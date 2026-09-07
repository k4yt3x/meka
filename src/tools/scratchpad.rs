//! Scratchpad: session-scoped, persisted key/value text store. Oversized tool outputs are
//! automatically redirected here and replaced inline with a preview + handle, keeping the
//! conversation context bounded. Provides write/read/edit/list/delete operations plus a regex
//! search mode.

use std::collections::HashMap;

use async_trait::async_trait;
use uuid::Uuid;

use super::{
    Tool, ToolOutput,
    util::{MAX_SEARCH_MATCHES, resolve_session_id, search_lines},
};
use crate::{
    conversation::{ContentBlock, Message, ToolResultContent},
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
    store::{ScratchpadEntry, Store},
    text::format_size,
};

/// Tool result text blocks larger than this (in bytes) are persisted to the database and replaced
/// with a preview + handle in the conversation context.
pub(crate) const MAX_INLINE_RESULT_BYTES: usize = 30_000;

/// Number of bytes included in the inline preview.
const PREVIEW_BYTES: usize = 2_000;

/// Default byte limit when reading back a persisted output.
const DEFAULT_READ_LIMIT: usize = 30_000;

/// Build a map from tool_use_id to (tool_name, input) for the ToolUse blocks in an assistant
/// message.
fn build_tool_use_map(assistant_message: &Message) -> HashMap<String, (String, serde_json::Value)> {
    let mut map = HashMap::new();
    for block in &assistant_message.content {
        if let ContentBlock::ToolUse { id, name, input } = block {
            map.insert(id.clone(), (name.clone(), input.clone()));
        }
    }
    map
}

fn build_scratchpad_reference(name: &str, size: usize) -> String {
    format!(
        "Output saved to scratchpad '{name}' ({size} bytes). \
         Use scratchpad_read to access it.",
    )
}

fn build_large_output_preview(name: &str, text: &str) -> String {
    let size = text.len();
    let preview_end = text.floor_char_boundary(PREVIEW_BYTES.min(size));
    let preview = &text[..preview_end];
    let has_more = preview_end < size;

    let mut replacement = format!(
        "<large-output name=\"{}\" size=\"{}\">\n\
         Output too large ({}). Read with `scratchpad_read`. Use \
         `limit: {}` to load the full content in one call, or page \
         with `offset`/`limit` if a partial read is enough.\n\n\
         Preview (first {} bytes):\n\
         {}",
        name,
        size,
        format_size(size),
        size,
        preview_end,
        preview,
    );
    if has_more {
        replacement.push_str("\n...");
    }
    replacement.push_str("\n</large-output>");
    replacement
}

/// Save tool results to the scratchpad when the agent explicitly requested it via the `scratchpad`
/// parameter on a tool call. Replaces the inline result with a brief reference.
///
/// `inherited_names` are the entries a parent lent this session read-only. The seven
/// `scratchpad_*` tools refuse to write those, and this door has to as well: it is the universal
/// parameter every tool takes, so `execute_command({.., scratchpad: "<inherited>"})` wrote a local
/// row under the parent's name and every later `scratchpad_read` found the shadow first. The
/// result stays inline, with the same refusal the tools give, so the model learns why.
pub(crate) async fn save_explicit_scratchpad_results(
    store: &Store,
    session_id: Uuid,
    inherited_names: &[String],
    assistant_message: &Message,
    results: &mut [ContentBlock],
) -> Result<()> {
    let tool_use_map = build_tool_use_map(assistant_message);

    for block in results.iter_mut() {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        {
            let scratchpad_name = tool_use_map
                .get(tool_use_id.as_str())
                .and_then(|(_, input)| input.get("scratchpad"))
                .and_then(|v| v.as_str());

            let Some(name) = scratchpad_name else {
                continue;
            };

            let text = ContentBlock::tool_result_text_content(content);
            if text.is_empty() {
                continue;
            }

            if is_inherited(inherited_names, name) {
                content.push(ToolResultContent::Text {
                    text: format!(
                        "\n\n[Not saved to the scratchpad: {}]",
                        inherited_refusal(name)
                    ),
                });
                continue;
            }

            let size = text.len();
            store.save_scratchpad_entry(session_id, name, &text).await?;

            *content = vec![ToolResultContent::Text {
                text: build_scratchpad_reference(name, size),
            }];
        }
    }
    Ok(())
}

/// Check each text block in tool results. If oversized, persist to DB and replace with a preview +
/// handle. Names are derived from the tool call's `scratchpad_hint` (MCP adapters) or the tool name
/// otherwise, with a numeric suffix on collision. `hints` is typically the per-turn map owned by
/// the agent; empty is fine.
pub(crate) async fn persist_oversized_results(
    store: &Store,
    session_id: Uuid,
    assistant_message: &Message,
    results: &mut [ContentBlock],
    hints: &std::collections::HashMap<String, String>,
) -> Result<()> {
    let tool_use_map = build_tool_use_map(assistant_message);
    let mut counter: usize = 0;

    for block in results.iter_mut() {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        {
            let base_name = hints.get(tool_use_id.as_str()).cloned().unwrap_or_else(|| {
                tool_use_map
                    .get(tool_use_id.as_str())
                    .map(|(name, _)| name.clone())
                    .unwrap_or_else(|| "unknown".to_string())
            });

            for item in content.iter_mut() {
                if let ToolResultContent::Text { text } = item {
                    if text.len() <= MAX_INLINE_RESULT_BYTES {
                        continue;
                    }

                    counter += 1;
                    // Unique for the life of the session, not just within this call: `counter`
                    // restarts at zero on every invocation (this runs once per assistant message)
                    // and `save_scratchpad_entry` is `INSERT OR REPLACE` keyed on `(session,
                    // name)`, so a name built from the counter alone would let a later turn's
                    // spill silently replace an earlier one under the handle the model still
                    // holds. The `tool_use_id` is provider-generated and unique per call.
                    let name = format!("{}_{}_{}", base_name, short_call_id(tool_use_id), counter);

                    store.save_scratchpad_entry(session_id, &name, text).await?;

                    *text = build_large_output_preview(&name, text);
                }
            }
        }
    }
    Ok(())
}

/// A short, name-safe slice of a provider tool-call id, for disambiguating scratchpad entries.
///
/// Ids run to ~30 characters (`toolu_01A9…`, `call_abc…`) and the whole thing in every entry name
/// would make `scratchpad_list` unreadable and the handles tedious for the model to quote back. The
/// tail is used rather than the head because providers put their fixed prefix at the front, so the
/// last few characters carry the entropy. Restricted to `[A-Za-z0-9]` because the name is also an
/// entry key.
fn short_call_id(tool_use_id: &str) -> String {
    let cleaned: String = tool_use_id
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect();
    let tail: String = cleaned
        .chars()
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if tail.is_empty() {
        "call".to_string()
    } else {
        tail
    }
}

pub(super) struct ScratchpadWriteTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    /// Names the parent has lent this sub-agent read-only. Writing to any of these is rejected so
    /// the child can't silently shadow the parent's copy. Empty on the root agent's registry.
    pub(crate) inherited_names: Vec<String>,
}

fn inherited_write_error(name: &str) -> Result<ToolOutput> {
    Ok(ToolOutput::text(inherited_refusal(name), true))
}

/// The one sentence every write door gives for an inherited name.
fn inherited_refusal(name: &str) -> String {
    format!(
        "Scratchpad entry '{name}' is inherited read-only from the parent. \
         Pick a different name (e.g. \"{name}_local\") for your own scratchpad state.",
    )
}

fn is_inherited(inherited_names: &[String], name: &str) -> bool {
    inherited_names.iter().any(|candidate| candidate == name)
}

#[async_trait]
impl Tool for ScratchpadWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_write".to_string(),
            description: "Store content in the scratchpad under the given name: a \
                session-scoped working memory that persists across turns without consuming \
                conversation context. If the name already exists, the content is overwritten. \
                Use this to save intermediate results, extracted text, accumulated data, or \
                research notes. You can also save tool output directly by adding a 'scratchpad' \
                parameter to any tool call. When you are a sub-agent, names inherited \
                read-only from the parent are refused here. Use a different name (e.g. \
                'name_local') for your own state."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the scratchpad entry."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to store."
                    }
                },
                "required": ["name", "content"]
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
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_write".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;
        let content = input["content"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_write".to_string(),
                message: "missing 'content' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, name) {
            return inherited_write_error(name);
        }

        let session_id = resolve_session_id(&self.site.session_id, "scratchpad_write")?;

        self.store
            .save_scratchpad_entry(session_id, name, content)
            .await?;

        Ok(ToolOutput::text(
            format!(
                "Stored {} bytes as scratchpad entry '{}'",
                content.len(),
                name,
            ),
            false,
        ))
    }
}

pub(super) struct ScratchpadReadTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    /// Sub-agent fallback: when the read misses the active (child) session, retry against this
    /// parent session for names listed in [`Self::inherited_names`]. `None` on the root agent's
    /// registry; no fallback path is taken.
    pub(crate) parent_session_id: Option<Uuid>,
    /// Allowlist of parent-scoped scratchpad names the sub-agent is permitted to read. Empty on
    /// the root agent.
    pub(crate) inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_read".to_string(),
            description: format!(
                "Read or search a scratchpad entry by name. Default returns {DEFAULT_READ_LIMIT} \
                 bytes from offset; pass a larger `limit` (no hard cap) to load the full \
                 entry in one call, or page with `offset`/`limit` for partial reads. Provide \
                 `regex` to return matching lines (max {MAX_SEARCH_MATCHES}) instead of a byte range. Also \
                 used to access content referenced by <large-output> tags. Pass the `size` \
                 value from the tag as `limit` when you intend to read everything. When this \
                 is a sub-agent and the name is not found locally, looks up names from the \
                 parent's inherited allowlist (see the system-prompt section if any).",
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry name."
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Byte offset to start reading from; one that falls inside a \
                                        multi-byte character moves back to that character's first \
                                        byte. Default: 0."
                    },
                    "limit": {
                        "type": "integer",
                        "default": DEFAULT_READ_LIMIT,
                        "description": format!(
                            "Maximum bytes to return; a range that would end inside a multi-byte \
                             character stops before it. Default: {DEFAULT_READ_LIMIT}."
                        )
                    },
                    "regex": {
                        "type": "string",
                        "description": format!(
                            "If provided, search the entry with this regex pattern \
                             and return matching lines (max {MAX_SEARCH_MATCHES} matches) instead of a byte range."
                        )
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
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_read".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        let session_id = resolve_session_id(&self.site.session_id, "scratchpad_read")?;

        let mut content = self.store.load_scratchpad_entry(session_id, name).await?;

        // Sub-agent inheritance: fall back to the parent's scratchpad if the child miss matches an
        // allowlisted name. Read-only: writes still target the child session, so the parent's
        // audit trail is untouched.
        if content.is_none()
            && let Some(parent_sid) = self.parent_session_id
            && self.inherited_names.iter().any(|n| n == name)
        {
            content = self.store.load_scratchpad_entry(parent_sid, name).await?;
        }

        let content = content.ok_or_else(|| MekaError::ToolExecution {
            tool_name: "scratchpad_read".to_string(),
            message: format!("scratchpad entry '{name}' not found"),
        })?;

        if let Some(pattern) = input.get("regex").and_then(|v| v.as_str()) {
            return search_lines(&content, pattern, "scratchpad_read");
        }

        read_mode(&content, &input)
    }
}

fn read_mode(content: &str, input: &serde_json::Value) -> Result<ToolOutput> {
    let offset = usize::try_from(input["offset"].as_u64().unwrap_or(0)).unwrap_or(usize::MAX);
    let limit = usize::try_from(input["limit"].as_u64().unwrap_or(DEFAULT_READ_LIMIT as u64))
        .unwrap_or(usize::MAX);
    let total = content.len();

    if offset >= total {
        return Ok(ToolOutput::text(
            format!("Offset {offset} exceeds content length ({total} bytes)"),
            true,
        ));
    }

    let start = content.floor_char_boundary(offset);
    let end = content.floor_char_boundary(start.saturating_add(limit).min(total));
    let slice = &content[start..end];

    Ok(ToolOutput::text(
        format!("{slice}\n\n(showing bytes {start}..{end} of {total})"),
        false,
    ))
}

pub(super) struct ScratchpadEditTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    /// See [`ScratchpadWriteTool::inherited_names`].
    pub(crate) inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadEditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_edit".to_string(),
            description: "Edit a scratchpad entry in place. Provide 'content' to fully \
                overwrite, or 'old_string'/'new_string' for targeted string replacement \
                (like edit_file). When you are a sub-agent, names inherited read-only \
                from the parent are refused. Copy the content into your own entry first \
                if you need to mutate it."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry name to edit."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full replacement content (mutually exclusive with `old_string`/`new_string`)."
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "default": false,
                        "description": "Replace every occurrence. Default: false."
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
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, name) {
            return inherited_write_error(name);
        }

        let session_id = resolve_session_id(&self.site.session_id, "scratchpad_edit")?;

        // Full overwrite mode
        if let Some(new_content) = input.get("content").and_then(|v| v.as_str()) {
            let updated = self
                .store
                .update_scratchpad_entry(session_id, name, new_content)
                .await?;

            return if updated {
                Ok(ToolOutput::text(
                    format!(
                        "Scratchpad entry '{}' overwritten ({} bytes)",
                        name,
                        new_content.len()
                    ),
                    false,
                ))
            } else {
                Ok(ToolOutput::text(
                    format!("Scratchpad entry '{name}' not found"),
                    true,
                ))
            };
        }

        // String replacement mode
        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: "provide either 'content' for full overwrite \
                    or 'old_string'/'new_string' for replacement"
                    .to_string(),
            })?;
        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: "missing 'new_string' parameter".to_string(),
            })?;
        let replace_all = input["replace_all"].as_bool().unwrap_or(false);

        let existing = self
            .store
            .load_scratchpad_entry(session_id, name)
            .await?
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: format!("scratchpad entry '{name}' not found"),
            })?;

        if !existing.contains(old_string) {
            return Ok(ToolOutput::text(
                format!(
                    "Error: '{}' not found in scratchpad entry '{}'",
                    super::util::truncate_string(old_string, 100),
                    name,
                ),
                true,
            ));
        }

        let (updated_content, count) = if replace_all {
            let count = existing.matches(old_string).count();
            (existing.replace(old_string, new_string), count)
        } else {
            (existing.replacen(old_string, new_string, 1), 1)
        };

        self.store
            .update_scratchpad_entry(session_id, name, &updated_content)
            .await?;

        Ok(ToolOutput::text(
            format!(
                "Scratchpad entry '{}': replaced {} occurrence(s) ({} bytes)",
                name,
                count,
                updated_content.len(),
            ),
            false,
        ))
    }
}

pub(super) struct ScratchpadListTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    /// Sub-agent fallback: list also enumerates parent entries filtered by
    /// [`Self::inherited_names`], rendered in a trailing `(inherited)` section. `None` on the
    /// root agent.
    pub(crate) parent_session_id: Option<Uuid>,
    /// Allowlist of parent-scoped scratchpad names visible to this sub-agent. Empty on the primary
    /// agent.
    pub(crate) inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_list".to_string(),
            description: "List scratchpad entries in the current session with their name, size, \
                creation time, and origin. Sub-agent entries inherited read-only from the \
                parent session appear in the same table with origin `inherited`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
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
    ) -> Result<ToolOutput> {
        let session_id = resolve_session_id(&self.site.session_id, "scratchpad_list")?;

        let own = self.store.list_scratchpad_entries(session_id).await?;

        // Inherited (parent) entries, filtered to the allowlist so the sub-agent never sees parent
        // state it wasn't explicitly granted.
        let mut rows: Vec<(ScratchpadEntry, &'static str)> =
            own.into_iter().map(|entry| (entry, "own")).collect();
        if let Some(parent_sid) = self.parent_session_id
            && !self.inherited_names.is_empty()
        {
            let parent_entries = self.store.list_scratchpad_entries(parent_sid).await?;
            rows.extend(
                parent_entries
                    .into_iter()
                    .filter(|entry| self.inherited_names.iter().any(|n| n == &entry.name))
                    .map(|entry| (entry, "inherited")),
            );
        }

        if rows.is_empty() {
            return Ok(ToolOutput::text("Scratchpad is empty.".to_string(), false));
        }

        let table_rows: Vec<Vec<String>> = rows
            .iter()
            .map(|(entry, origin)| {
                vec![
                    entry.name.clone(),
                    format_size(entry.size),
                    entry.created_at[..19.min(entry.created_at.len())].to_string(),
                    origin.to_string(),
                ]
            })
            .collect();
        let mut output =
            crate::text::format_columns(&["Name", "Size", "Created", "Origin"], &table_rows);
        output.push_str(&format!("\n{} entries total", rows.len()));

        Ok(ToolOutput::text(output, false))
    }
}

pub(super) struct ScratchpadMergeTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    /// See [`ScratchpadReadTool::parent_session_id`]. Sources may be inherited from the parent's
    /// scratchpad.
    pub(crate) parent_session_id: Option<Uuid>,
    /// See [`ScratchpadWriteTool::inherited_names`]. The `target` is blocked if listed here;
    /// sources are also matched against this set for the parent-fallback read.
    pub(crate) inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadMergeTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_merge".to_string(),
            description: "Combine scratchpad entries into one entry without routing the bytes \
                through the conversation, for collecting parallel sub-agent reports or any \
                accumulated data. The entries go in the order given: `sources` first, in the \
                order listed, then every own entry whose name starts with `prefix`, in name \
                order. `format` controls the join: `concat_with_headers` (default) puts a \
                `--- name ---` line before each entry's content, `concat` joins the contents \
                with a newline, `json_array` parses each content as JSON (quoting one that is \
                not) into one compact JSON array. The sources are kept as they are; nothing is \
                deleted, and `target` is overwritten if it exists. A sub-agent cannot merge into \
                a name inherited read-only from the parent, but may list inherited entries in \
                `sources`; `prefix` selects only its own entries."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Names of scratchpad entries to combine, in this order. \
                                        Optional when `prefix` is given."
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Also combine every own entry whose name starts with this, \
                                        in name order, after `sources`. `target` itself is never \
                                        selected."
                    },
                    "target": {
                        "type": "string",
                        "description": "Name to store the merged result under. Overwrites if it already exists."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["concat_with_headers", "concat", "json_array"],
                        "default": "concat_with_headers",
                        "description": "How to join the source entries. Default: `concat_with_headers`."
                    }
                },
                "required": ["target"]
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
        let target = input["target"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_merge".to_string(),
                message: "missing 'target' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, target) {
            return inherited_write_error(target);
        }

        let mut sources: Vec<String> = match input.get("sources") {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::Array(items)) => items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect(),
            Some(_) => {
                return Err(MekaError::ToolExecution {
                    tool_name: "scratchpad_merge".to_string(),
                    message: "'sources' must be an array of entry names".to_string(),
                });
            }
        };
        let prefix = input.get("prefix").and_then(|value| value.as_str());
        if sources.is_empty() && prefix.is_none() {
            return Ok(ToolOutput::text(
                "scratchpad_merge: name at least one entry in 'sources', or give a 'prefix'"
                    .to_string(),
                true,
            ));
        }

        let format = input
            .get("format")
            .and_then(|value| value.as_str())
            .unwrap_or("concat_with_headers");

        let session_id = resolve_session_id(&self.site.session_id, "scratchpad_merge")?;

        // The prefix selects this session's own entries, in name order so the merge is the same
        // however the entries were written, after whatever `sources` named. The target is left
        // out: a merge into `report-all` from `report-` must not fold the previous result back in.
        if let Some(prefix) = prefix {
            let mut matched: Vec<String> = self
                .store
                .list_scratchpad_entries(session_id)
                .await?
                .into_iter()
                .map(|entry| entry.name)
                .filter(|name| {
                    name.starts_with(prefix) && name != target && !sources.contains(name)
                })
                .collect();
            matched.sort();
            sources.extend(matched);
            if sources.is_empty() {
                return Ok(ToolOutput::text(
                    format!("scratchpad_merge: no scratchpad entry starts with '{prefix}'"),
                    true,
                ));
            }
        }

        // Resolve each source. Inheritance-aware: child first, then parent if the name is
        // allowlisted; same fallback path as ScratchpadReadTool. Any missing source aborts the
        // merge, so the target is never partially written.
        let mut loaded: Vec<(String, String)> = Vec::with_capacity(sources.len());
        for name in &sources {
            let mut content = self.store.load_scratchpad_entry(session_id, name).await?;
            if content.is_none()
                && let Some(parent_sid) = self.parent_session_id
                && self.inherited_names.iter().any(|n| n == name)
            {
                content = self.store.load_scratchpad_entry(parent_sid, name).await?;
            }
            let content = content.ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_merge".to_string(),
                message: format!("scratchpad entry '{name}' not found"),
            })?;
            loaded.push((name.clone(), content));
        }

        let merged = match format {
            "concat" => loaded
                .iter()
                .map(|(_, body)| body.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            "json_array" => {
                let values: Vec<serde_json::Value> = loaded
                    .iter()
                    .map(|(_, body)| {
                        serde_json::from_str::<serde_json::Value>(body)
                            .unwrap_or_else(|_| serde_json::Value::String(body.clone()))
                    })
                    .collect();
                serde_json::to_string(&values).map_err(|error| MekaError::ToolExecution {
                    tool_name: "scratchpad_merge".to_string(),
                    message: format!("failed to serialize merged JSON array: {error}"),
                })?
            }
            "concat_with_headers" => {
                let mut output = String::new();
                for (index, (name, body)) in loaded.iter().enumerate() {
                    if index > 0 {
                        output.push('\n');
                    }
                    output.push_str(&format!("--- {name} ---\n"));
                    output.push_str(body);
                }
                output
            }
            other => {
                return Ok(ToolOutput::text(
                    format!(
                        "scratchpad_merge: unknown `format` value '{other}' (expected \
                         'concat_with_headers', 'concat', or 'json_array')",
                    ),
                    true,
                ));
            }
        };

        let merged_size = merged.len();
        self.store
            .save_scratchpad_entry(session_id, target, &merged)
            .await?;

        Ok(ToolOutput::text(
            format!(
                "Merged {} entries into scratchpad entry '{}' ({} bytes)",
                loaded.len(),
                target,
                merged_size,
            ),
            false,
        ))
    }
}

pub(super) struct ScratchpadDeleteTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    /// See [`ScratchpadWriteTool::inherited_names`].
    pub(crate) inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_delete".to_string(),
            description: "Delete a scratchpad entry by name to free up space. When you are \
                a sub-agent, names inherited read-only from the parent cannot be deleted."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry name to delete."
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
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_delete".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, name) {
            return inherited_write_error(name);
        }

        let session_id = resolve_session_id(&self.site.session_id, "scratchpad_delete")?;

        let deleted = self.store.delete_scratchpad_entry(session_id, name).await?;

        if deleted {
            Ok(ToolOutput::text(
                format!("Scratchpad entry '{name}' deleted"),
                false,
            ))
        } else {
            Ok(ToolOutput::text(
                format!("Scratchpad entry '{name}' not found"),
                true,
            ))
        }
    }
}

pub(super) struct ScratchpadRenameTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    /// See [`ScratchpadWriteTool::inherited_names`].
    pub(crate) inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadRenameTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_rename".to_string(),
            description: "Rename a scratchpad entry from `old` to `new` without round-tripping \
                the content through the conversation. Errors if `old` doesn't exist, if `new` \
                already exists, or (for sub-agents) if either name is inherited read-only from \
                the parent."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "old": {
                        "type": "string",
                        "description": "Current scratchpad entry name."
                    },
                    "new": {
                        "type": "string",
                        "description": "Replacement scratchpad entry name."
                    }
                },
                "required": ["old", "new"]
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
        let old = input["old"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_rename".to_string(),
                message: "missing 'old' parameter".to_string(),
            })?;
        let new = input["new"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_rename".to_string(),
                message: "missing 'new' parameter".to_string(),
            })?;

        // Block both ends. Renaming away from an inherited source would be a no-op against the
        // parent's row but implies the sub-agent owns the name; renaming to an inherited target
        // would create a child shadow. Either way: reject early with the same error text the other
        // mutators use, naming the offending entry.
        if is_inherited(&self.inherited_names, old) {
            return inherited_write_error(old);
        }
        if is_inherited(&self.inherited_names, new) {
            return inherited_write_error(new);
        }

        if old == new {
            return Ok(ToolOutput::text(
                format!("Scratchpad entry '{old}' is already named that"),
                true,
            ));
        }

        let session_id = resolve_session_id(&self.site.session_id, "scratchpad_rename")?;

        let outcome = self
            .store
            .rename_scratchpad_entry(session_id, old, new)
            .await?;

        match outcome {
            crate::store::RenameOutcome::Renamed => Ok(ToolOutput::text(
                format!("Renamed scratchpad entry '{old}' to '{new}'"),
                false,
            )),
            crate::store::RenameOutcome::NotFound => Ok(ToolOutput::text(
                format!("Scratchpad entry '{old}' not found"),
                true,
            )),
            crate::store::RenameOutcome::TargetExists => Ok(ToolOutput::text(
                format!(
                    "Scratchpad entry '{new}' already exists; delete it first or pick a \
                     different name",
                ),
                true,
            )),
        }
    }
}

pub(super) struct ScratchpadLoadFileTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    /// See [`ScratchpadWriteTool::inherited_names`].
    pub(crate) inherited_names: Vec<String>,
}

#[async_trait]
impl Tool for ScratchpadLoadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_load_file".to_string(),
            description: "Read a file's contents directly into the scratchpad without routing \
                the bytes through the conversation. Useful for staging a large captured log or \
                document that you want to hand to sub-agents via `inherit_scratchpad` on \
                `agent_spawn`; the model never sees the payload. UTF-8 text only; binary \
                files are refused with the detected MIME type. For binary content, pass the \
                file path directly to whatever tool will consume it; sub-agents inherit the \
                parent's filesystem access. Overwrites an existing entry of the same name. \
                Sub-agents cannot load into a name inherited read-only from the parent."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read."
                    },
                    "name": {
                        "type": "string",
                        "description": "Name to store the contents under in the scratchpad."
                    }
                },
                "required": ["path", "name"]
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
        let path = input["path"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_load_file".to_string(),
                message: "missing 'path' parameter".to_string(),
            })?
            .to_string();
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_load_file".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        if is_inherited(&self.inherited_names, name) {
            return inherited_write_error(name);
        }

        let session_id = resolve_session_id(&self.site.session_id, "scratchpad_load_file")?;

        // Resolve a relative path against the session cwd before canonicalizing, so `/cd` is
        // honored (canonicalize alone would resolve relative to the process cwd).
        let resolved = crate::workspace::resolve_against_cwd(&self.site.cwd, &path);
        let canonical =
            super::util::canonicalize_for_tool("scratchpad_load_file", &resolved).await?;
        super::util::refuse_private_read("scratchpad_load_file", &self.site, &canonical)?;

        // Raw bytes rather than a `String`, so a UTF-8 failure can be sniffed once and the model
        // told what kind of binary it tried to load. The happy path pays one extra allocation,
        // negligible at the sizes this tool is meant for.
        let bytes = super::file::read_file_bytes(&canonical)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "scratchpad_load_file".to_string(),
                message: format!("failed to read '{path}': {error}"),
            })?;

        let content = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(utf8_error) => {
                let bytes = utf8_error.as_bytes();
                let detected = infer::get(bytes)
                    .map(|kind| format!(" Detected MIME type: {}.", kind.mime_type()))
                    .unwrap_or_default();
                return Ok(ToolOutput::text(
                    format!(
                        "'{}' is not valid UTF-8: {}.{}",
                        path,
                        utf8_error.utf8_error(),
                        detected,
                    ),
                    true,
                ));
            }
        };

        let byte_count = content.len();
        self.store
            .save_scratchpad_entry(session_id, name, &content)
            .await?;

        Ok(ToolOutput::text(
            format!("Loaded {byte_count} bytes from '{path}' into scratchpad entry '{name}'",),
            false,
        ))
    }
}

pub(super) struct ScratchpadSaveFileTool {
    /// The write boundary, shared with `write_file`. This tool reads as the scratchpad's
    /// `write_file` and lands bytes at a path the user named, so it is fenced the same way.
    pub(crate) scope: crate::workspace::WriteScope,
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
    /// See [`ScratchpadReadTool::parent_session_id`].
    pub(crate) parent_session_id: Option<Uuid>,
    /// See [`ScratchpadReadTool::inherited_names`].
    pub(crate) inherited_names: Vec<String>,
    /// The same tracker `write_file` and `edit_file` stamp.
    ///
    /// This tool is described to the model as the scratchpad's `write_file`, and it lands bytes at
    /// a path the user named, so it has to leave the same record: otherwise the tracker keeps the
    /// pre-save stamp and the next `write_file` or `edit_file` on that path is refused as if
    /// something else had written it.
    pub(crate) read_tracker: crate::tools::ReadTracker,
}

#[async_trait]
impl Tool for ScratchpadSaveFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_save_file".to_string(),
            description: "Write the contents of a scratchpad entry to a file on disk without \
                routing the bytes through the conversation. Useful for persisting a sub-agent's \
                report or a large extracted result. Mirrors `write_file`: creates parent \
                directories, refuses to replace an existing file unless `force` is set, UTF-8 \
                only. A path outside the workspace roots is refused unless the level is \
                `unrestricted`. Sub-agents can save inherited entries (read from parent, write \
                to disk) without copying through the model."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry to read from."
                    },
                    "path": {
                        "type": "string",
                        "description": "The file path to write to."
                    },
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Proceed despite the file already existing, or existing but \
                                        being unreadable. Default: false."
                    }
                },
                "required": ["name", "path"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Workspace
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let name = input["name"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_save_file".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;
        let path = input["path"]
            .as_str()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "scratchpad_save_file".to_string(),
                message: "missing 'path' parameter".to_string(),
            })?
            .to_string();

        let session_id = resolve_session_id(&self.site.session_id, "scratchpad_save_file")?;

        // Inherited-fallback read: mirror ScratchpadReadTool. Lets a sub- agent flush a parent's
        // allowlisted entry to disk without having to first copy it into its own scratchpad.
        let mut content = self.store.load_scratchpad_entry(session_id, name).await?;
        if content.is_none()
            && let Some(parent_sid) = self.parent_session_id
            && self.inherited_names.iter().any(|n| n == name)
        {
            content = self.store.load_scratchpad_entry(parent_sid, name).await?;
        }
        let content = content.ok_or_else(|| MekaError::ToolExecution {
            tool_name: "scratchpad_save_file".to_string(),
            message: format!("scratchpad entry '{name}' not found"),
        })?;

        // Shared with `write_file` rather than mirrored, because the two must agree on the file
        // they name and on the lock they take, and a copy of the resolution here agreed on neither.
        // Both are dispatched concurrently from one assistant message and both write through a temp
        // file derived from the target, so two calls naming one path could interleave.
        let (target, _write_guard) = super::file::resolve_write_target(
            "scratchpad_save_file",
            &self.site.cwd,
            &self.scope,
            &path,
        )
        .await?;

        // Asked under the write lock, so the answer is still true when the write happens. This
        // tool reads as the scratchpad's `write_file`, so it refuses to replace an existing file
        // the same way, and `force` is the same escape hatch, spelled the same way.
        let force = input
            .get("force")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        // "Cannot tell" is not "does not exist": a target that is there but cannot be stat'ed (a
        // symlink loop, an I/O error) must not be written over without `force`. `write_file`
        // refuses the same case with the same escape hatch.
        let replaced = match tokio::fs::metadata(&target).await {
            Ok(meta) => Some(meta.len()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) if force => {
                let path = target.display();
                tracing::debug!(
                    "scratchpad_save_file: failed to stat '{path}' ({error}); `force` is set, \
                     writing anyway"
                );
                None
            }
            Err(error) => {
                return Ok(ToolOutput::text(
                    format!(
                        "Error: '{path}' exists but could not be read ({error}), so meka cannot tell what \
                         saving would replace. Pass force=true to overwrite."
                    ),
                    true,
                ));
            }
        };
        if let Some(existing_bytes) = replaced
            && !force
        {
            return Ok(ToolOutput::text(
                format!(
                    "Error: '{path}' already exists ({existing_bytes} bytes) and saving would replace it. Read it \
                     first if you need what it says, pick another path, or set force=true."
                ),
                true,
            ));
        }

        let byte_count = content.len();
        super::file::write_file_bytes(&target, content.as_bytes())
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "scratchpad_save_file".to_string(),
                message: format!("failed to write '{path}': {error}"),
            })?;

        // Stamped like any other write meka performs, so the next `write_file` or `edit_file` on
        // this path does not mistake meka's own bytes for someone else's. `FileRoute::Local`
        // because this wrote to disk directly rather than through an editor delegate.
        super::file::record_write(
            &self.read_tracker,
            target.clone(),
            super::file::FileRoute::Local,
            &content,
        )
        .await;

        Ok(ToolOutput::text(
            format!(
                "Saved {} bytes from scratchpad entry '{}' to '{}'{}",
                byte_count,
                name,
                path,
                match replaced {
                    Some(existing_bytes) => format!(", replacing {existing_bytes} bytes"),
                    None => String::new(),
                }
            ),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::conversation::{ContentBlock, Role};

    fn session_id_for_test(uuid: Uuid) -> crate::session::SharedSessionId {
        crate::session::SharedSessionId::new(Some(uuid))
    }

    fn make_assistant_message(tool_calls: Vec<(&str, &str, serde_json::Value)>) -> Message {
        Message {
            role: Role::Assistant,
            content: tool_calls
                .into_iter()
                .map(|(id, name, input)| ContentBlock::ToolUse {
                    id: id.to_string(),
                    name: name.to_string(),
                    input,
                })
                .collect(),
        }
    }

    // -- persist_oversized_results --

    /// `scratchpad_save_file` is fenced by the scope it was built with, and stamps the tracker.
    ///
    /// Every other test here builds the tool with `WriteScope::unconfined()`, so replacing
    /// `&self.scope` with a fresh unconfined scope would leave them green, and this tool is
    /// described to the model as the scratchpad's `write_file`, so that would be a full write door
    /// outside the boundary at `workspace`. A save that is not recorded makes the next
    /// `write_file`/`edit_file` on the same path refuse meka's own bytes as someone else's.
    #[tokio::test]
    async fn saving_a_file_is_fenced_by_its_scope_and_records_the_write() {
        let temp = tempfile::tempdir().expect("tempdir");
        let workspace = crate::workspace::canonical_for_test(temp.path()).join("work");
        std::fs::create_dir_all(&workspace).expect("workspace");
        let outside = crate::workspace::canonical_for_test(temp.path()).join("outside");
        std::fs::create_dir_all(&outside).expect("outside");

        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "report", "FINDINGS")
            .await
            .expect("seed");

        let read_tracker: crate::tools::ReadTracker =
            std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));
        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::clone(&read_tracker),
            scope: crate::workspace::WriteScope::confined(vec![workspace.clone()]),
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::SharedCwd::new(workspace.clone()))
                .with_session_id(session_id_for_test(session_id)),
        };

        // Outside the one root: refused, and nothing lands.
        let escaped = outside.join("leaked.txt");
        let refused = tool
            .execute(
                serde_json::json!({
                    "name": "report",
                    "path": escaped.to_str().expect("path"),
                    "force": true,
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        let refused_text = match refused {
            Err(error) => error.to_string(),
            Ok(output) => {
                crate::conversation::ContentBlock::tool_result_text_content(&output.content)
            }
        };
        assert!(
            !escaped.exists(),
            "a save outside the workspace must not land: {refused_text}"
        );

        // Inside it: lands, and the tracker records that meka was the writer.
        let inside = workspace.join("report.txt");
        tool.execute(
            serde_json::json!({
                "name": "report",
                "path": inside.to_str().expect("path"),
                "force": true,
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("a save inside the workspace must succeed");
        assert_eq!(
            std::fs::read_to_string(&inside).expect("read back"),
            "FINDINGS"
        );
        assert!(
            read_tracker.read().await.contains_key(&inside),
            "the write must be stamped, or the next write_file blames someone else for it"
        );
    }

    #[tokio::test]
    async fn persist_oversized_results_replaces_large_text() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let large_text = "x".repeat(MAX_INLINE_RESULT_BYTES + 1000);
        let assistant_msg =
            make_assistant_message(vec![("call-1", "execute_command", serde_json::json!({}))]);
        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: large_text.clone(),
            }],
            is_error: false,
        }];

        persist_oversized_results(
            &manager,
            session_id,
            &assistant_msg,
            &mut results,
            &std::collections::HashMap::new(),
        )
        .await
        .expect("persist");

        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            let text = ContentBlock::tool_result_text_content(content);
            assert!(text.contains("<large-output"));
            // The tool name still leads, so the handle stays recognizable; the call-id tail after
            // it is what makes it unique (see the collision test below).
            assert!(text.contains("name=\"execute_command_"), "{}", text);
            assert!(text.contains("scratchpad_read"));
            assert!(!text.contains(&large_text));
        } else {
            panic!("expected ToolResult");
        }

        let name = format!("execute_command_{}_1", short_call_id("call-1"));
        let loaded = manager
            .load_scratchpad_entry(session_id, &name)
            .await
            .expect("load");
        assert_eq!(loaded, Some(large_text));
    }

    /// The counter restarts on every call, so a name built from it alone repeats across turns, and
    /// `save_scratchpad_entry` is `INSERT OR REPLACE`, so the later spill would silently destroy
    /// the earlier one while the model still held the first handle.
    #[tokio::test]
    async fn spilled_outputs_from_different_calls_do_not_overwrite_each_other() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let mut names = Vec::new();
        for (call_id, body) in [("call-turn1", 'a'), ("call-turn3", 'b')] {
            let text = body.to_string().repeat(MAX_INLINE_RESULT_BYTES + 100);
            let assistant_msg =
                make_assistant_message(vec![(call_id, "execute_command", serde_json::json!({}))]);
            let mut results = vec![ContentBlock::ToolResult {
                tool_use_id: call_id.to_string(),
                content: vec![ToolResultContent::Text { text: text.clone() }],
                is_error: false,
            }];
            persist_oversized_results(
                &manager,
                session_id,
                &assistant_msg,
                &mut results,
                &std::collections::HashMap::new(),
            )
            .await
            .expect("persist");
            names.push((
                format!("execute_command_{}_1", short_call_id(call_id)),
                text,
            ));
        }

        assert_ne!(names[0].0, names[1].0, "two calls must not share a name");
        for (name, expected) in &names {
            assert_eq!(
                manager
                    .load_scratchpad_entry(session_id, name)
                    .await
                    .expect("load"),
                Some(expected.clone()),
                "entry {name} was overwritten"
            );
        }
    }

    #[tokio::test]
    async fn persist_oversized_results_leaves_small_text() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let small_text = "hello world".to_string();
        let assistant_msg =
            make_assistant_message(vec![("call-1", "execute_command", serde_json::json!({}))]);
        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: small_text.clone(),
            }],
            is_error: false,
        }];

        persist_oversized_results(
            &manager,
            session_id,
            &assistant_msg,
            &mut results,
            &std::collections::HashMap::new(),
        )
        .await
        .expect("persist");

        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            assert_eq!(ContentBlock::tool_result_text_content(content), small_text);
        }
    }

    // -- save_explicit_scratchpad_results --

    #[tokio::test]
    async fn explicit_scratchpad_save() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let assistant_msg = make_assistant_message(vec![(
            "call-1",
            "execute_command",
            serde_json::json!({"command": "echo hi", "scratchpad": "cmd_output"}),
        )]);

        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: "hi\n".to_string(),
            }],
            is_error: false,
        }];

        save_explicit_scratchpad_results(&manager, session_id, &[], &assistant_msg, &mut results)
            .await
            .expect("save");

        // Result should be replaced with a reference
        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            let text = ContentBlock::tool_result_text_content(content);
            assert!(text.contains("cmd_output"));
            assert!(text.contains("scratchpad_read"));
            assert!(!text.contains("hi\n"));
        }

        // Content should be in the DB
        let loaded = manager
            .load_scratchpad_entry(session_id, "cmd_output")
            .await
            .expect("load");
        assert_eq!(loaded, Some("hi\n".to_string()));
    }

    /// The universal `scratchpad` parameter is the eighth write door for an inherited name, and
    /// the one that let a worker shadow the parent's entry with a local row of the same name.
    #[tokio::test]
    async fn an_explicit_save_under_an_inherited_name_writes_nothing_and_says_why() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let assistant_msg = make_assistant_message(vec![(
            "call-1",
            "execute_command",
            serde_json::json!({"command": "make", "scratchpad": "build_log"}),
        )]);
        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: "the worker's own output".to_string(),
            }],
            is_error: false,
        }];

        save_explicit_scratchpad_results(
            &manager,
            session_id,
            &["build_log".to_string()],
            &assistant_msg,
            &mut results,
        )
        .await
        .expect("no error: the result is delivered inline");

        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "build_log")
                .await
                .expect("load"),
            None,
            "no local row may shadow the inherited entry"
        );
        let ContentBlock::ToolResult { content, .. } = &results[0] else {
            panic!("a tool result");
        };
        let text = ContentBlock::tool_result_text_content(content);
        assert!(text.contains("the worker's own output"), "{text}");
        assert!(text.contains("inherited read-only"), "{text}");
    }

    #[tokio::test]
    async fn explicit_scratchpad_not_requested() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let assistant_msg = make_assistant_message(vec![(
            "call-1",
            "execute_command",
            serde_json::json!({"command": "echo hi"}),
        )]);

        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: "hi\n".to_string(),
            }],
            is_error: false,
        }];

        save_explicit_scratchpad_results(&manager, session_id, &[], &assistant_msg, &mut results)
            .await
            .expect("save");

        // Result should be unchanged
        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            assert_eq!(ContentBlock::tool_result_text_content(content), "hi\n");
        }
    }

    #[tokio::test]
    async fn explicit_scratchpad_ignores_non_scratchpad_keys() {
        // When a tool uses `from_scratchpad` (an input-source parameter) rather than `scratchpad`
        // (the output-destination convention), the agent-layer save must not touch the
        // pre-existing scratchpad entry.
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "img", "original base64 data")
            .await
            .expect("seed");

        let assistant_msg = make_assistant_message(vec![(
            "call-1",
            "render_image",
            serde_json::json!({"from_scratchpad": "img"}),
        )]);

        let mut results = vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: vec![ToolResultContent::Text {
                text: "[Image rendered from scratchpad \"img\"]".to_string(),
            }],
            is_error: false,
        }];

        save_explicit_scratchpad_results(&manager, session_id, &[], &assistant_msg, &mut results)
            .await
            .expect("save");

        // Pre-existing scratchpad entry should be untouched.
        let loaded = manager
            .load_scratchpad_entry(session_id, "img")
            .await
            .expect("load");
        assert_eq!(loaded.as_deref(), Some("original base64 data"));

        // Result content should also be unchanged (no rewriting to a reference).
        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            assert_eq!(
                ContentBlock::tool_result_text_content(content),
                "[Image rendered from scratchpad \"img\"]"
            );
        }
    }

    // -- scratchpad_write --

    #[tokio::test]
    async fn scratchpad_write_stores_an_entry() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadWriteTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "greeting", "content": "hello world"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        assert!(result.text_content().contains("greeting"));

        let loaded = manager
            .load_scratchpad_entry(session_id, "greeting")
            .await
            .expect("load");
        assert_eq!(loaded, Some("hello world".to_string()));
    }

    #[tokio::test]
    async fn scratchpad_write_overwrites() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "notes", "old content")
            .await
            .expect("save");

        let tool = ScratchpadWriteTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        tool.execute(
            serde_json::json!({"name": "notes", "content": "new content"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("execute");

        let loaded = manager
            .load_scratchpad_entry(session_id, "notes")
            .await
            .expect("load");
        assert_eq!(loaded, Some("new content".to_string()));
    }

    // -- scratchpad_read --

    #[tokio::test]
    async fn scratchpad_read_returns_the_stored_entry() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "data", "line1\nline2\nline3\n")
            .await
            .expect("save");

        let tool = ScratchpadReadTool {
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "data"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        let text = result.text_content();
        assert!(text.contains("line1"));
        assert!(text.contains("line3"));
    }

    #[tokio::test]
    async fn scratchpad_read_with_offset_and_limit() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "abc", "abcdefghij")
            .await
            .expect("save");

        let tool = ScratchpadReadTool {
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "abc", "offset": 3, "limit": 4}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        let text = result.text_content();
        assert!(text.contains("defg"));
        assert!(text.contains("showing bytes 3..7 of 10"));
    }

    /// `offset` and `limit` are bytes, and the description says what happens to a boundary inside
    /// a character: the start moves back to the character's first byte, the end stops before it.
    #[tokio::test]
    async fn a_byte_range_inside_a_character_is_moved_to_its_boundaries() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        // `é` is bytes 1..3 and `ü` bytes 4..6, so offset 2 is inside the first and the range's
        // end at 5 inside the second.
        manager
            .save_scratchpad_entry(session_id, "accents", "hélüo")
            .await
            .expect("save");
        let tool = ScratchpadReadTool {
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "accents", "offset": 2, "limit": 4}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        let text = result.text_content();
        assert!(text.starts_with("él\n"), "{text:?}");
        assert!(text.contains("showing bytes 1..4 of 7"), "{text:?}");
    }

    #[tokio::test]
    async fn scratchpad_read_search_mode() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(
                session_id,
                "fruits",
                "apple\nbanana\napricot\ncherry\navocado\n",
            )
            .await
            .expect("save");

        let tool = ScratchpadReadTool {
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "fruits", "regex": "^a"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        let text = result.text_content();
        assert!(text.contains("1:apple"));
        assert!(text.contains("3:apricot"));
        assert!(text.contains("5:avocado"));
        assert!(!text.contains("banana"));
    }

    #[tokio::test]
    async fn scratchpad_read_not_found() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadReadTool {
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool.execute(
            serde_json::json!({"name": "nonexistent"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        );
        assert!(result.await.is_err());
    }

    // -- scratchpad_edit --

    #[tokio::test]
    async fn scratchpad_edit_overwrite() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "doc", "old content")
            .await
            .expect("save");

        let tool = ScratchpadEditTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "doc", "content": "new content"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        assert!(result.text_content().contains("overwritten"));

        let loaded = manager
            .load_scratchpad_entry(session_id, "doc")
            .await
            .expect("load");
        assert_eq!(loaded, Some("new content".to_string()));
    }

    #[tokio::test]
    async fn scratchpad_edit_replacement() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "doc", "hello world hello")
            .await
            .expect("save");

        let tool = ScratchpadEditTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "doc", "old_string": "hello", "new_string": "hi"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.text_content().contains("1 occurrence(s)"));

        let loaded = manager
            .load_scratchpad_entry(session_id, "doc")
            .await
            .expect("load");
        assert_eq!(loaded, Some("hi world hello".to_string()));
    }

    #[tokio::test]
    async fn scratchpad_edit_replace_all() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "doc", "foo bar foo baz foo")
            .await
            .expect("save");

        let tool = ScratchpadEditTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({
                    "name": "doc",
                    "old_string": "foo",
                    "new_string": "qux",
                    "replace_all": true
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.text_content().contains("3 occurrence(s)"));

        let loaded = manager
            .load_scratchpad_entry(session_id, "doc")
            .await
            .expect("load");
        assert_eq!(loaded, Some("qux bar qux baz qux".to_string()));
    }

    // -- scratchpad_list --

    #[tokio::test]
    async fn scratchpad_list_empty() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadListTool {
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.text_content().contains("empty"));
    }

    #[tokio::test]
    async fn scratchpad_list_with_entries() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "notes", "content one")
            .await
            .expect("save");
        manager
            .save_scratchpad_entry(session_id, "data", "content two")
            .await
            .expect("save");

        let tool = ScratchpadListTool {
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        let text = result.text_content();
        assert!(text.contains("notes"));
        assert!(text.contains("data"));
        assert!(text.contains("2 entries total"));
    }

    // -- scratchpad_delete --

    #[tokio::test]
    async fn scratchpad_delete_removes_the_entry() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "temp", "temp data")
            .await
            .expect("save");

        let tool = ScratchpadDeleteTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "temp"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        assert!(result.text_content().contains("deleted"));

        let loaded = manager
            .load_scratchpad_entry(session_id, "temp")
            .await
            .expect("load");
        assert_eq!(loaded, None);
    }

    #[tokio::test]
    async fn scratchpad_delete_not_found() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadDeleteTool {
            store: manager,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "nonexistent"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
    }

    // -- sub-agent inheritance --

    #[tokio::test]
    async fn inherited_scratchpad_read_falls_back_to_parent() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "captured", "parent payload")
            .await
            .expect("seed parent");

        let tool = ScratchpadReadTool {
            store: manager.clone(),
            parent_session_id: Some(parent),
            inherited_names: vec!["captured".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "captured"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read inherited");

        assert!(!result.is_error);
        assert!(result.text_content().contains("parent payload"));
    }

    #[tokio::test]
    async fn inherited_scratchpad_prefers_child_when_both_present() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "shared", "parent value")
            .await
            .expect("seed parent");
        manager
            .save_scratchpad_entry(child, "shared", "child value")
            .await
            .expect("seed child");

        let tool = ScratchpadReadTool {
            store: manager,
            parent_session_id: Some(parent),
            inherited_names: vec!["shared".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "shared"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read shadowed");

        assert!(result.text_content().contains("child value"));
    }

    #[tokio::test]
    async fn inherited_scratchpad_read_blocks_names_not_in_allowlist() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "secret", "do not leak")
            .await
            .expect("seed parent");

        // allowlist only mentions a different name; the parent's "secret" entry must stay
        // invisible.
        let tool = ScratchpadReadTool {
            store: manager,
            parent_session_id: Some(parent),
            inherited_names: vec!["unrelated".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "secret"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(result.is_err(), "secret name must not be readable");
    }

    #[tokio::test]
    async fn inherited_scratchpad_list_respects_allowlist() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "shared_research", "p1")
            .await
            .expect("seed parent");
        manager
            .save_scratchpad_entry(parent, "private_note", "p2")
            .await
            .expect("seed parent");
        manager
            .save_scratchpad_entry(child, "own_note", "c1")
            .await
            .expect("seed child");

        let tool = ScratchpadListTool {
            store: manager,
            parent_session_id: Some(parent),
            inherited_names: vec!["shared_research".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };

        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("list");

        let text = result.text_content();
        assert!(text.contains("own_note"));
        assert!(text.contains("shared_research"));
        assert!(
            !text.contains("private_note"),
            "non-allowlisted parent entry must not appear, got: {text}"
        );
        // Unified-table contract: one Origin header, one totals line, no separate "inherited from
        // parent" section.
        assert_eq!(text.matches("Origin").count(), 1);
        assert!(text.contains("2 entries total"));
        assert!(!text.contains("inherited from parent"));
    }

    #[tokio::test]
    async fn inherited_scratchpad_list_handles_child_only_when_empty_allowlist() {
        // When no inheritance is configured, the list behaves exactly as before: no extra section,
        // no parent enumeration.
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "private", "do not leak")
            .await
            .expect("seed parent");

        let tool = ScratchpadListTool {
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };

        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("list");

        let text = result.text_content();
        assert!(!text.contains("private"));
        assert!(!text.contains("(inherited"));
    }

    #[tokio::test]
    async fn inherited_scratchpad_list_unified_table_has_origin_column() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "build_log", "p")
            .await
            .expect("seed parent");
        manager
            .save_scratchpad_entry(child, "analysis", "c")
            .await
            .expect("seed child");

        let tool = ScratchpadListTool {
            store: manager,
            parent_session_id: Some(parent),
            inherited_names: vec!["build_log".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };

        let text = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("list")
            .text_content();

        // Single header (not two sections) with the new `Origin` column.
        assert_eq!(
            text.matches("Origin").count(),
            1,
            "expected one Origin header, got: {text}",
        );
        assert!(text.contains("analysis"));
        assert!(text.contains("own"));
        assert!(text.contains("build_log"));
        assert!(text.contains("inherited"));
        // Old multi-section markers must be gone.
        assert!(!text.contains("inherited from parent"));
        assert!(!text.contains("inherited entries"));
        assert!(text.contains("2 entries total"));
    }

    #[tokio::test]
    async fn inherited_scratchpad_write_is_rejected() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "captured", "parent data")
            .await
            .expect("seed");

        let tool = ScratchpadWriteTool {
            store: manager.clone(),
            inherited_names: vec!["captured".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "captured", "content": "child override"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.is_error, "write to inherited name must error");
        assert!(result.text_content().contains("inherited read-only"));
        // Parent untouched.
        assert_eq!(
            manager
                .load_scratchpad_entry(parent, "captured")
                .await
                .unwrap(),
            Some("parent data".to_string())
        );
        // No child shadow row created.
        assert_eq!(
            manager
                .load_scratchpad_entry(child, "captured")
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn inherited_scratchpad_edit_is_rejected() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "captured", "parent data")
            .await
            .expect("seed");

        let tool = ScratchpadEditTool {
            store: manager.clone(),
            inherited_names: vec!["captured".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "captured", "content": "child override"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(result.text_content().contains("inherited read-only"));
        assert_eq!(
            manager
                .load_scratchpad_entry(parent, "captured")
                .await
                .unwrap(),
            Some("parent data".to_string())
        );
    }

    #[tokio::test]
    async fn inherited_scratchpad_delete_is_rejected() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "captured", "parent data")
            .await
            .expect("seed");

        let tool = ScratchpadDeleteTool {
            store: manager.clone(),
            inherited_names: vec!["captured".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "captured"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(result.text_content().contains("inherited read-only"));
        assert_eq!(
            manager
                .load_scratchpad_entry(parent, "captured")
                .await
                .unwrap(),
            Some("parent data".to_string())
        );
    }

    #[tokio::test]
    async fn write_to_unlisted_name_succeeds_without_touching_parent() {
        // Even when the parent has a same-named entry, if it isn't on the sub-agent's inherit
        // allowlist the child still writes its own independent row. (The block fires only on names
        // the parent actually granted; otherwise child sessions stay free to use any name.)
        // Parent's row is untouched.
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        manager
            .save_scratchpad_entry(parent, "shared", "parent original")
            .await
            .expect("seed");

        let write_tool = ScratchpadWriteTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };
        write_tool
            .execute(
                serde_json::json!({"name": "shared", "content": "child override"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");

        // Parent copy untouched.
        assert_eq!(
            manager
                .load_scratchpad_entry(parent, "shared")
                .await
                .expect("load parent"),
            Some("parent original".to_string())
        );
        // Child copy now holds its own version.
        assert_eq!(
            manager
                .load_scratchpad_entry(child, "shared")
                .await
                .expect("load child"),
            Some("child override".to_string())
        );
    }

    // -- session isolation --

    #[tokio::test]
    async fn sessions_have_independent_scratchpads() {
        let manager = Store::for_test().await;
        let session1 = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let session2 = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session1, "shared_name", "session1 data")
            .await
            .expect("save");
        manager
            .save_scratchpad_entry(session2, "shared_name", "session2 data")
            .await
            .expect("save");

        let loaded1 = manager
            .load_scratchpad_entry(session1, "shared_name")
            .await
            .expect("load");
        let loaded2 = manager
            .load_scratchpad_entry(session2, "shared_name")
            .await
            .expect("load");

        assert_eq!(loaded1, Some("session1 data".to_string()));
        assert_eq!(loaded2, Some("session2 data".to_string()));
    }

    // -- session lifecycle --

    #[tokio::test]
    async fn delete_session_removes_scratchpad() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "data", "content")
            .await
            .expect("save");

        manager.delete_session(session_id).await.expect("delete");

        let outputs = manager
            .load_all_scratchpad_entries(session_id)
            .await
            .expect("load");
        assert!(outputs.is_empty());
    }

    #[tokio::test]
    async fn clear_messages_removes_scratchpad() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        manager
            .save_scratchpad_entry(session_id, "data", "content")
            .await
            .expect("save");

        manager.clear_messages(session_id).await.expect("clear");

        let outputs = manager
            .load_all_scratchpad_entries(session_id)
            .await
            .expect("load");
        assert!(outputs.is_empty());
        assert!(manager.session_exists(session_id).await.expect("exists"));
    }

    // -- integration --

    #[tokio::test]
    async fn write_edit_read_list_delete_integration() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let session_id = session_id_for_test(session_id);

        let write_tool = ScratchpadWriteTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test().with_session_id(session_id.clone()),
        };
        write_tool
            .execute(
                serde_json::json!({"name": "test", "content": "hello world"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");

        let edit_tool = ScratchpadEditTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test().with_session_id(session_id.clone()),
        };
        edit_tool
            .execute(
                serde_json::json!({"name": "test", "old_string": "world", "new_string": "rust"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("edit");

        let read_tool = ScratchpadReadTool {
            store: manager.clone(),
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test().with_session_id(session_id.clone()),
        };
        let result = read_tool
            .execute(
                serde_json::json!({"name": "test"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read");
        assert!(result.text_content().contains("hello rust"));

        let list_tool = ScratchpadListTool {
            store: manager.clone(),
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test().with_session_id(session_id.clone()),
        };
        let result = list_tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("list");
        assert!(result.text_content().contains("1 entries total"));

        let delete_tool = ScratchpadDeleteTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test().with_session_id(session_id.clone()),
        };
        delete_tool
            .execute(
                serde_json::json!({"name": "test"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("delete");

        let result = list_tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("list");
        assert!(result.text_content().contains("empty"));
    }

    // -- scratchpad_load_file / scratchpad_save_file --

    #[tokio::test]
    async fn scratchpad_load_file_happy_path() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("input.txt");
        tokio::fs::write(&path, "hello scratchpad")
            .await
            .expect("write input");

        let tool = ScratchpadLoadFileTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_str().unwrap(), "name": "loaded"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("load");

        assert!(!result.is_error, "got: {}", result.text_content());
        assert!(result.text_content().contains("Loaded 16 bytes"));
        let stored = manager
            .load_scratchpad_entry(session_id, "loaded")
            .await
            .expect("load_scratchpad_entry");
        assert_eq!(stored.as_deref(), Some("hello scratchpad"));
    }

    #[tokio::test]
    async fn scratchpad_load_file_resolves_relative_path_against_cwd() {
        // A relative path resolves against the session cwd (like `read_file`), not the process
        // cwd, so `scratchpad_load_file` tracks `/cd`.
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        tokio::fs::write(dir.path().join("rel.txt"), "relative contents")
            .await
            .expect("write input");
        let cwd = crate::workspace::SharedCwd::new(dir.path().to_path_buf());

        let tool = ScratchpadLoadFileTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(cwd)
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": "rel.txt", "name": "loaded"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("load");

        assert!(!result.is_error, "got: {}", result.text_content());
        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "loaded")
                .await
                .expect("load_scratchpad_entry")
                .as_deref(),
            Some("relative contents"),
        );
    }

    /// A target that exists but cannot be stat'ed is refused without `force`. Read as absent, a
    /// symlink loop at the path would be written over and the save reported as a plain success.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_target_that_cannot_be_read_is_not_treated_as_absent() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "report", "final analysis")
            .await
            .expect("seed");
        let dir = tempfile::tempdir().expect("tempdir");
        // A link to itself: `metadata` fails with ELOOP, which is neither "exists" nor "absent".
        std::os::unix::fs::symlink(dir.path().join("out.txt"), dir.path().join("out.txt"))
            .expect("symlink");
        let cwd = crate::workspace::SharedCwd::new(dir.path().to_path_buf());
        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(cwd)
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "report", "path": "out.txt"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("the refusal is a tool error, not a failure");
        assert!(result.is_error, "got: {}", result.text_content());
        assert!(
            result.text_content().contains("could not be read"),
            "the refusal says why: {}",
            result.text_content()
        );
    }

    #[tokio::test]
    async fn scratchpad_save_file_resolves_relative_path_against_cwd() {
        // A relative save path lands under the session cwd, not the process cwd.
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "report", "final analysis")
            .await
            .expect("seed");
        let dir = tempfile::tempdir().expect("tempdir");
        let cwd = crate::workspace::SharedCwd::new(dir.path().to_path_buf());

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(cwd)
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "report", "path": "out.txt"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("save");

        assert!(!result.is_error, "got: {}", result.text_content());
        let written = tokio::fs::read_to_string(dir.path().join("out.txt"))
            .await
            .expect("read back");
        assert_eq!(written, "final analysis");
    }

    /// `scratchpad_save_file` has to take the same per-path write lock `write_file` does.
    ///
    /// Both are dispatched concurrently from one assistant message and both write through a temp
    /// file derived from the target, so two calls naming one path with no lock in common could
    /// interleave their write-then-rename and publish a spliced file. Asserted by holding the file
    /// tool's lock and showing the save cannot proceed behind it.
    #[tokio::test]
    async fn scratchpad_save_file_contends_with_write_file_for_the_same_path() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "report", "final analysis")
            .await
            .expect("seed");
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("out.txt");
        tokio::fs::write(&target, "existing")
            .await
            .expect("seed target");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::SharedCwd::new(dir.path().to_path_buf()))
                .with_session_id(session_id_for_test(session_id)),
        };

        // Whoever holds it, holds it against both tools: this is the lock `write_file` takes.
        let (_, held) = super::super::file::resolve_write_target(
            "write_file",
            &tool.site.cwd,
            &tool.scope,
            target.to_str().expect("path"),
        )
        .await
        .expect("take the write lock");

        let blocked = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            tool.execute(
                serde_json::json!({"name": "report", "path": "out.txt"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            ),
        )
        .await;
        assert!(
            blocked.is_err(),
            "the save must wait behind a write_file holding the same path",
        );

        drop(held);
        // `force`, because the fixture seeded `out.txt` before taking the lock and saving over an
        // existing file is refused without it. What this test is about is the lock, not the
        // clobber.
        let result = tool
            .execute(
                serde_json::json!({"name": "report", "path": "out.txt", "force": true}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("save once the lock is free");
        assert!(!result.is_error, "got: {}", result.text_content());
        assert_eq!(
            tokio::fs::read_to_string(&target).await.expect("read back"),
            "final analysis",
        );
    }

    /// Saving over a file that already exists is refused, and the confirmation says what it
    /// replaced when `force` allows it.
    ///
    /// This tool reads as the scratchpad's `write_file`, so a path the model named by mistake must
    /// not be gone with a success message on top of it.
    #[tokio::test]
    async fn scratchpad_save_file_refuses_to_replace_an_existing_file() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "report", "scratchpad bytes")
            .await
            .expect("seed");
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("out.txt");
        tokio::fs::write(&target, "THE USER WROTE THIS")
            .await
            .expect("seed target");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::SharedCwd::new(dir.path().to_path_buf()))
                .with_session_id(session_id_for_test(session_id)),
        };

        let refused = tool
            .execute(
                serde_json::json!({"name": "report", "path": "out.txt"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a refusal is a result, not an error");
        assert!(refused.is_error, "got: {}", refused.text_content());
        assert!(
            refused.text_content().contains("already exists"),
            "got: {}",
            refused.text_content()
        );
        assert_eq!(
            tokio::fs::read_to_string(&target).await.expect("read"),
            "THE USER WROTE THIS",
            "the file must survive the refusal"
        );

        let forced = tool
            .execute(
                serde_json::json!({"name": "report", "path": "out.txt", "force": true}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("force writes");
        assert!(!forced.is_error, "got: {}", forced.text_content());
        assert!(
            forced.text_content().contains("replacing 19 bytes"),
            "the confirmation must say what it replaced: {}",
            forced.text_content()
        );
    }

    #[tokio::test]
    async fn scratchpad_load_file_rejects_inherited_name() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("input.txt");
        tokio::fs::write(&path, "irrelevant")
            .await
            .expect("write input");

        let tool = ScratchpadLoadFileTool {
            store: manager.clone(),
            inherited_names: vec!["captured".to_string()],
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_session_id(session_id_for_test(child)),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_str().unwrap(), "name": "captured"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(result.text_content().contains("inherited read-only"));
        assert_eq!(
            manager
                .load_scratchpad_entry(child, "captured")
                .await
                .expect("load"),
            None,
            "no child shadow row should be created"
        );
    }

    #[tokio::test]
    async fn scratchpad_load_file_rejects_image_with_mime() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pic.png");
        // Minimal PNG signature + IHDR chunk bytes, enough for `infer` to fingerprint without
        // needing a syntactically valid image.
        let png_bytes: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk header
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1 dimensions
            0x08, 0x00, 0x00, 0x00, 0x00, // depth, type, etc.
            0xFF, 0xFE, 0xFD, 0xFC, // CRC placeholder + body, non-UTF-8
        ];
        tokio::fs::write(&path, png_bytes).await.expect("write png");

        let tool = ScratchpadLoadFileTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_str().unwrap(), "name": "img"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        let text = result.text_content();
        assert!(result.is_error, "got: {text}");
        assert!(text.contains("not valid UTF-8"), "message: {text}");
        assert!(text.contains("image/png"), "message: {text}");
        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "img")
                .await
                .unwrap(),
            None,
            "binary file must not produce a scratchpad row"
        );
    }

    #[tokio::test]
    async fn scratchpad_load_file_unknown_binary_has_no_mime_line() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("blob.bin");
        // A short run of 0xFF bytes won't match any `infer` signature.
        tokio::fs::write(&path, &[0xFF_u8; 8])
            .await
            .expect("write blob");

        let tool = ScratchpadLoadFileTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_str().unwrap(), "name": "blob"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        let text = result.text_content();
        assert!(result.is_error);
        assert!(text.contains("not valid UTF-8"));
        assert!(
            !text.contains("Detected MIME"),
            "no MIME line expected for unknown binary, message: {text}",
        );
    }

    #[tokio::test]
    async fn scratchpad_save_file_happy_path() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "report", "final analysis")
            .await
            .expect("seed");

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("subdir").join("out.txt");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "report", "path": path.to_str().unwrap()}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("save");

        assert!(!result.is_error, "got: {}", result.text_content());
        let written = tokio::fs::read_to_string(&path).await.expect("read back");
        assert_eq!(written, "final analysis");
    }

    #[tokio::test]
    async fn scratchpad_save_file_reads_inherited_from_parent() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        manager
            .save_scratchpad_entry(parent, "build_log", "parent-only payload")
            .await
            .expect("seed");

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("log.txt");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            store: manager,
            parent_session_id: Some(parent),
            inherited_names: vec!["build_log".to_string()],
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_session_id(session_id_for_test(child)),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "build_log", "path": path.to_str().unwrap()}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("save");

        assert!(!result.is_error, "got: {}", result.text_content());
        let written = tokio::fs::read_to_string(&path).await.expect("read back");
        assert_eq!(written, "parent-only payload");
    }

    #[tokio::test]
    async fn scratchpad_save_file_missing_entry_errors() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.txt");

        let tool = ScratchpadSaveFileTool {
            read_tracker: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            scope: crate::workspace::WriteScope::unconfined(),
            store: manager,
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "missing", "path": path.to_str().unwrap()}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(result.is_err(), "missing entry should propagate an error");
    }

    // -- scratchpad_rename --

    #[tokio::test]
    async fn scratchpad_rename_happy_path() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "draft", "payload")
            .await
            .expect("seed");

        let tool = ScratchpadRenameTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "draft", "new": "final"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("rename");

        assert!(!result.is_error, "got: {}", result.text_content());
        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "draft")
                .await
                .unwrap(),
            None,
            "old name must be gone after rename"
        );
        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "final")
                .await
                .unwrap(),
            Some("payload".to_string()),
        );
    }

    #[tokio::test]
    async fn scratchpad_rename_source_not_found() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let tool = ScratchpadRenameTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "absent", "new": "whatever"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(result.text_content().contains("not found"));
        // No row should appear under the target name.
        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "whatever")
                .await
                .unwrap(),
            None,
        );
    }

    #[tokio::test]
    async fn scratchpad_rename_target_already_exists() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "src", "src-content")
            .await
            .expect("seed src");
        manager
            .save_scratchpad_entry(session_id, "dst", "dst-content")
            .await
            .expect("seed dst");

        let tool = ScratchpadRenameTool {
            store: manager.clone(),
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "src", "new": "dst"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(result.text_content().contains("already exists"));
        // Both rows must be untouched.
        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "src")
                .await
                .unwrap(),
            Some("src-content".to_string()),
        );
        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "dst")
                .await
                .unwrap(),
            Some("dst-content".to_string()),
        );
    }

    #[tokio::test]
    async fn scratchpad_rename_blocks_inherited_source() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        manager
            .save_scratchpad_entry(parent, "captured", "parent-data")
            .await
            .expect("seed");

        let tool = ScratchpadRenameTool {
            store: manager.clone(),
            inherited_names: vec!["captured".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "captured", "new": "mine"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(result.text_content().contains("inherited read-only"));
        // Parent's row stays intact, child has no shadow under either name.
        assert_eq!(
            manager
                .load_scratchpad_entry(parent, "captured")
                .await
                .unwrap(),
            Some("parent-data".to_string()),
        );
        assert_eq!(
            manager.load_scratchpad_entry(child, "mine").await.unwrap(),
            None,
        );
    }

    #[tokio::test]
    async fn scratchpad_rename_blocks_inherited_target() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        manager
            .save_scratchpad_entry(child, "mine", "child-data")
            .await
            .expect("seed");

        let tool = ScratchpadRenameTool {
            store: manager.clone(),
            inherited_names: vec!["captured".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };
        let result = tool
            .execute(
                serde_json::json!({"old": "mine", "new": "captured"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
        assert!(result.text_content().contains("inherited read-only"));
        // Child's source row must stay intact, no shadow under inherited name.
        assert_eq!(
            manager.load_scratchpad_entry(child, "mine").await.unwrap(),
            Some("child-data".to_string()),
        );
        assert_eq!(
            manager
                .load_scratchpad_entry(child, "captured")
                .await
                .unwrap(),
            None,
        );
    }

    // -- scratchpad_merge --

    #[tokio::test]
    async fn scratchpad_merge_concat_with_headers_default() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        for (name, body) in [("a", "first"), ("b", "second"), ("c", "third")] {
            manager
                .save_scratchpad_entry(session_id, name, body)
                .await
                .expect("seed");
        }

        let tool = ScratchpadMergeTool {
            store: manager.clone(),
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"sources": ["a", "b", "c"], "target": "merged"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("merge");

        assert!(!result.is_error, "got: {}", result.text_content());
        let stored = manager
            .load_scratchpad_entry(session_id, "merged")
            .await
            .expect("load")
            .expect("present");
        assert!(stored.contains("--- a ---"));
        assert!(stored.contains("--- b ---"));
        assert!(stored.contains("--- c ---"));
        assert!(stored.contains("first"));
        assert!(stored.contains("second"));
        assert!(stored.contains("third"));
    }

    /// `prefix` adds every own entry that starts with it, in name order, after the named sources,
    /// and never the target itself; naming nothing and matching nothing is refused rather than
    /// written as an empty entry.
    #[tokio::test]
    async fn scratchpad_merge_prefix_selects_own_entries_in_name_order() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        // Written out of name order, with an unrelated entry and a stale target in the way.
        for (name, body) in [
            ("report-2", "second"),
            ("other", "elsewhere"),
            ("report-1", "first"),
            ("report-all", "stale"),
            ("intro", "preface"),
        ] {
            manager
                .save_scratchpad_entry(session_id, name, body)
                .await
                .expect("seed");
        }
        let tool = ScratchpadMergeTool {
            store: manager.clone(),
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };

        let result = tool
            .execute(
                serde_json::json!({
                    "sources": ["intro"],
                    "prefix": "report-",
                    "target": "report-all",
                    "format": "concat",
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("merge");
        assert!(!result.is_error, "got: {}", result.text_content());
        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "report-all")
                .await
                .expect("load")
                .expect("present"),
            "preface\nfirst\nsecond",
            "named sources first, then the prefix matches in name order, never the target"
        );
        assert!(
            manager
                .load_scratchpad_entry(session_id, "report-1")
                .await
                .expect("load")
                .is_some(),
            "sources are kept"
        );

        let nothing = tool
            .execute(
                serde_json::json!({"prefix": "missing-", "target": "empty"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a refusal is a tool error, not a failure");
        assert!(nothing.is_error);
        assert!(
            manager
                .load_scratchpad_entry(session_id, "empty")
                .await
                .expect("load")
                .is_none(),
            "nothing matched, so nothing is written"
        );
    }

    #[tokio::test]
    async fn scratchpad_merge_json_array_parses_valid_and_quotes_invalid() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "obj", r#"{"k":1}"#)
            .await
            .expect("seed obj");
        manager
            .save_scratchpad_entry(session_id, "num", "42")
            .await
            .expect("seed num");
        manager
            .save_scratchpad_entry(session_id, "plain", "not json")
            .await
            .expect("seed plain");

        let tool = ScratchpadMergeTool {
            store: manager.clone(),
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };
        tool.execute(
            serde_json::json!({
                "sources": ["obj", "num", "plain"],
                "target": "combined",
                "format": "json_array"
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("merge");

        let stored = manager
            .load_scratchpad_entry(session_id, "combined")
            .await
            .expect("load")
            .expect("present");
        let parsed: serde_json::Value = serde_json::from_str(&stored).expect("valid JSON");
        let array = parsed.as_array().expect("array");
        assert_eq!(array.len(), 3);
        assert_eq!(array[0]["k"], serde_json::json!(1));
        assert_eq!(array[1], serde_json::json!(42));
        assert_eq!(array[2], serde_json::json!("not json"));
    }

    #[tokio::test]
    async fn scratchpad_merge_blocks_inherited_target() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        manager
            .save_scratchpad_entry(child, "src", "data")
            .await
            .expect("seed");

        let tool = ScratchpadMergeTool {
            store: manager.clone(),
            parent_session_id: Some(parent),
            inherited_names: vec!["shadow".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };
        let result = tool
            .execute(
                serde_json::json!({"sources": ["src"], "target": "shadow"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute");
        assert!(result.is_error);
        assert!(result.text_content().contains("inherited read-only"));
        assert_eq!(
            manager
                .load_scratchpad_entry(child, "shadow")
                .await
                .unwrap(),
            None,
        );
    }

    #[tokio::test]
    async fn scratchpad_merge_reads_inherited_sources() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = manager
            .create_child_session(
                parent,
                None,
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        manager
            .save_scratchpad_entry(parent, "shared", "parent payload")
            .await
            .expect("seed parent");
        manager
            .save_scratchpad_entry(child, "mine", "child payload")
            .await
            .expect("seed child");

        let tool = ScratchpadMergeTool {
            store: manager.clone(),
            parent_session_id: Some(parent),
            inherited_names: vec!["shared".to_string()],
            site: crate::session::ToolSite::for_test().with_session_id(session_id_for_test(child)),
        };
        tool.execute(
            serde_json::json!({"sources": ["shared", "mine"], "target": "combined"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("merge");

        let stored = manager
            .load_scratchpad_entry(child, "combined")
            .await
            .expect("load")
            .expect("present");
        assert!(stored.contains("parent payload"));
        assert!(stored.contains("child payload"));
    }

    #[tokio::test]
    async fn scratchpad_merge_missing_source_aborts_without_writing() {
        let manager = Store::for_test().await;
        let session_id = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        manager
            .save_scratchpad_entry(session_id, "real", "stuff")
            .await
            .expect("seed");

        let tool = ScratchpadMergeTool {
            store: manager.clone(),
            parent_session_id: None,
            inherited_names: Vec::new(),
            site: crate::session::ToolSite::for_test()
                .with_session_id(session_id_for_test(session_id)),
        };
        let result = tool
            .execute(
                serde_json::json!({"sources": ["real", "missing"], "target": "out"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(result.is_err(), "missing source must propagate an error");
        // Target must not have been written.
        assert_eq!(
            manager
                .load_scratchpad_entry(session_id, "out")
                .await
                .unwrap(),
            None,
            "target row must not exist after a failed merge",
        );
    }
}
