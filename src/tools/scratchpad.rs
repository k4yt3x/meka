//! Scratchpad: session-scoped, persisted key/value text store. Oversized
//! tool outputs are automatically redirected here and replaced inline with a
//! preview + handle, keeping the conversation context bounded. Provides
//! write/read/edit/list/delete operations plus a regex search mode.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::{ContentBlock, Message, ToolDefinition, ToolResultContent};
use crate::session::SessionManager;

use super::util::{MAX_SEARCH_MATCHES, search_lines};
use super::{Tool, ToolOutput};

/// Tool result text blocks larger than this are persisted to the database and
/// replaced with a preview + handle in the conversation context.
pub const MAX_INLINE_RESULT_CHARS: usize = 30_000;

/// Number of characters included in the inline preview.
const PREVIEW_CHARS: usize = 2_000;

/// Default character limit when reading back a persisted output.
const DEFAULT_READ_LIMIT: usize = 30_000;

pub(crate) fn format_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.1} KB", bytes as f64 / 1_024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

async fn resolve_session_id(
    session_id: &Arc<RwLock<Option<Uuid>>>,
    tool_name: &str,
) -> Result<Uuid> {
    session_id
        .read()
        .await
        .ok_or_else(|| AgshError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: "no active session".to_string(),
        })
}

/// Build a map from tool_use_id to (tool_name, input) for the ToolUse blocks
/// in an assistant message.
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
        "Output saved to scratchpad \"{}\" ({} characters). \
         Use scratchpad_read to access it.",
        name, size,
    )
}

fn build_large_output_preview(name: &str, text: &str) -> String {
    let size = text.len();
    let preview_end = text.floor_char_boundary(PREVIEW_CHARS.min(size));
    let preview = &text[..preview_end];
    let has_more = preview_end < size;

    let mut replacement = format!(
        "<large-output name=\"{}\" size=\"{}\">\n\
         Output too large ({}). Read with `scratchpad_read` — use \
         `limit: {}` to load the full content in one call, or page \
         with `offset`/`limit` if a partial read is enough.\n\n\
         Preview (first {} characters):\n\
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

/// Save tool results to the scratchpad when the agent explicitly requested it
/// via the `scratchpad` parameter on a tool call. Replaces the inline result
/// with a brief reference.
pub async fn save_explicit_scratchpad_results(
    session_manager: &SessionManager,
    session_id: Uuid,
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

            let size = text.len();
            session_manager
                .save_tool_output(session_id, name, &text)
                .await?;

            *content = vec![ToolResultContent::Text {
                text: build_scratchpad_reference(name, size),
            }];
        }
    }
    Ok(())
}

/// Check each text block in tool results. If oversized, persist to DB
/// and replace with a preview + handle. Names are derived from the tool
/// call's `scratchpad_hint` (MCP adapters) or the tool name otherwise, with
/// a numeric suffix on collision. `hints` is typically the per-turn map
/// owned by the agent — empty is fine.
pub async fn persist_oversized_results(
    session_manager: &SessionManager,
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
                    if text.len() <= MAX_INLINE_RESULT_CHARS {
                        continue;
                    }

                    counter += 1;
                    let name = format!("{}_{}", base_name, counter);

                    session_manager
                        .save_tool_output(session_id, &name, text)
                        .await?;

                    *text = build_large_output_preview(&name, text);
                }
            }
        }
    }
    Ok(())
}

pub(super) struct ScratchpadWriteTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
}

#[async_trait]
impl Tool for ScratchpadWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_write".to_string(),
            description: "Store content in the scratchpad under the given name \u{2014} a \
                session-scoped working memory that persists across turns without consuming \
                conversation context. If the name already exists, the content is overwritten. \
                Use this to save intermediate results, extracted text, accumulated data, or \
                research notes. You can also save tool output directly by adding a 'scratchpad' \
                parameter to any tool call."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the scratchpad entry"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to store"
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let name = input["name"]
            .as_str()
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "scratchpad_write".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;
        let content = input["content"]
            .as_str()
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "scratchpad_write".to_string(),
                message: "missing 'content' parameter".to_string(),
            })?;

        let session_id = resolve_session_id(&self.session_id, "scratchpad_write").await?;

        self.session_manager
            .save_tool_output(session_id, name, content)
            .await?;

        Ok(ToolOutput::text(
            format!(
                "Stored {} characters as scratchpad entry \"{}\"",
                content.len(),
                name,
            ),
            false,
        ))
    }
}

pub(super) struct ScratchpadReadTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
}

#[async_trait]
impl Tool for ScratchpadReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_read".to_string(),
            description: format!(
                "Read or search a scratchpad entry by name. Default returns {} \
                 characters from offset; pass a larger `limit` (no hard cap) to load the full \
                 entry in one call, or page with `offset`/`limit` for partial reads. Provide \
                 `regex` to return matching lines (max {}) instead of a character range. Also \
                 used to access content referenced by <large-output> tags — pass the `size` \
                 value from the tag as `limit` when you intend to read everything.",
                DEFAULT_READ_LIMIT, MAX_SEARCH_MATCHES,
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry name"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Character offset to start reading from. Default: 0."
                    },
                    "limit": {
                        "type": "integer",
                        "description": format!("Maximum characters to return. Default: {}.", DEFAULT_READ_LIMIT)
                    },
                    "regex": {
                        "type": "string",
                        "description": format!(
                            "If provided, search the entry with this regex pattern \
                             and return matching lines (max {} matches) instead of a character range.",
                            MAX_SEARCH_MATCHES,
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let name = input["name"]
            .as_str()
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "scratchpad_read".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        let session_id = resolve_session_id(&self.session_id, "scratchpad_read").await?;

        let content = self
            .session_manager
            .load_tool_output(session_id, name)
            .await?
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "scratchpad_read".to_string(),
                message: format!("scratchpad entry \"{}\" not found", name),
            })?;

        if let Some(pattern) = input.get("regex").and_then(|v| v.as_str()) {
            return search_lines(&content, pattern, "scratchpad_read");
        }

        read_mode(&content, &input)
    }
}

fn read_mode(content: &str, input: &serde_json::Value) -> Result<ToolOutput> {
    let offset = input["offset"].as_u64().unwrap_or(0) as usize;
    let limit = input["limit"].as_u64().unwrap_or(DEFAULT_READ_LIMIT as u64) as usize;
    let total = content.len();

    if offset >= total {
        return Ok(ToolOutput::text(
            format!(
                "Offset {} exceeds content length ({} characters)",
                offset, total
            ),
            true,
        ));
    }

    let start = content.floor_char_boundary(offset);
    let end = content.floor_char_boundary((start + limit).min(total));
    let slice = &content[start..end];

    Ok(ToolOutput::text(
        format!(
            "{}\n\n(showing characters {}..{} of {})",
            slice, start, end, total
        ),
        false,
    ))
}

pub(super) struct ScratchpadEditTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
}

#[async_trait]
impl Tool for ScratchpadEditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_edit".to_string(),
            description: "Edit a scratchpad entry in place. Provide 'content' to fully \
                overwrite, or 'old_string'/'new_string' for targeted string replacement \
                (like edit_file)."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry name to edit"
                    },
                    "content": {
                        "type": "string",
                        "description": "Full replacement content (mutually exclusive with old_string/new_string)"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace all occurrences. Defaults to false."
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let name = input["name"]
            .as_str()
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        let session_id = resolve_session_id(&self.session_id, "scratchpad_edit").await?;

        // Full overwrite mode
        if let Some(new_content) = input.get("content").and_then(|v| v.as_str()) {
            let updated = self
                .session_manager
                .update_tool_output(session_id, name, new_content)
                .await?;

            return if updated {
                Ok(ToolOutput::text(
                    format!(
                        "Scratchpad entry \"{}\" overwritten ({} characters)",
                        name,
                        new_content.len()
                    ),
                    false,
                ))
            } else {
                Ok(ToolOutput::text(
                    format!("Scratchpad entry \"{}\" not found", name),
                    true,
                ))
            };
        }

        // String replacement mode
        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: "provide either 'content' for full overwrite \
                    or 'old_string'/'new_string' for replacement"
                    .to_string(),
            })?;
        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: "missing 'new_string' parameter".to_string(),
            })?;
        let replace_all = input["replace_all"].as_bool().unwrap_or(false);

        let existing = self
            .session_manager
            .load_tool_output(session_id, name)
            .await?
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "scratchpad_edit".to_string(),
                message: format!("scratchpad entry \"{}\" not found", name),
            })?;

        if !existing.contains(old_string) {
            return Ok(ToolOutput::text(
                format!(
                    "Error: '{}' not found in scratchpad entry \"{}\"",
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

        self.session_manager
            .update_tool_output(session_id, name, &updated_content)
            .await?;

        Ok(ToolOutput::text(
            format!(
                "Scratchpad entry \"{}\": replaced {} occurrence(s) ({} characters)",
                name,
                count,
                updated_content.len(),
            ),
            false,
        ))
    }
}

pub(super) struct ScratchpadListTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
}

#[async_trait]
impl Tool for ScratchpadListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_list".to_string(),
            description: "List all scratchpad entries in the current session with their name, \
                size, and creation time."
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let session_id = resolve_session_id(&self.session_id, "scratchpad_list").await?;

        let entries = self.session_manager.list_tool_outputs(session_id).await?;

        if entries.is_empty() {
            return Ok(ToolOutput::text("Scratchpad is empty.".to_string(), false));
        }

        let mut output = format!("{:<24} {:<10} {}\n", "Name", "Size", "Created");
        for entry in &entries {
            output.push_str(&format!(
                "{:<24} {:<10} {}\n",
                entry.name,
                format_size(entry.size),
                &entry.created_at[..19.min(entry.created_at.len())],
            ));
        }
        output.push_str(&format!("\n{} entries total", entries.len()));

        Ok(ToolOutput::text(output, false))
    }
}

pub(super) struct ScratchpadDeleteTool {
    pub session_manager: SessionManager,
    pub session_id: Arc<RwLock<Option<Uuid>>>,
}

#[async_trait]
impl Tool for ScratchpadDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "scratchpad_delete".to_string(),
            description: "Delete a scratchpad entry by name to free up space.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The scratchpad entry name to delete"
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let name = input["name"]
            .as_str()
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "scratchpad_delete".to_string(),
                message: "missing 'name' parameter".to_string(),
            })?;

        let session_id = resolve_session_id(&self.session_id, "scratchpad_delete").await?;

        let deleted = self
            .session_manager
            .delete_tool_output(session_id, name)
            .await?;

        if deleted {
            Ok(ToolOutput::text(
                format!("Scratchpad entry \"{}\" deleted", name),
                false,
            ))
        } else {
            Ok(ToolOutput::text(
                format!("Scratchpad entry \"{}\" not found", name),
                true,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::provider::{ContentBlock, Role};

    fn text_content(output: &ToolOutput) -> String {
        ContentBlock::tool_result_text_content(&output.content)
    }

    async fn test_manager() -> SessionManager {
        SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("failed to open in-memory database")
    }

    fn test_session_id(uuid: Uuid) -> Arc<RwLock<Option<Uuid>>> {
        Arc::new(RwLock::new(Some(uuid)))
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

    #[tokio::test]
    async fn test_persist_oversized_results_replaces_large_text() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        let large_text = "x".repeat(MAX_INLINE_RESULT_CHARS + 1000);
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
            assert!(text.contains("name=\"execute_command_1\""));
            assert!(text.contains("scratchpad_read"));
            assert!(!text.contains(&large_text));
        } else {
            panic!("expected ToolResult");
        }

        let loaded = manager
            .load_tool_output(session_id, "execute_command_1")
            .await
            .expect("load");
        assert_eq!(loaded, Some(large_text));
    }

    #[tokio::test]
    async fn test_persist_oversized_results_leaves_small_text() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

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
    async fn test_explicit_scratchpad_save() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

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

        save_explicit_scratchpad_results(&manager, session_id, &assistant_msg, &mut results)
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
            .load_tool_output(session_id, "cmd_output")
            .await
            .expect("load");
        assert_eq!(loaded, Some("hi\n".to_string()));
    }

    #[tokio::test]
    async fn test_explicit_scratchpad_not_requested() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

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

        save_explicit_scratchpad_results(&manager, session_id, &assistant_msg, &mut results)
            .await
            .expect("save");

        // Result should be unchanged
        if let ContentBlock::ToolResult { content, .. } = &results[0] {
            assert_eq!(ContentBlock::tool_result_text_content(content), "hi\n");
        }
    }

    #[tokio::test]
    async fn test_explicit_scratchpad_ignores_non_scratchpad_keys() {
        // Regression test for the `render_image` clobber bug: when a tool uses
        // `from_scratchpad` (as an input-source param) instead of `scratchpad`
        // (the output-destination convention), the agent-layer save must not
        // touch the pre-existing scratchpad entry.
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "img", "original base64 data")
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

        save_explicit_scratchpad_results(&manager, session_id, &assistant_msg, &mut results)
            .await
            .expect("save");

        // Pre-existing scratchpad entry should be untouched.
        let loaded = manager
            .load_tool_output(session_id, "img")
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
    async fn test_scratchpad_write() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        let tool = ScratchpadWriteTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "greeting", "content": "hello world"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("greeting"));

        let loaded = manager
            .load_tool_output(session_id, "greeting")
            .await
            .expect("load");
        assert_eq!(loaded, Some("hello world".to_string()));
    }

    #[tokio::test]
    async fn test_scratchpad_write_overwrites() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "notes", "old content")
            .await
            .expect("save");

        let tool = ScratchpadWriteTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
        };

        tool.execute(
            serde_json::json!({"name": "notes", "content": "new content"}),
            CancellationToken::new(),
        )
        .await
        .expect("execute");

        let loaded = manager
            .load_tool_output(session_id, "notes")
            .await
            .expect("load");
        assert_eq!(loaded, Some("new content".to_string()));
    }

    // -- scratchpad_read --

    #[tokio::test]
    async fn test_scratchpad_read() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "data", "line1\nline2\nline3\n")
            .await
            .expect("save");

        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "data"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        let text = text_content(&result);
        assert!(text.contains("line1"));
        assert!(text.contains("line3"));
    }

    #[tokio::test]
    async fn test_scratchpad_read_with_offset_and_limit() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "abc", "abcdefghij")
            .await
            .expect("save");

        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "abc", "offset": 3, "limit": 4}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        let text = text_content(&result);
        assert!(text.contains("defg"));
        assert!(text.contains("showing characters 3..7 of 10"));
    }

    #[tokio::test]
    async fn test_scratchpad_read_search_mode() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(
                session_id,
                "fruits",
                "apple\nbanana\napricot\ncherry\navocado\n",
            )
            .await
            .expect("save");

        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "fruits", "regex": "^a"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        let text = text_content(&result);
        assert!(text.contains("1:apple"));
        assert!(text.contains("3:apricot"));
        assert!(text.contains("5:avocado"));
        assert!(!text.contains("banana"));
    }

    #[tokio::test]
    async fn test_scratchpad_read_not_found() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        let tool = ScratchpadReadTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
        };

        let result = tool.execute(
            serde_json::json!({"name": "nonexistent"}),
            CancellationToken::new(),
        );
        assert!(result.await.is_err());
    }

    // -- scratchpad_edit --

    #[tokio::test]
    async fn test_scratchpad_edit_overwrite() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "doc", "old content")
            .await
            .expect("save");

        let tool = ScratchpadEditTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "doc", "content": "new content"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("overwritten"));

        let loaded = manager
            .load_tool_output(session_id, "doc")
            .await
            .expect("load");
        assert_eq!(loaded, Some("new content".to_string()));
    }

    #[tokio::test]
    async fn test_scratchpad_edit_replacement() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "doc", "hello world hello")
            .await
            .expect("save");

        let tool = ScratchpadEditTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "doc", "old_string": "hello", "new_string": "hi"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(text_content(&result).contains("1 occurrence(s)"));

        let loaded = manager
            .load_tool_output(session_id, "doc")
            .await
            .expect("load");
        assert_eq!(loaded, Some("hi world hello".to_string()));
    }

    #[tokio::test]
    async fn test_scratchpad_edit_replace_all() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "doc", "foo bar foo baz foo")
            .await
            .expect("save");

        let tool = ScratchpadEditTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(
                serde_json::json!({
                    "name": "doc",
                    "old_string": "foo",
                    "new_string": "qux",
                    "replace_all": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(text_content(&result).contains("3 occurrence(s)"));

        let loaded = manager
            .load_tool_output(session_id, "doc")
            .await
            .expect("load");
        assert_eq!(loaded, Some("qux bar qux baz qux".to_string()));
    }

    // -- scratchpad_list --

    #[tokio::test]
    async fn test_scratchpad_list_empty() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        let tool = ScratchpadListTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("execute");

        assert!(text_content(&result).contains("empty"));
    }

    #[tokio::test]
    async fn test_scratchpad_list_with_entries() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "notes", "content one")
            .await
            .expect("save");
        manager
            .save_tool_output(session_id, "data", "content two")
            .await
            .expect("save");

        let tool = ScratchpadListTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("execute");

        let text = text_content(&result);
        assert!(text.contains("notes"));
        assert!(text.contains("data"));
        assert!(text.contains("2 entries total"));
    }

    // -- scratchpad_delete --

    #[tokio::test]
    async fn test_scratchpad_delete() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "temp", "temp data")
            .await
            .expect("save");

        let tool = ScratchpadDeleteTool {
            session_manager: manager.clone(),
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "temp"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("deleted"));

        let loaded = manager
            .load_tool_output(session_id, "temp")
            .await
            .expect("load");
        assert_eq!(loaded, None);
    }

    #[tokio::test]
    async fn test_scratchpad_delete_not_found() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        let tool = ScratchpadDeleteTool {
            session_manager: manager,
            session_id: test_session_id(session_id),
        };

        let result = tool
            .execute(
                serde_json::json!({"name": "nonexistent"}),
                CancellationToken::new(),
            )
            .await
            .expect("execute");

        assert!(result.is_error);
    }

    // -- session isolation --

    #[tokio::test]
    async fn test_sessions_have_independent_scratchpads() {
        let manager = test_manager().await;
        let session1 = manager.create_session().await.expect("create");
        let session2 = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session1, "shared_name", "session1 data")
            .await
            .expect("save");
        manager
            .save_tool_output(session2, "shared_name", "session2 data")
            .await
            .expect("save");

        let loaded1 = manager
            .load_tool_output(session1, "shared_name")
            .await
            .expect("load");
        let loaded2 = manager
            .load_tool_output(session2, "shared_name")
            .await
            .expect("load");

        assert_eq!(loaded1, Some("session1 data".to_string()));
        assert_eq!(loaded2, Some("session2 data".to_string()));
    }

    // -- session lifecycle --

    #[tokio::test]
    async fn test_delete_session_removes_scratchpad() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "data", "content")
            .await
            .expect("save");

        manager.delete_session(session_id).await.expect("delete");

        let outputs = manager
            .load_all_tool_outputs(session_id)
            .await
            .expect("load");
        assert!(outputs.is_empty());
    }

    #[tokio::test]
    async fn test_clear_messages_removes_scratchpad() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");

        manager
            .save_tool_output(session_id, "data", "content")
            .await
            .expect("save");

        manager.clear_messages(session_id).await.expect("clear");

        let outputs = manager
            .load_all_tool_outputs(session_id)
            .await
            .expect("load");
        assert!(outputs.is_empty());
        assert!(manager.session_exists(session_id).await.expect("exists"));
    }

    // -- integration --

    #[tokio::test]
    async fn test_write_edit_read_list_delete_integration() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create");
        let sid = test_session_id(session_id);

        let write_tool = ScratchpadWriteTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
        };
        write_tool
            .execute(
                serde_json::json!({"name": "test", "content": "hello world"}),
                CancellationToken::new(),
            )
            .await
            .expect("write");

        let edit_tool = ScratchpadEditTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
        };
        edit_tool
            .execute(
                serde_json::json!({"name": "test", "old_string": "world", "new_string": "rust"}),
                CancellationToken::new(),
            )
            .await
            .expect("edit");

        let read_tool = ScratchpadReadTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
        };
        let result = read_tool
            .execute(
                serde_json::json!({"name": "test"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        assert!(text_content(&result).contains("hello rust"));

        let list_tool = ScratchpadListTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
        };
        let result = list_tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("list");
        assert!(text_content(&result).contains("1 entries total"));

        let delete_tool = ScratchpadDeleteTool {
            session_manager: manager.clone(),
            session_id: sid.clone(),
        };
        delete_tool
            .execute(
                serde_json::json!({"name": "test"}),
                CancellationToken::new(),
            )
            .await
            .expect("delete");

        let result = list_tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("list");
        assert!(text_content(&result).contains("empty"));
    }
}
