use std::path::PathBuf;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use base64::Engine;

use crate::error::{AgshError, Result};
use crate::image::{ImageHandling, classify_extension, prepare_image_payload};
use crate::permission::Permission;
use crate::provider::{ImageSource, ToolDefinition, ToolResultContent};

use super::util::{require_str, truncate_string};
use super::{ReadTracker, Tool, ToolOutput};

pub(super) struct ReadFileTool {
    pub read_tracker: ReadTracker,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read the contents of a file at the given path. Supported raster \
                          image files (PNG, JPEG, GIF, WebP, BMP, TIFF, ICO, HDR, EXR, \
                          TGA, PNM, QOI, DDS, Farbfeld) are returned as a multimodal \
                          content block; non-native formats are transparently converted \
                          to PNG. Only read image files if the current model supports \
                          vision input."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-based). Optional."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Optional."
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["path"]
            }),
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
        let path = input["path"]
            .as_str()
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "read_file".to_string(),
                message: "missing 'path' parameter".to_string(),
            })?
            .to_string();

        let canonical = tokio::fs::canonicalize(&path)
            .await
            .unwrap_or_else(|_| PathBuf::from(&path));

        // Detect image files and return multimodal content, converting
        // non-native formats (TIFF, ICO, etc.) to PNG along the way.
        let extension = PathBuf::from(&path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        let handling = extension
            .as_deref()
            .map(classify_extension)
            .unwrap_or(ImageHandling::Unsupported);

        if !matches!(handling, ImageHandling::Unsupported) {
            let data = tokio::fs::read(&path)
                .await
                .map_err(|error| AgshError::ToolExecution {
                    tool_name: "read_file".to_string(),
                    message: format!("failed to read '{}': {}", path, error),
                })?;

            let (media_type, payload) = match prepare_image_payload(handling, &data) {
                Ok(pair) => pair,
                Err(message) => {
                    return Ok(ToolOutput::text(
                        format!("Error: image '{}': {}", path, message),
                        true,
                    ));
                }
            };

            let base64_data = base64::engine::general_purpose::STANDARD.encode(&payload);

            self.read_tracker.write().await.insert(canonical);

            return Ok(ToolOutput {
                content: vec![
                    ToolResultContent::Text {
                        text: format!("[Image: {}]", path),
                    },
                    ToolResultContent::Image {
                        source: ImageSource {
                            source_type: "base64".to_string(),
                            media_type: media_type.to_string(),
                            data: base64_data,
                        },
                    },
                ],
                is_error: false,
            });
        }

        const DEFAULT_LINE_LIMIT: usize = 2000;

        let offset = input["offset"].as_u64().map(|value| value as usize);
        let limit = input["limit"].as_u64().map(|value| value as usize);

        let content =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|error| AgshError::ToolExecution {
                    tool_name: "read_file".to_string(),
                    message: format!("failed to read '{}': {}", path, error),
                })?;

        let total_lines = content.lines().count();
        let effective_offset = offset.unwrap_or(0);
        let effective_limit = limit.unwrap_or(DEFAULT_LINE_LIMIT);

        let result: String = content
            .lines()
            .skip(effective_offset)
            .take(effective_limit)
            .collect::<Vec<_>>()
            .join("\n");

        let result = if offset.is_none() && limit.is_none() && total_lines > DEFAULT_LINE_LIMIT {
            format!(
                "{}\n\n... (showing first {} of {} lines, use offset/limit to read more)",
                result, DEFAULT_LINE_LIMIT, total_lines,
            )
        } else {
            result
        };

        self.read_tracker.write().await.insert(canonical);

        Ok(ToolOutput::text(result, false))
    }
}

pub(super) struct EditFileTool {
    pub read_tracker: ReadTracker,
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_string(),
            description: "Make a string replacement in a file. By default replaces the first occurrence of 'old_string' with 'new_string'. Set 'replace_all' to true to replace every occurrence. The file must have been read with read_file first unless 'force' is set to true.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to edit"
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
                        "description": "If true, replace all occurrences instead of just the first. Defaults to false."
                    },
                    "force": {
                        "type": "boolean",
                        "description": "If true, bypass the requirement to read the file first. Defaults to false."
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Write
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let path = require_str(&input, "path", "edit_file")?;
        let old_string = require_str(&input, "old_string", "edit_file")?;
        let new_string = require_str(&input, "new_string", "edit_file")?;
        let replace_all = input["replace_all"].as_bool().unwrap_or(false);
        let force = input["force"].as_bool().unwrap_or(false);

        if !force {
            let canonical = tokio::fs::canonicalize(&path)
                .await
                .unwrap_or_else(|_| PathBuf::from(&path));
            if !self.read_tracker.read().await.contains(&canonical) {
                return Ok(ToolOutput::text(
                    format!(
                        "Error: file '{}' must be read before editing. \
                         Use read_file first, or set force=true to bypass.",
                        path
                    ),
                    true,
                ));
            }
        }

        let content =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|error| AgshError::ToolExecution {
                    tool_name: "edit_file".to_string(),
                    message: format!("failed to read '{}': {}", path, error),
                })?;

        if !content.contains(&old_string) {
            return Ok(ToolOutput::text(
                format!(
                    "Error: '{}' not found in '{}'",
                    truncate_string(&old_string, 100),
                    path
                ),
                true,
            ));
        }

        let (new_content, count) = if replace_all {
            let count = content.matches(&old_string).count();
            (content.replace(&old_string, &new_string), count)
        } else {
            (content.replacen(&old_string, &new_string, 1), 1)
        };

        tokio::fs::write(&path, &new_content)
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "edit_file".to_string(),
                message: format!("failed to write '{}': {}", path, error),
            })?;

        Ok(ToolOutput::text(
            format!(
                "Successfully edited '{}': replaced {} occurrence(s)",
                path, count
            ),
            false,
        ))
    }
}

pub(super) struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Create or overwrite a file with the given content.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Write
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let path = require_str(&input, "path", "write_file")?;
        let content = require_str(&input, "content", "write_file")?;

        let file_path = PathBuf::from(&path);
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| AgshError::ToolExecution {
                    tool_name: "write_file".to_string(),
                    message: format!("failed to create directories for '{}': {}", path, error),
                })?;
        }

        tokio::fs::write(&path, &content)
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "write_file".to_string(),
                message: format!("failed to write '{}': {}", path, error),
            })?;

        Ok(ToolOutput::text(
            format!("Successfully wrote {} bytes to '{}'", content.len(), path),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use tokio::sync::RwLock;

    use super::*;
    use crate::provider::ContentBlock;

    fn text_content(output: &ToolOutput) -> String {
        ContentBlock::tool_result_text_content(&output.content)
    }

    fn test_tracker() -> ReadTracker {
        Arc::new(RwLock::new(HashSet::new()))
    }

    #[tokio::test]
    async fn test_read_file() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\n").expect("failed to write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
        };
        let result = tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("line1"));
        assert!(text_content(&result).contains("line3"));
    }

    #[tokio::test]
    async fn test_read_file_with_offset_and_limit() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "line0\nline1\nline2\nline3\nline4\n").expect("failed to write");

        let tool = ReadFileTool {
            read_tracker: test_tracker(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "offset": 1,
                    "limit": 2
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("line1"));
        assert!(text_content(&result).contains("line2"));
        assert!(!text_content(&result).contains("line0"));
        assert!(!text_content(&result).contains("line3"));
    }

    #[tokio::test]
    async fn test_write_and_read_file() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("output.txt");

        let write_tool = WriteFileTool;
        let write_result = write_tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "content": "hello world"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");
        assert!(!write_result.is_error);

        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "hello world");
    }

    #[tokio::test]
    async fn test_edit_file() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tracker = test_tracker();
        // Read the file first to satisfy read-before-edit
        let read_tool = ReadFileTool {
            read_tracker: tracker.clone(),
        };
        read_tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("read should succeed");

        let tool = EditFileTool {
            read_tracker: tracker,
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "hello rust");
    }

    #[tokio::test]
    async fn test_edit_file_replace_all() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "foo bar foo baz foo").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "foo",
                    "new_string": "qux",
                    "replace_all": true,
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("3 occurrence(s)"));
        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "qux bar qux baz qux");
    }

    #[tokio::test]
    async fn test_edit_file_replace_all_default_false() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "foo bar foo baz foo").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "foo",
                    "new_string": "qux",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("1 occurrence(s)"));
        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "qux bar foo baz foo");
    }

    #[tokio::test]
    async fn test_edit_file_not_found_string() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "nonexistent",
                    "new_string": "replacement",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(text_content(&result).contains("not found"));
    }

    #[tokio::test]
    async fn test_edit_without_read_fails() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(text_content(&result).contains("must be read before editing"));
    }

    #[tokio::test]
    async fn test_edit_with_force_bypasses_read_check() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tool = EditFileTool {
            read_tracker: test_tracker(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust",
                    "force": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        let content = std::fs::read_to_string(&file_path).expect("failed to read");
        assert_eq!(content, "hello rust");
    }

    #[tokio::test]
    async fn test_read_then_edit_succeeds() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tracker = test_tracker();

        let read_tool = ReadFileTool {
            read_tracker: tracker.clone(),
        };
        read_tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("read should succeed");

        let edit_tool = EditFileTool {
            read_tracker: tracker,
        };
        let result = edit_tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "world",
                    "new_string": "rust"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn test_read_file_a_edit_file_b_fails() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_a = temp_dir.path().join("a.txt");
        let file_b = temp_dir.path().join("b.txt");
        std::fs::write(&file_a, "content a").expect("failed to write");
        std::fs::write(&file_b, "content b").expect("failed to write");

        let tracker = test_tracker();

        let read_tool = ReadFileTool {
            read_tracker: tracker.clone(),
        };
        read_tool
            .execute(
                serde_json::json!({"path": file_a.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("read should succeed");

        let edit_tool = EditFileTool {
            read_tracker: tracker,
        };
        let result = edit_tool
            .execute(
                serde_json::json!({
                    "path": file_b.to_str().expect("path"),
                    "old_string": "content",
                    "new_string": "modified"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(text_content(&result).contains("must be read before editing"));
    }
}
