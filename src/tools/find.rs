use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::ToolDefinition;

use super::util::{redirects_to_scratchpad, require_str};
use super::{Tool, ToolOutput};

pub(super) struct FindFilesTool;

#[async_trait]
impl Tool for FindFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "find_files".to_string(),
            description: "Find files matching a glob pattern (e.g., '**/*.rs', 'src/*.txt'). \
                          Avoid overly broad searches: scanning a large tree can take \
                          a long time and will hit many directories the user has no \
                          read permission for, producing noisy errors. Start with the \
                          smallest `path` and most specific pattern that plausibly \
                          contains the answer; if that returns nothing, widen the \
                          `path` by one level or loosen the pattern, and repeat. Only \
                          fall back to a tree-wide scan if targeted attempts have all \
                          failed. Inline results are capped at 200 entries; use the \
                          `scratchpad` parameter to collect an unbounded result set."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files against. Prefer narrow patterns over broad ones like `**/*`."
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to current directory. Prefer the smallest subtree that can answer the question."
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["pattern"]
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
        let pattern = require_str(&input, "pattern", "find_files")?;
        let base_path = input["path"].as_str().map(|s| s.to_string());

        let full_pattern = match &base_path {
            Some(base) => format!("{}/{}", base.trim_end_matches('/'), pattern),
            None => pattern.clone(),
        };

        // Cap result count for inline use; lift it when redirecting output to
        // the scratchpad so the agent can collect an unbounded result set.
        let max_results = if redirects_to_scratchpad(&input) {
            usize::MAX
        } else {
            200
        };

        let result = tokio::task::spawn_blocking(move || {
            let mut matches = Vec::new();
            let mut truncated = false;
            match glob::glob(&full_pattern) {
                Ok(paths) => {
                    for entry in paths {
                        match entry {
                            Ok(path) => {
                                matches.push(path.display().to_string());
                                if matches.len() >= max_results {
                                    truncated = true;
                                    break;
                                }
                            }
                            Err(error) => {
                                tracing::warn!("glob error: {}", error);
                            }
                        }
                    }
                }
                Err(error) => {
                    return Err(AgshError::ToolExecution {
                        tool_name: "find_files".to_string(),
                        message: format!("invalid glob pattern '{}': {}", full_pattern, error),
                    });
                }
            }
            Ok((matches, truncated, max_results))
        })
        .await
        .map_err(|error| AgshError::ToolExecution {
            tool_name: "find_files".to_string(),
            message: format!("task join error: {}", error),
        })??;

        let (matches, truncated, max_results) = result;
        if matches.is_empty() {
            Ok(ToolOutput::text(
                "No files found matching the pattern.".to_string(),
                false,
            ))
        } else {
            let mut output = matches.join("\n");
            if truncated {
                output.push_str(&format!(
                    "\n\n... (truncated, showing first {} results)",
                    max_results
                ));
            }
            Ok(ToolOutput::text(output, false))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ContentBlock;

    fn text_content(output: &ToolOutput) -> String {
        ContentBlock::tool_result_text_content(&output.content)
    }

    #[tokio::test]
    async fn test_find_files() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(temp_dir.path().join("a.txt"), "").expect("failed");
        std::fs::write(temp_dir.path().join("b.txt"), "").expect("failed");
        std::fs::write(temp_dir.path().join("c.rs"), "").expect("failed");

        let tool = FindFilesTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("a.txt"));
        assert!(text_content(&result).contains("b.txt"));
        assert!(!text_content(&result).contains("c.rs"));
    }

    #[tokio::test]
    async fn test_find_files_inline_capped_at_200() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..250 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(text_content(&result).contains("truncated, showing first 200"));
    }

    #[tokio::test]
    async fn test_find_files_scratchpad_lifts_cap() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..250 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path"),
                    "scratchpad": "paths"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            !text.contains("truncated"),
            "expected no truncation marker when scratchpad set, got: {:.200}...",
            text
        );
        // All 250 entries should be listed.
        let line_count = text.lines().count();
        assert!(
            line_count >= 250,
            "expected >= 250 entries, got {}",
            line_count
        );
    }
}
