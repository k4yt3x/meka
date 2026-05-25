//! `find_files` tool: glob-pattern file discovery.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{
    Tool, ToolOutput,
    util::{redirects_to_scratchpad, require_str},
};
use crate::{
    error::{AgshError, Result},
    permission::Permission,
    provider::ToolDefinition,
};

/// Default inline result cap when the agent isn't redirecting to the
/// scratchpad and didn't pass an explicit `limit`. Single source of truth
/// for the description and the runtime default.
const DEFAULT_INLINE_RESULTS: usize = 500;

pub(super) struct FindFilesTool {
    pub cwd: crate::agent::SharedCwd,
}

#[async_trait]
impl Tool for FindFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "find_files".to_string(),
            description: format!(
                "Find files matching a glob pattern (e.g., '**/*.rs', 'src/*.txt'). \
                 Avoid overly broad searches: scanning a large tree can take \
                 a long time and will hit many directories the user has no \
                 read permission for, producing noisy errors. Start with the \
                 smallest `path` and most specific pattern that plausibly \
                 contains the answer; if that returns nothing, widen the \
                 `path` by one level or loosen the pattern, and repeat. Only \
                 fall back to a tree-wide scan if targeted attempts have all \
                 failed. Inline results default to {} entries; pass `limit` to \
                 raise the cap or `scratchpad` to collect them all. \
                 Multiple independent find_files calls in one assistant message \
                 run in parallel.",
                DEFAULT_INLINE_RESULTS,
            ),
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
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": format!(
                            "Maximum results to return. Defaults to {} when output is inline, \
                             unbounded when `scratchpad` is set. Pass an explicit value to \
                             override either default.",
                            DEFAULT_INLINE_RESULTS,
                        )
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["pattern"]
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
        let pattern = require_str(&input, "pattern", "find_files")?;
        // Resolve the optional `path` against the agent's per-session
        // cwd so the search runs in the right tree regardless of where
        // the process was launched. Absolute paths pass through unchanged.
        let base_path = input["path"]
            .as_str()
            .map(|raw| crate::agent::resolve_against_cwd(&self.cwd, raw))
            .unwrap_or_else(|| crate::agent::cwd_snapshot(&self.cwd));
        let full_pattern = format!(
            "{}/{}",
            base_path.to_string_lossy().trim_end_matches('/'),
            pattern,
        );

        // Cap precedence:
        //   1. explicit `limit` parameter — honoured verbatim
        //   2. no limit + `scratchpad` set — unbounded (preserves the "collect everything" escape
        //      hatch)
        //   3. otherwise — DEFAULT_INLINE_RESULTS
        let explicit_limit = input
            .get("limit")
            .and_then(|value| value.as_u64())
            .map(|value| usize::try_from(value).unwrap_or(usize::MAX));
        let cap = match explicit_limit {
            Some(limit) => limit,
            None if redirects_to_scratchpad(&input) => usize::MAX,
            None => DEFAULT_INLINE_RESULTS,
        };

        let result = tokio::task::spawn_blocking(move || {
            let mut matches: Vec<String> = Vec::new();
            // Total continues past the storage cap so we can report
            // the real count of matches in the truncation message —
            // glob walks are FS-metadata only, so the extra iteration
            // past the cap is cheap.
            let mut total: usize = 0;
            match glob::glob(&full_pattern) {
                Ok(paths) => {
                    for entry in paths {
                        match entry {
                            Ok(path) => {
                                total += 1;
                                if matches.len() < cap {
                                    matches.push(path.display().to_string());
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
            Ok((matches, total, cap))
        })
        .await
        .map_err(|error| AgshError::ToolExecution {
            tool_name: "find_files".to_string(),
            message: format!("task join error: {}", error),
        })??;

        let (matches, total, cap) = result;
        if matches.is_empty() {
            Ok(ToolOutput::text(
                "No files found matching the pattern.".to_string(),
                false,
            ))
        } else {
            let mut output = matches.join("\n");
            if total > matches.len() {
                output.push_str(&format!(
                    "\n\n... (showed first {} of {} matches; refine `pattern` to narrow, \
                     pass `limit: <n>` to raise the cap, or pass `scratchpad: \"name\"` to \
                     collect them all)",
                    cap, total,
                ));
            }
            Ok(ToolOutput::text(output, false))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests::text_content;

    #[tokio::test]
    async fn test_find_files() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(temp_dir.path().join("a.txt"), "").expect("failed");
        std::fs::write(temp_dir.path().join("b.txt"), "").expect("failed");
        std::fs::write(temp_dir.path().join("c.rs"), "").expect("failed");

        let tool = FindFilesTool {
            cwd: crate::agent::test_cwd(),
        };
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
    async fn test_find_files_inline_default_cap_reports_total() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..600 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::agent::test_cwd(),
        };
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

        let text = text_content(&result);
        assert!(
            text.contains("showed first 500 of 600 matches"),
            "expected real total in truncation message, got: {:.300}",
            text,
        );
    }

    #[tokio::test]
    async fn test_find_files_limit_overrides_default() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..600 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::agent::test_cwd(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path"),
                    "limit": 100
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(text.contains("showed first 100 of 600 matches"));
        // 100 entries plus the trailing truncation line.
        let path_lines = text.lines().filter(|line| line.ends_with(".txt")).count();
        assert_eq!(path_lines, 100);
    }

    #[tokio::test]
    async fn test_find_files_scratchpad_lifts_cap() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..600 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::agent::test_cwd(),
        };
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
            !text.contains("showed first"),
            "expected no truncation marker when scratchpad set, got: {:.200}...",
            text,
        );
        let line_count = text.lines().filter(|l| l.ends_with(".txt")).count();
        assert!(
            line_count >= 600,
            "expected >= 600 entries, got {}",
            line_count
        );
    }

    #[tokio::test]
    async fn test_find_files_explicit_limit_with_scratchpad_caps() {
        // Regression: an explicit `limit` should beat the scratchpad
        // "unbounded" default — the agent might legitimately want a
        // bounded scratchpad collection.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..600 {
            std::fs::write(temp_dir.path().join(format!("f{}.txt", i)), "").expect("write");
        }

        let tool = FindFilesTool {
            cwd: crate::agent::test_cwd(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "*.txt",
                    "path": temp_dir.path().to_str().expect("path"),
                    "scratchpad": "paths",
                    "limit": 50
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(text.contains("showed first 50 of 600 matches"));
    }
}
