//! `search_contents` tool: ripgrep-style content search powered by the `grep-*` crates. Honors
//! `.gitignore` and supports glob filtering.

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

/// Inline match cap when the agent isn't redirecting to the scratchpad. Single source of truth for
/// the description and the runtime cap.
const MAX_INLINE_MATCHES: usize = 100;

pub(super) struct SearchContentsTool {
    pub cwd: crate::agent::SharedCwd,
}

#[async_trait]
impl Tool for SearchContentsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_contents".to_string(),
            description: format!(
                "Search file contents using a regex pattern (powered by ripgrep). \
                 Avoid overly broad searches: scanning a large tree is slow \
                 and will hit many directories the user has no read permission \
                 for, producing noisy errors. Start with the smallest `path` \
                 and a tight `glob` filter that plausibly contains the match; \
                 if that returns nothing, widen the `path` by one level or \
                 loosen the `glob`, and repeat. Only fall back to a tree-wide \
                 scan if targeted attempts have all failed. Inline results are \
                 capped at {} matches; use the `scratchpad` parameter to \
                 collect an unbounded result set. Multiple independent \
                 search_contents calls in one assistant message run in parallel.",
                MAX_INLINE_MATCHES,
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in. Defaults to current directory. Prefer the smallest subtree that can answer the question."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.rs'). Strongly recommended when searching directories to avoid scanning unrelated files."
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
        let pattern = require_str(&input, "pattern", "search_contents")?;
        // Resolve the optional `path` against the agent's per-session cwd so the search runs in the
        // right tree regardless of where the process was launched.
        let search_path = input["path"]
            .as_str()
            .map(|raw| crate::agent::resolve_against_cwd(&self.cwd, raw))
            .unwrap_or_else(|| crate::agent::cwd_snapshot(&self.cwd))
            .to_string_lossy()
            .into_owned();
        let file_glob = input["glob"].as_str().map(|s| s.to_string());
        // Cap match count for inline use; lift it when redirecting output to the scratchpad so the
        // agent can collect an unbounded result set.
        let max_results = if redirects_to_scratchpad(&input) {
            usize::MAX
        } else {
            MAX_INLINE_MATCHES
        };

        let result = tokio::task::spawn_blocking(move || {
            search_with_grep(&pattern, &search_path, file_glob.as_deref(), max_results)
        })
        .await
        .map_err(|error| AgshError::ToolExecution {
            tool_name: "search_contents".to_string(),
            message: format!("task join error: {}", error),
        })??;

        if result.is_empty() {
            Ok(ToolOutput::text("No matches found.".to_string(), false))
        } else {
            Ok(ToolOutput::text(result, false))
        }
    }
}

fn search_with_grep(
    pattern: &str,
    search_path: &str,
    file_glob: Option<&str>,
    max_results: usize,
) -> Result<String> {
    use grep_regex::RegexMatcherBuilder;

    // Cap the compiled-regex automaton and DFA cache sizes so an LLM-supplied pattern like
    // `a{10_000_000}` can't exhaust host memory during compile.
    const PATTERN_SIZE_LIMIT: usize = 1 << 20;
    const DFA_SIZE_LIMIT: usize = 1 << 20;

    let matcher = RegexMatcherBuilder::new()
        .size_limit(PATTERN_SIZE_LIMIT)
        .dfa_size_limit(DFA_SIZE_LIMIT)
        .build(pattern)
        .map_err(|error| AgshError::ToolExecution {
            tool_name: "search_contents".to_string(),
            message: format!("invalid or oversized regex '{}': {}", pattern, error),
        })?;

    let mut results = Vec::new();
    let path = std::path::Path::new(search_path);

    if path.is_file() {
        search_file(&matcher, path, &mut results)?;
    } else if path.is_dir() {
        let glob_pattern = match file_glob {
            Some(g) => Some(
                glob::Pattern::new(g).map_err(|error| AgshError::ToolExecution {
                    tool_name: "search_contents".to_string(),
                    message: format!("invalid glob pattern '{}': {}", g, error),
                })?,
            ),
            None => None,
        };

        walk_directory(path, &matcher, &glob_pattern, &mut results)?;
    } else {
        return Err(AgshError::ToolExecution {
            tool_name: "search_contents".to_string(),
            message: format!("path '{}' does not exist", search_path),
        });
    }

    if results.len() > max_results {
        results.truncate(max_results);
        results.push(format!(
            "... (truncated, showing first {} matches)",
            max_results
        ));
    }

    Ok(results.join("\n"))
}

fn search_file(
    matcher: &grep_regex::RegexMatcher,
    path: &std::path::Path,
    results: &mut Vec<String>,
) -> Result<()> {
    use grep_searcher::{Searcher, sinks::UTF8};

    let mut searcher = Searcher::new();
    if let Err(error) = searcher.search_path(
        matcher,
        path,
        UTF8(|line_number, line| {
            results.push(format!(
                "{}:{}:{}",
                path.display(),
                line_number,
                line.trim_end()
            ));
            Ok(true)
        }),
    ) {
        tracing::debug!("search_path failed for {}: {}", path.display(), error);
    }

    Ok(())
}

fn walk_directory(
    directory: &std::path::Path,
    matcher: &grep_regex::RegexMatcher,
    glob_pattern: &Option<glob::Pattern>,
    results: &mut Vec<String>,
) -> Result<()> {
    // Iterative traversal via an explicit work-stack: a recursive walk would overflow the call
    // stack on a pathologically deep directory tree.
    let mut pending: Vec<std::path::PathBuf> = vec![directory.to_path_buf()];

    while let Some(dir) = pending.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for entry in entries {
            let Ok(entry) = entry else { continue };
            let path = entry.path();

            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if file_name.starts_with('.') || file_name == "target" || file_name == "node_modules" {
                continue;
            }

            // `entry.file_type()` does not follow symlinks: a symlinked directory reports as a
            // symlink, not a dir, so it is never descended into. That removes any symlink-cycle
            // risk while still letting symlinked *files* be searched via the path-based `is_file()`
            // check below.
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                pending.push(path);
            } else if path.is_file() {
                if let Some(pattern) = glob_pattern
                    && !pattern.matches(file_name)
                {
                    continue;
                }
                search_file(matcher, &path, results)?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests::text_content;

    #[tokio::test]
    async fn test_search_contents() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(
            temp_dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again\n",
        )
        .expect("failed");

        let tool = SearchContentsTool {
            cwd: crate::agent::test_cwd(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "hello",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("hello world"));
        assert!(text_content(&result).contains("hello again"));
    }

    #[tokio::test]
    async fn test_search_contents_deeply_nested_tree() {
        // Exercises the iterative work-stack traversal: a file buried many directory levels deep
        // must still be found. A recursive walk would recurse once per level; the iterative version
        // uses a heap stack.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let mut deep = temp_dir.path().to_path_buf();
        for _ in 0..300 {
            deep.push("d");
        }
        std::fs::create_dir_all(&deep).expect("create nested tree");
        std::fs::write(deep.join("buried.txt"), "needle here\n").expect("write");

        let tool = SearchContentsTool {
            cwd: crate::agent::test_cwd(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "needle",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("needle here"));
    }

    #[tokio::test]
    async fn test_search_contents_inline_capped_at_100() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        // One file with 150 matching lines.
        let content = (0..150).map(|_| "match\n").collect::<String>();
        std::fs::write(temp_dir.path().join("many.txt"), content).expect("write");

        let tool = SearchContentsTool {
            cwd: crate::agent::test_cwd(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(text_content(&result).contains("truncated, showing first 100"));
    }

    #[tokio::test]
    async fn test_search_contents_invalid_glob_errors() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp_dir.path().join("a.txt"), "match").expect("write");

        let tool = SearchContentsTool {
            cwd: crate::agent::test_cwd(),
        };
        let err = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path"),
                    "glob": "[unclosed",
                }),
                CancellationToken::new(),
            )
            .await
            .expect_err("invalid glob must be rejected, not silently scan everything");
        let message = format!("{}", err);
        assert!(
            message.contains("invalid glob pattern"),
            "unexpected error: {}",
            message
        );
    }

    #[tokio::test]
    async fn test_search_contents_scratchpad_lifts_cap() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let content = (0..150).map(|_| "match\n").collect::<String>();
        std::fs::write(temp_dir.path().join("many.txt"), content).expect("write");

        let tool = SearchContentsTool {
            cwd: crate::agent::test_cwd(),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path"),
                    "scratchpad": "matches"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            !text.contains("truncated"),
            "expected no truncation marker when scratchpad set"
        );
        let match_lines = text.lines().filter(|l| l.contains("match")).count();
        assert!(
            match_lines >= 150,
            "expected >= 150 match lines, got {}",
            match_lines
        );
    }
}
