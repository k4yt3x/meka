use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::ToolDefinition;

use super::util::require_str;
use super::{Tool, ToolOutput};

pub(super) struct FindFilesTool;

#[async_trait]
impl Tool for FindFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "find_files".to_string(),
            description: "Find files matching a glob pattern (e.g., '**/*.rs', 'src/*.txt')."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files against"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to current directory."
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

        let result = tokio::task::spawn_blocking(move || {
            let mut matches = Vec::new();
            match glob::glob(&full_pattern) {
                Ok(paths) => {
                    for entry in paths {
                        match entry {
                            Ok(path) => matches.push(path.display().to_string()),
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
            Ok(matches)
        })
        .await
        .map_err(|error| AgshError::ToolExecution {
            tool_name: "find_files".to_string(),
            message: format!("task join error: {}", error),
        })??;

        if result.is_empty() {
            Ok(ToolOutput {
                content: "No files found matching the pattern.".to_string(),
                is_error: false,
            })
        } else {
            Ok(ToolOutput {
                content: result.join("\n"),
                is_error: false,
            })
        }
    }
}

pub(super) struct SearchContentsTool;

#[async_trait]
impl Tool for SearchContentsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_contents".to_string(),
            description: "Search file contents using a regex pattern (powered by ripgrep)."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in. Defaults to current directory."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.rs'). Optional."
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
        let pattern = require_str(&input, "pattern", "search_contents")?;
        let search_path = input["path"].as_str().unwrap_or(".").to_string();
        let file_glob = input["glob"].as_str().map(|s| s.to_string());

        let result = tokio::task::spawn_blocking(move || {
            search_with_grep(&pattern, &search_path, file_glob.as_deref())
        })
        .await
        .map_err(|error| AgshError::ToolExecution {
            tool_name: "search_contents".to_string(),
            message: format!("task join error: {}", error),
        })??;

        if result.is_empty() {
            Ok(ToolOutput {
                content: "No matches found.".to_string(),
                is_error: false,
            })
        } else {
            Ok(ToolOutput {
                content: result,
                is_error: false,
            })
        }
    }
}

fn search_with_grep(pattern: &str, search_path: &str, file_glob: Option<&str>) -> Result<String> {
    use grep_regex::RegexMatcher;

    let matcher = RegexMatcher::new(pattern).map_err(|error| AgshError::ToolExecution {
        tool_name: "search_contents".to_string(),
        message: format!("invalid regex '{}': {}", pattern, error),
    })?;

    let mut results = Vec::new();
    let path = std::path::Path::new(search_path);

    if path.is_file() {
        search_file(&matcher, path, &mut results)?;
    } else if path.is_dir() {
        let glob_pattern = file_glob.and_then(|g| glob::Pattern::new(g).ok());

        walk_directory(path, &matcher, &glob_pattern, &mut results)?;
    } else {
        return Err(AgshError::ToolExecution {
            tool_name: "search_contents".to_string(),
            message: format!("path '{}' does not exist", search_path),
        });
    }

    let max_results = 100;
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
    use grep_searcher::Searcher;
    use grep_searcher::sinks::UTF8;

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
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(_) => return Ok(()),
    };

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();

        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name.starts_with('.') || file_name == "target" || file_name == "node_modules" {
            continue;
        }

        if path.is_dir() {
            walk_directory(&path, matcher, glob_pattern, results)?;
        } else if path.is_file() {
            if let Some(pattern) = glob_pattern
                && !pattern.matches(file_name)
            {
                continue;
            }
            search_file(matcher, &path, results)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(result.content.contains("a.txt"));
        assert!(result.content.contains("b.txt"));
        assert!(!result.content.contains("c.rs"));
    }

    #[tokio::test]
    async fn test_search_contents() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(
            temp_dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again\n",
        )
        .expect("failed");

        let tool = SearchContentsTool;
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
        assert!(result.content.contains("hello world"));
        assert!(result.content.contains("hello again"));
    }
}
