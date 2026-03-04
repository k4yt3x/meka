use std::path::PathBuf;

use async_trait::async_trait;
use html2md::rewrite_html;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::ToolDefinition;

#[derive(Debug)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn required_permission(&self) -> Permission;
    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput>;
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|tool| tool.definition().name == name)
            .map(|tool| tool.as_ref())
    }

    pub fn definitions_for_permission(&self, permission: Permission) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .filter(|tool| permission.allows(tool.required_permission()))
            .map(|tool| tool.definition())
            .collect()
    }

    pub fn build_default(user_agent: String) -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(ReadFileTool));
        registry.register(Box::new(EditFileTool));
        registry.register(Box::new(WriteFileTool));
        registry.register(Box::new(FindFilesTool));
        registry.register(Box::new(SearchContentsTool));
        registry.register(Box::new(FetchUrlTool {
            user_agent: user_agent.clone(),
        }));
        registry.register(Box::new(WebSearchTool {
            user_agent: user_agent.clone(),
        }));
        registry.register(Box::new(ExecuteCommandTool));
        registry
    }
}

// ---------------------------------------------------------------------------
// read_file
// ---------------------------------------------------------------------------

struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read the contents of a file at the given path.".to_string(),
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

        let offset = input["offset"].as_u64().map(|value| value as usize);
        let limit = input["limit"].as_u64().map(|value| value as usize);

        let content =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|error| AgshError::ToolExecution {
                    tool_name: "read_file".to_string(),
                    message: format!("failed to read '{}': {}", path, error),
                })?;

        let result = match (offset, limit) {
            (Some(offset), Some(limit)) => content
                .lines()
                .skip(offset)
                .take(limit)
                .collect::<Vec<_>>()
                .join("\n"),
            (Some(offset), None) => content.lines().skip(offset).collect::<Vec<_>>().join("\n"),
            (None, Some(limit)) => content.lines().take(limit).collect::<Vec<_>>().join("\n"),
            (None, None) => content,
        };

        Ok(ToolOutput {
            content: result,
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// edit_file
// ---------------------------------------------------------------------------

struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_string(),
            description: "Make a string replacement in a file. Replaces the first occurrence of 'old_string' with 'new_string'.".to_string(),
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

        let content =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|error| AgshError::ToolExecution {
                    tool_name: "edit_file".to_string(),
                    message: format!("failed to read '{}': {}", path, error),
                })?;

        if !content.contains(&old_string) {
            return Ok(ToolOutput {
                content: format!(
                    "Error: '{}' not found in '{}'",
                    truncate_string(&old_string, 100),
                    path
                ),
                is_error: true,
            });
        }

        let new_content = content.replacen(&old_string, &new_string, 1);
        tokio::fs::write(&path, &new_content)
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "edit_file".to_string(),
                message: format!("failed to write '{}': {}", path, error),
            })?;

        Ok(ToolOutput {
            content: format!("Successfully edited '{}'", path),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// write_file
// ---------------------------------------------------------------------------

struct WriteFileTool;

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

        Ok(ToolOutput {
            content: format!("Successfully wrote {} bytes to '{}'", content.len(), path),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// find_files
// ---------------------------------------------------------------------------

struct FindFilesTool;

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

// ---------------------------------------------------------------------------
// search_contents
// ---------------------------------------------------------------------------

struct SearchContentsTool;

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
        let glob_pattern = file_glob.map(|g| glob::Pattern::new(g).ok()).flatten();

        walk_directory(path, &matcher, &glob_pattern, &mut results)?;
    } else {
        return Err(AgshError::ToolExecution {
            tool_name: "search_contents".to_string(),
            message: format!("path '{}' does not exist", search_path),
        });
    }

    // Limit output to prevent overwhelming the LLM context
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
    let _ = searcher.search_path(
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
    );

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

        // Skip hidden files and common non-text directories
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name.starts_with('.') || file_name == "target" || file_name == "node_modules" {
            continue;
        }

        if path.is_dir() {
            walk_directory(&path, matcher, glob_pattern, results)?;
        } else if path.is_file() {
            if let Some(pattern) = glob_pattern {
                if !pattern.matches(file_name) {
                    continue;
                }
            }
            search_file(matcher, &path, results)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// execute_command
// ---------------------------------------------------------------------------

struct ExecuteCommandTool;

#[async_trait]
impl Tool for ExecuteCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "execute_command".to_string(),
            description: "Execute a shell command and return its output.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Timeout in milliseconds. Defaults to 30000 (30 seconds)."
                    }
                },
                "required": ["command"]
            }),
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Write
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let command = require_str(&input, "command", "execute_command")?;
        let timeout_ms = input["timeout_ms"].as_u64().unwrap_or(30000);

        #[cfg(windows)]
        let mut command_builder = {
            let mut cmd = tokio::process::Command::new("powershell");
            cmd.arg("-Command").arg(&command);
            cmd
        };

        #[cfg(not(windows))]
        let mut command_builder = {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(&command);
            cmd
        };

        let mut child = command_builder
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "execute_command".to_string(),
                message: format!("failed to spawn command: {}", error),
            })?;

        let timeout_duration = std::time::Duration::from_millis(timeout_ms);

        // wait_with_output() consumes the child, so use wait() + manual
        // stdout/stderr reading instead to allow kill on cancellation.
        tokio::select! {
            _ = cancellation.cancelled() => {
                let _ = child.kill().await;
                Err(AgshError::Interrupted)
            }
            _ = tokio::time::sleep(timeout_duration) => {
                let _ = child.kill().await;
                Ok(ToolOutput {
                    content: format!("Command timed out after {}ms", timeout_ms),
                    is_error: true,
                })
            }
            status = child.wait() => {
                let status = status.map_err(|error| AgshError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message: format!("failed to wait for command: {}", error),
                })?;

                let exit_code = status.code().unwrap_or(-1);

                // Read whatever was captured in the pipes
                let mut stdout_content = String::new();
                let mut stderr_content = String::new();

                if let Some(mut stdout) = child.stdout.take() {
                    use tokio::io::AsyncReadExt;
                    let _ = stdout.read_to_string(&mut stdout_content).await;
                }
                if let Some(mut stderr) = child.stderr.take() {
                    use tokio::io::AsyncReadExt;
                    let _ = stderr.read_to_string(&mut stderr_content).await;
                }

                let mut result_text = String::new();
                if !stdout_content.is_empty() {
                    result_text.push_str(&stdout_content);
                }
                if !stderr_content.is_empty() {
                    if !result_text.is_empty() {
                        result_text.push_str("\n--- stderr ---\n");
                    }
                    result_text.push_str(&stderr_content);
                }
                if exit_code != 0 {
                    result_text.push_str(&format!("\nExit code: {}", exit_code));
                }

                Ok(ToolOutput {
                    content: if result_text.is_empty() {
                        "(no output)".to_string()
                    } else {
                        result_text
                    },
                    is_error: exit_code != 0,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fetch_url
// ---------------------------------------------------------------------------

struct FetchUrlTool {
    user_agent: String,
}

#[async_trait]
impl Tool for FetchUrlTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_url".to_string(),
            description: "Fetch a web page and return its content as markdown text.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
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
        let url = require_str(&input, "url", "fetch_url")?;

        let client = reqwest::Client::builder()
            .user_agent(&self.user_agent)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("failed to create HTTP client: {}", error),
            })?;

        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("failed to fetch '{}': {}", url, error),
            })?;

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolOutput {
                content: format!("HTTP {} for '{}'", status, url),
                is_error: true,
            });
        }

        let html = response
            .text()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("failed to read response body: {}", error),
            })?;

        let markdown = rewrite_html(&html, false);

        // Truncate to avoid overwhelming the LLM context
        let max_length = 50000;
        let content = if markdown.len() > max_length {
            format!(
                "{}\n\n... (truncated, showing first {} characters)",
                &markdown[..max_length],
                max_length
            )
        } else {
            markdown
        };

        Ok(ToolOutput {
            content,
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// web_search
// ---------------------------------------------------------------------------

struct WebSearchTool {
    user_agent: String,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web and return results. Supports DuckDuckGo (default), \
                Google, and Bing search engines."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "engine": {
                        "type": "string",
                        "description": "Search engine to use: 'duckduckgo' (default), 'google', or 'bing'",
                        "enum": ["duckduckgo", "google", "bing"]
                    }
                },
                "required": ["query"]
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
        let query = require_str(&input, "query", "web_search")?;
        let engine = input
            .get("engine")
            .and_then(|engine| engine.as_str())
            .unwrap_or("duckduckgo");

        let client = reqwest::Client::builder()
            .user_agent(&self.user_agent)
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "web_search".to_string(),
                message: format!("failed to create HTTP client: {}", error),
            })?;

        let (url, query_params): (&str, Vec<(&str, &str)>) = match engine {
            "google" => (
                "https://www.google.com/search",
                vec![("q", query.as_str()), ("hl", "en")],
            ),
            "bing" => ("https://www.bing.com/search", vec![("q", query.as_str())]),
            _ => (
                "https://html.duckduckgo.com/html/",
                vec![("q", query.as_str())],
            ),
        };

        let response = client
            .get(url)
            .query(&query_params)
            .send()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "web_search".to_string(),
                message: format!("search request failed: {}", error),
            })?;

        let html = response
            .text()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "web_search".to_string(),
                message: format!("failed to read search response: {}", error),
            })?;

        let results = match engine {
            "google" => parse_google_results(&html),
            "bing" => parse_bing_results(&html),
            _ => parse_duckduckgo_results(&html),
        };

        if results.is_empty() {
            Ok(ToolOutput {
                content: "No search results found.".to_string(),
                is_error: false,
            })
        } else {
            Ok(ToolOutput {
                content: results,
                is_error: false,
            })
        }
    }
}

fn parse_duckduckgo_results(html: &str) -> String {
    // DuckDuckGo HTML results have a structure where each result is in a
    // div with class "result". We extract titles and snippets using simple
    // string matching since we don't want a full HTML parser dependency
    // just for this.
    let mut results = Vec::new();
    let mut position = 0;

    while let Some(result_start) = html[position..].find("class=\"result__a\"") {
        let absolute_start = position + result_start;

        // Extract the URL from href
        let url = extract_href(&html[..absolute_start + 200], absolute_start);

        // Extract the title text (between > and </a>)
        let title = if let Some(tag_end) = html[absolute_start..].find('>') {
            let text_start = absolute_start + tag_end + 1;
            if let Some(close_tag) = html[text_start..].find("</a>") {
                strip_html_tags(&html[text_start..text_start + close_tag])
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Extract the snippet (class="result__snippet")
        let snippet =
            if let Some(snippet_start) = html[absolute_start..].find("class=\"result__snippet\"") {
                let snippet_abs = absolute_start + snippet_start;
                if let Some(tag_end) = html[snippet_abs..].find('>') {
                    let text_start = snippet_abs + tag_end + 1;
                    if let Some(close_tag) = html[text_start..].find("</") {
                        strip_html_tags(&html[text_start..text_start + close_tag])
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

        if !title.is_empty() {
            let mut result_text = format!("{}. **{}**", results.len() + 1, title.trim());
            if let Some(url) = &url {
                result_text.push_str(&format!("\n   URL: {}", url));
            }
            if !snippet.is_empty() {
                result_text.push_str(&format!("\n   {}", snippet.trim()));
            }
            results.push(result_text);

            if results.len() >= 10 {
                break;
            }
        }

        position = absolute_start + 1;
    }

    results.join("\n\n")
}

fn extract_href(html: &str, near_position: usize) -> Option<String> {
    // Look backward from position to find href="..."
    let search_start = near_position.saturating_sub(500);
    let search_region = &html[search_start..near_position];

    if let Some(href_pos) = search_region.rfind("href=\"") {
        let url_start = search_start + href_pos + 6;
        if let Some(url_end) = html[url_start..].find('"') {
            let url = &html[url_start..url_start + url_end];
            // DuckDuckGo wraps URLs in a redirect, extract the actual URL
            if let Some(uddg_pos) = url.find("uddg=") {
                let encoded_url = &url[uddg_pos + 5..];
                let decoded = urldecode(encoded_url);
                // Strip any trailing parameters after the URL
                let clean_url = decoded.split('&').next().unwrap_or(&decoded);
                return Some(clean_url.to_string());
            }
            return Some(url.to_string());
        }
    }
    None
}

fn urldecode(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars();

    while let Some(character) = chars.next() {
        if character == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            } else {
                result.push('%');
                result.push_str(&hex);
            }
        } else if character == '+' {
            result.push(' ');
        } else {
            result.push(character);
        }
    }

    result
}

fn strip_html_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;

    for character in html.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(character),
            _ => {}
        }
    }

    // Decode common HTML entities
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn parse_google_results(html: &str) -> String {
    let mut results = Vec::new();
    let mut position = 0;

    // Google search results typically appear in <div class="g"> blocks.
    // Within each block, the title is in an <h3> tag and the URL in an <a href="...">.
    while let Some(block_start) = html[position..].find("<div class=\"g\"") {
        let absolute_start = position + block_start;
        let block_end = html[absolute_start..]
            .find("<div class=\"g\"")
            .map(|next| {
                // Avoid matching the same block we just found by searching past it
                if next == 0 {
                    html[absolute_start + 20..]
                        .find("<div class=\"g\"")
                        .map(|p| absolute_start + 20 + p)
                        .unwrap_or(html.len())
                } else {
                    absolute_start + next
                }
            })
            .unwrap_or(html.len());

        let block = &html[absolute_start..block_end];

        // Extract URL from first <a href="...">
        let url = if let Some(href_start) = block.find("href=\"") {
            let url_start = href_start + 6;
            block[url_start..]
                .find('"')
                .map(|url_end| block[url_start..url_start + url_end].to_string())
                .filter(|url| url.starts_with("http"))
        } else {
            None
        };

        // Extract title from <h3>...</h3>
        let title = if let Some(h3_start) = block.find("<h3") {
            if let Some(tag_end) = block[h3_start..].find('>') {
                let text_start = h3_start + tag_end + 1;
                block[text_start..]
                    .find("</h3>")
                    .map(|close| strip_html_tags(&block[text_start..text_start + close]))
            } else {
                None
            }
        } else {
            None
        };

        if let Some(title) = title {
            if !title.trim().is_empty() {
                let mut result_text = format!("{}. **{}**", results.len() + 1, title.trim());
                if let Some(url) = &url {
                    result_text.push_str(&format!("\n   URL: {}", url));
                }
                results.push(result_text);

                if results.len() >= 10 {
                    break;
                }
            }
        }

        position = absolute_start + 1;
    }

    results.join("\n\n")
}

fn parse_bing_results(html: &str) -> String {
    let mut results = Vec::new();
    let mut position = 0;

    // Bing organic results appear in <li class="b_algo"> blocks.
    while let Some(block_start) = html[position..].find("class=\"b_algo\"") {
        let absolute_start = position + block_start;
        let block_end = html[absolute_start + 14..]
            .find("class=\"b_algo\"")
            .map(|next| absolute_start + 14 + next)
            .unwrap_or(html.len());

        let block = &html[absolute_start..block_end];

        // Extract URL from <a href="...">
        let url = if let Some(href_start) = block.find("href=\"") {
            let url_start = href_start + 6;
            block[url_start..]
                .find('"')
                .map(|url_end| block[url_start..url_start + url_end].to_string())
                .filter(|url| url.starts_with("http"))
        } else {
            None
        };

        // Extract title from <a ...>...</a> inside <h2>
        let title = if let Some(h2_start) = block.find("<h2") {
            if let Some(a_start) = block[h2_start..].find("<a ") {
                let a_abs = h2_start + a_start;
                if let Some(tag_end) = block[a_abs..].find('>') {
                    let text_start = a_abs + tag_end + 1;
                    block[text_start..]
                        .find("</a>")
                        .map(|close| strip_html_tags(&block[text_start..text_start + close]))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Extract snippet from <p> or <div class="b_caption">
        let snippet = block.find("class=\"b_caption\"").and_then(|cap_start| {
            block[cap_start..].find("<p").and_then(|p_start| {
                let p_abs = cap_start + p_start;
                block[p_abs..].find('>').and_then(|tag_end| {
                    let text_start = p_abs + tag_end + 1;
                    block[text_start..]
                        .find("</p>")
                        .map(|close| strip_html_tags(&block[text_start..text_start + close]))
                })
            })
        });

        if let Some(title) = title {
            if !title.trim().is_empty() {
                let mut result_text = format!("{}. **{}**", results.len() + 1, title.trim());
                if let Some(url) = &url {
                    result_text.push_str(&format!("\n   URL: {}", url));
                }
                if let Some(snippet) = &snippet {
                    let snippet = snippet.trim();
                    if !snippet.is_empty() {
                        result_text.push_str(&format!("\n   {}", snippet));
                    }
                }
                results.push(result_text);

                if results.len() >= 10 {
                    break;
                }
            }
        }

        position = absolute_start + 1;
    }

    results.join("\n\n")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn require_str(input: &serde_json::Value, field: &str, tool_name: &str) -> Result<String> {
    input[field]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| AgshError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("missing '{}' parameter", field),
        })
}

fn truncate_string(string: &str, max_length: usize) -> &str {
    if string.len() <= max_length {
        string
    } else {
        &string[..max_length]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_registry() {
        let registry = ToolRegistry::build_default("test-agent/0.1".to_string());
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(registry.get("edit_file").is_some());
        assert!(registry.get("find_files").is_some());
        assert!(registry.get("search_contents").is_some());
        assert!(registry.get("execute_command").is_some());
        assert!(registry.get("fetch_url").is_some());
        assert!(registry.get("web_search").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_permission_filtering() {
        let registry = ToolRegistry::build_default("test-agent/0.1".to_string());

        let none_tools = registry.definitions_for_permission(Permission::None);
        assert!(none_tools.is_empty());

        let read_tools = registry.definitions_for_permission(Permission::Read);
        assert!(read_tools.iter().any(|t| t.name == "read_file"));
        assert!(read_tools.iter().any(|t| t.name == "find_files"));
        assert!(!read_tools.iter().any(|t| t.name == "write_file"));
        assert!(!read_tools.iter().any(|t| t.name == "execute_command"));

        let write_tools = registry.definitions_for_permission(Permission::Write);
        assert!(write_tools.iter().any(|t| t.name == "read_file"));
        assert!(write_tools.iter().any(|t| t.name == "write_file"));
        assert!(write_tools.iter().any(|t| t.name == "execute_command"));
    }

    #[tokio::test]
    async fn test_read_file() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\n").expect("failed to write");

        let tool = ReadFileTool;
        let result = tool
            .execute(
                serde_json::json!({"path": file_path.to_str().expect("path")}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(result.content.contains("line1"));
        assert!(result.content.contains("line3"));
    }

    #[tokio::test]
    async fn test_read_file_with_offset_and_limit() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "line0\nline1\nline2\nline3\nline4\n").expect("failed to write");

        let tool = ReadFileTool;
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
        assert!(result.content.contains("line1"));
        assert!(result.content.contains("line2"));
        assert!(!result.content.contains("line0"));
        assert!(!result.content.contains("line3"));
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

        let tool = EditFileTool;
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
    async fn test_edit_file_not_found_string() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let file_path = temp_dir.path().join("edit.txt");
        std::fs::write(&file_path, "hello world").expect("failed to write");

        let tool = EditFileTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "path": file_path.to_str().expect("path"),
                    "old_string": "nonexistent",
                    "new_string": "replacement"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
        assert!(result.content.contains("not found"));
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

    #[tokio::test]
    async fn test_execute_command() {
        let tool = ExecuteCommandTool;
        let result = tool
            .execute(
                serde_json::json!({"command": "echo hello"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert_eq!(result.content.trim(), "hello");
    }

    #[tokio::test]
    async fn test_execute_command_failure() {
        let tool = ExecuteCommandTool;
        let result = tool
            .execute(
                serde_json::json!({"command": "false"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("<b>hello</b>"), "hello");
        assert_eq!(strip_html_tags("no tags here"), "no tags here");
        assert_eq!(strip_html_tags("&amp; test"), "& test");
    }

    #[test]
    fn test_urldecode() {
        assert_eq!(urlecode("hello%20world"), "hello world");
        assert_eq!(urlecode("test%2Fpath"), "test/path");
        assert_eq!(urlecode("a+b"), "a b");
    }

    fn urlecode(input: &str) -> String {
        urldecode(input)
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 5), "hello");
    }
}
