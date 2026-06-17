//! Small shared helpers for tool-input parsing and validation.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use tokio::sync::RwLock;
use uuid::Uuid;

use super::ToolOutput;
use crate::error::{MekaError, Result};

/// Default cap for regex-search-mode hits; shared by `read_file` and `scratchpad_read`.
pub(super) const MAX_SEARCH_MATCHES: usize = 100;

/// Resolve the active session id for a session-scoped tool, erroring if no session is open. Shared
/// by the scratchpad and recall tool families.
pub(super) async fn resolve_session_id(
    session_id: &Arc<RwLock<Option<Uuid>>>,
    tool_name: &str,
) -> Result<Uuid> {
    session_id
        .read()
        .await
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: "no active session".to_string(),
        })
}

pub(super) fn require_str(
    input: &serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<String> {
    input[field]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("missing '{}' parameter", field),
        })
}

/// Compile an LLM-supplied regex pattern with bounded compile memory so a pathological pattern like
/// `a{10_000_000}` cannot exhaust the host's RAM during compilation. The `regex` crate's NFA/DFA
/// engines already avoid catastrophic backtracking at *match* time; the remaining DoS surface is
/// the one-time cost of building the automaton, which this bounds.
pub(super) fn compile_user_regex(pattern: &str, tool_name: &str) -> Result<regex::Regex> {
    const PATTERN_SIZE_LIMIT: usize = 1 << 20;
    const DFA_SIZE_LIMIT: usize = 1 << 20;

    regex::RegexBuilder::new(pattern)
        .size_limit(PATTERN_SIZE_LIMIT)
        .dfa_size_limit(DFA_SIZE_LIMIT)
        .build()
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("invalid or oversized regex '{}': {}", pattern, error),
        })
}

/// Resolve the path the LLM provided to a canonical absolute path, with all symlink components
/// pre-resolved. Used by file tools to close a TOCTOU window where a symlink in the supplied path
/// could be swapped between the permission check and the actual I/O. Callers should use the
/// returned `PathBuf` for every subsequent filesystem operation; never re-open the original raw
/// string.
///
/// Errors when the path cannot be resolved (target missing, parent not a directory, permission
/// denied, etc.). For `write_file` where the target file may not exist yet, callers must
/// canonicalize the *parent* directory (which they create first) and re-join the filename. Falling
/// back to the raw path on failure would leave `..`/symlink components in parent directories
/// unresolved, defeating the TOCTOU protection.
pub(super) async fn canonicalize_for_tool(tool_name: &str, path: &Path) -> Result<PathBuf> {
    tokio::fs::canonicalize(path)
        .await
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("failed to resolve path '{}': {}", path.display(), error),
        })
}

/// Run a user-supplied regex against `content` line-by-line, returning `line_number:line` rows in a
/// `ToolOutput`. Caps results at [`MAX_SEARCH_MATCHES`] and reports the total when truncated.
/// `tool_name` is used only for the regex-compile error path.
pub(super) fn search_lines(content: &str, pattern: &str, tool_name: &str) -> Result<ToolOutput> {
    let re = compile_user_regex(pattern, tool_name)?;

    let mut matches = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        if re.is_match(line) {
            matches.push(format!("{}:{}", line_num + 1, line));
            if matches.len() >= MAX_SEARCH_MATCHES {
                break;
            }
        }
    }

    if matches.is_empty() {
        return Ok(ToolOutput::text(
            "No matches found for the given regex pattern.".to_string(),
            false,
        ));
    }

    let total_matches = if matches.len() >= MAX_SEARCH_MATCHES {
        let remaining: usize = content
            .lines()
            .skip(matches.len())
            .filter(|line| re.is_match(line))
            .count();
        matches.len() + remaining
    } else {
        matches.len()
    };

    let mut result = matches.join("\n");
    if total_matches > MAX_SEARCH_MATCHES {
        result.push_str(&format!(
            "\n\n... (showing first {} of {} matches)",
            MAX_SEARCH_MATCHES, total_matches,
        ));
    }

    Ok(ToolOutput::text(result, false))
}

pub(super) fn truncate_string(string: &str, max_length: usize) -> &str {
    if string.len() <= max_length {
        string
    } else {
        &string[..string.floor_char_boundary(max_length)]
    }
}

/// Whether the caller is redirecting this tool's output into the scratchpad via the `scratchpad`
/// parameter. Tools that internally cap result counts or output length should lift those caps when
/// this returns true, because the scratchpad is an overflow buffer and truncation defeats its
/// purpose.
pub(super) fn redirects_to_scratchpad(input: &serde_json::Value) -> bool {
    input
        .get("scratchpad")
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 5), "hello");
    }

    #[test]
    fn test_compile_user_regex_rejects_oversized() {
        // Pattern that compiles to a gigantic automaton; must be rejected by the size limit rather
        // than consume host memory.
        let result = compile_user_regex("a{10000000}", "test_tool");
        assert!(result.is_err(), "oversized pattern should be rejected");
    }

    #[test]
    fn test_compile_user_regex_accepts_normal_pattern() {
        let re = compile_user_regex(r"\d+", "test_tool").expect("normal pattern compiles");
        assert!(re.is_match("abc 123"));
    }

    #[test]
    fn test_compile_user_regex_rejects_invalid_syntax() {
        let result = compile_user_regex("[invalid", "test_tool");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_canonicalize_for_tool_resolves_existing() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("a.txt");
        std::fs::write(&file_path, "x").expect("write");

        let canonical = canonicalize_for_tool("test_tool", &file_path)
            .await
            .expect("canonicalize");
        assert_eq!(
            canonical,
            std::fs::canonicalize(&file_path).expect("canonical")
        );
    }

    #[tokio::test]
    async fn test_canonicalize_for_tool_errors_on_missing() {
        let result = canonicalize_for_tool(
            "test_tool",
            std::path::Path::new("/this/path/definitely/does/not/exist-xyzzy"),
        )
        .await;
        let err = result.expect_err("missing path should error");
        let message = err.to_string();
        assert!(
            message.contains("failed to resolve path"),
            "unexpected error message: {}",
            message,
        );
    }

    #[test]
    fn test_redirects_to_scratchpad() {
        assert!(redirects_to_scratchpad(
            &serde_json::json!({ "scratchpad": "img" })
        ));
        assert!(!redirects_to_scratchpad(
            &serde_json::json!({ "scratchpad": "" })
        ));
        assert!(!redirects_to_scratchpad(&serde_json::json!({})));
        assert!(!redirects_to_scratchpad(
            &serde_json::json!({ "from_scratchpad": "img" })
        ));
    }
}
