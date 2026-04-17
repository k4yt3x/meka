use crate::error::{AgshError, Result};

pub(super) fn require_str(
    input: &serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<String> {
    input[field]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| AgshError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("missing '{}' parameter", field),
        })
}

pub(super) fn truncate_string(string: &str, max_length: usize) -> &str {
    if string.len() <= max_length {
        string
    } else {
        &string[..string.floor_char_boundary(max_length)]
    }
}

/// Whether the caller is redirecting this tool's output into the scratchpad
/// via the `scratchpad` parameter. Tools that internally cap result counts or
/// output length should lift those caps when this returns true, because the
/// scratchpad is an overflow buffer and truncation defeats its purpose.
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
