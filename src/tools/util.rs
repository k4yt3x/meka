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

pub(super) fn ceil_char_boundary(string: &str, index: usize) -> usize {
    if index >= string.len() {
        return string.len();
    }
    let mut boundary = index;
    while boundary < string.len() && !string.is_char_boundary(boundary) {
        boundary += 1;
    }
    boundary
}

pub(super) fn truncate_string(string: &str, max_length: usize) -> &str {
    if string.len() <= max_length {
        string
    } else {
        &string[..string.floor_char_boundary(max_length)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 5), "hello");
    }
}
