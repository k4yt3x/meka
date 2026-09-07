//! String hygiene helpers for data crossing the MCP boundary. Strips invisible/control characters
//! that could be used for prompt injection or UI spoofing (RTL/LTR overrides, zero-width joiners,
//! C0/C1 controls) and normalizes MCP server names to the alphabet accepted by provider tool
//! schemas.

/// Reserved server names that collide with meka internals or with the tool namespace separator.
/// Connection requests for these names are rejected.
pub(crate) const RESERVED_SERVER_NAMES: &[&str] = &["meka", "ide"];

/// Normalize a user-supplied MCP server name into the alphabet accepted as the `<server>` segment
/// of a `mcp__<server>__<tool>` tool name.
///
/// Any character outside `[A-Za-z0-9_-]` is replaced with `_`; runs of `_` are collapsed, and
/// leading/trailing `_` are trimmed. Empty results are mapped to `"mcp_server"` (the caller is
/// expected to reject that via the reserved check anyway, but this keeps the string non-empty).
pub(crate) fn normalize_server_name(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_underscore = false;
    for ch in input.chars() {
        let mapped = if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            ch
        } else {
            '_'
        };
        if mapped == '_' {
            if last_underscore {
                continue;
            }
            last_underscore = true;
        } else {
            last_underscore = false;
        }
        out.push(mapped);
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "mcp_server".to_string()
    } else {
        trimmed
    }
}

/// Returns true when the given name is reserved (either an explicit entry in
/// [`RESERVED_SERVER_NAMES`] or starts with the `mcp_` prefix that meka uses for internal tools).
pub(crate) fn is_reserved_server_name(name: &str) -> bool {
    if RESERVED_SERVER_NAMES
        .iter()
        .any(|r| r.eq_ignore_ascii_case(name))
    {
        return true;
    }
    name.to_ascii_lowercase().starts_with("mcp_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::sanitize_text;

    #[test]
    fn sanitize_strips_rtl_override() {
        let input = "ok\u{202E}reversed";
        assert_eq!(sanitize_text(input), "okreversed");
    }

    #[test]
    fn sanitize_strips_zero_width_space() {
        let input = "a\u{200B}b\u{200C}c\u{200D}d";
        assert_eq!(sanitize_text(input), "abcd");
    }

    #[test]
    fn sanitize_strips_bom() {
        let input = "\u{FEFF}hello";
        assert_eq!(sanitize_text(input), "hello");
    }

    #[test]
    fn sanitize_keeps_emoji() {
        let input = "hi 🦀 from rust";
        assert_eq!(sanitize_text(input), "hi 🦀 from rust");
    }

    #[test]
    fn sanitize_keeps_cjk() {
        let input = "日本語と한국어";
        assert_eq!(sanitize_text(input), "日本語と한국어");
    }

    #[test]
    fn sanitize_keeps_newlines_and_tabs() {
        let input = "line1\nline2\tcol";
        assert_eq!(sanitize_text(input), "line1\nline2\tcol");
    }

    /// `\r` is the one control that can forge UI without an escape sequence: it moves the cursor
    /// back to column zero, so a server can overwrite a line meka already wrote. Every render site
    /// for server-supplied text relies on this function, so the strip has to happen here.
    #[test]
    fn sanitize_strips_carriage_return() {
        assert_eq!(
            sanitize_text("\r[mcp elicit: trusted] paste your token:"),
            "[mcp elicit: trusted] paste your token:"
        );
        assert_eq!(sanitize_text("done\r\nnext"), "done\nnext");
    }

    #[test]
    fn sanitize_strips_c0_control() {
        let input = "a\x01b\x08c";
        assert_eq!(sanitize_text(input), "abc");
    }

    #[test]
    fn sanitize_strips_c1_control() {
        let input = "a\u{0085}b"; // NEL
        assert_eq!(sanitize_text(input), "ab");
    }

    #[test]
    fn a_clean_server_name_normalizes_to_itself() {
        assert_eq!(normalize_server_name("postgres"), "postgres");
    }

    #[test]
    fn normalize_replaces_invalid() {
        assert_eq!(normalize_server_name("my.server"), "my_server");
        assert_eq!(normalize_server_name("a b c"), "a_b_c");
    }

    #[test]
    fn normalize_collapses_runs() {
        assert_eq!(normalize_server_name("a...b"), "a_b");
    }

    #[test]
    fn normalize_trims_underscores() {
        assert_eq!(normalize_server_name("...foo..."), "foo");
    }

    #[test]
    fn normalize_empty_result() {
        assert_eq!(normalize_server_name("..."), "mcp_server");
    }

    #[test]
    fn normalize_keeps_dashes() {
        assert_eq!(normalize_server_name("my-server-1"), "my-server-1");
    }

    #[test]
    fn reserved_server_names_are_recognized_case_insensitively() {
        assert!(is_reserved_server_name("meka"));
        assert!(is_reserved_server_name("IDE"));
        assert!(is_reserved_server_name("mcp_resources"));
        assert!(!is_reserved_server_name("postgres"));
    }
}
