//! Environment-variable substitution for MCP server config strings.
//!
//! Supports `${VAR}` and `${VAR:-default}` syntax. Missing variables with no default leave the
//! literal `${VAR}` in place (matches the behavior of Claude Code) and accumulate a warning.
//! Applied to every string field that could reasonably reference a user secret: stdio
//! `command`/`args`/`env` values, and HTTP `url`/`headers` values.

use std::collections::HashMap;

use crate::config::McpServerConfig;

/// Expand `${VAR}` / `${VAR:-default}` in `input`, consulting `lookup` (defaults to process
/// environment). Returns the expanded string alongside the names of any variables that were missing
/// and had no default.
pub(crate) fn expand_env_vars<F>(input: &str, mut lookup: F) -> (String, Vec<String>)
where
    F: FnMut(&str) -> Option<String>,
{
    let mut out = String::with_capacity(input.len());
    let mut missing = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Fast path: any byte that is not the start of a `${…}` opener is copied verbatim by
        // reading the next UTF-8 scalar and pushing the whole codepoint. Byte-level `as char` would
        // mangle multi-byte sequences (e.g. `café` → `cafÃ©`).
        if bytes[i] != b'$' || i + 1 >= bytes.len() || bytes[i + 1] != b'{' {
            // `i` is always on a char boundary: it advances by `ch.len_utf8()` or to the byte
            // after an ASCII `}`.
            let rest = &input[i..];
            #[allow(
                clippy::expect_used,
                reason = "the `while i < bytes.len()` guard keeps `rest` non-empty"
            )]
            let ch = rest.chars().next().expect("non-empty slice");
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        // Find matching `}`.
        let start = i + 2;
        let Some(end_offset) = bytes[start..].iter().position(|&b| b == b'}') else {
            // Unterminated `${`: emit verbatim and stop scanning.
            out.push_str(&input[i..]);
            break;
        };
        let end = start + end_offset;
        let body = &input[start..end];
        // Split on `:-` (limit 2 so `:-` can appear inside the default).
        let (var_name, default) = match body.split_once(":-") {
            Some((name, def)) => (name.trim(), Some(def)),
            None => (body.trim(), None),
        };
        if var_name.is_empty() {
            // `${}`: emit verbatim.
            out.push_str(&input[i..=end]);
        } else {
            match lookup(var_name) {
                Some(value) => out.push_str(&value),
                None => match default {
                    Some(def) => out.push_str(def),
                    None => {
                        missing.push(var_name.to_string());
                        out.push_str(&input[i..=end]);
                    }
                },
            }
        }
        i = end + 1;
    }
    (out, missing)
}

/// What `${VAR}` expansion could not resolve in a server's config.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Unresolved {
    /// Every name left literal, in first-seen order.
    pub(crate) names: Vec<String>,
    /// Whether any of them sat in `headers` or `env`, the two maps a credential lives in. A
    /// literal `Bearer ${TOKEN}` sent to a third party is a request that cannot succeed and a
    /// string the operator meant to keep in the environment; the server is refused rather than
    /// connected.
    pub(crate) in_secret_bearing_fields: bool,
}

/// Walk every expandable string inside `config` and apply [`expand_env_vars`] with the process
/// environment.
pub(crate) fn expand_server_config(config: &mut McpServerConfig) -> Unresolved {
    let mut unresolved = Unresolved::default();
    let mut record = |missing: Vec<String>, secret_bearing: bool| {
        if secret_bearing && !missing.is_empty() {
            unresolved.in_secret_bearing_fields = true;
        }
        for name in missing {
            if !unresolved.names.contains(&name) {
                unresolved.names.push(name);
            }
        }
    };

    let lookup = |name: &str| std::env::var(name).ok();

    if let Some(command) = &config.command {
        let (expanded, missing) = expand_env_vars(command, lookup);
        record(missing, false);
        config.command = Some(expanded);
    }
    if let Some(args) = &mut config.args {
        for arg in args {
            let (expanded, missing) = expand_env_vars(arg, lookup);
            record(missing, false);
            *arg = expanded;
        }
    }
    if let Some(env) = &mut config.env {
        let mut new_env: HashMap<String, String> = HashMap::with_capacity(env.len());
        for (key, value) in env.iter() {
            let (expanded, missing) = expand_env_vars(value, lookup);
            record(missing, true);
            new_env.insert(key.clone(), expanded);
        }
        *env = new_env;
    }
    if let Some(url) = &config.url {
        let (expanded, missing) = expand_env_vars(url, lookup);
        record(missing, false);
        config.url = Some(expanded);
    }
    if let Some(headers) = &mut config.headers {
        let mut new_headers: HashMap<String, String> = HashMap::with_capacity(headers.len());
        for (key, value) in headers.iter() {
            let (expanded, missing) = expand_env_vars(value, lookup);
            record(missing, true);
            new_headers.insert(key.clone(), expanded);
        }
        *headers = new_headers;
    }
    // The helper is a command line like `command`.
    if let Some(helper) = &config.headers_helper {
        let (expanded, missing) = expand_env_vars(helper, lookup);
        record(missing, false);
        config.headers_helper = Some(expanded);
    }
    unresolved
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed<'a>(map: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name: &str| {
            map.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn a_set_variable_expands_in_place() {
        let (out, missing) = expand_env_vars("hello ${FOO} world", fixed(&[("FOO", "bar")]));
        assert_eq!(out, "hello bar world");
        assert!(missing.is_empty());
    }

    #[test]
    fn a_missing_variable_takes_its_default() {
        let (out, missing) = expand_env_vars("${MISSING:-fallback}", fixed(&[]));
        assert_eq!(out, "fallback");
        assert!(missing.is_empty());
    }

    #[test]
    fn keeps_literal_when_missing_and_no_default() {
        let (out, missing) = expand_env_vars("x${MISSING}y", fixed(&[]));
        assert_eq!(out, "x${MISSING}y");
        assert_eq!(missing, vec!["MISSING".to_string()]);
    }

    #[test]
    fn preserves_colon_dash_in_default() {
        let (out, missing) = expand_env_vars("${MISSING:-a:-b:-c}", fixed(&[]));
        assert_eq!(out, "a:-b:-c");
        assert!(missing.is_empty());
    }

    #[test]
    fn an_unterminated_reference_passes_through() {
        let (out, missing) = expand_env_vars("tail ${ENDLESS", fixed(&[]));
        assert_eq!(out, "tail ${ENDLESS");
        assert!(missing.is_empty());
    }

    #[test]
    fn dollar_without_brace_is_passthrough() {
        let (out, missing) = expand_env_vars("$FOO", fixed(&[("FOO", "bar")]));
        assert_eq!(out, "$FOO");
        assert!(missing.is_empty());
    }

    #[test]
    fn empty_braces_pass_through() {
        let (out, _) = expand_env_vars("${}", fixed(&[]));
        assert_eq!(out, "${}");
    }

    #[test]
    fn multiple_vars_same_missing_dedup() {
        let (_out, missing) = expand_env_vars("${X} ${X} ${Y}", fixed(&[]));
        // Not deduped at this layer; the caller dedupes across fields.
        assert_eq!(missing, vec![
            "X".to_string(),
            "X".to_string(),
            "Y".to_string()
        ]);
    }

    #[test]
    fn preserves_multibyte_utf8_outside_placeholders() {
        let (out, missing) = expand_env_vars("café=${FOO} 日本語🦀", fixed(&[("FOO", "bar")]));
        assert_eq!(out, "café=bar 日本語🦀");
        assert!(missing.is_empty());
    }

    #[test]
    fn preserves_multibyte_utf8_with_missing_var() {
        let (out, missing) = expand_env_vars("ümlaut ${M} 末", fixed(&[]));
        assert_eq!(out, "ümlaut ${M} 末");
        assert_eq!(missing, vec!["M".to_string()]);
    }

    /// The walk over a whole server, rather than the substitution itself.
    ///
    /// Uses names that are never set and the `:-default` form, so it exercises both directions
    /// without touching the process environment, which the rest of the suite shares.
    #[test]
    fn the_walk_expands_every_field_and_reports_what_it_could_not_resolve() {
        let mut config: McpServerConfig = toml::from_str(
            r#"
name = "api"
transport = "http"
url = "https://${MEKA_TEST_UNSET_HOST:-example.test}/mcp"
headers = { X-Tenant = "${MEKA_TEST_UNSET_TENANT}", X-Fixed = "plain" }
"#,
        )
        .expect("the fixture parses");

        let missing = expand_server_config(&mut config);

        assert_eq!(
            config.url.as_deref(),
            Some("https://example.test/mcp"),
            "a default must be applied"
        );
        let headers = config.headers.as_ref().expect("headers");
        assert_eq!(
            headers.get("X-Tenant").map(String::as_str),
            Some("${MEKA_TEST_UNSET_TENANT}"),
            "an unresolvable name is left literal rather than blanked"
        );
        assert_eq!(headers.get("X-Fixed").map(String::as_str), Some("plain"));
        assert_eq!(
            missing.names,
            vec!["MEKA_TEST_UNSET_TENANT".to_string()],
            "only the one with no default is reported, and the warning depends on this list"
        );
        assert!(
            missing.in_secret_bearing_fields,
            "a header is where a credential lives, so the server is refused rather than connected"
        );
    }

    /// A name unresolved in `command` or `url` is reported but does not bar the server: neither is
    /// where a credential lives, and the request it produces fails on its own.
    #[test]
    fn an_unresolved_name_outside_headers_and_env_does_not_bar_the_server() {
        let mut config: McpServerConfig = toml::from_str(
            "name = \"api\"\ntransport = \"stdio\"\ncommand = \"${MEKA_TEST_UNSET_BIN}\"\n",
        )
        .expect("the fixture parses");
        let unresolved = expand_server_config(&mut config);
        assert_eq!(unresolved.names, vec!["MEKA_TEST_UNSET_BIN".to_string()]);
        assert!(!unresolved.in_secret_bearing_fields);
    }

    /// `headers_helper` is a command line like `command`, and was the one field the walk skipped.
    #[test]
    fn the_headers_helper_is_expanded_too() {
        let mut config: McpServerConfig = toml::from_str(
            "name = \"api\"\ntransport = \"http\"\nurl = \"https://example.test/mcp\"\nheaders_helper = \"${MEKA_TEST_UNSET_HELPER:-helper --token}\"\n",
        )
        .expect("the fixture parses");
        let unresolved = expand_server_config(&mut config);
        assert!(unresolved.names.is_empty(), "{unresolved:?}");
        assert_eq!(config.headers_helper.as_deref(), Some("helper --token"));
    }

    /// A server with nothing to expand reports nothing, so startup is silent.
    #[test]
    fn the_walk_reports_nothing_when_there_is_nothing_to_resolve() {
        let mut config: McpServerConfig = toml::from_str(
            "name = \"api\"\ntransport = \"http\"\nurl = \"https://example.test/mcp\"\n",
        )
        .expect("the fixture parses");
        assert!(expand_server_config(&mut config).names.is_empty());
    }
}
