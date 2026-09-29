//! Environment-variable substitution for MCP server config strings.
//!
//! Supports `${VAR}` and `${VAR:-default}` syntax. Missing variables with no default leave the
//! literal `${VAR}` in place (matches the behavior of Claude Code) and accumulate a warning.
//! Applied to every string field: stdio `command`/`args`/`env` values, HTTP `url`/`headers`
//! values and `headers_helper`. A miss in a field that reaches the other side (`args`, `env`,
//! `url`, `headers`) bars the server, so a literal `${TOKEN}` is never sent to anyone; a miss in
//! `command` or `headers_helper` names a program that will not be found, and fails on its own.

use std::collections::HashMap;

use crate::config::{Expansion, McpServerConfig, expand_env_vars};

/// What `${VAR}` expansion could not resolve in a server's config.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Unresolved {
    /// Every name left literal, in first-seen order.
    pub(crate) names: Vec<String>,
    /// Every reference the grammar does not accept, as written and in first-seen order; see
    /// [`crate::config::Expansion::malformed`].
    pub(crate) malformed: Vec<String>,
    /// Whether any of them sat in `args`, `env`, `url` or `headers`, the fields that reach the
    /// other side. A literal `Bearer ${TOKEN}` sent to a third party is a request that cannot
    /// succeed and a string the operator meant to keep in the environment; the server is refused
    /// rather than connected.
    pub(crate) in_secret_bearing_fields: bool,
}

impl Unresolved {
    /// What was left literal, for the refusal and the warning that name it.
    pub(crate) fn describe(&self) -> String {
        let mut parts = Vec::new();
        if !self.names.is_empty() {
            parts.push(format!(
                "environment variable(s) {:?} are unset",
                self.names
            ));
        }
        if !self.malformed.is_empty() {
            parts.push(format!(
                "reference(s) {:?} are not `${{VAR}}` or `${{VAR:-default}}`",
                self.malformed
            ));
        }
        parts.join(" and ")
    }
}

/// Walk every expandable string inside `config` and apply [`expand_env_vars`] with the process
/// environment.
pub(crate) fn expand_server_config(config: &mut McpServerConfig) -> Unresolved {
    let mut unresolved = Unresolved::default();
    let mut record = |expansion: Expansion, secret_bearing: bool| -> String {
        let Expansion {
            text,
            missing,
            malformed,
        } = expansion;
        if secret_bearing && (!missing.is_empty() || !malformed.is_empty()) {
            unresolved.in_secret_bearing_fields = true;
        }
        for name in missing {
            if !unresolved.names.contains(&name) {
                unresolved.names.push(name);
            }
        }
        for reference in malformed {
            if !unresolved.malformed.contains(&reference) {
                unresolved.malformed.push(reference);
            }
        }
        text
    };

    let lookup = |name: &str| std::env::var(name).ok();

    if let Some(command) = &config.command {
        config.command = Some(record(expand_env_vars(command, lookup), false));
    }
    if let Some(args) = &mut config.args {
        for arg in args {
            *arg = record(expand_env_vars(arg, lookup), true);
        }
    }
    if let Some(env) = &mut config.env {
        let mut new_env: HashMap<String, String> = HashMap::with_capacity(env.len());
        for (key, value) in env.iter() {
            new_env.insert(key.clone(), record(expand_env_vars(value, lookup), true));
        }
        *env = new_env;
    }
    if let Some(url) = &config.url {
        config.url = Some(record(expand_env_vars(url, lookup), true));
    }
    if let Some(headers) = &mut config.headers {
        let mut new_headers: HashMap<String, String> = HashMap::with_capacity(headers.len());
        for (key, value) in headers.iter() {
            new_headers.insert(key.clone(), record(expand_env_vars(value, lookup), true));
        }
        *headers = new_headers;
    }
    // The helper is a path meka runs, like `command`: a miss names a program that will not be
    // found, and fails on its own.
    if let Some(helper) = &config.headers_helper {
        config.headers_helper = Some(record(expand_env_vars(helper, lookup), false));
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
        let expansion = expand_env_vars("hello ${FOO} world", fixed(&[("FOO", "bar")]));
        assert_eq!(expansion.text, "hello bar world");
        assert!(expansion.missing.is_empty() && expansion.malformed.is_empty());
    }

    #[test]
    fn a_missing_variable_takes_its_default() {
        let expansion = expand_env_vars("${MISSING:-fallback}", fixed(&[]));
        assert_eq!(expansion.text, "fallback");
        assert!(expansion.missing.is_empty() && expansion.malformed.is_empty());
    }

    #[test]
    fn keeps_literal_when_missing_and_no_default() {
        let expansion = expand_env_vars("x${MISSING}y", fixed(&[]));
        assert_eq!(expansion.text, "x${MISSING}y");
        assert_eq!(expansion.missing, vec!["MISSING".to_string()]);
        assert!(expansion.malformed.is_empty());
    }

    #[test]
    fn preserves_colon_dash_in_default() {
        let expansion = expand_env_vars("${MISSING:-a:-b:-c}", fixed(&[]));
        assert_eq!(expansion.text, "a:-b:-c");
        assert!(expansion.missing.is_empty() && expansion.malformed.is_empty());
    }

    /// Left as written and reported as malformed rather than missing: nothing the environment
    /// could supply satisfies it, and a caller that fails closed on a miss must fail closed on this
    /// too, or a dropped `}` turns a credential's placeholder into the credential sent.
    #[test]
    fn an_unterminated_reference_is_malformed() {
        let expansion = expand_env_vars("tail ${ENDLESS", fixed(&[]));
        assert_eq!(expansion.text, "tail ${ENDLESS");
        assert!(expansion.missing.is_empty());
        assert_eq!(expansion.malformed, vec!["${ENDLESS".to_string()]);
    }

    /// A default that opens another reference is refused rather than cut at the inner `}`: read
    /// as `${A:-${B}` plus a literal `}`, the text would expand to `${B}` with `B` never reported.
    #[test]
    fn a_default_that_opens_another_reference_is_malformed() {
        let expansion = expand_env_vars("${A:-${B}}", fixed(&[]));
        assert_eq!(expansion.text, "${A:-${B}}");
        assert!(expansion.missing.is_empty());
        assert_eq!(expansion.malformed, vec!["${A:-${B}".to_string()]);
    }

    #[test]
    fn dollar_without_brace_is_passthrough() {
        let expansion = expand_env_vars("$FOO", fixed(&[("FOO", "bar")]));
        assert_eq!(expansion.text, "$FOO");
        assert!(expansion.missing.is_empty() && expansion.malformed.is_empty());
    }

    #[test]
    fn empty_braces_are_malformed() {
        let expansion = expand_env_vars("${}", fixed(&[]));
        assert_eq!(expansion.text, "${}");
        assert_eq!(expansion.malformed, vec!["${}".to_string()]);
    }

    #[test]
    fn multiple_vars_same_missing_dedup() {
        let expansion = expand_env_vars("${X} ${X} ${Y}", fixed(&[]));
        // Not deduped at this layer; the caller dedupes across fields.
        assert_eq!(expansion.missing, vec![
            "X".to_string(),
            "X".to_string(),
            "Y".to_string()
        ]);
    }

    #[test]
    fn preserves_multibyte_utf8_outside_placeholders() {
        let expansion = expand_env_vars("café=${FOO} 日本語🦀", fixed(&[("FOO", "bar")]));
        assert_eq!(expansion.text, "café=bar 日本語🦀");
        assert!(expansion.missing.is_empty());
    }

    #[test]
    fn preserves_multibyte_utf8_with_missing_var() {
        let expansion = expand_env_vars("ümlaut ${M} 末", fixed(&[]));
        assert_eq!(expansion.text, "ümlaut ${M} 末");
        assert_eq!(expansion.missing, vec!["M".to_string()]);
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

    /// A name unresolved in `url` or `args` bars the server as one in `headers` does: a key in a
    /// query string or a token on a command line is as much a credential, and the literal would
    /// reach the other side.
    #[test]
    fn an_unresolved_name_in_the_url_or_the_arguments_bars_the_server() {
        let mut config: McpServerConfig = toml::from_str(
            "name = \"api\"\ntransport = \"http\"\nurl = \"https://example.test/sse?key=${MEKA_TEST_UNSET_KEY}\"\n",
        )
        .expect("the fixture parses");
        let unresolved = expand_server_config(&mut config);
        assert_eq!(unresolved.names, vec!["MEKA_TEST_UNSET_KEY".to_string()]);
        assert!(
            unresolved.in_secret_bearing_fields,
            "a query-string key is a credential"
        );

        let mut config: McpServerConfig = toml::from_str(
            "name = \"cli\"\ntransport = \"stdio\"\ncommand = \"tool\"\nargs = [\"--token\", \"${MEKA_TEST_UNSET_TOKEN}\"]\n",
        )
        .expect("the fixture parses");
        let unresolved = expand_server_config(&mut config);
        assert_eq!(unresolved.names, vec!["MEKA_TEST_UNSET_TOKEN".to_string()]);
        assert!(
            unresolved.in_secret_bearing_fields,
            "a token on a command line is a credential"
        );
    }

    /// A malformed reference in a field that reaches the other side bars the server as an unset
    /// name does: `Bearer ${TOKEN` with a dropped brace is not a credential, and would be sent as
    /// one.
    #[test]
    fn a_malformed_reference_in_the_headers_bars_the_server() {
        let mut config: McpServerConfig = toml::from_str(
            "name = \"api\"\ntransport = \"http\"\nurl = \"https://example.test/mcp\"\n\
             headers = { Authorization = \"Bearer ${MEKA_TEST_UNSET_TOKEN\" }\n",
        )
        .expect("the fixture parses");
        let unresolved = expand_server_config(&mut config);
        assert!(unresolved.names.is_empty(), "{unresolved:?}");
        assert_eq!(unresolved.malformed, vec![
            "${MEKA_TEST_UNSET_TOKEN".to_string()
        ]);
        assert!(unresolved.in_secret_bearing_fields);
        assert!(
            unresolved.describe().contains("${MEKA_TEST_UNSET_TOKEN"),
            "{}",
            unresolved.describe()
        );
    }

    /// A name unresolved in `command` is reported but does not bar the server: it names a program
    /// that will not be found, and the spawn fails on its own without sending anything anywhere.
    #[test]
    fn an_unresolved_name_in_the_command_does_not_bar_the_server() {
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
            "name = \"api\"\ntransport = \"http\"\nurl = \"https://example.test/mcp\"\nheaders_helper = \"${MEKA_TEST_UNSET_HELPER:-helpers/token.sh}\"\n",
        )
        .expect("the fixture parses");
        let unresolved = expand_server_config(&mut config);
        assert!(unresolved.names.is_empty(), "{unresolved:?}");
        assert_eq!(config.headers_helper.as_deref(), Some("helpers/token.sh"));
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
