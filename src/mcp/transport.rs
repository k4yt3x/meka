//! Transport construction helpers for MCP: building stdio child commands and the streamable-HTTP
//! transport config (headers, auth tokens, dynamic header helpers).

use tokio::process::Command;

use crate::{
    config::McpServerConfig,
    error::{MekaError, Result},
};

/// Build a [`Command`] for a stdio MCP server, wrapping shell shims in `cmd /c` on Windows so
/// `npx`, `*.cmd`, and `*.bat` executables can be launched directly as a command string. Unix paths
/// pass through unchanged.
pub(crate) fn build_stdio_command(command_str: &str, args: &[String]) -> Command {
    #[cfg(windows)]
    {
        let lower = command_str.to_ascii_lowercase();
        let is_shim = lower == "npx"
            || lower == "yarn"
            || lower == "pnpm"
            || lower.ends_with(".cmd")
            || lower.ends_with(".bat")
            || lower.ends_with(".ps1");
        if is_shim {
            // The args are not shell-quoted here: `Command` escapes them for CreateProcess's
            // lpCommandLine itself.
            let mut cmd = Command::new("cmd");
            cmd.arg("/c").arg(command_str).args(args);
            return cmd;
        }
    }
    let _ = (command_str, args);
    let mut cmd = Command::new(command_str);
    cmd.args(args);
    cmd
}

/// Build the shared HTTP transport config (URL, bearer token, custom headers) used by both the auth
/// and no-auth paths.
pub(super) async fn build_http_transport_config(
    server_name: &str,
    config: &McpServerConfig,
    // The server's stored bearer, already read from `mcp_credentials`. Passed in rather than
    // looked up here because a transport has no business knowing where a secret lives.
    bearer: Option<&str>,
) -> Result<rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig> {
    let url = config
        .url
        .as_deref()
        .ok_or_else(|| MekaError::McpConnection {
            server_name: server_name.to_string(),
            message: "http transport requires `url`".to_string(),
        })?;

    let mut transport_config =
        rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(url);

    if let Some(token) = bearer {
        transport_config = transport_config.auth_header(token.to_string());
    }

    // Merge dynamic headers from the optional `headers_helper` script on top of the static
    // `headers` map (dynamic values override static ones).
    let mut merged_headers: std::collections::HashMap<String, String> =
        config.headers.clone().unwrap_or_default();
    if let Some(script) = &config.headers_helper {
        let dynamic = run_headers_helper(server_name, url, script).await?;
        merged_headers.extend(dynamic);
    }

    if !merged_headers.is_empty() {
        let mut header_map = std::collections::HashMap::new();
        for (key, value) in &merged_headers {
            let header_name =
                reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
                    MekaError::McpConnection {
                        server_name: server_name.to_string(),
                        message: format!("invalid header name '{key}': {error}"),
                    }
                })?;
            let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|error| {
                MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!("invalid header value for '{key}': {error}"),
                }
            })?;
            header_map.insert(header_name, header_value);
        }
        transport_config = transport_config.custom_headers(header_map);
    }

    Ok(transport_config)
}

/// Execute `headers_helper` and parse its stdout as an `Name: Value\n` stream, returning a map
/// merged into the HTTP transport's custom headers.
///
/// The script runs on every connect (not per request) with a 15-second wall-clock timeout, as the
/// root of a tree of its own, so a timeout or a connect that is dropped ends whatever it started.
/// `MEKA_MCP_SERVER_NAME` and `MEKA_MCP_SERVER_URL` are injected so one helper can serve multiple
/// servers.
async fn run_headers_helper(
    server_name: &str,
    url: &str,
    script: &str,
) -> Result<std::collections::HashMap<String, String>> {
    use std::process::Stdio;

    use crate::process::{Capture, Ended, Keep};

    let connection_error = |message: String| MekaError::McpConnection {
        server_name: server_name.to_string(),
        message,
    };

    // Absolute, or relative to the directory `config.toml` is in: never to the process's working
    // directory, which would run a different helper per launch directory for one config, and
    // never a bare name searched on `PATH`, which would run whatever the path holds under that
    // name.
    let script_path = std::path::Path::new(script);
    let resolved: std::path::PathBuf = if script_path.is_absolute() {
        script_path.to_path_buf()
    } else {
        let Some(config_dir) = crate::paths::meka_config_dir() else {
            return Err(connection_error(format!(
                "`headers_helper` '{script}' is relative and no config directory resolves to \
                 place it beside"
            )));
        };
        config_dir.join(script)
    };
    if !resolved.is_file() {
        return Err(connection_error(format!(
            "`headers_helper` '{script}' is not a file at {}",
            resolved.display()
        )));
    }

    let mut command = Command::new(&resolved);
    command
        .env("MEKA_MCP_SERVER_NAME", server_name)
        .env("MEKA_MCP_SERVER_URL", url)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = crate::process::spawn_tree(&mut command).map_err(|error| {
        connection_error(format!(
            "failed to spawn `headers_helper` '{script}': {error}"
        ))
    })?;

    // stdout is the header list, rarely more than a few KiB; stderr is repeated verbatim in the
    // error message, so it is kept tight.
    const MAX_HELPER_STDOUT_BYTES: usize = 64 * crate::text::KIB;
    const MAX_HELPER_STDERR_BYTES: usize = 4 * crate::text::KIB;
    // How long the helper's output is waited for once it has exited: a process it started and
    // left behind can hold the pipe open, and the connect is not held for it.
    const HELPER_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

    // Read while the helper runs, not after it exits: one that writes more than a pipe holds
    // blocks until it is read, and would otherwise sit out the timeout. What is past the cap is
    // read and dropped, for the same reason.
    let capture = Capture {
        stdout: Keep::Head(MAX_HELPER_STDOUT_BYTES),
        stderr: Keep::Head(MAX_HELPER_STDERR_BYTES),
        sink: None,
    };
    let finished = crate::process::run(
        child,
        capture,
        Some(std::time::Duration::from_secs(15)),
        None,
        HELPER_DRAIN_TIMEOUT,
    )
    .await
    .map_err(|error| {
        connection_error(format!(
            "failed to wait for `headers_helper` '{script}': {error}"
        ))
    })?;
    let status = match finished.ended {
        Ended::Exited(status) => status,
        // Nothing stops the helper and nothing spends a budget, so its timeout is the one way it
        // is ended.
        Ended::TimedOut | Ended::Stopped | Ended::OutputBound => {
            return Err(connection_error(format!(
                "`headers_helper` '{script}' timed out after 15s"
            )));
        }
    };
    if finished.drained.stopped_early {
        tracing::debug!(
            "`headers_helper` '{script}' left its output open past its exit; its headers are what \
             it printed until then"
        );
    }

    if !status.success() {
        let stderr_text = String::from_utf8_lossy(&finished.drained.stderr.bytes);
        return Err(connection_error(format!(
            "`headers_helper` '{}' exited with status {}: {}",
            script,
            status.code().unwrap_or(-1),
            stderr_text.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&finished.drained.stdout.bytes);
    parse_header_lines(&stdout).map_err(|message| {
        connection_error(format!("`headers_helper` '{script}' output: {message}"))
    })
}

fn parse_header_lines(
    text: &str,
) -> std::result::Result<std::collections::HashMap<String, String>, String> {
    let mut out = std::collections::HashMap::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| format!("line {} missing ':' separator", line_no + 1))?;
        out.insert(name.trim().to_string(), value.trim().to_string());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::*;

    fn http_server(name: &str) -> McpServerConfig {
        // Parsed rather than constructed field-by-field, so a field added to `McpServerConfig`
        // does not silently make this fixture unrepresentative of what a user actually writes.
        let toml = format!(
            "name = \"{name}\"\ntransport = \"http\"\nurl = \"https://api.example.com/mcp\"\n"
        );
        toml::from_str(&toml).expect("the fixture parses")
    }

    /// The one place a stored bearer becomes a request header. Everything upstream of here can be
    /// right and the server still sees no `Authorization` if this drops it.
    #[tokio::test]
    async fn a_stored_bearer_becomes_the_authorization_header() {
        let config = build_http_transport_config(
            "api",
            &http_server("api"),
            Some("bearer-not-a-real-token"),
        )
        .await
        .expect("builds");

        assert_eq!(&*config.uri, "https://api.example.com/mcp");
        assert_eq!(
            config.auth_header.as_deref(),
            Some("bearer-not-a-real-token")
        );
    }

    /// A server that authenticates through a flow has no stored bearer, and must send no
    /// `Authorization` of its own: rmcp's auth layer sets that header itself.
    #[tokio::test]
    async fn no_bearer_means_no_authorization_header() {
        let config = build_http_transport_config("api", &http_server("api"), None)
            .await
            .expect("builds");
        assert!(config.auth_header.is_none());
    }

    /// The bearer and `headers` are separate inputs and both have to survive.
    #[tokio::test]
    async fn a_bearer_and_custom_headers_coexist() {
        let mut server = http_server("api");
        server.headers = Some(
            [("X-Tenant-Id".to_string(), "acme".to_string())]
                .into_iter()
                .collect(),
        );

        let config = build_http_transport_config("api", &server, Some("bearer-not-a-real-token"))
            .await
            .expect("builds");

        assert_eq!(
            config.auth_header.as_deref(),
            Some("bearer-not-a-real-token")
        );
        assert_eq!(
            config
                .custom_headers
                .get(&reqwest::header::HeaderName::from_static("x-tenant-id"))
                .and_then(|value| value.to_str().ok()),
            Some("acme")
        );
    }

    #[test]
    fn parse_header_lines_basic() {
        let map = parse_header_lines("X-Token: abc\nX-Env: prod\n").expect("parses");
        assert_eq!(map.get("X-Token").map(String::as_str), Some("abc"));
        assert_eq!(map.get("X-Env").map(String::as_str), Some("prod"));
    }

    #[test]
    fn parse_header_lines_skips_blank_and_comment_lines() {
        let map = parse_header_lines("# a comment\n\n  \nX-Real: yes\n").expect("parses");
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("X-Real").map(String::as_str), Some("yes"));
    }

    #[test]
    fn parse_header_lines_trims_surrounding_whitespace() {
        let map = parse_header_lines("  Name  :   value with spaces  \n").expect("parses");
        assert_eq!(
            map.get("Name").map(String::as_str),
            Some("value with spaces")
        );
    }

    #[test]
    fn parse_header_lines_value_may_contain_colons() {
        // Only the first ':' separates; URL values must survive intact.
        let map = parse_header_lines("Location: https://example.com/x\n").expect("parses");
        assert_eq!(
            map.get("Location").map(String::as_str),
            Some("https://example.com/x")
        );
    }

    /// `headers_helper` is found beside `config.toml` or where an absolute path says, and a bare
    /// name is not searched on `PATH`: a helper that runs whatever the launch directory or the
    /// path holds under that name is a persistence primitive.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_headers_helper_is_found_beside_the_config_and_nowhere_else() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("tempdir");
        let script = dir.path().join("helper.sh");
        std::fs::write(&script, "#!/bin/sh\nprintf 'X-Test: found\\n'\n").expect("write");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).expect("chmod");
        // SAFETY: guarded above.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let beside = run_headers_helper("s", "http://example.test", "helper.sh").await;
        let absolute =
            run_headers_helper("s", "http://example.test", &script.to_string_lossy()).await;
        let bare = run_headers_helper("s", "http://example.test", "sh").await;
        // SAFETY: guarded above.
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        for (how, headers) in [("beside the config", beside), ("absolute", absolute)] {
            assert_eq!(
                headers.expect(how).get("X-Test").map(String::as_str),
                Some("found"),
                "{how}"
            );
        }
        let error = bare
            .expect_err("a bare name is not searched on PATH")
            .to_string();
        assert!(error.contains("is not a file"), "{error}");
    }

    /// A helper that writes more than a pipe holds, to stderr here, is read while it runs rather
    /// than left blocked on the write until the timeout kills it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_chatty_headers_helper_still_finishes() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let script = dir.path().join("helper.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\nhead -c 262144 /dev/zero | tr '\\0' x >&2\nprintf 'X-Test: found\\n'\n",
        )
        .expect("write");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).expect("chmod");
        let started = std::time::Instant::now();
        let headers = run_headers_helper("s", "http://example.test", &script.to_string_lossy())
            .await
            .expect("the helper's headers");
        assert_eq!(headers.get("X-Test").map(String::as_str), Some("found"));
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
    }

    /// A helper that exits having started something that keeps its stdout open still answers
    /// within its budget, with what it printed: the connect is not held until whatever it left
    /// behind lets go of the pipe.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_helper_that_leaves_its_stdout_held_still_answers() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let script = dir.path().join("helper.sh");
        let held_by = dir.path().join("held-by");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf 'X-Test: found\\n'\nsleep 30 &\necho $! > '{}'\n",
                held_by.display()
            ),
        )
        .expect("write");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).expect("chmod");

        let answered = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            run_headers_helper("s", "http://example.test", &script.to_string_lossy()),
        )
        .await;
        // The `sleep` outlives a helper that exited on its own, so it is ended here, by the pid it
        // wrote, rather than left for its 30 seconds.
        let pid: libc::pid_t = std::fs::read_to_string(&held_by)
            .expect("the helper wrote the pid")
            .trim()
            .parse()
            .expect("a pid");
        // SAFETY: `kill(2)` has no preconditions.
        unsafe { libc::kill(pid, libc::SIGKILL) };

        let headers = answered
            .expect("the helper answers within its budget")
            .expect("the helper's headers");
        assert_eq!(headers.get("X-Test").map(String::as_str), Some("found"));
    }

    #[test]
    fn parse_header_lines_rejects_missing_separator() {
        let err = parse_header_lines("Valid: ok\nbroken line\n").expect_err("must fail");
        assert!(err.contains("line 2"), "error should cite line 2: {err}");
    }

    #[test]
    fn build_stdio_command_passes_through_program_and_args() {
        let args = vec!["--flag".to_string(), "value".to_string()];
        let cmd = build_stdio_command("my-server", &args);
        let std_cmd = cmd.as_std();
        assert_eq!(std_cmd.get_program(), OsStr::new("my-server"));
        let collected: Vec<_> = std_cmd.get_args().collect();
        assert_eq!(collected, vec![OsStr::new("--flag"), OsStr::new("value")]);
    }

    #[cfg(windows)]
    #[test]
    fn build_stdio_command_wraps_npx_in_cmd_shim() {
        let args = vec!["my-pkg".to_string()];
        let cmd = build_stdio_command("npx", &args);
        let std_cmd = cmd.as_std();
        assert_eq!(std_cmd.get_program(), OsStr::new("cmd"));
        let collected: Vec<_> = std_cmd.get_args().collect();
        assert_eq!(collected, vec![
            OsStr::new("/c"),
            OsStr::new("npx"),
            OsStr::new("my-pkg")
        ]);
    }
}
