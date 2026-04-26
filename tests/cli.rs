//! End-to-end CLI smoke tests. These shell out to the built `agsh` binary
//! (`env!("CARGO_BIN_EXE_agsh")`) so they exercise the same entry point
//! users hit on the command line. They cover surface-level invariants that
//! unit tests can't reach: argument-parser wiring, `--help` output, and the
//! exit status of trivial subcommands.

use std::process::Command;

fn agsh() -> Command {
    Command::new(env!("CARGO_BIN_EXE_agsh"))
}

#[test]
fn version_flag_prints_version_and_exits_zero() {
    let output = agsh()
        .arg("--version")
        .output()
        .expect("failed to spawn agsh");
    assert!(
        output.status.success(),
        "agsh --version exited non-zero: {:?}",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("agsh "),
        "expected version output to start with 'agsh ', got: {}",
        stdout
    );
}

#[test]
fn help_flag_lists_subcommands() {
    let output = agsh().arg("--help").output().expect("failed to spawn agsh");
    assert!(output.status.success(), "agsh --help exited non-zero");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in ["setup", "export", "delete", "list"] {
        assert!(
            stdout.contains(expected),
            "--help output missing subcommand '{}':\n{}",
            expected,
            stdout
        );
    }
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    let output = agsh()
        .arg("--definitely-not-a-flag")
        .output()
        .expect("failed to spawn agsh");
    assert!(
        !output.status.success(),
        "agsh accepted an unknown flag without erroring"
    );
}

/// Run `agsh` with an isolated config + data directory so host state
/// (e.g. `~/.config/agsh/config.toml`) doesn't leak in, and the test's
/// writes don't spill out. Sets `AGSH_CONFIG_DIR` and `AGSH_DATA_DIR` —
/// the only env vars that work on every platform (`dirs::config_dir()`
/// and `dirs::data_dir()` ignore `XDG_*` on macOS/Windows). Without the
/// data-dir override, parallel CLI tests collide on a shared
/// `%APPDATA%/agsh/sessions.db` on Windows and hit SQLite lock contention.
fn run_isolated(dir: &std::path::Path, args: &[&str]) -> std::process::Output {
    agsh()
        .args(args)
        .env("AGSH_CONFIG_DIR", dir.join("agsh"))
        .env("AGSH_DATA_DIR", dir.join("data").join("agsh"))
        .env("XDG_CONFIG_HOME", dir)
        .env("HOME", dir)
        .env("XDG_DATA_HOME", dir.join("data"))
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn agsh {:?}: {}", args, err))
}

#[test]
fn mcp_list_with_empty_config_prints_no_servers_and_exits_zero() {
    // Isolate the config dir so the host's real `~/.config/agsh` doesn't
    // leak into the test. `AGSH_CONFIG_DIR` is the only env var that
    // works on every platform (see `run_isolated` for details).
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(dir.path(), &["mcp", "list"]);
    assert!(
        output.status.success(),
        "agsh mcp list exited non-zero: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("no MCP servers configured"),
        "expected 'no MCP servers configured' in stdout, got: {}",
        stdout
    );
}

#[test]
fn mcp_add_http_positional_url_persists_server() {
    // Notion-style happy path: positional URL, transport auto-detected
    // from the URL scheme, no --url flag required. `--no-login` keeps
    // the test hermetic — we just want to confirm `add` wrote the
    // entry, not that we can drive an end-to-end OAuth flow.
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(
        dir.path(),
        &[
            "mcp",
            "add",
            "notion",
            "https://mcp.notion.com/mcp",
            "--no-login",
        ],
    );
    assert!(
        output.status.success(),
        "agsh mcp add failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let list = run_isolated(dir.path(), &["mcp", "list"]);
    assert!(list.status.success());
    let stdout = String::from_utf8_lossy(&list.stdout);
    assert!(
        stdout.contains("notion") && stdout.contains("https://mcp.notion.com/mcp"),
        "mcp list should show the added server: {}",
        stdout
    );
}

#[test]
fn mcp_add_stdio_positional_command_and_args() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(
        dir.path(),
        &[
            "mcp",
            "add",
            "pg",
            "npx",
            "-y",
            "@modelcontextprotocol/server-postgres",
        ],
    );
    assert!(
        output.status.success(),
        "stdio add should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let get = run_isolated(dir.path(), &["mcp", "get", "pg"]);
    let stdout = String::from_utf8_lossy(&get.stdout);
    assert!(stdout.contains("transport:   stdio"), "{}", stdout);
    assert!(stdout.contains("npx"), "{}", stdout);
    assert!(
        stdout.contains("@modelcontextprotocol/server-postgres"),
        "{}",
        stdout
    );
}

#[test]
fn mcp_disable_sets_disabled_flag() {
    let dir = tempfile::tempdir().expect("tempdir");
    let add = run_isolated(
        dir.path(),
        &["mcp", "add", "flaky", "npx", "-y", "mcp-flaky"],
    );
    assert!(
        add.status.success(),
        "add: {}",
        String::from_utf8_lossy(&add.stderr)
    );

    let disable = run_isolated(dir.path(), &["mcp", "disable", "flaky"]);
    assert!(
        disable.status.success(),
        "disable: {}",
        String::from_utf8_lossy(&disable.stderr)
    );

    let config_path = dir.path().join("agsh").join("config.toml");
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        toml_text.contains("disabled = true"),
        "expected disabled = true in config, got:\n{}",
        toml_text
    );

    let enable = run_isolated(dir.path(), &["mcp", "enable", "flaky"]);
    assert!(
        enable.status.success(),
        "enable: {}",
        String::from_utf8_lossy(&enable.stderr)
    );
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        !toml_text.contains("disabled = true"),
        "disabled flag should be cleared, got:\n{}",
        toml_text
    );
}

#[test]
fn mcp_add_with_disabled_flag_persists() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(
        dir.path(),
        &[
            "mcp",
            "add",
            "staging",
            "https://mcp.example.com/mcp",
            "--no-login",
            "--disabled",
        ],
    );
    assert!(
        output.status.success(),
        "add --disabled: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let config_path = dir.path().join("agsh").join("config.toml");
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        toml_text.contains("disabled = true"),
        "expected disabled = true from --disabled flag, got:\n{}",
        toml_text
    );
}

#[test]
fn mcp_add_http_without_url_fails() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(dir.path(), &["mcp", "add", "broken", "--transport", "http"]);
    assert!(
        !output.status.success(),
        "http without URL must be rejected — stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("http transport needs a URL") || stderr.contains("URL"),
        "error should mention URL: {}",
        stderr
    );
}

#[test]
fn mcp_add_no_login_prints_skip_hint_when_probe_says_auth_required() {
    // Probing the real Notion endpoint classifies as AuthRequired;
    // `--no-login` must surface the "run `agsh mcp login` later" hint
    // rather than entering the OAuth flow. The hint goes to tracing at
    // info level — default filter is `warn`, so we pass `-v` to lift
    // the floor and read the message from stderr.
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(
        dir.path(),
        &[
            "-v",
            "mcp",
            "add",
            "notion",
            "https://mcp.notion.com/mcp",
            "--no-login",
        ],
    );
    assert!(
        output.status.success(),
        "mcp add should succeed even when probe says auth required: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("skipping auto-login"),
        "expected skip hint in stderr, got: {}",
        stderr
    );
    assert!(
        stderr.contains("agsh mcp login notion"),
        "expected follow-up command in stderr, got: {}",
        stderr
    );
}

#[cfg(unix)]
#[test]
fn mcp_add_rollback_on_sigint_during_auto_login() {
    // Reproduces the "user hits Ctrl-C while the OAuth flow is waiting
    // for the browser callback" scenario: start `agsh mcp add` without
    // --no-login against a server that requires auth, wait until the
    // auto-login is clearly in progress, send SIGINT, then confirm
    // nothing remains in config.toml.
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut child = agsh()
        // `-v` so the `running OAuth authorisation` info log is
        // visible — we use it as the "auto-login has started" signal
        // before sending SIGINT.
        .args(["-v", "mcp", "add", "notion", "https://mcp.notion.com/mcp"])
        .env("AGSH_CONFIG_DIR", dir.path().join("agsh"))
        .env("XDG_CONFIG_HOME", dir.path())
        .env("HOME", dir.path())
        .env("XDG_DATA_HOME", dir.path().join("data"))
        // Decouple stdin from the test harness so the paste-mode read
        // doesn't hang waiting on a terminal that isn't there.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn agsh mcp add");

    // Wait until we've seen the "running OAuth authorisation" line so
    // we know the child is past the write + probe and is inside the
    // SIGINT-covered post-persist section. The signpost now lives on
    // stderr (via tracing), not stdout. We drain into `captured` so
    // the subsequent rollback log lines are preserved across the
    // SIGINT for the final assertion.
    let stderr = child.stderr.take().expect("child stderr");
    let mut reader = BufReader::new(stderr);
    let mut captured = String::new();
    let mut saw_running_line = false;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while std::time::Instant::now() < deadline {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                captured.push_str(&line);
                if line.contains("running OAuth authorisation") {
                    saw_running_line = true;
                    break;
                }
            }
            Err(_) => break,
        }
    }
    assert!(
        saw_running_line,
        "child never reached the auto-login stage within 15s; stderr so far:\n{}",
        captured
    );

    // Send SIGINT to the child — same signal a user gets from Ctrl-C.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGINT);
    }

    // Drain the rest of stderr until the child exits so we can assert
    // on the rollback log lines.
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => captured.push_str(&line),
            Err(_) => break,
        }
    }

    let status = child.wait().expect("wait on agsh");
    assert!(
        !status.success(),
        "agsh should exit non-zero after SIGINT during auto-login"
    );
    assert!(
        captured.contains("interrupted") && captured.contains("rolling back"),
        "expected interrupted/rollback message in stderr, got:\n{}",
        captured
    );

    // Verify the entry was rolled out of config.toml.
    let config_path = dir.path().join("agsh").join("config.toml");
    let config_contents = std::fs::read_to_string(&config_path).unwrap_or_default();
    assert!(
        !config_contents.contains("notion"),
        "rolled-back entry must not remain in config.toml; got:\n{}",
        config_contents
    );
}

#[test]
fn mcp_add_tool_filter_and_permission_flags_round_trip() {
    // --allow-tool, --disable-tool, and --tool-permission should land
    // as allowed_tools, disabled_tools, and a [tool_permissions] sub-
    // table on the server entry in config.toml. We also validate one
    // parse error so the flag is actually enforced at add time.
    let dir = tempfile::tempdir().expect("tempdir");

    // Rejection path: missing '=' in --tool-permission.
    let bad = run_isolated(
        dir.path(),
        &[
            "mcp",
            "add",
            "broken",
            "https://mcp.example.com/mcp",
            "--no-login",
            "--tool-permission",
            "just-a-name",
        ],
    );
    assert!(
        !bad.status.success(),
        "bad --tool-permission should reject: {}",
        String::from_utf8_lossy(&bad.stdout)
    );

    // Happy path: all three fields populate correctly.
    let output = run_isolated(
        dir.path(),
        &[
            "mcp",
            "add",
            "notion",
            "https://mcp.notion.com/mcp",
            "--no-login",
            "--allow-tool",
            "notion-search",
            "--allow-tool",
            "notion-fetch",
            "--disable-tool",
            "notion-delete-pages",
            "--tool-permission",
            "notion-create-pages=write",
            "--tool-permission",
            "notion-update-page=write",
        ],
    );
    assert!(
        output.status.success(),
        "mcp add should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let config_path = dir.path().join("agsh").join("config.toml");
    let contents = std::fs::read_to_string(&config_path).expect("read config");
    // Check the allow/block arrays and the nested permissions table.
    assert!(
        contents.contains("allowed_tools"),
        "config missing allowed_tools:\n{}",
        contents
    );
    assert!(
        contents.contains("notion-search") && contents.contains("notion-fetch"),
        "allowed_tools entries missing:\n{}",
        contents
    );
    assert!(
        contents.contains("disabled_tools") && contents.contains("notion-delete-pages"),
        "disabled_tools missing:\n{}",
        contents
    );
    assert!(
        contents.contains("tool_permissions"),
        "config missing [tool_permissions]:\n{}",
        contents
    );
    assert!(
        contents.contains("notion-create-pages")
            && contents.contains("notion-update-page")
            && contents.contains("write"),
        "tool_permissions entries missing:\n{}",
        contents
    );
}

#[test]
fn mcp_add_oauth_writes_auth_block() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = run_isolated(
        dir.path(),
        &[
            "mcp",
            "add",
            "notion",
            "https://mcp.notion.com/mcp",
            "--auth",
            "oauth",
            "--scope",
            "read",
            "--scope",
            "write",
            "--no-login",
        ],
    );
    assert!(
        output.status.success(),
        "oauth add should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Read back the config.toml we wrote.
    let config_path = dir.path().join("agsh").join("config.toml");
    let contents = std::fs::read_to_string(&config_path).expect("read config");
    assert!(contents.contains("type = \"oauth\""), "{}", contents);
    assert!(contents.contains("read"), "{}", contents);
    assert!(contents.contains("write"), "{}", contents);
}
