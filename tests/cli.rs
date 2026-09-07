// Most of these run a turn against the scripted provider, which a release build has only with
// `mock-provider`; without it there is nothing to run.
#![cfg(any(debug_assertions, feature = "mock-provider"))]
// See the matching allow in `tests/acp.rs` for the rationale: integration tests panic on failure
// by design.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "tests panic on failure by design, and indexing a JSON document is the readable form"
)]

//! End-to-end CLI smoke tests. These shell out to the built `meka` binary
//! (`env!("CARGO_BIN_EXE_meka")`) so they exercise the same entry point users hit on the command
//! line. They cover surface-level invariants that unit tests can't reach: argument-parser wiring,
//! `--help` output, and the exit status of trivial subcommands.

use std::process::Command;

#[path = "harness/support.rs"]
mod support;

use support::{Install, meka};

/// What an MCP endpoint that wants OAuth answers to an unauthenticated request: the shape
/// `src/mcp/auth.rs` classifies as `AuthRequired`.
const AUTH_CHALLENGE: &str = "HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer realm=\"mcp\"\r\nContent-Length: \
     0\r\n\r\n";

/// A local stand-in for a server that wants OAuth. The MCP endpoint itself answers the probe with
/// a challenge; every other path (the OAuth discovery documents) is held open, so a login that
/// starts against it parks in discovery instead of failing, which is what lets a test interrupt it.
fn spawn_auth_required_mcp() -> String {
    let port = support::spawn_http_listener(|request| {
        if request.path.starts_with("/mcp") {
            AUTH_CHALLENGE.to_string()
        } else {
            std::thread::sleep(std::time::Duration::from_secs(60));
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_string()
        }
    });
    format!("http://127.0.0.1:{port}/mcp")
}

#[test]
fn version_flag_prints_version_and_exits_zero() {
    let output = meka()
        .arg("--version")
        .output()
        .expect("failed to spawn meka");
    assert!(
        output.status.success(),
        "meka --version exited non-zero: {:?}",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("meka "),
        "expected version output to start with 'meka ', got: {stdout}"
    );
}

#[test]
fn help_flag_lists_subcommands() {
    let output = meka().arg("--help").output().expect("failed to spawn meka");
    assert!(output.status.success(), "meka --help exited non-zero");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in ["account", "profile", "session", "history", "mcp", "acp"] {
        assert!(
            stdout.contains(expected),
            "--help output missing subcommand '{expected}':\n{stdout}"
        );
    }
}

#[test]
fn session_subcommand_help_lists_actions() {
    let output = meka()
        .args(["session", "--help"])
        .output()
        .expect("failed to spawn meka");
    assert!(
        output.status.success(),
        "meka session --help exited non-zero"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in ["list", "export", "delete"] {
        assert!(
            stdout.contains(expected),
            "session --help missing action '{expected}':\n{stdout}"
        );
    }
}

#[test]
fn history_subcommand_help_lists_actions() {
    let output = meka()
        .args(["history", "--help"])
        .output()
        .expect("failed to spawn meka");
    assert!(
        output.status.success(),
        "meka history --help exited non-zero"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in ["list", "clear"] {
        assert!(
            stdout.contains(expected),
            "history --help missing action '{expected}':\n{stdout}"
        );
    }
}

#[test]
fn acp_subcommand_help_describes_protocol() {
    // Verifies the `acp` subcommand is wired up. Full JSON-RPC handshake coverage lives in
    // `tests/acp.rs` against the mock-provider build; this smoke test stops at `--help`.
    let output = meka()
        .args(["acp", "--help"])
        .output()
        .expect("failed to spawn meka acp --help");
    assert!(
        output.status.success(),
        "meka acp --help exited non-zero: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("ACP")
            || stdout.contains("Agent Client Protocol")
            || stdout.contains("stdio"),
        "meka acp --help should mention the protocol or transport:\n{stdout}",
    );
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    let output = meka()
        .arg("--definitely-not-a-flag")
        .output()
        .expect("failed to spawn meka");
    assert!(
        !output.status.success(),
        "meka accepted an unknown flag without erroring"
    );
}

/// Run `meka` against `install` to completion, with the mock provider off. The callers here that
/// reach a provider at all are reading how a real one fails to build (no credential, no profile),
/// which a scripted answer would hide.
fn run_isolated(install: &Install, args: &[&str]) -> std::process::Output {
    install
        .meka(args)
        .env("MEKA_MOCK_PROVIDER", "0")
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn meka {args:?}: {err}"))
}

/// `Conversation::rewind(0)` returns `None` unconditionally, so without an explicit guard the
/// caller reports it as the session having "fewer than 0 turn(s)". Rejected before the session is
/// even looked up, which is why a nonexistent id still produces the argument error. The HTTP
/// surface already answers 422 here (`rewind_rejects_zero_turns` in `tests/serve.rs`); this keeps
/// the CLI in step.
#[test]
fn session_rewind_rejects_zero_turns_without_describing_the_conversation() {
    let install = Install::new();
    let id = "00000000-0000-4000-8000-000000000000";
    let output = run_isolated(&install, &["session", "rewind", id, "-n", "0"]);
    assert!(
        !output.status.success(),
        "-n 0 must fail, got: {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("`-n` must be 1 or more"),
        "expected the argument to be blamed, got: {stderr}"
    );
    assert!(
        !stderr.contains("fewer than 0"),
        "must not describe the conversation as having fewer than 0 turns: {stderr}"
    );
}

/// A full id that names no session is refused by every `meka session` door in the one sentence
/// `MekaError::SessionNotFound` renders, and the refusal reaches the exit code.
#[test]
fn a_session_id_that_names_nothing_is_refused_by_name_on_every_door() {
    let install = Install::new();
    let id = "00000000-0000-4000-8000-000000000000";
    for arguments in [
        vec!["session", "show", id],
        vec!["session", "export", id, "-o", "-"],
        vec!["session", "rewind", id, "-n", "1"],
        vec!["session", "fork", id],
        vec!["session", "delete", id],
    ] {
        let output = run_isolated(&install, &arguments);
        assert!(
            !output.status.success(),
            "{arguments:?} must fail on an id that is not there, got: {:?}",
            output.status
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("session '{id}' not found")),
            "{arguments:?} must name the missing session, got: {stderr}"
        );
    }
}

#[test]
fn mcp_list_with_empty_config_prints_no_servers_and_exits_zero() {
    let install = Install::new();
    let output = run_isolated(&install, &["mcp", "list"]);
    assert!(
        output.status.success(),
        "meka mcp list exited non-zero: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    // The empty case is a status note, not the data a script asked for, so it goes to stderr and
    // stdout stays clean enough to pipe.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No MCP servers."),
        "expected 'No MCP servers.' on stderr, got: {stderr}"
    );
    assert!(
        stdout.trim().is_empty(),
        "stdout must carry no placeholder row, got: {stdout}"
    );
}

/// The `agent_*` family is four rows on stdout like everything else, read from
/// `agent_tool_catalog`: the listing builds a real registry, and those tools carry an `Arc<dyn
/// Provider>` it has no credential for, so without that catalog they would be a sentence on stderr.
/// This is the wiring the unit tests in `src/tools/subagent.rs` cannot see.
#[test]
fn tools_list_puts_the_agent_family_in_the_table() {
    let install = Install::new();
    let output = run_isolated(&install, &["tools", "list"]);
    assert!(
        output.status.success(),
        "meka tools list exited non-zero: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for name in [
        "agent_spawn",
        "agent_list",
        "agent_followup",
        "agent_delete",
    ] {
        let row = stdout
            .lines()
            .find(|line| line.starts_with(&format!("{name} ")))
            .unwrap_or_else(|| panic!("'{name}' must have a row, got:\n{stdout}"));
        assert!(
            row.contains("enabled"),
            "'{name}' must read as enabled by default, got: {row}"
        );
    }
    // Sorted with the rest rather than appended, so the family arrives as one block at the top.
    let names: Vec<&str> = stdout
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted, "the table must stay sorted by name");
}

/// Denying `agent_spawn` takes the three lifecycle tools with it, so all four have to read as
/// disabled. Listing `agent_list` as enabled here would describe a session nobody can have.
#[test]
fn tools_list_reports_the_whole_agent_family_as_denied_with_agent_spawn() {
    let install = Install::new();
    let config_dir = install.config_dir();
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        "[tools]\ndisabled_tools = [\"agent_spawn\"]\n",
    )
    .expect("write config.toml");
    let output = run_isolated(&install, &["tools", "list"]);
    assert!(output.status.success(), "{:?}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    for name in [
        "agent_spawn",
        "agent_list",
        "agent_followup",
        "agent_delete",
    ] {
        let row = stdout
            .lines()
            .find(|line| line.starts_with(&format!("{name} ")))
            .unwrap_or_else(|| panic!("'{name}' must have a row, got:\n{stdout}"));
        assert!(
            row.contains("disabled"),
            "'{name}' goes with agent_spawn, got: {row}"
        );
    }
}

/// `session.subagent_max_depth = 0` is the documented way to turn delegation off, and the listing
/// missed it entirely while the family was described in prose.
#[test]
fn tools_list_reports_the_agent_family_as_denied_at_depth_zero() {
    let install = Install::new();
    let config_dir = install.config_dir();
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        "[session]\nsubagent_max_depth = 0\n",
    )
    .expect("write config.toml");
    let output = run_isolated(&install, &["tools", "list"]);
    assert!(output.status.success(), "{:?}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    for name in [
        "agent_spawn",
        "agent_list",
        "agent_followup",
        "agent_delete",
    ] {
        let row = stdout
            .lines()
            .find(|line| line.starts_with(&format!("{name} ")))
            .unwrap_or_else(|| panic!("'{name}' must have a row, got:\n{stdout}"));
        assert!(
            row.contains("disabled"),
            "'{name}' needs a depth budget, got: {row}"
        );
    }
}

#[test]
fn mcp_add_http_positional_url_persists_server() {
    // Notion-style happy path: positional URL, transport auto-detected from the URL scheme, no
    // --url flag required. `--no-login` keeps the test hermetic; we just want to confirm `add`
    // wrote the entry, not that we can drive an end-to-end OAuth flow.
    let install = Install::new();
    let output = run_isolated(&install, &[
        "mcp",
        "add",
        "notion",
        "https://mcp.notion.com/mcp",
        "--no-login",
    ]);
    assert!(
        output.status.success(),
        "meka mcp add failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let list = run_isolated(&install, &["mcp", "list"]);
    assert!(list.status.success());
    let stdout = String::from_utf8_lossy(&list.stdout);
    assert!(
        stdout.contains("notion") && stdout.contains("https://mcp.notion.com/mcp"),
        "mcp list should show the added server: {stdout}"
    );
}

#[test]
fn mcp_add_stdio_positional_command_and_args() {
    let install = Install::new();
    let output = run_isolated(&install, &[
        "mcp",
        "add",
        "pg",
        "npx",
        "-y",
        "@modelcontextprotocol/server-postgres",
    ]);
    assert!(
        output.status.success(),
        "stdio add should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let get = run_isolated(&install, &["mcp", "get", "pg"]);
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
    let install = Install::new();
    let add = run_isolated(&install, &["mcp", "add", "flaky", "npx", "-y", "mcp-flaky"]);
    assert!(
        add.status.success(),
        "add: {}",
        String::from_utf8_lossy(&add.stderr)
    );

    let disable = run_isolated(&install, &["mcp", "disable", "flaky"]);
    assert!(
        disable.status.success(),
        "disable: {}",
        String::from_utf8_lossy(&disable.stderr)
    );

    let config_path = install.config_dir().join("config.toml");
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        toml_text.contains("disabled = true"),
        "expected disabled = true in config, got:\n{toml_text}"
    );

    let enable = run_isolated(&install, &["mcp", "enable", "flaky"]);
    assert!(
        enable.status.success(),
        "enable: {}",
        String::from_utf8_lossy(&enable.stderr)
    );
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        !toml_text.contains("disabled = true"),
        "disabled flag should be cleared, got:\n{toml_text}"
    );
}

#[test]
fn mcp_add_with_disabled_flag_persists() {
    let install = Install::new();
    let output = run_isolated(&install, &[
        "mcp",
        "add",
        "staging",
        "https://mcp.example.com/mcp",
        "--no-login",
        "--disabled",
    ]);
    assert!(
        output.status.success(),
        "add --disabled: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let config_path = install.config_dir().join("config.toml");
    let toml_text = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        toml_text.contains("disabled = true"),
        "expected disabled = true from --disabled flag, got:\n{toml_text}"
    );
}

#[test]
fn mcp_add_http_without_url_fails() {
    let install = Install::new();
    let output = run_isolated(&install, &["mcp", "add", "broken", "--transport", "http"]);
    assert!(
        !output.status.success(),
        "http without URL must be rejected, stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("http transport needs a URL") || stderr.contains("URL"),
        "error should mention URL: {stderr}"
    );
}

#[test]
fn mcp_add_no_login_prints_skip_hint_when_probe_says_auth_required() {
    // The probe classifies a 401 with a bearer challenge as AuthRequired; `--no-login` must surface
    // the "run `meka mcp login` later" hint rather than entering the OAuth flow. The hint goes to
    // tracing at info level; default filter is `warn`, so we pass `-v` to lift the floor and read
    // the message from stderr. A local listener rather than a real endpoint, so the suite passes
    // with the network unplugged.
    let install = Install::new();
    let url = spawn_auth_required_mcp();
    let output = run_isolated(&install, &[
        "-v",
        "mcp",
        "add",
        "notion",
        &url,
        "--no-login",
    ]);
    assert!(
        output.status.success(),
        "mcp add should succeed even when probe says auth required: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("skipping auto-login"),
        "expected skip hint in stderr, got: {stderr}"
    );
    assert!(
        stderr.contains("meka mcp login notion"),
        "expected follow-up command in stderr, got: {stderr}"
    );
}

#[cfg(unix)]
#[test]
fn mcp_add_rollback_on_sigint_during_auto_login() {
    // Reproduces the "user hits Ctrl-C while the OAuth flow is waiting for the browser callback"
    // scenario: start `meka mcp add` without --no-login against a server that requires auth, wait
    // until the auto-login is clearly in progress, send SIGINT, then confirm nothing remains in
    // config.toml.
    use std::{
        io::{BufRead, BufReader},
        process::Stdio,
    };

    let install = Install::new();
    let url = spawn_auth_required_mcp();
    // `-v` so the `running OAuth authorization` info log is visible; we use it as the
    // "auto-login has started" signal before sending SIGINT.
    let mut child = install
        .meka(&["-v", "mcp", "add", "notion", &url])
        // Decouple stdin from the test harness so the paste-mode read doesn't hang waiting on a
        // terminal that isn't there.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn meka mcp add");

    // Wait until we've seen the "running OAuth authorization" line so we know the child is past the
    // write + probe and is inside the SIGINT-covered post-persist section. The signpost now lives
    // on stderr (via tracing), not stdout. We drain into `captured` so the subsequent rollback log
    // lines are preserved across the SIGINT for the final assertion.
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
                if line.contains("running OAuth authorization") {
                    saw_running_line = true;
                    break;
                }
            }
            Err(_) => break,
        }
    }
    assert!(
        saw_running_line,
        "child never reached the auto-login stage within 15s; stderr so far:\n{captured}"
    );

    // Send SIGINT to the child, same signal a user gets from Ctrl-C.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGINT);
    }

    // Drain the rest of stderr until the child exits so we can assert on the rollback log lines.
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => captured.push_str(&line),
            Err(_) => break,
        }
    }

    let status = child.wait().expect("wait on meka");
    assert!(
        !status.success(),
        "meka should exit non-zero after SIGINT during auto-login"
    );
    assert!(
        captured.contains("interrupted") && captured.contains("rolling back"),
        "expected interrupted/rollback message in stderr, got:\n{captured}"
    );

    // Verify the entry was rolled out of config.toml.
    let config_path = install.config_dir().join("config.toml");
    let config_contents = std::fs::read_to_string(&config_path).unwrap_or_default();
    assert!(
        !config_contents.contains("notion"),
        "rolled-back entry must not remain in config.toml; got:\n{config_contents}"
    );
}

#[test]
fn mcp_add_tool_filter_and_permission_flags_round_trip() {
    // --allow-tool, --disable-tool, and --tool-permission should land as allowed_tools,
    // disabled_tools, and a [tool_permissions] sub- table on the server entry in config.toml. We
    // also validate one parse error so the flag is actually enforced at add time.
    let install = Install::new();

    // Rejection path: missing '=' in --tool-permission.
    let bad = run_isolated(&install, &[
        "mcp",
        "add",
        "broken",
        "https://mcp.example.com/mcp",
        "--no-login",
        "--tool-permission",
        "just-a-name",
    ]);
    assert!(
        !bad.status.success(),
        "bad --tool-permission should reject: {}",
        String::from_utf8_lossy(&bad.stdout)
    );

    // Happy path: all three fields populate correctly.
    let output = run_isolated(&install, &[
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
        "notion-create-pages=unrestricted",
        "--tool-permission",
        "notion-update-page=unrestricted",
        "--eager-load-tool",
        "notion-search",
    ]);
    assert!(
        output.status.success(),
        "mcp add should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let config_path = install.config_dir().join("config.toml");
    let contents = std::fs::read_to_string(&config_path).expect("read config");
    // Check the allow/block arrays and the nested permissions table.
    assert!(
        contents.contains("allowed_tools"),
        "config missing allowed_tools:\n{contents}"
    );
    assert!(
        contents.contains("notion-search") && contents.contains("notion-fetch"),
        "allowed_tools entries missing:\n{contents}"
    );
    assert!(
        contents.contains("disabled_tools") && contents.contains("notion-delete-pages"),
        "disabled_tools missing:\n{contents}"
    );
    // The one flag of the four that reached the parser and not the file: accepted, reported as a
    // success, and dropped, so every session paid the `load_tool` round trip it was meant to skip.
    assert!(
        contents.contains("eager_load_tools"),
        "eager_load_tools missing:\n{contents}"
    );
    assert!(
        contents.contains("tool_permissions"),
        "config missing [tool_permissions]:\n{contents}"
    );
    assert!(
        contents.contains("notion-create-pages")
            && contents.contains("notion-update-page")
            && contents.contains("unrestricted"),
        "tool_permissions entries missing:\n{contents}"
    );
}

#[test]
fn mcp_add_oauth_writes_auth_block() {
    let install = Install::new();
    let output = run_isolated(&install, &[
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
    ]);
    assert!(
        output.status.success(),
        "oauth add should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Read back the config.toml we wrote.
    let config_path = install.config_dir().join("config.toml");
    let contents = std::fs::read_to_string(&config_path).expect("read config");
    assert!(contents.contains("type = \"oauth\""), "{}", contents);
    assert!(contents.contains("read"), "{}", contents);
    assert!(contents.contains("write"), "{}", contents);
}

/// `meka skill remove` must wait on the store lock, like every other skill door.
///
/// Going around `delete_skill` with its own `remove_dir_all` would complete in 70 ms against a lock
/// every other door waits on, able to delete a skill directory while a `skill_write` or `PUT
/// /v1/skills` is composing and renaming `SKILL.md` inside it.
#[test]
fn skill_remove_waits_for_a_store_lock_another_process_holds() {
    let install = Install::new();
    let skills = install.config_dir().join("skills");
    std::fs::create_dir_all(skills.join("victim")).expect("skill dir");
    std::fs::write(
        skills.join("victim").join("SKILL.md"),
        "---\nname: victim\ndescription: a skill\n---\nbody\n",
    )
    .expect("seed");

    // Hold the lock the way another meka would: the store root itself, or on Windows the sidecar
    // that `LockFileEx` needs because it refuses a directory handle.
    #[cfg(unix)]
    let lock_file = std::fs::File::open(&skills).expect("open the store root");
    #[cfg(windows)]
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(skills.join(".meka-store.lock"))
        .expect("open the store lock");
    lock_file.lock().expect("hold the store lock");

    let mut blocked = install
        .meka(&["skill", "remove", "victim"])
        .spawn()
        .expect("spawn meka skill remove");
    std::thread::sleep(std::time::Duration::from_millis(750));
    assert!(
        blocked.try_wait().expect("try_wait").is_none(),
        "the delete must wait for the lock this test is holding"
    );
    assert!(
        skills.join("victim").join("SKILL.md").exists(),
        "and must not have removed anything yet"
    );

    drop(lock_file);
    assert!(blocked.wait().expect("wait").success(), "then it completes");
    assert!(!skills.join("victim").exists(), "and the skill is gone");
}

/// Run `meka` isolated, with a config file, and a prompt that will fail on the missing provider.
///
/// The prompt is what forces full config resolution: `session list` and friends short-circuit
/// before `ResolvedConfig::resolve` runs, so a flag that only affects the resolved config is
/// unobservable through them. Failing on "no provider profiles configured" is the expected end of
/// every call here; what the tests read is what was warned on the way there.
fn resolve_config(install: &Install, config: &str, args: &[&str]) -> std::process::Output {
    if !config.is_empty() {
        install.write_config(config);
    }
    // The mock would answer the prompt; the failure to build a real provider is where these runs
    // are meant to end.
    install
        .meka(args)
        .args(["-p", "hi"])
        .env("MEKA_MOCK_PROVIDER", "0")
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn meka {args:?}: {err}"))
}

/// A level meka does not have fails at every door it can be spelled at, and never resolves quietly.
///
/// `Permission` is read as a grant *and* as a requirement, so a surface that quietly mapped an
/// unknown string onto some level would admit tools at authority nobody chose. The value of
/// refusing depends entirely on every surface doing it, rather than one of them keeping a private
/// table.
#[test]
fn a_level_meka_does_not_have_is_refused_at_the_flag_and_in_the_config_file() {
    let install = Install::new();

    let flag = run_isolated(&install, &["--permission", "elevated", "session", "list"]);
    assert!(!flag.status.success(), "an unknown level must not start");
    let stderr = String::from_utf8_lossy(&flag.stderr);
    assert!(
        stderr.contains("workspace") && stderr.contains("unrestricted"),
        "the refusal has to list the levels meka does have: {stderr}"
    );

    // The file is refused where it is parsed, the way an unknown key is, rather than warned about
    // and run at a level the user did not write.
    let file = resolve_config(
        &install,
        "[permissions]\ndefault = \"elevated\"\nenabled = [\"read\", \"elevated\"]\n",
        &[],
    );
    assert!(
        !file.status.success(),
        "a config naming a level meka does not have must not start"
    );
    let stderr = String::from_utf8_lossy(&file.stderr);
    assert!(
        stderr.contains("elevated") && stderr.contains("unrestricted"),
        "the refusal names the value and the levels meka does have: {stderr}"
    );
}

/// `workspace` is spellable everywhere the other levels are, and reaches config resolution.
#[test]
fn the_workspace_level_is_accepted_at_the_flag_and_in_the_config_file() {
    let install = Install::new();

    let flag = resolve_config(&install, "", &["--permission", "workspace"]);
    let stderr = String::from_utf8_lossy(&flag.stderr);
    assert!(
        stderr.contains("no profile configured"),
        "resolution must get past the permission flag to the profile: {stderr}"
    );
    assert!(
        !stderr.contains("invalid value"),
        "`workspace` must not be rejected by the parser: {stderr}"
    );

    let file = resolve_config(
        &install,
        "[permissions]\ndefault = \"workspace\"\nenabled = [\"read\", \"workspace\"]\n",
        &[],
    );
    let stderr = String::from_utf8_lossy(&file.stderr);
    assert!(
        !stderr.contains("ignoring invalid"),
        "neither [permissions] key may treat `workspace` as unknown: {stderr}"
    );
}

/// `--writable-root` naming a path that does not exist warns, and does not fail the run.
///
/// Both halves matter. A root that cannot be canonicalized is dropped from the boundary by
/// `writable_roots`, so without the warning the user learns about it from a refused write naming a
/// boundary they believed included the path. And a build directory that does not exist *yet* is a
/// legitimate root, so this cannot be an error: the boundary is recomputed on every write.
#[test]
fn an_unresolvable_writable_root_warns_without_failing_the_run() {
    let install = Install::new();
    let missing = install.root().join("not-created-yet");
    let output = resolve_config(&install, "", &[
        "--writable-root",
        missing.to_str().expect("path"),
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--writable-root") && stderr.contains("does not resolve"),
        "an unresolvable root must say so: {stderr}"
    );
    assert!(
        stderr.contains("no profile configured"),
        "and must not be what stops the run: {stderr}"
    );

    // The existing case stays quiet, so the warning means something when it appears.
    std::fs::create_dir_all(&missing).expect("create the root");
    let output = resolve_config(&install, "", &[
        "--writable-root",
        missing.to_str().expect("path"),
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("does not resolve"),
        "a root that exists must not warn: {stderr}"
    );
}

/// `--continue` and `--resume` name *this run's session*, and neither long-lived host has one:
/// each creates a session per `session/new` or `POST /v1/sessions`. Parsing and doing nothing would
/// be worse than it sounds: `-c` / `-r` set `session_resume`, which switches off the
/// default-profile check a host with no configured default needs most, so `meka -c acp` would write
/// a session row naming the empty profile and fail its first turn complaining about a session it
/// had created moments earlier.
#[test]
fn the_long_lived_hosts_refuse_the_flags_that_name_one_session() {
    let install = Install::new();
    for host in ["acp", "serve"] {
        for flag in [vec!["--continue"], vec!["--resume", "0e5f"]] {
            // Isolated, like every other CLI test. A regression in the guard would otherwise reach
            // the host's real startup, and `meka serve` would bind the port in the *developer's*
            // `config.toml` and run until the harness gave up -- a hang rather than a failure.
            let mut args = flag.clone();
            args.push(host);
            let output = run_isolated(&install, &args);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success(),
                "meka {} {host} should be refused, got: {stderr}",
                flag.join(" ")
            );
            assert!(
                stderr.contains(flag[0]) && stderr.contains("does not take"),
                "meka {} {host} must say why: {stderr}",
                flag.join(" ")
            );
        }
    }
}

/// A `config.toml` meka cannot parse must not stop the commands that exist to repair one.
///
/// `meka mcp remove` and `meka profile remove` edit the raw document through `toml_edit` and never
/// parse it into a `ConfigFile`, which is exactly what makes them the way out of a config an
/// unknown key or a bad value has made unloadable. Gating the whole subcommand path on a readable
/// config would close that door: the rule that the ledger must not adopt a profile it inferred from
/// a parse error belongs at the ledger, not one level higher where every subcommand would refuse.
///
/// The ledger's own protection is asserted where it lives, in
/// `store::migrations::tests::an_unreadable_config_refuses_to_stamp_carried_sessions_but_not_an_empty_store`:
/// a store with sessions to stamp is refused and left at its old version, and one with nothing to
/// stamp opens normally. That split is what lets both properties hold at once.
#[test]
fn an_unparseable_config_still_lets_the_commands_that_repair_it_run() {
    let install = Install::new();
    let config_dir = install.config_dir();
    std::fs::create_dir_all(&config_dir).expect("config dir");
    // Valid TOML that `serde` rejects, which is the shape the repair path is for:
    // `deny_unknown_fields` refuses the whole file over one stray key, while `toml_edit` still
    // parses it, so the document can be edited even though the config cannot be loaded. A
    // *syntax* error defeats `toml_edit` too and has never been repairable from the CLI; that
    // is not what this guards.
    let config = "default_profile = \"work\"\n\n[accounts.work]\nbackend = \
                  \"anthropic-messages\"\n\n[profiles.work]\naccount = \"work\"\nmodel = \
                  \"some-model\"\nstray_unknown_key = 1\n";
    std::fs::write(config_dir.join("config.toml"), config).expect("write config.toml");

    let output = run_isolated(&install, &["profile", "remove", "work"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "`meka profile remove` must run on the very config it exists to repair: {stderr}"
    );
    let after = std::fs::read_to_string(config_dir.join("config.toml")).expect("read config back");
    assert!(
        !after.contains("[profiles.work]"),
        "the profile was not actually removed, so the repair did not happen:\n{after}"
    );

    // The readers are the other half of the split and must keep refusing: answering "No MCP servers
    // configured" out of a file meka could not read would state something false.
    // `run_mcp_subcommand` branches on exactly this, and the two halves are what let a broken
    // config be both survivable and repairable.
    //
    // A second directory, because the repair above has by now *fixed* the first one: removing the
    // profile took the stray key with it.
    let unrepaired = Install::new();
    unrepaired.write_config(config);
    let output = run_isolated(&unrepaired, &["mcp", "list"]);
    assert!(
        !output.status.success(),
        "`meka mcp list` must not answer out of a config it could not read"
    );
}

/// `--profile` is deliberately *not* refused above: it selects which configured profile the host
/// defaults to, which is a property of the host rather than of one session. A guard that lumped it
/// in with the four would take a real capability away.
#[test]
fn a_long_lived_host_still_takes_provider() {
    let install = Install::new();
    // One configured profile, so the refusal is "no profile named X" rather than "no profiles
    // configured". Seeded rather than inherited: run un-isolated, this test read the developer's
    // own `config.toml` and passed only because it happened to have a profile in it.
    let config_dir = install.config_dir();
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        "default_profile = \"work\"\n\n[accounts.work]\nbackend = \"anthropic-messages\"\n\n\
         [profiles.work]\naccount = \"work\"\nmodel = \"m\"\n",
    )
    .expect("write config");

    let output = run_isolated(&install, &[
        "--profile",
        "definitely-not-configured",
        "serve",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no profile named"),
        "--profile must reach profile selection rather than the flag guard: {stderr}"
    );
}

/// Write a `config.toml` with `profiles` configured and `default_profile` naming `default`.
///
/// Every endpoint is port 9, which discards, so a turn that got as far as the network could not
/// reach anything. The tests below never get that far: they run with the scripted provider.
fn write_provider_config(install: &Install, default: &str, profiles: &[&str]) {
    let mut config = format!(
        "default_profile = \"{default}\"\n\n[permissions]\ndefault = \"read\"\nenabled = \
         [\"read\"]\n"
    );
    // One account per profile, of the same name, so a test can reason about either by one word.
    for profile in profiles {
        config.push_str(&format!(
            "\n[accounts.{profile}]\nbackend = \"openai-chat-completions\"\nbase_url = \
             \"http://127.0.0.1:9/\"\n\n[profiles.{profile}]\naccount = \"{profile}\"\nmodel = \
             \"{profile}-model\"\n"
        ));
    }
    install.write_config(&config);
}

/// Run one `meka` turn against the scripted mock provider, which is how these tests get a session
/// row without a credential or a network.
///
/// The mock is compiled into debug builds only (`MEKA_MOCK_PROVIDER=1`), which is what `cargo test`
/// builds; `tests/multiprocess.rs` rests on the same thing.
fn run_scripted(install: &Install, args: &[&str]) -> std::process::Output {
    scripted_command(install, args)
        .output()
        .unwrap_or_else(|error| panic!("failed to spawn meka {args:?}: {error}"))
}

/// The command [`run_scripted`] runs, not yet spawned, for a test that has to shape its stdin.
fn scripted_command(install: &Install, args: &[&str]) -> Command {
    install.write_script(
        r#"[[{"type":"text","text":"ok"},{"type":"message_end","stop_reason":"end_turn"}]]"#,
    );
    install.meka(args)
}

/// `-p -` is the prompt on stdin, whole, with the newline a shell appends trimmed off.
#[test]
fn the_prompt_can_be_read_from_stdin() {
    let install = Install::new();
    write_provider_config(&install, "mock", &["mock"]);
    let mut child = scripted_command(&install, &["-p", "-", "--oneshot"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn meka");
    {
        use std::io::Write;
        let mut stdin = child.stdin.take().expect("piped stdin");
        stdin.write_all(b"from a pipe\n").expect("write the prompt");
    }
    let output = child.wait_with_output().expect("wait for meka");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let words: String = store(&install)
        .query_row(
            "SELECT content FROM messages WHERE role = 'user_blocks' ORDER BY id LIMIT 1",
            [],
            |row| row.get(0),
        )
        .expect("the turn's user row");
    assert!(
        words.contains("\"from a pipe\""),
        "the words as typed, without the trailing newline: {words}"
    );
}

/// An empty stdin is a mistake, not a turn: nothing is sent and no session is made.
#[test]
fn an_empty_stdin_prompt_is_refused() {
    let install = Install::new();
    write_provider_config(&install, "mock", &["mock"]);
    let output = scripted_command(&install, &["-p", "-", "--oneshot"])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn meka");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("read nothing from stdin"), "{stderr}");
    assert!(
        !install
            .root()
            .join("data")
            .join("meka")
            .join("meka.db")
            .exists(),
        "nothing was sent, so no store was opened"
    );
}

/// `--format` shapes a one-shot run's stdout; without `--oneshot` it is refused, not ignored.
#[test]
fn format_without_oneshot_is_refused() {
    let install = Install::new();
    write_provider_config(&install, "mock", &["mock"]);
    let output = run_scripted(&install, &["--format", "json", "-p", "hi"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("`--format` needs `--oneshot`"), "{stderr}");
}

/// Ctrl+C ends a one-shot run with 130, as it ends every other host, not with 0 over a partial
/// answer, and what streamed before the press is kept. The press waits for a notice the script
/// raises right after its first text, which the console prints to stderr as it arrives; the
/// session row is no signal, since it lands as the turn begins, before the first delta, and a
/// press in that gap leaves nothing to keep.
#[cfg(unix)]
#[test]
fn an_interrupted_oneshot_run_exits_130_and_keeps_what_streamed() {
    use std::io::Read;
    let install = Install::new();
    write_provider_config(&install, "mock", &["mock"]);
    write_slow_script(&install);
    let mut child = install
        .meka(&["--oneshot", "-p", "slow"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn meka");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let mut chrome = Vec::new();
    let mut buffer = [0u8; 4096];
    // A closed pipe here means the run ended on its own, which the status assertion reports.
    while !String::from_utf8_lossy(&chrome).contains("text-has-streamed") {
        let read = stderr.read(&mut buffer).expect("read stderr");
        if read == 0 {
            break;
        }
        chrome.extend_from_slice(&buffer[..read]);
    }
    // SAFETY: `child.id()` is a live process this test spawned and still owns.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGINT);
    }
    stderr.read_to_end(&mut chrome).expect("drain stderr");
    let status = child.wait().expect("wait for meka");
    let mut answer = String::new();
    child
        .stdout
        .take()
        .expect("piped stdout")
        .read_to_string(&mut answer)
        .expect("read stdout");
    let chrome = String::from_utf8_lossy(&chrome);
    assert_eq!(
        status.code(),
        Some(130),
        "an interrupted run exits like an interrupted REPL: {chrome}"
    );
    assert!(chrome.contains("(interrupted)"), "{chrome}");
    assert!(
        answer.contains("starting") && !answer.contains("never reached"),
        "what streamed before the press is kept, and nothing after it: {answer:?}"
    );
}

/// Under `--format json` an interrupted run still prints its one report, and it says
/// `interrupted`. Whether the report carries partial text depends on when the press landed, so
/// that claim lives in the raw-mode twin above, where the press waits for the text.
#[cfg(unix)]
#[test]
fn an_interrupted_json_oneshot_run_still_reports() {
    let install = Install::new();
    write_provider_config(&install, "mock", &["mock"]);
    write_slow_script(&install);
    let child = install
        .meka(&["--oneshot", "--format", "json", "-p", "slow"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn meka");
    // The row is written as the turn begins, after the interrupt handler is installed, so it is
    // the signal that a SIGINT now lands on a live turn rather than on a process still starting.
    support::wait_until(
        "the session row",
        std::time::Duration::from_secs(20),
        || {
            install.database().exists()
                && store(&install)
                    .query_row("SELECT count(*) FROM sessions", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .is_ok_and(|rows| rows > 0)
        },
    );
    // SAFETY: `child.id()` is a live process this test spawned and still owns.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGINT);
    }
    let output = child.wait_with_output().expect("wait for meka");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(130), "{stderr}");
    assert!(stderr.contains("(interrupted)"), "{stderr}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|error| panic!("one object on stdout ({error}): {stdout:?}"));
    assert_eq!(report["stop_reason"], "interrupted");
}

/// A script whose answer starts at once, says so on stderr, and then holds for longer than any
/// test waits.
#[cfg(unix)]
fn write_slow_script(install: &Install) {
    install.write_script(
        r#"[[{"type":"text","text":"starting"},
            {"type":"notice","message":"text-has-streamed"},
            {"type":"sleep","ms":30000},
            {"type":"text","text":"never reached"},
            {"type":"message_end","stop_reason":"end_turn"}]]"#,
    );
}

/// `[display].show_session_id_on_create` holds under `--format json`: the id line goes to stderr,
/// and stdout is still the one object.
#[test]
fn a_json_oneshot_run_prints_the_new_session_id_on_stderr_when_asked() {
    let install = Install::new();
    write_provider_config(&install, "mock", &["mock"]);
    let config_path = install.config_dir().join("config.toml");
    let mut config = std::fs::read_to_string(&config_path).expect("read config.toml");
    config.push_str(
        "\n[display]\nshow_session_id_on_create = true\nshow_session_id_on_exit = false\n",
    );
    install.write_config(&config);
    let output = run_scripted(&install, &["--oneshot", "--format", "json", "-p", "hi"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    let id = only_session(&install);
    assert!(
        stderr.contains("Creating new session") && stderr.contains(&id),
        "the id line is printed as the REPL prints it: {stderr:?}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.lines().count(),
        1,
        "stdout is the report alone: {stdout:?}"
    );
    let report: serde_json::Value = serde_json::from_str(stdout.trim()).expect("one object");
    assert_eq!(report["session_id"], id);
}

/// The store an isolated run left behind, read directly: what is being set up is a row shape no
/// command produces on purpose.
fn store(install: &Install) -> rusqlite::Connection {
    rusqlite::Connection::open(install.database()).expect("open the store")
}

/// The id of the one session a test made.
fn only_session(install: &Install) -> String {
    store(install)
        .query_row("SELECT id FROM sessions", [], |row| row.get::<_, String>(0))
        .expect("exactly one session")
}

/// The working directory the one session recorded.
fn only_session_cwd(install: &Install) -> Option<String> {
    store(install)
        .query_row("SELECT cwd FROM sessions", [], |row| {
            row.get::<_, Option<String>>(0)
        })
        .expect("exactly one session")
}

/// [`run_scripted`] with the process started somewhere specific and running a script the caller
/// wrote, which is what the working-directory tests below need: the whole question is what the
/// session does when the shell is *not* where the session is.
fn run_scripted_from(
    install: &Install,
    working_directory: &std::path::Path,
    script_json: &str,
    args: &[&str],
) -> std::process::Output {
    install.write_script(script_json);
    install
        .meka(args)
        .current_dir(working_directory)
        .output()
        .unwrap_or_else(|error| panic!("failed to spawn meka {args:?}: {error}"))
}

/// A scripted turn that writes `marker.txt` with a *relative* path, so where the file lands is
/// where the session's working directory actually was. More direct than reading the rendering:
/// `write_file` resolves against the same `SharedCwd` every other tool does.
const WRITE_A_MARKER: &str = r#"[
  [{"type":"tool_use_start","id":"call-1","name":"write_file"},
   {"type":"tool_use_end","input":{"path":"marker.txt","content":"here"}},
   {"type":"message_end","stop_reason":"tool_use"}],
  [{"type":"text","text":"done"},{"type":"message_end","stop_reason":"end_turn"}]
]"#;

/// [`WRITE_A_MARKER`] with a sentence ahead of the call, for the report's `text`: what the model
/// said before calling the tool and after must read as two paragraphs, not one run-on sentence.
const NARRATED_MARKER: &str = r#"[
  [{"type":"text","text":"Writing the marker."},
   {"type":"tool_use_start","id":"call-1","name":"write_file"},
   {"type":"tool_use_end","input":{"path":"marker.txt","content":"here"}},
   {"type":"message_end","stop_reason":"tool_use"}],
  [{"type":"text","text":"done"},{"type":"message_end","stop_reason":"end_turn"}]
]"#;

/// `--format json` prints the turn as one object and nothing else on stdout: the words, every call
/// with its input and outcome, the stop reason, the usage, and which session and profile ran it.
#[test]
fn a_json_one_shot_prints_one_object_and_nothing_else() {
    let install = Install::new();
    write_capable_config(&install);
    let work = install.root().join("work");
    std::fs::create_dir_all(&work).expect("work dir");
    let output = run_scripted_from(&install, &work, NARRATED_MARKER, &[
        "--oneshot",
        "-p",
        "write the marker",
        "--format",
        "json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|error| {
        panic!("stdout must be exactly one JSON object ({error}): {stdout:?}")
    });
    assert!(stdout.ends_with('\n'), "newline-terminated: {stdout:?}");
    assert_eq!(report["session_id"], only_session(&install));
    assert_eq!(report["profile"], "mock");
    assert_eq!(report["stop_reason"], "end_turn");
    assert_eq!(report["text"], "Writing the marker.\n\ndone");
    assert_eq!(
        report["tool_calls"],
        serde_json::json!([{
            "name": "write_file",
            "input": {"path": "marker.txt", "content": "here"},
            "is_error": false,
        }])
    );
    for key in [
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ] {
        assert!(report["usage"][key].is_number(), "usage.{key}: {report}");
    }
    assert!(work.join("marker.txt").exists(), "the call ran");
}

/// A one-shot run has nobody to answer an approval prompt, so a gated call is denied and the run
/// says which tool was refused: a warning on stderr on the plain path, a `notices` entry under
/// `--format json`. Without either, a run whose every gated call was refused reads as a model that
/// chose not to use its tools.
#[test]
fn a_one_shot_with_approvals_on_refuses_each_gated_tool_and_says_so() {
    for json in [false, true] {
        let install = Install::new();
        let config_dir = install.config_dir();
        std::fs::create_dir_all(&config_dir).expect("config dir");
        std::fs::write(
            config_dir.join("config.toml"),
            "default_profile = \"mock\"\n\n[accounts.mock]\nbackend = \"anthropic-messages\"\n\n\
             [profiles.mock]\naccount = \"mock\"\nmodel = \"claude-sonnet-4-5\"\n\n[permissions]\n\
             default = \"read\"\napprovals = true\nenabled = [\"read\"]\n",
        )
        .expect("write config.toml");
        let work = install.root().join("work");
        std::fs::create_dir_all(&work).expect("work dir");
        let mut args = vec!["--oneshot", "-p", "write the marker"];
        if json {
            args.extend(["--format", "json"]);
        }
        let output = run_scripted_from(&install, &work, WRITE_A_MARKER, &args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stderr}");
        assert!(
            !work.join("marker.txt").exists(),
            "the refused call must not have run"
        );
        if json {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let report: serde_json::Value = serde_json::from_str(stdout.trim())
                .unwrap_or_else(|error| panic!("one JSON object expected ({error}): {stdout:?}"));
            let notices = report["notices"].as_array().expect("notices array");
            assert!(
                notices.iter().any(|notice| {
                    notice["level"] == "warn"
                        && notice["text"].as_str().is_some_and(|text| {
                            text.contains("'write_file'") && text.contains("refused without asking")
                        })
                }),
                "the refusal names the tool in the report: {report}"
            );
            assert_eq!(report["tool_calls"][0]["is_error"], true, "{report}");
        } else {
            assert!(
                stderr.contains("'write_file'") && stderr.contains("refused without asking"),
                "the refusal names the tool on stderr: {stderr}"
            );
        }
    }
}

/// A provider advisory is part of what the turn produced, so the JSON report carries it in the
/// shape the HTTP API's turn response uses; a run that drops it reads as a turn nobody warned.
#[test]
fn a_json_one_shot_reports_the_turn_s_notices() {
    const NOTICED: &str = r#"[
  [{"type":"notice","message":"upstream trimmed the context"},
   {"type":"text","text":"ok"},
   {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = Install::new();
    write_capable_config(&install);
    let output = run_scripted_from(&install, install.root(), NOTICED, &[
        "--oneshot",
        "-p",
        "hi",
        "--format",
        "json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|error| panic!("one JSON object expected ({error}): {stdout:?}"));
    assert_eq!(
        report["notices"],
        serde_json::json!([{"level": "info", "text": "upstream trimmed the context"}]),
        "{report}"
    );
}

/// A config that lets `write_file` run, since the marker above is the whole measurement.
fn write_capable_config(install: &Install) {
    install.write_config(
        "default_profile = \"mock\"\n\n[accounts.mock]\nbackend = \"anthropic-messages\"\n\n\
         [profiles.mock]\naccount = \"mock\"\nmodel = \"claude-sonnet-4-5\"\n\n[permissions]\n\
         default = \"workspace\"\nenabled = [\"read\", \"workspace\"]\n",
    );
}

/// Reasoning reaches the terminal whether or not the turn streamed.
///
/// The blocking path returns the message whole, with no event channel to have put its thinking on
/// while it was being written, so `run_turn` has to re-emit each block by hand. A frontend renders
/// reasoning from those events and from nothing else, which makes the loop that re-emits them the
/// only thing standing between a `--no-stream` turn and no sign the model reasoned at all.
///
/// Asserted on stderr, which is where reasoning belongs: it is chrome, and a pipe reading stdout
/// must see only the answer.
#[test]
fn reasoning_is_shown_on_a_turn_that_did_not_stream() {
    const REASONED: &str = r#"[
  [{"type":"thinking_delta","text":"weighing the options"},
   {"type":"thinking_complete"},
   {"type":"text","text":"the answer"},
   {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    for extra in [vec![], vec!["--no-stream"]] {
        let install = Install::new();
        let config_dir = install.config_dir();
        std::fs::create_dir_all(&config_dir).expect("config dir");
        std::fs::write(
            config_dir.join("config.toml"),
            "default_profile = \"mock\"\n\n[accounts.mock]\nbackend = \
             \"anthropic-messages\"\n\n[profiles.mock]\naccount = \"mock\"\nmodel = \
             \"claude-sonnet-4-5\"\n\n[thinking]\nshow_content = true\n",
        )
        .expect("write config.toml");

        let mut args = extra.clone();
        args.extend(["--oneshot", "-p", "ponder"]);
        let run = run_scripted_from(&install, install.root(), REASONED, &args);
        let stderr = String::from_utf8_lossy(&run.stderr);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(run.status.success(), "run failed with {extra:?}: {stderr}");
        assert!(
            stderr.contains("weighing the options"),
            "reasoning missing from stderr with {extra:?}:\n{stderr}"
        );
        assert!(
            !stdout.contains("weighing the options"),
            "reasoning must not reach stdout with {extra:?}:\n{stdout}"
        );
        assert!(
            stdout.contains("the answer"),
            "the answer belongs on stdout with {extra:?}:\n{stdout}"
        );
    }
}

/// A resumed session reopens where it was recorded, not where this shell happens to be.
///
/// At `workspace` the working directory *is* the writable boundary, so taking the shell's would
/// silently widen it: resume a project session from `$HOME` and the whole home directory becomes
/// writable, with a scheduled job able to fire before the user can react.
#[test]
fn a_resumed_session_opens_in_the_directory_it_recorded() {
    let install = Install::new();
    write_capable_config(&install);
    let project = install.root().join("project");
    let elsewhere = install.root().join("elsewhere");
    std::fs::create_dir_all(&project).expect("project dir");
    std::fs::create_dir_all(&elsewhere).expect("elsewhere dir");

    let created = run_scripted_from(
        &install,
        &project,
        r#"[[{"type":"text","text":"ok"},{"type":"message_end","stop_reason":"end_turn"}]]"#,
        &["--oneshot", "-p", "start here"],
    );
    assert!(
        created.status.success(),
        "first run failed: {}",
        String::from_utf8_lossy(&created.stderr)
    );

    let resumed = run_scripted_from(&install, &elsewhere, WRITE_A_MARKER, &[
        "--oneshot",
        "-c",
        "-p",
        "write the marker",
    ]);
    assert!(
        resumed.status.success(),
        "resume failed: {}",
        String::from_utf8_lossy(&resumed.stderr)
    );

    assert!(
        project.join("marker.txt").exists(),
        "the resumed turn must run in the session's own directory; stderr:\n{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert!(
        !elsewhere.join("marker.txt").exists(),
        "the resumed turn must not run in the directory the shell happened to be in",
    );
}

/// ...and resuming does not rewrite the column either. An unattended `--oneshot -c` from cron or a
/// unit at `/` would otherwise repoint a long-lived session, taking every scheduled tool-gate --
/// which is re-checked in this directory -- with it.
#[test]
fn a_resumed_session_does_not_rewrite_its_recorded_directory() {
    let install = Install::new();
    write_capable_config(&install);
    let project = install.root().join("project");
    let elsewhere = install.root().join("elsewhere");
    std::fs::create_dir_all(&project).expect("project dir");
    std::fs::create_dir_all(&elsewhere).expect("elsewhere dir");

    let simple =
        r#"[[{"type":"text","text":"ok"},{"type":"message_end","stop_reason":"end_turn"}]]"#;
    run_scripted_from(&install, &project, simple, &[
        "--oneshot",
        "-p",
        "start here",
    ]);
    let before = only_session_cwd(&install).expect("the first run records a directory");

    run_scripted_from(&install, &elsewhere, simple, &["--oneshot", "-c", "again"]);
    assert_eq!(
        only_session_cwd(&install),
        Some(before),
        "a resume reads the recorded directory and leaves it alone",
    );
}

/// A recorded directory that has since been removed must not stop the session opening: warn, fall
/// back to where the process is, and carry on.
#[test]
fn a_resumed_session_falls_back_when_its_recorded_directory_is_gone() {
    let install = Install::new();
    write_capable_config(&install);
    let project = install.root().join("project");
    let elsewhere = install.root().join("elsewhere");
    std::fs::create_dir_all(&project).expect("project dir");
    std::fs::create_dir_all(&elsewhere).expect("elsewhere dir");

    let simple =
        r#"[[{"type":"text","text":"ok"},{"type":"message_end","stop_reason":"end_turn"}]]"#;
    run_scripted_from(&install, &project, simple, &[
        "--oneshot",
        "-p",
        "start here",
    ]);
    std::fs::remove_dir_all(&project).expect("remove the recorded directory");

    let resumed = run_scripted_from(&install, &elsewhere, WRITE_A_MARKER, &[
        "-v",
        "--oneshot",
        "-c",
        "-p",
        "write the marker",
    ]);
    assert!(
        resumed.status.success(),
        "a missing directory must not stop the session opening: {}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert!(
        elsewhere.join("marker.txt").exists(),
        "the fallback must be where the process is; stderr:\n{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    let stderr = String::from_utf8_lossy(&resumed.stderr);
    assert!(
        stderr.contains("no longer exists"),
        "the fallback must say so rather than silently relocating the session: {stderr}"
    );
}

/// Resuming a session whose recorded profile has left `config.toml` must fail the process, not just
/// print about it.
///
/// An interactive host that rendered the refusal and returned `Ok(())` would make `meka -r <id>;
/// echo $?` say `0` for a session it refused to open, and every supervisor and wrapper script would
/// read that as success. `--oneshot` on the same session, and a fresh session with an unresolvable
/// `default_profile` in either host, exit 1; the resume path in the REPL host must too.
#[test]
fn resuming_a_session_whose_profile_is_gone_exits_nonzero() {
    let install = Install::new();
    write_provider_config(&install, "ghost", &["alpha", "ghost"]);
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(
        created.status.success(),
        "the first turn should have created a session: {}",
        String::from_utf8_lossy(&created.stderr)
    );
    let id = only_session(&install);

    // What `meka profile remove ghost` or a hand edit leaves behind: the row still names it.
    write_provider_config(&install, "alpha", &["alpha"]);

    let refused = run_isolated(&install, &["-r", &id]);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("no profile named 'ghost'"),
        "the resume should have been refused by name: {stderr}"
    );
    assert!(
        !refused.status.success(),
        "a refused resume must exit non-zero, got {:?}: {stderr}",
        refused.status
    );
    // The hint adds the one thing the refusal above it cannot: the session id, and the only command
    // that rewrites a row's binding.
    assert!(
        stderr.contains(&format!("meka -r {id} --profile alpha")),
        "the hint should give the command that repins this session: {stderr}"
    );
    // And it adds nothing else. `profile add` here would have to invent the deleted profile's
    // account and `--model`, which meka never saw: `ghost` may have been on another account and
    // another model, so the command would create a different profile under the name the session
    // wants. The refusal above already says to restore it from config.toml, which is the honest
    // version of the same advice.
    assert!(
        !stderr.contains("profile add"),
        "the hint must not suggest recreating a profile whose type and model it cannot know: \
         {stderr}"
    );
}

/// `--profile` on a resume rewrites the row, which is the whole point of it being a repin rather
/// than a per-run override.
///
/// `apply_session_repin` could be replaced with `Ok(())` and every test stayed green: the resume
/// succeeded, the run used the new profile for its one turn, and the row silently kept the old one,
/// so the *next* resume went back. The row is the fact; this asserts the row.
#[test]
fn a_resume_with_provider_rewrites_the_row() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha", "beta"]);
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(
        created.status.success(),
        "the first turn should have created a session: {}",
        String::from_utf8_lossy(&created.stderr)
    );
    let id = only_session(&install);
    assert_eq!(recorded_profile(&install, &id), "alpha");

    let moved = run_scripted(&install, &[
        "-r",
        &id,
        "--profile",
        "beta",
        "--oneshot",
        "-p",
        "hi",
    ]);
    assert!(
        moved.status.success(),
        "the repinned resume should run: {}",
        String::from_utf8_lossy(&moved.stderr)
    );
    assert_eq!(
        recorded_profile(&install, &id),
        "beta",
        "the row must hold the new profile, or the next resume goes back to the old one"
    );
}

/// A `--profile` naming nothing configured is refused before anything is written, by the check
/// that reads the configured set rather than by the later failure to build a provider.
///
/// Both refuse in the same sentence now that both ask `require_profile`, so the row is what tells
/// them apart: the door refuses before the repin is committed, and a build failure comes after.
#[test]
fn a_resume_with_an_unconfigured_profile_is_refused_by_name() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(created.status.success());
    let id = only_session(&install);

    let refused = run_scripted(&install, &[
        "-r",
        &id,
        "--profile",
        "ghost",
        "--oneshot",
        "-p",
        "hi",
    ]);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "must not run: {stderr}");
    assert!(
        stderr.contains("no profile named 'ghost' (configured: alpha)"),
        "the refusal must name the profile and list the configured ones: {stderr}"
    );
    assert_eq!(
        recorded_profile(&install, &id),
        "alpha",
        "a refused repin must leave the row alone"
    );
}

/// `meka profile set` is the successor to the retired `--model`: it writes the key, leaves the
/// rest of the file alone, and leaves behind a config the next process can still start on.
///
/// End to end rather than at `set_profile_field`, because the unit test cannot see the last of
/// those: a write that parses in isolation can still produce a file that fails at startup.
#[test]
fn profile_set_writes_the_key_and_leaves_a_config_the_next_run_can_start_on() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);

    // A comment beside the key, which is exactly what a whole-table rewrite would eat.
    let config_path = install.config_dir().join("config.toml");
    let annotated = std::fs::read_to_string(&config_path)
        .expect("read config")
        .replace(
            "model = \"alpha-model\"",
            "model = \"alpha-model\" # the model, annotated",
        );
    std::fs::write(&config_path, annotated).expect("write config");

    let set = run_isolated(&install, &[
        "profile",
        "set",
        "alpha",
        "model",
        "swapped-model",
    ]);
    assert!(
        set.status.success(),
        "set should succeed: {}",
        String::from_utf8_lossy(&set.stderr)
    );

    let written = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        written.contains("model = \"swapped-model\""),
        "the new model is written: {written}"
    );
    assert!(
        written.contains("# the model, annotated"),
        "the comment beside the changed key survives: {written}"
    );
    assert!(
        written.contains("account = \"alpha\""),
        "the profile's other keys survive: {written}"
    );

    // The edited file still starts a process and runs a turn, which is what a whole class of
    // botched writes would break. Deliberately *not* a claim that the new model reached the wire:
    // `run_scripted`'s reply is fixed text, so nothing here can observe which model was built, and
    // saying otherwise would describe a guard this does not have.
    let turn = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(
        turn.status.success(),
        "the turn should run: {}",
        String::from_utf8_lossy(&turn.stderr)
    );

    // And `--unset` on the model is refused: a profile stating no model cannot run, so the write
    // door declines to produce one and the file keeps what it had.
    let unset = run_isolated(&install, &["profile", "set", "alpha", "model", "--unset"]);
    assert!(
        !unset.status.success(),
        "unsetting the model must be refused"
    );
    let stderr = String::from_utf8_lossy(&unset.stderr);
    assert!(
        stderr.contains("model"),
        "the refusal names the model: {stderr}"
    );
    let kept = std::fs::read_to_string(&config_path).expect("read config");
    assert!(
        kept.contains("model = \"swapped-model\""),
        "a refused write leaves the file as it was: {kept}"
    );
}

/// The retired per-profile override flags are gone from the parser, not merely ignored.
///
/// A flag that parses and does nothing is worse than one that does not parse: a script pinning a
/// model would keep exiting 0 while every turn ran on the profile's own. clap's unknown-argument
/// error is the honest answer, and it is what tells the user to look for the new door.
#[test]
fn the_retired_profile_override_flags_no_longer_parse() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    for flag in [
        vec!["--model", "pinned-model"],
        vec!["--base-url", "https://example.invalid"],
        vec!["--thinking", "off"],
        vec!["--thinking-budget", "2048"],
    ] {
        let mut args = flag.clone();
        args.extend(["--oneshot", "-p", "hi"]);
        let output = run_isolated(&install, &args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success(),
            "meka {} must not run: {stderr}",
            flag.join(" ")
        );
        assert!(
            stderr.contains("unexpected argument") && stderr.contains(flag[0]),
            "meka {} must be refused by name: {stderr}",
            flag.join(" ")
        );
    }
}

/// A setup failure that is *not* a missing profile must not offer to repin the session.
///
/// The gate could be replaced with `true` and nothing noticed, which turns every failed start into
/// advice to move a session that is bound exactly where it belongs. Here the profile is configured
/// and merely has no credential, so repinning fixes nothing.
#[test]
fn a_credential_failure_does_not_advise_repinning_a_session() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(created.status.success());
    let id = only_session(&install);

    // No `MEKA_MOCK_PROVIDER`, so the real credential lookup runs and finds nothing stored.
    let refused = run_isolated(&install, &["-r", &id]);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("no stored credential"),
        "this test needs the credential failure, not some other one: {stderr}"
    );
    assert!(
        !stderr.contains("Move this session onto"),
        "the session's profile is configured, so repinning is the wrong advice: {stderr}"
    );
}

/// The profile a session's row currently names.
fn recorded_profile(install: &Install, id: &str) -> String {
    store(install)
        .query_row("SELECT profile FROM sessions WHERE id = ?1", [id], |row| {
            row.get(0)
        })
        .expect("the session row")
}

/// The permission level the session row records.
fn recorded_permission(install: &Install, id: &str) -> Option<String> {
    store(install)
        .query_row(
            "SELECT permission FROM sessions WHERE id = ?1",
            [id],
            |row| row.get(0),
        )
        .expect("the session row")
}

/// `--permission` on a resume lands with the repin, after the profile is known to run. Written
/// first, a run that then failed to start left the row at a level it never ran at, for the
/// scheduler's gate re-check and every other reader of the row.
#[test]
fn a_refused_resume_leaves_the_recorded_permission_alone() {
    let install = Install::new();
    // `beta` is configured, so the repin passes the membership check and fails later, where the
    // profile has to produce a provider: it has no stored credential and this run has no mock.
    write_provider_config(&install, "alpha", &["alpha", "beta"]);
    // The level the resume asks for has to be one the config allows, or the run is refused before
    // the resume and the row is never in question.
    let config_path = install.config_dir().join("config.toml");
    let widened = std::fs::read_to_string(&config_path)
        .expect("read config")
        .replace(
            "enabled = [\"read\"]",
            "enabled = [\"read\", \"unrestricted\"]",
        );
    std::fs::write(&config_path, widened).expect("write config");
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(created.status.success());
    let id = only_session(&install);
    let before = recorded_permission(&install, &id);
    assert_eq!(before.as_deref(), Some("read"));

    let refused = run_isolated(&install, &[
        "-r",
        &id,
        "--profile",
        "beta",
        "--permission",
        "unrestricted",
        "--oneshot",
        "-p",
        "hi",
    ]);
    assert!(
        !refused.status.success(),
        "must not run: {}",
        String::from_utf8_lossy(&refused.stderr)
    );
    assert_eq!(
        recorded_profile(&install, &id),
        "alpha",
        "a repin that could not produce a provider must leave the row alone"
    );
    assert_eq!(
        recorded_permission(&install, &id),
        before,
        "a run that did not start must not have moved the level"
    );
}

/// The success path of the same door: a resume that runs records the level it was asked for, so
/// the next resume without `--permission` starts where this one left the row.
#[test]
fn a_resume_that_runs_records_the_level_it_was_asked_for() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let config_path = install.config_dir().join("config.toml");
    let widened = std::fs::read_to_string(&config_path)
        .expect("read config")
        .replace(
            "enabled = [\"read\"]",
            "enabled = [\"read\", \"unrestricted\"]",
        );
    std::fs::write(&config_path, widened).expect("write config");
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(created.status.success());
    let id = only_session(&install);
    assert_eq!(recorded_permission(&install, &id).as_deref(), Some("read"));

    let resumed = run_scripted(&install, &[
        "-r",
        &id,
        "--permission",
        "unrestricted",
        "--oneshot",
        "-p",
        "hi",
    ]);
    assert!(
        resumed.status.success(),
        "the resume must run: {}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert_eq!(
        recorded_permission(&install, &id).as_deref(),
        Some("unrestricted"),
        "a resume that ran must have recorded the level it was asked for"
    );
}

/// The same, without `--profile`: the session's own profile fails to produce a provider. The
/// level was written ahead of the build whenever there was no repin to wait for, so this door
/// moved the row while the `--profile` one did not.
#[test]
fn a_refused_resume_without_a_repin_leaves_the_recorded_permission_alone() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let config_path = install.config_dir().join("config.toml");
    let widened = std::fs::read_to_string(&config_path)
        .expect("read config")
        .replace(
            "enabled = [\"read\"]",
            "enabled = [\"read\", \"unrestricted\"]",
        );
    std::fs::write(&config_path, widened).expect("write config");
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(created.status.success());
    let id = only_session(&install);
    assert_eq!(recorded_permission(&install, &id).as_deref(), Some("read"));

    // No mock, so `alpha` has no credential and the build refuses.
    let refused = run_isolated(&install, &[
        "-r",
        &id,
        "--permission",
        "unrestricted",
        "--oneshot",
        "-p",
        "hi",
    ]);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "must not run: {stderr}");
    assert!(
        stderr.contains("no stored credential"),
        "this test needs the credential failure, not some other one: {stderr}"
    );
    assert_eq!(
        recorded_permission(&install, &id).as_deref(),
        Some("read"),
        "a run that did not start must not have moved the level"
    );
}

/// A one-shot turn that fails after detaching a command still waits for that command, reports
/// it, and exits non-zero. Returning the error early dropped the runtime with the task parked at an
/// await, so the child ran on untracked and its row stayed `running`.
#[test]
fn a_failed_oneshot_turn_still_waits_for_its_detached_work() {
    let install = Install::new();
    // `unrestricted`, so the command runs without a sandbox whatever this host offers: the
    // measurement is the exit path, not the shell.
    let config_dir = install.config_dir();
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        "default_profile = \"mock\"\n\n[accounts.mock]\nbackend = \"anthropic-messages\"\n\n\
         [profiles.mock]\naccount = \"mock\"\nmodel = \"claude-sonnet-4-5\"\n\n[permissions]\n\
         default = \"unrestricted\"\nenabled = [\"read\", \"unrestricted\"]\n\n[background]\n\
         enabled = true\n",
    )
    .expect("write config.toml");
    let output = run_scripted_from(
        &install,
        install.root(),
        r#"[
          [{"type":"tool_use_start","id":"call-1","name":"execute_command"},
           {"type":"tool_use_end","input":{"command":"sleep 1; echo finished-late","background":true}},
           {"type":"message_end","stop_reason":"tool_use"}],
          [{"type":"fail","message":"the provider fell over"}]
        ]"#,
        &["--oneshot", "-p", "run it"],
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "the turn failed: {stderr}");
    assert!(
        stderr.contains("the provider fell over"),
        "the failure is still reported: {stderr}"
    );
    assert!(
        stderr.contains("[Background task reporting"),
        "the detached command was waited for and reported on the way out: {stderr}"
    );
    let status: String = store(&install)
        .query_row("SELECT status FROM background_tasks", [], |row| row.get(0))
        .expect("the one task row");
    assert_ne!(
        status, "running",
        "the task finished before the process left"
    );
}

/// A sub-agent spawned with `writable_roots` writes under those directories and nowhere else,
/// through the real wiring: the session's own permission handle, the worker registry's write
/// fence, and the row the spawn leaves behind.
///
/// The worker's paths are relative on purpose. `inside.txt` lands where its working directory
/// is, which is the first root and not the parent's directory; `../outside.txt` is inside the
/// parent's workspace and still refused, since the worker's boundary is the list and nothing
/// of the parent's.
#[test]
fn a_sub_agent_bounded_by_writable_roots_writes_only_there() {
    let install = Install::new();
    write_capable_config(&install);
    let work = install.work_dir();
    let sub = work.join("sub");
    std::fs::create_dir_all(&sub).expect("sub dir");

    let output = run_scripted_from(
        &install,
        &work,
        r#"[
          [{"type":"tool_use_start","id":"call-1","name":"agent_spawn"},
           {"type":"tool_use_end","input":{"prompt":"write both","writable_roots":["sub"]}},
           {"type":"message_end","stop_reason":"tool_use"}],
          [{"type":"tool_use_start","id":"call-2","name":"write_file"},
           {"type":"tool_use_end","input":{"path":"inside.txt","content":"in"}},
           {"type":"message_end","stop_reason":"tool_use"}],
          [{"type":"tool_use_start","id":"call-3","name":"write_file"},
           {"type":"tool_use_end","input":{"path":"../outside.txt","content":"out"}},
           {"type":"message_end","stop_reason":"tool_use"}],
          [{"type":"text","text":"worker done"},{"type":"message_end","stop_reason":"end_turn"}],
          [{"type":"text","text":"dispatched"},{"type":"message_end","stop_reason":"end_turn"}]
        ]"#,
        &["--oneshot", "-p", "delegate"],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_eq!(
        std::fs::read_to_string(sub.join("inside.txt")).expect("written under the root"),
        "in"
    );
    assert!(
        !work.join("outside.txt").exists(),
        "a write into the parent's workspace but outside the worker's root must be refused"
    );
    let (cwd, spec): (String, String) = store(&install)
        .query_row(
            "SELECT cwd, subagent_spec_json FROM sessions WHERE parent_session_id IS NOT NULL",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("exactly one worker row");
    // meka records the canonical path without Windows' verbatim `\\?\` prefix, which
    // `std::fs::canonicalize` includes there.
    let expected = std::fs::canonicalize(&sub).expect("canonical");
    let expected = expected
        .to_string_lossy()
        .trim_start_matches(r"\\?\")
        .to_string();
    assert_eq!(
        cwd, expected,
        "the worker's working directory is the first root, spelled canonically"
    );
    assert!(
        spec.contains("\"writable_roots\":[\""),
        "the spawn terms carry the bounds: {spec}"
    );
}

/// `meka -r <worker-id>` refuses, on the CLI door the HTTP test cannot reach.
///
/// The sibling of `tests/serve.rs`'s `a_worker_session_refuses_a_turn_posted_straight_at_it`, and
/// not redundant with it: that one covers `build_session_agent`, while the REPL and `--oneshot` go
/// through `create_agent_from_config`. Two call sites, and deleting the guard from *this* one left
/// the whole suite green -- which is the untested-wiring shape the sub-agent refusal was added to
/// close in the first place.
///
/// The worker is spawned for real, because the id has to come from where a user's would: a listing.
/// Fabricating a row with a `parent_session_id` would test the predicate again rather than the
/// wiring.
#[test]
fn a_worker_session_refuses_a_resume_from_the_command_line() {
    let install = Install::new();
    write_capable_config(&install);

    let spawned = run_scripted_from(
        &install,
        install.root(),
        r#"[
          [{"type":"tool_use_start","id":"call-1","name":"agent_spawn"},
           {"type":"tool_use_end","input":{"prompt":"count the files","permission":"read"}},
           {"type":"message_end","stop_reason":"tool_use"}],
          [{"type":"text","text":"worker done"},{"type":"message_end","stop_reason":"end_turn"}],
          [{"type":"text","text":"dispatched"},{"type":"message_end","stop_reason":"end_turn"}]
        ]"#,
        &["--oneshot", "-p", "spawn one"],
    );
    assert!(
        spawned.status.success(),
        "the spawning turn has to succeed for there to be a worker: {}",
        String::from_utf8_lossy(&spawned.stderr)
    );

    let worker: String = store(&install)
        .query_row(
            "SELECT id FROM sessions WHERE parent_session_id IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .expect("the spawn should have left exactly one worker row");

    let refused = run_scripted(&install, &[
        "--oneshot",
        "-r",
        &worker,
        "-p",
        "drive it directly",
    ]);
    assert!(
        !refused.status.success(),
        "a worker must not take a turn from this door"
    );
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("agent_followup"),
        "the refusal must name the door that can drive it: {stderr}"
    );

    // The interactive host too, since the two order their startup independently.
    //
    // The `meka profile add` assertion below states an outcome, not a mechanism:
    // `resolve_session_resume` refuses before `run_interactive` reaches the builder at all, so the
    // hint is unreachable rather than suppressed. It is kept because the outcome is still what a
    // user must get (a refusal naming `agent_followup` and no advice to configure a profile), but
    // it guards no suppression of its own.
    let interactive = run_scripted(&install, &["-r", &worker]);
    let stderr = String::from_utf8_lossy(&interactive.stderr);
    assert!(
        stderr.contains("agent_followup"),
        "the REPL host refuses a worker by the same rule: {stderr}"
    );
    assert!(
        !stderr.contains("meka profile add"),
        "and must not follow it with advice to configure a profile, which is not the \
         problem: {stderr}"
    );

    // And the refusal comes before `--profile` repins the row, which is the whole reason it sits
    // ahead of `apply_session_repin` in both hosts rather than merely inside the builders. A run
    // that refuses to touch a session must not have already rewritten it on the way to saying so.
    let config = install.config_dir().join("config.toml");
    let mut toml = std::fs::read_to_string(&config).expect("read config.toml");
    toml.push_str("\n[profiles.second]\naccount = \"alpha\"\nmodel = \"other-model\"\n");
    std::fs::write(&config, toml).expect("write config.toml");

    // Both columns a resume can rewrite, not just the provider: `--profile` is computed in
    // `resolve_session_resume` and committed by `apply_session_repin` afterwards, while
    // `--permission` is committed by `resolve_session_resume` itself, so a refusal placed between
    // them covered one and missed the other. Reading only `provider` was how that stayed green.
    let row = |id: &str| -> (Option<String>, Option<String>) {
        store(&install)
            .query_row(
                "SELECT profile, permission FROM sessions WHERE id = ?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("the worker row is readable")
    };
    let before = row(&worker);

    let repinned = run_scripted(&install, &[
        "--oneshot",
        "--profile",
        "second",
        "-r",
        &worker,
        "drive it on another profile",
    ]);
    assert!(
        !repinned.status.success(),
        "naming a profile does not make a worker drivable: {}",
        String::from_utf8_lossy(&repinned.stderr)
    );
    assert_eq!(
        before,
        row(&worker),
        "a refused run must leave the row alone, not rewrite it on the way to the refusal"
    );

    // Both hosts, because they order these two calls independently and the assertion above only
    // reaches `run_oneshot`. Swapping them in `run_interactive` alone left this test green.
    let repinned = run_scripted(&install, &["--profile", "second", "-r", &worker]);
    assert!(
        !repinned.status.success(),
        "the REPL host refuses it too: {}",
        String::from_utf8_lossy(&repinned.stderr)
    );
    assert_eq!(
        before,
        row(&worker),
        "and leaves the row alone by the same ordering"
    );

    // `--permission` is the sibling flag, and the one an ordering check is likeliest to miss: it is
    // written a function earlier than the repin, so a refusal placed between the two lets it
    // through. The value it falsifies travels into `session list`, `GET /v1/sessions/{id}` and
    // every archive made from this store.
    let repermissioned = run_scripted(&install, &[
        "--oneshot",
        "--permission",
        "read",
        "-r",
        &worker,
        "drive it at another level",
    ]);
    assert!(
        !repermissioned.status.success(),
        "naming a permission does not make a worker drivable: {}",
        String::from_utf8_lossy(&repermissioned.stderr)
    );
    assert_eq!(
        before,
        row(&worker),
        "and the refusal comes before the level is recorded, or the row claims the worker ran at \
         a level it never ran at"
    );
}

/// A one-shot run carries an outcome that was waiting, and prints one that lands during it.
///
/// `--oneshot` is the fourth host and the one it is easiest to forget: it has no REPL loop, no
/// poller and exactly one turn. A resume inherits whatever the last process left undelivered -- a
/// cancellation, or a task the session-load sweep retires as `interrupted` -- and without the fold
/// that report reached nobody until the run was already over.
#[test]
fn a_oneshot_run_carries_an_outcome_that_was_waiting() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let config = install.config_dir().join("config.toml");
    let mut text = std::fs::read_to_string(&config).expect("read config.toml");
    text.push_str("\n[background]\nenabled = true\n");
    std::fs::write(&config, text).expect("write config.toml");

    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(
        created.status.success(),
        "the first turn should have created a session: {}",
        String::from_utf8_lossy(&created.stderr)
    );
    let id = only_session(&install);

    // Exactly what `/tasks cancel` leaves behind for the next process to carry.
    let now = chrono::Utc::now().to_rfc3339();
    store(&install)
        .execute(
            "INSERT INTO background_tasks \
             (id, session_id, tool_name, label, status, outcome, started_at, finished_at) \
             VALUES (?1, ?2, 'execute_command', 'sleep 900', 'canceled', NULL, ?3, ?3)",
            rusqlite::params![uuid::Uuid::new_v4().to_string(), &id, now],
        )
        .expect("seed the canceled task");

    let resumed = run_scripted(&install, &["-r", &id, "--oneshot", "-p", "what happened?"]);
    assert!(
        resumed.status.success(),
        "the resumed run should have succeeded: {}",
        String::from_utf8_lossy(&resumed.stderr)
    );

    let carrier: String = store(&install)
        .query_row(
            "SELECT content FROM messages WHERE session_id = ?1 AND role IN ('user', 'user_blocks') \
             ORDER BY id DESC LIMIT 1",
            rusqlite::params![&id],
            |row| row.get(0),
        )
        .expect("the resumed turn's user message");
    assert!(
        carrier.contains("was canceled"),
        "the outcome must ride inside the one turn this run has: {carrier}"
    );
    assert!(
        carrier.contains("what happened?"),
        "and the prompt has to still be there: {carrier}"
    );

    let delivered: Option<String> = store(&install)
        .query_row(
            "SELECT delivered_at FROM background_tasks WHERE session_id = ?1",
            rusqlite::params![&id],
            |row| row.get(0),
        )
        .expect("read the task");
    assert!(
        delivered.is_some(),
        "and riding a turn is a delivery, so the row must be stamped"
    );
}

/// A reader that hangs up mid-answer ends the run cleanly.
///
/// `meka --oneshot … | head -1` is the shape one-shot mode exists for, and a short-lived reader is
/// ordinary, so the write that finds the closed pipe has to be a failure the renderer reports
/// rather than one it panics on. A panicking write is bad enough on its own; paired with a `Drop`
/// that flushes on the way out it is worse, because the second panic during the unwind is not an
/// exit code at all but `SIGABRT`.
///
/// Every delta below closes one paragraph and opens another, which is what a real provider streams
/// and what leaves a partial paragraph buffered: a flush drains only what it can render, so a
/// buffer emptied by the failing flush leaves the tail nothing to print and hides the fault.
#[cfg(unix)]
#[test]
fn a_reader_that_hangs_up_mid_answer_does_not_abort_the_run() {
    use std::io::Read as _;

    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);

    let deltas: Vec<String> = (1..40)
        .map(|index| {
            format!(
                r#"{{"type":"text","text":"end of paragraph {index}.\n\nthe start of paragraph {} which is not yet"}}"#,
                index + 1
            )
        })
        .collect();
    install.write_script(format!(
        r#"[[{},{{"type":"message_end","stop_reason":"end_turn"}}]]"#,
        deltas.join(",")
    ));

    let mut child = install
        .meka(&["--oneshot", "-p", "write something long"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn meka");

    // One read, then hang up: `head -1` in process form. Reading first is what makes the close land
    // mid-stream rather than before the first write.
    {
        let mut stdout = child.stdout.take().expect("piped stdout");
        let mut first = [0u8; 1];
        // The child may not have written yet, and either way the point is to close the pipe.
        if let Err(error) = stdout.read(&mut first) {
            panic!("reading one byte from the child failed: {error}");
        }
    }
    let status = child.wait().expect("wait for meka");

    use std::os::unix::process::ExitStatusExt as _;
    assert!(
        status.signal().is_none(),
        "the run died on signal {:?} rather than exiting; a panicking `Drop` on an unwinding \
         thread aborts, and the last episode's flush is the one write certain to fail again",
        status.signal()
    );
    assert!(
        status.success(),
        "and it exited {:?}: a reader that stops reading is its own decision, not a failure of \
         the run",
        status.code()
    );
}

/// A stdout that will not take the answer fails the run, and says so once.
///
/// The counterpart to the broken pipe above, and the reason the two are told apart: a reader that
/// hangs up chose to, but a full disk did not, and `meka -p … > out.txt` reporting success over an
/// empty file is the worse failure of the two. Once, because `text_delta` runs per streamed delta
/// and a real answer is hundreds of them.
#[cfg(target_os = "linux")]
#[test]
fn a_stdout_that_will_not_take_the_answer_fails_the_run_once() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);

    let deltas: Vec<String> = (1..40)
        .map(|index| {
            format!(
                r#"{{"type":"text","text":"end of paragraph {index}.\n\nthe start of paragraph {} which is not yet"}}"#,
                index + 1
            )
        })
        .collect();
    install.write_script(format!(
        r#"[[{},{{"type":"message_end","stop_reason":"end_turn"}}]]"#,
        deltas.join(",")
    ));

    let full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .expect("/dev/full");
    let run = install
        .meka(&["--oneshot", "-p", "write something long"])
        .stdout(std::process::Stdio::from(full))
        .output()
        .expect("spawn meka");

    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success(),
        "losing the answer to a full disk has to reach the exit code: {stderr}"
    );
    // Counting the log line, not the phrase: the error report below it says the same thing, and a
    // count that leaned on the two being worded differently would be a test choosing the wording.
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.contains("WARN") && line.contains("did not reach stdout"))
            .count(),
        1,
        "and it has to be said once, not once per delta: {stderr}"
    );
}

/// The first message labels a session under one name on every surface: `title`, in `session show`
/// and at the head of the `session list` column, the same word the HTTP and ACP records carry.
#[test]
fn a_session_is_labeled_title_in_show_and_list() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let created = run_scripted(&install, &["--oneshot", "-p", "name   the\nsession"]);
    assert!(created.status.success(), "seed a session");
    let id = only_session(&install);

    let shown = run_isolated(&install, &["session", "show", &id]);
    let stdout = String::from_utf8_lossy(&shown.stdout);
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("title:") && line.ends_with("name the session")),
        "`session show` labels the first message `title`, collapsed to one line: {stdout}"
    );
    assert!(
        !stdout.contains("opening"),
        "the old label is gone from the record: {stdout}"
    );

    let listed = run_isolated(&install, &["session", "list"]);
    let stdout = String::from_utf8_lossy(&listed.stdout);
    let header = stdout.lines().next().unwrap_or_default();
    assert!(
        header.ends_with("Title"),
        "the listing's last column is the title: {header:?}"
    );
    assert!(
        stdout.contains("name the session"),
        "and the row shows it: {stdout}"
    );
}

/// Every command that prints survives a reader that hangs up, and none of them calls that failure.
///
/// The renderer was converted first and the rest of the CLI was not, which left `session export
/// --output - | head` crashing where `--oneshot | head` had just been fixed. `CLAUDE.md` names the
/// litmus test -- `meka … 2>/dev/null | next-tool` -- so a `print!` that panics is a broken
/// contract in any of them, not only the streaming one.
#[cfg(unix)]
#[test]
fn no_command_dies_because_its_reader_stopped_reading() {
    use std::os::fd::FromRawFd as _;

    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(created.status.success(), "seed a session");
    let id = only_session(&install);

    // Every command below has to actually reach stdout, or it proves nothing: one that finds
    // nothing to list says so on stderr and writes no bytes at all, so the pipe never breaks.
    let skill = install.config_dir().join("skills").join("demo");
    std::fs::create_dir_all(&skill).expect("skill dir");
    // Bigger than a pipe buffer on purpose. Dropping the read end races the child, and a command
    // whose whole answer fits in the buffer wins that race and exits cleanly even when its writes
    // panic -- which is how a listing can pass while the per-line printer beside it crashes.
    let long = "d".repeat(128 * 1024);
    std::fs::write(
        skill.join("SKILL.md"),
        format!("---\nname: demo\ndescription: {long}\n---\n\nBody.\n"),
    )
    .expect("write SKILL.md");
    let config = install.config_dir().join("config.toml");
    let mut text = std::fs::read_to_string(&config).expect("read config.toml");
    text.push_str(
        "\n[[mcp.servers]]\nname = \"demo\"\ntransport = \"stdio\"\ncommand = \"true\"\n",
    );
    std::fs::write(&config, text).expect("write config.toml");

    // The `get` commands first. Those print a line at a time, so the reader's hangup lands between
    // two writes -- which is what actually crashed. A listing emits its whole table in one write
    // that fits the pipe buffer, so it survives even unfixed and proves nothing on its own.
    for args in [
        vec!["skill", "get", "demo"],
        vec!["mcp", "get", "demo"],
        vec!["instructions", "show"],
        vec!["session", "export", id.as_str(), "--output", "-"],
        vec!["session", "list"],
        vec!["session", "show", id.as_str()],
        vec!["profile", "list"],
        vec!["account", "list"],
        vec!["mcp", "list"],
        vec!["skill", "list"],
        vec!["memory", "list"],
        vec!["schedule", "list"],
    ] {
        let mut ends = [0 as libc::c_int; 2];
        // SAFETY: `pipe` fills two descriptors this test then owns.
        assert_eq!(unsafe { libc::pipe(ends.as_mut_ptr()) }, 0, "pipe");
        // SAFETY: closing the read end this test just made, and handing the write end to `Stdio`,
        // which takes ownership of it.
        let gone = unsafe {
            libc::close(ends[0]);
            std::process::Stdio::from_raw_fd(ends[1])
        };
        let mut child = install
            .meka(&args)
            .stdout(gone)
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap_or_else(|error| panic!("spawn meka {args:?}: {error}"));
        let status = child.wait().expect("wait for meka");

        use std::os::unix::process::ExitStatusExt as _;
        assert!(
            status.signal().is_none(),
            "`meka {}` died on signal {:?}",
            args.join(" "),
            status.signal()
        );
        assert!(
            status.success(),
            "`meka {}` exited {:?}; a reader that stops reading is its own decision",
            args.join(" "),
            status.code()
        );
    }
}

/// A destination the user named is not a pipeline, and losing it is still a failure.
///
/// `meka … | head` exits 0 because the reader chose to stop, and that carve-out has to mean the
/// reader of *stdout*. `--output <fifo>` raises the same `BrokenPipe` from a place the user pointed
/// at by name; answering 0 there reports success over data that never landed, which is worse than
/// the crash the carve-out replaced.
#[cfg(unix)]
#[test]
fn a_named_destination_that_goes_away_is_still_a_failure() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    // A transcript bigger than a pipe buffer, or the whole export lands before the reader leaves
    // and there is no broken pipe left to test.
    let paragraph = "e".repeat(400);
    let deltas: Vec<String> = (0..200)
        .map(|_| format!(r#"{{"type":"text","text":"{paragraph}\n\n"}}"#))
        .collect();
    let script = format!(
        r#"[[{},{{"type":"message_end","stop_reason":"end_turn"}}]]"#,
        deltas.join(",")
    );
    assert!(
        run_scripted_from(&install, install.root(), &script, &[
            "--oneshot",
            "-p",
            "hello"
        ])
        .status
        .success(),
        "seed a session"
    );
    let id = only_session(&install);

    let fifo = install.root().join("sink");
    let path = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).expect("path");
    // SAFETY: a nul-terminated path this test owns; `mkfifo` writes nothing back.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0, "mkfifo");

    // Opened before the child and non-blocking, so neither side can wait on the other: opening a
    // fifo for writing blocks until a reader arrives, and for reading until a writer does. A thread
    // that blocks in `open` would hang the suite instead of failing it whenever meka exits before
    // it gets that far -- an unknown session, an unreadable config, a refused lock.
    use std::os::unix::fs::OpenOptionsExt as _;
    let reader = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&fifo)
        .expect("open the fifo for reading");

    let run = install
        .meka(&["session", "export", id.as_str(), "--output"])
        .arg(&fifo)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn meka");

    // Take a little and hang up, so the export breaks partway rather than never starting.
    {
        use std::io::Read as _;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut taken = [0u8; 8];
        while std::time::Instant::now() < deadline {
            match (&reader).read(&mut taken) {
                Ok(count) if count > 0 => break,
                _ => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        drop(reader);
    }
    let run = run.wait_with_output().expect("wait for meka");

    assert!(
        !run.status.success(),
        "the export never landed, so the command did not do what it was asked: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// An export to a regular path is owner-only, as `memory export` writes: a transcript carries tool
/// output and whatever was pasted, so it must not land at the umask's mode.
#[cfg(unix)]
#[test]
fn an_export_file_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;

    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(created.status.success());
    let id = only_session(&install);
    let out = install.root().join("transcript.md");

    let exported = run_isolated(&install, &[
        "session",
        "export",
        &id,
        "--output",
        out.to_str().expect("path"),
    ]);
    assert!(
        exported.status.success(),
        "{}",
        String::from_utf8_lossy(&exported.stderr)
    );
    let mode = std::fs::metadata(&out)
        .expect("the export")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600, "an export is private data");
}

/// `[session].retention` must not take the session `-r` names. The sweep spares only a
/// session some process holds, so it has to run after the resume has taken its lock; run first, it
/// deleted the conversation the user had just listed and then failed on "no session matches".
#[test]
fn retention_spares_the_session_being_resumed_and_still_sweeps_the_rest() {
    let install = Install::new();
    write_provider_config(&install, "default", &["default"]);
    for prompt in ["first", "second"] {
        let output = run_scripted(&install, &["--oneshot", "-p", prompt]);
        assert!(
            output.status.success(),
            "seeding turn failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let ids: Vec<String> = {
        let store = store(&install);
        let stale = (chrono::Utc::now() - chrono::Duration::days(40)).to_rfc3339();
        store
            .execute("UPDATE sessions SET updated_at = ?1", rusqlite::params![
                stale
            ])
            .expect("age both sessions past the window");
        let mut statement = store
            .prepare("SELECT id FROM sessions ORDER BY rowid")
            .expect("prepare");
        statement
            .query_map([], |row| row.get::<_, String>(0))
            .expect("query")
            .collect::<Result<Vec<_>, _>>()
            .expect("ids")
    };
    assert_eq!(ids.len(), 2, "two sessions to start with");

    let config_path = install.config_dir().join("config.toml");
    let mut config = std::fs::read_to_string(&config_path).expect("read config");
    config.push_str("\n[session]\nretention = \"30d\"\n");
    std::fs::write(&config_path, config).expect("write config");

    let resumed = run_scripted(&install, &["-r", &ids[0], "--oneshot", "-p", "again"]);
    assert!(
        resumed.status.success(),
        "resuming an aged session must not be defeated by the retention sweep: {}",
        String::from_utf8_lossy(&resumed.stderr)
    );

    let store = store(&install);
    let mut statement = store
        .prepare("SELECT id FROM sessions ORDER BY rowid")
        .expect("prepare");
    let remaining: Vec<String> = statement
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query")
        .collect::<Result<Vec<_>, _>>()
        .expect("ids");
    assert_eq!(
        remaining,
        vec![ids[0].clone()],
        "the resumed session survives and the other aged one is swept"
    );
}

/// The one JSON document a `--format json` command printed, after checking it exited 0.
fn json_stdout(output: std::process::Output) -> serde_json::Value {
    assert!(
        output.status.success(),
        "exited {:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "stdout is not one JSON document ({error}): {}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

/// Every listing under `--format json` is one `{"<nouns>": [...]}` document on stdout, the envelope
/// around an empty array when there is nothing to list, and nothing on stderr: a script reading it
/// never has to tell "no rows" from "the command did not run". The plain rendering of the same
/// empty listing says so on stderr and writes nothing to stdout.
#[test]
fn an_empty_listing_in_json_is_an_empty_envelope_and_a_quiet_stderr() {
    let install = Install::new();
    for (command, nouns) in [
        (vec!["session", "list"], "sessions"),
        (vec!["account", "list"], "accounts"),
        (vec!["profile", "list"], "profiles"),
        (vec!["mcp", "list"], "servers"),
        (vec!["schedule", "list"], "jobs"),
        (vec!["memory", "list"], "memories"),
        (vec!["history", "list"], "history"),
        (vec!["skill", "list"], "skills"),
    ] {
        let mut arguments = command.clone();
        arguments.extend(["--format", "json"]);
        let output = run_isolated(&install, &arguments);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.is_empty(),
            "{command:?} must say nothing on stderr under json, got: {stderr}"
        );
        let document = json_stdout(output);
        let object = document
            .as_object()
            .unwrap_or_else(|| panic!("{command:?} must print an object: {document}"));
        assert_eq!(
            object.len(),
            1,
            "{command:?}: one key, the nouns: {document}"
        );
        assert_eq!(
            object
                .get(nouns)
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(0),
            "{command:?} must print {{\"{nouns}\": []}}: {document}"
        );

        let plain = run_isolated(&install, &command);
        assert!(plain.status.success(), "{command:?}: {:?}", plain.status);
        assert!(
            plain.stdout.is_empty(),
            "{command:?} must print no placeholder row: {}",
            String::from_utf8_lossy(&plain.stdout)
        );
        let stderr = String::from_utf8_lossy(&plain.stderr);
        assert!(
            stderr.starts_with("No ") && stderr.trim_end().ends_with('.'),
            "{command:?} must say `No <nouns>.` on stderr, got: {stderr}"
        );
    }
}

/// `session list --format json` and `session show --format json` print the row's fields under the
/// names the HTTP API uses, ids in full, and no field only a running host could answer.
#[test]
fn session_list_and_show_print_the_session_as_json() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let created = run_scripted(&install, &["--oneshot", "-p", "name the session"]);
    assert!(
        created.status.success(),
        "seed a session: {}",
        String::from_utf8_lossy(&created.stderr)
    );
    let id = only_session(&install);

    let listed = json_stdout(run_isolated(&install, &[
        "session", "list", "--format", "json",
    ]));
    let sessions = listed["sessions"].as_array().expect("an array");
    assert_eq!(sessions.len(), 1, "{listed}");
    let row = &sessions[0];
    assert_eq!(row["id"], id);
    assert_eq!(row["profile"], "alpha");
    assert_eq!(row["title"], "name the session");
    assert_eq!(row["permission"], "read");
    assert_eq!(row["approvals"], false);
    assert!(
        row["created_at"].is_string() && row["updated_at"].is_string() && row["cwd"].is_string(),
        "{row}"
    );
    assert!(
        row.get("turn_in_flight").is_none() && row.get("capabilities").is_none(),
        "nothing this reader cannot answer, and no capabilities a CLI session never declared: {row}"
    );
    assert!(
        row.get("parent_id").is_none(),
        "a root session's optional is omitted, not null: {row}"
    );

    let shown = json_stdout(run_isolated(&install, &[
        "session",
        "show",
        &id[..8],
        "--format",
        "json",
    ]));
    assert_eq!(shown, *row, "show prints the object the listing carries");
}

/// `account list` and `profile list` under `--format json`: the config's own fields, the account's
/// backend on each profile, `active` on the one a session gets by default, and never a secret.
#[test]
fn account_and_profile_lists_print_json_under_the_http_names() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha", "beta"]);

    let accounts = json_stdout(run_isolated(&install, &[
        "account", "list", "--format", "json",
    ]));
    let rows = accounts["accounts"].as_array().expect("an array");
    assert_eq!(rows.len(), 2, "{accounts}");
    assert_eq!(rows[0]["name"], "alpha");
    assert_eq!(rows[0]["backend"], "openai-chat-completions");
    assert_eq!(rows[0]["base_url"], "http://127.0.0.1:9/");
    assert_eq!(rows[0]["authenticated"], "no");

    let profiles = json_stdout(run_isolated(&install, &[
        "profile", "list", "--format", "json",
    ]));
    let rows = profiles["profiles"].as_array().expect("an array");
    let alpha = rows
        .iter()
        .find(|row| row["name"] == "alpha")
        .expect("alpha");
    assert_eq!(alpha["account"], "alpha");
    assert_eq!(alpha["backend"], "openai-chat-completions");
    assert_eq!(alpha["model"], "alpha-model");
    assert_eq!(alpha["active"], true);
    let beta = rows.iter().find(|row| row["name"] == "beta").expect("beta");
    assert_eq!(beta["active"], false);
}

/// `mcp list` and `mcp get` under `--format json` carry the configuration and the kinds of secret,
/// never a value that could be one: `env` and `headers` reach the document as their keys alone.
#[test]
fn mcp_list_and_get_print_the_server_as_json_without_secrets() {
    let install = Install::new();
    let added = run_isolated(&install, &[
        "mcp",
        "add",
        "--env",
        "PGPASSWORD=hunter2",
        "--tool-permission",
        "query=read",
        "pg",
        "npx",
        "-y",
        "@modelcontextprotocol/server-postgres",
    ]);
    assert!(
        added.status.success(),
        "mcp add: {}",
        String::from_utf8_lossy(&added.stderr)
    );

    let listed = json_stdout(run_isolated(&install, &["mcp", "list", "--format", "json"]));
    let server = &listed["servers"][0];
    assert_eq!(server["name"], "pg");
    assert_eq!(server["transport"], "stdio");
    assert_eq!(server["command"], "npx");
    assert_eq!(
        server["args"],
        serde_json::json!(["-y", "@modelcontextprotocol/server-postgres"])
    );
    assert_eq!(server["required"], false);
    assert_eq!(server["disabled"], false);
    assert!(
        server.get("url").is_none(),
        "a stdio server has no url: {server}"
    );

    let detail = json_stdout(run_isolated(&install, &[
        "mcp", "get", "pg", "--format", "json",
    ]));
    assert_eq!(detail["name"], "pg");
    assert_eq!(detail["command"], "npx");
    assert_eq!(detail["env_keys"], serde_json::json!(["PGPASSWORD"]));
    assert_eq!(detail["tool_permissions"]["query"], "read");
    assert!(
        detail.get("credentials").is_none(),
        "nothing is stored for a server never logged in: {detail}"
    );
    let text = detail.to_string();
    assert!(
        !text.contains("hunter2"),
        "an env value may be a secret and must not reach the document: {text}"
    );
}

/// `memory list`, `get` and `show` under `--format json`: the fields the HTTP view carries plus the
/// read count, with the body riding only on `show`.
#[test]
fn memory_list_get_and_show_print_json_and_only_show_carries_the_body() {
    let install = Install::new();
    let added = run_isolated(&install, &[
        "memory",
        "add",
        "tz",
        "--description",
        "K4YT3X is in UTC+8",
        "--priority",
        "2",
        "--tag",
        "people",
        "--body",
        "detail line",
    ]);
    assert!(
        added.status.success(),
        "memory add: {}",
        String::from_utf8_lossy(&added.stderr)
    );

    let listed = json_stdout(run_isolated(&install, &[
        "memory", "list", "--format", "json",
    ]));
    let row = &listed["memories"][0];
    assert_eq!(row["name"], "tz");
    assert_eq!(row["description"], "K4YT3X is in UTC+8");
    assert_eq!(row["priority"], 2);
    assert_eq!(row["tags"], serde_json::json!(["people"]));
    assert_eq!(row["read_count"], 0);
    assert!(
        row["recorded_at"].is_string() && row["updated_at"].is_string(),
        "{row}"
    );
    assert!(
        row.get("body").is_none(),
        "the listing leaves the body out: {row}"
    );

    let got = json_stdout(run_isolated(&install, &[
        "memory", "get", "tz", "--format", "json",
    ]));
    assert_eq!(got, *row, "get prints the listing's object");

    let shown = json_stdout(run_isolated(&install, &[
        "memory", "show", "tz", "--format", "json",
    ]));
    assert_eq!(shown["name"], "tz");
    assert_eq!(shown["body"], "detail line");
}

/// `skill list`, `get` and `show` under `--format json`: the palette's fields plus where each skill
/// is, the frontmatter in full on `get`, and the body as the agent receives it on `show`.
#[test]
fn skill_list_get_and_show_print_json() {
    let install = Install::new();
    let added = run_isolated(&install, &[
        "skill",
        "add",
        "demo",
        "--description",
        "Demonstrates the listing",
        "--priority",
        "3",
        "--metadata",
        "author=Jane Doe",
    ]);
    assert!(
        added.status.success(),
        "skill add: {}",
        String::from_utf8_lossy(&added.stderr)
    );

    let listed = json_stdout(run_isolated(&install, &[
        "skill", "list", "--format", "json",
    ]));
    let row = &listed["skills"][0];
    assert_eq!(row["name"], "demo");
    assert_eq!(row["description"], "Demonstrates the listing");
    assert_eq!(row["priority"], 3);
    assert_eq!(row["author"], "Jane Doe");
    assert_eq!(row["external"], false);
    let source_dir = row["source_dir"].as_str().expect("a path");
    assert!(source_dir.ends_with("demo"), "{source_dir}");

    let got = json_stdout(run_isolated(&install, &[
        "skill", "get", "demo", "--format", "json",
    ]));
    assert_eq!(got["name"], "demo");
    assert_eq!(got["source_dir"], row["source_dir"]);
    assert!(
        got["body_path"]
            .as_str()
            .is_some_and(|path| path.ends_with("SKILL.md")),
        "{got}"
    );
    assert_eq!(got["metadata"]["author"], "Jane Doe");
    assert!(
        got.get("body").is_none(),
        "get prints the frontmatter, not the body: {got}"
    );

    let shown = json_stdout(run_isolated(&install, &[
        "skill", "show", "demo", "--format", "json",
    ]));
    assert!(
        shown["body"]
            .as_str()
            .is_some_and(|body| body.contains(source_dir)),
        "show carries the body as the agent receives it, header included: {shown}"
    );
}

/// `schedule list` and `schedule show` under `--format json` print each job with the fields
/// `GET /v1/schedule` uses, ids in full, and the optional fields absent rather than null.
#[test]
fn schedule_list_and_show_print_the_job_as_json() {
    let install = Install::new();
    write_provider_config(&install, "alpha", &["alpha"]);
    let created = run_scripted(&install, &["--oneshot", "-p", "hello"]);
    assert!(
        created.status.success(),
        "seed a session: {}",
        String::from_utf8_lossy(&created.stderr)
    );
    let session = only_session(&install);
    let job = "7f3a1b2c-0000-4000-8000-000000000000";
    let now = chrono::Utc::now();
    store(&install)
        .execute(
            "INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, created_at, \
             next_fire_at) VALUES (?1, ?2, 'every', '1h', 'watch the thing', ?3, ?4)",
            rusqlite::params![
                job,
                session,
                now.to_rfc3339(),
                (now + chrono::Duration::hours(1)).to_rfc3339()
            ],
        )
        .expect("plant a job");

    let listed = json_stdout(run_isolated(&install, &[
        "schedule", "list", "--format", "json",
    ]));
    let row = &listed["jobs"][0];
    assert_eq!(row["id"], job);
    assert_eq!(row["session_id"], session);
    assert_eq!(row["schedule"], "every 1h");
    assert_eq!(row["prompt"], "watch the thing");
    assert!(
        row["next_fire_at"].is_string() && row["created_at"].is_string(),
        "{row}"
    );
    assert!(
        row.get("gate").is_none()
            && row.get("withheld").is_none()
            && row.get("last_fired_at").is_none(),
        "an ungated job that has never fired carries none of the optional fields: {row}"
    );

    let shown = json_stdout(run_isolated(&install, &[
        "schedule",
        "show",
        &job[..8],
        "--format",
        "json",
    ]));
    assert_eq!(shown, *row, "show prints the object the listing carries");
}

/// `history list --format json` is `{"history": [...]}`, oldest first like the plain lines.
#[test]
fn history_list_prints_json() {
    let install = Install::new();
    // Any command opens the store, which is what creates the table the rows below go into.
    let opened = run_isolated(&install, &["session", "list"]);
    assert!(opened.status.success(), "{:?}", opened.status);
    let store = store(&install);
    for (entry, at) in [
        ("first", "2026-01-01T00:00:00Z"),
        ("second", "2026-01-02T00:00:00Z"),
    ] {
        store
            .execute(
                "INSERT INTO prompt_history (command_line, created_at) VALUES (?1, ?2)",
                rusqlite::params![entry, at],
            )
            .expect("plant history");
    }
    let listed = json_stdout(run_isolated(&install, &[
        "history", "list", "--format", "json",
    ]));
    assert_eq!(listed["history"], serde_json::json!(["first", "second"]));
}

/// `tools list --format json` carries what the table shows: the effective level, where it came
/// from, and whether the config admits the tool, beside the description the table cuts short.
#[test]
fn tools_list_prints_json_with_the_source_and_visibility_of_each_tool() {
    let install = Install::new();
    let config_dir = install.config_dir();
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        "[tools]\ndisabled_tools = [\"agent_spawn\"]\n\n[tools.tool_permissions]\nread_file = \
         \"none\"\n",
    )
    .expect("write config.toml");

    let listed = json_stdout(run_isolated(&install, &[
        "tools", "list", "--format", "json",
    ]));
    let tools = listed["tools"].as_array().expect("an array");
    let find = |name: &str| {
        tools
            .iter()
            .find(|tool| tool["name"] == name)
            .unwrap_or_else(|| panic!("{name} must be listed: {listed}"))
    };
    let read_file = find("read_file");
    assert_eq!(read_file["required_permission"], "none");
    assert_eq!(read_file["permission_source"], "override");
    assert_eq!(read_file["enabled"], true);
    let execute = find("execute_command");
    assert_eq!(execute["permission_source"], "builtin");
    assert!(
        execute["description"]
            .as_str()
            .is_some_and(|description| description.len() > 60),
        "the document carries the whole description, not the table's 60 columns: {execute}"
    );
    assert_eq!(find("agent_list")["enabled"], false);
    for tool in tools {
        assert!(tool["deferred"].is_boolean(), "{tool}");
    }
}
