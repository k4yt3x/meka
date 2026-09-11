// Every prompt here is answered by the scripted provider, which a release build has only with
// `mock-provider`; without it there is nothing to run.
#![cfg(any(debug_assertions, feature = "mock-provider"))]
// Integration-test files are their own crate, so the `#![cfg_attr(test, allow(...))]` in
// `src/main.rs` doesn't reach here. Mirror it explicitly: tests rely on `.unwrap()` / `.expect()`
// for clear panic-on-failure semantics, and asserting against panics is the standard idiom.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "tests panic on failure by design, and indexing a JSON document is the readable form"
)]

//! End-to-end ACP integration tests. Spawn the real `meka acp` binary with `MEKA_MOCK_PROVIDER=1`
//! so a scripted [`crate::provider::mock::MockProvider`] drives deterministic `session/prompt`
//! round-trips. Tests verify the tool-call lifecycle, permission round-trip, session lifecycle
//! (load / resume / list / close), slash-skill invocation, set_mode flow, and the `fs/*` delegation
//! path.
//!
//! # Test shape
//!
//! Tests should use [`AcpTestHarness`] (and its [`AcpTestHarnessBuilder`] for tests that need to
//! seed the tempdir before spawn). The harness collapses spawn / init / session/new boilerplate to
//! ~3 lines and the [`AcpTestHarnessBuilder::pre_spawn`] hook covers tests whose mock script must
//! reference an on-disk path inside the tempdir.
//!
//! A handful of tests use the inline `tempfile::tempdir + Command::spawn + stdin/stdout pipes +
//! read_until` shape because the harness contract can't model what they need:
//! - **Multi-spawn persistence tests** (`acp_session_load_replays_persisted_turn`,
//!   `acp_session_resume_adopts_without_replay`, `acp_session_list_filters_by_cwd`,
//!   `acp_session_list_paginates_across_cursor_boundary`) seed a second child process against the
//!   same on-disk session store the first child wrote to. The harness owns its tempdir and has no
//!   "respawn against this existing tempdir" hook.
//! - **Pre-initialize protocol tests** (`acp_initialize_clamps_far_future_version_to_latest`) need
//!   to send a non-default `protocolVersion`, which the harness bakes in during `build()`.
//! - **Bespoke timing tests** (`acp_session_cancel_interrupts_running_prompt`,
//!   `acp_multi_session_parallel_prompts_dont_serialize`) rely on precise `Instant::now()`
//!   measurements outside the harness's per-request window.
//!
//! Everything else (the tool-call lifecycle, permission flows, delegation paths, sub-agent
//! forwarding, and per-session isolation) sits on the harness.

use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, ChildStdin, Stdio},
    time::{Duration, Instant},
};

#[path = "harness/support.rs"]
mod support;

use support::Install;

/// Test harness that owns the child process, stdio pipes, and a per-request window. Wraps the spawn
/// / `initialize` / `session/new` boilerplate so each test stays focused on the behavior it
/// exercises. See the module header for the inline-pattern exceptions.
struct AcpTestHarness {
    install: Install,
    child: Child,
    stdin: ChildStdin,
    reader: support::TimedLines,
    /// Drained by the spawned reader thread and kept alive so that thread can finish cleanly.
    #[allow(dead_code, reason = "held so the reader thread can finish; never read")]
    stderr_handle: std::thread::JoinHandle<String>,
    next_id: u64,
    window: Duration,
}

/// Boxed pre-spawn hook. Type-aliased to keep the builder field declaration readable
/// (clippy::type_complexity).
type PreSpawnHook = Box<dyn FnOnce(&Path) -> serde_json::Value>;

/// Fluent builder for [`AcpTestHarness`]. Tests that need to pre-populate files inside the spawned
/// process's tempdir use [`Self::pre_spawn`] to run a closure with the resolved `config_dir`
/// *before* the child starts. The mock script can reference paths set up there.
#[allow(dead_code, reason = "not every test uses every builder field")]
#[derive(Default)]
struct AcpTestHarnessBuilder {
    config: String,
    script: Option<serde_json::Value>,
    capabilities: serde_json::Value,
    pre_spawn: Option<PreSpawnHook>,
    config_window: Option<Duration>,
}

#[allow(dead_code, reason = "not every test uses every builder method")]
impl AcpTestHarnessBuilder {
    fn config(mut self, toml: &str) -> Self {
        self.config = toml.to_string();
        self
    }

    fn script(mut self, value: serde_json::Value) -> Self {
        self.script = Some(value);
        self
    }

    fn script_opt(mut self, value: Option<serde_json::Value>) -> Self {
        self.script = value;
        self
    }

    fn capabilities(mut self, value: serde_json::Value) -> Self {
        self.capabilities = value;
        self
    }

    /// Run `f` with the resolved `config_dir` *before* spawn. The returned JSON value replaces the
    /// script (so the closure can reference real paths under the tempdir).
    fn pre_spawn<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&Path) -> serde_json::Value + 'static,
    {
        self.pre_spawn = Some(Box::new(f));
        self
    }

    /// Override the default 15s per-request window.
    fn window(mut self, duration: Duration) -> Self {
        self.config_window = Some(duration);
        self
    }

    fn build(self) -> AcpTestHarness {
        let install = Install::new();
        install.write_config(&self.config);

        let script = if let Some(f) = self.pre_spawn {
            f(&install.config_dir())
        } else {
            self.script
                .unwrap_or_else(|| serde_json::Value::Array(Vec::new()))
        };
        install.write_script(&script);

        let mut child = install
            .meka(&["acp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn meka acp");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let reader = support::TimedLines::spawn(stdout);
        let stderr_handle = support::drain(stderr_pipe);
        let window = self.config_window.unwrap_or(Duration::from_secs(15));
        let mut harness = AcpTestHarness {
            install,
            child,
            stdin,
            reader,
            stderr_handle,
            next_id: 0,
            window,
        };
        let _ = harness.request(
            "initialize",
            serde_json::json!({
                "protocolVersion": 1,
                "clientCapabilities": self.capabilities,
            }),
        );
        harness
    }
}

/// A fixture path in the harness's work directory, which sits beside `config_dir`. A file *under*
/// the config dir is inside meka's own directory, which `read_file` refuses below `unrestricted`.
/// The directory sessions from [`AcpTestHarness::new_session`] work in, from inside a `pre_spawn`
/// closure that is handed the config directory beside it. An approved write at `read` lands only
/// under the workspace roots, so a script that expects its write to succeed targets this.
fn work_dir_beside(config_dir: &Path) -> std::path::PathBuf {
    config_dir
        .parent()
        .expect("the config directory sits under the harness's temp root")
        .join("work")
}

fn fixture_beside(config_dir: &Path, name: &str) -> std::path::PathBuf {
    config_dir
        .parent()
        .expect("the config dir sits under the temp root")
        .join("work")
        .join(name)
}

#[allow(dead_code, reason = "not every test uses every helper")]
impl AcpTestHarness {
    /// Spin up `meka acp` against a fresh tempdir with `config_toml` pre-written and
    /// `MEKA_MOCK_PROVIDER` enabled (with an empty script unless `script` is supplied).
    /// Initialize the connection but don't create a session yet.
    fn spawn(config_toml: &str, script: Option<serde_json::Value>) -> Self {
        Self::spawn_with_capabilities(config_toml, script, serde_json::json!({}))
    }

    /// As [`Self::spawn`], but pass `client_capabilities` to the `initialize` handler. Tests that
    /// exercise the `fs.*` or `terminal` delegation paths flip the relevant capability bits here.
    fn spawn_with_capabilities(
        config_toml: &str,
        script: Option<serde_json::Value>,
        client_capabilities: serde_json::Value,
    ) -> Self {
        Self::builder()
            .config(config_toml)
            .script_opt(script)
            .capabilities(client_capabilities)
            .build()
    }

    fn builder() -> AcpTestHarnessBuilder {
        AcpTestHarnessBuilder::default()
    }

    fn config_dir(&self) -> std::path::PathBuf {
        self.install.config_dir()
    }

    /// The directory sessions from [`Self::new_session`] work in: a project directory, not meka's
    /// own config directory, which the write fence refuses at `workspace`.
    fn work_dir(&self) -> std::path::PathBuf {
        self.install.work_dir()
    }

    /// The store the spawned process is using, for a test that has to act as a *second* writer of a
    /// row meka reads. See `tests/multiprocess.rs` for the same reasoning at length.
    fn database(&self) -> std::path::PathBuf {
        self.install.database()
    }

    /// Send a JSON-RPC request and return the parsed response. Uses a monotonically increasing
    /// request id; tests don't need to pick ids themselves. Convenience wrapper over
    /// [`Self::send_request`] + [`Self::await_response`].
    fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.send_request(method, params);
        self.await_response(id)
    }

    /// Fire a JSON-RPC request and return the id without waiting. Use [`Self::await_response`] or
    /// [`Self::collect_updates`] to pick up the response later. Useful when a test fires a prompt,
    /// observes intermediate notifications, then collects the final response.
    fn send_request(&mut self, method: &str, params: serde_json::Value) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{request}").expect("write request");
        id
    }

    /// Fire a JSON-RPC notification (no id, no response). Used for `session/cancel`.
    fn notify(&mut self, method: &str, params: serde_json::Value) {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{notification}").expect("write notification");
    }

    /// Block until the response for `id` arrives. Side-channel notifications + meka-issued requests
    /// on the same connection are silently dropped (the latter is the right call only when the test
    /// isn't expected to provoke any).
    fn await_response(&mut self, id: u64) -> serde_json::Value {
        self.await_response_with_dispatch(id, |_| None)
    }

    /// Block until the response for `id` arrives, dispatching any meka-issued requests through
    /// `handler`. The handler returns `Some(response)` to answer, or `None` to ignore.
    fn await_response_with_dispatch<F>(&mut self, id: u64, mut handler: F) -> serde_json::Value
    where
        F: FnMut(&serde_json::Value) -> Option<serde_json::Value>,
    {
        let needle = format!("\"id\":{id}");
        let lines = read_until_with_dispatch(
            &mut self.reader,
            &mut self.stdin,
            Instant::now() + self.window,
            |value| handler(value),
            |line| response_matches(line, &needle),
        );
        let line = match lines.iter().find(|line| response_matches(line, &needle)) {
            Some(line) => line.clone(),
            None => {
                let collected = lines.join("");
                panic!("no response for id={id}; transcript:\n{collected}",);
            }
        };
        serde_json::from_str(&line).unwrap_or_else(|error| {
            panic!("response for id={id} was not JSON ({error}): {line}");
        })
    }

    /// Drain every `session/update` for `session_id` that meka has already emitted, with no prompt
    /// outstanding.
    ///
    /// Every other update helper is keyed to a pending `session/prompt`; a scheduled turn has none,
    /// which is the property under test. Termination comes from a `session/list` sent afterwards:
    /// stdio preserves order, so its response cannot arrive before the notifications queued ahead
    /// of it, and waiting on a reply meka is guaranteed to send means a regression fails the
    /// test instead of blocking the suite on a `read_line` that never returns.
    fn drain_unsolicited_updates(&mut self, session_id: &str) -> Vec<serde_json::Value> {
        let id = self.send_request("session/list", serde_json::json!({}));
        let needle = format!("\"id\":{id}");
        let session_id_owned = session_id.to_string();
        let lines = read_until_with_dispatch(
            &mut self.reader,
            &mut self.stdin,
            Instant::now() + self.window,
            |_| None,
            |line| response_matches(line, &needle),
        );
        lines
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|value| value["method"] == "session/update")
            .filter(|value| value["params"]["sessionId"] == session_id_owned)
            .collect()
    }

    /// Collect every `session/update` notification for `session_id` plus the eventual response for
    /// `id`. Captures everything the agent emits during a prompt turn, which is the single most
    /// reused pattern in the existing test file. Side-channel meka-issued requests are silently
    /// ignored; if a test expects them, use [`Self::collect_updates_with_dispatch`] instead.
    fn collect_updates(
        &mut self,
        session_id: &str,
        id: u64,
    ) -> (Vec<serde_json::Value>, serde_json::Value) {
        self.collect_updates_with_dispatch(session_id, id, |_| None)
    }

    /// As [`Self::collect_updates`], but dispatch meka-issued requests via `handler`. Used by tests
    /// that watch the session/update stream *and* answer fs / terminal delegation.
    ///
    /// The free [`read_until_with_dispatch`] only invokes its dispatch closure on JSON-RPC
    /// *requests* (those with both `method` and `id`). Notifications carry `method` but no `id`, so
    /// we can't piggy-back on it; drive a parallel loop here that also captures `session/update`
    /// notifications for the target `session_id`.
    fn collect_updates_with_dispatch<F>(
        &mut self,
        session_id: &str,
        id: u64,
        mut handler: F,
    ) -> (Vec<serde_json::Value>, serde_json::Value)
    where
        F: FnMut(&serde_json::Value) -> Option<serde_json::Value>,
    {
        let needle = format!("\"id\":{id}");
        let mut updates: Vec<serde_json::Value> = Vec::new();
        let mut response: Option<serde_json::Value> = None;
        let mut transcript = String::new();
        let deadline = Instant::now() + self.window;
        while let Some(line) = self.reader.next_line(deadline) {
            transcript.push_str(&line);
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            if value["method"] == "session/update"
                && value["params"]["sessionId"].as_str() == Some(session_id)
            {
                updates.push(value.clone());
                continue;
            }
            if value.get("method").is_some()
                && value.get("id").is_some()
                && let Some(reply) = handler(&value)
            {
                let _ = writeln!(self.stdin, "{reply}");
                continue;
            }
            if line.contains(&needle) && response_matches(&line, &needle) {
                response = Some(value);
                break;
            }
        }
        let response = response
            .unwrap_or_else(|| panic!("no response for id={id}; transcript:\n{transcript}",));
        (updates, response)
    }

    /// Create a session in `config_dir` and return its id. Most tests do this once at start of a
    /// scenario.
    fn new_session(&mut self) -> String {
        let cwd = self.install.work_dir();
        let response = self.request(
            "session/new",
            serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
        );
        response["result"]["sessionId"]
            .as_str()
            .unwrap_or_else(|| panic!("session/new did not return a sessionId: {response}"))
            .to_string()
    }

    /// Fire a `session/prompt` against `session_id` and return the request id. Pair with
    /// [`Self::collect_updates`] or [`Self::await_response`] to read the result.
    fn prompt(&mut self, session_id: &str, text: &str) -> u64 {
        self.send_request(
            "session/prompt",
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": text }],
            }),
        )
    }

    /// Fire a `session/cancel` notification for `session_id`.
    fn cancel(&mut self, session_id: &str) {
        self.notify(
            "session/cancel",
            serde_json::json!({ "sessionId": session_id }),
        );
    }

    /// One-shot `session/set_mode` round-trip.
    fn set_mode(&mut self, session_id: &str, mode: &str) -> serde_json::Value {
        self.request(
            "session/set_mode",
            serde_json::json!({
                "sessionId": session_id,
                "modeId": mode,
            }),
        )
    }

    /// One-shot `session/close` round-trip.
    fn close_session(&mut self, session_id: &str) -> serde_json::Value {
        self.request(
            "session/close",
            serde_json::json!({ "sessionId": session_id }),
        )
    }
}

/// Returns `true` when `line` is the response (or error) message for the given `id` needle. Used by
/// [`AcpTestHarness`] helpers that stop reading once the awaited response arrives. The secondary
/// check on `result` / `error` filters out incoming meka-issued *requests* that happen to share an
/// id with our response we're awaiting (the dispatch loop renumbers those, but
/// belt-and-suspenders).
fn response_matches(line: &str, needle: &str) -> bool {
    if !line.contains(needle) {
        return false;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
        return false;
    };
    value.get("result").is_some() || value.get("error").is_some()
}

impl Drop for AcpTestHarness {
    fn drop(&mut self) {
        // Best-effort cleanup; we already closed stdin or killed the child in `drain_stderr` for
        // failing tests, but a happy-path test that returns normally still needs the child reaped.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A deadline `seconds` from now, for one wait.
///
/// Each wait gets its own budget rather than a share of one. A single `Instant` computed at the top
/// of a test is a stopwatch, not a timeout: it is checked *before* each `read_line`, so once it
/// passes, every later read returns nothing instantly and the test fails with an empty transcript
/// naming a request the child was never given time to answer. On a loaded runner a slow child start
/// spent the whole budget during `initialize` and `session/new`, and the first real assertion of
/// nine different tests failed on Windows CI for a timeout none of them had actually reached.
fn window(seconds: u64) -> Instant {
    Instant::now() + Duration::from_secs(seconds)
}

/// Read lines until either `f` returns `true`, EOF, or the deadline elapses. Collects every line
/// read so test failures can dump the JSON-RPC stream for diagnosis.
///
/// The deadline bounds each read, not only the gaps between them: a child that goes silent
/// mid-request fails the test at the deadline instead of blocking the suite in `read_line`.
fn read_until<F>(reader: &mut support::TimedLines, deadline: Instant, mut f: F) -> Vec<String>
where
    F: FnMut(&str) -> bool,
{
    let mut lines = Vec::new();
    while let Some(line) = reader.next_line(deadline) {
        let stop = f(&line);
        lines.push(line);
        if stop {
            return lines;
        }
    }
    lines
}

/// Variant of [`read_until`] that also answers incoming JSON-RPC *requests* from meka. Any line
/// that parses to a JSON object with both a `method` and an `id` field is treated as a meka-issued
/// request; `dispatch` is invoked with the parsed value and its `Some(response)` return value is
/// written back to meka's stdin. Tests use this to play the client side of the `fs/*` and
/// `terminal/*` round-trips.
fn read_until_with_dispatch<W, D, F>(
    reader: &mut support::TimedLines,
    stdin: &mut W,
    deadline: Instant,
    mut dispatch: D,
    mut stop: F,
) -> Vec<String>
where
    W: Write,
    D: FnMut(&serde_json::Value) -> Option<serde_json::Value>,
    F: FnMut(&str) -> bool,
{
    let mut lines = Vec::new();
    while let Some(line) = reader.next_line(deadline) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line)
            && value.get("method").is_some()
            && value.get("id").is_some()
            && let Some(response) = dispatch(&value)
        {
            // Failure to write to stdin means the child is gone; surface it via the deadline loop
            // rather than panicking from inside the helper.
            let _ = writeln!(stdin, "{response}");
        }
        let should_stop = stop(&line);
        lines.push(line);
        if should_stop {
            return lines;
        }
    }
    lines
}

#[test]
fn acp_tool_call_lifecycle_round_trips_through_mock_provider() {
    // Fake config + credential so `create_agent_from_config` builds a real provider stack. The
    // mock swap inside `run_acp` then replaces the provider before any HTTP call is attempted.
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            // Target file for the scripted `read_file` call. Real tool runs against this path, so
            // it must exist. In the work directory beside the config dir: `read_file` refuses
            // meka's own directory below `unrestricted`.
            let target = fixture_beside(config_dir, "target.txt");
            std::fs::write(&target, "hello from mock test\n").expect("write target");
            serde_json::json!([
                [
                    { "type": "text", "text": "reading the file...\n" },
                    { "type": "tool_use_start", "id": "call_1", "name": "read_file" },
                    { "type": "tool_use_end", "input": { "path": target.to_str().unwrap() } },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "text", "text": "done!" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "read the target file");
    let (updates, response) = harness.collect_updates(&session_id, id);

    let saw_tool_call = updates.iter().any(|value| {
        let update = &value["params"]["update"];
        if update["sessionUpdate"] != "tool_call" {
            return false;
        }
        assert_eq!(
            update["kind"], "read",
            "expected tool kind 'read': {update}"
        );
        assert_eq!(
            update["status"], "in_progress",
            "expected tool_call status in_progress: {update}",
        );
        // The title carries the tool's name, then the resolved primary argument (the path): not
        // the bare name alone, and not a second vocabulary.
        let title = update["title"].as_str().unwrap_or("");
        assert!(
            title.starts_with("read_file ") && title.contains("target.txt"),
            "tool_call title should be 'read_file <path>': {update}",
        );
        true
    });
    assert!(
        saw_tool_call,
        "expected a session/update with sessionUpdate=tool_call; updates: {updates:?}",
    );
    assert!(
        updates.iter().any(|value| {
            let update = &value["params"]["update"];
            update["sessionUpdate"] == "tool_call_update" && update["status"] == "completed"
        }),
        "expected a tool_call_update with status=completed; updates: {updates:?}",
    );
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "expected stopReason=end_turn; full response: {response}",
    );
}

/// The `todo` tool surfaces as a `plan` session/update with one entry per item.
#[test]
fn acp_todo_tool_emits_plan_update() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|_dir| {
            serde_json::json!([
                [
                    { "type": "text", "text": "planning...\n" },
                    { "type": "tool_use_start", "id": "call_todo", "name": "todo" },
                    {
                        "type": "tool_use_end",
                        "input": { "title": "Work", "items": ["First", "Second"] }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "text", "text": "done" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "make a plan");
    let (updates, response) = harness.collect_updates(&session_id, id);

    let plan = updates
        .iter()
        .find(|value| value["params"]["update"]["sessionUpdate"] == "plan")
        .unwrap_or_else(|| panic!("expected a plan session/update; updates: {updates:?}"));
    let entries = plan["params"]["update"]["entries"]
        .as_array()
        .expect("plan entries array");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["content"], "First");
    assert_eq!(entries[0]["status"], "pending");
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

/// The first turn of a fresh session emits a `session_info_update` carrying the title (the first
/// user message preview: the words, not the agent's context block).
#[test]
fn acp_first_turn_emits_session_info_update_title() {
    let script = serde_json::json!([[
        { "type": "text", "text": "ok" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "explain the build system");
    let (updates, _response) = harness.collect_updates(&session_id, id);

    let info = updates
        .iter()
        .find(|value| value["params"]["update"]["sessionUpdate"] == "session_info_update")
        .unwrap_or_else(|| panic!("expected a session_info_update; updates: {updates:?}"));
    assert_eq!(
        info["params"]["update"]["title"],
        "explain the build system"
    );
}

/// A provider advisory reaches the editor as an assistant-message chunk prefixed `[meka]`.
///
/// ACP has no notice primitive, so this chunk is the *only* way an advisory reaches a client, and
/// the prefix is the only thing letting one filter or style it. The degrade-and-retry uses it to
/// say that content was removed from the turn, which makes this the difference between an editor
/// user being told and a tool result quietly changing under them.
///
/// Deleting the `FrontendEvent::Notice` arm that produces it left every suite green: it was the one
/// mutant the sweep over this changeset missed.
#[test]
fn acp_forwards_a_provider_notice_as_a_prefixed_assistant_chunk() {
    let script = serde_json::json!([[
        { "type": "notice", "message": "context is filling up" },
        { "type": "text", "text": "ok" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "hello");
    let (updates, _response) = harness.collect_updates(&session_id, id);

    let chunks: Vec<&str> = updates
        .iter()
        .filter(|value| value["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        .filter_map(|value| value["params"]["update"]["content"]["text"].as_str())
        .collect();
    assert!(
        chunks.contains(&"[meka] context is filling up"),
        "the advisory must reach the client, prefixed: {chunks:?}"
    );
}

/// `session/prompt` reports session-cumulative token usage on the response, alongside the
/// per-turn `usage_update` notification that carries the context gauge.
#[test]
fn acp_prompt_response_carries_token_usage() {
    let script = serde_json::json!([[
        { "type": "text", "text": "ok" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "hello");
    let (_updates, response) = harness.collect_updates(&session_id, id);

    let usage = &response["result"]["usage"];
    assert!(
        usage.is_object(),
        "expected usage on the prompt response; full response: {response}"
    );
    // Every counter is reported, including the cache tiers, rather than left off the wire. The
    // values are all zero because the mock provider emits no token-usage events by design (see
    // `provider::mock`), so this pins the shape and the wiring, not the arithmetic.
    for field in [
        "totalTokens",
        "inputTokens",
        "outputTokens",
        "cachedReadTokens",
        "cachedWriteTokens",
    ] {
        assert!(
            usage[field].is_u64(),
            "expected {field} on the usage object; usage: {usage}"
        );
    }
    // Omitted deliberately: meka doesn't meter reasoning separately from output.
    assert!(usage["thoughtTokens"].is_null(), "usage: {usage}");
}

/// A turn that reports `tool_use` but carries no tool-call block leaves the agent with nothing to
/// run and nothing to show. It happens for real: an OpenAI-compatible endpoint that coalesces its
/// final delta into the `finish_reason` chunk can have that delta dropped, so the tool call
/// vanishes and only the stop reason survives. The client must still get a visible message and a
/// terminal stop reason rather than a silent turn it waits on forever.
#[test]
fn acp_turn_with_tool_use_stop_but_no_tool_call_still_reports_to_the_client() {
    let script = serde_json::json!([[
        { "type": "message_end", "stop_reason": "tool_use" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "search the web");
    let (updates, response) = harness.collect_updates(&session_id, id);

    let chunk = updates
        .iter()
        .find(|value| value["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        .unwrap_or_else(|| {
            panic!("expected an agent_message_chunk standing in for the empty turn; updates: {updates:?}")
        });
    let text = chunk["params"]["update"]["content"]["text"]
        .as_str()
        .unwrap_or_default();
    assert!(
        text.contains("empty response"),
        "the stand-in notice must reach the client verbatim; got {text:?}"
    );

    // No tool call was made, so the turn ends normally. A client that saw `tool_use` with no
    // `tool_call` update and no terminal stop reason would sit waiting on a call that never came.
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "full response: {response}"
    );
    assert!(
        !updates
            .iter()
            .any(|value| value["params"]["update"]["sessionUpdate"] == "tool_call"),
        "no tool call was made, so none must be announced; updates: {updates:?}"
    );
}

/// Outcome the test wants the synthetic ACP client to send back when the agent issues
/// `session/request_permission`.
#[derive(Debug, Clone, Copy)]
enum PermissionAnswer {
    AllowOnce,
    RejectOnce,
}

/// An ACP session at `workspace` writes inside its cwd and is refused outside it.
///
/// The ACP surface had no `workspace` session at all: its three mentions of the word are about ACP
/// *workspace folders*, which are unrelated. So the fence could stop being applied to a session
/// `session/new` created and nothing on this surface would notice.
///
/// Asserts disk state rather than the tool-call status, because a tool that never ran and one the
/// fence refused report the same way.
#[test]
fn an_acp_session_at_workspace_is_fenced_to_its_cwd() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "workspace"
enabled = ["read", "workspace", "unrestricted"]
"#;
    let outside_dir = tempfile::tempdir().expect("tempdir");
    let outside = outside_dir.path().join("escaped.txt");
    let outside_for_script = outside.clone();

    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(move |config_dir| {
            // The session's cwd is the harness's work directory beside the config dir, so this one
            // is inside the boundary. Not the config dir itself: meka's own directories are
            // refused at `workspace` whatever the roots are.
            let inside = config_dir
                .parent()
                .expect("the config dir sits under the tempdir")
                .join("work")
                .join("inside.txt");
            serde_json::json!([
                [
                    { "type": "tool_use_start", "id": "call_in", "name": "write_file" },
                    {
                        "type": "tool_use_end",
                        "input": { "path": inside.to_str().unwrap(), "content": "in" }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "tool_use_start", "id": "call_out", "name": "write_file" },
                    {
                        "type": "tool_use_end",
                        "input": {
                            "path": outside_for_script.to_str().unwrap(),
                            "content": "out"
                        }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "text", "text": "done" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();

    let inside = harness.work_dir().join("inside.txt");
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "write both");
    let _ = harness.await_response(id);

    assert!(
        inside.exists(),
        "a write inside the session cwd must land at workspace"
    );
    assert_eq!(std::fs::read_to_string(&inside).expect("read back"), "in");
    assert!(
        !outside.exists(),
        "a write outside every root must be refused at workspace"
    );
}

/// Drive a full `meka acp` permission round-trip with the mock provider. The scripted turn calls
/// `write_file` (which, above `read` with approvals on, triggers a `session/request_permission`);
/// the test auto-responds with the configured outcome and asserts the resulting tool-call status.
fn run_permission_scenario(answer: PermissionAnswer) {
    // `read` with approvals on: a write is above the level, so it triggers the round-trip we want
    // to exercise.
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
approvals = true
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target = work_dir_beside(config_dir).join("out.txt");
            serde_json::json!([
                [
                    { "type": "text", "text": "writing the file...\n" },
                    { "type": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "type": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": "hello" }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "text", "text": "done!" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "write the file");

    let option_id = match answer {
        PermissionAnswer::AllowOnce => "allow_once",
        PermissionAnswer::RejectOnce => "reject_once",
    };
    let mut saw_permission_request = false;
    let (updates, response) = harness.collect_updates_with_dispatch(&session_id, id, |value| {
        match value["method"].as_str() {
            Some("session/request_permission") => {
                saw_permission_request = true;
                // What the editor is asked to approve: the tool's name, every argument (so the
                // write's content is on screen, not only its path), and sticky options that name
                // the tool in the same words as the title.
                let tool_call = &value["params"]["toolCall"];
                assert!(
                    tool_call["title"]
                        .as_str()
                        .is_some_and(|title| title.starts_with("write_file ")),
                    "the permission title opens with the tool's name: {value}"
                );
                assert_eq!(
                    tool_call["rawInput"]["content"], "hello",
                    "rawInput carries the call's arguments: {value}"
                );
                let content_text = tool_call["content"][0]["content"]["text"]
                    .as_str()
                    .unwrap_or_default();
                assert!(
                    content_text.contains("\"content\": \"hello\""),
                    "the content block shows the arguments to a client that renders content \
                     rather than rawInput: {value}"
                );
                let option_names: Vec<&str> = value["params"]["options"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|option| option["name"].as_str())
                    .collect();
                assert!(
                    option_names.contains(&"Always allow any write_file")
                        && option_names.contains(&"Always deny any write_file"),
                    "the sticky options name the tool the way the title does: {option_names:?}"
                );
                Some(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": value["id"].clone(),
                    "result": {
                        "outcome": { "outcome": "selected", "optionId": option_id }
                    }
                }))
            }
            _ => None,
        }
    });

    assert!(
        saw_permission_request,
        "expected a session/request_permission from the agent; updates: {updates:?}",
    );

    let status = updates
        .iter()
        .filter_map(|value| {
            let update = &value["params"]["update"];
            if update["sessionUpdate"] != "tool_call_update" {
                return None;
            }
            update["status"].as_str().map(str::to_string)
        })
        .next_back()
        .unwrap_or_else(|| {
            panic!("expected a tool_call_update with a status; updates: {updates:?}",)
        });
    match answer {
        PermissionAnswer::AllowOnce => assert_eq!(
            status, "completed",
            "allow_once should let write_file complete; updates: {updates:?}",
        ),
        PermissionAnswer::RejectOnce => assert_eq!(
            status, "failed",
            "reject_once should fail the tool call; updates: {updates:?}",
        ),
    }
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "expected stopReason=end_turn after permission outcome was handled; full response: {response}",
    );
}

#[test]
fn acp_permission_allow_once_runs_tool_and_completes_turn() {
    // A session with approvals on where the client answers `allow_once` must actually run the
    // gated tool.
    run_permission_scenario(PermissionAnswer::AllowOnce);
}

#[test]
fn acp_permission_reject_once_fails_tool_but_completes_turn() {
    run_permission_scenario(PermissionAnswer::RejectOnce);
}

/// a session that ran one prompt + tool round-trip is closed and then loaded by id; the load
/// handler must replay `user_message_chunk`, `agent_message_chunk`, `tool_call` (read kind), and a
/// `tool_call_update` with `status=completed` before responding with `LoadSessionResponse`.
#[test]
fn acp_session_load_replays_persisted_turn() {
    let install = Install::new();
    let config_dir = install.config_dir();

    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    install.write_config(config_toml);

    // Beside the config dir, not in it: `read_file` refuses meka's own directory below
    // `unrestricted`.
    let target = install.root().join("target.txt");
    std::fs::write(&target, "hello from reload test\n").expect("write target");

    let script = serde_json::json!([
        [
            { "type": "text", "text": "reading the file...\n" },
            { "type": "tool_use_start", "id": "call_1", "name": "read_file" },
            { "type": "tool_use_end", "input": { "path": target.to_str().unwrap() } },
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "done!" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    install.write_script(&script);

    // First run: drive one prompt to populate the session, capture sessionId, then exit cleanly.
    let session_id = {
        let mut child = install
            .meka(&["acp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn meka acp");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = support::TimedLines::spawn(stdout);
        let stderr_handle = std::thread::spawn(move || {
            let mut buffer = String::new();
            let mut stderr_reader = BufReader::new(stderr_pipe);
            while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
            buffer
        });

        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("initialize");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

        let new_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": config_dir, "mcpServers": [] }
        });
        writeln!(stdin, "{new_request}").expect("session/new");
        let new_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":2"));
        let new_line = new_lines
            .iter()
            .find(|line| line.contains("\"id\":2"))
            .expect("session/new response");
        let new_response: serde_json::Value =
            serde_json::from_str(new_line).expect("session/new JSON parses");
        let session_id = new_response["result"]["sessionId"]
            .as_str()
            .expect("sessionId is a string")
            .to_string();

        let prompt_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "read the target file" }]
            }
        });
        writeln!(stdin, "{prompt_request}").expect("write session/prompt");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":3"));

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        let _ = stderr_handle.join();
        session_id
    };

    // Second run: load the persisted session and assert the replay stream.
    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn meka acp #2");

    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = support::TimedLines::spawn(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("initialize");
    let init_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));
    // Confirm session-management capabilities were advertised.
    let init_response: serde_json::Value = serde_json::from_str(
        init_lines
            .iter()
            .find(|line| line.contains("\"id\":1"))
            .expect("init response"),
    )
    .expect("init parses");
    assert_eq!(
        init_response["result"]["agentCapabilities"]["loadSession"],
        true,
    );
    assert!(
        init_response["result"]["agentCapabilities"]["sessionCapabilities"]["list"].is_object(),
        "expected sessionCapabilities.list to be advertised; got: {init_response}",
    );

    let load_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "session/load",
        "params": {
            "sessionId": session_id,
            "cwd": config_dir,
            "mcpServers": []
        }
    });
    writeln!(stdin, "{load_request}").expect("session/load");
    let load_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":4"));

    let mut saw_user_chunk = false;
    let mut saw_agent_chunk = false;
    let mut saw_tool_call = false;
    let mut saw_tool_call_update_completed = false;
    let mut load_response: Option<serde_json::Value> = None;
    for line in &load_lines {
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(error) => panic!("stdout line is not valid JSON-RPC: {line} ({error})",),
        };
        if value["method"] == "session/update" {
            let update = &value["params"]["update"];
            match update["sessionUpdate"].as_str() {
                Some("user_message_chunk") => saw_user_chunk = true,
                Some("agent_message_chunk") => saw_agent_chunk = true,
                Some("tool_call") => {
                    assert_eq!(update["kind"], "read");
                    saw_tool_call = true;
                }
                Some("tool_call_update") if update["status"] == "completed" => {
                    saw_tool_call_update_completed = true;
                }
                _ => {}
            }
        }
        if value["id"] == 4 {
            load_response = Some(value);
        }
    }

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let dump = || load_lines.join("");

    assert!(
        saw_user_chunk,
        "replay must emit user_message_chunk; stream:\n{}\nSTDERR:\n{}",
        dump(),
        stderr_handle.join().unwrap_or_default(),
    );
    assert!(
        saw_agent_chunk,
        "replay must emit agent_message_chunk; stream:\n{}",
        dump()
    );
    assert!(
        saw_tool_call,
        "replay must emit tool_call; stream:\n{}",
        dump()
    );
    assert!(
        saw_tool_call_update_completed,
        "replay must emit tool_call_update completed; stream:\n{}",
        dump(),
    );

    let response =
        load_response.unwrap_or_else(|| panic!("no LoadSessionResponse; stream:\n{}", dump()));
    assert!(
        response["result"].is_object(),
        "expected an object result for session/load: {response}",
    );
}

/// Poll `child.try_wait()` until the process exits or `timeout` elapses. Returns the exit status,
/// or `None` on timeout (the caller decides whether that's a failure and is responsible for killing
/// the child).
fn wait_for_exit(child: &mut Child, timeout: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            _ => return None,
        }
    }
}

/// Regression: `meka acp` must exit (releasing its session `flock`) when the client disconnects
/// (stdin EOF), instead of lingering as an orphan that pins the lock. Run 1 takes the lock via
/// `session/new` + a prompt, then drops stdin WITHOUT `session/close` or `kill`; the process must
/// exit on its own. Run 2 then loads the same session from a fresh process and must succeed (not
/// `SessionLocked`), proving run 1 released the lock by exiting.
#[test]
fn acp_exits_and_releases_lock_on_stdin_eof() {
    let install = Install::new();
    let config_dir = install.config_dir();

    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    install.write_config(config_toml);

    let script = serde_json::json!([
        [
            { "type": "text", "text": "done!" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    install.write_script(&script);

    // Run 1: take the session lock, then disconnect by dropping stdin (no session/close, no kill).
    let session_id = {
        let mut child = install
            .meka(&["acp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn meka acp");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let stderr_handle = std::thread::spawn(move || {
            let mut buffer = String::new();
            let mut stderr_reader = BufReader::new(stderr_pipe);
            while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
            buffer
        });
        let mut reader = support::TimedLines::spawn(stdout);

        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("initialize");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

        let new_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": config_dir, "mcpServers": [] }
        });
        writeln!(stdin, "{new_request}").expect("session/new");
        let new_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":2"));
        let session_id = serde_json::from_str::<serde_json::Value>(
            new_lines
                .iter()
                .find(|line| line.contains("\"id\":2"))
                .expect("session/new response"),
        )
        .expect("parse")["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        // One prompt so the session lock is definitely held.
        let prompt_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": { "sessionId": session_id, "prompt": [{ "type": "text", "text": "hello" }] }
        });
        writeln!(stdin, "{prompt_request}").expect("session/prompt");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":3"));

        // The thread behind `reader` keeps draining stdout, so a shutdown-time write cannot block
        // the child; `reader` stays alive to the end of the test for that reason. Disconnect by
        // closing stdin. Crucially: NO `child.kill()` -- the process must exit itself.
        drop(stdin);

        let exited = wait_for_exit(&mut child, Duration::from_secs(10)).is_some();
        if !exited {
            let _ = child.kill();
            let _ = child.wait();
        }
        drop(reader);
        assert!(
            exited,
            "meka acp did not exit within 10s of stdin EOF (orphaned, lock still held).\nSTDERR:\n{}",
            stderr_handle.join().unwrap_or_default(),
        );
        session_id
    };

    // Run 2: a fresh process must be able to lock + load the same session.
    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn meka acp #2");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });
    let mut reader = support::TimedLines::spawn(stdout);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("initialize");
    let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

    let load_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "session/load",
        "params": { "sessionId": session_id, "cwd": config_dir, "mcpServers": [] }
    });
    writeln!(stdin, "{load_request}").expect("session/load");
    let load_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":4"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let response_line = load_lines
        .iter()
        .find(|line| line.contains("\"id\":4"))
        .unwrap_or_else(|| {
            panic!(
                "no session/load response; run 1 may not have released the lock.\nSTDERR:\n{}",
                stderr_handle.join().unwrap_or_default(),
            )
        });
    let response: serde_json::Value =
        serde_json::from_str(response_line).expect("parse session/load response");
    assert!(
        response.get("error").is_none(),
        "session/load must succeed after run 1 exited; got error (lock not released?): {response}",
    );
}

/// a `session/list` with a `cwd` filter must only return sessions whose persisted cwd matches.
#[test]
fn acp_session_list_filters_by_cwd() {
    let install = Install::new();

    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    install.write_config(config_toml);

    // Two distinct cwds that both physically exist (ACP server only stores the path; existence
    // doesn't matter for the filter, but tools later resolved against it would fail if absent).
    let cwd_a = install.root().join("proj-a");
    let cwd_b = install.root().join("proj-b");
    std::fs::create_dir_all(&cwd_a).expect("mkdir cwd_a");
    std::fs::create_dir_all(&cwd_b).expect("mkdir cwd_b");

    let script = serde_json::json!([
        [
            { "type": "text", "text": "ack" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    install.write_script(&script);

    // Helper: launch one `meka acp`, send initialize + session/new (with the given cwd) +
    // session/prompt, return the sessionId.
    let create_one = |session_cwd: &std::path::Path| -> String {
        let mut child = install
            .meka(&["acp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = support::TimedLines::spawn(stdout);
        let _stderr_handle = std::thread::spawn(move || {
            let mut buffer = String::new();
            let mut stderr_reader = BufReader::new(stderr_pipe);
            while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
            buffer
        });
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("init");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

        let new_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": session_cwd, "mcpServers": [] }
        });
        writeln!(stdin, "{new_request}").expect("session/new");
        let new_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":2"));
        let new_line = new_lines
            .iter()
            .find(|line| line.contains("\"id\":2"))
            .expect("session/new response");
        let new_response: serde_json::Value =
            serde_json::from_str(new_line).expect("parse session/new");
        let session_id = new_response["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        // Drive a no-op prompt so the session is persisted with a message (otherwise the title
        // would be empty; not required by the assertion but matches the realistic shape).
        let prompt_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "ping" }]
            }
        });
        writeln!(stdin, "{prompt_request}").expect("prompt");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":3"));

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        session_id
    };

    let id_a = create_one(&cwd_a);
    let _id_b = create_one(&cwd_b);

    // Second invocation issues session/list filtered to cwd_a.
    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn list child");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = support::TimedLines::spawn(stdout);
    let _stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

    let list_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "session/list",
        "params": { "cwd": cwd_a }
    });
    writeln!(stdin, "{list_req}").expect("session/list");
    let list_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":5"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let list_line = list_lines
        .iter()
        .find(|line| line.contains("\"id\":5"))
        .expect("session/list response");
    let list_response: serde_json::Value =
        serde_json::from_str(list_line).expect("parse session/list");
    let sessions = list_response["result"]["sessions"]
        .as_array()
        .expect("sessions array");
    assert_eq!(
        sessions.len(),
        1,
        "expected exactly one session matching cwd_a; got: {list_response}",
    );
    assert_eq!(sessions[0]["sessionId"], id_a);
}

/// `session/resume` adopts an existing session id without replaying. The handler should not emit
/// any `session/update` notifications, but should leave the slot populated so subsequent prompts
/// can proceed (smoke test: a follow-up prompt succeeds).
#[test]
fn acp_session_resume_adopts_without_replay() {
    let install = Install::new();
    let config_dir = install.config_dir();

    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    install.write_config(config_toml);

    // The script must serve two prompts: one for the first run, one for the follow-up after resume.
    let script = serde_json::json!([
        [
            { "type": "text", "text": "first response" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "follow up" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    install.write_script(&script);

    // First run: create a session, run one prompt, capture the id.
    let session_id = {
        let mut child = install
            .meka(&["acp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn first");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = support::TimedLines::spawn(stdout);
        let _stderr_handle = std::thread::spawn(move || {
            let mut buffer = String::new();
            let mut stderr_reader = BufReader::new(stderr_pipe);
            while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
            buffer
        });
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("init");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

        let new_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": config_dir, "mcpServers": [] }
        });
        writeln!(stdin, "{new_request}").expect("session/new");
        let new_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":2"));
        let session_id = serde_json::from_str::<serde_json::Value>(
            new_lines
                .iter()
                .find(|line| line.contains("\"id\":2"))
                .expect("session/new response"),
        )
        .expect("parse")["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        let prompt_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "first" }]
            }
        });
        writeln!(stdin, "{prompt_request}").expect("first prompt");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":3"));

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        session_id
    };

    // Second run: resume + a follow-up prompt.
    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn resume");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = support::TimedLines::spawn(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

    let resume_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "session/resume",
        "params": {
            "sessionId": session_id,
            "cwd": config_dir,
            "mcpServers": []
        }
    });
    writeln!(stdin, "{resume_req}").expect("session/resume");
    let resume_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":6"));

    // The `available_commands_update` push is allowed (and expected) on resume. What must NOT
    // appear is a replay update: `user_message_chunk`, `agent_message_chunk`, `tool_call`, or
    // `tool_call_update`.
    let mut saw_replay_update = false;
    let mut resume_response: Option<serde_json::Value> = None;
    for line in &resume_lines {
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if value["method"] == "session/update" {
            let kind = value["params"]["update"]["sessionUpdate"]
                .as_str()
                .unwrap_or_default();
            if matches!(
                kind,
                "user_message_chunk"
                    | "agent_message_chunk"
                    | "agent_thought_chunk"
                    | "tool_call"
                    | "tool_call_update"
            ) {
                saw_replay_update = true;
            }
        }
        if value["id"] == 6 {
            resume_response = Some(value);
        }
    }
    assert!(
        !saw_replay_update,
        "session/resume must NOT emit replay updates; stream:\n{}",
        resume_lines.join(""),
    );
    let resume_response = resume_response.unwrap_or_else(|| {
        panic!(
            "no ResumeSessionResponse; stream:\n{}",
            resume_lines.join(""),
        )
    });
    assert!(
        resume_response["result"].is_object(),
        "resume must succeed: {resume_response}",
    );

    // Follow-up prompt confirms the slot is active.
    let prompt_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "session/prompt",
        "params": {
            "sessionId": session_id,
            "prompt": [{ "type": "text", "text": "follow up" }]
        }
    });
    writeln!(stdin, "{prompt_request}").expect("follow-up prompt");
    let prompt_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":7"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let prompt_response: serde_json::Value = serde_json::from_str(
        prompt_lines
            .iter()
            .find(|line| line.contains("\"id\":7"))
            .unwrap_or_else(|| {
                panic!(
                    "no follow-up prompt response; STDERR:\n{}",
                    stderr_handle.join().unwrap_or_default(),
                )
            }),
    )
    .expect("parse follow-up");
    assert_eq!(prompt_response["result"]["stopReason"], "end_turn");
}

/// `session/close` clears the active slot so a subsequent `session/new` succeeds within the same
/// process.
#[test]
fn acp_session_close_clears_slot_for_subsequent_new() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    let first_id = harness.new_session();

    // Second session/new without close: must succeed (multi-session ACP now allows N concurrent
    // sessions per process).
    let second_id = harness.new_session();
    assert_ne!(
        second_id, first_id,
        "second session/new must mint a fresh sessionId",
    );

    // Close the first session.
    let close_response = harness.close_session(&first_id);
    assert!(
        close_response["result"].is_object(),
        "expected ok result for session/close: {close_response}",
    );

    // Re-closing the first session must error; it's gone.
    let reclose = harness.close_session(&first_id);
    assert!(
        reclose["error"].is_object(),
        "re-closing a removed session must error: {reclose}",
    );
}

/// a skill installed under `$MEKA_CONFIG_DIR/skills/` shows up in the `available_commands_update`
/// push that follows `session/new`, AND the `NewSessionResponse` carries the configured mode
/// picker.
#[test]
fn acp_session_new_advertises_skills_and_modes() {
    // Provider stub + a non-default enabled set so we can assert exactly which modes get
    // advertised.
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let skill_dir = config_dir.join("skills").join("demo-skill");
            std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
            std::fs::write(
                skill_dir.join("SKILL.md"),
                "---\ndescription: a demo skill\n---\ndo stuff\n",
            )
            .expect("write SKILL.md");
            // Empty script; we never run a turn.
            serde_json::json!([])
        })
        .build();

    // session/new fires before any session/update notifications, so we can't filter notifications
    // by session_id up front. Send the request manually, then walk the stream picking up the
    // intermediate `available_commands_update` notification(s) and the eventual response
    // together.
    let cwd = harness.config_dir();
    let id = harness.send_request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    let needle = format!("\"id\":{id}");
    let mut saw_skill = false;
    let mut new_response: Option<serde_json::Value> = None;
    let mut transcript = String::new();
    let deadline = Instant::now() + harness.window;
    while Instant::now() < deadline {
        let mut line = String::new();
        match harness.reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        transcript.push_str(&line);
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if value["method"] == "session/update" {
            let update = &value["params"]["update"];
            if update["sessionUpdate"] == "available_commands_update"
                && let Some(cmds) = update["availableCommands"].as_array()
                && cmds.iter().any(|c| c["name"] == "demo-skill")
            {
                saw_skill = true;
            }
        }
        if line.contains(&needle) && response_matches(&line, &needle) {
            new_response = Some(value);
            break;
        }
    }
    assert!(
        saw_skill,
        "expected available_commands_update with demo-skill; transcript:\n{transcript}",
    );
    let response = new_response.unwrap_or_else(|| {
        panic!("no session/new response; transcript:\n{transcript}");
    });
    let modes = &response["result"]["modes"];
    let ids: Vec<String> = modes["availableModes"]
        .as_array()
        .expect("availableModes")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(ids, vec!["read", "unrestricted"]);
    assert_eq!(modes["currentModeId"], "read");
}

/// a `session/prompt` whose text is `/<skill-name>` resolves to the rendered skill body before
/// being handed to the agent. Asserts the prompt completes successfully (the alternative, the
/// helper returning an error from an unknown skill, would surface as a JSON-RPC error response
/// instead of `stopReason=end_turn`).
#[test]
fn acp_session_prompt_invokes_skill_by_slash_name() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let skill_dir = config_dir.join("skills").join("hello");
            std::fs::create_dir_all(&skill_dir).expect("mkdir");
            std::fs::write(
                skill_dir.join("SKILL.md"),
                "---\ndescription: say hi\n---\nrespond with a greeting\n",
            )
            .expect("write SKILL.md");
            serde_json::json!([[
                { "type": "text", "text": "hello from agent" },
                { "type": "message_end", "stop_reason": "end_turn" }
            ]])
        })
        .build();
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "/hello but be brief");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "skill invocation should run a normal turn: {response}",
    );
}

/// A `session/prompt` whose first text token *looks* like a skill invocation but doesn't match an
/// installed skill must pass through to the model rather than erroring. The original rejection
/// broke paste UX: pasted text like `/usr local lib` or the start of a sentence like `/etc and so
/// on` would be parsed as `name="usr"` / `name="etc"`, validated as a syntactically-OK skill name,
/// then rejected with `InvalidParams "unknown skill"`. The model can respond with "I don't know
/// that command" if the user genuinely meant `/skill-name`.
#[test]
fn acp_session_prompt_passes_through_unknown_skill_name() {
    let script = serde_json::json!([[
        { "type": "text", "text": "ok" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "/unknown-skill but otherwise valid text");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "unknown skill name must pass through to the model, not error: {response}",
    );
}

/// A `/<skill>` invocation that resolves to an installed skill but whose body file becomes
/// unreadable between scan and invocation must surface as JSON-RPC `InternalError` (-32603), not
/// `InvalidParams` (-32602). The client's request was syntactically valid and named a real
/// skill; the failure is a server-side disk problem.
///
/// We trigger this by chmod-ing SKILL.md to 0 after the skills cache has registered it. The cache's
/// disk-snapshot key is the file's mtime (see `disk_snapshot` in `src/skills.rs`), which `chmod`
/// doesn't touch, so the cache happily serves the stale entry while `load_skill_body` fails with
/// EACCES.
#[cfg(unix)]
#[test]
fn acp_session_prompt_skill_body_unreadable_is_internal_error() {
    use std::os::unix::fs::PermissionsExt;

    let mut harness = AcpTestHarness::builder()
        .config(ACP_INVALID_PARAMS_CONFIG)
        .pre_spawn(|config_dir| {
            let skill_dir = config_dir.join("skills").join("doomed");
            std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
            std::fs::write(
                skill_dir.join("SKILL.md"),
                "---\ndescription: will become unreadable\n---\nbody\n",
            )
            .expect("write SKILL.md");
            serde_json::json!([])
        })
        .build();
    let session_id = harness.new_session();
    let skill_md = harness
        .config_dir()
        .join("skills")
        .join("doomed")
        .join("SKILL.md");
    // chmod 0 after the in-process cache has already scanned the file during `session/new`. Since
    // mtime is unchanged, the cache's snapshot-equality check serves the stale skill list and
    // load_skill_body hits EACCES.
    std::fs::set_permissions(&skill_md, std::fs::Permissions::from_mode(0o000))
        .expect("chmod 0 SKILL.md");

    let id = harness.prompt(&session_id, "/doomed");
    let response = harness.await_response(id);

    // Restore permissions before assertions so a panic doesn't break tempdir cleanup.
    let _ = std::fs::set_permissions(&skill_md, std::fs::Permissions::from_mode(0o644));

    let error = response["error"]
        .as_object()
        .unwrap_or_else(|| panic!("expected JSON-RPC error: {response}"));
    assert_eq!(
        error["code"].as_i64(),
        Some(-32603),
        "skill body load failure must map to InternalError (-32603), not InvalidParams; got: {response}",
    );
    let data = error["data"]
        .as_str()
        .unwrap_or_else(|| panic!("expected error.data to carry the detail string: {response}"));
    assert!(
        data.contains("failed to load skill 'doomed'"),
        "error.data should mention the doomed skill name and load failure; got: {data}",
    );
}

/// `session/set_mode` flips the active permission level and emits `current_mode_update`. A request
/// for a mode outside the enabled set returns a JSON-RPC error.
#[test]
fn acp_session_set_mode_flips_permission_and_emits_update() {
    const CONFIG: &str = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "workspace"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);

    // Confirm advertised modes match the enabled set.
    let new_response = harness.request(
        "session/new",
        serde_json::json!({
            "cwd": harness.config_dir(),
            "mcpServers": []
        }),
    );
    let ids: Vec<String> = new_response["result"]["modes"]["availableModes"]
        .as_array()
        .expect("availableModes")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(ids, vec!["read", "workspace"]);
    let session_id = new_response["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();

    // Valid set_mode: read → workspace. The `current_mode_update` notification arrives via the same
    // session/update channel, which `request` discards; collect it via a small ad-hoc loop instead
    // by issuing the request and watching for the notification before the response.
    let set_id = harness.send_request(
        "session/set_mode",
        serde_json::json!({ "sessionId": session_id, "modeId": "workspace" }),
    );
    let (updates, set_response) = harness.collect_updates(&session_id, set_id);
    assert!(
        updates.iter().any(|u| {
            u["params"]["update"]["sessionUpdate"] == "current_mode_update"
                && u["params"]["update"]["currentModeId"] == "workspace"
        }),
        "expected current_mode_update with currentModeId=workspace; updates: {updates:?}",
    );
    assert!(
        set_response["result"].is_object(),
        "set_mode must succeed: {set_response}",
    );

    // Invalid set_mode: unrestricted is not in the enabled set.
    let bad_response = harness.set_mode(&session_id, "unrestricted");
    assert!(
        bad_response["error"].is_object(),
        "set_mode for a disabled mode must error: {bad_response}",
    );
}

/// when the client advertises `fs.read_text_file`, a `read_file` tool call delegates to
/// `fs/read_text_file` rather than touching the disk. The mock provider scripts the tool use; the
/// test harness intercepts the outgoing fs request and answers with canned content.
#[test]
fn acp_fs_read_text_file_is_delegated_when_capability_offered() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .capabilities(serde_json::json!({
            "fs": { "readTextFile": true, "writeTextFile": false },
            "terminal": false
        }))
        .pre_spawn(|config_dir| {
            // Real on-disk file with one content; the delegate returns *different* content, so the
            // assertion proves the delegate path was used.
            let target = fixture_beside(config_dir, "delegated.txt");
            std::fs::write(&target, "ON DISK\n").expect("write target");
            serde_json::json!([
                [
                    { "type": "text", "text": "reading..." },
                    { "type": "tool_use_start", "id": "call_read", "name": "read_file" },
                    { "type": "tool_use_end", "input": { "path": target.to_str().unwrap() } },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "text", "text": "done" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let target = harness.work_dir().join("delegated.txt");
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "read it");

    let mut saw_fs_read_request = false;
    let _ = harness.await_response_with_dispatch(id, |value| {
        if value["method"] == "fs/read_text_file" {
            saw_fs_read_request = true;
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": { "content": "FROM EDITOR BUFFER\n" }
            }))
        } else {
            None
        }
    });

    assert!(
        saw_fs_read_request,
        "expected a fs/read_text_file request from meka",
    );

    // The on-disk file is untouched (we only wrote `ON DISK` once before the test). The delegate
    // returned different content, proving the tool used the delegate result rather than reading the
    // disk.
    assert_eq!(std::fs::read_to_string(&target).expect("read"), "ON DISK\n");
}

/// when the client advertises `fs.write_text_file`, a `write_file` tool call delegates to
/// `fs/write_text_file` and does NOT touch the local disk. The test harness intercepts the request,
/// replies ok, and asserts no local file was created.
#[test]
fn acp_fs_write_text_file_is_delegated_when_capability_offered() {
    let content_to_write = "hello from delegated write";
    // `unrestricted` so the agent's permission gate doesn't refuse `write_file` before we even
    // reach the delegation seam.
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .capabilities(serde_json::json!({
            "fs": { "readTextFile": true, "writeTextFile": true },
            "terminal": false
        }))
        .pre_spawn(move |config_dir| {
            let target = config_dir.join("delegated-write.txt");
            serde_json::json!([
                [
                    { "type": "text", "text": "writing..." },
                    { "type": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "type": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": content_to_write }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "text", "text": "done" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    // `write_file` canonicalizes the parent directory before handing the path to the delegate, so
    // the expected path matches `/private/var/...` on macOS rather than the `/var/...` tempdir
    // returns from `config_dir()`. Stripped of the `\\?\` prefix the way meka strips it, because
    // this is compared against a path meka reports rather than one the test constructs.
    // `canonicalize` alone yields the verbatim spelling, which is the one meka never emits, so the
    // mismatch is Windows-only.
    let target_dir = {
        let canonical = std::fs::canonicalize(harness.config_dir()).expect("canonicalize tempdir");
        let text = canonical.to_string_lossy();
        match text.strip_prefix(r"\\?\") {
            Some(stripped) => std::path::PathBuf::from(stripped),
            None => canonical,
        }
    };
    let target = target_dir.join("delegated-write.txt");
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "write it");

    let mut saw_fs_write = false;
    let mut delegated_path: Option<String> = None;
    let mut delegated_content: Option<String> = None;
    let _ = harness.await_response_with_dispatch(id, |value| match value["method"].as_str() {
        Some("fs/read_text_file") => {
            // Pre-read for diff metadata: return file-not-found shaped error so write_file falls
            // back to None old_text.
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "error": { "code": -32603, "message": "file not open" }
            }))
        }
        Some("fs/write_text_file") => {
            saw_fs_write = true;
            delegated_path = value["params"]["path"].as_str().map(String::from);
            delegated_content = value["params"]["content"].as_str().map(String::from);
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": {}
            }))
        }
        _ => None,
    });

    assert!(saw_fs_write, "expected a fs/write_text_file request");
    assert_eq!(
        delegated_path.as_deref(),
        Some(target.to_str().unwrap()),
        "delegate received wrong path"
    );
    assert_eq!(
        delegated_content.as_deref(),
        Some(content_to_write),
        "delegate received wrong content"
    );
    // No local file on disk; the delegate handled the write.
    assert!(
        !target.exists(),
        "meka wrote a local file despite delegating: {}",
        target.display(),
    );
}

/// when the client does NOT advertise `fs.write_text_file`, `write_file` falls back to a local disk
/// write. No `fs/write_text_file` request should appear.
#[test]
fn acp_write_file_falls_back_to_local_when_no_capability() {
    let content_to_write = "wrote locally";
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]
"#;
    // No capabilities advertised; the harness default is `{}`.
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(move |config_dir| {
            let target = config_dir.join("local-write.txt");
            serde_json::json!([
                [
                    { "type": "text", "text": "writing..." },
                    { "type": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "type": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": content_to_write }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "text", "text": "done" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let target = harness.config_dir().join("local-write.txt");
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "write it");

    let mut saw_fs_write = false;
    let _ = harness.await_response_with_dispatch(id, |value| {
        if value["method"] == "fs/write_text_file" {
            saw_fs_write = true;
        }
        None
    });

    assert!(
        !saw_fs_write,
        "fs/write_text_file must NOT be issued without capability"
    );
    let written = std::fs::read_to_string(&target).expect("local write should have happened");
    assert_eq!(written, content_to_write);
}

/// The capabilities Zed advertises, in the shape that matters here: `terminal` says it implements
/// `terminal/*` requests, and the separate `_meta.terminal_output` key says it renders agent-owned
/// terminals from meka's `_meta` frames. Only the latter gates the terminal rendering.
fn zed_shaped_capabilities() -> serde_json::Value {
    serde_json::json!({
        "terminal": true,
        "_meta": { "terminal_output": true },
    })
}

/// `execute_command` always runs in meka's own (sandboxed) child process, never in the client's
/// terminal, whatever the permission level and whatever the client advertises. `unrestricted` is
/// the level where delegating would be tempting, and `read` is the level where it would be a
/// sandbox bypass. Guards both by asserting no `terminal/*` traffic and that the output is the
/// local shell's, not the client's.
#[test]
fn acp_execute_command_never_leaves_meka() {
    for level in ["unrestricted", "read"] {
        let config_toml = format!(
            r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "{level}"
enabled = ["read", "unrestricted"]
"#
        );
        let script = serde_json::json!([
            [
                { "type": "text", "text": "running..." },
                { "type": "tool_use_start", "id": "call_exec", "name": "execute_command" },
                { "type": "tool_use_end", "input": { "command": "echo ran-inside-meka" } },
                { "type": "message_end", "stop_reason": "tool_use" }
            ],
            [
                { "type": "text", "text": "done" },
                { "type": "message_end", "stop_reason": "end_turn" }
            ]
        ]);
        let mut harness = AcpTestHarness::spawn_with_capabilities(
            &config_toml,
            Some(script),
            zed_shaped_capabilities(),
        );
        let session_id = harness.new_session();
        let id = harness.prompt(&session_id, "run it");

        let mut terminal_methods: Vec<String> = Vec::new();
        let (updates, _response) =
            harness.collect_updates_with_dispatch(&session_id, id, |value| {
                if let Some(method) = value["method"].as_str()
                    && method.starts_with("terminal/")
                {
                    terminal_methods.push(method.to_string());
                }
                // Approve anything the agent asks about, so the tool actually runs.
                if value["method"] == "session/request_permission" {
                    return Some(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": value["id"].clone(),
                        "result": { "outcome": { "outcome": "selected", "optionId": "allow_once" } }
                    }));
                }
                None
            });

        assert!(
            terminal_methods.is_empty(),
            "{level} level must run execute_command inside meka; saw {terminal_methods:?}",
        );
        // Either terminal status will do. Whether the command *succeeds* is a property of the
        // host's sandbox, not of this test: on a runner where the platform sandbox rejects the
        // profile, a run at `read` legitimately reports `failed`. What must hold everywhere
        // is that the call reached a terminal state without leaving meka.
        let status = updates
            .iter()
            .map(|u| &u["params"]["update"])
            .filter(|u| u["sessionUpdate"] == "tool_call_update")
            .filter_map(|u| u["status"].as_str())
            .next_back()
            .unwrap_or_else(|| {
                panic!("{level}: no tool_call_update carried a status: {updates:#?}")
            });
        assert!(
            status == "completed" || status == "failed",
            "{level}: unexpected terminal status {status:?}",
        );

        // When it did run, the output must be meka's own shell rather than a delegated one.
        if status == "completed" {
            let streamed: String = updates
                .iter()
                .filter_map(|u| {
                    u["params"]["update"]["_meta"]["terminal_output"]["data"]
                        .as_str()
                        .map(str::to_string)
                })
                .collect();
            assert!(
                streamed.contains("ran-inside-meka"),
                "{level}: expected the local shell's output; got {streamed:?}",
            );
        }
    }
}

/// `unrestricted`, so `execute_command` is not subject to the sandbox's availability on the test
/// host.
const ACP_UNRESTRICTED_CONFIG: &str = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]
"#;

/// A command's output reaches the client while it is still running, not only when it exits. The
/// scripted command prints, sleeps past the throttle interval, then prints again, so a correct
/// implementation emits at least one `tool_call_update` carrying `first` but not yet `second`.
/// Before live output existed, the only update for a tool call was the one at completion, and a
/// long build showed a bare spinner for its whole duration.
#[test]
fn acp_execute_command_streams_output_while_running() {
    let script = serde_json::json!([
        [
            { "type": "text", "text": "running..." },
            { "type": "tool_use_start", "id": "call_exec", "name": "execute_command" },
            {
                "type": "tool_use_end",
                "input": { "command": "echo first; sleep 0.5; echo second" }
            },
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_UNRESTRICTED_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "run it");
    let (updates, response) = harness.collect_updates_with_dispatch(&session_id, id, |_value| None);

    assert_eq!(response["result"]["stopReason"], "end_turn");

    let exec_updates: Vec<&serde_json::Value> = updates
        .iter()
        .map(|u| &u["params"]["update"])
        .filter(|u| u["sessionUpdate"] == "tool_call_update" && u["toolCallId"] == "call_exec")
        .collect();

    let partial = exec_updates.iter().find(|u| {
        let rendered = u["content"].to_string();
        rendered.contains("first") && !rendered.contains("second")
    });
    assert!(
        partial.is_some(),
        "expected an in-progress update carrying only the first line; got: {exec_updates:#?}",
    );
    assert!(
        partial.expect("checked above")["status"].is_null(),
        "a live-output update must not claim the call finished",
    );

    let completed = exec_updates
        .iter()
        .find(|u| u["status"] == "completed")
        .unwrap_or_else(|| panic!("expected a completed update; got: {exec_updates:#?}"));
    let rendered = completed["content"].to_string();
    assert!(
        rendered.contains("first") && rendered.contains("second"),
        "the final update must carry the whole output; got {rendered}",
    );
}

/// A terminal-capable client gets the agent-owned terminal channel: the call announces a terminal
/// so the client can register it, each update carries only the *new* bytes (the client owns the
/// scrollback, so re-sending would duplicate), and the call ends with a real exit status.
///
/// This is what makes output visible in Zed. A `kind: execute` tool call whose content is a text
/// block renders with no expansion affordance at all, so the output is sent but never shown.
#[test]
fn acp_terminal_capable_client_gets_an_agent_owned_terminal() {
    let script = serde_json::json!([
        [
            { "type": "text", "text": "running..." },
            { "type": "tool_use_start", "id": "call_exec", "name": "execute_command" },
            {
                "type": "tool_use_end",
                "input": { "command": "echo alpha; sleep 0.4; echo beta; exit 7" }
            },
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn_with_capabilities(
        ACP_UNRESTRICTED_CONFIG,
        Some(script),
        zed_shaped_capabilities(),
    );
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "run it");
    let (updates, _response) =
        harness.collect_updates_with_dispatch(&session_id, id, |_value| None);

    let for_call = |kind: &str| -> Vec<serde_json::Value> {
        updates
            .iter()
            .map(|u| &u["params"]["update"])
            .filter(|u| u["toolCallId"] == "call_exec")
            .filter(|u| !u["_meta"][kind].is_null())
            .cloned()
            .collect()
    };

    // The terminal has to be announced before anything references it; without a matching create,
    // clients park the output against an id they were never told about and show an empty terminal.
    let announced = for_call("terminal_info");
    assert_eq!(
        announced.len(),
        1,
        "expected exactly one terminal_info; got {announced:#?}",
    );
    assert_eq!(announced[0]["sessionUpdate"], "tool_call");
    assert_eq!(announced[0]["content"][0]["type"], "terminal");
    assert_eq!(announced[0]["content"][0]["terminalId"], "call_exec");

    // Output is appended, so no chunk may repeat what an earlier one already delivered.
    let output_frames = for_call("terminal_output");
    let chunks: Vec<&str> = output_frames
        .iter()
        .filter_map(|u| u["_meta"]["terminal_output"]["data"].as_str())
        .collect();
    assert!(
        chunks.len() >= 2,
        "expected the output to arrive in pieces while the command ran; got {chunks:?}",
    );
    assert_eq!(
        // The shell is PowerShell on Windows and `sh` elsewhere, and the two disagree about line
        // endings. What is being asserted is that the appended chunks reassemble to the command's
        // output exactly once, which is independent of that.
        chunks.concat().replace("\r\n", "\n"),
        "alpha\nbeta\n",
        "appended chunks must reassemble to the command's output exactly once",
    );
    assert!(
        chunks.iter().all(|chunk| !chunk.is_empty()),
        "an empty append is a wasted notification; got {chunks:?}",
    );

    // The real exit code, not a synthesized 1: a terminal shows the status, and 7 is only
    // available because the shell tool reports it structurally.
    let exits = for_call("terminal_exit");
    assert_eq!(exits.len(), 1, "expected exactly one terminal_exit");
    assert_eq!(exits[0]["_meta"]["terminal_exit"]["exit_code"], 7);
    assert_eq!(exits[0]["status"], "failed");
    assert_eq!(
        exits[0]["content"][0]["type"], "terminal",
        "the completed call must keep the terminal, not swap in a flattened copy",
    );
}

/// `terminal: true` alone must not trigger terminal rendering. That capability means "I implement
/// `terminal/*` requests", which is about running commands *in the client*; a client can offer it
/// and have no idea what meka's `_meta` terminal frames mean. Sending a terminal content block to
/// such a client resolves to nothing and displays no output at all, which is the exact failure the
/// terminal path exists to fix. Only `_meta.terminal_output` licenses it.
#[test]
fn acp_terminal_capability_alone_does_not_enable_terminal_rendering() {
    let script = serde_json::json!([
        [
            { "type": "tool_use_start", "id": "call_exec", "name": "execute_command" },
            { "type": "tool_use_end", "input": { "command": "echo capability-check" } },
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn_with_capabilities(
        ACP_UNRESTRICTED_CONFIG,
        Some(script),
        // `terminal/*` implemented, agent-owned terminal frames not understood.
        serde_json::json!({ "terminal": true }),
    );
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "run it");
    let (updates, _response) =
        harness.collect_updates_with_dispatch(&session_id, id, |_value| None);

    let call_updates: Vec<&serde_json::Value> = updates
        .iter()
        .map(|u| &u["params"]["update"])
        .filter(|u| u["toolCallId"] == "call_exec")
        .collect();
    assert!(
        call_updates
            .iter()
            .all(|u| u["content"][0]["type"] != "terminal"),
        "a terminal block here would render as nothing; got {call_updates:#?}",
    );
    assert!(
        call_updates.iter().any(|u| {
            u["content"][0]["content"]["text"]
                .as_str()
                .is_some_and(|text| text.contains("capability-check"))
        }),
        "expected the console text fallback; got {call_updates:#?}",
    );
}

/// A client that never advertised `terminal` keeps the text rendering, because a terminal block it
/// cannot resolve renders as nothing at all.
#[test]
fn acp_client_without_terminal_capability_gets_console_text() {
    let script = serde_json::json!([
        [
            { "type": "tool_use_start", "id": "call_exec", "name": "execute_command" },
            { "type": "tool_use_end", "input": { "command": "echo plain-text-path" } },
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn_with_capabilities(
        ACP_UNRESTRICTED_CONFIG,
        Some(script),
        serde_json::json!({ "terminal": false }),
    );
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "run it");
    let (updates, _response) =
        harness.collect_updates_with_dispatch(&session_id, id, |_value| None);

    let call_updates: Vec<&serde_json::Value> = updates
        .iter()
        .map(|u| &u["params"]["update"])
        .filter(|u| u["toolCallId"] == "call_exec")
        .collect();
    assert!(
        call_updates.iter().all(|u| u["_meta"].is_null()),
        "no terminal _meta may be sent to a client that can't resolve a terminal; got \
         {call_updates:#?}",
    );
    assert!(
        call_updates
            .iter()
            .all(|u| u["content"][0]["type"] != "terminal"),
        "no terminal content block either; got {call_updates:#?}",
    );
    let completed = call_updates
        .iter()
        .find(|u| u["status"] == "completed")
        .unwrap_or_else(|| panic!("expected a completed update; got {call_updates:#?}"));
    assert!(
        completed["content"][0]["content"]["text"]
            .as_str()
            .is_some_and(|text| text.contains("plain-text-path")),
        "expected a console text block carrying the output; got {completed:#?}",
    );
}

/// Build a JSON-RPC error response for a request the dispatch closure wants to fail.
fn jsonrpc_error(id: serde_json::Value, message: &str) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": -32603, "message": message }
    })
}

/// Spec: `session/cancel` MUST resolve the in-flight prompt with `stopReason: "cancelled"`. The
/// mock provider stalls mid-turn via a `Sleep` event so the test can fire the cancel notification
/// while the agent loop is parked inside `provider.stream`. Regression guard for the bug where
/// `Mutex<ServerState>` was held across `agent.run_turn().await`, which serialized the cancel
/// notification behind the prompt and made cancellation effectively useless.
#[test]
fn acp_session_cancel_interrupts_running_prompt() {
    let install = Install::new();
    let config_dir = install.config_dir();

    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    install.write_config(config_toml);

    // Round 1: a short "starting" delta so the test knows the turn started, then a 5s sleep that
    // races against cancel. If cancel arrives in time the mock returns early; the agent loop's
    // post-stream cancellation check breaks with Interrupted. If cancel is starved (bug regressed),
    // the sleep finishes, the text "done" + end_turn fires, and the response carries `end_turn`.
    // The assertion below catches that.
    let script = serde_json::json!([
        [
            { "type": "text", "text": "starting..." },
            { "type": "sleep", "ms": 5000 },
            { "type": "text", "text": "done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    install.write_script(&script);

    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = support::TimedLines::spawn(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

    let new_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "session/new",
        "params": { "cwd": config_dir, "mcpServers": [] }
    });
    writeln!(stdin, "{new_request}").expect("session/new");
    let new_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":2"));
    let session_id = serde_json::from_str::<serde_json::Value>(
        new_lines
            .iter()
            .find(|line| line.contains("\"id\":2"))
            .expect("session/new"),
    )
    .expect("parse")["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();

    let prompt_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "session/prompt",
        "params": {
            "sessionId": session_id,
            "prompt": [{ "type": "text", "text": "stall then cancel" }]
        }
    });
    writeln!(stdin, "{prompt_request}").expect("prompt");

    // Wait until we've seen the "starting..." chunk so we know the turn is actually parked inside
    // the mock's sleep; firing cancel any earlier might race the prompt setup.
    let start_deadline = Instant::now() + Duration::from_secs(5);
    let _ = read_until(&mut reader, start_deadline, |line| {
        line.contains("starting...")
    });

    // Fire session/cancel. If the bug regressed, this notification would queue behind the state
    // mutex; the assertion below would see stopReason=end_turn after the 5s sleep finishes.
    let cancel_notif = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "session/cancel",
        "params": { "sessionId": session_id }
    });
    writeln!(stdin, "{cancel_notif}").expect("cancel");

    // Tight deadline: cancel should resolve well before the 5s sleep would have completed
    // naturally. Allow generous slack for CI variance but well short of 5s.
    let response_deadline = Instant::now() + Duration::from_secs(3);
    let lines = read_until(&mut reader, response_deadline, |line| {
        line.contains("\"id\":3")
    });

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let response_line = lines
        .iter()
        .find(|line| line.contains("\"id\":3"))
        .unwrap_or_else(|| {
            panic!(
                "no PromptResponse before deadline, cancel was likely starved.\nSTDERR:\n{}\nstream:\n{}",
                stderr_handle.join().unwrap_or_default(),
                lines.join(""),
            )
        });
    let response: serde_json::Value =
        serde_json::from_str(response_line).expect("parse PromptResponse");
    assert_eq!(
        response["result"]["stopReason"], "cancelled",
        "session/cancel must resolve the in-flight prompt with canceled; got: {response}",
    );
}

/// Regression: a turn interrupted mid-stream must persist the partial assistant text so it survives
/// resume. Previously the partial was appended only in memory and discarded on exit, so resume
/// showed only the user prompt. Round 1 streams a partial answer then stalls in a `sleep`; the test
/// fires `session/cancel` once the partial has streamed. Round 2 loads the session and asserts the
/// replay carries the partial answer (and not the post-interrupt text, which never streamed).
#[test]
fn acp_interrupted_turn_persists_partial_assistant_text() {
    let install = Install::new();
    let config_dir = install.config_dir();

    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    install.write_config(config_toml);

    // A partial answer streams, then the turn stalls in a 5s sleep that races cancellation. The
    // text after the sleep must never stream once cancel fires, and so must never be persisted.
    let script = serde_json::json!([
        [
            { "type": "text", "text": "partial answer before interrupt" },
            { "type": "sleep", "ms": 5000 },
            { "type": "text", "text": "TEXT-AFTER-INTERRUPT-MUST-NOT-PERSIST" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    install.write_script(&script);

    // Round 1: prompt, wait for the partial to stream, fire cancel, capture sessionId, exit
    // cleanly.
    let session_id = {
        let mut child = install
            .meka(&["acp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn meka acp");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = support::TimedLines::spawn(stdout);
        let stderr_handle = std::thread::spawn(move || {
            let mut buffer = String::new();
            let mut stderr_reader = BufReader::new(stderr_pipe);
            while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
            buffer
        });

        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("initialize");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

        let new_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": config_dir, "mcpServers": [] }
        });
        writeln!(stdin, "{new_request}").expect("session/new");
        let new_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":2"));
        let session_id = serde_json::from_str::<serde_json::Value>(
            new_lines
                .iter()
                .find(|line| line.contains("\"id\":2"))
                .expect("session/new response"),
        )
        .expect("parse")["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        let prompt_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "interrupt me mid-turn" }]
            }
        });
        writeln!(stdin, "{prompt_request}").expect("session/prompt");

        // Wait until the partial answer has streamed so the agent has it buffered, then cancel.
        let start_deadline = Instant::now() + Duration::from_secs(5);
        let _ = read_until(&mut reader, start_deadline, |line| {
            line.contains("partial answer before interrupt")
        });
        let cancel_notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "session/cancel",
            "params": { "sessionId": session_id }
        });
        writeln!(stdin, "{cancel_notif}").expect("cancel");

        // The prompt should resolve (canceled) well before the 5s sleep would finish.
        let response_deadline = Instant::now() + Duration::from_secs(3);
        let lines = read_until(&mut reader, response_deadline, |line| {
            line.contains("\"id\":3")
        });
        let response: serde_json::Value = serde_json::from_str(
            lines
                .iter()
                .find(|line| line.contains("\"id\":3"))
                .unwrap_or_else(|| {
                    panic!(
                        "no PromptResponse before deadline; cancel was likely starved.\nSTDERR:\n{}",
                        stderr_handle.join().unwrap_or_default(),
                    )
                }),
        )
        .expect("parse PromptResponse");
        assert_eq!(
            response["result"]["stopReason"], "cancelled",
            "prompt must resolve as canceled; got: {response}",
        );

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        session_id
    };

    // Round 2: load the persisted session; the replay must carry the partial answer.
    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn meka acp #2");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = support::TimedLines::spawn(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("initialize");
    let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

    let load_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "session/load",
        "params": {
            "sessionId": session_id,
            "cwd": config_dir,
            "mcpServers": []
        }
    });
    writeln!(stdin, "{load_request}").expect("session/load");
    let load_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":4"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let replay = load_lines.join("");
    assert!(
        replay.contains("partial answer before interrupt"),
        "session/load replay must carry the interrupted turn's partial answer; stream:\n{}\nSTDERR:\n{}",
        replay,
        stderr_handle.join().unwrap_or_default(),
    );
    assert!(
        !replay.contains("TEXT-AFTER-INTERRUPT-MUST-NOT-PERSIST"),
        "post-interrupt text must not be persisted; stream:\n{replay}",
    );
}

/// when the client advertises both `fs.readTextFile` and `fs.writeTextFile`, `edit_file` delegates
/// both halves and does not touch the local disk: the delegated read+write composition.
#[test]
fn acp_edit_file_delegates_when_both_fs_capabilities_offered() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]
"#;
    // edit_file canonicalizes the target before reaching the delegation seam, so the path must
    // exist on disk. Seed it with a *different* content from what the delegate will serve; this
    // proves the editor's in-buffer view (via fs/read_text_file) wins over the on-disk bytes.
    // `force=true` skips the read-before-edit gate (we're not testing that path).
    let disk_content = "this should NOT be edited\n";
    let editor_content = "alpha\nbeta\n";
    let expected_new_content = "alpha\nGAMMA\n";

    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .capabilities(serde_json::json!({
            "fs": { "readTextFile": true, "writeTextFile": true }
        }))
        .pre_spawn(move |config_dir| {
            let target = config_dir.join("delegated-edit.txt");
            std::fs::write(&target, disk_content).expect("seed local file");
            serde_json::json!([
                [
                    { "type": "text", "text": "editing..." },
                    { "type": "tool_use_start", "id": "call_edit", "name": "edit_file" },
                    {
                        "type": "tool_use_end",
                        "input": {
                            "path": target.to_str().unwrap(),
                            "old_string": "beta",
                            "new_string": "GAMMA",
                            "force": true
                        }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "text", "text": "done" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let target = harness.config_dir().join("delegated-edit.txt");
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "edit it");

    let mut saw_fs_read = false;
    let mut saw_fs_write = false;
    let mut delegated_content: Option<String> = None;
    let _ = harness.await_response_with_dispatch(id, |value| match value["method"].as_str() {
        Some("fs/read_text_file") => {
            saw_fs_read = true;
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": { "content": editor_content }
            }))
        }
        Some("fs/write_text_file") => {
            saw_fs_write = true;
            delegated_content = value["params"]["content"].as_str().map(String::from);
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": {}
            }))
        }
        _ => None,
    });

    assert!(saw_fs_read, "edit_file must delegate the read half");
    assert!(saw_fs_write, "edit_file must delegate the write half");
    assert_eq!(
        delegated_content.as_deref(),
        Some(expected_new_content),
        "delegated write content didn't reflect the edit \
         applied to the editor's in-buffer view"
    );
    // On-disk content must be untouched; delegation bypassed the local filesystem.
    let on_disk = std::fs::read_to_string(&target).expect("read seeded file");
    assert_eq!(
        on_disk, disk_content,
        "edit_file modified the local file despite delegating to fs/write_text_file"
    );
}

/// a sub-agent's permission prompt must forward through `PermissionForwardingFrontend` to the
/// parent's ACP connection. The parent triggers `agent_spawn`; the sub-agent runs in `ask` mode
/// (inherited) and attempts `write_file`, which fires a `session/request_permission` on the
/// *parent's* connection. Test answers `allow_once` and asserts the request was observed.
#[test]
fn acp_subagent_permission_forwards_to_parent_client() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
approvals = true
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target = work_dir_beside(config_dir).join("subagent-write.txt");
            // The mock provider is shared between parent and sub-agent (same `Arc<dyn Provider>`),
            // so rounds drain in the order they're consumed: parent → sub-agent → sub-agent →
            // parent.
            serde_json::json!([
                // Parent round 1: spawn the sub-agent.
                [
                    { "type": "text", "text": "spawning sub-agent..." },
                    { "type": "tool_use_start", "id": "call_spawn", "name": "agent_spawn" },
                    {
                        "type": "tool_use_end",
                        "input": { "prompt": "write the file" }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                // Sub-agent round 1: write_file → triggers permission.
                [
                    { "type": "text", "text": "writing..." },
                    { "type": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "type": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": "subagent wrote me" }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                // Sub-agent round 2: final report.
                [
                    { "type": "text", "text": "wrote the file" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ],
                // Parent round 2: final report.
                [
                    { "type": "text", "text": "sub-agent finished" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .window(Duration::from_secs(30))
        .build();
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "delegate the write");

    let mut saw_permission_request = false;
    let (_updates, _response) = harness.collect_updates_with_dispatch(&session_id, id, |value| {
        match value["method"].as_str() {
            Some("session/request_permission") => {
                saw_permission_request = true;
                Some(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": value["id"].clone(),
                    "result": {
                        "outcome": { "outcome": "selected", "optionId": "allow_once" }
                    }
                }))
            }
            _ => None,
        }
    });

    assert!(
        saw_permission_request,
        "sub-agent's write_file must forward a session/request_permission \
         through the parent connection",
    );
}

/// `session/list` paginates with an opaque cursor. Seed PAGE_SIZE + a few sessions in the same cwd;
/// first call must return a `nextCursor`; passing that cursor back must return the remaining rows;
/// both pages combined must equal the seeded set.
#[test]
fn acp_session_list_paginates_across_cursor_boundary() {
    let install = Install::new();

    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    install.write_config(config_toml);

    let cwd = install.root().join("proj");
    std::fs::create_dir_all(&cwd).expect("mkdir cwd");

    let script = serde_json::json!([
        [
            { "type": "text", "text": "ack" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    install.write_script(&script);

    // `acp::handle_list_sessions` uses PAGE_SIZE = 50. Seed PAGE_SIZE + 3 sessions so the second
    // page is non-empty but small enough to keep the test fast.
    const PAGE_SIZE: usize = 50;
    const TOTAL: usize = PAGE_SIZE + 3;

    let create_one = || -> String {
        let mut child = install
            .meka(&["acp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr_pipe = child.stderr.take().expect("stderr");
        let mut reader = support::TimedLines::spawn(stdout);
        let _stderr_handle = std::thread::spawn(move || {
            let mut buffer = String::new();
            let mut stderr_reader = BufReader::new(stderr_pipe);
            while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
            buffer
        });

        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
        )
        .expect("init");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":1"));

        let new_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": cwd.clone(), "mcpServers": [] }
        });
        writeln!(stdin, "{new_request}").expect("session/new");
        let new_lines = read_until(&mut reader, window(15), |line| line.contains("\"id\":2"));
        let session_id = serde_json::from_str::<serde_json::Value>(
            new_lines
                .iter()
                .find(|line| line.contains("\"id\":2"))
                .expect("session/new"),
        )
        .expect("parse")["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        // One trivial prompt so the session has a row to surface.
        let prompt_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "ping" }]
            }
        });
        writeln!(stdin, "{prompt_request}").expect("prompt");
        let _ = read_until(&mut reader, window(15), |line| line.contains("\"id\":3"));

        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        session_id
    };

    let mut seeded: std::collections::HashSet<String> = std::collections::HashSet::new();
    for _ in 0..TOTAL {
        seeded.insert(create_one());
    }
    assert_eq!(seeded.len(), TOTAL, "test seeded duplicate session ids");

    // Now drive two session/list calls. The first returns the first page + a cursor; the second
    // uses the cursor to fetch the remainder.
    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn list child");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = support::TimedLines::spawn(stdout);
    let _stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, window(30), |line| line.contains("\"id\":1"));

    let list_req_a = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "session/list",
        "params": { "cwd": cwd.clone() }
    });
    writeln!(stdin, "{list_req_a}").expect("list page 1");
    let lines_a = read_until(&mut reader, window(30), |line| line.contains("\"id\":5"));
    let line_a = lines_a
        .iter()
        .find(|line| line.contains("\"id\":5"))
        .expect("list page 1 response");
    let response_a: serde_json::Value = serde_json::from_str(line_a).expect("parse");
    let sessions_a = response_a["result"]["sessions"]
        .as_array()
        .expect("sessions array")
        .clone();
    let cursor = response_a["result"]["nextCursor"]
        .as_str()
        .expect("page 1 must carry a nextCursor for TOTAL > PAGE_SIZE")
        .to_string();
    assert_eq!(
        sessions_a.len(),
        PAGE_SIZE,
        "page 1 should be exactly PAGE_SIZE rows"
    );

    let list_req_b = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "session/list",
        "params": { "cwd": cwd.clone(), "cursor": cursor }
    });
    writeln!(stdin, "{list_req_b}").expect("list page 2");
    let lines_b = read_until(&mut reader, window(30), |line| line.contains("\"id\":6"));
    let line_b = lines_b
        .iter()
        .find(|line| line.contains("\"id\":6"))
        .expect("list page 2 response");
    let response_b: serde_json::Value = serde_json::from_str(line_b).expect("parse");
    let sessions_b = response_b["result"]["sessions"]
        .as_array()
        .expect("sessions array")
        .clone();

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    assert_eq!(
        sessions_b.len(),
        TOTAL - PAGE_SIZE,
        "page 2 should be the remaining rows"
    );

    // Both pages combined must equal the seeded set (no dropped rows, no duplicates between pages).
    let mut combined: std::collections::HashSet<String> = std::collections::HashSet::new();
    for entry in sessions_a.iter().chain(sessions_b.iter()) {
        if let Some(id) = entry["sessionId"].as_str() {
            combined.insert(id.to_string());
        }
    }
    assert_eq!(
        combined, seeded,
        "paginated set must equal seeded set; missing or duplicated rows"
    );
}

/// `allow_always` sticks: a subsequent `session/prompt` that hits the same write tool MUST NOT
/// trigger another `session/request_permission` round-trip. Regression guard for the sticky-allow
/// store. Two consecutive prompts in one session.
#[test]
fn acp_permission_allow_always_skips_second_prompt() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
approvals = true
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target_a = work_dir_beside(config_dir).join("a.txt");
            let target_b = work_dir_beside(config_dir).join("b.txt");
            // Two complete turns; both invoke write_file. Only the first should provoke a
            // permission round-trip.
            serde_json::json!([
                // Turn 1 round 1.
                [
                    { "type": "text", "text": "writing a..." },
                    { "type": "tool_use_start", "id": "call_a", "name": "write_file" },
                    {
                        "type": "tool_use_end",
                        "input": { "path": target_a.to_str().unwrap(), "content": "a" }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                // Turn 1 round 2.
                [
                    { "type": "text", "text": "done a" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ],
                // Turn 2 round 1.
                [
                    { "type": "text", "text": "writing b..." },
                    { "type": "tool_use_start", "id": "call_b", "name": "write_file" },
                    {
                        "type": "tool_use_end",
                        "input": { "path": target_b.to_str().unwrap(), "content": "b" }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                // Turn 2 round 2.
                [
                    { "type": "text", "text": "done b" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .window(Duration::from_secs(30))
        .build();
    let session_id = harness.new_session();

    // Turn 1: write_file → request_permission (allow_always).
    let id_1 = harness.prompt(&session_id, "write a");
    let mut prompts_for_turn_1 = 0_usize;
    let _ = harness.await_response_with_dispatch(id_1, |value| {
        if value["method"] == "session/request_permission" {
            prompts_for_turn_1 += 1;
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": {
                    "outcome": { "outcome": "selected", "optionId": "allow_always" }
                }
            }))
        } else {
            None
        }
    });

    // Turn 2: same tool; sticky allow must suppress the round-trip.
    let id_2 = harness.prompt(&session_id, "write b");
    let mut prompts_for_turn_2 = 0_usize;
    let _ = harness.await_response_with_dispatch(id_2, |value| {
        if value["method"] == "session/request_permission" {
            prompts_for_turn_2 += 1;
            // Defensive answer just in case, but the assertion below catches the sticky-allow
            // regression.
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": {
                    "outcome": { "outcome": "selected", "optionId": "allow_once" }
                }
            }))
        } else {
            None
        }
    });

    assert_eq!(
        prompts_for_turn_1, 1,
        "turn 1 should have triggered exactly one permission round-trip",
    );
    assert_eq!(
        prompts_for_turn_2, 0,
        "turn 2 should have skipped permission entirely (allow_always sticky)",
    );
}

// `acp_initialize_negotiates_unknown_protocol_version` was superseded by
// `acp_initialize_clamps_far_future_version_to_latest` (below); stricter: asserts the clamp lands
// on `ProtocolVersion::LATEST`, not just "some number".

/// Two `session/new` calls succeed and produce independent session ids. Each session has its own
/// conversation; prompts route to the right one via `session/update`'s `sessionId` field.
#[test]
fn acp_multi_session_create_and_isolate_messages() {
    let script = serde_json::json!([
        [
            { "type": "text", "text": "A says hello" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "B says hello" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid_a = harness.new_session();
    let sid_b = harness.new_session();
    assert_ne!(sid_a, sid_b, "second session/new must mint a distinct id",);

    for (session_id, expected_text) in [(&sid_a, "A says hello"), (&sid_b, "B says hello")] {
        let id = harness.prompt(session_id, "go");
        let (updates, _) = harness.collect_updates(session_id, id);
        let saw_correct_chunk = updates.iter().any(|u| {
            u["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
                && u["params"]["update"]["content"]["text"].as_str() == Some(expected_text)
        });
        assert!(
            saw_correct_chunk,
            "session {session_id} did not receive its expected agent_message_chunk; updates: {updates:?}",
        );
    }
}

/// Two sessions prompting in parallel: A stalls in a long sleep, B completes a fast prompt. With
/// multi-session ACP, B's response must arrive *well before* A's, proving the per-session mutex
/// design lets sessions parallelize (the single-session `Mutex<ServerState>` would have serialized
/// them).
#[test]
fn acp_multi_session_parallel_prompts_dont_serialize() {
    let install = Install::new();
    let config_dir = install.config_dir();

    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    install.write_config(config_toml);

    // Windows CI workers have noticeably slower stdio IPC, so give A a longer stall and B a more
    // generous threshold while keeping the parallelism check (A:B ratio still ≥ 2:1).
    let (a_stall_ms, b_threshold) = if cfg!(target_os = "windows") {
        (10_000_u64, Duration::from_secs(5))
    } else {
        (4_000_u64, Duration::from_secs(2))
    };

    // Two rounds, in the order they'll be drained:
    //   1. Session A's prompt: long sleep then "A done".
    //   2. Session B's prompt: short response.
    let script = serde_json::json!([
        [
            { "type": "text", "text": "A starting" },
            { "type": "sleep", "ms": a_stall_ms },
            { "type": "text", "text": "A done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "B done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    install.write_script(&script);

    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = support::TimedLines::spawn(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });
    let deadline = Instant::now() + Duration::from_millis(a_stall_ms + 10_000);

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#,
    )
    .expect("init");
    let _ = read_until(&mut reader, deadline, |line| line.contains("\"id\":1"));

    // Open two sessions.
    let mut session_ids = Vec::new();
    for id in [2, 3] {
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/new",
            "params": { "cwd": config_dir.clone(), "mcpServers": [] }
        });
        writeln!(stdin, "{req}").expect("session/new");
        let needle = format!("\"id\":{id}");
        let lines = read_until(&mut reader, deadline, |line| line.contains(&needle));
        let line = lines
            .iter()
            .find(|line| line.contains(&needle))
            .expect("session/new response");
        let response: serde_json::Value = serde_json::from_str(line).expect("parse");
        session_ids.push(
            response["result"]["sessionId"]
                .as_str()
                .expect("sessionId")
                .to_string(),
        );
    }

    // Fire prompt A (will stall in 4s sleep), then immediately prompt B (should complete fast).
    let test_start = Instant::now();
    let prompt_a = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 100,
        "method": "session/prompt",
        "params": {
            "sessionId": session_ids[0].clone(),
            "prompt": [{ "type": "text", "text": "go A" }]
        }
    });
    writeln!(stdin, "{prompt_a}").expect("prompt A");

    // Wait for A's "A starting" delta to surface before firing B, so we know A holds the runtime
    // mutex and isn't merely queued. A blind `sleep(300ms)` was the previous approach, but it was
    // flake-prone on loaded CI; the deterministic marker mirrors what
    // `acp_session_cancel_interrupts_running_prompt` already does.
    let barrier = Instant::now() + Duration::from_secs(5);
    let _ = read_until(&mut reader, barrier, |line| line.contains("A starting"));

    let prompt_b = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 101,
        "method": "session/prompt",
        "params": {
            "sessionId": session_ids[1].clone(),
            "prompt": [{ "type": "text", "text": "go B" }]
        }
    });
    writeln!(stdin, "{prompt_b}").expect("prompt B");

    // Read responses for both. Track when each id is observed.
    let mut a_finish: Option<Duration> = None;
    let mut b_finish: Option<Duration> = None;
    while a_finish.is_none() || b_finish.is_none() {
        if Instant::now() > deadline {
            break;
        }
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            if value["id"] == 100 {
                a_finish = Some(test_start.elapsed());
            } else if value["id"] == 101 {
                b_finish = Some(test_start.elapsed());
            }
        }
    }

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let a = a_finish.expect("session A never responded");
    let b = b_finish.expect("session B never responded");

    // B must finish *substantially* before A. A is stalled, so B should return well within the
    // threshold. If the design serialized B behind A, B would take ≥ a_stall_ms.
    assert!(
        b < b_threshold,
        "session B took {:?} (threshold {:?}), looks serialized behind A's {}ms stall;\nSTDERR:\n{}",
        b,
        b_threshold,
        a_stall_ms,
        stderr_handle.join().unwrap_or_default(),
    );
    assert!(
        a > b,
        "session A finished before B ({a:?} vs {b:?}), script ordering wrong?",
    );
}

/// Per-session permission cells: setting mode on session A doesn't leak to session B. Sessions use
/// the same builtin permission space but their `SharedPermission` cells are independent
/// per-session, so a `session/set_mode` on A only affects A.
#[test]
fn acp_multi_session_set_mode_isolated() {
    const CONFIG: &str = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);
    let sid_a = harness.new_session();
    let sid_b = harness.new_session();

    // Use `collect_updates` against sid_a to gather A's notifications during the set_mode
    // round-trip. Notifications for sid_b appear in the same stream; collect a second pass
    // afterwards by inspecting the raw transcript to be sure none leaked.
    let set_id = harness.send_request(
        "session/set_mode",
        serde_json::json!({ "sessionId": sid_a, "modeId": "unrestricted" }),
    );
    // Track session-id of every current_mode_update we observe by inline-collecting alongside the
    // response.
    let sid_a_owned = sid_a;
    let sid_b_owned = sid_b;
    let mut saw_a_update_on_a = false;
    let mut saw_a_update_on_b = false;
    let needle = format!("\"id\":{set_id}");
    let deadline = Instant::now() + harness.window;
    while Instant::now() < deadline {
        let mut line = String::new();
        match harness.reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line)
            && value["method"] == "session/update"
            && value["params"]["update"]["sessionUpdate"] == "current_mode_update"
        {
            match value["params"]["sessionId"].as_str() {
                Some(s) if s == sid_a_owned => saw_a_update_on_a = true,
                Some(s) if s == sid_b_owned => saw_a_update_on_b = true,
                _ => {}
            }
        }
        if line.contains(&needle) && response_matches(&line, &needle) {
            break;
        }
    }
    assert!(
        saw_a_update_on_a,
        "session A must receive current_mode_update for its own set_mode",
    );
    assert!(
        !saw_a_update_on_b,
        "session B must NOT receive A's current_mode_update, modes are per-session",
    );
}

/// `session/cancel` fires only the target session's token. Session A stalls, B prompts normally. We
/// cancel A; A resolves with `cancelled` while B continues to `end_turn`.
#[test]
fn acp_multi_session_cancel_fires_only_target_session() {
    let script = serde_json::json!([
        // Session A: stall 5s. Cancel arrives before sleep ends → canceled.
        [
            { "type": "text", "text": "A stalling" },
            { "type": "sleep", "ms": 5000 },
            { "type": "text", "text": "A done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        // Session B: short response.
        [
            { "type": "text", "text": "B done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarnessBuilder::default()
        .config(ACP_INVALID_PARAMS_CONFIG)
        .script(script)
        .window(Duration::from_secs(20))
        .build();
    let sid_a = harness.new_session();
    let sid_b = harness.new_session();

    let id_a = harness.prompt(&sid_a, "stall");
    // Wait for A to actually start streaming before firing cancel.
    let start_deadline = Instant::now() + Duration::from_secs(3);
    let _ = read_until(&mut harness.reader, start_deadline, |line| {
        line.contains("A stalling")
    });
    harness.cancel(&sid_a);
    let id_b = harness.prompt(&sid_b, "go");

    // Poll for both responses arriving in any order.
    let mut a_stop: Option<String> = None;
    let mut b_stop: Option<String> = None;
    let deadline = Instant::now() + harness.window;
    while a_stop.is_none() || b_stop.is_none() {
        if Instant::now() > deadline {
            break;
        }
        let mut line = String::new();
        if harness.reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            if value["id"].as_u64() == Some(id_a)
                && let Some(reason) = value["result"]["stopReason"].as_str()
            {
                a_stop = Some(reason.to_string());
            }
            if value["id"].as_u64() == Some(id_b)
                && let Some(reason) = value["result"]["stopReason"].as_str()
            {
                b_stop = Some(reason.to_string());
            }
        }
    }

    assert_eq!(
        a_stop.as_deref(),
        Some("cancelled"),
        "session A must resolve canceled",
    );
    assert_eq!(
        b_stop.as_deref(),
        Some("end_turn"),
        "session B's cancel must NOT have fired, only A was canceled",
    );
}

/// `session/close` arriving while a prompt is still running must:
///   1. cancel the in-flight prompt (it resolves with `cancelled`),
///   2. return success on the close request itself,
///   3. cause re-close on the same id to error (slot is gone),
///   4. cause subsequent `session/prompt` against the closed id to error.
///
/// The architecture relies on the sibling cancellation cell so close can fire the token *without*
/// contending on the runtime mutex the in-flight prompt holds. If that wiring regressed, this test
/// would hang past the deadline.
#[test]
fn acp_session_close_while_prompt_in_flight_cancels_and_rejects_followups() {
    // Single round: a starting chunk, a 5s sleep that close should race against, then never-reached
    // completion text. If close doesn't cancel, the test will see end_turn after the full sleep.
    let script = serde_json::json!([[
        { "type": "text", "text": "starting..." },
        { "type": "sleep", "ms": 5000 },
        { "type": "text", "text": "done" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();

    // Fire the stalled prompt, then wait for it to actually start streaming so we know the turn is
    // parked in the 5s sleep.
    let prompt_id = harness.prompt(&session_id, "stall");
    let start_deadline = Instant::now() + Duration::from_secs(5);
    let _ = read_until(&mut harness.reader, start_deadline, |line| {
        line.contains("starting...")
    });

    // Fire close. The sibling cancellation cell pattern means this never blocks on the runtime
    // mutex.
    let close_id = harness.send_request(
        "session/close",
        serde_json::json!({ "sessionId": session_id.clone() }),
    );

    // Both prompt_id (prompt canceled) and close_id (close ok) must arrive well before the 5s
    // sleep would have finished.
    let response_deadline = Instant::now() + Duration::from_secs(3);
    let mut prompt_stop_reason: Option<String> = None;
    let mut close_result_seen = false;
    while prompt_stop_reason.is_none() || !close_result_seen {
        if Instant::now() > response_deadline {
            break;
        }
        let mut line = String::new();
        if harness.reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            if value["id"].as_u64() == Some(prompt_id)
                && let Some(reason) = value["result"]["stopReason"].as_str()
            {
                prompt_stop_reason = Some(reason.to_string());
            }
            if value["id"].as_u64() == Some(close_id) && value["result"].is_object() {
                close_result_seen = true;
            }
        }
    }

    assert_eq!(
        prompt_stop_reason.as_deref(),
        Some("cancelled"),
        "in-flight prompt must resolve canceled when session is closed mid-turn",
    );
    assert!(
        close_result_seen,
        "close request itself must return success even while a prompt is in flight",
    );

    // Re-close: must error.
    let re_close = harness.close_session(&session_id);
    assert!(
        re_close["error"].is_object(),
        "re-closing a closed session must error: {re_close}",
    );

    // Prompt against the closed id: must error.
    let stale_prompt_id = harness.prompt(&session_id, "ghost");
    let stale = harness.await_response(stale_prompt_id);
    assert!(
        stale["error"].is_object(),
        "prompting a closed session must error: {stale}",
    );
}

// === Input-validation error-path tests ==============================
//
// Each handler that takes a `sessionId`, `modeId`, or rich `prompt` content array must reject
// malformed input with a JSON-RPC `InvalidParams` (`-32602`) error, not a generic `InternalError`
// (`-32603`). Use the `AcpTestHarness` helper so the boilerplate stays out of these tests' way.

const ACP_INVALID_PARAMS: i64 = -32602;

fn assert_invalid_params(response: &serde_json::Value, context: &str) {
    let error = response["error"]
        .as_object()
        .unwrap_or_else(|| panic!("{context}: expected error response, got: {response}"));
    let code = error
        .get("code")
        .and_then(|c| c.as_i64())
        .unwrap_or_else(|| panic!("{context}: error missing numeric code: {response}"));
    assert_eq!(
        code, ACP_INVALID_PARAMS,
        "{context}: expected -32602 InvalidParams, got code {code}: {response}",
    );
}

const ACP_INVALID_PARAMS_CONFIG: &str = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;

/// `session/prompt` against an unknown `sessionId` must error with `InvalidParams`. An `audio`
/// content block likewise: meka accepts `text` / `resource_link` / `resource` / `image` (when
/// vision is on) but never `audio`, so it is a client contract violation.
#[test]
fn acp_session_prompt_rejects_unknown_session_and_audio_block() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);

    let unknown = harness.request(
        "session/prompt",
        serde_json::json!({
            "sessionId": "00000000-0000-0000-0000-000000000000",
            "prompt": [{ "type": "text", "text": "hi" }]
        }),
    );
    assert_invalid_params(&unknown, "prompt with unknown sessionId");

    // Open a real session and send an audio block (an unsupported content type); must yield
    // InvalidParams during content parsing, before any turn work.
    let session_id = harness.new_session();
    let bad_block = harness.request(
        "session/prompt",
        serde_json::json!({
            "sessionId": session_id,
            "prompt": [{
                "type": "audio",
                "data": "AAAA",
                "mimeType": "audio/wav"
            }]
        }),
    );
    assert_invalid_params(&bad_block, "prompt with audio content block");
}

/// With `vision = false` on the selected profile, meka advertises `image: false` and rejects image
/// content blocks with `InvalidParams` (the rejection happens during parsing, before any turn).
#[test]
fn acp_session_prompt_rejects_image_when_vision_disabled() {
    const NO_VISION_CONFIG: &str = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
vision = false
"#;
    let mut harness = AcpTestHarness::spawn(NO_VISION_CONFIG, None);

    let session_id = harness.new_session();
    let rejected = harness.request(
        "session/prompt",
        serde_json::json!({
            "sessionId": session_id,
            "prompt": [{
                "type": "image",
                "data": "AAAA",
                "mimeType": "image/png"
            }]
        }),
    );
    assert_invalid_params(&rejected, "image block with vision disabled");
}

/// `session/load` rejects malformed UUIDs and refuses to re-load a session that's already open
/// (closing first is the correct flow). Both arms must report `InvalidParams`, not generic
/// `InternalError`.
#[test]
fn acp_session_load_rejects_malformed_uuid_and_already_loaded() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);

    let bad_uuid = harness.request(
        "session/load",
        serde_json::json!({
            "sessionId": "not-a-uuid-at-all",
            "cwd": harness.config_dir(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&bad_uuid, "load with malformed UUID");

    // Open a real session and immediately try to reload it.
    let session_id = harness.new_session();
    let already = harness.request(
        "session/load",
        serde_json::json!({
            "sessionId": session_id,
            "cwd": harness.config_dir(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&already, "load already-open session");
}

/// `session/resume` rejects malformed UUIDs, unknown ids, and ids for sessions already occupying
/// the active slot, all client-side mistakes that map to `InvalidParams`. The already-loaded guard
/// at `src/acp.rs:1695` mirrors the `session/load` one tested in
/// `acp_session_load_rejects_malformed_uuid_and_already_loaded`.
#[test]
fn acp_session_resume_rejects_malformed_uuid_unknown_and_already_loaded() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);

    let bad_uuid = harness.request(
        "session/resume",
        serde_json::json!({
            "sessionId": "not-a-uuid",
            "cwd": harness.config_dir(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&bad_uuid, "resume with malformed UUID");

    let unknown = harness.request(
        "session/resume",
        serde_json::json!({
            "sessionId": "00000000-0000-0000-0000-000000000000",
            "cwd": harness.config_dir(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&unknown, "resume with unknown UUID");

    // Open a real session and immediately try to resume it. The session is already in the active
    // map, so the resume guard rejects with `InvalidParams`.
    let session_id = harness.new_session();
    let already = harness.request(
        "session/resume",
        serde_json::json!({
            "sessionId": session_id,
            "cwd": harness.config_dir(),
            "mcpServers": []
        }),
    );
    assert_invalid_params(&already, "resume already-active session");
}

/// `session/set_mode` rejects an unknown mode id and rejects a valid-but-disabled mode (configured
/// `enabled` array doesn't list it). Both arms are input validation: `InvalidParams`.
#[test]
fn acp_session_set_mode_rejects_unknown_and_disabled() {
    let config = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read"]
"#;
    let mut harness = AcpTestHarness::spawn(config, None);
    let session_id = harness.new_session();

    let unknown = harness.request(
        "session/set_mode",
        serde_json::json!({
            "sessionId": session_id,
            "modeId": "definitely-not-a-mode"
        }),
    );
    assert_invalid_params(&unknown, "set_mode with unknown mode id");

    // `unrestricted` is a valid mode id (parse_mode_id succeeds) but it's not in the configured
    // `enabled` list, so try_set rejects it.
    let disabled = harness.request(
        "session/set_mode",
        serde_json::json!({
            "sessionId": session_id,
            "modeId": "unrestricted"
        }),
    );
    assert_invalid_params(&disabled, "set_mode with disabled mode");
}

/// Tightens the existing protocol-version test: meka must clamp far-future versions to
/// `ProtocolVersion::LATEST` (currently V1), not echo the requested value verbatim. A naive echo
/// would let a future client think we support a version we haven't shipped.
#[test]
fn acp_initialize_clamps_far_future_version_to_latest() {
    let install = Install::new();
    install.write_config(ACP_INVALID_PARAMS_CONFIG);

    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = support::TimedLines::spawn(stdout);
    let stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });

    // Far-future version, well past anything the schema crate would ever produce. Must come back
    // clamped to LATEST (V1).
    let init_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": { "protocolVersion": 9999 }
    });
    writeln!(stdin, "{init_request}").expect("init");
    let lines = read_until(&mut reader, window(10), |line| line.contains("\"id\":1"));

    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let line = lines
        .iter()
        .find(|line| line.contains("\"id\":1"))
        .unwrap_or_else(|| {
            panic!(
                "no initialize response.\nSTDERR:\n{}\nstdout:\n{}",
                stderr_handle.join().unwrap_or_default(),
                lines.join(""),
            )
        });
    let response: serde_json::Value = serde_json::from_str(line).expect("parse init");
    let result = response["result"]
        .as_object()
        .unwrap_or_else(|| panic!("initialize must succeed, got: {response}"));
    let negotiated = result
        .get("protocolVersion")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("missing numeric protocolVersion in: {response}"));
    // Exactly LATEST, which is V1 today. A `<=` here let an agent that echoed 0, or ignored the
    // field, pass a test named for clamping. When the SDK ships a stable V2 this line moves with
    // it, which is the point: the negotiated version is a fact worth pinning.
    assert_eq!(
        negotiated, 1,
        "a far-future protocolVersion must be clamped to LATEST (V1 today)"
    );
}

// === Mock-provider-driven coverage ==================================
//
// These tests round-trip features that the mock provider couldn't emit before (`ThinkingDelta` /
// `ThinkingComplete` and the `MaxTokens` stop reason) through the full ACP pipeline.

/// A `ThinkingDelta` + `ThinkingComplete` pair from the provider maps to a `session/update`
/// notification with `sessionUpdate: "agent_thought_chunk"` carrying the thinking text. The text
/// body is the only thing the editor needs; the `signature` field is opaque pass-through and not
/// currently surfaced in the notification.
#[test]
fn acp_session_prompt_emits_agent_thought_chunk_for_thinking_block() {
    let script = serde_json::json!([[
        { "type": "thinking_delta", "text": "weighing options... " },
        { "type": "thinking_delta", "text": "considering safety" },
        { "type": "thinking_complete", "signature": null },
        { "type": "text", "text": "ok" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "think first");
    let (updates, response) = harness.collect_updates(&session_id, id);

    let thought = updates
        .iter()
        .find(|u| u["params"]["update"]["sessionUpdate"] == "agent_thought_chunk")
        .unwrap_or_else(|| panic!("missing agent_thought_chunk; updates: {updates:?}"));
    let text = thought["params"]["update"]["content"]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("thought chunk missing text body: {thought}"));
    assert!(
        text.contains("weighing options") || text.contains("considering safety"),
        "agent_thought_chunk text should carry the scripted thinking content; got: {text}",
    );
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

/// Under `redact-thinking`, the provider streams a `thinking_complete` carrying only a signature
/// (no `thinking_delta`). The agent must retain that block for replay continuity without surfacing
/// a spurious empty `agent_thought_chunk`: the turn completes normally and the only visible output
/// is the assistant text.
#[test]
fn acp_session_prompt_keeps_empty_thinking_with_signature_quietly() {
    let script = serde_json::json!([[
        { "type": "thinking_complete", "signature": "OPAQUE_SIGNATURE_BLOB" },
        { "type": "text", "text": "answer" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "go");
    let (updates, response) = harness.collect_updates(&session_id, id);

    // No empty thought chunk: the redacted (text-less) thinking block stays off-screen.
    assert!(
        !updates
            .iter()
            .any(|u| u["params"]["update"]["sessionUpdate"] == "agent_thought_chunk"),
        "empty thinking block must not emit an agent_thought_chunk; updates: {updates:?}",
    );
    let saw_answer = updates.iter().any(|u| {
        u["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
            && u["params"]["update"]["content"]["text"]
                .as_str()
                .is_some_and(|t| t.contains("answer"))
    });
    assert!(
        saw_answer,
        "assistant text should reach the client; got: {updates:?}"
    );
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

/// A thinking-only round (no visible text) ending in `end_turn` must not end the turn silently. The
/// agent nudges once for a user-visible response and re-issues, so the scripted second round's text
/// reaches the client. Mirrors Claude Code's `query_thinking_only_response` recovery. Without the
/// nudge the first round's `end_turn` would end the turn and the second round would never run.
#[test]
fn acp_session_prompt_nudges_thinking_only_turn() {
    let script = serde_json::json!([
        [
            { "type": "thinking_delta", "text": "pondering silently" },
            { "type": "thinking_complete", "signature": null },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "recovered answer" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "go");
    let (updates, response) = harness.collect_updates(&session_id, id);

    // The second round runs only if the nudge fired after the thinking-only first round.
    let saw_recovered = updates.iter().any(|u| {
        u["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
            && u["params"]["update"]["content"]["text"]
                .as_str()
                .is_some_and(|text| text.contains("recovered answer"))
    });
    assert!(
        saw_recovered,
        "thinking-only turn must nudge and surface the second round's text; updates: {updates:?}",
    );
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

/// `MockStopReason::MaxTokens` propagates end-to-end as `stopReason: "max_tokens"` on the
/// `PromptResponse`. The mock already had the enum variant; this test plugs the gap in integration
/// coverage.
#[test]
fn acp_session_prompt_max_tokens_stop_reason() {
    let script = serde_json::json!([[
        { "type": "text", "text": "truncated mid-thought" },
        { "type": "message_end", "stop_reason": "max_tokens" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "go");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "max_tokens",
        "MaxTokens script → stopReason='max_tokens'; got: {response}",
    );
}

/// Per-session frontend routing: a thinking block scripted for session A produces an
/// `agent_thought_chunk` on A's notifications and *not* on B's. Regression guard against
/// cross-session leakage of `session/update` traffic. The mock drains rounds in FIFO order across
/// sessions, so we send the only round to session A first then drive B's prompt through an empty
/// round (no `thinking_*` events).
#[test]
fn acp_session_prompt_thought_chunk_routes_per_session() {
    let script = serde_json::json!([
        [
            { "type": "thinking_delta", "text": "session A only" },
            { "type": "thinking_complete", "signature": null },
            { "type": "text", "text": "A response" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "B response" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let sid_a = harness.new_session();
    let sid_b = harness.new_session();

    let id_a = harness.prompt(&sid_a, "go A");
    let (updates_a, _) = harness.collect_updates(&sid_a, id_a);

    let id_b = harness.prompt(&sid_b, "go B");
    let (updates_b, _) = harness.collect_updates(&sid_b, id_b);

    assert!(
        updates_a
            .iter()
            .any(|u| u["params"]["update"]["sessionUpdate"] == "agent_thought_chunk"),
        "session A must observe its agent_thought_chunk; updates: {updates_a:?}",
    );
    assert!(
        updates_b
            .iter()
            .all(|u| u["params"]["update"]["sessionUpdate"] != "agent_thought_chunk"),
        "session B must not see A's agent_thought_chunk; updates: {updates_b:?}",
    );
}

/// Non-Interrupted `Agent::run_turn` errors must surface as a JSON-RPC `error` on the
/// `session/prompt` response (the `Err(error)` arm at `src/acp.rs:1514`). Scripted via the `Fail`
/// mock event so no real provider call is made; the agent loop's stream handler turns the provider
/// error into `MekaError::Provider`, `run_turn` propagates it, and the ACP handler maps it to
/// `internal_error`.
#[test]
fn acp_session_prompt_surfaces_provider_error_as_jsonrpc_error() {
    let script = serde_json::json!([[
        { "type": "fail", "message": "scripted provider failure" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "go");
    let response = harness.await_response(id);
    assert!(
        response.get("error").is_some(),
        "non-Interrupted run_turn error must surface as JSON-RPC error; got: {response}",
    );
    assert!(
        response["result"].is_null(),
        "JSON-RPC response carries either result or error, not both; got: {response}",
    );
    // `agent_client_protocol::util::internal_error` sets the standard `"Internal error"` JSON-RPC
    // message and stuffs the explanatory string into `data`.
    let message = response["error"]["message"]
        .as_str()
        .unwrap_or_else(|| panic!("error.message must be a string: {response}"));
    assert_eq!(message, "Internal error");
    let code = response["error"]["code"]
        .as_i64()
        .unwrap_or_else(|| panic!("error.code must be an integer: {response}"));
    assert_eq!(code, -32603, "internal_error → JSON-RPC code -32603");
    let data = response["error"]["data"]
        .as_str()
        .unwrap_or_else(|| panic!("error.data must carry the detail string: {response}"));
    assert!(
        data.contains("the provider rejected or failed this turn"),
        "error.data should carry meka's own sentence about the failure; got: {data}",
    );
    assert!(
        data.contains("scripted provider failure"),
        "and the upstream's own text, which `relay_provider_errors` leaves on by default; got: \
         {data}",
    );
}

/// What an editor is told about a failed turn follows the operator's switch, as it does over HTTP.
///
/// `meka acp` formatted every failure as `meka turn failed: {error}`, so an upstream body naming
/// the operator's account with the provider went to the client whatever `[serve]
/// relay_provider_errors` said -- and a deployment that had turned it off believed the text was
/// withheld everywhere, because the only surface it had checked was the HTTP one.
///
/// Both settings, because only one of them is the default. meka's own sentence is asserted in both,
/// which is what makes the member additive rather than a replacement for it.
#[test]
fn an_acp_turn_failure_relays_the_upstream_only_when_the_operator_asked() {
    let secret = "acct-0f3c-operator-only";
    for (relay, expect_relayed) in [
        ("", true),
        ("\n[serve]\nrelay_provider_errors = false\n", false),
    ] {
        let script = serde_json::json!([[
            { "type": "fail", "message": format!("API returned status 401: {{\"account_uuid\":\"{secret}\"}}") }
        ]]);
        let config = format!("{ACP_INVALID_PARAMS_CONFIG}{relay}");
        let mut harness = AcpTestHarness::spawn(&config, Some(script));
        let session_id = harness.new_session();
        let id = harness.prompt(&session_id, "go");
        let response = harness.await_response(id);

        let data = response["error"]["data"].as_str().unwrap_or_else(|| {
            panic!("a failed turn must answer with an error carrying `data`: {response}")
        });
        assert!(
            data.contains("the provider rejected or failed this turn"),
            "meka's own sentence is the same either way: {data}"
        );
        assert_eq!(
            data.contains(secret),
            expect_relayed,
            "with `{relay}` the upstream body must {} the client: {data}",
            if expect_relayed { "reach" } else { "not reach" }
        );
        // The whole error object, so a member moved elsewhere in the payload still counts as
        // having reached the client.
        if !expect_relayed {
            assert!(
                !response["error"].to_string().contains(secret),
                "the upstream body reached the client by another route: {response}"
            );
        }
    }
}

/// A load refused by the builder must not have moved the session first.
///
/// `session/load` wrote the client's `cwd` and `additionalDirectories` onto the row before
/// `build_session_runtime` could refuse, so a load that failed -- a profile that has left
/// `config.toml`, an account with no stored credential -- came back an error having already
/// repointed the session. `cwd` is the writable boundary at `workspace` and the directory a
/// scheduled gate is re-checked in, so the next process to open that session ran it somewhere the
/// user never asked for, on the strength of a request meka had declined.
///
/// The refusal used here is the profile leaving `config.toml`, moved by a second connection to the
/// store, which is what `an_acp_scheduled_fire_refuses_a_profile_the_row_no_longer_names` uses for
/// the same reason: it is the one builder refusal a test can produce without a credential.
///
/// `session/resume` is the sibling door and is exercised in the same run.
#[test]
fn an_acp_load_or_resume_refused_by_the_builder_leaves_the_workspace_alone() {
    for method in ["session/load", "session/resume"] {
        let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
        let session_id = harness.new_session();
        let closed = harness.close_session(&session_id);
        assert!(closed["result"].is_object(), "close must succeed: {closed}");

        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        let moved = connection
            .execute("UPDATE sessions SET profile = 'retired' WHERE id = ?1", [
                &session_id,
            ])
            .expect("repin the session");
        assert_eq!(moved, 1, "the repin matched no row");
        let cwd_before: Option<String> = connection
            .query_row(
                "SELECT cwd FROM sessions WHERE id = ?1",
                [&session_id],
                |row| row.get(0),
            )
            .expect("read the recorded directory");
        drop(connection);

        // A directory that exists and differs from the session's, so a handler that got as far as
        // the write leaves a difference this test can see.
        let elsewhere = harness.config_dir().join("elsewhere");
        std::fs::create_dir_all(&elsewhere)
            .expect("create the directory the load would move it to");

        let refused = harness.request(
            method,
            serde_json::json!({
                "sessionId": session_id,
                "cwd": elsewhere.to_string_lossy(),
                "mcpServers": [],
            }),
        );
        assert!(
            refused.get("error").is_some(),
            "{method}: a profile the configuration no longer has must be refused: {refused}"
        );

        let connection = rusqlite::Connection::open(harness.database()).expect("reopen the store");
        let cwd_after: Option<String> = connection
            .query_row(
                "SELECT cwd FROM sessions WHERE id = ?1",
                [&session_id],
                |row| row.get(0),
            )
            .expect("the row is still readable");
        assert_eq!(
            cwd_before,
            cwd_after,
            "{method}: refusing after the write is not an equivalent answer; the session was \
             repointed at {}",
            elsewhere.display()
        );
    }
}

/// A transient failure (`FailRetryable`) on the first attempt must be retried automatically:
/// `Agent::run_streaming` re-invokes `provider.stream()`, consuming the mock's next scripted round,
/// and the turn completes successfully with exactly the second round's content: no duplication,
/// no error surfaced to the client.
#[test]
fn acp_session_prompt_retries_transient_provider_error_then_succeeds() {
    let script = serde_json::json!([
        [{ "type": "fail_retryable", "message": "overloaded", "retry_after_secs": null }],
        [
            { "type": "text", "text": "recovered" },
            { "type": "message_end", "stop_reason": "end_turn" },
        ],
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "go");
    let (updates, response) = harness.collect_updates(&session_id, id);

    assert!(
        response.get("error").is_none(),
        "the retry must be invisible to the client; got error: {response}",
    );

    let text_chunks: Vec<&str> = updates
        .iter()
        .filter(|u| u["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        .filter_map(|u| u["params"]["update"]["content"]["text"].as_str())
        .collect();
    assert_eq!(
        text_chunks,
        vec!["recovered"],
        "exactly one clean copy of the recovered round's text, no duplication from the failed \
         first attempt; updates: {updates:?}",
    );
}

/// `FailRetryable` repeated past `MAX_PROVIDER_RETRIES` must exhaust the retry budget and surface
/// the failure to the client as a normal JSON-RPC error, exactly like a non-retryable failure does.
#[test]
fn acp_session_prompt_exhausts_retries_then_surfaces_error() {
    // MAX_PROVIDER_RETRIES (2) retries + the initial attempt = 3 total attempts, all failing.
    let script = serde_json::json!([
        [{ "type": "fail_retryable", "message": "overloaded 1", "retry_after_secs": null }],
        [{ "type": "fail_retryable", "message": "overloaded 2", "retry_after_secs": null }],
        [{ "type": "fail_retryable", "message": "overloaded 3", "retry_after_secs": null }],
    ]);
    // The default backoff (1s + 2s = 3s of sleeping) plus process overhead is comfortably under
    // 15s, but give this one extra headroom since it's the slowest test in the suite by design.
    let mut harness = AcpTestHarness::builder()
        .config(ACP_INVALID_PARAMS_CONFIG)
        .script(script)
        .window(std::time::Duration::from_secs(25))
        .build();
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "go");
    let response = harness.await_response(id);

    assert!(
        response.get("error").is_some(),
        "exhausted retries must surface as a JSON-RPC error; got: {response}",
    );
    let data = response["error"]["data"]
        .as_str()
        .unwrap_or_else(|| panic!("error.data must carry the detail string: {response}"));
    // The last attempt's message is what propagates.
    assert!(
        data.contains("overloaded 3"),
        "error.data should carry the final attempt's message; got: {data}",
    );
}

/// A non-retryable failure (`Fail`, mapping to a plain `MekaError::Provider`) must still fail
/// immediately with no backoff delay, which guards against over-broadening the retry
/// classification to errors that were never meant to retry.
#[test]
fn acp_session_prompt_plain_fail_is_not_retried() {
    let script = serde_json::json!([[{ "type": "fail", "message": "permanent failure" }]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "go");
    let started = std::time::Instant::now();
    let response = harness.await_response(id);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "a non-retryable failure must not incur any backoff delay; took {:?}",
        started.elapsed(),
    );
    assert!(
        response.get("error").is_some(),
        "plain Fail must still surface as a JSON-RPC error; got: {response}",
    );
    let data = response["error"]["data"]
        .as_str()
        .unwrap_or_else(|| panic!("error.data must carry the detail string: {response}"));
    assert!(
        data.contains("permanent failure"),
        "error.data should propagate the underlying provider error text; got: {data}",
    );
}

/// Once the frontend has already shown text this attempt, a subsequent transient error in the
/// SAME round must NOT trigger a retry: retrying after the user has seen partial output would
/// duplicate/corrupt what's on screen. The mock's per-round-is-one-`stream()`-call model means a
/// `FailRetryable` after a `Text` event in the same round exercises exactly this: the driver sends
/// the text (setting `content_started`), then fails, and `run_streaming` must propagate the error
/// immediately instead of consuming a second round.
#[test]
fn acp_session_prompt_does_not_retry_once_content_shown() {
    let script = serde_json::json!([
        [
            { "type": "text", "text": "partial" },
            { "type": "fail_retryable", "message": "overloaded mid-stream", "retry_after_secs": null },
        ],
        [
            { "type": "text", "text": "should never be reached" },
            { "type": "message_end", "stop_reason": "end_turn" },
        ],
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "go");
    let (updates, response) = harness.collect_updates(&session_id, id);

    assert!(
        response.get("error").is_some(),
        "a transient error after content was shown must surface immediately, not retry; got: {response}",
    );
    let text_chunks: Vec<&str> = updates
        .iter()
        .filter(|u| u["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        .filter_map(|u| u["params"]["update"]["content"]["text"].as_str())
        .collect();
    assert_eq!(
        text_chunks,
        vec!["partial"],
        "only the pre-failure text must appear; the second round must never be consumed: {updates:?}",
    );
}

/// `session/request_permission` failure marks the connection as disconnected so the agent loop
/// bails out promptly. A spec-conformant client always answers `Selected` / `Cancelled`, so any
/// `Err` from `block_task` (channel closed or peer JSON-RPC error) signals a broken/malformed
/// client. The agent denies the tool call, completes the current iteration, then short-circuits the
/// next iteration via `Frontend::client_disconnected()`; the turn resolves `cancelled`, not
/// `end_turn`.
#[test]
fn acp_session_prompt_request_permission_failure_marks_disconnect() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
approvals = true
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target = work_dir_beside(config_dir).join("would-write.txt");
            serde_json::json!([
                [
                    { "type": "text", "text": "writing..." },
                    { "type": "tool_use_start", "id": "call_write", "name": "write_file" },
                    {
                        "type": "tool_use_end",
                        "input": { "path": target.to_str().unwrap(), "content": "hi" }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                // Second round would emit "done" + end_turn, but the disconnect-mark must
                // short-circuit the loop before it streams. If the mark wires up correctly this
                // round is never drained.
                [
                    { "type": "text", "text": "done" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "write it");

    let (_updates, response) = harness.collect_updates_with_dispatch(&session_id, id, |value| {
        match value["method"].as_str() {
            Some("session/request_permission") => Some(jsonrpc_error(
                value["id"].clone(),
                "synthetic client error response",
            )),
            _ => None,
        }
    });

    assert_eq!(
        response["result"]["stopReason"], "cancelled",
        "permission-Err must mark disconnect → next loop iter short-circuits; got: {response}",
    );
}

// === fs/read_text_file line + limit =================================

/// `read_file` asks the editor for the whole document even when the model passed `offset`/`limit`,
/// and windows what comes back locally.
///
/// Pushing the window down to `fs/read_text_file` looked tidier and cost two things. The freshness
/// stamp recorded the slice rather than the document, so the next `edit_file` compared a slice
/// against the whole buffer and refused with a false "changed in the editor". And a response of
/// exactly `limit` lines was indistinguishable from a file that ended there, so a truncated read
/// was handed to the model with no notice.
#[test]
fn acp_fs_read_text_file_fetches_the_whole_document_and_windows_locally() {
    let on_disk_marker = "DO-NOT-READ-ME-FROM-DISK\n".repeat(100);
    let on_disk_marker_for_seed = on_disk_marker.clone();
    let mut harness = AcpTestHarness::builder()
        .config(ACP_INVALID_PARAMS_CONFIG)
        .capabilities(serde_json::json!({
            "fs": { "readTextFile": true, "writeTextFile": false }
        }))
        .pre_spawn(move |config_dir| {
            let target = fixture_beside(config_dir, "delegated-line-limit.txt");
            std::fs::write(&target, &on_disk_marker_for_seed).expect("write target");
            serde_json::json!([
                [
                    { "type": "text", "text": "reading partial..." },
                    { "type": "tool_use_start", "id": "call_read", "name": "read_file" },
                    {
                        "type": "tool_use_end",
                        "input": { "path": target.to_string_lossy(), "offset": 9, "limit": 50 }
                    },
                    { "type": "message_end", "stop_reason": "tool_use" }
                ],
                [
                    { "type": "text", "text": "done" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let target = harness.work_dir().join("delegated-line-limit.txt");
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "read partial");

    let mut saw_request = false;
    let mut observed_line = serde_json::Value::Null;
    let mut observed_limit = serde_json::Value::Null;
    let _ = harness.await_response_with_dispatch(id, |value| match value["method"].as_str() {
        Some("fs/read_text_file") => {
            saw_request = true;
            observed_line = value["params"]["line"].clone();
            observed_limit = value["params"]["limit"].clone();
            Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": value["id"].clone(),
                "result": { "content": "DELEGATED CONTENT FROM EDITOR" }
            }))
        }
        _ => None,
    });

    assert!(saw_request, "the read must still be delegated");
    assert!(
        observed_line.is_null(),
        "the window is applied locally, so no line is sent: {observed_line}",
    );
    assert!(
        observed_limit.is_null(),
        "the window is applied locally, so no limit is sent: {observed_limit}",
    );
    // On-disk file is untouched, proving the delegate path won.
    assert_eq!(
        std::fs::read_to_string(&target).expect("read on-disk"),
        on_disk_marker,
        "on-disk file content should be unchanged",
    );
}

// === V2 protocol conformance ========================================

/// `ContentBlock::ResourceLink` is part of the ACP baseline, so meka flattens the link into a tag
/// the model can see rather than refusing it with `InvalidParams`.
#[test]
fn acp_session_prompt_accepts_resource_link_baseline() {
    let script = serde_json::json!([[
        { "type": "text", "text": "ack" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    // session/prompt with a resource_link block; assert no error.
    let id = harness.send_request(
        "session/prompt",
        serde_json::json!({
            "sessionId": session_id,
            "prompt": [
                { "type": "text", "text": "describe this:" },
                {
                    "type": "resource_link",
                    "name": "README.md",
                    "uri": "file:///tmp/README.md",
                    "description": "project readme"
                }
            ]
        }),
    );
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "resource_link content block must be accepted, not error: {response}",
    );
}

/// An embedded `resource` block (an @-mention's inlined contents) is accepted and flattened into a
/// `<resource>` tag, not rejected. meka advertises `embeddedContext: true`.
#[test]
fn acp_session_prompt_accepts_embedded_resource() {
    let script = serde_json::json!([[
        { "type": "text", "text": "ack" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.send_request(
        "session/prompt",
        serde_json::json!({
            "sessionId": session_id,
            "prompt": [
                { "type": "text", "text": "summarize:" },
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///tmp/notes.txt",
                        "text": "the meeting notes",
                        "mimeType": "text/plain"
                    }
                }
            ]
        }),
    );
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "embedded resource block must be accepted, not error: {response}",
    );
}

/// An `image` content block is accepted when the profile has vision on (the default), and the turn
/// runs to completion.
#[test]
fn acp_session_prompt_accepts_image_with_vision() {
    let script = serde_json::json!([[
        { "type": "text", "text": "i see it" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    // A 1x1 transparent PNG, base64-encoded, and it has to be a real one: meka decodes every
    // attachment before forwarding it, so a hand-mangled PNG with a bad IDAT checksum would be
    // refused here rather than exercise the path.
    let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    let id = harness.send_request(
        "session/prompt",
        serde_json::json!({
            "sessionId": session_id,
            "prompt": [
                { "type": "text", "text": "what is this?" },
                { "type": "image", "data": png_b64, "mimeType": "image/png" }
            ]
        }),
    );
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "image block must be accepted when vision is on: {response}",
    );
}

/// `session/cancel` yields a `Cancelled` stop reason even when the cancellation manifests as
/// a non-`Interrupted` provider error. Script a `Sleep` followed by a `Fail`; fire cancel during
/// the sleep; assert `stopReason: canceled` rather than the JSON-RPC error the `Fail` would
/// otherwise produce.
#[test]
fn acp_session_prompt_canceled_after_provider_error() {
    let script = serde_json::json!([[
        { "type": "text", "text": "starting..." },
        { "type": "sleep", "ms": 5000 },
        { "type": "fail", "message": "would-be internal error" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "go");

    // Wait for "starting..." to confirm the turn is parked in the sleep, then fire cancel.
    let barrier = Instant::now() + Duration::from_secs(3);
    let _ = read_until(&mut harness.reader, barrier, |line| {
        line.contains("starting...")
    });
    harness.cancel(&session_id);
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "cancelled",
        "post-cancel error must surface as Cancelled, not internal_error: {response}",
    );
}

/// A `session/cancel` sent straight after a `session/prompt` stops that prompt.
///
/// The window `cancel_armed_through` exists for. `session/prompt` is spawned off the dispatch
/// loop while `session/cancel` is handled on it, so the cancel can reach the session before the
/// turn has published a token. Both interleavings must end the same way: whichever arrives first,
/// the turn the editor asked to stop is the turn that stops.
#[test]
fn acp_a_cancel_sent_straight_after_a_prompt_stops_it() {
    // First turn completes, so the second is dispatched with no turn running and nothing in the
    // cell, which is the state the race needs. The sleep is long enough that the turn cannot
    // finish on its own.
    let script = serde_json::json!([
        [
            { "type": "text", "text": "first done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "second starting..." },
            { "type": "sleep", "ms": 5000 },
            { "type": "text", "text": "second done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();

    let id_1 = harness.prompt(&session_id, "first");
    let response_1 = harness.await_response(id_1);
    assert_eq!(response_1["result"]["stopReason"], "end_turn");

    // Fired back to back with no wait between them, so the cancel is dispatched while the prompt
    // is still on its way to its handler.
    let id_2 = harness.prompt(&session_id, "second");
    harness.cancel(&session_id);
    let response_2 = harness.await_response(id_2);
    assert_eq!(
        response_2["result"]["stopReason"], "cancelled",
        "a cancel racing its own prompt must still stop it: {response_2}",
    );
}

/// Canceling a turn that is actually running must not disarm the next one. The latch exists for
/// the cancel no turn received; a cancel a live turn consumed has already done its work.
#[test]
fn acp_canceling_a_running_turn_leaves_the_next_one_alone() {
    // First turn sleeps so the cancel lands mid-flight. Second turn is short: it must run.
    let script = serde_json::json!([
        [
            { "type": "text", "text": "first starting..." },
            { "type": "sleep", "ms": 5000 },
            { "type": "text", "text": "first done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "second done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();

    // Cancel only once the turn is provably streaming, so this tests the in-flight door rather
    // than racing the between-turns one it is meant to be distinguished from.
    let id_1 = harness.prompt(&session_id, "first");
    let barrier = Instant::now() + Duration::from_secs(3);
    let started = read_until(&mut harness.reader, barrier, |line| {
        line.contains("first starting...")
    });
    assert!(
        started
            .iter()
            .any(|line| line.contains("first starting...")),
        "first turn never began streaming",
    );
    harness.cancel(&session_id);
    let response_1 = harness.await_response(id_1);
    assert_eq!(response_1["result"]["stopReason"], "cancelled");

    let id_2 = harness.prompt(&session_id, "second");
    let response_2 = harness.await_response(id_2);
    assert_eq!(
        response_2["result"]["stopReason"], "end_turn",
        "a cancel the previous turn consumed must not carry into the next prompt: {response_2}",
    );
}

/// A cancel with no prompt on its way is spent on nothing, not saved for a later one.
///
/// The latch is for a prompt the editor has already sent, so a stop with nothing pending has
/// nothing to stop. Canceling twice is the way a user reaches this without meaning to: the second
/// click lands after the turn resolved, and latching it kills whatever they type next.
#[test]
fn acp_a_cancel_with_nothing_pending_does_not_touch_a_later_prompt() {
    let script = serde_json::json!([
        [
            { "type": "text", "text": "first starting..." },
            { "type": "sleep", "ms": 5000 },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "second done" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();

    let id_1 = harness.prompt(&session_id, "first");
    let barrier = Instant::now() + Duration::from_secs(3);
    let started = read_until(&mut harness.reader, barrier, |line| {
        line.contains("first starting...")
    });
    assert!(
        started
            .iter()
            .any(|line| line.contains("first starting...")),
        "first turn never began streaming",
    );
    harness.cancel(&session_id);
    assert_eq!(
        harness.await_response(id_1)["result"]["stopReason"],
        "cancelled"
    );

    // The second stop, now that the turn it would have interrupted is already resolved.
    harness.cancel(&session_id);
    let next = harness.prompt(&session_id, "second");
    let response = harness.await_response(next);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "a stop with nothing to stop must not be saved for the next prompt: {response}",
    );
}

/// A prompt refused before it runs still spends the cancel that was latched for it.
///
/// The latch belongs to one prompt. Left set, it is spent on the next prompt instead, which is the
/// same defect one door along: the editor's stop kills a turn the user typed afterwards.
#[test]
fn acp_a_refused_prompt_spends_the_cancel_latched_for_it() {
    let script = serde_json::json!([[
        { "type": "text", "text": "ran" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();

    // Fired without waiting, then canceled straight away, so the cancel is handled while this
    // prompt is still on its way and is latched for it. The `audio` block is then refused during
    // content validation, before the turn ever resolves a session or publishes a token.
    let refused_id = harness.send_request(
        "session/prompt",
        serde_json::json!({
            "sessionId": session_id,
            "prompt": [{ "type": "audio", "data": "AAAA", "mimeType": "audio/wav" }],
        }),
    );
    harness.cancel(&session_id);
    let refused = harness.await_response(refused_id);
    assert!(
        refused["error"].is_object(),
        "an audio block must be refused: {refused}",
    );

    let next = harness.prompt(&session_id, "after the refusal");
    let response = harness.await_response(next);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "a refused prompt must not leave its cancel for the next one: {response}",
    );
}

/// A cancel armed for a prompt that is then refused is not handed to a prompt sent after it.
///
/// A client that pipelines is the only way to reach this, but it is reachable: the refused prompt
/// leaves the arming set, and the prompt behind it in the queue spends it. The sequence number is
/// what separates them, so the second prompt must run even though the first never consumed its own
/// cancel.
///
/// Repeated because one pass is not reliably the case under test. The arming only happens if the
/// cancel is dequeued before the refused prompt's own turn reaches the check that refuses it, and
/// the two orderings are externally identical, so a single pass silently tests nothing about one
/// time in ten. Measured at 45 of 50 with the fix reverted; over ten passes an escape needs every
/// one of them to fall the same way.
#[test]
fn acp_a_refused_prompt_does_not_hand_its_cancel_to_the_next_one() {
    let turn = serde_json::json!([
        { "type": "text", "text": "the queued prompt ran" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]);
    const PASSES: usize = 10;
    let script = serde_json::Value::Array(vec![turn; PASSES]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();

    for pass in 0..PASSES {
        // All three written back to back with no waiting, so the dispatch loop takes them in
        // order: the refused prompt is counted, the cancel is armed through it, and only then is
        // the queued prompt dispatched. Whitespace-only text is refused well after that, past an
        // await.
        let refused_id = harness.send_request(
            "session/prompt",
            serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "   " }],
            }),
        );
        harness.cancel(&session_id);
        let queued_id = harness.prompt(&session_id, "queued behind the refusal");

        let refused = harness.await_response(refused_id);
        assert!(
            refused["error"].is_object(),
            "pass {pass}: a whitespace-only prompt must be refused: {refused}",
        );
        let queued = harness.await_response(queued_id);
        assert_eq!(
            queued["result"]["stopReason"], "end_turn",
            "pass {pass}: a cancel armed before this prompt existed must not stop it: {queued}",
        );
    }
}

/// `session/set_mode` does not take the runtime mutex: a mid-turn level change takes effect without
/// waiting for the turn to finish.
#[test]
fn acp_session_set_mode_during_long_prompt_does_not_block() {
    let script = serde_json::json!([[
        { "type": "text", "text": "running..." },
        { "type": "sleep", "ms": 2000 },
        { "type": "text", "text": "done" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    const CONFIG: &str = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, Some(script));
    let session_id = harness.new_session();
    let prompt_id = harness.prompt(&session_id, "go");

    // Wait for the turn to start streaming before firing set_mode.
    let barrier = Instant::now() + Duration::from_secs(3);
    let _ = read_until(&mut harness.reader, barrier, |line| {
        line.contains("running...")
    });

    // set_mode while the turn is mid-sleep must return promptly (well under the sleep's 2s).
    let start = Instant::now();
    let set_response = harness.set_mode(&session_id, "unrestricted");
    let elapsed = start.elapsed();
    assert!(
        set_response["result"].is_object(),
        "set_mode must succeed mid-turn: {set_response}",
    );
    assert!(
        elapsed < Duration::from_secs(1),
        "set_mode must not block on the runtime mutex; took {elapsed:?}",
    );

    let prompt_response = harness.await_response(prompt_id);
    assert_eq!(prompt_response["result"]["stopReason"], "end_turn");
}

/// `session/cancel` during an approval prompt resolves the turn promptly. Without the race against
/// the cancellation token, the agent hangs inside `request_permission` until the client answers.
#[test]
fn acp_session_request_permission_canceled_by_session_cancel() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
approvals = true
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            let target = work_dir_beside(config_dir).join("doomed.txt");
            serde_json::json!([[
                { "type": "text", "text": "writing..." },
                { "type": "tool_use_start", "id": "call_w", "name": "write_file" },
                {
                    "type": "tool_use_end",
                    "input": { "path": target.to_str().unwrap(), "content": "x" }
                },
                { "type": "message_end", "stop_reason": "tool_use" }
            ]])
        })
        .build();
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "write");

    // Watch for the permission request to fire, then cancel without answering. The turn should
    // resolve `cancelled`.
    let sid_clone = session_id.clone();
    let mut saw_permission = false;
    let mut cancel_fired = false;
    let needle = format!("\"id\":{id}");
    let mut response: Option<serde_json::Value> = None;
    let deadline = Instant::now() + harness.window;
    while Instant::now() < deadline {
        let mut line = String::new();
        if harness.reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if value["method"] == "session/request_permission" {
            saw_permission = true;
            if !cancel_fired {
                let cancel_notif = serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": { "sessionId": sid_clone }
                });
                writeln!(harness.stdin, "{cancel_notif}").expect("write cancel");
                cancel_fired = true;
            }
        }
        if line.contains(&needle) && response_matches(&line, &needle) {
            response = Some(value);
            break;
        }
    }
    let response = response.unwrap_or_else(|| panic!("no prompt response"));
    assert!(saw_permission, "permission request must have fired");
    assert_eq!(
        response["result"]["stopReason"], "cancelled",
        "cancel during request_permission must resolve as Cancelled: {response}",
    );
}

/// `protocolVersion: 0` is the schema's parse-failure sentinel and is rejected with
/// `InvalidParams`, not silently clamped.
#[test]
fn acp_initialize_rejects_protocol_version_zero() {
    let install = Install::new();
    install.write_config(ACP_INVALID_PARAMS_CONFIG);

    let mut child = install
        .meka(&["acp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let stderr_pipe = child.stderr.take().expect("stderr");
    let mut reader = support::TimedLines::spawn(stdout);
    let _stderr_handle = std::thread::spawn(move || {
        let mut buffer = String::new();
        let mut stderr_reader = BufReader::new(stderr_pipe);
        while stderr_reader.read_line(&mut buffer).unwrap_or(0) > 0 {}
        buffer
    });

    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":0}}}}"#,
    )
    .expect("init");
    let lines = read_until(&mut reader, window(10), |line| line.contains("\"id\":1"));
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let line = lines
        .iter()
        .find(|line| line.contains("\"id\":1"))
        .expect("init response");
    let response: serde_json::Value = serde_json::from_str(line).expect("parse");
    assert_eq!(
        response["error"]["code"].as_i64(),
        Some(-32602),
        "protocolVersion 0 must be rejected with InvalidParams; got: {response}",
    );
}

/// Concurrent same-session prompt rejection. Two `session/prompt`s on the same `sessionId`: the
/// first stalls, the second must return `InvalidParams "session already has a prompt in flight"`
/// while the first still resolves normally.
#[test]
fn acp_session_prompt_rejects_concurrent_prompt_same_session() {
    let script = serde_json::json!([[
        { "type": "text", "text": "stalling" },
        { "type": "sleep", "ms": 2000 },
        { "type": "text", "text": "done" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id_a = harness.prompt(&session_id, "first");

    // Wait until A is mid-sleep, then fire B.
    let barrier = Instant::now() + Duration::from_secs(3);
    let _ = read_until(&mut harness.reader, barrier, |line| {
        line.contains("stalling")
    });

    let id_b = harness.prompt(&session_id, "second");
    let response_b = harness.await_response(id_b);
    assert_invalid_params(&response_b, "second concurrent prompt");

    // A still completes normally.
    let response_a = harness.await_response(id_a);
    assert_eq!(response_a["result"]["stopReason"], "end_turn");
}

/// Refusal stop reason: Claude `stop_reason: "refusal"` is mapped to `MockStopReason::Refusal` and
/// surfaces as the spec's `refusal` stop reason in the response. Mock needs the variant; add it.
#[test]
fn acp_session_prompt_refusal_stop_reason() {
    let script = serde_json::json!([[
        { "type": "text", "text": "I cannot help with that." },
        { "type": "message_end", "stop_reason": "refusal" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "do something disallowed");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "refusal",
        "refusal stop_reason must surface as ACP `refusal`: {response}",
    );
}

/// An *empty* refusal: Claude streams `stop_reason: "refusal"` with no body (as fable-5 does after
/// a search surfaces disallowed content). meka must still surface a visible stand-in message
/// instead of a blank turn. Regression for session fad3ed41, where the turn rendered nothing and
/// persisted an empty `[]` assistant message.
#[test]
fn acp_empty_refusal_surfaces_standin_message() {
    let script = serde_json::json!([[
        { "type": "message_end", "stop_reason": "refusal" }
    ]]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "trigger an empty refusal");
    let (updates, response) = harness.collect_updates(&session_id, id);
    assert_eq!(
        response["result"]["stopReason"], "refusal",
        "empty refusal must still surface as ACP `refusal`: {response}",
    );
    let dump = format!("{updates:?}");
    assert!(
        dump.contains("declined to respond"),
        "empty refusal must surface a stand-in agent_message_chunk; updates: {dump}",
    );
}

/// Regression: tool calls must run off the *presence* of `tool_use` blocks, not the reported stop
/// reason. Providers mislabel it - OpenAI Codex reports `completed` for a tool turn, and Claude
/// occasionally reports `end_turn` with `tool_use` present. Here the mock emits a complete
/// `read_file` call but ends the turn with `stop_reason: "end_turn"`; meka must still execute the
/// tool (and the turn completes normally) instead of orphaning the call and breaking the next
/// request.
#[test]
fn acp_tool_calls_execute_despite_non_tool_use_stop_reason() {
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"
"#;
    let mut harness = AcpTestHarness::builder()
        .config(config_toml)
        .pre_spawn(|config_dir| {
            // In the work directory beside the config dir: `read_file` refuses meka's own
            // directory below `unrestricted`.
            let target = fixture_beside(config_dir, "target.txt");
            std::fs::write(&target, "hello from mock test\n").expect("write target");
            serde_json::json!([
                [
                    { "type": "text", "text": "reading the file...\n" },
                    { "type": "tool_use_start", "id": "call_1", "name": "read_file" },
                    { "type": "tool_use_end", "input": { "path": target.to_str().unwrap() } },
                    // A complete tool call whose stop reason is not "tool_use".
                    { "type": "message_end", "stop_reason": "end_turn" }
                ],
                [
                    { "type": "text", "text": "done!" },
                    { "type": "message_end", "stop_reason": "end_turn" }
                ]
            ])
        })
        .build();
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "read the target file");
    let (updates, response) = harness.collect_updates(&session_id, id);

    // The tool must have executed despite the end_turn stop reason.
    assert!(
        updates.iter().any(|value| {
            let update = &value["params"]["update"];
            update["sessionUpdate"] == "tool_call_update" && update["status"] == "completed"
        }),
        "tool must execute even with a non-tool_use stop reason; updates: {updates:?}",
    );
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "turn should complete normally after the tool round; full response: {response}",
    );
}

/// The capability is the switch that makes a multi-root client send anything at all: Zed reads
/// `sessionCapabilities.additionalDirectories` and, when it's absent, silently drops every
/// workspace folder but the first before it ever reaches meka.
#[test]
fn acp_advertises_additional_directories_capability() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    let response = harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));
    assert!(
        response["result"]["agentCapabilities"]["sessionCapabilities"]["additionalDirectories"]
            .is_object(),
        "expected additionalDirectories to be advertised; got: {response}",
    );
}

/// A session created with extra roots reports them back on `session/list`, which is what a client
/// rebuilds its workspace from when picking a session out of history.
#[test]
fn acp_session_new_accepts_and_reports_additional_directories() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));

    let cwd = harness.config_dir();
    let extra = cwd.join("shared");
    std::fs::create_dir_all(&extra).expect("mkdir extra root");

    let response = harness.request(
        "session/new",
        serde_json::json!({
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [extra],
        }),
    );
    let session_id = response["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/new failed: {response}"))
        .to_string();

    let listed = harness.request("session/list", serde_json::json!({}));
    let sessions = listed["result"]["sessions"]
        .as_array()
        .unwrap_or_else(|| panic!("session/list failed: {listed}"));
    let row = sessions
        .iter()
        .find(|row| row["sessionId"] == session_id.as_str())
        .unwrap_or_else(|| panic!("session not listed: {listed}"));
    assert_eq!(
        row["additionalDirectories"],
        serde_json::json!([extra]),
        "session/list must report the roots the session was opened with; got: {row}",
    );
}

/// Every ACP door takes `cwd` through the one acceptor: a file is refused as `InvalidParams`
/// before a row exists, and a directory named through a symlink is recorded as the directory
/// itself, the spelling `/cd` and `POST /v1/sessions` record too.
#[test]
fn acp_session_new_accepts_a_cwd_only_as_an_existing_directory_spelled_canonically() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));

    let file = harness.config_dir().join("not-a-directory");
    std::fs::write(&file, b"x").expect("write the file");
    let refused = harness.request(
        "session/new",
        serde_json::json!({ "cwd": file, "mcpServers": [] }),
    );
    assert_invalid_params(&refused, "session/new on a file");
    assert!(
        refused["error"]["data"]
            .as_str()
            .is_some_and(|detail| detail.contains("not a directory")),
        "the refusal names the rule: {refused}"
    );
    let store = rusqlite::Connection::open(harness.database()).expect("open the store");
    let rows: i64 = store
        .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
        .expect("count the rows");
    assert_eq!(rows, 0, "a refused cwd leaves no row behind");
    drop(store);

    #[cfg(unix)]
    {
        let real = harness.config_dir().join("real");
        std::fs::create_dir(&real).expect("mkdir");
        let link = harness.config_dir().join("link");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        let created = harness.request(
            "session/new",
            serde_json::json!({ "cwd": link, "mcpServers": [] }),
        );
        let session_id = created["result"]["sessionId"]
            .as_str()
            .unwrap_or_else(|| panic!("session/new failed: {created}"))
            .to_string();
        let store = rusqlite::Connection::open(harness.database()).expect("open the store");
        let recorded: String = store
            .query_row(
                "SELECT cwd FROM sessions WHERE id = ?1",
                [&session_id],
                |row| row.get(0),
            )
            .expect("read the row");
        assert_eq!(
            std::path::PathBuf::from(recorded),
            std::fs::canonicalize(&real).expect("canonical"),
            "the row records the directory, not the link"
        );
    }
}

/// The spec requires absolute paths, and meka has no defensible base to resolve a relative one
/// against: joining to `cwd` would invent a root the client never named.
#[test]
fn acp_session_new_rejects_relative_additional_directory() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));

    let response = harness.request(
        "session/new",
        serde_json::json!({
            "cwd": harness.config_dir(),
            "mcpServers": [],
            "additionalDirectories": ["relative/path"],
        }),
    );
    assert_eq!(
        response["error"]["code"], -32602,
        "expected invalid_params; got: {response}",
    );
}

#[test]
fn acp_advertises_fork_capability() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    let response = harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));
    assert!(
        response["result"]["agentCapabilities"]["sessionCapabilities"]["fork"].is_object(),
        "expected fork to be advertised; got: {response}",
    );
}

/// `session/fork` refuses a sub-agent's id, and says so as a caller error before copying anything.
///
/// A fork of a sub-agent is a sibling under the same parent (`Store::fork_session_locked`), so
/// the copy is a worker too and `build_session_runtime` refuses to build it. That refusal alone is
/// safe but useless to a client: it arrives as `InternalError` -- for something the caller got
/// wrong -- naming the *copy's* id, which the client has never seen and which `discard_failed_fork`
/// has already deleted. The HTTP door answers 422 up front for the same reason; this is its
/// sibling, and it was left open when that one was closed.
///
/// The worker id comes from the store rather than from a listing, because ACP's `session/list`
/// shows root sessions and a client holding a sub-agent id got it some other way.
#[test]
fn acp_session_fork_refuses_a_sub_agent_before_copying_it() {
    let spawn_round = [
        serde_json::json!({ "type": "tool_use_start", "id": "t1", "name": "agent_spawn" }),
        serde_json::json!({ "type": "tool_use_end", "input": {"prompt": "count", "permission": "read"} }),
        serde_json::json!({ "type": "message_end", "stop_reason": "tool_use" }),
    ];
    let plain = [
        serde_json::json!({ "type": "text", "text": "ok" }),
        serde_json::json!({ "type": "message_end", "stop_reason": "end_turn" }),
    ];
    // The parent's spawning turn, the worker's own reply, then the parent's closing text.
    let script = serde_json::json!([spawn_round, plain, plain]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let source_id = harness.new_session();
    let id = harness.prompt(&source_id, "spawn a worker");
    harness.collect_updates(&source_id, id);

    let store = rusqlite::Connection::open(harness.database()).expect("open the store");
    let worker: String = store
        .query_row(
            "SELECT id FROM sessions WHERE parent_session_id IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .expect("the spawn should have left exactly one worker row");
    let before: i64 = store
        .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
        .expect("count");
    drop(store);

    let refused = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": worker,
            "cwd": harness.config_dir(),
            "mcpServers": [],
        }),
    );
    assert_invalid_params(&refused, "session/fork on a sub-agent");
    // `data`, not `message`: `invalid_params_error` puts the sentence there and leaves `message`
    // as the protocol's fixed "Invalid params".
    let detail = refused["error"]["data"].as_str().unwrap_or_default();
    assert!(
        detail.contains(&worker) && detail.contains("agent_followup"),
        "the refusal must name the id the client sent and the door that can continue it: {refused}"
    );

    let store = rusqlite::Connection::open(harness.database()).expect("reopen the store");
    let after: i64 = store
        .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
        .expect("count");
    assert_eq!(
        before, after,
        "refusing before the copy is the point; a rolled-back copy is a worse answer, not an \
         equivalent one"
    );
}

/// `session/load` refuses a sub-agent before it locks or writes its row.
///
/// The sibling of the `session/fork` test above, and the door that most needed one: the guard was
/// written for `session/load` and landed on `session/resume`, so the method editors actually call
/// kept every side effect. Nothing caught it, because no test existed for either handler's guard.
///
/// Asserts the write, not only the refusal. Reaching `build_session_runtime` is what makes this
/// visible from outside: by then the handler has taken the worker's file lock, rewritten `cwd`,
/// retired the background work its parent left running and replaced its roots, and answers
/// `-32603 Internal error` for something the caller got wrong. `cwd` is the one a test can see
/// cheaply, and it is not cosmetic: at `workspace` it is the writable boundary, and it is the
/// directory a scheduled gate is re-checked in.
#[test]
fn acp_session_load_refuses_a_sub_agent_before_touching_its_row() {
    let spawn_round = [
        serde_json::json!({ "type": "tool_use_start", "id": "t1", "name": "agent_spawn" }),
        serde_json::json!({ "type": "tool_use_end", "input": {"prompt": "count", "permission": "read"} }),
        serde_json::json!({ "type": "message_end", "stop_reason": "tool_use" }),
    ];
    let plain = [
        serde_json::json!({ "type": "text", "text": "ok" }),
        serde_json::json!({ "type": "message_end", "stop_reason": "end_turn" }),
    ];
    let script = serde_json::json!([spawn_round, plain, plain]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let source_id = harness.new_session();
    let id = harness.prompt(&source_id, "spawn a worker");
    harness.collect_updates(&source_id, id);

    let store = rusqlite::Connection::open(harness.database()).expect("open the store");
    let (worker, cwd_before): (String, Option<String>) = store
        .query_row(
            "SELECT id, cwd FROM sessions WHERE parent_session_id IS NOT NULL",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("the spawn should have left exactly one worker row");
    drop(store);

    // A directory that exists and differs from the worker's, so a handler that got as far as the
    // `cwd` write would leave a difference this test can see.
    let elsewhere = harness.config_dir().join("elsewhere");
    std::fs::create_dir_all(&elsewhere).expect("create the directory the load would move it to");

    let refused = harness.request(
        "session/load",
        serde_json::json!({
            "sessionId": worker,
            "cwd": elsewhere.to_string_lossy(),
            "mcpServers": [],
        }),
    );
    assert_invalid_params(&refused, "session/load on a sub-agent");
    let detail = refused["error"]["data"].as_str().unwrap_or_default();
    assert!(
        detail.contains(&worker) && detail.contains("agent_followup"),
        "the refusal must name the id the client sent and the door that can continue it: {refused}"
    );

    let store = rusqlite::Connection::open(harness.database()).expect("reopen the store");
    let cwd_after: Option<String> = store
        .query_row("SELECT cwd FROM sessions WHERE id = ?1", [&worker], |row| {
            row.get(0)
        })
        .expect("the worker row is still readable");
    assert_eq!(
        cwd_before, cwd_after,
        "refusing before the write is the point; a refusal that has already moved the session is \
         not an equivalent answer"
    );
}

/// `session/fork` mints a new session carrying the source's conversation, and leaves the source
/// open: both are addressable afterwards.
#[test]
fn acp_session_fork_creates_a_usable_copy() {
    let turn = [
        serde_json::json!({ "type": "text", "text": "ok" }),
        serde_json::json!({ "type": "message_end", "stop_reason": "end_turn" }),
    ];
    // One script entry per prompt: seed the source, then a turn on the fork, then one on the
    // source to prove forking left it usable.
    let script = serde_json::json!([turn, turn, turn]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));
    let source_id = harness.new_session();
    let id = harness.prompt(&source_id, "hello");
    harness.collect_updates(&source_id, id);

    let forked = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": harness.config_dir(),
            "mcpServers": [],
        }),
    );
    let fork_id = forked["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/fork failed: {forked}"))
        .to_string();
    assert_ne!(fork_id, source_id, "the fork is a distinct session");

    // The title is derived from the first user message, so matching titles mean the source's
    // conversation actually came across rather than the fork starting empty.
    let listed = harness.request("session/list", serde_json::json!({}));
    let sessions = listed["result"]["sessions"]
        .as_array()
        .unwrap_or_else(|| panic!("session/list failed: {listed}"));
    let title_of = |wanted: &str| -> serde_json::Value {
        sessions
            .iter()
            .find(|row| row["sessionId"] == wanted)
            .unwrap_or_else(|| panic!("session {wanted} not listed: {listed}"))["title"]
            .clone()
    };
    assert_eq!(title_of(&fork_id), serde_json::json!("hello"));
    assert_eq!(title_of(&fork_id), title_of(&source_id));

    // The copy is immediately promptable, which is only true if its conversation loaded cleanly.
    let id = harness.prompt(&fork_id, "again");
    let (_updates, response) = harness.collect_updates(&fork_id, id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "the forked session must accept a prompt; got: {response}",
    );

    // Forking does not close the source.
    let id = harness.prompt(&source_id, "still here");
    let (_updates, response) = harness.collect_updates(&source_id, id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "forking must leave the source session usable; got: {response}",
    );
}

/// ACP models fork as a session-*creation* request, so the workspace comes from the request rather
/// than the source. The source here has no extra roots, so the fork reporting one proves the
/// request's list was applied instead of inherited.
#[test]
fn acp_session_fork_applies_the_requests_additional_directories() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));

    let cwd = harness.config_dir();
    let extra = cwd.join("shared");
    std::fs::create_dir_all(&extra).expect("mkdir extra root");

    let response = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    let source_id = response["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/new failed: {response}"))
        .to_string();

    let forked = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [extra],
        }),
    );
    let fork_id = forked["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/fork failed: {forked}"))
        .to_string();

    let listed = harness.request("session/list", serde_json::json!({}));
    let sessions = listed["result"]["sessions"]
        .as_array()
        .unwrap_or_else(|| panic!("session/list failed: {listed}"));
    let row_of = |wanted: &str| {
        sessions
            .iter()
            .find(|row| row["sessionId"] == wanted)
            .unwrap_or_else(|| panic!("session {wanted} not listed: {listed}"))
    };
    assert_eq!(
        row_of(&fork_id)["additionalDirectories"],
        serde_json::json!([extra]),
        "the fork must carry the roots the request named; got: {}",
        row_of(&fork_id),
    );
    assert!(
        row_of(&source_id)["additionalDirectories"].is_null(),
        "and must not write them back onto the source; got: {}",
        row_of(&source_id),
    );
}

/// The realistic client flow: pick a session out of `session/list` and branch from it without
/// opening it first. Fork reads the database, so the source never has to be in the active map --
/// and forking must not implicitly adopt it either, or the client would be left holding a session
/// it never asked to open (and its on-disk lock).
#[test]
fn acp_session_fork_works_on_a_session_that_was_never_loaded() {
    let turn = [
        serde_json::json!({ "type": "text", "text": "ok" }),
        serde_json::json!({ "type": "message_end", "stop_reason": "end_turn" }),
    ];
    let script = serde_json::json!([turn, turn]);
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(script));

    // Seed a session, then close it so only its persisted row remains.
    let source_id = harness.new_session();
    let id = harness.prompt(&source_id, "seed");
    harness.collect_updates(&source_id, id);
    let closed = harness.request(
        "session/close",
        serde_json::json!({ "sessionId": source_id }),
    );
    assert!(closed["error"].is_null(), "session/close failed: {closed}");

    let forked = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": harness.config_dir(),
            "mcpServers": [],
        }),
    );
    let fork_id = forked["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("forking an unloaded session failed: {forked}"))
        .to_string();

    // The fork is live...
    let id = harness.prompt(&fork_id, "go");
    let (_updates, response) = harness.collect_updates(&fork_id, id);
    assert_eq!(response["result"]["stopReason"], "end_turn");

    // ...and the source stayed closed rather than being silently adopted.
    let reclose = harness.request(
        "session/close",
        serde_json::json!({ "sessionId": source_id }),
    );
    assert!(
        !reclose["error"].is_null(),
        "forking must not adopt the source session; got: {reclose}",
    );
}

#[test]
fn acp_session_fork_rejects_bad_input() {
    let mut harness = AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, None);
    harness.request("initialize", serde_json::json!({ "protocolVersion": 1 }));

    let cwd = harness.config_dir();
    let response = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    let source_id = response["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("session/new failed: {response}"))
        .to_string();

    let unknown = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": "9f1d4c2e-0000-4000-8000-000000000000",
            "cwd": cwd,
            "mcpServers": [],
        }),
    );
    assert_eq!(
        unknown["error"]["code"], -32602,
        "an unknown source must be invalid_params; got: {unknown}",
    );

    let relative_root = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": ["relative/path"],
        }),
    );
    assert_eq!(
        relative_root["error"]["code"], -32602,
        "expected invalid_params for a relative additional root; got: {relative_root}",
    );
}

const ACP_SCHEDULE_CONFIG: &str = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "unrestricted"
enabled = ["read", "unrestricted"]

[schedule]
poll_interval = "1s"
"#;

/// A scheduled job fires in an ACP session and its turn reaches the editor unsolicited: no
/// `session/prompt` is outstanding when the notifications arrive.
///
/// This is the property the whole ACP host rests on. Zed's `handle_session_notification` dispatches
/// on session id with no in-flight-prompt check, so an agent-initiated turn renders; if that ever
/// stopped being true the feature would run invisibly, and this test is what would notice.
#[test]
fn acp_scheduled_job_fires_without_a_prompt() {
    let script = serde_json::json!([
        // Turn 1: the agent schedules a one-shot two seconds out.
        [
            { "type": "tool_use_start", "id": "call_sched", "name": "schedule_create" },
            { "type": "tool_use_end", "input": {
                "prompt": "ACP_DELIVERED_MARKER",
                "at": "2s"
            }},
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "scheduled" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        // Turn 2 is the fire. Nothing on the client side asks for it.
        [
            { "type": "text", "text": "ACP_SCHEDULED_REPLY" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::builder()
        .config(ACP_SCHEDULE_CONFIG)
        .script(script)
        .window(Duration::from_secs(45))
        .build();

    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "remind me in two seconds");
    let (_updates, _response) = harness.collect_updates(&session_id, id);

    // Nothing is outstanding now: the prompt above has been answered. Anything that arrives from
    // here is agent-initiated. Wait for the scheduled turn to have written its reply rather than
    // a fixed span past the job's due time: the reply's notifications are sent before the row is
    // written, so once the row exists they are already queued ahead of the `session/list` the
    // drain below terminates on.
    let replied_by = Instant::now() + Duration::from_secs(45);
    loop {
        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        let replies: i64 = connection
            .query_row(
                "SELECT count(*) FROM messages WHERE session_id = ?1 AND role = 'assistant' \
                 AND content LIKE '%ACP_SCHEDULED_REPLY%'",
                rusqlite::params![&session_id],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if replies > 0 {
            break;
        }
        assert!(
            Instant::now() < replied_by,
            "the scheduled job never fired, or its reply was never written"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    let updates = harness.drain_unsolicited_updates(&session_id);

    let text_of = |kind: &str| -> String {
        updates
            .iter()
            .map(|update| &update["params"]["update"])
            .filter(|update| update["sessionUpdate"] == kind)
            .filter_map(|update| update["content"]["text"].as_str())
            .collect::<Vec<_>>()
            .join("")
    };

    // The prompt is shown as a user message, so the transcript explains the reply rather than
    // presenting an answer to a question the editor never saw asked.
    let user_text = text_of("user_message_chunk");
    assert!(
        user_text.contains("ACP_DELIVERED_MARKER"),
        "the job's prompt must be pushed as a user message; updates were:\n{updates:#?}",
    );
    assert!(
        user_text.contains("Scheduled job"),
        "and it must be marked as scheduled; updates were:\n{updates:#?}",
    );

    let agent_text = text_of("agent_message_chunk");
    assert!(
        agent_text.contains("ACP_SCHEDULED_REPLY"),
        "the agent's reply to the scheduled turn must reach the client; updates were:\n{updates:#?}",
    );
}

/// A scheduled turn has no `session/prompt` response to carry its outcome, so when it fails the
/// editor would otherwise show the job's prompt and nothing under it. The failure is said as a
/// `[meka warn]` chunk, the way the REPL prints it and `meka serve` posts a `failed` webhook.
#[test]
fn acp_reports_a_failed_scheduled_turn_as_a_warn_notice() {
    let script = serde_json::json!([
        [
            { "type": "tool_use_start", "id": "call_sched", "name": "schedule_create" },
            { "type": "tool_use_end", "input": {
                "prompt": "ACP_DELIVERED_MARKER",
                "at": "2s"
            }},
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "scheduled" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        // The fire: the provider is down for it.
        [
            { "type": "fail", "message": "ACP_PROVIDER_DOWN" }
        ]
    ]);
    let mut harness = AcpTestHarness::builder()
        .config(ACP_SCHEDULE_CONFIG)
        .script(script)
        .window(Duration::from_secs(45))
        .build();

    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "remind me in two seconds");
    let (_updates, _response) = harness.collect_updates(&session_id, id);

    // Nothing is outstanding now; whatever arrives is the fire's doing. Drained until the failure
    // notice shows, rather than after a fixed wait past the due time.
    let deadline = Instant::now() + Duration::from_secs(40);
    let mut chunks: Vec<String> = Vec::new();
    loop {
        chunks.extend(
            harness
                .drain_unsolicited_updates(&session_id)
                .iter()
                .map(|update| &update["params"]["update"])
                .filter(|update| update["sessionUpdate"] == "agent_message_chunk")
                .filter_map(|update| update["content"]["text"].as_str())
                .map(str::to_string),
        );
        if chunks.iter().any(|chunk| chunk.starts_with("[meka warn]")) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the failed fire was never reported to the editor; chunks were:\n{chunks:#?}"
        );
        std::thread::sleep(Duration::from_millis(250));
    }
    let notice = chunks
        .iter()
        .find(|chunk| chunk.starts_with("[meka warn]"))
        .expect("checked by the loop");
    assert!(
        notice.contains("scheduled job") && notice.contains("failed"),
        "the notice must say which turn failed: {notice}"
    );
    assert!(
        notice.contains("ACP_PROVIDER_DOWN"),
        "and carry the error, not only that there was one: {notice}"
    );
}

/// A compaction is the one thing that changes what the model can see without the editor doing
/// anything, and the automatic ones fire with nobody asking. It is said as a `[meka]` chunk so a
/// user whose next reply forgets the morning has been told why.
#[test]
fn acp_reports_a_compaction_as_a_meka_notice() {
    let script = serde_json::json!([
        [
            { "type": "tool_use_start", "id": "tu_1", "name": "context_compact" },
            { "type": "tool_use_end", "input": {} },
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        // The summarizer draws first; the model's own reply is the round after it.
        [
            { "type": "text", "text": "a summary" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "the reply" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let config = format!("{ACP_INVALID_PARAMS_CONFIG}\n[session]\ncompact_checkpoint = false\n");
    let mut harness = AcpTestHarness::spawn(&config, Some(script));
    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "compact yourself");
    let (updates, response) = harness.collect_updates(&session_id, id);

    let chunks: Vec<&str> = updates
        .iter()
        .filter(|value| value["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        .filter_map(|value| value["params"]["update"]["content"]["text"].as_str())
        .collect();
    let notice = chunks
        .iter()
        .find(|chunk| chunk.starts_with("[meka] compacted the conversation"))
        .unwrap_or_else(|| panic!("the compaction must reach the editor: {chunks:?}"));
    assert!(
        notice.contains("compaction 1"),
        "and say which compaction it was: {notice}"
    );
    assert_eq!(response["result"]["stopReason"], "end_turn");
}

/// The same, through `session/resume` rather than `session/load`.
///
/// `handle_resume_session` carries its own copy of the permission restore, and nothing exercised
/// it: `a_mode_set_through_acp_survives_a_reload` covers the twin in `handle_load_session`, and
/// `acp_session_resume_adopts_without_replay` asserts nothing about the level. Deleting the resume
/// path's block left the suite green -- and the result is a session running at the config default
/// while its row claims a higher level, which the scheduler's fire-time re-check then trusts. Fail
/// open, reached from the side that re-check cannot see.
#[test]
fn a_mode_set_through_acp_survives_a_resume() {
    const CONFIG: &str = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);
    let cwd = harness.config_dir();

    let new_response = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    assert_eq!(new_response["result"]["modes"]["currentModeId"], "read");
    let session_id = new_response["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();

    let set = harness.set_mode(&session_id, "unrestricted");
    assert!(set["result"].is_object(), "set_mode must succeed: {set}");

    let closed = harness.request(
        "session/close",
        serde_json::json!({ "sessionId": session_id }),
    );
    assert!(closed["result"].is_object(), "close must succeed: {closed}");

    let resumed = harness.request(
        "session/resume",
        serde_json::json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [] }),
    );
    assert_eq!(
        resumed["result"]["modes"]["currentModeId"], "unrestricted",
        "the resumed session fell back to the process default instead of the level its row \
         records: {resumed}"
    );
}

/// A level set through `session/set_mode` survives a reload.
///
/// Two halves of one invariant. `set_mode` only moved the in-memory cell, so the session row kept
/// whatever it was created with; the scheduler's live gate re-check reads that row, which meant a
/// gate authored after cycling to `unrestricted` was refused forever and one authored before
/// cycling down to `read` kept firing. Persisting alone is not enough either:
/// `build_session_runtime` seeds the permission from process config, so a reloaded session would
/// run at the default while its row claimed the level the gate check would then trust -- the same
/// fail-open from the other side.
///
/// Asserted through `currentModeId`, which is the client-visible form of the live cell, so this
/// fails if either the write or the restore is dropped.
#[test]
fn a_mode_set_through_acp_survives_a_reload() {
    const CONFIG: &str = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);
    let cwd = harness.config_dir();

    let new_response = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    assert_eq!(
        new_response["result"]["modes"]["currentModeId"], "read",
        "a fresh session starts at the configured default"
    );
    let session_id = new_response["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();

    let set = harness.set_mode(&session_id, "unrestricted");
    assert!(set["result"].is_object(), "set_mode must succeed: {set}");

    // Drop it from the live map so the reload rebuilds from the row rather than reusing the entry.
    let closed = harness.request(
        "session/close",
        serde_json::json!({ "sessionId": session_id }),
    );
    assert!(
        closed["result"].is_object(),
        "session/close must succeed: {closed}"
    );

    let loaded = harness.request(
        "session/load",
        serde_json::json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [] }),
    );
    assert_eq!(
        loaded["result"]["modes"]["currentModeId"], "unrestricted",
        "the reloaded session fell back to the process default instead of the level it was set to: \
         {loaded}"
    );
}

/// A turn runs on the profile the session's *row* names, not on whichever one the agent happened
/// to be assembled with.
///
/// Parking a resolved profile on the session entry whenever `session/set_config_option` could not
/// take the runtime mutex, with only `session/prompt` draining it, would let a scheduled fire or a
/// background-outcome turn run on the profile the user had left, and bill that account, while the
/// row, both pickers and the reported window all said otherwise. There is no park: the row is the
/// only carrier, and all three turn entry points read it.
///
/// The row is moved here by a second connection to the store rather than through
/// `session/set_config_option`, for the reason `tests/multiprocess.rs` gives at length: what is
/// being checked is that meka reads a change it did not make itself, which is exactly the position
/// the out-of-band entry points are in.
///
/// `/status` is the observation because it reports the model off `runtime.agent`'s own binding, and
/// it runs as a turn -- so it goes through the same door a scheduled fire does.
#[test]
fn an_acp_turn_follows_the_provider_its_row_names() {
    let config_toml = r#"
default_profile = "alpha"

[accounts.alpha]
backend = "anthropic-messages"

[profiles.alpha]
account = "alpha"
model = "model-from-alpha"

[accounts.beta]
backend = "anthropic-messages"

[profiles.beta]
account = "beta"
model = "model-from-beta"
"#;
    let mut harness = AcpTestHarness::spawn(config_toml, None);
    let session_id = harness.new_session();

    let id = harness.prompt(&session_id, "/status");
    let (updates, _response) = harness.collect_updates(&session_id, id);
    assert!(
        agent_text(&updates).contains("model-from-alpha"),
        "the session should start on the default profile; updates: {updates:?}"
    );

    let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
    let moved = connection
        .execute("UPDATE sessions SET profile = 'beta' WHERE id = ?1", [
            &session_id,
        ])
        .expect("repin the session");
    assert_eq!(moved, 1, "the repin matched no row");
    drop(connection);

    let id = harness.prompt(&session_id, "/status");
    let (updates, _response) = harness.collect_updates(&session_id, id);
    let text = agent_text(&updates);
    assert!(
        text.contains("model-from-beta"),
        "the turn ran on the profile the agent was built with rather than the one its row names; \
         updates: {updates:?}"
    );
    assert!(
        !text.contains("model-from-alpha"),
        "and must not still be reporting the old one: {updates:?}"
    );
}

/// Every `agent_message_chunk` in `updates`, concatenated.
fn agent_text(updates: &[serde_json::Value]) -> String {
    updates
        .iter()
        .filter(|value| value["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        .filter_map(|value| value["params"]["update"]["content"]["text"].as_str())
        .collect()
}

/// Whether an attachment is admissible follows the session's *row*, not a flag cached when the
/// session was created.
///
/// `docs/book/src/usage/acp.md` promises that a session moved onto a text-only profile refuses
/// attachments even on a connection whose `initialize` advertised `image`, and nothing defended it:
/// the entry carried a `vision` boolean that `session/set_config_option` pushed at, so every test
/// that moved a profile also moved the flag by hand and could not tell the two apart.
///
/// The row is moved here by a second connection to the store, exactly as
/// `an_acp_turn_follows_the_provider_its_row_names` does and for the same reason: what is being
/// checked is that meka reads a change it did not make in this process.
#[test]
fn an_acp_prompt_judges_an_image_against_the_profile_its_row_names() {
    let config_toml = r#"
default_profile = "seeing"

[accounts.seeing]
backend = "anthropic-messages"

[profiles.seeing]
account = "seeing"
model = "model-with-eyes"
vision = true

[accounts.blind]
backend = "anthropic-messages"

[profiles.blind]
account = "blind"
model = "model-without-eyes"
vision = false
"#;
    let mut harness = AcpTestHarness::spawn(config_toml, None);
    let session_id = harness.new_session();

    // A real 1x1 PNG: the payload is decoded before the profile is consulted, so a broken one
    // would be refused for being broken and never reach the decision under test.
    let image = |session_id: &str| {
        serde_json::json!({
            "sessionId": session_id,
            "prompt": [{
                "type": "image",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
                "mimeType": "image/png"
            }]
        })
    };

    // On `seeing`, the block is admitted: it gets past parsing and fails later, on the empty mock
    // script, rather than being refused as an unsupported content type.
    let accepted = harness.request("session/prompt", image(&session_id));
    let rendered = accepted.to_string();
    assert!(
        !rendered.contains("vision"),
        "a vision-enabled profile must admit the block: {accepted}"
    );

    let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
    let moved = connection
        .execute("UPDATE sessions SET profile = 'blind' WHERE id = ?1", [
            &session_id,
        ])
        .expect("repin the session");
    assert_eq!(moved, 1, "the repin matched no row");
    drop(connection);

    let rejected = harness.request("session/prompt", image(&session_id));
    assert_invalid_params(
        &rejected,
        "image block after the row moved to a text-only profile",
    );
    assert!(
        rejected.to_string().contains("vision"),
        "the refusal must name why: {rejected}"
    );
}

/// A scheduled fire is a turn, so it runs on the profile the session's row names -- or it does not
/// run at all.
///
/// A fire is the entry point most exposed to a parked profile: with only `session/prompt` draining
/// one, a fire would go on billing the account the user had left.
/// `an_acp_turn_follows_the_provider_its_row_names` covers the prompt path; this covers
/// `run_wakeup`.
///
/// The row is moved to a profile that is not configured, because that is the one difference a
/// scripted provider cannot hide: the mock stands in for every profile, so a fire on the wrong one
/// is indistinguishable from a fire on the right one *unless* the right one cannot resolve. With
/// the fix the job defers and nothing reaches the editor; without it the turn runs on whatever the
/// agent was assembled with and the reply arrives.
#[test]
fn an_acp_scheduled_fire_refuses_a_profile_the_row_no_longer_names() {
    let script = serde_json::json!([
        [
            { "type": "tool_use_start", "id": "call_sched", "name": "schedule_create" },
            { "type": "tool_use_end", "input": {
                "prompt": "ACP_DELIVERED_MARKER",
                "at": "2s"
            }},
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "scheduled" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        // Only reached if the fire runs on the agent's own profile instead of the row's.
        [
            { "type": "text", "text": "ACP_SCHEDULED_REPLY" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]);
    let mut harness = AcpTestHarness::builder()
        .config(ACP_SCHEDULE_CONFIG)
        .script(script)
        .window(Duration::from_secs(45))
        .build();

    let session_id = harness.new_session();
    let id = harness.prompt(&session_id, "remind me in two seconds");
    let (_updates, _response) = harness.collect_updates(&session_id, id);

    // Somebody other than this connection repins the session onto a profile that has since left
    // `config.toml` -- the state `look_up_profile` refuses by name.
    let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
    let moved = connection
        .execute("UPDATE sessions SET profile = 'retired' WHERE id = ?1", [
            &session_id,
        ])
        .expect("repin the session");
    assert_eq!(moved, 1, "the repin matched no row");
    drop(connection);

    // Past the 2s due time and several 1s ticks.
    std::thread::sleep(Duration::from_secs(6));
    let updates = harness.drain_unsolicited_updates(&session_id);
    let rendered = format!("{updates:#?}");

    assert!(
        !rendered.contains("ACP_SCHEDULED_REPLY"),
        "the fire ran on the profile the agent was assembled with rather than the one its row \
         names; updates were:\n{rendered}"
    );
    assert!(
        !rendered.contains("ACP_DELIVERED_MARKER"),
        "and its prompt must not have been pushed either, since no turn should have started; \
         updates were:\n{rendered}"
    );
}

/// `session/set_config_option` is how an ACP client moves a session onto another provider profile,
/// and until now nothing drove it at all.
///
/// A mutation sweep replaced the whole handler with `Ok(())`, emptied `build_config_options`,
/// deleted the `!` from its configured-profile check and flipped the `!=` that decides whether the
/// row is written, and the suite stayed green through every one. The row is what the next turn
/// resolves against, so a switch that does not reach it is a switch that did not happen; this
/// asserts the row by reading the value back out of `configOptions`, which `build_config_options`
/// sources from the row rather than from the live agent.
#[test]
fn acp_set_config_option_moves_the_session_onto_another_profile() {
    const CONFIG: &str = r#"
default_profile = "mock"

[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[accounts.other]
backend = "openai-responses"

[profiles.other]
account = "other"
model = "gpt-5.6-sol"

[permissions]
default = "read"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);
    let cwd = harness.config_dir();
    let created = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    let session_id = created["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();

    // The pickers are advertised at creation, so a client has something to render before the first
    // prompt.
    let provider_option = |response: &serde_json::Value| -> serde_json::Value {
        response["result"]["configOptions"]
            .as_array()
            .unwrap_or_else(|| panic!("no configOptions: {response}"))
            .iter()
            .find(|option| option["id"] == "profile")
            .unwrap_or_else(|| panic!("no provider option: {response}"))
            .clone()
    };
    let created_option = provider_option(&created);
    assert_eq!(
        created_option["currentValue"], "mock",
        "a new session starts on the host default: {created_option}"
    );
    let offered: Vec<String> = created_option["options"]
        .as_array()
        .expect("select options")
        .iter()
        .map(|option| option["value"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(offered, vec!["mock".to_string(), "other".to_string()]);

    // The switch itself. `currentValue` comes back off the row, so this fails if the row was not
    // written -- which is exactly what flipping the no-op filter to `==` produces.
    let switched = harness.request(
        "session/set_config_option",
        serde_json::json!({ "sessionId": session_id, "configId": "profile", "value": "other" }),
    );
    assert_eq!(
        provider_option(&switched)["currentValue"],
        "other",
        "the row must name the new profile: {switched}"
    );

    // And a name the config does not have is refused rather than recorded, because a row naming
    // nothing would fail every later turn on this session.
    let refused = harness.request(
        "session/set_config_option",
        serde_json::json!({ "sessionId": session_id, "configId": "profile", "value": "ghost" }),
    );
    assert!(
        refused["error"].is_object(),
        "an unconfigured profile must be refused: {refused}"
    );
    // Against the whole error object: ACP puts the detail in `data`, and `message` is the generic
    // JSON-RPC "Invalid params".
    let detail = refused["error"].to_string();
    assert!(
        detail.contains("no profile named 'ghost'") && detail.contains("other"),
        "the refusal should name the problem and list the configured profiles: {detail}"
    );

    // The refusal must not have moved the row on its way out.
    let after = harness.request(
        "session/set_config_option",
        serde_json::json!({ "sessionId": session_id, "configId": "profile", "value": "other" }),
    );
    assert_eq!(provider_option(&after)["currentValue"], "other");
}

/// A profile switch while a prompt holds the session is refused with `InvalidParams`, the answer
/// a second prompt gets, and the row stays where it was. Writing the row and deferring the agent's
/// move to the next turn would leave the two disagreeing for the length of the turn.
#[test]
fn acp_set_config_option_refuses_a_profile_switch_while_a_turn_is_in_flight() {
    const CONFIG: &str = r#"
default_profile = "mock"

[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[accounts.other]
backend = "openai-responses"

[profiles.other]
account = "other"
model = "gpt-5.6-sol"
"#;
    let script = serde_json::json!([[
        { "type": "text", "text": "stalling" },
        { "type": "sleep", "ms": 2000 },
        { "type": "text", "text": "done" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let mut harness = AcpTestHarness::spawn(CONFIG, Some(script));
    let session_id = harness.new_session();
    let prompt_id = harness.prompt(&session_id, "first");
    let barrier = Instant::now() + Duration::from_secs(3);
    let _ = read_until(&mut harness.reader, barrier, |line| {
        line.contains("stalling")
    });

    let refused = harness.request(
        "session/set_config_option",
        serde_json::json!({ "sessionId": session_id, "configId": "profile", "value": "other" }),
    );
    assert_invalid_params(&refused, "profile switch during a turn");
    assert!(
        refused["error"].to_string().contains("turn is in flight"),
        "the refusal should say what is holding the session: {refused}"
    );

    let response = harness.await_response(prompt_id);
    assert_eq!(response["result"]["stopReason"], "end_turn");

    // The row itself: the refused switch must not have written it.
    let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
    let recorded: String = connection
        .query_row(
            "SELECT profile FROM sessions WHERE id = ?1",
            [&session_id],
            |row| row.get(0),
        )
        .expect("the session's row");
    assert_eq!(recorded, "mock", "the row must still name the old profile");
}

/// `approvals` is a session config option like `profile`: advertised at creation, written to the
/// row when set, and read back off it, so a resume and a scheduled fire see what the editor set.
#[test]
fn acp_set_config_option_turns_approvals_on_and_records_it() {
    const CONFIG: &str = r#"
default_profile = "mock"

[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[permissions]
default = "read"
enabled = ["read", "unrestricted"]
"#;
    let mut harness = AcpTestHarness::spawn(CONFIG, None);
    let cwd = harness.config_dir();
    let created = harness.request(
        "session/new",
        serde_json::json!({ "cwd": cwd, "mcpServers": [] }),
    );
    let session_id = created["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();
    let approvals_option = |response: &serde_json::Value| -> serde_json::Value {
        response["result"]["configOptions"]
            .as_array()
            .unwrap_or_else(|| panic!("no configOptions: {response}"))
            .iter()
            .find(|option| option["id"] == "approvals")
            .unwrap_or_else(|| panic!("no approvals option: {response}"))
            .clone()
    };
    assert_eq!(
        approvals_option(&created)["currentValue"],
        false,
        "off unless the config says so: {created}"
    );

    let switched = harness.request(
        "session/set_config_option",
        // The protocol flattens the value into the request: a `type` discriminator beside the
        // `value`, where a picker sends a bare value id and no `type`.
        serde_json::json!({
            "sessionId": session_id,
            "configId": "approvals",
            "type": "boolean",
            "value": true,
        }),
    );
    assert_eq!(
        approvals_option(&switched)["currentValue"],
        true,
        "the option reports the switch: {switched}"
    );
    let store = rusqlite::Connection::open(harness.database()).expect("open the store");
    let recorded: i64 = store
        .query_row(
            "SELECT approvals FROM sessions WHERE id = ?1",
            [&session_id],
            |row| row.get(0),
        )
        .expect("the session row");
    assert_eq!(
        recorded, 1,
        "the switch is on the row, where a resume and a scheduled fire read it"
    );
}

/// An ACP scheduled fire carries a cancellation that was waiting, as `meka serve`'s does.
///
/// The ACP half of the third door, and the half with no coverage: deleting the fold there left the
/// suite green. The job row is seeded directly so the fire is due immediately; what is under test
/// is what the turn's prompt carries, not the scheduler's arithmetic.
#[test]
fn an_acp_scheduled_fire_carries_a_cancellation_that_was_waiting() {
    let script = serde_json::json!([[
        { "type": "text", "text": "ran the job" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[background]
enabled = true

[schedule]
poll_interval = "200ms"
"#;
    let mut harness = AcpTestHarness::spawn(config_toml, Some(script));
    let session_id = harness.new_session();
    {
        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        let now = chrono::Utc::now();
        connection
            .execute(
                "INSERT INTO background_tasks \
                 (id, session_id, tool_name, label, status, outcome, started_at, finished_at) \
                 VALUES (?1, ?2, 'execute_command', 'sleep 900', 'canceled', NULL, ?3, ?3)",
                rusqlite::params![
                    uuid::Uuid::new_v4().to_string(),
                    &session_id,
                    now.to_rfc3339()
                ],
            )
            .expect("seed the canceled task");
        // Due a minute ago, so the very next sweep fires it.
        connection
            .execute(
                "INSERT INTO scheduled_jobs \
                 (id, session_id, kind, spec, prompt, gate_kind, gate_spec_json, gate_last_output, \
                  gate_permission, created_at, last_fired_at, next_fire_at) \
                 VALUES (?1, ?2, 'every', '60s', 'PROBE_ACP_FIRE', NULL, NULL, NULL, NULL, ?3, \
                         NULL, ?4)",
                rusqlite::params![
                    uuid::Uuid::new_v4().to_string(),
                    &session_id,
                    now.to_rfc3339(),
                    (now - chrono::Duration::seconds(60)).to_rfc3339(),
                ],
            )
            .expect("seed the due job");
    }

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        let fired: Option<String> = connection
            .query_row(
                "SELECT content FROM messages WHERE session_id = ?1 AND role IN ('user', 'user_blocks') \
                 AND content LIKE '%PROBE_ACP_FIRE%' ORDER BY id ASC LIMIT 1",
                rusqlite::params![&session_id],
                |row| row.get(0),
            )
            .ok();
        if let Some(text) = fired {
            assert!(
                text.contains("was canceled"),
                "the job's prompt must carry the outcome that was waiting: {text}"
            );
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the scheduled job never fired"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

/// `session/cancel` stops a turn a scheduled job started, not only one the editor prompted.
///
/// A scheduled turn publishes its token the same way, so the editor's stop button reaches it. That
/// was asserted only by a comment: the whole path is driven by the poller, so no prompt response
/// carries its stop reason and nothing here observed it. What proves it is the far side of the
/// script's sleep never being reached.
#[test]
fn an_acp_scheduled_turn_stops_when_the_editor_cancels() {
    let script = serde_json::json!([[
        { "type": "text", "text": "job turn running" },
        { "type": "sleep", "ms": 5000 },
        { "type": "text", "text": "PAST_THE_CANCEL" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[schedule]
poll_interval = "200ms"
"#;
    let mut harness = AcpTestHarness::spawn(config_toml, Some(script));
    let session_id = harness.new_session();
    {
        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        let now = chrono::Utc::now();
        // Due a minute ago, so the very next sweep fires it.
        connection
            .execute(
                "INSERT INTO scheduled_jobs \
                 (id, session_id, kind, spec, prompt, gate_kind, gate_spec_json, gate_last_output, \
                  gate_permission, created_at, last_fired_at, next_fire_at) \
                 VALUES (?1, ?2, 'every', '3600s', 'PROBE_CANCEL_FIRE', NULL, NULL, NULL, NULL, \
                         ?3, NULL, ?4)",
                rusqlite::params![
                    uuid::Uuid::new_v4().to_string(),
                    &session_id,
                    now.to_rfc3339(),
                    (now - chrono::Duration::seconds(60)).to_rfc3339(),
                ],
            )
            .expect("seed the due job");
    }

    // Cancel only once the job's turn is provably streaming, or the cancel lands between turns and
    // the latch would carry it instead, which is a different path.
    let barrier = Instant::now() + Duration::from_secs(20);
    let started = read_until(&mut harness.reader, barrier, |line| {
        line.contains("job turn running")
    });
    assert!(
        started.iter().any(|line| line.contains("job turn running")),
        "the scheduled job never started a turn",
    );
    harness.cancel(&session_id);

    // Past the script's sleep, so an uncanceled turn would have written its far side by now.
    std::thread::sleep(Duration::from_secs(8));
    let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
    let reached: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?1 \
             AND content LIKE '%PAST_THE_CANCEL%'",
            rusqlite::params![&session_id],
            |row| row.get(0),
        )
        .expect("count messages");
    assert_eq!(
        reached, 0,
        "session/cancel must stop a scheduled turn; it ran on to the end of its script",
    );
}

/// The ACP poller delivers a completed outcome as a turn, exactly once.
///
/// Nothing drove `deliver_outcomes` at all. A completed task is the case that *does* warrant a
/// turn, which is what makes it the one that exercises this function: the claim, the stamp, the
/// turn, and that no later sweep repeats it.
///
/// Its claimed-filter is not pinned here and cannot be by a test with one claimer -- the filter is
/// a no-op until something else claims the same row concurrently. The rule itself is covered by
/// `background::tests::a_report_carries_only_the_outcomes_the_stamp_won`.
#[test]
fn the_acp_poller_delivers_a_finished_task_once() {
    let script = serde_json::json!([[
        { "type": "text", "text": "noted the build finished" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[background]
enabled = true

[schedule]
poll_interval = "200ms"
"#;
    let mut harness = AcpTestHarness::spawn(config_toml, Some(script));
    let session_id = harness.new_session();
    {
        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        let now = chrono::Utc::now().to_rfc3339();
        connection
            .execute(
                "INSERT INTO background_tasks \
                 (id, session_id, tool_name, label, status, outcome, started_at, finished_at) \
                 VALUES (?1, ?2, 'execute_command', 'cargo build', 'completed', '42 passed', \
                         ?3, ?3)",
                rusqlite::params![uuid::Uuid::new_v4().to_string(), &session_id, now],
            )
            .expect("seed the finished task");
    }

    // The poller has one round to spend. A second delivery would find none and the turn would fail.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        let carried: i64 = connection
            .query_row(
                "SELECT count(*) FROM messages WHERE session_id = ?1 AND role IN ('user', 'user_blocks') \
                 AND content LIKE '%cargo build%'",
                rusqlite::params![&session_id],
                |row| row.get(0),
            )
            .expect("count");
        if carried > 0 {
            assert_eq!(carried, 1, "the outcome must be reported exactly once");
            let delivered: Option<String> = connection
                .query_row(
                    "SELECT delivered_at FROM background_tasks WHERE session_id = ?1",
                    rusqlite::params![&session_id],
                    |row| row.get(0),
                )
                .expect("read the task");
            assert!(
                delivered.is_some(),
                "and stamped, so no later sweep repeats it"
            );
            // Several more poll intervals: a sweep that re-delivered would add a second message.
            std::thread::sleep(std::time::Duration::from_secs(2));
            let again: i64 = connection
                .query_row(
                    "SELECT count(*) FROM messages WHERE session_id = ?1 AND role IN ('user', 'user_blocks') \
                     AND content LIKE '%cargo build%'",
                    rusqlite::params![&session_id],
                    |row| row.get(0),
                )
                .expect("count");
            assert_eq!(again, 1, "and it stays reported once");
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the poller never delivered the finished task"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

/// The ACP poller leaves a cancellation alone instead of spending a turn on it.
///
/// The sibling of the fold test below: one proves the outcome arrives, this proves it does not
/// arrive as a turn of its own. Both halves are needed, because either failure mode alone looks
/// like the other passing.
#[test]
fn the_acp_poller_does_not_spend_a_turn_on_a_cancellation() {
    // One round only. A poller that delivers would consume it and the prompt below would fail.
    let script = serde_json::json!([[
        { "type": "text", "text": "answered" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[background]
enabled = true

[schedule]
poll_interval = "200ms"
"#;
    let mut harness = AcpTestHarness::spawn(config_toml, Some(script));
    let session_id = harness.new_session();
    {
        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        connection
            .execute(
                "INSERT INTO background_tasks \
                 (id, session_id, tool_name, label, status, outcome, started_at, finished_at) \
                 VALUES (?1, ?2, 'execute_command', 'sleep 900', 'canceled', NULL, ?3, ?3)",
                rusqlite::params![
                    uuid::Uuid::new_v4().to_string(),
                    &session_id,
                    chrono::Utc::now().to_rfc3339(),
                ],
            )
            .expect("seed the canceled task");
    }

    // Several poll intervals, so a delivering poller has every chance to prove itself.
    std::thread::sleep(std::time::Duration::from_secs(2));
    {
        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        let messages: i64 = connection
            .query_row(
                "SELECT count(*) FROM messages WHERE session_id = ?1",
                rusqlite::params![&session_id],
                |row| row.get(0),
            )
            .expect("count messages");
        assert_eq!(
            messages, 0,
            "a cancellation must add no message of its own: it would be a turn boundary with no \
             turn behind it"
        );
        let delivered: Option<String> = connection
            .query_row(
                "SELECT delivered_at FROM background_tasks WHERE session_id = ?1",
                rusqlite::params![&session_id],
                |row| row.get(0),
            )
            .expect("read the task");
        assert!(
            delivered.is_none(),
            "and it must stay in the pool for the next real turn to carry"
        );
    }

    // The round the poller must not have eaten is this prompt's.
    let id = harness.prompt(&session_id, "what is in this CSV?");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "the prompt must still have a round to run on: {response}"
    );
}

/// A canceled task rides the editor's next prompt, in ACP as it does in `meka serve`.
///
/// ACP has no test for any of the background-outcome path, so every guard in it -- the poller's
/// `wakes_a_host` branch, this fold, the `PromptRetention` it runs under -- could be deleted with
/// the suite green. Seeded straight into the store rather than run for real: what is under test is
/// what the editor's prompt carries, not whether `sleep` works.
#[test]
fn a_canceled_task_rides_the_editors_next_prompt() {
    let script = serde_json::json!([[
        { "type": "text", "text": "answered" },
        { "type": "message_end", "stop_reason": "end_turn" }
    ]]);
    let config_toml = r#"
[accounts.mock]
backend = "anthropic-messages"

[profiles.mock]
account = "mock"
model = "claude-sonnet-4-5"

[background]
enabled = true
"#;
    let mut harness = AcpTestHarness::spawn(config_toml, Some(script));
    let session_id = harness.new_session();

    // A terminal, undelivered, unannounced task: exactly what `/task cancel` leaves behind.
    {
        let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
        connection
            .execute(
                "INSERT INTO background_tasks \
                 (id, session_id, tool_name, label, status, outcome, started_at, finished_at) \
                 VALUES (?1, ?2, 'execute_command', 'sleep 900', 'canceled', NULL, ?3, ?3)",
                rusqlite::params![
                    uuid::Uuid::new_v4().to_string(),
                    &session_id,
                    chrono::Utc::now().to_rfc3339(),
                ],
            )
            .expect("seed the canceled task");
    }

    let id = harness.prompt(&session_id, "what is in this CSV?");
    let response = harness.await_response(id);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "the prompt must run: {response}"
    );

    let connection = rusqlite::Connection::open(harness.database()).expect("open the store");
    let user_text: String = connection
        .query_row(
            "SELECT content FROM messages WHERE session_id = ?1 AND role IN ('user', 'user_blocks') \
             ORDER BY id DESC LIMIT 1",
            rusqlite::params![&session_id],
            |row| row.get(0),
        )
        .expect("the prompt's user message");
    assert!(
        user_text.contains("was canceled"),
        "the outcome must ride inside the editor's own prompt: {user_text}"
    );
    assert!(
        user_text.contains("what is in this CSV?"),
        "and the prompt has to still be there: {user_text}"
    );

    let delivered: Option<String> = connection
        .query_row(
            "SELECT delivered_at FROM background_tasks WHERE session_id = ?1",
            rusqlite::params![&session_id],
            |row| row.get(0),
        )
        .expect("read the task");
    assert!(
        delivered.is_some(),
        "and riding a turn is a delivery, so it must be stamped"
    );
}

/// `session/fork` refuses a source this editor is prompting, as it refuses one another process is
/// writing and for the same reason: the turn persisted its prompt before the provider answered,
/// so the copy would end on a prompt nothing answered. `InvalidParams`, the answer a second prompt
/// on the session gets.
#[test]
fn acp_session_fork_refuses_a_source_with_a_prompt_in_flight() {
    // The leading chunk is the starting gun: the fork is sent once it has been seen, so the turn
    // is parked in the sleep and holds the conversation.
    let turn = [
        serde_json::json!({ "type": "text", "text": "starting..." }),
        serde_json::json!({ "type": "sleep", "ms": 1500 }),
        serde_json::json!({ "type": "text", "text": "done" }),
        serde_json::json!({ "type": "message_end", "stop_reason": "end_turn" }),
    ];
    let mut harness =
        AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(serde_json::json!([turn])));
    let source_id = harness.new_session();
    let prompt = harness.prompt(&source_id, "slow one");
    let seen = read_until(&mut harness.reader, window(15), |line| {
        line.contains("starting...")
    });
    assert!(
        seen.iter().any(|line| line.contains("starting...")),
        "the premise: the turn must be under way before the fork is sent"
    );

    let refused = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": harness.config_dir(),
            "mcpServers": [],
        }),
    );
    assert_invalid_params(&refused, "session/fork on a source with a prompt in flight");
    let detail = refused["error"]["data"].as_str().unwrap_or_default();
    assert!(
        detail.contains("turn is in flight") && detail.contains("fork"),
        "the refusal names what it refused and why: {refused}"
    );

    let (_updates, response) = harness.collect_updates(&source_id, prompt);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "and the turn it declined to interrupt completes; got: {response}",
    );

    let store = rusqlite::Connection::open(harness.database()).expect("open the store");
    let sessions: i64 = store
        .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
        .expect("count");
    assert_eq!(sessions, 1, "no copy was written");
    let messages: i64 = store
        .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
        .expect("count");
    assert_eq!(
        messages, 2,
        "the source's conversation is what the turn alone produced"
    );
    drop(store);

    // Between turns the same request is what it always was.
    let forked = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": harness.config_dir(),
            "mcpServers": [],
        }),
    );
    assert!(
        forked["result"]["sessionId"].is_string(),
        "a source between turns forks: {forked}"
    );
}

/// `session/fork` refuses a source another process is writing, as `meka session fork` and
/// `POST /v1/sessions/{id}/fork` do: a copy taken mid-turn ends on a user message nothing
/// answered. A source this editor has open is its own and is not probed; one it has closed is, and
/// a second descriptor from this test stands in for the other process.
#[test]
fn acp_session_fork_refuses_a_source_another_process_holds() {
    let turn = [
        serde_json::json!({ "type": "text", "text": "ok" }),
        serde_json::json!({ "type": "message_end", "stop_reason": "end_turn" }),
    ];
    let mut harness =
        AcpTestHarness::spawn(ACP_INVALID_PARAMS_CONFIG, Some(serde_json::json!([turn])));

    let source_id = harness.new_session();
    let id = harness.prompt(&source_id, "seed");
    harness.collect_updates(&source_id, id);
    let closed = harness.request(
        "session/close",
        serde_json::json!({ "sessionId": source_id }),
    );
    assert!(closed["error"].is_null(), "session/close failed: {closed}");

    let lock_path = harness
        .install
        .data_dir()
        .join("locks")
        .join(format!("{source_id}.lock"));
    let held = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .unwrap_or_else(|error| panic!("open {}: {error}", lock_path.display()));
    held.try_lock()
        .expect("the closed session's lock is free for this test to take");

    let refused = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": harness.config_dir(),
            "mcpServers": [],
        }),
    );
    assert_invalid_params(&refused, "session/fork on a source another process holds");
    let detail = refused["error"]["data"].as_str().unwrap_or_default();
    assert!(
        detail.contains("another process") && detail.contains(&source_id),
        "the refusal must say who has it and name the source: {refused}"
    );
    let listed = harness.request("session/list", serde_json::json!({}));
    assert_eq!(
        listed["result"]["sessions"].as_array().map(Vec::len),
        Some(1),
        "and no copy was left behind: {listed}"
    );

    drop(held);
    let forked = harness.request(
        "session/fork",
        serde_json::json!({
            "sessionId": source_id,
            "cwd": harness.config_dir(),
            "mcpServers": [],
        }),
    );
    assert!(
        forked["result"]["sessionId"].is_string(),
        "a released source forks: {forked}"
    );
}
