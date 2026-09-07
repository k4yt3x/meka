//! The REPL, driven through a real pseudo-terminal.
//!
//! Everything else in `tests/` runs meka with pipes, which the interactive shell refuses: reedline
//! needs a terminal, and additionally queries the cursor position (DSR, `ESC[6n`) during its first
//! paint, so even `script -qec` fails. That left `run_repl` and everything downstream of it -- the
//! whole `[display]` blank-line contract -- with no automated coverage at all, and a dozen spacing
//! defects accumulated in it unnoticed.
//!
//! This owns the master side of a pty: it answers DSR itself and feeds input whenever the child
//! goes quiet, which is the only "it is at a prompt again" signal available without parsing
//! reedline's paint. The captured bytes are then replayed through the handful of control sequences
//! meka and reedline actually emit, because a raw comparison would be reading carriage returns and
//! erase-to-end-of-line as though they were content.
//!
//! Unix only, and only where the scripted provider (`MEKA_MOCK_PROVIDER`) is compiled in, which a
//! release build does without the `mock-provider` feature.

#![cfg(all(unix, any(debug_assertions, feature = "mock-provider")))]
// Same rationale as the other integration tests: a failed assumption here is a broken test, and
// panicking says so at the point it broke.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "tests panic on failure by design, and indexing a JSON document is the readable form"
)]

use std::{
    os::unix::io::RawFd,
    time::{Duration, Instant},
};

#[path = "harness/support.rs"]
mod support;

use support::Install;

/// An install whose one profile points at port 9, which discards, so a run here cannot reach a
/// provider even with the mock off.
fn repl_install(
    newline_before_prompt: bool,
    newline_after_prompt: bool,
    extra_display: &str,
) -> Install {
    repl_install_with_extra(
        newline_before_prompt,
        newline_after_prompt,
        extra_display,
        "",
    )
}

/// [`repl_install`] plus whole config tables the `[display]` slot cannot hold.
fn repl_install_with_extra(
    newline_before_prompt: bool,
    newline_after_prompt: bool,
    extra_display: &str,
    extra_tables: &str,
) -> Install {
    let install = Install::new();
    install.write_config(&format!(
        "default_profile = \"default\"\n\n\
         [permissions]\ndefault = \"read\"\nenabled = [\"read\"]\n\n\
         [display]\nnewline_before_prompt = {newline_before_prompt}\n\
         newline_after_prompt = {newline_after_prompt}\n{extra_display}\n\
         [accounts.default]\nbackend = \"openai-chat-completions\"\n\
         base_url = \"http://127.0.0.1:9/\"\n\n\
         [profiles.default]\naccount = \"default\"\nmodel = \"mock-model\"\n{extra_tables}"
    ));
    install
}

/// Run one REPL session, sending `inputs` line by line, and return the rows the terminal would
/// show.
fn run_repl(install: &Install, script: &str, inputs: &[&str]) -> Vec<String> {
    install.write_script(script);
    let captured = drive(install, inputs);
    replay(&captured)
}

/// Fork a pty, exec meka on the child side, and drive the master side to completion.
fn drive(install: &Install, inputs: &[&str]) -> Vec<u8> {
    drive_to(install, inputs, None)
}

/// [`drive`] with the child's stderr sent to `stderr` instead of the pty, so a test can tell the
/// two streams apart: the pty then carries stdout alone. It is stderr that leaves, not stdout,
/// because the line editor draws on stdout and reads its cursor position back through the pty.
fn drive_to(install: &Install, inputs: &[&str], stderr: Option<&std::path::Path>) -> Vec<u8> {
    // Built before the fork: the child must not allocate between `fork` and `exec`.
    let stderr_path = stderr.map(|path| {
        std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).expect("a path without NUL")
    });
    let mut master: RawFd = -1;
    // SAFETY: `forkpty` with null pointers for the optional out-params is the documented way to get
    // a pty pair plus a child. The child branch below does nothing but set state and `exec`, which
    // is the one thing that is defined after `fork` in a threaded process.
    let pid = unsafe {
        libc::forkpty(
            &mut master,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert!(pid >= 0, "forkpty failed");

    if pid == 0 {
        // SAFETY: single-threaded from here to `execv`; every call below is async-signal-safe or a
        // libc wrapper this process is allowed to use before exec.
        unsafe {
            // Drop ECHO so our own DSR replies do not come back as input.
            let mut attrs: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut attrs) == 0 {
                attrs.c_lflag &= !libc::ECHO;
                libc::tcsetattr(0, libc::TCSANOW, &attrs);
            }
        }
        if let Some(path) = &stderr_path {
            // SAFETY: still single-threaded before `exec`; `open` and `dup2` are async-signal-safe.
            unsafe {
                let fd = libc::open(
                    path.as_ptr(),
                    libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
                    0o600,
                );
                if fd < 0 || libc::dup2(fd, 2) < 0 {
                    std::process::exit(126);
                }
            }
        }
        // Not spawned: this process *is* the child, and `exec` replaces it.
        let error = exec_meka(install);
        // Only reachable if exec failed.
        eprintln!("exec failed: {error}");
        std::process::exit(127);
    }

    let captured = pump(master, inputs);
    // SAFETY: `master` is the fd `forkpty` handed back and is still open here.
    unsafe {
        libc::close(master);
        libc::kill(pid, libc::SIGKILL);
        let mut status = 0;
        libc::waitpid(pid, &mut status, 0);
    }
    captured
}

fn exec_meka(install: &Install) -> std::io::Error {
    use std::os::unix::process::CommandExt;
    let mut command = support::meka();
    command
        .arg("-c")
        .current_dir(install.work_dir())
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("TERM", "xterm-256color")
        .env("COLUMNS", "100")
        .env("LINES", "40")
        // Everything here asserts where meka's own rows land, so a row the *host* decides to add
        // shifts every position under it. The sandbox advisories are exactly that: a runner with
        // Landlock but no Bubblewrap, or an older Landlock ABI, prints two `warn!` lines above the
        // first banner, and whether it does is a fact about the kernel the suite happens to run on.
        // Silenced by module rather than by level, because a retry warning reaching the screen is
        // itself under test (see `a_warning_raised_during_a_turn_is_not_erased_by_the_turn`).
        //
        // The two `rmcp` directives are meka's own defaults, restated because `RUST_LOG` replaces
        // the filter outright rather than adding to it, and dropping them lets an MCP transport's
        // reconnect chatter print rows of its own.
        .env(
            "RUST_LOG",
            "warn,meka::sandbox=error,rmcp::transport::common::client_side_sse=error,\
             rmcp::transport::worker=off",
        );
    install.env(&mut command).exec()
}

/// Read the master side, answering DSR and sending the next input whenever the child falls quiet.
fn pump(master: RawFd, inputs: &[&str]) -> Vec<u8> {
    let mut captured = Vec::new();
    let mut pending: Vec<&str> = inputs.to_vec();
    pending.reverse();
    let mut buffer = [0u8; 65536];
    let mut last_activity = Instant::now();
    // The first send waits longer: the child has a store to migrate and MCP probes to time out
    // before it draws anything.
    let mut quiet_needed = Duration::from_millis(2500);
    let deadline = Instant::now() + Duration::from_secs(90);

    while Instant::now() < deadline {
        // SAFETY: `master` is open, and the buffer outlives the call.
        let read = unsafe {
            let mut poll = libc::pollfd {
                fd: master,
                events: libc::POLLIN,
                revents: 0,
            };
            if libc::poll(&mut poll, 1, 250) > 0 && poll.revents & libc::POLLIN != 0 {
                libc::read(master, buffer.as_mut_ptr().cast(), buffer.len())
            } else {
                -1
            }
        };

        if read > 0 {
            let chunk = &buffer[..read as usize];
            captured.extend_from_slice(chunk);
            last_activity = Instant::now();
            if find(chunk, b"\x1b[6n") {
                // Any plausible position will do: reedline only needs the column to lay its prompt
                // out, and re-derives the row.
                write_all(master, b"\x1b[1;1R");
            }
            continue;
        }
        if read == 0 {
            break;
        }

        if last_activity.elapsed() > quiet_needed {
            match pending.pop() {
                Some(line) => {
                    write_all(master, line.as_bytes());
                    // A lone control byte is a keystroke, not a line: `^C` has to reach the tty's
                    // line discipline as itself, and a trailing carriage return would be a second
                    // keystroke the test did not ask for.
                    if line.len() > 1 || !line.starts_with(|c: char| c.is_control()) {
                        write_all(master, b"\r");
                    }
                    last_activity = Instant::now();
                    quiet_needed = Duration::from_millis(1500);
                }
                // Everything sent and the child has gone quiet: it has exited or is idle at a
                // prompt, and either way there is nothing left to capture.
                None if last_activity.elapsed() > Duration::from_secs(4) => break,
                None => {}
            }
        }
    }
    captured
}

fn write_all(fd: RawFd, bytes: &[u8]) {
    let mut written = 0;
    while written < bytes.len() {
        // SAFETY: writing a sub-slice of a live buffer to an open fd.
        let count =
            unsafe { libc::write(fd, bytes[written..].as_ptr().cast(), bytes.len() - written) };
        if count <= 0 {
            return;
        }
        written += count as usize;
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Replay the capture onto a grid of rows, so a blank line means what it looks like.
///
/// Only the sequences meka and reedline actually emit are modeled: SGR (dropped), erase-to-end-of-
/// line, move-to-column, carriage return, backspace and newline. Anything else is skipped as a unit
/// rather than printed, which is what keeps escape bytes out of the rows a test asserts on.
fn replay(data: &[u8]) -> Vec<String> {
    let mut rows: Vec<Vec<u8>> = vec![Vec::new()];
    let mut column = 0usize;
    let mut index = 0usize;

    while index < data.len() {
        if data[index] == 0x1b && index + 1 < data.len() && data[index + 1] == b'[' {
            let mut end = index + 2;
            while end < data.len() && !data[end].is_ascii_alphabetic() {
                end += 1;
            }
            if end >= data.len() {
                break;
            }
            let parameters = &data[index + 2..end];
            match data[end] {
                // Erase to end of line: truncate the row at the cursor.
                b'K' => {
                    let row = rows.last_mut().expect("a row is always present");
                    row.truncate(column.min(row.len()));
                }
                b'G' => column = 0,
                _ => {}
            }
            let _ = parameters;
            index = end + 1;
            continue;
        }
        match data[index] {
            b'\r' => column = 0,
            b'\n' => {
                rows.push(Vec::new());
                column = 0;
            }
            0x08 => column = column.saturating_sub(1),
            // A lone escape (an OSC or a sequence we do not model) -- skip the byte.
            0x1b => {}
            byte => {
                let row = rows.last_mut().expect("a row is always present");
                while row.len() < column {
                    row.push(b' ');
                }
                if column < row.len() {
                    row[column] = byte;
                } else {
                    row.push(byte);
                }
                column += 1;
            }
        }
        index += 1;
    }

    rows.into_iter()
        .map(|row| String::from_utf8_lossy(&row).trim_end().to_string())
        .collect()
}

/// The rows between the line that contains `after` and the next one containing `before`.
fn between<'a>(rows: &'a [String], after: &str, before: &str) -> &'a [String] {
    let start = rows
        .iter()
        .position(|row| row.contains(after))
        .unwrap_or_else(|| panic!("no row contains {after:?} in {rows:#?}"));
    let end = rows[start + 1..]
        .iter()
        .position(|row| row.contains(before))
        .unwrap_or_else(|| panic!("no row after {after:?} contains {before:?} in {rows:#?}"))
        + start
        + 1;
    &rows[start + 1..end]
}

fn blanks(rows: &[String]) -> usize {
    rows.iter().filter(|row| row.is_empty()).count()
}

const TOOLS_THEN_TEXT: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"schedule_list"},
  {"type":"tool_use_end","input":{}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"All cleared."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;

const TWO_TURNS: &str = r#"[
 [{"type":"text","text":"First answer."},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"tool_use_start","id":"t1","name":"schedule_list"},
  {"type":"tool_use_end","input":{}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"Second answer."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;

/// A thinking block, a transient failure, its retry, then a second turn.
///
/// The thinking delta is what puts the cursor on a `RowState::Transient` line -- the live
/// indicator -- so the warning that follows lands on a row the console is about to erase. Without
/// it the warning prints onto an empty row and survives whether or not the relay settles anything,
/// which is a test that passes for the wrong reason.
///
/// A transient failure, its retry, then a second turn. The retry consumes a round of its own:
/// `MockEvent::FailRetryable` returns from the stream the moment it is reached, so anything after
/// it in the same round is never emitted.
const RETRIES_MID_TURN: &str = r#"[
 [{"type":"thinking_delta","text":"weighing the options"},
  {"type":"fail_retryable","message":"529 overloaded","retry_after_secs":1}],
 [{"type":"text","text":"recovered answer"},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"text","text":"second answer"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;

const FAILS_MID_ANSWER: &str = r#"[
 [{"type":"text","text":"partial answer before the failure"},
  {"type":"fail","message":"provider exploded"}],
 [{"type":"text","text":"recovered"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;

/// The shape every other case is a variation on: one blank after the line you typed, one before the
/// next prompt, and the tool indicator and the answer separated by the block machine. `/rewind all`
/// is refused with a message rather than rewinding one turn and reporting a count the user had not
/// asked for. Driven through the REPL rather than the parser alone, so the dispatch arm that prints
/// the refusal is what is under test. `/approvals on` flips the switch for the session and writes
/// it to the row, where a resume from any host and a scheduled fire read it.
#[test]
fn approvals_switched_on_in_the_repl_are_recorded_on_the_row() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &["/approvals on", "hi", "/exit"]);
    assert!(
        rows.iter().any(|row| row.contains("Approvals set to: on")),
        "the switch is confirmed on screen: {rows:#?}"
    );
    let connection = rusqlite::Connection::open(install.database()).expect("open the store");
    let recorded: i64 = connection
        .query_row("SELECT approvals FROM sessions", [], |row| row.get(0))
        .expect("exactly one session row");
    assert_eq!(recorded, 1, "the row carries the switch");
}

/// `always` at the approval prompt answers for the tool until the session ends: the second call to
/// it is not put to the user, and both writes land. Driven through the terminal because the answer
/// travels from the editor thread back to the frontend, and only the real channel proves it gets
/// there as a sticky decision rather than a bare yes.
#[test]
fn an_always_answer_at_the_approval_prompt_covers_the_next_call_to_the_tool() {
    const TWO_WRITES: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"write_file"},
  {"type":"tool_use_end","input":{"path":"a.txt","content":"a"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"wrote a"},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"tool_use_start","id":"t2","name":"write_file"},
  {"type":"tool_use_end","input":{"path":"b.txt","content":"b"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"wrote b"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TWO_WRITES, &[
        "/approvals on",
        "write a",
        "always",
        "write b",
        "/exit",
    ]);
    let prompts = rows
        .iter()
        .filter(|row| row.contains("[approval] write_file"))
        .count();
    assert_eq!(
        prompts, 1,
        "the first write is put to the user and the second is covered by `always`: {rows:#?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.contains("Allow? (Y/n/always/never)")),
        "the question names the sticky answers: {rows:#?}"
    );
    let work = install.root().join("work");
    assert!(work.join("a.txt").exists(), "the approved write ran");
    assert!(
        work.join("b.txt").exists(),
        "the second write ran without a prompt: {rows:#?}"
    );
}

/// `always` answers for the rest of the session it was given in, and `/fork` moves the REPL into
/// another session: the same tool is put to the user again in the copy. Driven through the
/// terminal because the handoff is the loop's, not the frontend's, and only the real loop proves
/// the answer is forgotten when it moves.
#[test]
fn an_always_answer_does_not_survive_a_fork() {
    const TWO_WRITES: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"write_file"},
  {"type":"tool_use_end","input":{"path":"a.txt","content":"a"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"wrote a"},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"tool_use_start","id":"t2","name":"write_file"},
  {"type":"tool_use_end","input":{"path":"b.txt","content":"b"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"wrote b"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TWO_WRITES, &[
        "/approvals on",
        "write a",
        "always",
        "/fork",
        "write b",
        "y",
        "/exit",
    ]);
    assert!(
        rows.iter().any(|row| row.contains("Forked session")),
        "the fork has to have happened for the rest to mean anything: {rows:#?}"
    );
    let prompts = rows
        .iter()
        .filter(|row| row.contains("[approval] write_file"))
        .count();
    assert_eq!(
        prompts, 2,
        "the write in the copy is put to the user again, since `always` was given in the \
         original: {rows:#?}"
    );
    let work = install.root().join("work");
    assert!(work.join("a.txt").exists(), "the approved write ran");
    assert!(
        work.join("b.txt").exists(),
        "and so did the one approved in the copy: {rows:#?}"
    );
}

/// A scheduled job's prompt is echoed before the reply it triggers, dimmed like a notice, so the
/// answer that appears while the user is at the prompt is not the model speaking unprompted.
#[test]
fn a_scheduled_fire_echoes_its_prompt_before_the_reply() {
    const SCHEDULE_THEN_FIRE: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"schedule_create"},
  {"type":"tool_use_end","input":{"prompt":"REPL_DELIVERED_MARKER","at":"1s"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"scheduled"},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"text","text":"REPL_SCHEDULED_REPLY"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install =
        repl_install_with_extra(true, true, "", "\n[schedule]\npoll_interval = \"200ms\"\n");
    let rows = run_repl(&install, SCHEDULE_THEN_FIRE, &["remind me", "/exit"]);
    let screen = rows.join("\n");
    assert!(
        screen.contains("REPL_SCHEDULED_REPLY"),
        "the fire's reply must reach the screen: {rows:#?}"
    );
    let echoed = rows
        .iter()
        .position(|row| row.contains("[Scheduled job") && row.contains("fired"));
    let replied = rows
        .iter()
        .position(|row| row.contains("REPL_SCHEDULED_REPLY"));
    assert!(
        matches!((echoed, replied), (Some(echo), Some(reply)) if echo < reply),
        "the job's prompt is shown above the reply it triggered: {rows:#?}"
    );
    assert!(
        screen.contains("REPL_DELIVERED_MARKER"),
        "and the prompt's own words are what is shown: {rows:#?}"
    );
}

#[test]
fn a_rewind_with_a_bad_count_is_refused_on_screen() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &["/rewind all", "/exit"]);
    assert!(
        rows.iter()
            .any(|row| row.contains("/rewind takes a turn count of 1 or more, not 'all'")),
        "the refusal must reach the screen, quoting the value as a value: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains("Rewound")),
        "nothing may be rewound on a refused count: {rows:#?}"
    );
}

/// A known command missing its argument is refused as that command, and an unknown one is refused
/// by its command word alone: the rest of the line was never read, so echoing it would say it was.
#[test]
fn a_half_typed_command_is_refused_for_what_it_is() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &[
        "/mcp reconnect",
        "/mcp frob",
        "/frob a b",
        "/exit",
    ]);
    assert!(
        rows.iter()
            .any(|row| row.contains("/mcp reconnect takes a server name")),
        "a verb without its server is a known command missing its argument: {rows:#?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.contains("'frob' is not an `/mcp` verb")),
        "an unknown verb is an `/mcp` mistake, not an unknown command: {rows:#?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.contains("Unknown command: /frob. Type /help")),
        "the unknown command is named by its word alone: {rows:#?}"
    );
    assert!(
        !rows
            .iter()
            .any(|row| row.contains("Unknown command: /mcp") || row.contains("/frob a b.")),
        "neither line echoes what was never read: {rows:#?}"
    );
}

/// Ctrl+C during `/compact` is annotated the way Ctrl+C during a turn is. The one place a manual
/// compaction can still end in an interrupt is the wait before a retry, so the summarizer's call is
/// scripted to be refused with a long `Retry-After`, with the checkpoint turn off so that call is
/// the one made.
#[test]
fn an_interrupted_compaction_is_annotated_not_reported_as_an_error() {
    const RETRY_LATER: &str = r#"[
 [{"type":"text","text":"First answer."},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"fail_retryable","message":"busy","retry_after_secs":20}]
]"#;
    let install =
        repl_install_with_extra(true, true, "", "\n[session]\ncompact_checkpoint = false\n");
    let rows = run_repl(&install, RETRY_LATER, &["hi", "/compact", "\u{3}", "/exit"]);

    let body = between(&rows, "> /compact", "> /exit");
    assert!(
        body.iter().any(|row| row.contains("(interrupted)")),
        "the interrupt is annotated: {body:#?}"
    );
    assert!(
        !body.iter().any(|row| row.contains("interrupted by user")),
        "and not reported as an error: {body:#?}"
    );
}

#[test]
fn a_turn_is_bracketed_once_on_each_side() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TOOLS_THEN_TEXT, &["do the thing", "/exit"]);

    let body = between(&rows, "do the thing", "> /exit");
    assert_eq!(
        blanks(body),
        3,
        "one after the prompt, one between indicator and text, one before the next prompt: {body:#?}"
    );
    assert_eq!(body.first().map(String::as_str), Some(""));
    assert_eq!(body.last().map(String::as_str), Some(""));
}

/// The setting must hold past the first turn: suppressing the block machine along with the blank
/// makes the next turn's first block see the previous turn's last one and ask for a separator that
/// looks exactly like the blank the user disabled.
#[test]
fn disabling_both_blanks_leaves_no_prompt_spacing_on_any_turn() {
    let install = repl_install(false, false, "");
    let rows = run_repl(&install, TWO_TURNS, &["first", "second", "/exit"]);

    let first = between(&rows, "> first", "> second");
    assert_eq!(blanks(first), 0, "turn one is unspaced: {first:#?}");

    let second = between(&rows, "> second", "> /exit");
    assert_eq!(
        blanks(second),
        1,
        "only the indicator-to-text separator, which is not a prompt bracket: {second:#?}"
    );
}

/// A command that answers without running a turn is bracketed by the dispatcher, since no turn will
/// do it; otherwise its error prints flush against the line above *and* the prompt below.
#[test]
fn a_command_that_never_runs_a_turn_is_still_bracketed() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &["/skill nosuchskill", "/exit"]);

    let body = between(&rows, "/skill nosuchskill", "> /exit");
    assert_eq!(blanks(body), 2, "one blank on each side: {body:#?}");
    assert!(
        body.iter().any(|row| row.contains("no skill named")),
        "the error is what the brackets are around: {body:#?}"
    );
}

/// A turn that dies mid-answer holds whatever it streamed. Ending the *episode* flushes it, which
/// puts it under the turn it belongs to; flushed by the next turn's `TurnStarted` instead, it would
/// print beneath the following prompt as though it answered that.
#[test]
fn a_failed_turn_shows_its_partial_answer_in_its_own_turn() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, FAILS_MID_ANSWER, &["boom", "again", "/exit"]);

    let failed = between(&rows, "> boom", "> again");
    assert!(
        failed.iter().any(|row| row.contains("partial answer")),
        "the partial answer belongs to the turn that produced it: {failed:#?}"
    );

    let recovered = between(&rows, "> again", "> /exit");
    assert!(
        !recovered.iter().any(|row| row.contains("partial answer")),
        "and must not reappear under the next prompt: {recovered:#?}"
    );
    assert_eq!(
        recovered.first().map(String::as_str),
        Some(""),
        "the blank after the prompt is still the first thing: {recovered:#?}"
    );
}

/// The session-id notice is emitted before the turn starts, so left alone it prints above the blank
/// meant to separate it from the line the user typed.
#[test]
fn the_opening_blank_precedes_the_session_notice() {
    let install = repl_install(true, true, "show_session_id_on_create = true\n");
    let rows = run_repl(&install, TWO_TURNS, &["first", "/exit"]);

    let body = between(&rows, "> first", "> /exit");
    assert_eq!(
        body.first().map(String::as_str),
        Some(""),
        "nothing may slip above the opening blank: {body:#?}"
    );
    assert!(
        body.iter().any(|row| row.contains("Creating new session")),
        "the notice is still shown: {body:#?}"
    );
}

/// A run's outer edges border the shell's prompt rather than one of meka's, and a shell lays out
/// its own. Both blanks were spent against it anyway: one above `Continuing session:`, which is
/// meka's first word and belongs directly under the command line, and one below `Leaving session:`,
/// which is its last unless a background task is still running. See
/// [`the_shutdown_notice_reads_as_one_block_with_the_exit_banner`] for that case.
///
/// Two runs over one install, because the second is the only way to reach the resume banner: `-c`
/// needs a session the first run left behind.
#[test]
fn the_shell_s_prompt_gets_neither_blank() {
    let install = repl_install(true, true, "");

    let leaving = run_repl(&install, TWO_TURNS, &["first", "/exit"]);
    let last_word = leaving
        .iter()
        .position(|row| row.contains("Leaving session"))
        .unwrap_or_else(|| panic!("the exit banner is shown by default: {leaving:#?}"));
    assert!(
        leaving[last_word + 1..].iter().all(String::is_empty) && leaving.len() - last_word <= 2,
        "only the newline that ends the banner may follow it, not a blank line into the shell \
         prompt: {leaving:#?}"
    );

    let continuing = run_repl(&install, TWO_TURNS, &["/exit"]);
    assert!(
        continuing
            .first()
            .is_some_and(|row| row.contains("Continuing session")),
        "the resume banner is the first row, with no blank above it: {continuing:#?}"
    );
    // The other half, and the one the `printed` bookkeeping exists for: losing the opening blank
    // must not cost the episode its closing one. Without this the whole suite stays green while
    // the replayed history butts straight against the first prompt.
    let first_prompt = continuing
        .iter()
        .position(|row| row.contains("/exit"))
        .unwrap_or_else(|| panic!("the typed line is echoed at the prompt: {continuing:#?}"));
    assert!(
        first_prompt > 0 && continuing[first_prompt - 1].is_empty(),
        "the first prompt is bracketed like every later one: {continuing:#?}"
    );
}

/// A task still running at `/exit` puts a second line under the exit banner, and the two read as
/// one block.
///
/// They are both `Chrome`, which asks for no separator, so flush is what two of them do anywhere
/// else in a session. What makes it worth pinning is that an episode closes between them, early,
/// to settle the row before the session lock is released: a close that bracketed against a prompt
/// would put a blank there that nobody chose.
#[test]
fn the_shutdown_notice_reads_as_one_block_with_the_exit_banner() {
    let install = repl_install_with_extra(true, true, "", "\n[background]\nenabled = true\n");
    let script = r#"[
        [
            { "type": "tool_use_start", "id": "tu_1", "name": "execute_command" },
            { "type": "tool_use_end", "input": {"command": "sleep 120", "background": true} },
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "started it" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]"#;

    let rows = run_repl(&install, script, &["run it", "/exit"]);
    let banner = rows
        .iter()
        .position(|row| row.contains("Leaving session"))
        .unwrap_or_else(|| panic!("the exit banner is shown by default: {rows:#?}"));
    assert_eq!(
        rows.get(banner + 1).map(String::as_str),
        Some("(stopping 1 background task)"),
        "the notice is the very next row, with no blank between: {rows:#?}"
    );
    assert!(
        rows[banner + 2..].iter().all(String::is_empty) && rows.len() - banner <= 3,
        "and nothing follows it into the shell prompt: {rows:#?}"
    );
}

/// Most commands answer through the `cli` modules, which print for themselves and are invisible to
/// the console. They are bracketed by the dispatcher announcing on their behalf; without that the
/// blank lines follow only the output the console happens to render itself, and `/status` prints
/// none of it.
#[test]
fn a_command_that_prints_for_itself_is_still_bracketed() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &["/status", "/exit"]);

    let body = between(&rows, "> /status", "> /exit");
    assert_eq!(
        body.first().map(String::as_str),
        Some(""),
        "a blank after the line typed: {body:#?}"
    );
    assert_eq!(
        body.last().map(String::as_str),
        Some(""),
        "and one before the next prompt: {body:#?}"
    );
    assert!(
        body.iter().any(|row| row.contains("Session status")),
        "the table is what the brackets are around: {body:#?}"
    );
}

/// `/help` is answered by the REPL thread rather than the agent loop, through a printer the console
/// cannot see. It is bracketed by the same rule as everything else.
#[test]
fn help_is_bracketed_by_the_repl_thread_too() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &["/help", "/exit"]);

    let body = between(&rows, "> /help", "> /exit");
    assert_eq!(body.first().map(String::as_str), Some(""), "{body:#?}");
    assert_eq!(body.last().map(String::as_str), Some(""), "{body:#?}");
    assert!(
        body.iter().any(|row| row.contains("Shortcuts:")),
        "the help text is what the brackets are around: {body:#?}"
    );
}

/// The todo list paints its own surrounding blanks, so the block machine must not add more. This is
/// the one block that would double if it were spaced like the others.
#[test]
fn a_todo_list_is_not_double_spaced() {
    const TODO: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"todo"},
  {"type":"tool_use_end","input":{"title":"Work","items":["First","Second"]}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"Done."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TODO, &["plan it", "/exit"]);

    let body = between(&rows, "> plan it", "> /exit");
    assert!(
        body.iter().any(|row| row.contains("TODO: Work")),
        "the list rendered: {body:#?}"
    );
    assert!(
        !body
            .windows(2)
            .any(|pair| pair[0].is_empty() && pair[1].is_empty()),
        "no two blank lines ever sit together: {body:#?}"
    );
}

/// Thinking renders as its own block, inside the episode's brackets like any other.
#[test]
fn a_thinking_block_sits_inside_the_brackets() {
    const THINKING: &str = r#"[
 [{"type":"thinking_delta","text":"weighing the options"},
  {"type":"thinking_complete"},
  {"type":"text","text":"Here is the answer."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "\n[thinking]\nshow_content = true\n");
    let rows = run_repl(&install, THINKING, &["think about it", "/exit"]);

    let body = between(&rows, "> think about it", "> /exit");
    assert_eq!(body.first().map(String::as_str), Some(""), "{body:#?}");
    assert_eq!(body.last().map(String::as_str), Some(""), "{body:#?}");
    assert!(
        body.iter().any(|row| row.contains("weighing the options")),
        "the thinking block rendered: {body:#?}"
    );
    assert!(
        body.iter().any(|row| row.contains("Here is the answer")),
        "and the answer after it: {body:#?}"
    );
}

/// Reasoning arrives in chunks that stop wherever the provider's tokenizer did, so a block is one
/// block however it was cut up: one `Thinking... ` label, and words split across a chunk boundary
/// rejoined rather than shown broken.
///
/// The markers go too. Reasoning is markdown on the backends that emit a summary, where every part
/// opens with a `**Bold header**` line, so a renderer that shows the asterisks shows them on every
/// block those backends produce.
#[test]
fn a_thinking_block_split_across_deltas_renders_as_one() {
    const SPLIT: &str = r#"[
 [{"type":"thinking_delta","text":"**Weighing the op"},
  {"type":"thinking_delta","text":"tions**\n\nBoth are fine."},
  {"type":"thinking_complete"},
  {"type":"text","text":"Here is the answer."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "\n[thinking]\nshow_content = true\n");
    let rows = run_repl(&install, SPLIT, &["think about it", "/exit"]);

    let body = between(&rows, "> think about it", "> /exit");
    assert_eq!(
        body.iter()
            .filter(|row| row.contains("Thinking..."))
            .count(),
        1,
        "one label for one block: {body:#?}"
    );
    assert!(
        body.iter().any(|row| row.contains("Weighing the options")),
        "the split word was rejoined: {body:#?}"
    );
    assert!(
        !body.iter().any(|row| row.contains("**")),
        "the emphasis markers are styling now, not text: {body:#?}"
    );
    assert!(
        body.iter().any(|row| row.contains("Both are fine.")),
        "and the rest of the block followed: {body:#?}"
    );
    assert!(
        body.iter().any(|row| row.contains("Here is the answer")),
        "with the answer after it: {body:#?}"
    );
}

/// The default. The block collapses onto the label's row, and the emphasis in it is styling there
/// too: this is the line most people see, so it cannot be the one that still shows asterisks.
#[test]
fn without_show_content_a_thinking_block_is_one_formatted_line() {
    const SPLIT: &str = r#"[
 [{"type":"thinking_delta","text":"**Weighing the options.**\nThe second line."},
  {"type":"thinking_complete"},
  {"type":"text","text":"Here is the answer."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, SPLIT, &["think about it", "/exit"]);

    let body = between(&rows, "> think about it", "> /exit");
    assert!(
        body.iter()
            .any(|row| row.contains("Thinking... Weighing the options. The second line.")),
        "the preview keeps its inline label and flattens the break: {body:#?}"
    );
    assert!(
        !body.iter().any(|row| row.contains("**")),
        "and its markers are styling, not text: {body:#?}"
    );
}

/// A provider notice renders as a hint inside the turn, between the same brackets as everything
/// else rather than beside them.
#[test]
fn a_notice_renders_inside_the_brackets() {
    const NOTICE: &str = r#"[
 [{"type":"notice","message":"the model dropped an unsupported parameter"},
  {"type":"text","text":"Answered anyway."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, NOTICE, &["ask", "/exit"]);

    let body = between(&rows, "> ask", "> /exit");
    assert_eq!(body.first().map(String::as_str), Some(""), "{body:#?}");
    assert_eq!(body.last().map(String::as_str), Some(""), "{body:#?}");
    assert!(
        body.iter().any(|row| row.contains("unsupported parameter")),
        "the notice rendered: {body:#?}"
    );
}

/// `/profile <name>` is typed at the REPL thread but answered on the agent's side, so its
/// confirmation is the one piece of command output that neither the dispatcher nor a turn prints.
#[test]
fn a_forwarded_profile_switch_is_bracketed() {
    let install = repl_install(true, true, "");
    // A second profile to switch to; the first is what the session starts on.
    let extra = "\n[profiles.other]\naccount = \"default\"\nmodel = \"other-model\"\n";
    let config = install.root().join("meka").join("config.toml");
    let mut text = std::fs::read_to_string(&config).expect("read config");
    text.push_str(extra);
    std::fs::write(&config, text).expect("write config");

    let rows = run_repl(&install, TWO_TURNS, &["/profile other", "/exit"]);

    let body = between(&rows, "> /profile other", "> /exit");
    assert_eq!(body.first().map(String::as_str), Some(""), "{body:#?}");
    assert_eq!(body.last().map(String::as_str), Some(""), "{body:#?}");
    assert!(
        body.iter().any(|row| row.contains("Profile set to")),
        "the confirmation is what the brackets are around: {body:#?}"
    );
}

/// `/profile` with no argument names the profile this session runs on, then lists every configured
/// one with its account and the backend it speaks, under a heading styled like `/status`'s. It used
/// to be a comma-joined run of names with no heading, which stops fitting long before a user stops
/// adding accounts and never said what the list was for.
#[test]
fn profile_lists_one_profile_per_line_with_its_account_and_backend() {
    let install = repl_install(true, true, "");
    let config = install.root().join("meka").join("config.toml");
    let mut text = std::fs::read_to_string(&config).expect("read config");
    text.push_str(
        "\n[accounts.zzz-acct]\nbackend = \"anthropic-messages\"\n\n\
         [profiles.zzz-last]\naccount = \"zzz-acct\"\nmodel = \"other-model\"\n",
    );
    std::fs::write(&config, text).expect("write config");

    let rows = run_repl(&install, TWO_TURNS, &["/profile", "/exit"]);
    let body = between(&rows, "> /profile", "> /exit");

    let current = body
        .iter()
        .position(|row| row == "Current profile: default")
        .unwrap_or_else(|| panic!("the answer comes first: {body:#?}"));
    let heading = body
        .iter()
        .position(|row| row == "Configured profiles")
        .unwrap_or_else(|| panic!("the list says what it is: {body:#?}"));
    assert!(
        heading > current && body[heading - 1].is_empty(),
        "the heading follows, set apart from it: {body:#?}"
    );
    assert!(
        body[heading + 1].starts_with("- "),
        "with its list directly beneath, as `/status` heads its own block: {body:#?}"
    );
    assert!(
        body.iter()
            .any(|row| row == "- default (default, openai-chat-completions)"),
        "each profile is its own line, with its account and backend: {body:#?}"
    );
    assert!(
        body.iter()
            .any(|row| row == "- zzz-last (zzz-acct, anthropic-messages)"),
        "including the ones that are not current: {body:#?}"
    );
    assert!(
        !body.iter().any(|row| row.contains("Configured:")),
        "the comma-joined line is gone: {body:#?}"
    );
}

/// A successful `/cd` prints nothing, so it gets no blank lines; its failure prints, so it does.
#[test]
fn cd_is_spaced_only_when_it_has_something_to_say() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, TWO_TURNS, &[
        "/cd /nonexistent-xyz",
        "/cd /tmp",
        "/exit",
    ]);

    let failure = between(&rows, "/cd /nonexistent-xyz", "/cd /tmp");
    assert_eq!(blanks(failure), 2, "the error is bracketed: {failure:#?}");

    let success = between(&rows, "/cd /tmp", "> /exit");
    assert_eq!(
        blanks(success),
        0,
        "a silent command gets no blanks around nothing: {success:#?}"
    );
}

/// Ctrl+C during a turn. The notice is the one piece of chrome tempted to open with its own newline
/// to terminate whatever row it lands on: right when a row is open, a stray blank line when the
/// cursor is already at column zero, and no brackets at all when the turn has not printed anything
/// yet.
#[test]
fn an_interrupted_turn_is_annotated_and_bracketed() {
    const SLOW: &str = r#"[
 [{"type":"text","text":"starting the long answer"},
  {"type":"sleep","ms":8000},
  {"type":"text","text":"never reached"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, SLOW, &["slow", "\u{3}", "/exit"]);

    let body = between(&rows, "> slow", "> /exit");
    assert!(
        body.iter().any(|row| row.contains("(interrupted)")),
        "the interrupt is annotated, not announced as a sentence: {body:#?}"
    );
    assert!(
        !body.iter().any(|row| row.contains("never reached")),
        "the turn really was cut short: {body:#?}"
    );
    assert_eq!(
        body.first().map(String::as_str),
        Some(""),
        "still one blank after the line typed: {body:#?}"
    );
    assert_eq!(
        body.last().map(String::as_str),
        Some(""),
        "and one before the next prompt: {body:#?}"
    );
    assert!(
        !body
            .windows(2)
            .any(|pair| pair[0].is_empty() && pair[1].is_empty()),
        "and no stray blank from the notice's old leading newline: {body:#?}"
    );
}

/// A warning printed mid-turn reaches the screen and stays there, beside the answer it explains.
///
/// The retry path warns at default verbosity, which is the moment a user most needs to know why
/// nothing is happening, and nothing else covers that end to end: `tracing` goes straight to stderr
/// off-prompt, outside every renderer this file otherwise exercises.
///
/// **What this does not cover.** It does not reproduce the erasure `relay::RelayWriter::write`
/// settles the row to prevent -- neutering that call leaves this test green. The collision needs
/// the warning to land while the row is still the thinking indicator's, and which of the frontend's
/// draw and the agent's `warn!` wins that race is not something a script can pin. The settle itself
/// is guarded in `relay`'s own tests, where the row can be forced; this guards the plainer property
/// that the warning is not lost some *other* way, which is what a reader of the changelog entry
/// would want to know.
#[test]
fn a_warning_raised_during_a_turn_is_not_erased_by_the_turn() {
    let install = repl_install(true, true, "");
    let rows = run_repl(&install, RETRIES_MID_TURN, &["boom", "again", "/exit"]);

    let turn = between(&rows, "> boom", "> again");
    assert!(
        turn.iter().any(|row| row.contains("529 overloaded")),
        "the retry warning has to still be on screen at the end of the turn that raised it: \
         {turn:#?}"
    );
    assert!(
        turn.iter().any(|row| row.contains("recovered answer")),
        "and the answer it was waiting for has to follow it: {turn:#?}"
    );
}

/// Showing reasoning costs the turn its retry, rather than showing the reasoning twice.
///
/// The retry is gated on nothing model-produced having reached a consumer, and reasoning reaches
/// one only when it is being rendered. So the same failure recovers under the default (the line
/// shown is built from a completed block, which a failed attempt never reaches) and does not
/// recover here, where the deltas are already on screen and a second attempt would repeat them.
/// `a_warning_raised_during_a_turn_is_not_erased_by_the_turn` pins the other half with the same
/// script.
#[test]
fn showing_reasoning_trades_the_retry_for_not_repeating_it() {
    let install = repl_install(true, true, "\n[thinking]\nshow_content = true\n");
    let rows = run_repl(&install, RETRIES_MID_TURN, &["boom", "again", "/exit"]);

    let turn = between(&rows, "> boom", "> again");
    assert_eq!(
        turn.iter()
            .filter(|row| row.contains("weighing the options"))
            .count(),
        1,
        "the reasoning reached the screen more than once: {turn:#?}"
    );
    assert!(
        !turn.iter().any(|row| row.contains("recovered answer")),
        "the turn retried after its reasoning was already shown: {turn:#?}"
    );
}

/// A running token estimate never draws over reasoning already on the screen.
///
/// Claude's token-count beta reports an estimate from the *same* wire event that carries a visible
/// delta, so the two alternate for the whole block. Drawing the counter closes the streamed block
/// to free the row, and the next delta opens a fresh one behind a second `Thinking... ` label --
/// once per delta, which shreds one block into a labeled fragment per chunk.
///
/// Driven through a pty because the guard sits in front of terminal-only drawing:
/// `live_indicator_supported` is false in every ordinary test process, so nothing there can reach
/// the code this protects.
#[test]
fn a_token_estimate_never_shreds_the_reasoning_it_counts() {
    const COUNTED: &str = r#"[
 [{"type":"thinking_delta","text":"First I weigh it. "},
  {"type":"thinking_progress","estimated_tokens":16},
  {"type":"thinking_delta","text":"Then I settle it. "},
  {"type":"thinking_progress","estimated_tokens":32},
  {"type":"thinking_delta","text":"Then I say so."},
  {"type":"thinking_progress","estimated_tokens":48},
  {"type":"thinking_complete"},
  {"type":"text","text":"Here is the answer."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "[thinking]\nshow_content = true\n");
    let rows = run_repl(&install, COUNTED, &["think about it", "/exit"]);

    let labels = rows
        .iter()
        .filter(|row| row.contains("Thinking... "))
        .count();
    assert_eq!(
        labels, 1,
        "the block was relabeled once per estimate: {rows:#?}"
    );
    let reasoning: String = rows.join(" ");
    assert!(
        reasoning.contains("First I weigh it.")
            && reasoning.contains("Then I settle it.")
            && reasoning.contains("Then I say so."),
        "an estimate landed on top of a chunk: {rows:#?}"
    );
}

/// Resuming replays the reasoning it recorded, rendered the way the live turn rendered it.
///
/// The replay path builds its own renderer rather than sharing the live one's, so nothing but a
/// resume exercises it: delete the call and every other test stays green, which is how `/history`
/// and `resume_show_recent` could stop showing reasoning entirely without a failure.
///
/// Two runs against one install: the pty harness always passes `-c`, so the second resumes the
/// session the first recorded.
#[test]
fn a_resumed_session_replays_the_reasoning_it_recorded() {
    const REASONED: &str = r#"[
 [{"type":"thinking_delta","text":"**Weighing it** then deciding."},
  {"type":"thinking_complete"},
  {"type":"text","text":"Here is the answer."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(
        true,
        true,
        "resume_show_recent = 1\n\n[thinking]\nshow_content = true\n",
    );
    let first = run_repl(&install, REASONED, &["think about it", "/exit"]);
    assert!(
        first
            .iter()
            .any(|row| row.contains("Weighing it then deciding")),
        "the live turn has to render it before a replay can repeat it: {first:#?}"
    );

    let resumed = run_repl(&install, REASONED, &["/exit"]);
    assert!(
        resumed
            .iter()
            .any(|row| row.contains("Weighing it then deciding")),
        "the resumed session dropped the reasoning it recorded: {resumed:#?}"
    );
    assert!(
        !resumed.iter().any(|row| row.contains("**")),
        "and replays it rendered, not as source: {resumed:#?}"
    );
}

/// A canceled background task rides the next thing the user types, without spending a turn.
///
/// The REPL is the most-used host and had no coverage for any of this: the fold, the watcher's
/// refusal to wake, and the retention the carrier runs under were all deletable with the suite
/// green. Driven through the pty rather than seeded, so what is under test is the wiring rather
/// than the predicates the unit tests already pin.
///
/// The script holds exactly three rounds, and the watcher polls fast enough to tick several times
/// between the cancellation and the closing turn: one that woke *and delivered* would eat the third
/// round and leave that turn nothing to answer with.
///
/// Not pinned here: the watcher's own `wakes_a_host` gate. Removing it makes the watcher wake for a
/// batch the wake arm then declines to claim, so the conversation and the script are untouched and
/// only a prompt redraw is wasted. It is an optimization sitting in front of the real guard, and
/// catching it needs an assertion about drawing rather than about the conversation.
#[test]
fn a_canceled_task_rides_the_next_prompt_in_the_repl() {
    let install = repl_install_with_extra(
        true,
        true,
        "",
        "\n[background]\nenabled = true\n\n[schedule]\npoll_interval = \"200ms\"\n",
    );
    let script = r#"[
        [
            { "type": "tool_use_start", "id": "tu_1", "name": "execute_command" },
            { "type": "tool_use_end", "input": {"command": "sleep 120", "background": true} },
            { "type": "message_end", "stop_reason": "tool_use" }
        ],
        [
            { "type": "text", "text": "started it" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ],
        [
            { "type": "text", "text": "answered the question" },
            { "type": "message_end", "stop_reason": "end_turn" }
        ]
    ]"#;

    let rows = run_repl(&install, script, &[
        "run it",
        "/tasks cancel --all",
        "what is in this CSV?",
        "exit",
    ]);
    let screen = rows.join("\n");
    assert!(
        screen.contains("answered the question"),
        "the third round must still be there for the user's own turn: {screen}"
    );

    let connection = rusqlite::Connection::open(install.database()).expect("open the store");
    let user_messages: Vec<String> = {
        let mut statement = connection
            .prepare("SELECT content FROM messages WHERE role IN ('user', 'user_blocks') ORDER BY id ASC")
            .expect("prepare");
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .expect("query");
        rows.collect::<rusqlite::Result<Vec<String>>>()
            .expect("read")
    };

    assert_eq!(
        user_messages.len(),
        2,
        "two turns were typed and the cancellation must not have added a third: {user_messages:#?}"
    );
    let carrier = user_messages.last().expect("the second turn");
    assert!(
        carrier.contains("was canceled") && carrier.contains("what is in this CSV?"),
        "the outcome must ride inside the user's own message: {carrier}"
    );

    let delivered: Option<String> = connection
        .query_row(
            "SELECT delivered_at FROM background_tasks LIMIT 1",
            [],
            |row| row.get(0),
        )
        .expect("read the task");
    assert!(
        delivered.is_some(),
        "and riding a turn is a delivery, so the row must be stamped"
    );
}

/// A slash command's table is the user looking at the UI, so it goes to stderr with the rest of
/// the chrome, while the model's answer stays on stdout. Driven with stderr redirected to a file,
/// which a merged pty could not tell apart.
#[test]
fn a_slash_commands_table_goes_to_stderr_and_the_answer_stays_on_stdout() {
    const ANSWER: &str = r#"[
 [{"type":"text","text":"answer-on-stdout"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = repl_install(true, true, "");
    let mut add = support::meka();
    add.args([
        "memory",
        "add",
        "stream-audit-memory",
        "--description",
        "kept for the stream test",
    ]);
    let added = install.env(&mut add).output().expect("run memory add");
    assert!(
        added.status.success(),
        "memory add: {}",
        String::from_utf8_lossy(&added.stderr)
    );

    install.write_script(ANSWER);
    let stderr_path = install.work_dir().join("stderr.txt");
    let captured = drive_to(&install, &["/memory", "hello", "/exit"], Some(&stderr_path));
    let rows = replay(&captured);
    assert!(
        rows.iter().any(|row| row.contains("answer-on-stdout")),
        "the answer reaches the terminal through stdout: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains("stream-audit-memory")),
        "the table is not on stdout: {rows:#?}"
    );
    let stderr = std::fs::read_to_string(&stderr_path).expect("stderr file");
    assert!(
        stderr.contains("stream-audit-memory"),
        "the table is on stderr: {stderr:?}"
    );
}
