//! The interactive screen, driven through a real pseudo-terminal.
//!
//! Everything else in `tests/` runs meka with pipes, which the screen refuses: it draws on the
//! alternate screen and reads the keyboard. This owns the master side of a pty, feeds keys whenever
//! the child goes quiet, and reads the screen back through a terminal emulator, so a test asserts
//! on rows as a person would see them rather than on escape sequences.
//!
//! Unix only, and only where the scripted provider (`MEKA_MOCK_PROVIDER`) is compiled in, which a
//! release build does without the `mock-provider` feature.
#![cfg(all(unix, any(debug_assertions, feature = "mock-provider")))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "a test panics on failure by design"
)]

use std::{
    ffi::CString,
    os::{fd::RawFd, unix::ffi::OsStrExt},
    time::{Duration, Instant},
};

use base64::Engine as _;
use support::Install;

#[path = "harness/support.rs"]
mod support;

const ROWS: u16 = 40;
const COLUMNS: u16 = 100;

fn screen_install(permission: &str, extra_display: &str, extra_tables: &str) -> Install {
    let install = Install::new();
    install.write_config(&format!(
        "default_profile = \"default\"\n\n\
         [permissions]\ndefault = \"{permission}\"\nenabled = [\"{permission}\"]\n\n\
         [display]\n{extra_display}\n\
         [accounts.default]\nbackend = \"openai-chat-completions\"\n\
         base_url = \"http://127.0.0.1:9/\"\n\n\
         [profiles.default]\naccount = \"default\"\nmodel = \"mock-model\"\n{extra_tables}"
    ));
    install
}

/// What the screen showed just before each input was sent, in order. The screen rows are
/// trimmed on the right, and trailing blank rows are dropped.
type Screens = Vec<Vec<String>>;

/// Run one interactive session with `script`, sending `inputs` one at a time whenever the child
/// has been quiet for a moment, and return the screen as it stood before each.
fn run_screen(install: &Install, script: &str, inputs: &[&str]) -> Screens {
    install.write_script(script);
    drive(install, inputs)
}

/// The screen before the last input, which is the one every test ends with.
fn final_screen(screens: &Screens) -> &[String] {
    screens.last().expect("at least one input was sent")
}

fn drive(install: &Install, inputs: &[&str]) -> Screens {
    drive_run(install, inputs, false).screens
}

/// What one driven session came to.
struct Run {
    screens: Screens,
    /// The status the child exited with on its own by the time the driver gave up on it, or
    /// `None` when it had to be killed.
    exit: Option<i32>,
    /// Whether the child asked the terminal for the kitty keyboard protocol's disambiguated
    /// keys, which it does only where the terminal says it speaks it.
    enhanced: bool,
    /// The last text the child put on the terminal's clipboard, through OSC 52.
    clipboard: Option<String>,
}

/// [`drive`], with the terminal answering the child's keyboard protocol query as one that speaks
/// it when `kitty` is set, and as one that does not otherwise.
fn drive_run(install: &Install, inputs: &[&str], kitty: bool) -> Run {
    drive_run_with(install, &["-c"], inputs, kitty)
}

/// [`drive_run`], with `args` on meka's command line in place of the `-c` every other test runs
/// with.
fn drive_run_with(install: &Install, args: &[&str], inputs: &[&str], kitty: bool) -> Run {
    drive_run_sized(install, args, inputs, kitty, COLUMNS)
}

/// [`drive_run_with`] on a terminal `columns` wide, for what only shows at a width the default
/// does not reach.
fn drive_run_sized(
    install: &Install,
    args: &[&str],
    inputs: &[&str],
    kitty: bool,
    columns: u16,
) -> Run {
    let mut master: RawFd = -1;
    let size = libc::winsize {
        ws_row: ROWS,
        ws_col: columns,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let prepared = PreparedExec::meka(install, args);
    // SAFETY: `forkpty` with a null termios and a sized window is the documented way to get a pty
    // pair plus a child. The child branch below does nothing but `exec`, which is the one thing
    // that is defined after `fork` in a threaded process.
    let pid = unsafe {
        libc::forkpty(
            &mut master,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &size as *const libc::winsize as *mut libc::winsize,
        )
    };
    assert!(pid >= 0, "forkpty failed");
    if pid == 0 {
        // SAFETY: the child of a threaded process, so only what `PreparedExec::exec` promises.
        unsafe { prepared.exec() }
    }
    keep_master_here(master);
    let (screens, enhanced, clipboard) = pump(master, inputs, kitty, columns);
    // SAFETY: `master` is the fd `forkpty` handed back and is still open here, and `pid` is the
    // child it forked, reaped here and nowhere else.
    let exit = unsafe {
        libc::close(master);
        let mut status = 0;
        if libc::waitpid(pid, &mut status, libc::WNOHANG) == pid {
            Some(if libc::WIFEXITED(status) {
                libc::WEXITSTATUS(status)
            } else {
                -1
            })
        } else {
            libc::kill(pid, libc::SIGKILL);
            libc::waitpid(pid, &mut status, 0);
            None
        }
    };
    Run {
        screens,
        exit,
        enhanced,
        clipboard,
    }
}

/// Keep the pty's master out of every child this process forks from here on: `forkpty` hands it
/// back without close-on-exec, and tests fork in parallel, so another test's child would carry
/// a copy of it and keep that terminal open past the close that is meant to hang it up.
fn keep_master_here(master: RawFd) {
    // SAFETY: `master` is the fd `forkpty` just handed back.
    let flagged = unsafe { libc::fcntl(master, libc::F_SETFD, libc::FD_CLOEXEC) };
    assert!(flagged == 0, "close-on-exec on the master");
}

/// A `meka` command line with everything its exec needs laid out ahead of the fork: the child of
/// a threaded process may allocate nothing and take no lock between the fork and the exec, since
/// a lock another thread held at the fork is held forever in the child. Two tests forking at the
/// same instant are enough to hit that, and the child then hangs before `meka` ever runs.
struct PreparedExec {
    program: CString,
    directory: CString,
    /// One past the highest descriptor the child closes, read ahead of the fork since the child
    /// may not ask: the soft limit, within bounds, as a test process holds a few hundred at most
    /// and a limit in the millions would have the child closing for a while.
    descriptors: libc::c_int,
    /// Owned by the two vectors below the pointers, which point into them.
    _arguments: Vec<CString>,
    _environment: Vec<CString>,
    argv: Vec<*const libc::c_char>,
    envp: Vec<*const libc::c_char>,
}

impl PreparedExec {
    fn meka(install: &Install, args: &[&str]) -> Self {
        let mut command = support::meka();
        command
            .args(args)
            .current_dir(install.work_dir())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("TERM", "xterm-256color")
            // The sandbox advisories are facts about the kernel the suite runs on, and would be
            // rows of their own on the screen. The two `rmcp` directives are meka's defaults,
            // restated because `RUST_LOG` replaces the filter outright.
            .env(
                "RUST_LOG",
                "warn,meka::sandbox=error,rmcp::transport::common::client_side_sse=error,\
                 rmcp::transport::worker=off",
            );
        install.env(&mut command);
        let c_string = |text: &std::ffi::OsStr| CString::new(text.as_bytes()).expect("no NUL");
        let arguments: Vec<CString> = std::iter::once(command.get_program())
            .chain(command.get_args())
            .map(c_string)
            .collect();
        let environment: Vec<CString> = command
            .get_envs()
            .filter_map(|(key, value)| {
                let mut pair = key.to_os_string();
                pair.push("=");
                pair.push(value?);
                Some(c_string(&pair))
            })
            .collect();
        let with_end = |strings: &[CString]| -> Vec<*const libc::c_char> {
            strings
                .iter()
                .map(|string| string.as_ptr())
                .chain(std::iter::once(std::ptr::null()))
                .collect()
        };
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `getrlimit` writes the struct it is given.
        let read = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) };
        assert!(read == 0, "the descriptor limit");
        Self {
            descriptors: libc::c_int::try_from(limit.rlim_cur)
                .unwrap_or(libc::c_int::MAX)
                .clamp(1024, 65_536),
            program: c_string(command.get_program()),
            directory: c_string(install.work_dir().as_os_str()),
            argv: with_end(&arguments),
            envp: with_end(&environment),
            _arguments: arguments,
            _environment: environment,
        }
    }

    /// Replace the process with the command. Only `close`, `chdir`, `execve` and `_exit`, each
    /// async-signal-safe, which is what makes this callable in the child of a threaded process.
    /// Every descriptor past the three standard ones is closed first: a master another test
    /// opened between its `forkpty` and its close-on-exec flag would otherwise ride along.
    ///
    /// # Safety
    ///
    /// Call only in a child just forked from this process; it never returns.
    unsafe fn exec(&self) -> ! {
        // SAFETY: the strings and pointer arrays were laid out before the fork and are
        // NUL-terminated; nothing here allocates.
        unsafe {
            for descriptor in 3..self.descriptors {
                libc::close(descriptor);
            }
            libc::chdir(self.directory.as_ptr());
            libc::execve(
                self.program.as_ptr(),
                self.argv.as_ptr(),
                self.envp.as_ptr(),
            );
            libc::_exit(127)
        }
    }
}

/// Read the master side into a terminal emulator, snapshotting the screen and sending the next
/// input whenever the child falls quiet. Answers the child's keyboard protocol query the way a
/// terminal would, since crossterm waits two seconds for the answer, and reports whether the
/// child then asked for the protocol.
fn pump(
    master: RawFd,
    inputs: &[&str],
    kitty: bool,
    columns: u16,
) -> (Screens, bool, Option<String>) {
    let mut parser = vt100::Parser::new(ROWS, columns, 0);
    let mut screens = Vec::new();
    let mut enhanced = false;
    let mut raw: Vec<u8> = Vec::new();
    let mut pending: Vec<&str> = inputs.to_vec();
    pending.reverse();
    let mut buffer = [0u8; 65536];
    let mut last_activity = Instant::now();
    // Quiet is the screen standing still, not the child writing nothing: the status row counts
    // the seconds of a turn and the working row turns its spinner, so bytes keep coming while
    // the agent works. The last row is left out of the comparison and the working row is read
    // as one row for that reason.
    let mut settled = settled_rows(&parser);
    // The first send waits longer: the child has a store to migrate before it draws anything.
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
            if chunk.windows(4).any(|window| window == b"\x1b[?u") {
                // The flags report says "I speak it"; the device attributes alone say "I do not".
                let answer: &[u8] = if kitty {
                    b"\x1b[?0u\x1b[?1;2c"
                } else {
                    b"\x1b[?1;2c"
                };
                write_all(master, answer);
            }
            if chunk.windows(5).any(|window| window == b"\x1b[>1u") {
                enhanced = true;
            }
            raw.extend_from_slice(chunk);
            parser.process(chunk);
            let now = settled_rows(&parser);
            if now != settled {
                settled = now;
                last_activity = Instant::now();
            }
        } else if read == 0 {
            break;
        }
        // Judged whether or not bytes arrived: the working row's spinner has the child writing
        // every frame of a turn, and a capture mid-turn is what some tests are for.
        if last_activity.elapsed() > quiet_needed {
            match pending.pop() {
                Some(line) => {
                    screens.push(rows_of(&parser));
                    write_all(master, line.as_bytes());
                    // A lone control byte is a keystroke, not a line.
                    if line.len() > 1 || !line.starts_with(|c: char| c.is_control()) {
                        write_all(master, b"\r");
                    }
                    last_activity = Instant::now();
                    quiet_needed = Duration::from_millis(1500);
                }
                // Everything sent and the child has gone quiet: it has exited or is idle, and
                // either way there is nothing left to capture.
                None if last_activity.elapsed() > Duration::from_secs(4) => break,
                None => {}
            }
        }
    }
    (screens, enhanced, last_clipboard_write(&raw))
}

/// The text of the last OSC 52 clipboard write in `output`, as a terminal would have taken it.
fn last_clipboard_write(output: &[u8]) -> Option<String> {
    let marker = b"\x1b]52;c;";
    let start = output
        .windows(marker.len())
        .rposition(|window| window == marker)?
        + marker.len();
    let end = output[start..].iter().position(|&byte| byte == 0x07)? + start;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&output[start..end])
        .ok()?;
    String::from_utf8(decoded).ok()
}

/// Every row but the last, which is the status row while the composer is up.
fn settled_rows(parser: &vt100::Parser) -> Vec<String> {
    parser
        .screen()
        .rows(0, parser.screen().size().1)
        .take(usize::from(ROWS) - 1)
        .map(|row| row.trim_end().to_string())
        // The working row turns its spinner and counts its seconds for as long as the agent
        // works; for the quiet's purposes it is one row.
        .map(|row| {
            if is_working_row(&row) {
                "<working>".to_string()
            } else {
                row
            }
        })
        .collect()
}

/// Whether `row` is the working row above the composer: it opens with the spinner, or stands
/// still as a wait while the turn is the user's to answer.
fn is_working_row(row: &str) -> bool {
    row.starts_with("• Waiting for your answer")
        || row
            .chars()
            .next()
            .is_some_and(|glyph| "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏".contains(glyph))
}

fn rows_of(parser: &vt100::Parser) -> Vec<String> {
    let mut rows: Vec<String> = parser
        .screen()
        .rows(0, parser.screen().size().1)
        .map(|row| row.trim_end().to_string())
        .collect();
    while rows.last().is_some_and(String::is_empty) {
        rows.pop();
    }
    rows
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

/// The rows `first..=last` are a block of their own: exactly one blank row above and one below,
/// with a row of content beyond each.
fn assert_set_off_by_one_blank(rows: &[String], first: usize, last: usize, what: &str) {
    assert!(
        first >= 2 && rows[first - 1].is_empty() && !rows[first - 2].is_empty(),
        "{what}: exactly one blank above it: {rows:#?}"
    );
    assert!(
        rows.get(last + 1).is_some_and(String::is_empty)
            && rows.get(last + 2).is_some_and(|row| !row.is_empty()),
        "{what}: exactly one blank below it: {rows:#?}"
    );
}

const TWO_TURNS: &str = r#"[
 [{"type":"text","text":"First answer."},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"tool_use_start","id":"t1","name":"schedule_list"},
  {"type":"tool_use_end","input":{}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"Second answer."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;

const TWO_WRITES: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"file_write"},
  {"type":"tool_use_end","input":{"path":"a.txt","content":"a"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"wrote a"},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"tool_use_start","id":"t2","name":"file_write"},
  {"type":"tool_use_end","input":{"path":"b.txt","content":"b"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"wrote b"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;

/// The shape every other case is a variation on: the prompt in its block, the answer under
/// it, and the composer and status row at the bottom. `/approvals on` flips the switch for the
/// session and writes it to the row, where a resume from any host and a scheduled fire read it.
#[test]
fn a_turn_draws_the_prompt_and_its_answer_above_the_composer() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_TURNS, &["/approvals on", "hi", "/exit"]);
    let rows = final_screen(&screens);
    assert!(
        rows.iter().any(|row| row == "• Approvals set to: on"),
        "the switch is confirmed on screen: {rows:#?}"
    );
    let prompt = rows
        .iter()
        .position(|row| row == "> hi")
        .unwrap_or_else(|| panic!("the prompt in its block: {rows:#?}"));
    let answer = rows
        .iter()
        .position(|row| row == "  First answer.")
        .unwrap_or_else(|| panic!("the answer: {rows:#?}"));
    assert!(
        prompt < answer,
        "the answer sits under the prompt: {rows:#?}"
    );
    assert!(
        rows.last().is_some_and(|row| row.starts_with("  read ")),
        "the status row under the composer names the permission level: {rows:#?}"
    );
    let connection = rusqlite::Connection::open(install.database()).expect("open the store");
    let recorded: i64 = connection
        .query_row("SELECT approvals FROM sessions", [], |row| row.get(0))
        .expect("exactly one session row");
    assert_eq!(recorded, 1, "the row carries the switch");
}

/// `a` at the approval prompt allows the tool until the session ends: the second call to it is
/// not put to the user, and both writes land. Driven through the terminal because the answer
/// travels from the screen back to the frontend, and only the real channel proves it gets there
/// as a sticky decision rather than a bare yes.
#[test]
fn an_always_answer_at_the_approval_prompt_covers_the_next_call_to_the_tool() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_WRITES, &[
        "/approvals on",
        "write a",
        "a",
        "write b",
        "/exit",
    ]);
    let asked = &screens[2];
    assert!(
        asked.iter().any(|row| row == "  └ content: a")
            && asked
                .iter()
                .any(|row| row.contains("Allow for the session  a")),
        "the prompt shows the call's arguments as a tree and offers the sticky answers: \
         {asked:#?}"
    );
    assert!(
        !asked.iter().any(|row| row.starts_with("  read ")),
        "the prompt replaces the composer and the status row: {asked:#?}"
    );
    let rows = final_screen(&screens);
    let recorded = rows
        .iter()
        .filter(|row| row.contains("file_write allowed"))
        .count();
    assert_eq!(
        recorded, 1,
        "the first write is put to the user and the second is covered by `a`: {rows:#?}"
    );
    assert!(
        rows.iter()
            .any(|row| row == "• file_write allowed for the session"),
        "the decision is one row on the transcript: {rows:#?}"
    );
    assert!(
        rows.iter().any(|row| row == "• file_write a.txt  +1 -0")
            && rows.iter().any(|row| row == "  └ 1 + a"),
        "the write draws what it changed as a diff, counted on its row: {rows:#?}"
    );
    let work = install.root().join("work");
    assert!(work.join("a.txt").exists(), "the approved write ran");
    assert!(
        work.join("b.txt").exists(),
        "the second write ran without a question: {rows:#?}"
    );
}

/// `always` answers for the rest of the session it was given in, and `/fork` moves the screen
/// into another session: the same tool is put to the user again in the copy.
#[test]
fn an_always_answer_does_not_survive_a_fork() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_WRITES, &[
        "/approvals on",
        "write a",
        "a",
        "/fork",
        "write b",
        "y",
        "/exit",
    ]);
    let rows = final_screen(&screens);
    assert!(
        rows.iter().any(|row| row.contains("Forked session")),
        "the fork has to have happened for the rest to mean anything: {rows:#?}"
    );
    let recorded = rows
        .iter()
        .filter(|row| row.contains("file_write allowed"))
        .count();
    assert_eq!(
        recorded, 2,
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

/// A scheduled fire is announced above the reply it triggers, as a stage direction naming the job
/// and the opening of its prompt, so the answer that appears while the user is idle is not the
/// model speaking unprompted. The prompt the model is given is not echoed, the way no tool's
/// output is.
#[test]
fn a_scheduled_fire_is_announced_as_a_stage_direction_above_its_reply() {
    const SCHEDULE_THEN_FIRE: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"schedule_create"},
  {"type":"tool_use_end","input":{"prompt":"SCREEN_DELIVERED_MARKER","at":"1s"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"scheduled"},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"text","text":"SCREEN_SCHEDULED_REPLY"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = screen_install("read", "", "\n[schedule]\npoll_interval = \"200ms\"\n");
    let screens = run_screen(&install, SCHEDULE_THEN_FIRE, &["remind me", "/exit"]);
    let rows = final_screen(&screens);
    let announced = rows
        .iter()
        .position(|row| {
            row.starts_with("• scheduled job ") && row.ends_with(" fired: SCREEN_DELIVERED_MARKER")
        })
        .unwrap_or_else(|| panic!("the fire is a stage direction naming the job: {rows:#?}"));
    let replied = rows
        .iter()
        .position(|row| row.contains("SCREEN_SCHEDULED_REPLY"))
        .unwrap_or_else(|| panic!("the fire's reply must reach the screen: {rows:#?}"));
    assert!(
        announced < replied,
        "the direction is above the reply it accounts for: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains("[Scheduled job")),
        "the model's prompt is not echoed: {rows:#?}"
    );
    assert_set_off_by_one_blank(rows, announced, announced, "the fire's direction");
}

/// A refused command answers in a panel, quoting the value as a value, and the panel leaves no
/// row behind once Esc closes it.
#[test]
fn a_rewind_with_a_bad_count_is_refused_in_a_panel() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_TURNS, &["/rewind all", "\x1b", "/exit"]);
    let open = &screens[1];
    assert!(
        open.iter()
            .any(|row| row == "  /rewind takes a turn count of 1 or more, not 'all'"),
        "the refusal is the panel's body: {open:#?}"
    );
    assert_eq!(open.last().map(String::as_str), Some("  esc to close"));
    let rows = final_screen(&screens);
    assert!(
        !rows
            .iter()
            .any(|row| row.contains("/rewind takes") || row.contains("Rewound")),
        "nothing stays on the transcript, and nothing was rewound: {rows:#?}"
    );
}

/// What a command says back that is not a fact of the session is a panel, not a row: `/export`
/// before any turn has nothing to export, and says so until Esc.
#[test]
fn a_command_s_refusal_is_a_panel_that_leaves_no_row() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_TURNS, &["/export", "\x1b", "/exit"]);
    let open = &screens[1];
    assert!(
        open.iter()
            .any(|row| row == "  No active session to export."),
        "the refusal is the panel's body: {open:#?}"
    );
    assert_eq!(open.last().map(String::as_str), Some("  esc to close"));
    let rows = final_screen(&screens);
    assert!(
        !rows.iter().any(|row| row.contains("No active session")),
        "nothing stays on the transcript: {rows:#?}"
    );
}

/// `/memory` is a picker of the saved memories by name and description, and Enter shows the
/// picked one in full.
#[test]
fn the_memory_picker_shows_the_picked_memory() {
    const SAVES_A_MEMORY: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"memory_write"},
  {"type":"tool_use_end","input":{"name":"ramen-protocol","description":"What to do when the ramen runs out","body":"Buy more."}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"Saved."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, SAVES_A_MEMORY, &[
        "remember", "/memory", "\r", "\x1b", "/exit",
    ]);
    let picker = &screens[2];
    assert!(
        picker
            .iter()
            .any(|row| row == "  ramen-protocol  What to do when the ramen runs out"),
        "the picker lists the memory by name and description: {picker:#?}"
    );
    let shown = &screens[3];
    assert!(
        shown.iter().any(|row| row.contains("Buy more.")),
        "Enter shows the picked memory in full: {shown:#?}"
    );
}

/// A turn that fails mid-answer keeps what streamed on screen, with the failure under it.
#[test]
fn a_failed_turn_shows_its_partial_answer_in_its_own_turn() {
    const FAILS_MID_ANSWER: &str = r#"[
 [{"type":"text","text":"partial answer before the failure"},
  {"type":"fail","message":"provider exploded"}],
 [{"type":"text","text":"recovered"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, FAILS_MID_ANSWER, &["first", "second", "/exit"]);
    let rows = final_screen(&screens);
    let partial = rows
        .iter()
        .position(|row| row.contains("partial answer before the failure"))
        .unwrap_or_else(|| panic!("what streamed stays: {rows:#?}"));
    let error = rows
        .iter()
        .position(|row| row.starts_with("■ ") && row.contains("provider exploded"))
        .unwrap_or_else(|| panic!("the failure is said: {rows:#?}"));
    let recovered = rows
        .iter()
        .position(|row| row == "  recovered")
        .unwrap_or_else(|| panic!("the next turn answers: {rows:#?}"));
    assert!(
        partial < error && error < recovered,
        "the failure sits under its turn and ahead of the next: {rows:#?}"
    );
}

/// A session that recorded reasoning shows it again on resume, rendered rather than as source,
/// when the config asks for the whole block.
#[test]
fn a_resumed_session_replays_the_reasoning_it_recorded() {
    const REASONED: &str = r#"[
 [{"type":"thinking_delta","text":"**Weighing it** then deciding."},
  {"type":"thinking_complete"},
  {"type":"text","text":"Here is the answer."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = screen_install("read", "", "\n[thinking]\nshow_content = true\n");
    let first = run_screen(&install, REASONED, &["think about it", "/exit"]);
    let rows = final_screen(&first);
    assert!(
        rows.iter()
            .any(|row| row.contains("Weighing it then deciding")),
        "the live turn has to render it before a replay can repeat it: {rows:#?}"
    );
    let resumed = run_screen(&install, REASONED, &["/exit"]);
    let rows = final_screen(&resumed);
    assert!(
        rows.iter()
            .any(|row| row.contains("Weighing it then deciding")),
        "the resumed session dropped the reasoning it recorded: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains("**")),
        "and shows it rendered, not as source: {rows:#?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.starts_with("• Resuming session ")),
        "the resume is announced first: {rows:#?}"
    );
}

/// A resumed session holds the checklist it was left with: the item's id is the one the next
/// edit lands on.
#[test]
fn a_resumed_session_holds_the_checklist_it_was_left_with() {
    const FIRST: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"checklist_add"},
  {"type":"tool_use_end","input":{"items":["ask Sam"]}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"tool_use_start","id":"t2","name":"checklist_edit"},
  {"type":"tool_use_end","input":{"id":1,"status":"deferred","reason":"waiting on Sam"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"Waiting."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    const SECOND: &str = r#"[
 [{"type":"tool_use_start","id":"t3","name":"checklist_edit"},
  {"type":"tool_use_end","input":{"id":1,"status":"in_progress"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"On it."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = screen_install("read", "", "");
    let first = run_screen(&install, FIRST, &["plan it", "/exit"]);
    let rows = final_screen(&first);
    let row = rows
        .iter()
        .position(|row| row.contains("» ask Sam (waiting on Sam)"))
        .unwrap_or_else(|| panic!("{rows:#?}"));
    assert!(
        rows[row + 1].is_empty(),
        "the row says the item is parked and nothing sits under it: {rows:#?}"
    );
    let resumed = run_screen(&install, SECOND, &["carry on", "/exit"]);
    let rows = final_screen(&resumed);
    assert!(
        rows.iter().any(|row| row.contains("› ask Sam")),
        "the edit landed on the recovered id: {rows:#?}"
    );
}

/// The nudge is a stage direction on the transcript, set off on both sides like a tool call: the
/// model's reply above it, the next round below it, and a blank line between it and each.
#[test]
fn a_nudge_is_a_stage_direction_set_off_from_the_reply_and_the_next_round() {
    const SCRIPT: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"checklist_add"},
  {"type":"tool_use_end","input":{"items":["write the tests"]}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"I have planned it."},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"tool_use_start","id":"t2","name":"checklist_edit"},
  {"type":"tool_use_end","input":{"id":1,"status":"completed"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"Done."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, SCRIPT, &["plan and do it", "/exit"]);
    let rows = final_screen(&screens);
    let at = rows
        .iter()
        .position(|row| row == "• checklist: 1 open item, continuing, nudge 1 of 3")
        .unwrap_or_else(|| panic!("the nudge is a row of its own: {rows:#?}"));
    assert_set_off_by_one_blank(rows, at, at, "the nudge");
    assert!(
        rows[..at]
            .iter()
            .any(|row| row.contains("I have planned it.")),
        "the reply it annotates is above it: {rows:#?}"
    );
    assert!(
        rows[at..].iter().any(|row| row == "  Done."),
        "and the round it opens is below it: {rows:#?}"
    );
}

/// The whole arc on the screen, through its watcher: an item deferred on a running task lets the
/// turn end, the task's end wakes the screen, the report is a turn of its own, and that turn is
/// held by the item until the model disposes of it, with the stage direction saying so between
/// the model's note and the round that does.
///
/// The task is played by a thread on the store, because the script cannot name an id the real
/// tool would make up: recorded running once the session exists, and ended once the deferral is
/// on the record, so its end lands inside the deferring turn, the window the report has to wait
/// out.
#[test]
fn an_outcome_turn_is_held_by_the_item_deferred_on_its_task() {
    const TASK: &str = "1a2b3c4d-0000-4000-8000-000000000000";
    const SCRIPT: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"checklist_add"},
  {"type":"tool_use_end","input":{"items":["read the build log"]}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"sleep","ms":600},
  {"type":"tool_use_start","id":"t2","name":"checklist_edit"},
  {"type":"tool_use_end","input":{"id":1,"status":"deferred","reason":"building","task":"1a2b3c4d"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"Build started."},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"text","text":"Noted."},
  {"type":"message_end","stop_reason":"end_turn"}],
 [{"type":"tool_use_start","id":"t3","name":"checklist_edit"},
  {"type":"tool_use_end","input":{"id":1,"status":"completed"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"Read it."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = screen_install(
        "read",
        "",
        "\n[background]\nenabled = true\n\n[schedule]\npoll_interval = \"200ms\"\n",
    );
    let database = install.database();
    let player = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(60);
        let (connection, session_id) = loop {
            assert!(Instant::now() < deadline, "the session never appeared");
            if let Ok(connection) = rusqlite::Connection::open(&database)
                && let Ok(id) = connection.query_row("SELECT id FROM sessions LIMIT 1", [], |row| {
                    row.get::<_, String>(0)
                })
            {
                break (connection, id);
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        connection
            .execute(
                "INSERT INTO background_tasks (id, session_id, tool, label, status, started_at) \
                 VALUES (?1, ?2, 'shell_execute', 'make', 'running', ?3)",
                rusqlite::params![TASK, session_id, chrono::Utc::now().to_rfc3339()],
            )
            .expect("a running task");
        loop {
            assert!(Instant::now() < deadline, "the deferral never landed");
            let deferred: i64 = connection
                .query_row(
                    "SELECT COUNT(*) FROM messages WHERE content LIKE '%Item 1 is deferred%'",
                    [],
                    |row| row.get(0),
                )
                .expect("count");
            if deferred > 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        connection
            .execute(
                "UPDATE background_tasks SET status = 'completed', outcome = ?3, \
                 finished_at = ?2 WHERE id = ?1",
                rusqlite::params![TASK, chrono::Utc::now().to_rfc3339(), "ok\n"],
            )
            .expect("the task ends");
    });
    // The pause in the second round is where the thread records the task; it has to stay under
    // the driver's quiet window, or the driver reads the silence as idle and types the exit.
    let screens = run_screen(&install, SCRIPT, &["build and read", "/exit"]);
    player.join().expect("the task was played");

    let rows = final_screen(&screens);
    let body = &rows[rows
        .iter()
        .position(|row| row == "> build and read")
        .unwrap_or_else(|| panic!("the prompt: {rows:#?}"))..];
    let position = |needle: &str| {
        body.iter()
            .position(|row| row.contains(needle))
            .unwrap_or_else(|| panic!("no row contains {needle:?}: {body:#?}"))
    };
    let started = position("Build started.");
    let report = position("• background task 1a2b3c4d finished after ");
    let noted = position("Noted.");
    let nudge = position("• checklist: 1 open item, continuing, nudge 1 of 3");
    let read = position("Read it.");
    assert!(
        started < report && report < noted && noted < nudge && nudge < read,
        "the deferring turn ends, the report opens a turn, and the nudge holds it: {body:#?}"
    );
    assert_eq!(
        body.iter()
            .filter(|row| row.contains("continuing, nudge"))
            .count(),
        1,
        "the deferring turn itself was not nudged: {body:#?}"
    );
    assert!(body[report].ends_with(", reporting"), "{body:#?}");
    assert!(
        !body
            .iter()
            .any(|row| row.contains("[Background task reporting") || row == "ok"),
        "the prompt and the output stay with the model: {body:#?}"
    );
    assert_set_off_by_one_blank(body, report, report, "the report's direction");
}

/// A message typed while a turn runs goes to the inbox as a steer: shown at once as waiting, and
/// read by the loop at the turn's next round, after which the screen says so.
#[test]
fn a_message_typed_during_a_turn_is_a_steer_the_turn_reads() {
    const SLOW: &str = r#"[
 [{"type":"text","text":"Starting."},
  {"type":"sleep","ms":2000},
  {"type":"tool_use_start","id":"t1","name":"schedule_list"},
  {"type":"tool_use_end","input":{}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"Done after the steer."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, SLOW, &["go", "also check the logs", "/exit"]);
    let typed = &screens[2];
    assert!(
        typed.iter().any(|row| row == "> also check the logs"),
        "the steer is on screen the moment it is typed: {typed:#?}"
    );
    let rows = final_screen(&screens);
    let steer = rows
        .iter()
        .position(|row| row == "  (steer)")
        .unwrap_or_else(|| panic!("the steer is marked read once the loop took it: {rows:#?}"));
    let done = rows
        .iter()
        .position(|row| row == "  Done after the steer.")
        .unwrap_or_else(|| panic!("the turn went on: {rows:#?}"));
    assert!(steer < done, "{rows:#?}");
    assert!(
        !rows.iter().any(|row| row.contains("waiting")),
        "nothing is left waiting once the turn has read it: {rows:#?}"
    );
    let connection = rusqlite::Connection::open(install.database()).expect("open the store");
    let delivered: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM inbox_items WHERE class = 'steer' AND delivered_at IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .expect("the inbox table");
    assert_eq!(delivered, 1, "the steer went through the inbox");
}

/// A turn that would run for twenty seconds, for the tests that act on it from the keyboard.
const LONG_TURN: &str = r#"[
 [{"type":"text","text":"Starting."},
  {"type":"sleep","ms":20000},
  {"type":"text","text":" Never shown."},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;

/// Esc while the agent works stops the turn, the way the first Ctrl+C does.
#[test]
fn esc_stops_the_turn() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, LONG_TURN, &["go", "\x1b", "/exit"]);
    let rows = final_screen(&screens);
    assert!(
        rows.iter().any(|row| row == "! interrupted"),
        "the turn was stopped: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains("Never shown")),
        "nothing past the stop reached the screen: {rows:#?}"
    );
}

/// Leaving while the agent works stops the turn first, so the process is gone at once rather than
/// working on headless until the turn ends by itself.
#[test]
fn leaving_during_a_turn_stops_it() {
    let install = screen_install("read", "", "");
    install.write_script(LONG_TURN);
    let run = drive_run(&install, &["go", "/exit"], false);
    assert!(
        run.exit.is_some(),
        "meka was still running the turn after the screen left: {:#?}",
        run.screens
    );
}

/// Three Ctrl+C presses with nothing running leave the way `/exit` does, with the interrupt's
/// status: the host's own teardown runs rather than the screen cutting the process short.
#[test]
fn the_third_interrupt_press_leaves_with_status_130() {
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    let run = drive_run(&install, &["\x03", "\x03", "\x03"], false);
    assert_eq!(run.exit, Some(130), "{:#?}", run.screens);
}

/// Where the terminal speaks the kitty keyboard protocol, the screen asks for its disambiguated
/// keys, and Shift+Enter is then a newline in the composer rather than Enter.
#[test]
fn shift_enter_is_a_newline_under_the_kitty_keyboard_protocol() {
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    let run = drive_run(&install, &["one\x1b[13;2utwo", "/exit"], true);
    assert!(run.enhanced, "the screen asked for the protocol");
    let rows = final_screen(&run.screens);
    assert!(
        rows.iter().any(|row| row == "> one") && rows.iter().any(|row| row == "  two"),
        "the prompt is two lines: {rows:#?}"
    );
    let plain = drive_run(&install, &["/exit"], false);
    assert!(
        !plain.enhanced,
        "a terminal that does not speak it is not asked"
    );
}

/// A paste that is one path to an image file attaches the file, and Enter sends it with the
/// prompt: the transcript shows the image row, and the path is never typed.
#[test]
fn a_pasted_image_path_is_attached_and_sent() {
    let install = screen_install("read", "", "");
    let path = install.work_dir().join("dot.png");
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        2,
        2,
        image::Rgba([255, 0, 0, 255]),
    ))
    .write_to(&mut png, image::ImageFormat::Png)
    .expect("a png");
    std::fs::write(&path, png.into_inner()).expect("write the png");
    let paste = format!("\x1b[200~{}\x1b[201~", path.display());
    let screens = run_screen(&install, TWO_TURNS, &[&paste, "/exit"]);
    let rows = final_screen(&screens);
    assert!(
        rows.iter().any(|row| row == "• image"),
        "the image went with the prompt: {rows:#?}"
    );
    assert!(
        rows.iter().any(|row| row == "  First answer."),
        "the turn ran on it: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains("dot.png")),
        "the path was attached, not typed: {rows:#?}"
    );
}

/// `!` into the empty composer is the shell mode: the glyph turns to `!`, the status row says so,
/// and what is typed next runs as a command.
#[test]
fn a_bang_enters_the_shell_mode_and_enter_runs_the_command() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_TURNS, &["!", "echo shell-mode-ran", "/exit"]);
    let armed = &screens[1];
    let composer = armed
        .iter()
        .rposition(|row| row.starts_with("! "))
        .or_else(|| armed.iter().rposition(|row| row == "!"));
    assert!(
        composer.is_some(),
        "the composer's glyph is the shell's: {armed:#?}"
    );
    assert!(
        armed
            .last()
            .is_some_and(|row| row == "  Run command in shell"),
        "the status row says what Enter will do, and nothing else: {armed:#?}"
    );
    let rows = final_screen(&screens);
    assert!(
        rows.iter().any(|row| row == "> !echo shell-mode-ran")
            && rows.iter().any(|row| row == "  shell-mode-ran"),
        "the command ran and its output is on the transcript: {rows:#?}"
    );
}

/// A `!command` runs beside the screen rather than holding it, and Esc kills it with what it
/// started: the command's own prompt row stays, `! interrupted` marks the stop, and nothing it
/// would have printed after arrives.
#[test]
fn esc_stops_a_running_shell_command() {
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    let started = Instant::now();
    let run = drive_run(
        &install,
        &["!sleep 30; echo survived", "\x1b", "/exit"],
        false,
    );
    let rows = final_screen(&run.screens);
    assert!(
        rows.iter().any(|row| row == "> !sleep 30; echo survived"),
        "the command is on the transcript: {rows:#?}"
    );
    assert!(
        rows.iter().any(|row| row == "! interrupted"),
        "the stop is marked where a stopped turn's is: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row == "  survived"),
        "what the command would have gone on to print never arrives: {rows:#?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(25),
        "the command was killed rather than waited out: {:?}",
        started.elapsed()
    );
    assert_eq!(run.exit, Some(0), "{rows:#?}");
}

/// The terminal going away is a hangup: the screen leaves the way it does on a key, so the
/// `!command` it was running dies with it rather than being left behind, and the process ends on
/// its own exit, the interrupted one, rather than aborting on the dead terminal.
#[test]
fn a_hangup_kills_the_running_command_and_leaves() {
    let (exit, survivors, screens) = terminal_taken_away(false);
    assert_eq!(
        exit,
        Some(130),
        "the hangup is an interrupted leave: {screens:#?}"
    );
    assert!(survivors.is_empty(), "the command died with the screen");
}

/// The same end when no signal says the terminal is gone and only the reads and writes do, which
/// is the order macOS notices a closed pty in, and what a terminal held through another process
/// looks like anywhere.
#[test]
fn a_terminal_that_dies_without_a_signal_leaves_the_same_way() {
    let (exit, survivors, screens) = terminal_taken_away(true);
    assert_eq!(
        exit,
        Some(130),
        "the dead terminal is an interrupted leave: {screens:#?}"
    );
    assert!(survivors.is_empty(), "the command died with the screen");
}

/// Run the screen with a `!command` going, then close the terminal under it: the exit status it
/// ends with (`None` when it had to be killed), the command's surviving pids, and the screens.
/// With `detached`, the child lets go of the terminal as its controlling one first, so closing
/// it sends no hangup and only the failing reads and writes tell.
fn terminal_taken_away(detached: bool) -> (Option<i32>, Vec<i32>, Screens) {
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    // One marker per variant, so neither test reads the other's command as its own.
    let marker = if detached {
        "sleep 300.322"
    } else {
        "sleep 300.321"
    };
    let mut master: RawFd = -1;
    let size = libc::winsize {
        ws_row: ROWS,
        ws_col: COLUMNS,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let prepared = PreparedExec::meka(&install, &["-c"]);
    // SAFETY: as in `drive_run`.
    let pid = unsafe {
        libc::forkpty(
            &mut master,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &size as *const libc::winsize as *mut libc::winsize,
        )
    };
    assert!(pid >= 0, "forkpty failed");
    if pid == 0 {
        // SAFETY: the child of a threaded process: async-signal-safe calls only. A session leader
        // letting go of its terminal is sent the hangup itself, so it is ignored for that moment.
        unsafe {
            if detached {
                libc::signal(libc::SIGHUP, libc::SIG_IGN);
                libc::ioctl(0, libc::TIOCNOTTY);
                libc::signal(libc::SIGHUP, libc::SIG_DFL);
            }
            prepared.exec()
        }
    }
    keep_master_here(master);
    let (screens, ..) = pump(master, &[&format!("!{marker}")], false, COLUMNS);
    assert!(
        !processes_running(marker).is_empty(),
        "the command runs before the terminal goes: {screens:#?}"
    );
    // SAFETY: closing the master takes the terminal away; the child is reaped here and nowhere
    // else.
    let exit = unsafe {
        libc::close(master);
        let mut status = 0;
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if libc::waitpid(pid, &mut status, libc::WNOHANG) == pid {
                break Some(if libc::WIFEXITED(status) {
                    libc::WEXITSTATUS(status)
                } else {
                    -1
                });
            }
            if Instant::now() > deadline {
                libc::kill(pid, libc::SIGKILL);
                libc::waitpid(pid, &mut status, 0);
                break None;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    };
    let survivors = processes_running(marker);
    for survivor in &survivors {
        // SAFETY: a pid this test's own command line marks, found just above.
        unsafe {
            libc::kill(*survivor, libc::SIGKILL);
        }
    }
    (exit, survivors, screens)
}

/// The pids whose command line carries `marker`.
fn processes_running(marker: &str) -> Vec<i32> {
    // `/proc` where there is one; macOS has none, and `ps` answers the same question there.
    if let Ok(entries) = std::fs::read_dir("/proc") {
        let mut found = Vec::new();
        for entry in entries.flatten() {
            let Ok(pid) = entry.file_name().to_string_lossy().parse::<i32>() else {
                continue;
            };
            let Ok(cmdline) = std::fs::read(entry.path().join("cmdline")) else {
                continue;
            };
            if String::from_utf8_lossy(&cmdline)
                .replace('\0', " ")
                .contains(marker)
            {
                found.push(pid);
            }
        }
        return found;
    }
    let Ok(listed) = std::process::Command::new("ps")
        .args(["-A", "-o", "pid=,args="])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&listed.stdout)
        .lines()
        .filter_map(|line| {
            let (pid, command) = line.trim_start().split_once(' ')?;
            command.contains(marker).then(|| pid.parse().ok()).flatten()
        })
        .collect()
}

const TWO_WRITES_AT_ONCE: &str = r#"[
 [{"type":"tool_use_start","id":"t1","name":"file_write"},
  {"type":"tool_use_end","input":{"path":"a.txt","content":"a"}},
  {"type":"tool_use_start","id":"t2","name":"file_write"},
  {"type":"tool_use_end","input":{"path":"b.txt","content":"b"}},
  {"type":"message_end","stop_reason":"tool_use"}],
 [{"type":"text","text":"wrote both"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;

/// Two calls dispatched together both ask, one after the other, in whichever order their
/// futures reach the screen: the second waits behind the first rather than replacing it, so
/// each answer lands on the call it was given to and both writes run.
#[test]
fn two_calls_that_ask_at_once_are_asked_in_turn() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_WRITES_AT_ONCE, &[
        "/approvals on",
        "write both",
        "y",
        "y",
        "/exit",
    ]);
    let asked = |screen: &[String]| {
        ["a.txt", "b.txt"]
            .into_iter()
            .find(|file| screen.iter().any(|row| row == &format!("  │ path: {file}")))
    };
    let first = asked(&screens[2]).unwrap_or_else(|| panic!("one call is put: {:#?}", screens[2]));
    let second = asked(&screens[3]).unwrap_or_else(|| {
        panic!(
            "the other call is put once the first is answered: {:#?}",
            screens[3]
        )
    });
    assert_ne!(first, second, "each call is put once: {:#?}", screens[3]);
    let rows = final_screen(&screens);
    assert_eq!(
        rows.iter()
            .filter(|row| *row == "• file_write allowed")
            .count(),
        2,
        "each answer is one row: {rows:#?}"
    );
    let work = install.root().join("work");
    assert!(
        work.join("a.txt").exists() && work.join("b.txt").exists(),
        "both writes ran: {rows:#?}"
    );
}

const SLOW_ANSWER: &str = r#"[
 [{"type":"sleep","ms":4000},
  {"type":"text","text":"done at last"},
  {"type":"message_end","stop_reason":"end_turn"}]
]"#;

/// While the agent works, a row of its own above the composer, under the blank line, turns a
/// spinner and says what the agent is doing, for how long, and that Esc stops it; the row goes
/// when the turn ends.
#[test]
fn the_row_above_the_composer_says_what_the_agent_is_doing() {
    let install = screen_install("read", "", "");
    // Ctrl+L redraws and changes nothing, so each one is a capture a second and a half on; the
    // last falls after the answer however late the mock's sleep began. Pressed mid-turn, it is
    // also the redraw that must not wait on the terminal for anything.
    let screens = run_screen(&install, SLOW_ANSWER, &[
        "hi", "\x0c", "\x0c", "\x0c", "/exit",
    ]);
    let working = &screens[1];
    let at = working
        .iter()
        .position(|row| is_working_row(row))
        .unwrap_or_else(|| panic!("a working row mid-turn: {working:#?}"));
    let words = working[at]
        .get(working[at].find(' ').unwrap_or(0) + 1..)
        .unwrap_or("");
    let seconds = words
        .strip_prefix("Working · ")
        .and_then(|rest| rest.strip_suffix("s · Esc to stop"))
        .unwrap_or_else(|| panic!("the row says what runs and for how long: {words:?}"));
    assert!(seconds.parse::<u64>().is_ok(), "{words:?}");
    assert!(
        working[at + 1].starts_with('>') && working[at + 1].contains("Steer the agent"),
        "the row stands right above the composer: {working:#?}"
    );
    assert!(
        at > 0 && working[at - 1].is_empty(),
        "a blank line keeps the row off the conversation: {working:#?}"
    );
    let rows = final_screen(&screens);
    assert!(
        rows.iter().any(|row| row == "  done at last"),
        "the turn ended: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| is_working_row(row)),
        "the row goes with the turn: {rows:#?}"
    );
}

/// A terminal pastes a line break as the carriage return Enter sends; the composer takes it as
/// the line break it is, so a pasted paragraph is sent as its lines.
#[test]
fn a_pasted_carriage_return_is_a_line_break() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_TURNS, &[
        "\x1b[200~one\rtwo\x1b[201~",
        "/exit",
    ]);
    let rows = final_screen(&screens);
    let at = rows
        .iter()
        .position(|row| row == "> one")
        .unwrap_or_else(|| panic!("the first pasted line is the prompt row: {rows:#?}"));
    assert_eq!(
        rows.get(at + 1).map(String::as_str),
        Some("  two"),
        "the second line follows it: {rows:#?}"
    );
}

/// `meka -r` with nothing after it opens on the sessions to resume, newest first; Down and Enter
/// attach the older one, whose conversation then fills the screen.
#[test]
fn a_bare_resume_opens_the_picker_and_enter_attaches_the_pick() {
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    for topic in ["first topic", "second topic"] {
        let status = install
            .meka(&["--oneshot", "-p", topic])
            .current_dir(install.work_dir())
            .status()
            .expect("a scripted run");
        assert!(status.success(), "seeding '{topic}'");
    }
    let run = drive_run_with(&install, &["-r"], &["\x1b[B", "/exit"], false);
    let opening = &run.screens[0];
    let second = opening
        .iter()
        .position(|row| row.contains("second topic"))
        .unwrap_or_else(|| panic!("the newest session heads the list: {opening:#?}"));
    let first = opening
        .iter()
        .position(|row| row.contains("first topic"))
        .unwrap_or_else(|| panic!("the older session is listed: {opening:#?}"));
    assert!(second < first, "newest first: {opening:#?}");
    assert!(
        opening.iter().any(|row| row.contains("/ to filter")),
        "the list can be filtered: {opening:#?}"
    );
    let rows = final_screen(&run.screens);
    assert!(
        rows.iter().any(|row| row == "> first topic"),
        "the picked session's conversation is on the screen: {rows:#?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.starts_with("• Resuming session ")),
        "and the resume is announced: {rows:#?}"
    );
    assert_eq!(run.exit, Some(0), "{rows:#?}");
}

/// The picker is a table: the facts stand in columns of their own at the right edge, the same
/// on every row, a directory past its column cut from its start so its tail stays, and the title
/// has the rest of the row.
#[test]
fn a_picker_is_a_table_with_the_facts_in_their_columns() {
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    let deep = install
        .work_dir()
        .join("a/directory/deep/enough/to/take/the/facts/share/of/a/row");
    std::fs::create_dir_all(&deep).expect("a deep directory");
    let long_topic = "a title long enough to be cut ".repeat(5);
    for (topic, directory) in [("deep", deep), (long_topic.as_str(), install.work_dir())] {
        let status = install
            .meka(&["--oneshot", "-p", topic])
            .current_dir(directory)
            .status()
            .expect("a scripted run");
        assert!(status.success(), "seeding '{topic}'");
    }
    let run = drive_run_with(&install, &["-r"], &["\x1b", "/exit"], false);
    let opening = &run.screens[0];
    let row_of = |start: &str| {
        opening
            .iter()
            .find(|row| row.starts_with(start))
            .unwrap_or_else(|| panic!("a row opens with {start:?}: {opening:#?}"))
    };
    let cut = row_of("  a title long enough");
    let deep_row = row_of("  deep");
    let facts = "  default  just now";
    let column = cut
        .find(facts)
        .unwrap_or_else(|| panic!("the profile and age columns: {cut:?}"));
    assert_eq!(
        deep_row.find(facts),
        Some(column),
        "the same columns on every row: {cut:?} {deep_row:?}"
    );
    assert!(
        cut.contains("…  ~/work "),
        "the title is cut two columns short of the directory column: {cut:?}"
    );
    assert!(
        deep_row.contains("  …") && deep_row.contains("share/of/a/row  default"),
        "a long directory is cut from its start and fills its column: {deep_row:?}"
    );
    for row in [cut, deep_row] {
        assert_eq!(
            row.chars().count(),
            usize::from(COLUMNS) - 1,
            "the age column ends one column short of the edge: {row:?}"
        );
    }
    assert_eq!(run.exit, Some(0), "{opening:#?}");
}

/// On a terminal wider than the wire's title allows for, a picker row still fills: the title
/// carries as much of the first message as the row holds, cut two columns short of its facts.
#[test]
fn an_ultrawide_picker_row_holds_more_of_a_title_than_the_wire_carries() {
    const WIDE: u16 = 300;
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    let topic = ["wide"; 65].join(" ");
    let status = install
        .meka(&["--oneshot", "-p", &topic])
        .current_dir(install.work_dir())
        .status()
        .expect("a scripted run");
    assert!(status.success(), "seeding the long topic");
    let run = drive_run_sized(&install, &["-r"], &["\x1b", "/exit"], false, WIDE);
    let opening = &run.screens[0];
    let row = opening
        .iter()
        .find(|row| row.starts_with("  wide wide"))
        .unwrap_or_else(|| panic!("the long title is listed: {opening:#?}"));
    let (_, facts) = row
        .split_once("…  ")
        .unwrap_or_else(|| panic!("the title is cut two columns short of its facts: {row:?}"));
    assert!(
        !facts.starts_with(' ') && facts.ends_with("~/work  default  just now"),
        "the facts follow the title at once: {row:?}"
    );
    assert_eq!(
        row.chars().count(),
        usize::from(WIDE) - 1,
        "the facts end one column short of the edge: {row:?}"
    );
    assert_eq!(run.exit, Some(0), "{opening:#?}");
}

/// `/resume` from a running session switches the screen to the pick: the picked session's rows
/// take the place of the one left behind.
#[test]
fn resume_switches_the_running_screen_to_the_pick() {
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    let status = install
        .meka(&["--oneshot", "-p", "older topic"])
        .current_dir(install.work_dir())
        .status()
        .expect("a scripted run");
    assert!(status.success());
    let run = drive_run_with(&install, &[], &["hello", "/resume", "\r", "/exit"], false);
    let picker = &run.screens[2];
    assert!(
        picker.iter().any(|row| row.contains("older topic")),
        "the other session is offered: {picker:#?}"
    );
    let rows = final_screen(&run.screens);
    assert!(
        rows.iter().any(|row| row == "> older topic"),
        "the screen is on the picked session: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row == "> hello"),
        "the session left behind is off the screen: {rows:#?}"
    );
    assert_eq!(run.exit, Some(0), "{rows:#?}");
}

/// A command the host answers is refused while a turn runs, in a panel naming the command, and
/// does not run once the turn ends either: a `/compact` that ran after the turn it was typed
/// during would act on a conversation the user had not seen yet.
#[test]
fn a_host_command_typed_during_a_turn_is_refused() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, SLOW_ANSWER, &["hi", "/compact", "\x1b", "/exit"]);
    let refused = &screens[2];
    assert!(
        refused
            .iter()
            .any(|row| row == "  `/compact` cannot run while a turn is running."),
        "the refusal names the command: {refused:#?}"
    );
    let rows = final_screen(&screens);
    assert!(
        rows.iter().any(|row| row == "  done at last"),
        "the turn ended on its own: {rows:#?}"
    );
    assert!(
        !rows
            .iter()
            .any(|row| row.starts_with("• ") && row.to_lowercase().contains("compact")),
        "nothing was compacted once the turn ended: {rows:#?}"
    );
}

/// A new session opens on the banner: meka's mark beside the version, the profile with its model
/// and the directory, at the top under a blank row. The first prompt takes it down.
#[test]
fn a_new_session_opens_on_the_banner() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_TURNS, &["hi", "/exit"]);
    let opening = &screens[0];
    assert_eq!(
        opening[1],
        format!("  █▄ ▄█  meka v{}", env!("CARGO_PKG_VERSION")),
        "{opening:#?}"
    );
    assert_eq!(opening[2], "  █ ▀ █  default (mock-model)", "{opening:#?}");
    assert_eq!(opening[3], "  █   █  ~/work", "{opening:#?}");
    assert!(opening[0].is_empty(), "the blank row above: {opening:#?}");
    let rows = final_screen(&screens);
    assert!(
        rows.iter().any(|row| row == "> hi"),
        "the prompt is on the transcript: {rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains("  meka v")),
        "the banner goes once the conversation starts: {rows:#?}"
    );
}

/// Once the answer has pushed the prompt off the top, the prompt's first line sticks there as a
/// bar; a click on it brings the prompt to the top of the view.
#[test]
fn the_prompt_that_scrolled_off_sticks_at_the_top_and_a_click_brings_it_back() {
    let paragraphs: String = (1..=60).map(|n| format!("line {n}\\n\\n")).collect();
    let script = format!(
        r#"[[{{"type":"text","text":"{paragraphs}"}},{{"type":"message_end","stop_reason":"end_turn"}}]]"#
    );
    let install = screen_install("read", "", "");
    let click = "\x1b[<0;3;1M\x1b[<0;3;1m";
    let screens = run_screen(&install, &script, &["hi", click, "/exit"]);
    let flowing = &screens[1];
    assert_eq!(
        flowing[0], "> hi",
        "the prompt sticks at the top: {flowing:#?}"
    );
    assert!(
        flowing.iter().any(|row| row == "  line 60"),
        "while the view is at the end: {flowing:#?}"
    );
    // The jump lands on the prompt itself: nothing stands above the first prompt once the
    // banner has gone with it, so there is no bar to leave a row for and the view is at its top.
    let jumped = &screens[2];
    assert_eq!(
        jumped[0], "> hi",
        "the prompt itself is at the top now: {jumped:#?}"
    );
    assert_eq!(
        jumped[2], "  line 1",
        "with its answer under it: {jumped:#?}"
    );
    assert!(
        !jumped.iter().any(|row| row == "  line 60"),
        "and the end is no longer in view: {jumped:#?}"
    );
}

/// With two prompts off the top, a click on the bar brings the second to the top and the first
/// takes the bar; a second click brings the first.
#[test]
fn clicking_the_bar_walks_back_one_prompt_at_a_time() {
    let paragraphs: String = (1..=40).map(|n| format!("line {n}\\n\\n")).collect();
    let round = format!(
        r#"[{{"type":"text","text":"{paragraphs}"}},{{"type":"message_end","stop_reason":"end_turn"}}]"#
    );
    let script = format!("[{round},{round}]");
    let install = screen_install("read", "", "");
    let click = "\x1b[<0;3;1M\x1b[<0;3;1m";
    // Ctrl+L after each click forces another frame, which is where a view anchored one row
    // off would slide.
    let screens = run_screen(&install, &script, &[
        "one", "two", click, "\x0c", click, "\x0c", "/exit",
    ]);
    let at_end = &screens[2];
    assert_eq!(at_end[0], "> two", "the last prompt sticks: {at_end:#?}");
    let first_jump = &screens[4];
    assert_eq!(
        first_jump[0], "> one",
        "the prompt before takes the bar: {first_jump:#?}"
    );
    assert_eq!(
        first_jump[1], "> two",
        "with the last prompt under it: {first_jump:#?}"
    );
    let second_jump = &screens[6];
    assert_eq!(
        second_jump[0], "> one",
        "the first prompt is at the top, with nothing left to stick: {second_jump:#?}"
    );
    assert_eq!(second_jump[2], "  line 1", "{second_jump:#?}");
}

/// A drag over the transcript selects what it covers, and the release copies it to the
/// terminal's clipboard; a press and release in one place is still the click that toggles.
#[test]
fn a_drag_over_the_transcript_copies_what_it_covers() {
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    // SGR mouse: press at column 3 of row 3, drag to column 15, release there. The answer's row
    // is the third: the prompt, a blank, then `  First answer.`; the banner went with the prompt.
    let drag = "\x1b[<0;3;3M\x1b[<32;9;3M\x1b[<32;15;3M\x1b[<0;15;3m";
    let run = drive_run(&install, &["hi", drag, "/exit"], false);
    assert_eq!(
        run.clipboard.as_deref(),
        Some("First answer."),
        "{:#?}",
        run.screens
    );
    let lit = &run.screens[2];
    assert!(
        lit.last().is_some_and(|row| row.contains("copied 1 row")),
        "the status row says what was copied: {lit:#?}"
    );
}

/// `/log` shows what was logged at every level, debug lines included, where the transcript shows
/// only what `-v` allows.
#[test]
fn the_log_panel_keeps_what_the_transcript_does_not_show() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_TURNS, &["\x1b[Z", "/log", "\x1b", "/exit"]);
    let open = &screens[2];
    assert!(
        open.iter()
            .any(|row| row.contains("DEBUG") && row.contains("permission cycled to read")),
        "the debug line is in the pane: {open:#?}"
    );
    let rows = final_screen(&screens);
    assert!(
        !rows.iter().any(|row| row.contains("permission cycled")),
        "and not on the transcript: {rows:#?}"
    );
}

/// `/mcp` is a picker of the configured servers with their state, and Enter offers what can be
/// done with the picked one.
#[test]
fn the_mcp_picker_offers_the_server_s_actions() {
    let install = screen_install(
        "read",
        "",
        "\n[[mcp.servers]]\nname = \"demo\"\ntransport = \"stdio\"\ncommand = \"true\"\n",
    );
    let screens = run_screen(&install, TWO_TURNS, &["/mcp", "\r", "\x1b", "/exit"]);
    let picker = &screens[1];
    assert!(
        picker.iter().any(|row| row.starts_with("  demo  ")),
        "the server with its state: {picker:#?}"
    );
    let actions = &screens[2];
    assert!(
        actions.iter().any(|row| row == "demo")
            && actions.iter().any(|row| row.starts_with("  Reconnect"))
            && actions.iter().any(|row| row.starts_with("  Logout")),
        "the verbs under the server's name: {actions:#?}"
    );
}

/// Typing `/` opens the command list and Enter runs the pick, so `/he` and Enter is `/help`.
#[test]
fn the_command_list_opens_as_a_slash_is_typed_and_enter_runs_the_pick() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_TURNS, &["/he", "\x1b", "/exit"]);
    let shown = &screens[1];
    assert!(
        shown.iter().any(|row| row == "  Commands:"),
        "the help panel opened for the picked command: {shown:#?}"
    );
}

/// `/status` opens a panel in place of the composer, under no heading since the user typed the
/// command, and Esc closes it without a trace on the transcript: the command is not echoed and
/// its answer is not kept.
#[test]
fn a_status_panel_opens_under_the_transcript_and_leaves_no_row() {
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, TWO_TURNS, &["hi", "/status", "\x1b", "/exit"]);
    let open = &screens[2];
    assert_eq!(
        open.last().map(String::as_str),
        Some("  esc to close"),
        "the panel's footer takes the status row's place: {open:#?}"
    );
    assert!(
        !open.iter().any(|row| row == "/status"),
        "no heading repeats the command: {open:#?}"
    );
    assert!(
        open.iter().any(|row| row.starts_with("  Profile:")),
        "the status report is its body, at the panel's edge: {open:#?}"
    );
    let rows = final_screen(&screens);
    assert!(
        !rows
            .iter()
            .any(|row| row.contains("/status") || row.contains("Profile:")),
        "nothing of it stays on the transcript: {rows:#?}"
    );
    assert!(
        rows.last().is_some_and(|row| row.starts_with("  read ")),
        "the composer and the status row are back: {rows:#?}"
    );
}

/// A bare `/permission` is a picker over the enabled levels, the cursor on the current one; Down
/// and Enter move the session to the next, and the status row says so.
#[test]
fn the_permission_picker_sets_the_level() {
    let install = Install::new();
    install.write_config(
        "default_profile = \"default\"\n\n\
         [permissions]\ndefault = \"read\"\nenabled = [\"read\", \"workspace\"]\n\n\
         [accounts.default]\nbackend = \"openai-chat-completions\"\n\
         base_url = \"http://127.0.0.1:9/\"\n\n\
         [profiles.default]\naccount = \"default\"\nmodel = \"mock-model\"\n",
    );
    let screens = run_screen(&install, TWO_TURNS, &["/permission", "\x1b[B", "/exit"]);
    let picker = &screens[1];
    assert!(
        picker.iter().any(|row| row == "  read") && picker.iter().any(|row| row == "  workspace"),
        "the picker lists the enabled levels with the cursor on the current one: {picker:#?}"
    );
    let rows = final_screen(&screens);
    assert!(
        rows.iter()
            .any(|row| row == "• Permission level set to: workspace"),
        "the pick is one row on the transcript: {rows:#?}"
    );
    assert!(
        rows.last()
            .is_some_and(|row| row.starts_with("  workspace ")),
        "and the status row names the new level: {rows:#?}"
    );
}

/// `/skill` is a picker of the installed skills by name and description, and Enter runs the
/// picked one: the turn it starts is answered on the transcript.
#[test]
fn the_skill_picker_runs_the_picked_skill() {
    let install = screen_install("read", "", "");
    let skills = install.config_dir().join("skills");
    std::fs::create_dir_all(skills.join("rustbugs")).expect("skill dir");
    std::fs::write(
        skills.join("rustbugs").join("SKILL.md"),
        "---\nname: rustbugs\ndescription: Finds bugs in Rust\n---\nLook for bugs.\n",
    )
    .expect("seed the skill");
    let screens = run_screen(&install, TWO_TURNS, &["/skill", "\r", "/exit"]);
    let picker = &screens[1];
    assert!(
        picker
            .iter()
            .any(|row| row == "  rustbugs  Finds bugs in Rust"),
        "the picker lists the skill by name and description: {picker:#?}"
    );
    assert!(
        !picker.iter().any(|row| row == "/skill"),
        "no heading repeats the command: {picker:#?}"
    );
    let rows = final_screen(&screens);
    assert!(
        rows.iter().any(|row| row == "  First answer."),
        "the picked skill ran as a turn: {rows:#?}"
    );
}

/// Scrolling up shows the way back in the gap above the composer, and Esc takes it.
#[test]
fn scrolling_up_shows_the_pill_and_esc_returns_to_the_bottom() {
    let long: String = (1..=60)
        .map(|n| format!("line {n}\\n\\n"))
        .collect::<String>();
    let script = format!(
        r#"[[{{"type":"text","text":"{long}"}},{{"type":"message_end","stop_reason":"end_turn"}}]]"#
    );
    let install = screen_install("read", "", "");
    let screens = run_screen(&install, &script, &["hi", "\x1b[5~", "\x1b", "/exit"]);
    let scrolled = &screens[2];
    let pill = scrolled
        .iter()
        .position(|row| row.contains("↓ back to bottom"))
        .unwrap_or_else(|| panic!("the pill is shown while scrolled up: {scrolled:#?}"));
    assert!(
        scrolled[pill + 1].starts_with(">"),
        "it sits in the gap right above the composer: {scrolled:#?}"
    );
    assert!(
        !scrolled.iter().any(|row| row.contains("line 60")),
        "the bottom is out of view: {scrolled:#?}"
    );
    let rows = final_screen(&screens);
    assert!(
        !rows.iter().any(|row| row.contains("↓ back to bottom")),
        "Esc returned to the bottom and the pill is gone: {rows:#?}"
    );
    assert!(
        rows.iter().any(|row| row == "  line 60"),
        "the end of the answer is back in view: {rows:#?}"
    );
}

/// A resumed session opens on the end of its log and reads the rest as the view scrolls up: the
/// first page is on screen at once, the top of it brings the page above in without moving what
/// is shown, and Ctrl+End returns to the end with this run's rows after it.
#[test]
fn history_loads_as_the_view_scrolls_up() {
    let install = screen_install("read", "", "");
    let first = run_screen(&install, TWO_TURNS, &["first", "/exit"]);
    assert!(
        final_screen(&first)
            .iter()
            .any(|row| row == "  First answer."),
        "the session has a turn to resume"
    );
    // Rows past one page, written as the store would have written them, so the resume has a
    // page above the one it opens on.
    let connection = rusqlite::Connection::open(install.database()).expect("open the store");
    let session_id: String = connection
        .query_row("SELECT id FROM sessions LIMIT 1", [], |row| row.get(0))
        .expect("the session");
    for n in 1..=250 {
        connection
            .execute(
                "INSERT INTO messages (session_id, kind, content, created_at) \
                 VALUES (?1, 'assistant', ?2, ?3)",
                rusqlite::params![
                    session_id,
                    format!("seeded {n}"),
                    chrono::Utc::now().to_rfc3339()
                ],
            )
            .expect("a seeded row");
    }
    drop(connection);
    let resumed = run_screen(&install, TWO_TURNS, &[
        "\x1b[1;5H",
        "\x1b[1;5H",
        "\x1b[1;5F",
        "/exit",
    ]);
    let opened = &resumed[0];
    assert!(
        opened.iter().any(|row| row == "  seeded 250")
            && opened
                .iter()
                .any(|row| row.starts_with("• Resuming session ")),
        "the end of the log, then this run's banner: {opened:#?}"
    );
    assert!(
        !opened.iter().any(|row| row == "> first"),
        "the first turn is a page away: {opened:#?}"
    );
    let top = &resumed[1];
    assert_eq!(
        top.first().map(String::as_str),
        Some("> first"),
        "Ctrl+Home reaches the top of the page the screen opened on, the page read above it \
         puts the first prompt above the view, and the bar shows it: {top:#?}"
    );
    assert_eq!(
        top.get(2).map(String::as_str),
        Some("  seeded 52"),
        "and the row the view opened on holds its place under the bar: {top:#?}"
    );
    let older = &resumed[2];
    assert_eq!(
        older.first().map(String::as_str),
        Some("> first"),
        "Ctrl+Home again reaches the first turn: {older:#?}"
    );
    let rows = final_screen(&resumed);
    assert!(
        rows.iter().any(|row| row == "  seeded 250")
            && !rows.iter().any(|row| row.contains("↓ back to bottom")),
        "Ctrl+End is the end of the log again: {rows:#?}"
    );
}

/// A pipe on stdin or stdout is refused with the door a script should use.
#[test]
fn a_run_without_a_terminal_is_refused_and_names_oneshot() {
    let install = screen_install("read", "", "");
    install.write_script(TWO_TURNS);
    let output = install
        .meka(&[])
        .current_dir(install.work_dir())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run meka");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("needs a terminal") && stderr.contains("--oneshot"),
        "{stderr}"
    );
}
