//! Pieces every integration harness needs, kept in one place so no test crate carries its own copy.
//!
//! Integration tests are separate crates, so this is included per file with `#[path = "support.rs"]
//! mod support;` rather than through a `mod.rs`, which the project does not use.

#![allow(dead_code, reason = "not every test crate uses every helper")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "the tests this serves panic on failure by design"
)]

use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command},
    sync::mpsc,
    time::{Duration, Instant},
};

/// The `meka` binary under test.
pub(crate) fn meka() -> Command {
    Command::new(env!("CARGO_BIN_EXE_meka"))
}

/// An isolated config and data directory pair under one tempdir, plus the environment that points
/// a `meka` process at them and at the scripted mock provider.
///
/// `MEKA_CONFIG_DIR` and `MEKA_DATA_DIR` are the overrides that work on every platform
/// (`dirs::config_dir()` and `dirs::data_dir()` ignore `XDG_*` on macOS and Windows, and
/// `dirs::home_dir()` on Windows never reads the environment at all); `HOME` and the `XDG_*` pair
/// are set as well so a code path that consults them on Linux lands inside the tempdir too, never
/// in the developer's real profile. A test that needs a path outside meka's own directories names
/// one under [`Self::root`] rather than relying on `HOME`.
pub(crate) struct Install {
    temp: tempfile::TempDir,
}

impl Install {
    pub(crate) fn new() -> Self {
        let temp = tempfile::tempdir().expect("tempdir");
        let install = Self { temp };
        std::fs::create_dir_all(install.config_dir()).expect("create config dir");
        std::fs::create_dir_all(install.data_dir()).expect("create data dir");
        std::fs::create_dir_all(install.work_dir()).expect("create work dir");
        install
    }

    pub(crate) fn root(&self) -> &Path {
        self.temp.path()
    }

    pub(crate) fn config_dir(&self) -> PathBuf {
        self.temp.path().join("meka")
    }

    pub(crate) fn data_dir(&self) -> PathBuf {
        self.temp.path().join("data").join("meka")
    }

    pub(crate) fn database(&self) -> PathBuf {
        self.data_dir().join("meka.db")
    }

    /// A project directory beside meka's own, for a session to work in. A path *under* the config
    /// dir is inside meka's own directory, which the write fence refuses below `unrestricted`.
    pub(crate) fn work_dir(&self) -> PathBuf {
        self.temp.path().join("work")
    }

    pub(crate) fn script_path(&self) -> PathBuf {
        self.temp.path().join("script.json")
    }

    pub(crate) fn write_config(&self, toml: &str) {
        std::fs::write(self.config_dir().join("config.toml"), toml).expect("write config.toml");
    }

    /// Point the mock provider at `rounds`, a JSON array of rounds given either as a
    /// `serde_json::Value` or as source text. One script serves every process spawned from this
    /// install; each loads the file at startup and gets its own queue.
    pub(crate) fn write_script(&self, rounds: impl std::fmt::Display) -> PathBuf {
        let path = self.script_path();
        std::fs::write(&path, rounds.to_string()).expect("write script");
        path
    }

    /// Apply the isolation environment to `command`. The script variable is set only once a
    /// script has been written, so a command spawned without one sees the mock provider with an
    /// empty queue rather than a path to a file that does not exist.
    ///
    /// The mock stands in for every profile. A test that needs the real provider path (how a
    /// missing credential fails, which endpoint a process sends to) sets `MEKA_MOCK_PROVIDER` to
    /// `0` after this, which is the one value the switch does not read as on.
    pub(crate) fn env<'a>(&self, command: &'a mut Command) -> &'a mut Command {
        command
            .env("MEKA_CONFIG_DIR", self.config_dir())
            .env("MEKA_DATA_DIR", self.data_dir())
            .env("HOME", self.temp.path())
            .env("XDG_CONFIG_HOME", self.temp.path())
            .env("XDG_DATA_HOME", self.temp.path().join("data"))
            .env("MEKA_MOCK_PROVIDER", "1");
        if self.script_path().exists() {
            command.env("MEKA_MOCK_PROVIDER_SCRIPT", self.script_path());
        }
        command
    }

    /// A `meka` command pointed at this install.
    pub(crate) fn meka(&self, args: &[&str]) -> Command {
        let mut command = meka();
        command.args(args);
        self.env(&mut command);
        command
    }
}

impl Default for Install {
    fn default() -> Self {
        Self::new()
    }
}

/// Wait until `check` holds, or give up. Polling rather than sleeping a fixed span: these tests
/// wait on another process reaching a state, and the only constant that is reliably long enough on
/// a loaded machine is one that makes the suite slow for everybody.
pub(crate) fn wait_until(what: &str, timeout: Duration, mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if check() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out after {timeout:?} waiting for {what}");
}

/// Bind to an OS-assigned port and release it, so a server spawned next can claim it. A parallel
/// test can grab the port in between; callers that spawn a server retry on a bind failure.
pub(crate) fn ephemeral_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

/// Read one of a child's pipes to EOF on its own thread, so it cannot fill and block the process
/// writing to it. The thread hands back everything it read, for a failure message.
pub(crate) fn drain(pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(pipe);
        let mut accumulated = String::new();
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            accumulated.push_str(&line);
            line.clear();
        }
        accumulated
    })
}

/// Lines from a child's pipe, each read with a deadline.
///
/// A `BufReader` over a pipe blocks in `read_line` for as long as the child is silent, and
/// `cargo test` has no per-test timeout, so a child that deadlocks hangs the whole suite instead
/// of failing one test. Reading on a thread and receiving with a timeout is what turns that hang
/// into a failure that names the request the child never answered.
pub(crate) struct TimedLines {
    receiver: mpsc::Receiver<String>,
    /// Set once the reader thread hit EOF or an error, so a later read returns nothing at once
    /// rather than waiting out its deadline on a pipe that will never speak again.
    closed: bool,
}

impl TimedLines {
    /// The budget a bare [`Self::read_line`] gets when the caller names none. Long enough for the
    /// slowest legitimate wait in the suite, short enough that a hang is reported the same minute.
    pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

    pub(crate) fn spawn(read: impl Read + Send + 'static) -> Self {
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(read);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        Self {
            receiver,
            closed: false,
        }
    }

    /// The next line, or `None` when the deadline passes or the pipe is closed.
    pub(crate) fn next_line(&mut self, deadline: Instant) -> Option<String> {
        if self.closed {
            return None;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        match self.receiver.recv_timeout(remaining) {
            Ok(line) => Some(line),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.closed = true;
                None
            }
        }
    }

    /// `BufRead::read_line`'s shape, for the loops that were written against a `BufReader`:
    /// `Ok(0)` at EOF, and a `TimedOut` error when the child says nothing for
    /// [`Self::DEFAULT_TIMEOUT`]. Loops that treat any error as EOF (`unwrap_or(0)`, `Err(_) =>
    /// break`) therefore end on a hang instead of blocking, and the test fails on whatever it
    /// asserts next.
    pub(crate) fn read_line(&mut self, buffer: &mut String) -> std::io::Result<usize> {
        match self.next_line(Instant::now() + Self::DEFAULT_TIMEOUT) {
            Some(line) => {
                let read = line.len();
                buffer.push_str(&line);
                Ok(read)
            }
            None if self.closed => Ok(0),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("no line from the child within {:?}", Self::DEFAULT_TIMEOUT),
            )),
        }
    }
}

/// One HTTP request as a local listener received it.
pub(crate) struct CapturedRequest {
    pub(crate) method: String,
    pub(crate) path: String,
    /// Header names lower-cased.
    pub(crate) headers: HashMap<String, String>,
    pub(crate) body: String,
}

/// A local HTTP/1.1 listener that answers every request with whatever `handler` returns, byte
/// for byte, starting at the status line. Each connection is served on its own thread, so a
/// handler that deliberately stalls (to hold a client in a request while a test signals it) does
/// not block the next connection.
///
/// Enough of HTTP to stand in for a webhook receiver or an MCP endpoint's auth challenge: the
/// request line, headers, and a `Content-Length` body. Chunked bodies and keep-alive are not
/// handled; each connection carries one request.
pub(crate) fn spawn_http_listener<F>(handler: F) -> u16
where
    F: Fn(&CapturedRequest) -> String + Send + Sync + 'static,
{
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let port = listener.local_addr().expect("addr").port();
    let handler = std::sync::Arc::new(handler);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let handler = std::sync::Arc::clone(&handler);
            std::thread::spawn(move || {
                let Some(request) = read_request(&mut stream) else {
                    return;
                };
                let response = handler(&request);
                // A client that hung up before reading its answer is not this listener's
                // failure to report; the test asserts on what the client did.
                stream
                    .write_all(response.as_bytes())
                    .and_then(|()| stream.flush())
                    .ok();
            });
        }
    });
    port
}

fn read_request(stream: &mut std::net::TcpStream) -> Option<CapturedRequest> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).ok()? == 0 {
        return None;
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let mut headers = HashMap::new();
    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).ok()? == 0 {
            break;
        }
        if header.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim().to_string();
            if name == "content-length" {
                content_length = value.parse().unwrap_or(0);
            }
            headers.insert(name, value);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }
    Some(CapturedRequest {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).to_string(),
    })
}

/// How a server start ended.
pub(crate) enum Started {
    /// The child announced the address it bound; the handle collects the rest of its stderr.
    Ready(std::thread::JoinHandle<String>),
    /// The child exited before announcing; its exit status and stderr.
    Exited(std::process::ExitStatus, String),
    /// The child is still running but never announced inside `timeout`.
    TimedOut(std::thread::JoinHandle<String>),
}

/// Wait for `meka serve` to announce that it is listening on `bind`.
///
/// The announcement is the child's own `listening on <address>` line, matched with the exact
/// address, because nothing over HTTP identifies *this* child: a parallel test can win the race
/// for the port between `ephemeral_port` releasing it and this child binding it, and a liveness
/// probe then answers from the other test's server with the other test's config. Watching the
/// child as well means an exit is reported the moment it happens, with the logs that explain it,
/// instead of after the full timeout.
pub(crate) fn wait_for_serve(
    bind: &str,
    child: &mut Child,
    stderr: std::process::ChildStderr,
    timeout: Duration,
) -> Started {
    let needle = format!("listening on {bind}");
    let (ready_sender, ready_receiver) = mpsc::channel::<()>();
    let logs = std::thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut accumulated = String::new();
        let mut line = String::new();
        let mut announced = false;
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            accumulated.push_str(&line);
            if !announced && line.contains(&needle) {
                announced = true;
                // The waiter may have given up already; the logs are still worth keeping.
                ready_sender.send(()).ok();
            }
            line.clear();
        }
        accumulated
    });
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match ready_receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(()) => return Started::Ready(logs),
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {}
        }
        if let Ok(Some(status)) = child.try_wait() {
            let text = logs.join().unwrap_or_default();
            return Started::Exited(status, text);
        }
    }
    Started::TimedOut(logs)
}
