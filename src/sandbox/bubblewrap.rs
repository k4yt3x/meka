//! Bubblewrap: finding a binary that can be trusted to confine, and the smoke test that proves user
//! namespaces work here.

use super::*;

pub(super) fn probe_bubblewrap() -> BackendProbe {
    let Some(bwrap_path) = bwrap_on_path() else {
        return BackendProbe::Missing {
            reason: "bwrap not found on PATH".to_string(),
        };
    };

    match smoke_test_bwrap(&bwrap_path, BWRAP_PROBE_TIMEOUT) {
        SmokeResult::Success => BackendProbe::Ok(SandboxCapability::Bubblewrap { bwrap_path }),
        SmokeResult::UserNamespaceDenied { stderr } => BackendProbe::UserNamespaceDenied { stderr },
        SmokeResult::OtherFailure { reason } => BackendProbe::Missing { reason },
    }
}
pub(super) const BWRAP_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);
pub(super) const BWRAP_PROBE_POLL: std::time::Duration = std::time::Duration::from_millis(50);
pub(super) const BWRAP_STDERR_LIMIT: usize = 64 * crate::text::KIB;
/// Stderr substrings that indicate the kernel refused the user namespace request rather than some
/// other transient failure. Mirrors the fingerprint list Codex uses in
/// `codex-rs/sandboxing/src/bwrap.rs`.
pub(super) const USER_NAMESPACE_FAILURE_MARKERS: &[&str] = &[
    "loopback: Failed RTM_NEWADDR",
    "loopback: Failed RTM_NEWLINK",
    "setting up uid map: Permission denied",
    "No permissions to create a new namespace",
];
pub(super) enum SmokeResult {
    Success,
    UserNamespaceDenied { stderr: String },
    OtherFailure { reason: String },
}
/// Whether `path` is a directory only root can write to.
///
/// The test that decides whether a `bwrap` found there can be trusted. Group- and other-writable
/// are both disqualifying: a directory writable by any group the user is in is writable by the
/// user.
pub(super) fn only_root_can_write(path: &std::path::Path) -> bool {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    metadata.uid() == 0 && metadata.permissions().mode() & 0o022 == 0
}
/// A regular file with at least one execute bit set.
///
/// Named so the rule can be exercised on a file a test can actually create. Inline in
/// [`bwrap_on_path`] it was reachable only through `$PATH`, and both halves of the conjunction were
/// mutable without any test noticing.
pub(super) fn is_executable_file(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;

    metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
}
/// Whether a `bwrap` at `candidate`, found in `directory`, can be trusted to confine anything.
///
/// **Both** paths have to be root-only-writable. A root-owned binary in a directory the user can
/// write is one `mv` away from being the user's, so checking the binary alone is no check at all.
///
/// Split out from [`bwrap_on_path`] because the conjunction *is* the security property and cannot
/// be exercised through that function: the interesting inputs are mixed, one trusted path and one
/// not, and a test cannot create a root-owned file to supply them. As a named predicate it takes
/// two paths that exist on any host, `/usr/bin` and a temp dir, so the mixed cases become ordinary
/// arguments.
pub(super) fn trusted_to_confine(candidate: &std::path::Path, directory: &std::path::Path) -> bool {
    only_root_can_write(candidate) && only_root_can_write(directory)
}
/// Look up `bwrap` on `$PATH`, accepting only a binary that the user cannot replace.
///
/// The plain `$PATH` walk was an unconfinement primitive. `$PATH` on an ordinary desktop holds
/// several directories the user can write -- `~/.local/bin`, a cargo or go bin dir, a toolchain
/// shim dir -- and every one of them precedes `/usr/bin`. A `bwrap` planted in any of them is
/// executed verbatim by the spawn path, so a six-line shell script that `exec`s its final argument
/// turns every `read` and `workspace` shell command into an unconfined one. Nothing noticed:
/// [`smoke_test_bwrap`] runs `bwrap <flags> /bin/true`, which a shim satisfies by construction, so
/// the probe reported `Ok` and both the startup warning and the lazy hard-error gate stayed quiet.
/// Demonstrated end to end -- a confined command wrote outside every root and the file landed on
/// the host, where real `bwrap` refuses the same argv.
///
/// It is a persistence primitive rather than a one-shot: one turn at `unrestricted`, one approved
/// `ask` command, or any post-install hook buys unconfined shells in every later session,
/// including the sessions a user opens at `read` precisely because they do not trust the turn.
///
/// `macos_impl` has hardcoded `/usr/bin/sandbox-exec` since it was written, with a comment naming
/// this exact attack. Linux is the backend meka auto-prefers, and it was the one searching `$PATH`.
///
/// Checking ownership rather than hardcoding a list keeps the distributions that put it elsewhere
/// working -- NixOS serves it out of a root-owned `/nix/store` path, which passes -- while refusing
/// anything under a directory the user can write. A rejected candidate is `warn!`ed rather than
/// skipped silently, because "bubblewrap is installed but meka fell back to Landlock" is otherwise
/// indistinguishable from "bubblewrap is not installed".
pub(super) fn bwrap_on_path() -> Option<std::path::PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join("bwrap");
        let Ok(metadata) = std::fs::metadata(&candidate) else {
            continue;
        };
        if !is_executable_file(&metadata) {
            continue;
        }
        if trusted_to_confine(&candidate, &dir) {
            return Some(candidate);
        }
        let path = candidate.display();
        tracing::warn!(
            "ignoring {path} for sandboxing: it or its directory is writable by someone other \
             than root, so it cannot be trusted to confine anything"
        );
    }
    None
}
/// Run a `bwrap … /bin/true` smoke test with a short timeout.
///
/// The flag set mirrors the production-path argv in `src/tools/shell.rs` so a host that succeeds
/// here also succeeds at runtime; without it, a kernel that quietly rejects (say)
/// `--unshare-cgroup-try` or `--die-with-parent` would pass the probe and blow past the lazy
/// hard-error gate the first time `execute_command` ran. `--unshare-net` is added on top so the
/// probe stays self-contained (no outbound DNS / network calls), even though production keeps the
/// host network namespace.
pub(super) fn smoke_test_bwrap(
    bwrap_path: &std::path::Path,
    timeout: std::time::Duration,
) -> SmokeResult {
    use std::{io::Read, os::fd::AsRawFd};

    let mut command = std::process::Command::new(bwrap_path);
    command
        .args([
            "--new-session",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
            "--unshare-cgroup-try",
            "--unshare-net",
            "/bin/true",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());

    // Arm the parent-death signal here rather than leaving it to `--die-with-parent`.
    //
    // Measured, because the obvious reading is wrong in both directions. bwrap's own flag *does*
    // handle a `meka` that is SIGKILLed outright: kill the parent a second after spawning and the
    // sandbox goes with it. What it cannot cover is the window before bwrap reaches its own
    // `prctl`, several syscalls into a startup that includes a user-namespace handshake with its
    // child -- and a `bwrap` found parked for eleven days on this machine was blocked in exactly
    // that handshake, `read()`ing its sync eventfd, reparented to init with the flag in its argv.
    // Arming in `pre_exec` covers the child from before `execve`, which is the earliest point that
    // exists.
    //
    // The `getppid` re-read is what makes it a guarantee rather than a smaller window: the signal
    // only fires on a death that happens *after* the `prctl`, so a parent that died between fork
    // and this line would never deliver it. Reading the parent back afterwards catches that and
    // exits instead.
    //
    // `PR_SET_PDEATHSIG` tracks the parent *thread*, not the process, which is safe here only
    // because the caller blocks in the poll loop below for the child's whole life: the thread that
    // spawned it cannot retire while the child still matters. Do not lift this onto a spawn whose
    // child outlives the call.
    let parent = std::process::id() as libc::pid_t;
    // SAFETY: the closure runs after `fork` in the child and calls only async-signal-safe
    // functions (`prctl`, `getppid`, `_exit`), allocating nothing.
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(&mut command, move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != parent {
                libc::_exit(0);
            }
            Ok(())
        });
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return SmokeResult::OtherFailure {
                reason: format!("failed to spawn bwrap for smoke test: {error}"),
            };
        }
    };

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Drain stderr non-blocking. The grandchild is gone so nothing else will write; any
                // data already buffered is all we'll get.
                let stderr = match child.stderr.take() {
                    Some(mut handle) => {
                        let fd = handle.as_raw_fd();
                        // SAFETY: fcntl with F_GETFL/F_SETFL on a valid open file descriptor;
                        // failure is fine and just means we'll attempt a regular read that may
                        // block briefly on a closed pipe.
                        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
                        if flags >= 0 {
                            unsafe {
                                libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
                            }
                        }
                        let mut bytes = Vec::new();
                        let mut take = handle.by_ref().take(BWRAP_STDERR_LIMIT as u64);
                        if let Err(error) = take.read_to_end(&mut bytes)
                            && error.kind() != std::io::ErrorKind::WouldBlock
                        {
                            tracing::debug!("bwrap smoke test: stderr read failed: {error}");
                        }
                        String::from_utf8_lossy(&bytes).into_owned()
                    }
                    None => String::new(),
                };
                if status.success() {
                    return SmokeResult::Success;
                }
                if USER_NAMESPACE_FAILURE_MARKERS
                    .iter()
                    .any(|marker| stderr.contains(marker))
                {
                    return SmokeResult::UserNamespaceDenied { stderr };
                }
                let truncated_stderr = stderr.lines().next().unwrap_or("").trim().to_string();
                let reason = if truncated_stderr.is_empty() {
                    format!("bwrap smoke test failed (exit {:?})", status.code())
                } else {
                    format!(
                        "bwrap smoke test failed (exit {:?}): {}",
                        status.code(),
                        truncated_stderr
                    )
                };
                return SmokeResult::OtherFailure { reason };
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    reap_smoke_test_child(&mut child);
                    return SmokeResult::OtherFailure {
                        reason: format!(
                            "bwrap smoke test exceeded {}ms timeout",
                            timeout.as_millis()
                        ),
                    };
                }
                std::thread::sleep(BWRAP_PROBE_POLL);
            }
            Err(error) => {
                reap_smoke_test_child(&mut child);
                return SmokeResult::OtherFailure {
                    reason: format!("bwrap smoke test wait failed: {error}"),
                };
            }
        }
    }
}
/// Best-effort cleanup of a stuck smoke-test child: kill it and reap its status so we don't leave a
/// zombie. Errors are logged at debug level only; by this point the smoke test has already failed
/// and the caller is about to return a higher-priority error reason.
pub(super) fn reap_smoke_test_child(child: &mut std::process::Child) {
    if let Err(error) = child.kill() {
        tracing::debug!("bwrap smoke test: failed to kill stuck child: {error}");
    }
    if let Err(error) = child.wait() {
        tracing::debug!("bwrap smoke test: failed to reap child: {error}");
    }
}
