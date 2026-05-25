//! `execute_command` tool. Spawns a shell process, optionally constrained by
//! the platform sandbox (Landlock/sandbox-exec) when permissions are
//! read-only, and streams stdout/stderr back to the agent.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{Tool, ToolOutput, util::require_str};
use crate::{
    error::{AgshError, Result},
    permission::Permission,
    provider::ToolDefinition,
};

/// Default `timeout_ms` applied when the caller doesn't pass one.
/// Single source of truth for both the parameter unwrap and the
/// description shown to the agent.
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

pub(super) struct ExecuteCommandTool {
    pub sandbox_capability: crate::sandbox::SandboxCapability,
    /// Backend chosen in config (or auto-resolved). Read only by the
    /// Linux hard-error message in [`Tool::execute`]; on macOS /
    /// Windows the field is populated but unused, so suppress the
    /// "never read" lint there without hiding regressions on Linux.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub sandbox_backend: crate::config::SandboxBackend,
    /// Probe outcome for [`Self::sandbox_backend`]. Drives the
    /// hard-error path in read mode when the backend isn't usable
    /// (bwrap missing, user namespaces denied, etc.). When `Ok(_)`,
    /// [`Self::sandbox_capability`] mirrors the inner capability and
    /// the spawn path runs normally.
    pub backend_probe: crate::sandbox::BackendProbe,
    pub shared_permission: crate::permission::SharedPermission,
    pub sandbox_enabled: bool,
    pub cwd: crate::agent::SharedCwd,
    /// Non-`Read` modes delegate to the editor's hosted terminal
    /// (ACP `terminal/*`) when the client advertises support.
    /// `Read` mode bypasses delegation to keep the local
    /// Landlock / bwrap / sandbox-exec / Low-Integrity jail.
    pub frontend: Arc<dyn crate::frontend::Frontend>,
}

#[async_trait]
impl Tool for ExecuteCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "execute_command".to_string(),
            description: "Execute a shell command and return its output. On Unix the \
                command runs via `sh -c <command>` — POSIX `$VAR` expansion applies; \
                quote with single quotes or `\\$` to pass a literal `$`. On Windows \
                the command runs via `powershell.exe -Command <command>` — use \
                PowerShell syntax directly (e.g. `$var = ...`, `$env:PATH`); do NOT \
                wrap with another `powershell -Command` or the outer PowerShell will \
                expand your inner `$var` references to empty strings. In read mode \
                the command runs in a read-only sandbox where filesystem writes are \
                blocked. Multiple independent execute_command calls in one assistant \
                message run in parallel; use this for read-only commands and \
                serialize anything that mutates shared state (files, git, packages)."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": format!(
                            "Timeout in milliseconds. Defaults to {} ({} seconds).",
                            DEFAULT_TIMEOUT_MS,
                            DEFAULT_TIMEOUT_MS / 1000,
                        )
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["command"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        if self.sandbox_enabled
            && !matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::Unavailable
            )
        {
            Permission::Read
        } else {
            Permission::Write
        }
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let command = require_str(&input, "command", "execute_command")?;
        let timeout_ms = input["timeout_ms"].as_u64().unwrap_or(DEFAULT_TIMEOUT_MS);
        let permission = self.shared_permission.get();
        let sandboxed = self.sandbox_enabled && permission != Permission::Write;

        if sandboxed {
            // Configured backend isn't usable on this host. Hard-error
            // with the specific reason so the model can surface it via
            // `render::render_error` rather than treat the failure as
            // a tool result it could try to recover from.
            if let Some(reason) = crate::sandbox::backend_unavailable_reason(&self.backend_probe) {
                // `sandbox_backend` is Linux-only; on other platforms
                // there's nothing to reconfigure — the only escape
                // hatch is write mode.
                #[cfg(target_os = "linux")]
                let message = format!(
                    "configured sandbox backend ({}) is unavailable: {}. \
                     Switch to write mode (Shift+Tab) to run shell commands \
                     without a sandbox, or update [shell].sandbox_backend in \
                     your config.",
                    self.sandbox_backend, reason
                );
                #[cfg(not(target_os = "linux"))]
                let message = format!(
                    "sandbox is unavailable: {}. Switch to write mode \
                     (Shift+Tab) to run shell commands without a sandbox.",
                    reason
                );
                return Err(AgshError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message,
                });
            }
        }

        // Delegate to the editor's hosted terminal when offered.
        // `Read` mode skips delegation to preserve the local sandbox
        // jail — the editor's terminal has no equivalent.
        if !matches!(permission, Permission::Read) {
            let (program, args) = shell_invocation(&command);
            let spec = crate::frontend::DelegatedExecSpec {
                command: program,
                args,
                env: Vec::new(),
                cwd: Some(crate::agent::cwd_snapshot(&self.cwd)),
                timeout: Some(std::time::Duration::from_millis(timeout_ms)),
                output_byte_limit: Some(2_000_000),
                cancellation: cancellation.clone(),
            };
            if let Some(result) = self.frontend.delegate_execute(spec).await {
                let output = result.map_err(|error| AgshError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message: format!("delegated execute failed: {}", error),
                })?;
                return Ok(assemble_delegated_output(output));
            }
        }

        // Windows + sandboxed: spawn directly via CreateProcessAsUserW with a
        // Low-integrity token. This path can't go through tokio::process
        // because the stdlib gives no hook for injecting a custom token.
        #[cfg(windows)]
        if sandboxed
            && matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::LowIntegrity
            )
        {
            return run_windows_low_integrity(&command, timeout_ms, cancellation).await;
        }

        #[cfg(windows)]
        let mut command_builder = {
            // Wrap with the UTF-8 output prelude so pipe output matches
            // what the sandboxed path produces — both on Rust's side this
            // is decoded as UTF-8. Without the wrap, PowerShell 5.1
            // defaults to the legacy console code page and mangles
            // non-ASCII characters into `?`.
            let wrapped = crate::sandbox::wrap_command_with_utf8_output(&command);
            let mut cmd = tokio::process::Command::new("powershell.exe");
            cmd.arg("-NoProfile")
                .arg("-NonInteractive")
                .arg("-Command")
                .arg(&wrapped);
            cmd
        };

        #[cfg(target_os = "macos")]
        let mut command_builder = if sandboxed
            && matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::SandboxExec
            ) {
            let mut cmd = tokio::process::Command::new(crate::sandbox::SANDBOX_EXEC_PATH);
            cmd.arg("-p")
                .arg(crate::sandbox::SANDBOX_PROFILE_READONLY)
                .arg("sh")
                .arg("-c")
                .arg(&command);
            cmd
        } else {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(&command);
            cmd
        };

        #[cfg(target_os = "linux")]
        let mut command_builder = if sandboxed
            && let crate::sandbox::SandboxCapability::Bubblewrap { bwrap_path } =
                &self.sandbox_capability
        {
            // Bubblewrap path: `--ro-bind /` enforces "no writes",
            // `--unshare-*` cuts off PID / user / UTS / IPC views,
            // tmpfs masks over `/run`, `/tmp`, `/var/tmp`, and
            // `$XDG_RUNTIME_DIR` make the dbus and systemd-user
            // sockets unreachable so the agent can't `dbus-send`
            // state-changing methods. `--unshare-net` is intentionally
            // absent — network must stay open for `curl | pdftotext`
            // and similar pipelines.
            let mut cmd = tokio::process::Command::new(bwrap_path);
            cmd.args([
                "--new-session",
                "--die-with-parent",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/run",
                "--tmpfs",
                "/var/tmp",
                "--unshare-user",
                "--unshare-pid",
                "--unshare-uts",
                "--unshare-ipc",
                "--unshare-cgroup-try",
            ]);
            if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR")
                && std::path::Path::new(&xdg).is_absolute()
            {
                cmd.arg("--tmpfs").arg(&xdg);
            }
            cmd.arg("--").arg("sh").arg("-c").arg(&command);
            cmd
        } else {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(&command);
            cmd
        };

        // Unix: place the child in its own session/process group via
        // `setsid` so timeouts and cancellation can kill the whole tree
        // (including backgrounded grandchildren such as `(sleep 3600 &)`)
        // via `kill(-pgid, …)`. On Linux the Landlock setup runs in the
        // same closure — `pre_exec` overwrites rather than chains, so we
        // fold both steps into one. Landlock is applied ONLY for the
        // Landlock capability; under Bubblewrap, the `--ro-bind /`
        // mount layer already enforces "no writes" and layering both
        // is fragile to test across kernels.
        #[cfg(unix)]
        {
            #[cfg(target_os = "linux")]
            let landlock_abi: Option<i32> = if sandboxed {
                if let crate::sandbox::SandboxCapability::Landlock { abi_version } =
                    self.sandbox_capability
                {
                    Some(abi_version)
                } else {
                    None
                }
            } else {
                None
            };

            unsafe {
                command_builder.pre_exec(move || {
                    // SAFETY: `setsid(2)` is async-signal-safe and has no
                    // preconditions beyond "the caller isn't already a
                    // process group leader" — which is guaranteed for a
                    // freshly forked child process.
                    if libc::setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    #[cfg(target_os = "linux")]
                    if let Some(abi) = landlock_abi {
                        crate::sandbox::apply_landlock_readonly(abi)
                            .map_err(std::io::Error::from_raw_os_error)?;
                    }
                    #[cfg(not(target_os = "linux"))]
                    let _ = (); // landlock_abi unused on non-Linux Unix
                    Ok(())
                });
            }
        }

        // Scrub env before spawn so secrets in the parent process
        // (`ANTHROPIC_API_KEY`, `AWS_*`, `GITHUB_TOKEN`, …) can't ride
        // along into the read-mode child. Sandboxes block writes/IPC
        // but leave the network open, so leaked env is a live exfil
        // vector under prompt injection. Write mode keeps the parent
        // env (trusted-operation path). The Windows sandboxed branch
        // applies the same scrub inside `spawn_low_integrity_command`.
        #[cfg(unix)]
        if sandboxed {
            command_builder.env_clear();
            command_builder.envs(crate::sandbox::sandbox_child_env());
        }

        // Resolve commands against the agent's per-session cwd, not
        // the process cwd. `/cd` mutates the agent's cwd; this is how
        // it actually reaches the child.
        command_builder.current_dir(crate::agent::cwd_snapshot(&self.cwd));

        let mut child = command_builder
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "execute_command".to_string(),
                message: format!("failed to spawn command: {}", error),
            })?;

        let timeout_duration = std::time::Duration::from_millis(timeout_ms);

        // Drain stdout/stderr on dedicated tasks that start *before* the
        // wait. `tokio::process::Child::wait()` does not read the pipes;
        // a child writing past the OS pipe buffer (~64 KiB) would block in
        // `write()`, `wait()` would never return, and the call would
        // spuriously hit the timeout below. After the child's process
        // group exits the pipe write ends close and the drains hit EOF.
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdout_task = tokio::spawn(async move { read_to_string_best_effort(stdout).await });
        let stderr_task = tokio::spawn(async move { read_to_string_best_effort(stderr).await });

        // wait_with_output() consumes the child, so use wait() + manual
        // stdout/stderr reading instead to allow kill on cancellation.
        tokio::select! {
            _ = cancellation.cancelled() => {
                kill_child_tree(&mut child).await;
                stdout_task.abort();
                stderr_task.abort();
                Err(AgshError::Interrupted)
            }
            _ = tokio::time::sleep(timeout_duration) => {
                kill_child_tree(&mut child).await;
                stdout_task.abort();
                stderr_task.abort();
                Ok(ToolOutput::text(
                    format!("Command timed out after {}ms", timeout_ms),
                    true,
                ))
            }
            status = child.wait() => {
                let status = status.map_err(|error| AgshError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message: format!("failed to wait for command: {}", error),
                })?;

                let exit_code = status.code().unwrap_or(-1);
                // A backgrounded grandchild can keep the pipe open past the
                // direct child's exit; cap the drain so the tool call can't
                // hang, attaching a truncation note if the cap fires.
                let (stdout_content, stdout_timed_out) =
                    join_drain_with_timeout(stdout_task, DRAIN_TIMEOUT).await;
                let (stderr_content, stderr_timed_out) =
                    join_drain_with_timeout(stderr_task, DRAIN_TIMEOUT).await;

                // No output-length truncation here: the agent layer's
                // `persist_oversized_results` auto-persists any oversized
                // result to the scratchpad losslessly. Truncating here would
                // corrupt binary-in-base64 pipelines (see #1 in the trial
                // feedback).
                let mut output =
                    assemble_command_output(&stdout_content, &stderr_content, exit_code);
                if stdout_timed_out || stderr_timed_out {
                    append_drain_truncation_note(&mut output, stdout_timed_out, stderr_timed_out);
                }
                Ok(output)
            }
        }
    }
}

/// Terminate the child and — on Unix — its entire process group. Called
/// on timeout and on cancellation. On Unix we rely on the `setsid()` done
/// in `pre_exec`: the child's pid is also its pgid, so `kill(-pgid, …)`
/// reaches every backgrounded descendant it spawned (e.g.
/// `(sleep 3600 &)` survives a plain `child.kill()` but is caught here).
/// The fallback `child.kill().await` is a no-op on Unix once the group
/// has been signaled but still the right primitive on Windows.
async fn kill_child_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            let pgid = pid as libc::pid_t;
            // SAFETY: `kill(2)` is always safe to call; it just returns an
            // error if the target is gone. Sending to `-pgid` targets the
            // whole process group. Errors here usually mean the group
            // already exited; log at debug so an unkillable group still
            // leaves a trail without spamming default verbosity.
            let term_result = unsafe { libc::kill(-pgid, libc::SIGTERM) };
            if term_result != 0 {
                tracing::debug!(
                    "libc::kill(-{}, SIGTERM) failed: {}",
                    pgid,
                    std::io::Error::last_os_error()
                );
            }
            // Brief grace period so well-behaved children can shut down
            // cleanly before SIGKILL lands.
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            let kill_result = unsafe { libc::kill(-pgid, libc::SIGKILL) };
            if kill_result != 0 {
                tracing::debug!(
                    "libc::kill(-{}, SIGKILL) failed: {}",
                    pgid,
                    std::io::Error::last_os_error()
                );
            }
        }
    }
    if let Err(error) = child.kill().await {
        tracing::debug!("failed to kill child process: {}", error);
    }
}

/// Upper bound on draining a child's stdout/stderr after it has exited. A
/// backgrounded grandchild that inherited the pipe write handle can keep the
/// pipe open past the direct child's exit; rather than block the tool call we
/// cap the drain, abort it, and attach a truncation note.
const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

async fn read_to_string_best_effort<R>(reader: Option<R>) -> String
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let mut content = String::new();
    if let Some(mut reader) = reader
        && let Err(error) = reader.read_to_string(&mut content).await
    {
        tracing::debug!("failed to read child output: {}", error);
    }
    content
}

fn assemble_command_output(stdout: &str, stderr: &str, exit_code: i32) -> ToolOutput {
    let mut result_text = String::new();
    if !stdout.is_empty() {
        result_text.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !result_text.is_empty() {
            result_text.push_str("\n--- stderr ---\n");
        }
        result_text.push_str(stderr);
    }
    if exit_code != 0 {
        result_text.push_str(&format!("\nExit code: {}", exit_code));
    }

    ToolOutput::text(
        if result_text.is_empty() {
            "(no output)".to_string()
        } else {
            result_text
        },
        exit_code != 0,
    )
}

/// Windows-only: spawn via `CreateProcessAsUserW` with a Low-integrity token,
/// read stdout/stderr from the pipe `File`s, and wait/kill through blocking
/// tasks. Mirrors the timeout/cancellation semantics of the standard path.
///
/// Stdout/stderr are drained on dedicated tasks that start *before* the child
/// wait begins. Without that, a child that writes more than the pipe buffer
/// (1 MiB hinted; smaller if the kernel rounds down) before anyone reads will
/// block in `WriteFile`, the wait never returns, and the whole call times out
/// with truncated output. After the child exits or is killed, the pipe write
/// ends close and the drain tasks terminate at EOF.
///
/// # Drain timeouts
///
/// On Windows there is no atomic "kill process tree" primitive available in
/// this code path (a future refactor could wrap the child in a Job Object
/// with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`). Consequently, a grandchild
/// that inherits the pipe write handles can keep the pipe alive past the
/// direct child's exit — the drain tasks would then block on `ReadFile`
/// until the grandchild finally exits. To bound the tool-call wall time we
/// cap every drain await with [`DRAIN_TIMEOUT`]; on timeout the drain task
/// is aborted, any output already read is lost, and we attach a diagnostic
/// note so the model can reason about truncation.
#[cfg(windows)]
async fn run_windows_low_integrity(
    command: &str,
    timeout_ms: u64,
    cancellation: CancellationToken,
) -> Result<ToolOutput> {
    use std::{sync::Arc, time::Duration};

    // Bound the post-kill cleanup wait so a stuck `TerminateProcess` or a
    // drain task that somehow fails to reach EOF can't hang the tool
    // indefinitely. Two seconds is generous for kernel-side teardown.
    const POST_KILL_TIMEOUT: Duration = Duration::from_secs(2);

    let mut sandboxed = crate::sandbox::spawn_low_integrity_command(command).map_err(|error| {
        AgshError::ToolExecution {
            tool_name: "execute_command".to_string(),
            message: format!("failed to spawn sandboxed command: {}", error),
        }
    })?;

    let stdout = sandboxed.take_stdout().map(tokio::fs::File::from_std);
    let stderr = sandboxed.take_stderr().map(tokio::fs::File::from_std);

    let child = Arc::new(sandboxed);
    let timeout_duration = Duration::from_millis(timeout_ms);

    let stdout_task = tokio::spawn(async move { read_to_string_best_effort(stdout).await });
    let stderr_task = tokio::spawn(async move { read_to_string_best_effort(stderr).await });

    let wait_child = Arc::clone(&child);
    // `tokio::select!` requires the future passed to the happy-path branch
    // (`join = ...`) to be polled without consuming ownership of the handle,
    // because the other two branches need to move the same handle into
    // `abort_after_timeout` if their future resolves first. Polling
    // `&mut wait_handle` satisfies `JoinHandle`'s `Future` impl (it has a
    // `&mut self`-based `poll`) without committing the move until we know
    // which branch wins.
    let mut wait_handle = tokio::task::spawn_blocking(move || wait_child.wait_blocking());

    tokio::select! {
        _ = cancellation.cancelled() => {
            if let Err(error) = child.kill() {
                tracing::debug!("failed to kill sandboxed child: {}", error);
            }
            abort_after_timeout(wait_handle, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stdout_task, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stderr_task, POST_KILL_TIMEOUT).await;
            Err(AgshError::Interrupted)
        }
        _ = tokio::time::sleep(timeout_duration) => {
            if let Err(error) = child.kill() {
                tracing::debug!("failed to kill sandboxed child: {}", error);
            }
            abort_after_timeout(wait_handle, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stdout_task, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stderr_task, POST_KILL_TIMEOUT).await;
            Ok(ToolOutput::text(
                format!("Command timed out after {}ms", timeout_ms),
                true,
            ))
        }
        join = &mut wait_handle => {
            let status = join
                .map_err(|error| AgshError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message: format!("wait task panicked: {}", error),
                })?
                .map_err(|error| AgshError::ToolExecution {
                    tool_name: "execute_command".to_string(),
                    message: format!("failed to wait for command: {}", error),
                })?;

            let exit_code = status.code().unwrap_or(-1);
            let (stdout_content, stdout_timed_out) =
                join_drain_with_timeout(stdout_task, DRAIN_TIMEOUT).await;
            let (stderr_content, stderr_timed_out) =
                join_drain_with_timeout(stderr_task, DRAIN_TIMEOUT).await;
            if stdout_timed_out || stderr_timed_out {
                tracing::warn!(
                    "sandboxed command output drain timed out after {:?}; \
                     a background process may be holding the pipe open",
                    DRAIN_TIMEOUT
                );
            }
            let mut output =
                assemble_command_output(&stdout_content, &stderr_content, exit_code);
            if stdout_timed_out || stderr_timed_out {
                append_drain_truncation_note(
                    &mut output,
                    stdout_timed_out,
                    stderr_timed_out,
                );
            }
            Ok(output)
        }
    }
}

/// Await a `JoinHandle<String>` up to `timeout`. If the timeout expires the
/// task is aborted and an empty string is returned alongside `timed_out=true`
/// so the caller can surface a truncation note.
async fn join_drain_with_timeout(
    mut task: tokio::task::JoinHandle<String>,
    timeout: std::time::Duration,
) -> (String, bool) {
    tokio::select! {
        result = &mut task => match result {
            Ok(content) => (content, false),
            Err(error) => {
                tracing::debug!("drain task failed: {}", error);
                (String::new(), false)
            }
        },
        _ = tokio::time::sleep(timeout) => {
            task.abort();
            (String::new(), true)
        }
    }
}

/// Abort any pending `JoinHandle` after `timeout`. Used on cancel/timeout
/// cleanup paths where we don't need the task's output — just its termination.
#[cfg(windows)]
async fn abort_after_timeout<T: 'static>(
    mut handle: tokio::task::JoinHandle<T>,
    timeout: std::time::Duration,
) {
    tokio::select! {
        _ = &mut handle => {}
        _ = tokio::time::sleep(timeout) => {
            handle.abort();
        }
    }
}

fn append_drain_truncation_note(
    output: &mut ToolOutput,
    stdout_timed_out: bool,
    stderr_timed_out: bool,
) {
    let note = match (stdout_timed_out, stderr_timed_out) {
        (true, true) => {
            "\n(stdout/stderr drain timed out; output may be truncated — a background process likely held the pipe open past the child's exit)"
        }
        (true, false) => {
            "\n(stdout drain timed out; output may be truncated — a background process likely held the pipe open past the child's exit)"
        }
        (false, true) => {
            "\n(stderr drain timed out; output may be truncated — a background process likely held the pipe open past the child's exit)"
        }
        (false, false) => return,
    };
    if let Some(crate::provider::ToolResultContent::Text { text }) = output.content.last_mut() {
        text.push_str(note);
    }
}

/// Build the `(program, args)` pair an ACP `terminal/create` should
/// run for the user-supplied shell command. Mirrors the platform
/// choices the local spawn makes (`sh -c …` on Unix, PowerShell with
/// the UTF-8 prelude on Windows) so a delegated command behaves like
/// the local one.
fn shell_invocation(command: &str) -> (String, Vec<String>) {
    #[cfg(windows)]
    {
        let wrapped = crate::sandbox::wrap_command_with_utf8_output(command);
        ("powershell.exe".to_string(), vec![
            "-NoProfile".to_string(),
            "-NonInteractive".to_string(),
            "-Command".to_string(),
            wrapped,
        ])
    }
    #[cfg(not(windows))]
    {
        ("sh".to_string(), vec![
            "-c".to_string(),
            command.to_string(),
        ])
    }
}

/// Assemble a [`ToolOutput`] from a [`crate::frontend::DelegatedExecOutput`].
/// Matches the local execute_command's final-string shape so the model
/// can't tell whether the command ran locally or in the editor's
/// terminal: stdout (combined output here, since ACP returns one
/// stream) + an "Exit code: N" trailer for non-zero exits + a
/// truncation note when the editor dropped output.
fn assemble_delegated_output(output: crate::frontend::DelegatedExecOutput) -> ToolOutput {
    let mut text = output.output.clone();
    let exit_code = output.exit_code.unwrap_or(0);
    let is_error = exit_code != 0 || output.signal.is_some();
    if let Some(signal) = output.signal.as_deref() {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!("Terminated by signal: {}", signal));
    } else if exit_code != 0 {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!("Exit code: {}", exit_code));
    }
    if output.truncated {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(
            "(output truncated by the editor's terminal-buffer cap; rerun with a narrower \
             scope or pipe to a file for the full output)",
        );
    }
    ToolOutput::text(text, is_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests::text_content;

    fn test_shared_permission() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(
            Permission::Write,
            crate::permission::EnabledPermissions::ALL,
        )
    }

    /// Construct an `ExecuteCommandTool` for tests with a backend probe
    /// matching whatever the host actually supports. Tests that need a
    /// specific probe state (e.g. exercising the "backend unavailable"
    /// hard-error path) should build `ExecuteCommandTool` directly with
    /// the desired `BackendProbe` rather than going through this helper.
    fn test_tool(
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
    ) -> ExecuteCommandTool {
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        ExecuteCommandTool {
            sandbox_capability,
            sandbox_backend: crate::config::SandboxBackend::Landlock,
            backend_probe,
            shared_permission,
            sandbox_enabled,
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        }
    }

    #[tokio::test]
    async fn test_execute_command() {
        let tool = test_tool(test_shared_permission(), true);
        let result = tool
            .execute(
                serde_json::json!({"command": "echo hello"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert_eq!(text_content(&result).trim(), "hello");
    }

    /// Regression test for the orphaned-grandchild bug: a command that
    /// backgrounds a long-running helper (`(sleep 30 &)`) must have that
    /// helper killed when the tool times out, not outlive the agent. The
    /// child is placed in its own process group via `setsid` so the tool
    /// can signal the whole tree via `kill(-pgid, …)`.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_execute_command_timeout_kills_grandchild() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let marker = temp_dir.path().join("marker");
        let marker_str = marker.to_str().expect("utf-8 path").to_string();

        let tool = test_tool(test_shared_permission(), false);

        // The grandchild sleeps 3s then touches `marker`. If it survived
        // the timeout, the marker file will appear. The timeout is 300ms
        // and we wait 5s below for a definitive "did it survive?" answer.
        let script = format!(
            "( sleep 3 && : > '{}' ) & echo backgrounded; sleep 30",
            marker_str
        );
        let result = tool
            .execute(
                serde_json::json!({ "command": script, "timeout_ms": 300u64 }),
                CancellationToken::new(),
            )
            .await
            .expect("execute should not error");

        // Tool reports timeout.
        assert!(result.is_error);
        let text = text_content(&result);
        assert!(text.contains("timed out"), "got: {:?}", text);

        // Wait well past the grandchild's sleep-3s. If the marker
        // materializes, the grandchild wasn't killed — the bug is back.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        assert!(
            !marker.exists(),
            "grandchild survived timeout and created marker at {:?}",
            marker
        );
    }

    #[tokio::test]
    async fn test_execute_command_failure() {
        let tool = test_tool(test_shared_permission(), true);
        let result = tool
            .execute(
                serde_json::json!({"command": "false"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
    }

    /// When the configured sandbox backend isn't usable, read-mode
    /// `execute_command` must return `Err(AgshError::ToolExecution)`
    /// — *not* `Ok(ToolOutput { is_error: true })`. The hard error
    /// path is how the model is forced to surface the failure to the
    /// user rather than just retrying or describing it as a tool
    /// result.
    #[tokio::test]
    async fn test_execute_command_hard_errors_when_backend_unavailable() {
        let read_only_perm = crate::permission::SharedPermission::new(
            Permission::Read,
            crate::permission::EnabledPermissions::ALL,
        );
        let tool = ExecuteCommandTool {
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Bubblewrap,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "bwrap not found on PATH".to_string(),
            },
            shared_permission: read_only_perm,
            sandbox_enabled: true,
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({"command": "echo nope"}),
                CancellationToken::new(),
            )
            .await;
        match result {
            Err(AgshError::ToolExecution { tool_name, message }) => {
                assert_eq!(tool_name, "execute_command");
                // The Linux error path splices in the configured
                // backend's display name (`Bubblewrap`); the non-Linux
                // variant drops the Linux-specific config reference
                // and reads "sandbox is unavailable: ...". Both must
                // include the probe reason verbatim.
                #[cfg(target_os = "linux")]
                assert!(
                    message.contains("Bubblewrap"),
                    "expected backend display name in error: {}",
                    message
                );
                assert!(
                    message.contains("bwrap not found on PATH"),
                    "expected probe reason in error: {}",
                    message
                );
            }
            Err(other) => panic!("expected ToolExecution, got {:?}", other),
            Ok(output) => panic!("expected hard error, got Ok({:?})", text_content(&output)),
        }
    }

    /// When the tool is invoked at Write permission, an unavailable
    /// sandbox backend must NOT short-circuit the spawn — the user
    /// has explicitly opted out of sandboxing for this command.
    #[tokio::test]
    async fn test_execute_command_runs_without_sandbox_when_write_mode() {
        let write_perm = crate::permission::SharedPermission::new(
            Permission::Write,
            crate::permission::EnabledPermissions::ALL,
        );
        let tool = ExecuteCommandTool {
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Bubblewrap,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "bwrap not found on PATH".to_string(),
            },
            shared_permission: write_perm,
            sandbox_enabled: true,
            cwd: crate::agent::test_cwd(),
            frontend: Arc::new(crate::frontend::SilentFrontend),
        };
        let result = tool
            .execute(
                serde_json::json!({"command": "echo hello"}),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed in write mode");
        assert!(!result.is_error);
        assert_eq!(text_content(&result).trim(), "hello");
    }

    #[tokio::test]
    async fn test_execute_command_large_output_not_truncated() {
        // Output well over the old 30 KB cap — the tool must return it in
        // full. The agent layer handles oversize downstream.
        let tool = test_tool(test_shared_permission(), true);
        let result = tool
            .execute(
                // 50 000 "x" characters. POSIX-portable — uses `head` and `tr`
                // instead of bash brace expansion so it works under `dash`
                // (Debian/Ubuntu's default `/bin/sh`) as well as `bash`.
                serde_json::json!({
                    "command": "head -c 50000 /dev/zero | tr '\\0' x"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        let text = text_content(&result);
        assert!(
            !text.contains("(output truncated"),
            "no truncation marker expected, got: {:.200}...",
            text
        );
        assert!(
            text.trim().len() >= 50_000,
            "expected >= 50 000 chars, got {}",
            text.trim().len()
        );
    }

    /// Regression test for the stdout/stderr pipe deadlock on Unix: a
    /// command writing far more than the OS pipe buffer (~64 KiB on Linux)
    /// must complete without blocking. Before draining stdout/stderr on
    /// dedicated tasks that start *before* `child.wait()`, the child
    /// blocked in `write()`, `wait()` never returned, and the call hit a
    /// spurious timeout with truncated output.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_execute_command_large_output_no_deadlock() {
        let tool = test_tool(test_shared_permission(), true);
        // 5 MiB of 'x' — two orders of magnitude past any pipe buffer.
        let result = tool
            .execute(
                serde_json::json!({
                    "command": "head -c 5242880 /dev/zero | tr '\\0' x"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error, "large output spuriously flagged as error");
        let text = text_content(&result);
        assert!(
            !text.contains("drain timed out"),
            "unexpected drain-timeout note: {:.200}",
            text
        );
        assert!(
            text.trim().len() >= 5_242_880,
            "expected >= 5 MiB of output, got {}",
            text.trim().len()
        );
    }

    #[cfg(windows)]
    mod windows_sandbox {
        use super::*;

        fn read_permission() -> crate::permission::SharedPermission {
            crate::permission::SharedPermission::new(
                Permission::Read,
                crate::permission::EnabledPermissions::ALL,
            )
        }

        /// Build an `ExecuteCommandTool` for the Low-integrity Windows
        /// path. Mirrors `super::test_tool` (which always calls
        /// `sandbox::detect()` and would resolve to `LowIntegrity` on
        /// Windows anyway) but constructs the fields explicitly so the
        /// tests document the intended state.
        fn windows_test_tool(
            shared_permission: crate::permission::SharedPermission,
        ) -> ExecuteCommandTool {
            let sandbox_capability = crate::sandbox::SandboxCapability::LowIntegrity;
            let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
            ExecuteCommandTool {
                sandbox_capability,
                // `sandbox_backend` is Linux-only metadata; on Windows
                // the value is never read but the field must still be
                // populated. `Landlock` is the conventional placeholder.
                sandbox_backend: crate::config::SandboxBackend::Landlock,
                backend_probe,
                shared_permission,
                sandbox_enabled: true,
                cwd: crate::agent::test_cwd(),
            }
        }

        /// Under Low integrity, writing to the user's profile directory
        /// must be denied by the OS. The test probes a path under
        /// `%USERPROFILE%` and asserts the file is never created.
        #[tokio::test]
        async fn test_windows_sandbox_blocks_write_to_userprofile() {
            let probe_path = format!(
                "{}\\agsh-sandbox-probe.txt",
                std::env::var("USERPROFILE").expect("USERPROFILE must be set on Windows")
            );
            // Clean any stray file from an earlier failed run before starting.
            let _ = std::fs::remove_file(&probe_path);

            let tool = windows_test_tool(read_permission());
            let _ = tool
                .execute(
                    serde_json::json!({
                        "command": format!("echo hello > \"{}\"", probe_path),
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("execute should not error");

            let existed = std::path::Path::new(&probe_path).exists();
            // Defensive cleanup even if the assertion below fails.
            let _ = std::fs::remove_file(&probe_path);
            assert!(
                !existed,
                "Low-integrity sandbox should have blocked write to {}",
                probe_path
            );
        }

        /// A command that produces well over the default Windows pipe buffer
        /// (~4 KB) of output must complete without deadlocking and without
        /// truncation. Before the concurrent-drain fix, the child would block
        /// in `WriteFile` past the buffer, the wait would never return, and
        /// the tool would report a spurious timeout.
        #[tokio::test]
        async fn test_windows_sandbox_large_output_under_sandbox() {
            let tool = windows_test_tool(read_permission());
            // PowerShell builds a 262144-char string in memory then emits it
            // as one line. Total output is ~256 KB — well past any plausible
            // pipe buffer.
            let result = tool
                .execute(
                    serde_json::json!({
                        "command": "'x' * 262144",
                        "timeout_ms": 60000u64,
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("execute should not error");

            assert!(
                !result.is_error,
                "large-output command should not be flagged as an error"
            );
            let text = text_content(&result);
            let x_count = text.matches('x').count();
            assert!(
                x_count >= 262144,
                "expected >= 262144 'x' characters in output, got {}",
                x_count
            );
        }

        /// The child's stdin must be connected to `NUL`, not inherited from
        /// the agent's TTY and not left as an invalid handle. `$input`
        /// enumerates pipeline input; piped from NUL it yields zero objects.
        /// The command must complete promptly rather than hanging on a
        /// dangling stdin.
        #[tokio::test]
        async fn test_windows_sandbox_stdin_is_null() {
            let tool = windows_test_tool(read_permission());
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                tool.execute(
                    serde_json::json!({
                        "command": "($input | Measure-Object).Count",
                        "timeout_ms": 5000u64,
                    }),
                    CancellationToken::new(),
                ),
            )
            .await
            .expect("command must not hang waiting for stdin")
            .expect("execute should not error");

            assert!(!result.is_error);
            let text = text_content(&result);
            assert!(
                text.trim().starts_with('0'),
                "expected stdin-object count of 0, got {:?}",
                text
            );
        }

        /// Round-trip a grab-bag of tricky marker strings through PowerShell
        /// to confirm `quote_command_arg` + PowerShell's argv parser are
        /// inverses. Uses PowerShell single-quote literals internally so the
        /// test exercises our command-line encoding, not PS string rules.
        #[tokio::test]
        async fn test_windows_sandbox_quoting_roundtrip() {
            let tool = windows_test_tool(read_permission());

            let cases: &[&str] = &[
                "plain",
                "with spaces",
                r#"quotes "inside""#,
                r"back\slashes",
                "meta & chars | pipe > redir",
                "日本語 unicode",
            ];
            for marker in cases {
                // Escape ' as '' inside the PS single-quote literal.
                let script = format!("Write-Output '{}'", marker.replace('\'', "''"));
                let result = tool
                    .execute(
                        serde_json::json!({ "command": script, "timeout_ms": 10000u64 }),
                        CancellationToken::new(),
                    )
                    .await
                    .expect("execute should not error");
                assert!(!result.is_error, "command for marker {:?} errored", marker);
                let text = text_content(&result);
                assert!(
                    text.contains(marker),
                    "marker {:?} missing from output {:?}",
                    marker,
                    text
                );
            }
        }

        /// Regression test for the parent-env-inheritance leak: secrets set
        /// in the parent (API keys, OAuth tokens) must not appear in the
        /// sandboxed child's environment, because a Low-integrity child
        /// can still open outbound sockets and exfiltrate them.
        #[tokio::test]
        async fn test_windows_sandbox_scrubs_provider_api_keys() {
            // SAFETY: tests run under `cargo test`, which is single-threaded
            // per target by default for integration tests, and this env var
            // is scoped to the test's probe command. Acceptable for a test.
            unsafe {
                std::env::set_var("ANTHROPIC_API_KEY", "probe-12345-leaked");
            }

            let tool = windows_test_tool(read_permission());
            let result = tool
                .execute(
                    serde_json::json!({
                        "command": "$env:ANTHROPIC_API_KEY",
                        "timeout_ms": 10000u64,
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("execute should not error");

            unsafe {
                std::env::remove_var("ANTHROPIC_API_KEY");
            }

            let text = text_content(&result);
            assert!(
                !text.contains("probe-12345-leaked"),
                "parent API key leaked into sandboxed child env: {:?}",
                text
            );
        }

        /// Reads must still succeed under Low integrity. The hosts file is
        /// readable by Everyone on stock Windows, so it's a good probe.
        #[tokio::test]
        async fn test_windows_sandbox_allows_read() {
            let tool = windows_test_tool(read_permission());
            let result = tool
                .execute(
                    serde_json::json!({
                        "command": "type C:\\Windows\\System32\\drivers\\etc\\hosts",
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("execute should not error");

            assert!(
                !result.is_error,
                "reading %WINDIR%\\System32\\drivers\\etc\\hosts should succeed under Low integrity"
            );
        }
    }
}
