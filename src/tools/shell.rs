//! `shell_execute` tool. Spawns a shell process, optionally constrained by the platform sandbox
//! when permissions are read-only, and streams stdout/stderr back to the agent as it arrives.
//!
//! The sandbox is Landlock or Bubblewrap on Linux (see [`crate::sandbox`] for which is preferred),
//! `sandbox-exec` on macOS, and a low-integrity token on Windows, which is spawned through
//! `CreateProcessAsUserW` rather than `tokio::process` because the standard library offers no hook
//! for injecting one.

use std::sync::Arc;

use async_trait::async_trait;

use super::{Tool, ToolOutput, util::require_str};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
};

/// `timeout_ms` when the caller passes none. Single source of truth for both the parameter unwrap
/// and the description shown to the agent.
const DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// The directories Bubblewrap masks with a private tmpfs: where socket-on-disk IPC lives, replaced
/// by empty, writable, throwaway space.
#[cfg(target_os = "linux")]
fn bwrap_masks() -> Vec<std::path::PathBuf> {
    let mut masks: Vec<std::path::PathBuf> = ["/tmp", "/run", "/var/tmp"]
        .iter()
        .map(std::path::PathBuf::from)
        .collect();
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR")
        && std::path::Path::new(&xdg).is_absolute()
    {
        masks.push(xdg.into());
    }
    masks
}

/// Every bwrap argument up to the `--` separator, for `writable` roots.
///
/// Factored out of the spawn so the ordering rule below is testable without a live child. Order is
/// the whole correctness argument here and it is invisible in the resulting mount namespace: bwrap
/// applies operations onto the new root in sequence and the last one to touch a path wins, so a
/// bind placed before the tmpfs masks is silently undone by them. A workspace under `/tmp` (where
/// every test fixture and a fair number of real scratch directories live) would come out read-only
/// with no error from bwrap, no error from meka, and a level that quietly confines the shell to
/// nothing.
#[cfg(target_os = "linux")]
fn bwrap_args(
    writable: &[std::path::PathBuf],
    cwd: &std::path::Path,
    private: &[std::path::PathBuf],
) -> Vec<std::ffi::OsString> {
    // `--ro-bind /` enforces "no writes", `--unshare-*` cuts off PID / user / UTS / IPC views, and
    // the tmpfs masks over `/run`, `/tmp`, `/var/tmp` and `$XDG_RUNTIME_DIR` make the dbus and
    // systemd-user sockets unreachable so the agent cannot `dbus-send` state-changing methods.
    // `--unshare-net` is intentionally absent; network must stay open for `curl | pdftotext` and
    // similar pipelines.
    let mut args: Vec<std::ffi::OsString> = [
        "--new-session",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        "--unshare-cgroup-try",
    ]
    .iter()
    .map(std::ffi::OsString::from)
    .collect();

    for mask in bwrap_masks() {
        args.push("--tmpfs".into());
        args.push(mask.into());
    }

    // The working directory, read-only, after the masks and before the writable binds.
    //
    // Without it a session whose cwd is under a masked directory loses the directory entirely, and
    // bwrap's fallback is silent: `Command::current_dir` chdirs before `execve`, bwrap cannot
    // re-enter that path inside the new root, and it lands the child in `$HOME` instead, while
    // `file_read` and `file_search` run in-process and see the real files.
    //
    // Read-only because a writable root that happens to be the cwd is bound read-write by the loop
    // below, and a later mount wins.
    //
    // Skipped when the cwd is a masked directory, or an ancestor of one, because the same
    // last-mount-wins rule would otherwise undo every mask above it: a session at `/tmp` would see
    // the host's tmux socket, one at `$XDG_RUNTIME_DIR` the session bus, and one at `/` the host's
    // PIDs. That is the escape `is_system_root` exists to prevent, arriving through the one door it
    // does not guard: it filters the writable roots, and the cwd is bound whether or not it is one.
    //
    // Nothing is lost by skipping it. `--chdir` below is unconditional, and a masked directory
    // still exists inside the sandbox as the empty tmpfs, so the child lands there and sees what
    // the mask intends rather than being relocated to `$HOME`. A path merely under a mask
    // (`/tmp/work`) is not a masked root, so it still gets its bind and still works.
    if !crate::workspace::is_system_root(cwd) {
        args.push("--ro-bind-try".into());
        args.push(cwd.into());
        args.push(cwd.into());
    }

    for root in writable {
        // `--bind-try`, not `--bind`. A root is canonicalized when the confinement is resolved and
        // mounted a moment later; a concurrent `shell_execute` running `rm -rf` on it in between
        // makes plain `--bind` abort the whole spawn with a bwrap error the model cannot act on.
        // Landlock already degrades the same way (it skips a root it cannot open rather than
        // failing the command), and this is the same rule spelled in bwrap's own vocabulary.
        args.push("--bind-try".into());
        args.push(root.into());
        args.push(root.into());
    }

    // meka's own directories, masked after every bind so they stay hidden under a writable root
    // that contains them: later mounts win, which is what would otherwise let a root at `$HOME`
    // hand `~/.config/meka` and the credential store beside it to a confined shell. See
    // `workspace::private_directories` for what is in the list and why.
    for directory in private {
        args.push("--tmpfs".into());
        args.push(directory.into());
    }

    // Asked for explicitly rather than inherited through the pre-`execve` chdir, so that a cwd
    // bwrap cannot enter is a loud failure the model can read instead of a silent relocation to
    // `$HOME`. Last, so it applies to the mounts above it.
    args.push("--chdir".into());
    args.push(cwd.into());
    args
}

/// The command Bubblewrap execs when the kernel has a usable Landlock: meka itself, reached as
/// `program`, which enacts the ruleset after bwrap's mounts and then becomes the shell.
///
/// The workspace roots are writable; every mask and `/dev` are scratch. The masks because bwrap
/// already made them writable throwaway space and the layer must not take that back, though a
/// socket from outside that a bind put under one stays out of reach; `/dev` because bwrap's is the
/// private minimal set rather than the host's devices, and refusing an ioctl there would only break
/// pty allocation (`script`, `expect`). Private directories are not passed: the masks hide them,
/// and a walk inside the sandbox would find only empty tmpfs.
#[cfg(target_os = "linux")]
fn inner_confinement_argv(
    program: &std::path::Path,
    writable: &[std::path::PathBuf],
) -> Vec<std::ffi::OsString> {
    let mut argv: Vec<std::ffi::OsString> = vec![program.into(), "confine".into()];
    for root in writable {
        argv.push("--writable".into());
        argv.push(root.into());
    }
    let mut scratch = bwrap_masks();
    scratch.push("/dev".into());
    for directory in scratch {
        argv.push("--scratch".into());
        argv.push(directory.into());
    }
    argv.push("--".into());
    argv
}

/// The executable that enacts the Landlock layer inside the sandbox: this process's own, opened
/// through `/proc/self/exe`, which names the running image even after the file behind it was
/// replaced or deleted. Bubblewrap execs it through the descriptor rather than a path, so a
/// package upgrade under a running server does not break the shell until a restart.
///
/// An error rather than a fallback to bwrap alone: a command run one layer short would report
/// nothing, and `/proc/self/exe` is unreadable only on a host where `/proc` itself is missing.
#[cfg(target_os = "linux")]
fn inner_layer_executable() -> Result<std::fs::File> {
    #[cfg(test)]
    let opened = INNER_LAYER_EXECUTABLE
        .with(|slot| slot.borrow().as_ref().map(std::fs::File::try_clone))
        .unwrap_or_else(|| std::fs::File::open("/proc/self/exe"));
    #[cfg(not(test))]
    let opened = std::fs::File::open("/proc/self/exe");
    opened.map_err(|error| MekaError::ToolExecution {
        tool_name: "shell_execute".to_string(),
        message: format!(
            "failed to open meka's own executable for the sandbox's Landlock layer: {error}"
        ),
    })
}

#[cfg(all(test, target_os = "linux"))]
thread_local! {
    /// The executable [`inner_layer_executable`] hands to bwrap under test, when a test has set
    /// one: the built `meka`, since the test binary has no `confine` verb.
    static INNER_LAYER_EXECUTABLE: std::cell::RefCell<Option<std::fs::File>> =
        const { std::cell::RefCell::new(None) };
}

pub(crate) struct ExecuteCommandTool {
    /// The process's workspace-ACE ledger on Windows.
    ///
    /// A handle on the one `process_grants()` singleton, not a per-tool ledger. It reads as a
    /// field because that is how the tool reaches it, and the distinction matters: an ACE is
    /// machine state, so a per-registry ledger had a sub-agent's teardown revoke the ACEs its
    /// parent was still writing through. Released by `release_process_grants` at process exit
    /// rather than by `Drop`; see [`crate::sandbox::windows::WindowsGrants`] for what
    /// that does and does not cover.
    #[cfg(windows)]
    pub(crate) windows_grants: std::sync::Arc<crate::sandbox::windows::WindowsGrants>,
    /// The write boundary, shared with `file_write`. The shell derives its sandbox allow-list from
    /// the same [`crate::workspace::WriteScope`] the file tools fence against, so the two cannot
    /// disagree about where a write may land.
    pub(crate) scope: crate::workspace::WriteScope,
    pub(crate) sandbox_capability: crate::sandbox::SandboxCapability,
    /// Backend chosen in config (or auto-resolved). Read only by the Linux hard-error message in
    /// [`Tool::execute`]; on macOS / Windows the field is populated but unused, so suppress the
    /// "never read" lint there without hiding regressions on Linux.
    #[cfg_attr(
        not(target_os = "linux"),
        allow(dead_code, reason = "read only on the Linux spawn path")
    )]
    pub(crate) sandbox_backend: crate::config::SandboxBackend,
    /// Probe outcome for [`Self::sandbox_backend`]. Drives the hard-error path at `read` when
    /// the backend isn't usable (bwrap missing, user namespaces denied, etc.). When `Ok(_)`,
    /// [`Self::sandbox_capability`] mirrors the inner capability and the spawn path runs normally.
    pub(crate) backend_probe: crate::sandbox::BackendProbe,
    pub(crate) sandbox_enabled: bool,
    pub(crate) site: crate::session::ToolSite,
}

impl ExecuteCommandTool {
    /// Whether a command may be spawned at `permission` given that its confinement `sandboxed` it
    /// or did not. Asked by `execute` ahead of the spawn and by `refusal_at_level` ahead of the
    /// approval prompt, so the two cannot disagree about a refusal approval could never lift.
    ///
    /// `[shell].sandbox = false` unconfines every level. That is right for the levels that never
    /// promised a boundary and wrong for every level that did: left alone it would run the shell
    /// with no confinement while the file tools stayed fenced, so one config key would make the
    /// level mean two different things and the weaker meaning would be the silent one.
    ///
    /// Refused rather than hidden. `required_permission` cannot hide it: `Workspace.allows` is true
    /// for everything by design, because scope is meant to be enforced at the door rather than by
    /// withholding tools. Refusing at that door is the same shape as the write fence, and it can
    /// say what to do about it where a missing tool could not.
    ///
    /// `unrestricted` is the only level whose intent is `Unconfined`; every other level reaching an
    /// unconfined spawn is a configuration that cannot deliver what the level says. Keyed on
    /// `workspace` alone, the sibling case stays open: `[tools.tool_permissions]` overrides a
    /// tool's required level with no floor, so `shell_execute = "read"` plus `[shell].sandbox =
    /// false` would run a plain `sh -c` at `read`, with the full parent environment, since the
    /// scrub is gated on `sandboxed` too.
    fn admit_confinement(&self, permission: Permission, sandboxed: bool) -> Result<()> {
        if permission != Permission::Unrestricted && !sandboxed {
            return Err(MekaError::ToolExecution {
                tool_name: "shell_execute".to_string(),
                message: "`[shell].sandbox = false` leaves nothing to confine this command \
                          below `unrestricted`; set `[shell].sandbox = true`"
                    .to_string(),
            });
        }
        if !sandboxed {
            return Ok(());
        }
        // Configured backend isn't usable on this host. Hard-error with the specific reason so the
        // model can surface it via `render::render_error` rather than treat the failure as a tool
        // result it could try to recover from.
        let Some(reason) = crate::sandbox::backend_unavailable_reason(&self.backend_probe) else {
            return Ok(());
        };
        // `sandbox_backend` is Linux-only; on other platforms there's nothing to reconfigure. The
        // only escape hatch is `unrestricted`, which is also the only level whose confinement is
        // `Unconfined` and so never reaches this branch.
        #[cfg(target_os = "linux")]
        let message = format!(
            "configured sandbox backend ({}) is unavailable: {}; set `[shell].sandbox_backend` to \
             a usable one",
            self.sandbox_backend, reason
        );
        #[cfg(not(target_os = "linux"))]
        let message = format!(
            "sandbox is unavailable: {reason}. `unrestricted` runs shell commands without a \
             sandbox."
        );
        Err(MekaError::ToolExecution {
            tool_name: "shell_execute".to_string(),
            message,
        })
    }
}

#[async_trait]
impl Tool for ExecuteCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "shell_execute".to_string(),
            description: format!(
                "{} Each call starts in the session's working directory; shell state does not \
                 persist between calls. At restricted levels a suitable sandbox is required. \
                 Background execution keeps `timeout_ms` unchanged. A command that prints more \
                 than {} is stopped.",
                if cfg!(windows) {
                    "Run PowerShell syntax via `powershell.exe -Command`. Do not nest another `powershell -Command`."
                } else {
                    "Run a POSIX shell command via `sh -c`. Use POSIX quoting."
                },
                crate::text::format_size(MAX_OUTPUT_BYTES),
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "default": DEFAULT_TIMEOUT.as_millis(),
                        "description": format!(
                            "Timeout in milliseconds. Default: {} ({} seconds).",
                            DEFAULT_TIMEOUT.as_millis(),
                            DEFAULT_TIMEOUT.as_secs(),
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
            Permission::Unrestricted
        }
    }

    /// The level and the configuration decide whether anything can confine a command; the command
    /// itself and the user's answer do not.
    async fn refusal_at_level(
        &self,
        level: Permission,
        _input: &serde_json::Value,
    ) -> Option<ToolOutput> {
        let confinement = crate::sandbox::Confinement::resolve(
            self.sandbox_enabled,
            level,
            &self.scope,
            &self.site.cwd,
        );
        self.admit_confinement(level, confinement.is_sandboxed())
            .err()
            .map(|error| ToolOutput::from_error(&error))
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        let command = require_str(&input, "command", "shell_execute")?;
        let timeout = input["timeout_ms"]
            .as_u64()
            .map(std::time::Duration::from_millis)
            .unwrap_or(DEFAULT_TIMEOUT);
        let permission = self.site.permission.get();
        // Three states, resolved once: the rest of this function asks the `Confinement` rather than
        // re-deriving "is it sandboxed" from the level, which is how the read-only and
        // workspace-writable cases would drift apart.
        let confinement = crate::sandbox::Confinement::resolve(
            self.sandbox_enabled,
            permission,
            &self.scope,
            &self.site.cwd,
        );
        let sandboxed = confinement.is_sandboxed();
        self.admit_confinement(permission, sandboxed)?;

        // Windows + sandboxed: spawn directly via CreateProcessAsUserW with a Low-integrity token.
        // This path can't go through tokio::process because the stdlib gives no hook for injecting
        // a custom token.
        #[cfg(windows)]
        if sandboxed
            && matches!(
                self.sandbox_capability,
                crate::sandbox::SandboxCapability::LowIntegrity
            )
        {
            // Two different mechanisms, picked by what the level promises. A workspace confinement
            // needs the ACEs in place *before* the token names them, so the grant happens here
            // rather than inside the spawn.
            let windows_confinement = match confinement.writable() {
                [] => crate::sandbox::windows::WindowsConfinement::LowIntegrity,
                roots => {
                    // Off the async executor. `ensure` calls `SetNamedSecurityInfoW`, which
                    // propagates the inheritable ACE over the *entire* existing tree -- seconds to
                    // minutes for a large workspace -- synchronously, while holding the ledger
                    // mutex. On a tokio worker that stalls streaming, every other `meka serve`
                    // session, and cancellation. `clippy::await_holding_lock` does not catch this
                    // shape: there is no `.await` inside the lock, just a long blocking syscall.
                    let grants = Arc::clone(&self.windows_grants);
                    let owned: Vec<std::path::PathBuf> = roots.to_vec();
                    let granted = tokio::task::spawn_blocking(move || {
                        for root in &owned {
                            if let Err(error) = grants.ensure(root) {
                                return Err((root.clone(), error));
                            }
                        }
                        Ok(())
                    })
                    .await
                    .map_err(|error| MekaError::ToolExecution {
                        tool_name: "shell_execute".to_string(),
                        message: format!("granting workspace write access panicked: {error}"),
                    })?;

                    if let Err((root, error)) = &granted {
                        return Err(MekaError::ToolExecution {
                            tool_name: "shell_execute".to_string(),
                            message: format!(
                                "failed to make '{}' writable for the sandboxed shell: {}; a \
                                 workspace root on Windows must be a directory meka owns",
                                root.display(),
                                error
                            ),
                        });
                    }
                    crate::sandbox::windows::WindowsConfinement::WriteRestricted(roots.to_vec())
                }
            };
            let relay = OutputRelay::for_call(&context);
            return run_windows_sandboxed(
                &command,
                &windows_confinement,
                self.site.cwd.get(),
                timeout,
                cancellation,
                relay,
            )
            .await;
        }

        #[cfg(windows)]
        let mut command_builder = {
            // Wrap with the UTF-8 output prelude so pipe output matches what the sandboxed path
            // produces; both on Rust's side this is decoded as UTF-8. Without the wrap, PowerShell
            // 5.1 defaults to the legacy console code page and mangles non-ASCII characters into
            // `?`.
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
            let (profile, params) = crate::sandbox::seatbelt::sandbox_profile_for(
                confinement.writable(),
                &crate::workspace::private_directories(),
            );
            let mut cmd = tokio::process::Command::new(crate::sandbox::seatbelt::SANDBOX_EXEC_PATH);
            cmd.arg("-p").arg(&profile);
            // `-D KEY=value` pairs, so a path never has to survive SBPL string quoting.
            cmd.args(&params);
            cmd.arg("sh").arg("-c").arg(&command);
            cmd
        } else {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(&command);
            cmd
        };

        // The executable that enacts the Landlock layer inside Bubblewrap, held open from here to
        // the spawn: the sandbox execs it through the descriptor, which the `pre_exec` below lets
        // through, and the verb closes it again before becoming the command.
        #[cfg(target_os = "linux")]
        let inner_executable: Option<std::fs::File> = match (sandboxed, &self.sandbox_capability) {
            (
                true,
                crate::sandbox::SandboxCapability::Bubblewrap {
                    landlock_abi: Some(_),
                    ..
                },
            ) => Some(inner_layer_executable()?),
            _ => None,
        };

        #[cfg(target_os = "linux")]
        let mut command_builder = if sandboxed
            && let crate::sandbox::SandboxCapability::Bubblewrap { bwrap_path, .. } =
                &self.sandbox_capability
        {
            // Bubblewrap path: `--ro-bind /` enforces "no writes", `--unshare-*` cuts off PID /
            // user / UTS / IPC views, tmpfs masks over `/run`, `/tmp`, `/var/tmp`, and
            // `$XDG_RUNTIME_DIR` make the dbus and systemd-user sockets unreachable so the agent
            // can't `dbus-send` state-changing methods. `--unshare-net` is intentionally absent;
            // network must stay open for `curl | pdftotext` and similar pipelines. The Landlock
            // layer inside closes what the masks do not reach: a socket under `$HOME`, the
            // abstract namespace, device ioctls.
            let mut cmd = tokio::process::Command::new(bwrap_path);
            cmd.args(bwrap_args(
                confinement.writable(),
                &self.site.cwd.get(),
                &crate::workspace::private_directories(),
            ));
            cmd.arg("--");
            if let Some(executable) = &inner_executable {
                let program = format!(
                    "/proc/self/fd/{}",
                    std::os::fd::AsRawFd::as_raw_fd(executable)
                );
                cmd.args(inner_confinement_argv(
                    std::path::Path::new(&program),
                    confinement.writable(),
                ));
            }
            cmd.arg("sh").arg("-c").arg(&command);
            cmd
        } else {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(&command);
            cmd
        };

        // The Landlock dialect's ABI, when this command runs under it.
        #[cfg(target_os = "linux")]
        let landlock_abi: Option<i32> = match (sandboxed, &self.sandbox_capability) {
            (true, crate::sandbox::SandboxCapability::Landlock { abi_version }) => {
                Some(*abi_version)
            }
            _ => None,
        };

        // Unix: place the child in its own session/process group via `setsid` so timeouts and
        // cancellation can kill the whole tree (including backgrounded grandchildren such as
        // `(sleep 3600 &)`) via `kill(-pgid, …)`. On Linux the Landlock setup runs in the same
        // closure, because `pre_exec` overwrites rather than chains, and only for the Landlock
        // dialect: under Bubblewrap the ruleset is enacted inside the sandbox by `meka confine`,
        // after bwrap's mounts, because a domain that handles any filesystem right refuses
        // `mount(2)`.
        #[cfg(unix)]
        {
            // Planned here, in the parent, because `pre_exec` runs after `fork` in a
            // single-threaded child where reading the filesystem and allocating are not
            // async-signal-safe.
            // No scratch paths: Landlock alone offers a command no temporary directory, because
            // the only one it could offer is a real directory under the real `/tmp`, and below
            // `unrestricted` meka writes to nothing but its store. Bubblewrap's tmpfs is the
            // answer for a tool that needs one.
            #[cfg(target_os = "linux")]
            let grants: Vec<crate::sandbox::LandlockGrant> = match landlock_abi {
                Some(abi) => crate::sandbox::landlock_grants(
                    abi,
                    confinement.writable(),
                    &[],
                    &crate::workspace::private_directories(),
                ),
                None => Vec::new(),
            };

            unsafe {
                command_builder.pre_exec(move || {
                    // SAFETY: `setsid(2)` is async-signal-safe and has no preconditions beyond "the
                    // caller isn't already a process group leader", which is guaranteed for a
                    // freshly forked child process.
                    if libc::setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    // The layer's executable rides into Bubblewrap on its descriptor, which stays
                    // close-on-exec until here so no other child of meka inherits it. `fcntl(2)`
                    // is async-signal-safe.
                    #[cfg(target_os = "linux")]
                    if let Some(executable) = &inner_executable
                        && libc::fcntl(
                            std::os::fd::AsRawFd::as_raw_fd(executable),
                            libc::F_SETFD,
                            0,
                        ) == -1
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    #[cfg(target_os = "linux")]
                    if let Some(abi) = landlock_abi {
                        crate::sandbox::apply_landlock(abi, &grants)
                            .map_err(std::io::Error::from_raw_os_error)?;
                    }
                    Ok(())
                });
            }
        }

        // Scrub env before spawn so secrets in the parent process (`ANTHROPIC_API_KEY`, `AWS_*`,
        // `GITHUB_TOKEN`, …) cannot ride along into a confined child. Sandboxes block writes/IPC
        // but leave the network open, so leaked env is a live exfiltration vector under prompt
        // injection. Only `unrestricted` keeps the full parent environment, and an approved command
        // still runs in its level's sandbox with the scrubbed environment, as
        // `docs/book/src/tools/shell.md` states. The Windows sandboxed branch applies the same
        // scrub inside its own spawn.
        #[cfg(unix)]
        if sandboxed {
            command_builder.env_clear();
            command_builder.envs(crate::sandbox::sandbox_child_env());
        }

        // Resolve commands against the agent's per-session cwd, not the process cwd. `/cd` mutates
        // the agent's cwd; this is how it actually reaches the child.
        command_builder.current_dir(self.site.cwd.get());

        let mut child = command_builder
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "shell_execute".to_string(),
                message: format!("failed to spawn command: {error}"),
            })?;

        // Drain stdout/stderr on dedicated tasks that start *before* the wait.
        // `tokio::process::Child::wait()` does not read the pipes; a child writing past the OS pipe
        // buffer (~64 KiB) would block in `write()`, `wait()` would never return, and the call
        // would spuriously hit the timeout below. After the child's process group exits the pipe
        // write ends close and the drains hit EOF.
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let relay = OutputRelay::for_call(&context);
        let budget = Arc::new(OutputBudget::new(MAX_OUTPUT_BYTES));
        let stop = tokio_util::sync::CancellationToken::new();
        let stdout_task = tokio::spawn({
            let relay = relay.clone();
            let budget = Arc::clone(&budget);
            let stop = stop.clone();
            async move { drain_output(stdout, relay, budget, stop).await }
        });
        let stderr_task = tokio::spawn({
            let relay = relay.clone();
            let budget = Arc::clone(&budget);
            let stop = stop.clone();
            async move { drain_output(stderr, relay, budget, stop).await }
        });

        // wait_with_output() consumes the child, so use wait() + manual stdout/stderr reading
        // instead to allow kill on cancellation. `biased`, so a bound reached in the same instant
        // the child exits is reported as the bound: the random pick would otherwise call the
        // command clean while its last reads were never taken.
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                kill_child_tree(&mut child).await;
                stdout_task.abort();
                stderr_task.abort();
                Err(MekaError::Interrupted)
            }
            _ = budget.exhausted.cancelled() => {
                kill_child_tree(&mut child).await;
                let drained = collect_drains(stdout_task, stderr_task, &stop).await;
                Ok(killed_command_output(&drained, &output_bound_reason()))
            }
            _ = tokio::time::sleep(timeout) => {
                kill_child_tree(&mut child).await;
                let drained = collect_drains(stdout_task, stderr_task, &stop).await;
                Ok(killed_command_output(&drained, &timed_out_reason(timeout, &budget)))
            }
            status = child.wait() => {
                finish_command(status, stdout_task, stderr_task, &stop, &budget).await
            }
        }
    }
}

/// The result of a command that ran to its end: its exit status with what the drains collected.
///
/// The drains may still cross the bound here: a child exits once its last write fits in the pipe,
/// so it can finish within a pipe's worth of the bound and leave the crossing to the collection
/// below. That output is cut like a killed command's, and the result says so first, because a clean
/// exit code over an incomplete transcript is the one shape the record must never take.
async fn finish_command(
    status: std::io::Result<std::process::ExitStatus>,
    stdout_task: tokio::task::JoinHandle<String>,
    stderr_task: tokio::task::JoinHandle<String>,
    stop: &tokio_util::sync::CancellationToken,
    budget: &OutputBudget,
) -> Result<ToolOutput> {
    let status = status.map_err(|error| MekaError::ToolExecution {
        tool_name: "shell_execute".to_string(),
        message: format!("failed to wait for command: {error}"),
    })?;

    let drained = collect_drains(stdout_task, stderr_task, stop).await;
    if drained.stopped_early {
        tracing::warn!(
            "command output drain stopped after {DRAIN_TIMEOUT:?}; a background process may be \
             holding the pipe open"
        );
    }

    if budget.exhausted.is_cancelled() {
        return Ok(led_by_reason(&drained, &output_cut_note())
            .with_metadata(command_exit_metadata(&status)));
    }

    // No output-length truncation here: the agent layer's `persist_oversized_results`
    // auto-persists any oversized result to the scratchpad losslessly. Truncating here would
    // corrupt binary-in-base64 pipelines (see #1 in the trial feedback).
    let exit_code = status.code().unwrap_or(-1);
    let mut output = assemble_command_output(&drained.stdout, &drained.stderr, exit_code);
    if drained.stopped_early {
        append_drain_stopped_note(&mut output);
    }
    Ok(output.with_metadata(command_exit_metadata(&status)))
}

/// The result of a command meka killed, at its timeout or at the output bound: the reason, then
/// everything it printed up to the kill. A frontend rendering a terminal shows "terminated" rather
/// than an exit code.
fn killed_command_output(drained: &DrainedOutput, reason: &str) -> ToolOutput {
    led_by_reason(drained, reason).with_metadata(killed_exit_metadata())
}

/// A failed result that states `reason` first and carries what the command printed after it.
///
/// The reason leads because the output can be 64 MiB and every surface shows a result's head (the
/// scratchpad preview, a background outcome, the REPL banner), so a reason at the end would be the
/// one line nobody reads; the record holds all of it either way.
fn led_by_reason(drained: &DrainedOutput, reason: &str) -> ToolOutput {
    let mut text = reason.to_string();
    let printed = join_streams(&drained.stdout, &drained.stderr);
    if !printed.is_empty() {
        text.push('\n');
        text.push_str(&printed);
    }
    let mut output = ToolOutput::text(text, true);
    if drained.stopped_early {
        append_drain_stopped_note(&mut output);
    }
    output
}

/// Why a command was killed at its timeout, in the words the model and every test read. When the
/// drains also reached the bound while the command died, the result says that too, since the
/// transcript is then cut as well as ended.
fn timed_out_reason(timeout: std::time::Duration, budget: &OutputBudget) -> String {
    let mut reason = format!("Command timed out after {}ms", timeout.as_millis());
    if budget.exhausted.is_cancelled() {
        reason.push('\n');
        reason.push_str(&output_cut_note());
    }
    reason
}

/// Why a command was killed at the output bound. Names the bound, so the model reruns with a
/// narrower command rather than a longer timeout.
fn output_bound_reason() -> String {
    format!(
        "Command stopped: its output exceeded {}",
        crate::text::format_size(MAX_OUTPUT_BYTES)
    )
}

/// What a result says when the output reached the bound while the command was ending for another
/// reason, or on its own: the transcript stops there, whatever the exit status says.
fn output_cut_note() -> String {
    format!(
        "Output exceeded {}; what the command printed past that point is not included",
        crate::text::format_size(MAX_OUTPUT_BYTES)
    )
}

/// Terminate the child and, on Unix, its entire process group. Called on timeout and on
/// cancellation. On Unix the `setsid()` done in `pre_exec` makes the child's pid its pgid, so
/// `kill(-pgid, …)` reaches every backgrounded descendant it spawned (`(sleep 3600 &)` survives a
/// plain `child.kill()` but is caught here). The fallback `child.kill().await` is a no-op on Unix
/// once the group has been signaled but still the right primitive on Windows.
async fn kill_child_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            let pgid = pid as libc::pid_t;
            // SAFETY: `kill(2)` is always safe to call; it just returns an error if the target is
            // gone. Sending to `-pgid` targets the whole process group. Errors here usually mean
            // the group already exited; log at debug so an unkillable group still leaves a trail
            // without spamming default verbosity.
            let term_result = unsafe { libc::kill(-pgid, libc::SIGTERM) };
            if term_result != 0 {
                let error = std::io::Error::last_os_error();
                tracing::debug!("failed to send SIGTERM to process group {pgid}: {error}");
            }
            // Brief grace period so well-behaved children can shut down cleanly before SIGKILL
            // lands.
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            let kill_result = unsafe { libc::kill(-pgid, libc::SIGKILL) };
            if kill_result != 0 {
                let error = std::io::Error::last_os_error();
                tracing::debug!("failed to send SIGKILL to process group {pgid}: {error}");
            }
        }
    }
    if let Err(error) = child.kill().await {
        tracing::debug!("failed to kill child process: {error}");
    }
}

/// Upper bound on draining a child's stdout/stderr after it has exited or been killed. A
/// backgrounded grandchild that inherited the pipe write handle can keep the pipe open past the
/// direct child's exit; rather than block the tool call, the drains are told to stop and return
/// what they have, and the result says so.
const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Forwards a running command's output to the frontend as it is produced. Cloned into both drain
/// tasks so stdout and stderr land in one stream in arrival order, which is what a terminal shows.
///
/// `None` when there is no tool call to correlate against (direct tool construction in tests), in
/// which case draining behaves exactly as it did before deltas existed.
#[derive(Clone)]
struct OutputRelay {
    frontend: Arc<dyn crate::frontend::Frontend>,
    tool_call_id: String,
}

impl OutputRelay {
    /// The id has to be captured here rather than inside the drain task: it lives in a task-local
    /// scoped by `Agent::resolve_and_execute_tool`, and `tokio::spawn` does not inherit
    /// task-locals. The relay for a call a model made; a call with no tool-use id has nothing to
    /// tag chunks with.
    fn for_call(context: &crate::tools::ToolContext) -> Option<Self> {
        context.tool_call_id.clone().map(|tool_call_id| Self {
            frontend: Arc::clone(&context.frontend),
            tool_call_id,
        })
    }

    async fn send(&self, chunk: String) {
        self.frontend
            .emit(crate::frontend::FrontendEvent::ToolCallOutputDelta {
                id: self.tool_call_id.clone(),
                chunk,
            })
            .await;
    }
}

/// How much a command may print, stdout and stderr together, before meka stops it.
///
/// Every byte under the bound is kept: in memory while the command runs, then in the scratchpad
/// when the result is too large for the conversation. So a bound is the only thing standing between
/// a command that writes faster than the turn ends (`cat /dev/zero`, a runaway build log) and the
/// process going down with it. Reaching it kills the command and says so, the way the timeout does,
/// rather than dropping part of what was printed: an elided middle is a hole in the record for the
/// operator and the agent alike, and a stopped command is something the agent can rerun narrower.
///
/// Above the largest legitimate outputs measured (a whole `git log -p`, an `ls -lR /usr`, both in
/// the tens of MiB) and well below the runaway cases.
const MAX_OUTPUT_BYTES: usize = 64 * crate::text::MIB;

/// One command's share of [`MAX_OUTPUT_BYTES`], held by both of its drains.
///
/// Shared rather than split per stream, so a command that prints everything on one stream gets the
/// whole bound, and one that splits its output cannot double it.
struct OutputBudget {
    remaining: std::sync::atomic::AtomicUsize,
    /// Fires when a drain has spent the budget. The spawn waits on it beside the timeout.
    exhausted: tokio_util::sync::CancellationToken,
}

impl OutputBudget {
    fn new(bytes: usize) -> Self {
        Self {
            remaining: std::sync::atomic::AtomicUsize::new(bytes),
            exhausted: tokio_util::sync::CancellationToken::new(),
        }
    }

    /// Spend `bytes`. `false` when they did not fit, at which point the budget is exhausted and
    /// stays so; the chunk that crossed the line is kept, so the record ends on a whole read.
    fn spend(&self, bytes: usize) -> bool {
        use std::sync::atomic::Ordering;
        let before = self
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                Some(remaining.saturating_sub(bytes))
            })
            .unwrap_or(0);
        if before >= bytes {
            return true;
        }
        self.exhausted.cancel();
        false
    }
}

/// Read a child pipe until EOF, the budget runs out, or `stop` fires, relaying each chunk as it
/// arrives and returning everything read.
///
/// Reads bytes rather than `read_to_string` because a chunk boundary can fall inside a multi-byte
/// character: the trailing incomplete sequence is carried over to the next read instead of being
/// relayed as replacement characters. What is relayed still covers the whole stream, so the live
/// view is unaffected by how the reads happened to split.
///
/// Whichever way the read ends, the returned string is everything the drain took in. The spawn
/// decides what to say about a stream that ended early; this function never drops a byte it read.
async fn drain_output<R>(
    reader: Option<R>,
    relay: Option<OutputRelay>,
    budget: Arc<OutputBudget>,
    stop: tokio_util::sync::CancellationToken,
) -> String
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let Some(mut reader) = reader else {
        return String::new();
    };

    let mut content: Vec<u8> = Vec::new();
    let mut relayed = 0usize;
    let mut buffer = [0u8; 8192];
    loop {
        // `read` is cancel-safe on a pipe: a read that has not completed has copied nothing, so
        // stopping here loses nothing already produced.
        let read = tokio::select! {
            read = reader.read(&mut buffer) => read,
            _ = stop.cancelled() => break,
        };
        match read {
            Ok(0) => break,
            Ok(read) => {
                let chunk = &buffer[..read];
                content.extend_from_slice(chunk);

                if let Some(relay) = &relay {
                    // Relay only the prefix that is complete UTF-8; whatever trails an incomplete
                    // sequence stays behind for the next read to finish.
                    let pending = &content[relayed..];
                    let valid = match std::str::from_utf8(pending) {
                        Ok(text) => text.len(),
                        Err(error) => error.valid_up_to(),
                    };
                    if valid > 0 {
                        // `valid` is the length of a verified-UTF-8 prefix, so the lossy decode
                        // never actually substitutes anything; it is just the panic-free spelling.
                        let text = String::from_utf8_lossy(&pending[..valid]).into_owned();
                        relayed += valid;
                        relay.send(text).await;
                    }
                }

                // After the chunk is kept, so the record ends on what was actually read; the spawn
                // kills the command, and this drain has nothing more to take from it.
                if !budget.spend(read) {
                    break;
                }
            }
            Err(error) => {
                tracing::debug!("failed to read child output: {error}");
                break;
            }
        }
    }

    // Lossy rather than a hard error: a command that emits a stray non-UTF-8 byte (a progress bar
    // in a foreign encoding, a binary blob on stderr) should still hand the model everything else
    // it printed.
    String::from_utf8_lossy(&content).into_owned()
}

/// What the two drains returned, and whether they were stopped before the pipes closed.
struct DrainedOutput {
    stdout: String,
    stderr: String,
    /// A grandchild kept a pipe open past [`DRAIN_TIMEOUT`], so the drains were told to stop and
    /// anything printed after that is not in `stdout` or `stderr`.
    stopped_early: bool,
}

/// Join both drains, waiting at most [`DRAIN_TIMEOUT`] in total for the pipes to close.
///
/// A drain that has not reached EOF by then is told to stop and returns what it has, so a
/// backgrounded grandchild holding the pipe delays the result by the timeout and costs the output
/// it prints afterwards, never the output already read.
async fn collect_drains(
    mut stdout_task: tokio::task::JoinHandle<String>,
    mut stderr_task: tokio::task::JoinHandle<String>,
    stop: &tokio_util::sync::CancellationToken,
) -> DrainedOutput {
    let deadline = tokio::time::Instant::now() + DRAIN_TIMEOUT;
    let mut stopped_early = false;
    let stdout = join_drain_by(&mut stdout_task, deadline, stop, &mut stopped_early).await;
    let stderr = join_drain_by(&mut stderr_task, deadline, stop, &mut stopped_early).await;
    DrainedOutput {
        stdout,
        stderr,
        stopped_early,
    }
}

/// One drain's result by `deadline`. Past it, `stop` is fired and the drain is joined for what it
/// has, which it returns as soon as it sees the signal.
async fn join_drain_by(
    task: &mut tokio::task::JoinHandle<String>,
    deadline: tokio::time::Instant,
    stop: &tokio_util::sync::CancellationToken,
    stopped_early: &mut bool,
) -> String {
    let joined = match tokio::time::timeout_at(deadline, &mut *task).await {
        Ok(joined) => joined,
        Err(_elapsed) => {
            *stopped_early = true;
            stop.cancel();
            task.await
        }
    };
    joined.unwrap_or_else(|error| {
        tracing::debug!("drain task failed: {error}");
        String::new()
    })
}

/// Structured exit status for frontends that render a terminal. `ExitStatus::code()` is `None`
/// exactly when a signal ended the process, so the two are read as alternatives rather than both
/// being guessed from the same number.
fn command_exit_metadata(status: &std::process::ExitStatus) -> crate::frontend::ToolOutputMetadata {
    #[cfg(unix)]
    let signal = {
        use std::os::unix::process::ExitStatusExt;
        status.signal().map(signal_name)
    };
    #[cfg(not(unix))]
    let signal = None;
    crate::frontend::ToolOutputMetadata::CommandExit {
        exit_code: status.code(),
        signal,
    }
}

/// Symbolic name for the signals a command realistically dies from. The number alone would render
/// as `SIG9` in a client's terminal UI next to a `SIGKILL` we report elsewhere for the same kill,
/// so the two paths have to agree. Numbers outside this set are stable enough nowhere to name.
#[cfg(unix)]
fn signal_name(number: i32) -> String {
    match number {
        libc::SIGHUP => "SIGHUP".to_string(),
        libc::SIGINT => "SIGINT".to_string(),
        libc::SIGQUIT => "SIGQUIT".to_string(),
        libc::SIGABRT => "SIGABRT".to_string(),
        libc::SIGKILL => "SIGKILL".to_string(),
        libc::SIGSEGV => "SIGSEGV".to_string(),
        libc::SIGPIPE => "SIGPIPE".to_string(),
        libc::SIGTERM => "SIGTERM".to_string(),
        other => format!("SIG{other}"),
    }
}

/// Exit status for a command meka killed, at its timeout or at the output bound. The child is torn
/// down without its status being reaped, so there is nothing to read it from. On Unix the kill
/// really is a signal ([`kill_child_tree`]); Windows has no signals, and claiming one there would
/// put a name in the client's terminal that never existed on that platform.
fn killed_exit_metadata() -> crate::frontend::ToolOutputMetadata {
    crate::frontend::ToolOutputMetadata::CommandExit {
        exit_code: None,
        #[cfg(unix)]
        signal: Some("SIGKILL".to_string()),
        #[cfg(not(unix))]
        signal: None,
    }
}

/// Both streams as one text, stderr under its own divider when both have something to show.
fn join_streams(stdout: &str, stderr: &str) -> String {
    let mut text = String::new();
    if !stdout.is_empty() {
        text.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !text.is_empty() {
            text.push_str("\n--- stderr ---\n");
        }
        text.push_str(stderr);
    }
    text
}

fn assemble_command_output(stdout: &str, stderr: &str, exit_code: i32) -> ToolOutput {
    let mut result_text = join_streams(stdout, stderr);
    if exit_code != 0 {
        result_text.push_str(&format!("\nExit code: {exit_code}"));
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

/// Windows-only: spawn via `CreateProcessAsUserW` with a Low-integrity token, read stdout/stderr
/// from the pipe `File`s, and wait/kill through blocking tasks. Mirrors the timeout/cancellation
/// semantics of the standard path.
///
/// Stdout/stderr are drained on dedicated tasks that start *before* the child wait begins. Without
/// that, a child that writes more than the pipe buffer (1 MiB hinted; smaller if the kernel rounds
/// down) before anyone reads will block in `WriteFile`, the wait never returns, and the whole call
/// times out with truncated output. After the child exits or is killed, the pipe write ends close
/// and the drain tasks terminate at EOF.
///
/// # Drain timeouts
///
/// On Windows there is no atomic "kill process tree" primitive available in this code path (a
/// future refactor could wrap the child in a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
/// Consequently, a grandchild that inherits the pipe write handles can keep the pipe alive past the
/// direct child's exit; the drain tasks would then block on `ReadFile` until the grandchild
/// finally exits. To bound the tool-call wall time the drains are collected under
/// [`DRAIN_TIMEOUT`] and told to stop at it, keeping what they read.
#[cfg(windows)]
async fn run_windows_sandboxed(
    command: &str,
    confinement: &crate::sandbox::windows::WindowsConfinement,
    cwd: std::path::PathBuf,
    timeout: std::time::Duration,
    cancellation: tokio_util::sync::CancellationToken,
    relay: Option<OutputRelay>,
) -> Result<ToolOutput> {
    use std::{sync::Arc, time::Duration};

    // Bound the post-kill cleanup wait so a stuck `TerminateProcess` or a drain task that somehow
    // fails to reach EOF can't hang the tool indefinitely. Two seconds is generous for kernel-side
    // teardown.
    const POST_KILL_TIMEOUT: Duration = Duration::from_secs(2);

    let mut sandboxed =
        crate::sandbox::windows::spawn_sandboxed_command(command, confinement, &cwd).map_err(
            |error| MekaError::ToolExecution {
                tool_name: "shell_execute".to_string(),
                message: format!("failed to spawn sandboxed command: {error}"),
            },
        )?;

    let stdout = sandboxed.take_stdout().map(tokio::fs::File::from_std);
    let stderr = sandboxed.take_stderr().map(tokio::fs::File::from_std);

    let child = Arc::new(sandboxed);
    let budget = Arc::new(OutputBudget::new(MAX_OUTPUT_BYTES));
    let stop = tokio_util::sync::CancellationToken::new();
    let stdout_task = tokio::spawn({
        let relay = relay.clone();
        let budget = Arc::clone(&budget);
        let stop = stop.clone();
        async move { drain_output(stdout, relay, budget, stop).await }
    });
    let stderr_task = tokio::spawn({
        let budget = Arc::clone(&budget);
        let stop = stop.clone();
        async move { drain_output(stderr, relay, budget, stop).await }
    });

    let wait_child = Arc::clone(&child);
    // `tokio::select!` requires the future passed to the happy-path branch (`join = ...`) to be
    // polled without consuming ownership of the handle, because the other branches need to move
    // the same handle into `abort_after_timeout` if their future resolves first. Polling `&mut
    // wait_handle` satisfies `JoinHandle`'s `Future` impl (it has a `&mut self`-based `poll`)
    // without committing the move until we know which branch wins.
    let mut wait_handle = tokio::task::spawn_blocking(move || wait_child.wait_blocking());

    // `biased` for the reason the Unix spawn gives: a bound reached as the child exits is the
    // bound, not a clean exit.
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => {
            if let Err(error) = child.kill() {
                tracing::debug!("failed to kill sandboxed child: {error}");
            }
            abort_after_timeout(wait_handle, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stdout_task, POST_KILL_TIMEOUT).await;
            abort_after_timeout(stderr_task, POST_KILL_TIMEOUT).await;
            Err(MekaError::Interrupted)
        }
        _ = budget.exhausted.cancelled() => {
            if let Err(error) = child.kill() {
                tracing::debug!("failed to kill sandboxed child: {error}");
            }
            abort_after_timeout(wait_handle, POST_KILL_TIMEOUT).await;
            let drained = collect_drains(stdout_task, stderr_task, &stop).await;
            Ok(killed_command_output(&drained, &output_bound_reason()))
        }
        _ = tokio::time::sleep(timeout) => {
            if let Err(error) = child.kill() {
                tracing::debug!("failed to kill sandboxed child: {error}");
            }
            abort_after_timeout(wait_handle, POST_KILL_TIMEOUT).await;
            let drained = collect_drains(stdout_task, stderr_task, &stop).await;
            Ok(killed_command_output(&drained, &timed_out_reason(timeout, &budget)))
        }
        join = &mut wait_handle => {
            let status = join.map_err(|error| MekaError::ToolExecution {
                tool_name: "shell_execute".to_string(),
                message: format!("wait task panicked: {error}"),
            })?;
            finish_command(status, stdout_task, stderr_task, &stop, &budget).await
        }
    }
}

/// Abort any pending `JoinHandle` after `timeout`. Used on cancel/timeout cleanup paths where we
/// don't need the task's output, just its termination.
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

/// Tell the model that the drains were stopped with a pipe still open, so output printed after
/// that point by whatever held it is not in the result.
fn append_drain_stopped_note(output: &mut ToolOutput) {
    let note = format!(
        "\n(a background process kept the output pipe open for {} seconds past the command's \
         exit; anything it printed after that is not included)",
        DRAIN_TIMEOUT.as_secs()
    );
    if let Some(crate::conversation::ToolResultContent::Text { text }) = output.content.last_mut() {
        text.push_str(&note);
    }
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;

    fn shared_permission_for_test() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(
            Permission::Unrestricted,
            crate::permission::EnabledPermissions::ALL,
        )
    }

    /// Every text block of a tool output, joined.
    #[cfg(target_os = "linux")]
    pub(super) fn text_of(output: &ToolOutput) -> String {
        output
            .content
            .iter()
            .filter_map(|content| {
                if let crate::conversation::ToolResultContent::Text { text } = content {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// A frontend that keeps every output delta it is handed, so a test can see what the relay
    /// sent and when.
    #[derive(Default)]
    struct ChunkRecorder {
        chunks: std::sync::Mutex<Vec<String>>,
    }

    impl ChunkRecorder {
        fn relayed(&self) -> String {
            self.chunks.lock().expect("lock").concat()
        }
    }

    #[async_trait]
    impl crate::frontend::Frontend for ChunkRecorder {
        async fn emit(&self, event: crate::frontend::FrontendEvent) {
            if let crate::frontend::FrontendEvent::ToolCallOutputDelta { chunk, .. } = event
                && let Ok(mut chunks) = self.chunks.lock()
            {
                chunks.push(chunk);
            }
        }

        async fn request_permission(
            &self,
            _request: crate::frontend::PermissionRequest,
        ) -> crate::frontend::PermissionOutcome {
            crate::frontend::PermissionOutcome::Deny
        }
    }

    /// A relay into `recorder`, tagged as one tool call.
    fn relay_into(recorder: &Arc<ChunkRecorder>) -> Option<OutputRelay> {
        let frontend: Arc<dyn crate::frontend::Frontend> = recorder.clone();
        Some(OutputRelay {
            frontend,
            tool_call_id: "call_1".to_string(),
        })
    }

    /// A drain over `reader` with the real bound and nothing telling it to stop: what a command
    /// that behaves gets.
    async fn drain_for_test<R>(reader: R, relay: Option<OutputRelay>) -> String
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        drain_output(
            Some(reader),
            relay,
            Arc::new(OutputBudget::new(MAX_OUTPUT_BYTES)),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
    }

    /// Construct an `ExecuteCommandTool` for tests with a backend probe matching whatever the host
    /// actually supports. Tests that need a specific probe state (e.g. exercising the "backend
    /// unavailable" hard-error path) should build `ExecuteCommandTool` directly with the desired
    /// `BackendProbe` rather than going through this helper.
    pub(super) fn tool_for_test(
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
    ) -> ExecuteCommandTool {
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        ExecuteCommandTool {
            #[cfg(windows)]
            windows_grants: std::sync::Arc::new(crate::sandbox::windows::WindowsGrants::default()),
            scope: crate::workspace::WriteScope::unconfined(),
            sandbox_capability,
            sandbox_backend: crate::config::SandboxBackend::Landlock,
            backend_probe,
            sandbox_enabled,
            site: crate::session::ToolSite::for_test()
                .with_permission(shared_permission)
                .with_cwd(crate::workspace::cwd_for_test()),
        }
    }

    /// With `[shell].sandbox = false`, nothing can confine a command below `unrestricted`, and that
    /// follows from the level alone, so the tool states the refusal ahead of the approval prompt in
    /// the words `execute` would use. `unrestricted` promised no boundary and has nothing to say.
    #[tokio::test]
    async fn the_shell_states_its_confinement_refusal_ahead_of_the_prompt() {
        let tool = tool_for_test(shared_permission_for_test(), false);
        let input = serde_json::json!({"command": "true"});

        let refusal = tool
            .refusal_at_level(Permission::Read, &input)
            .await
            .expect("nothing confines the command at `read`");
        assert!(refusal.is_error);
        assert!(
            refusal.text_content().contains("`[shell].sandbox = false`"),
            "{}",
            refusal.text_content()
        );
        assert!(
            tool.refusal_at_level(Permission::Unrestricted, &input)
                .await
                .is_none(),
            "`unrestricted` runs unconfined by design"
        );
    }

    /// A real signal kill and meka's own timeout kill must spell the same signal the same way; a
    /// client's terminal shows this string, and `SIG9` next to `SIGKILL` for the same event reads
    /// as two different failures.
    #[cfg(unix)]
    #[test]
    fn signal_naming_is_consistent_between_a_real_kill_and_a_timeout() {
        assert_eq!(signal_name(libc::SIGKILL), "SIGKILL");
        assert_eq!(signal_name(libc::SIGTERM), "SIGTERM");
        assert_eq!(
            signal_name(4242),
            "SIG4242",
            "unknown signals stay unambiguous"
        );

        let crate::frontend::ToolOutputMetadata::CommandExit { exit_code, signal } =
            killed_exit_metadata()
        else {
            panic!("a kill must report a command exit");
        };
        assert_eq!(
            exit_code, None,
            "a killed command never produced an exit code"
        );
        assert_eq!(
            signal.as_deref(),
            Some(signal_name(libc::SIGKILL).as_str()),
            "the timeout path must name the signal the same way the reaped-status path does",
        );
    }

    /// `shell_execute` reports its exit status structurally, not only inside the prose the model
    /// reads, so a frontend rendering a terminal can show the real code instead of guessing from
    /// the error flag.
    #[tokio::test]
    async fn execute_command_reports_its_exit_code_as_metadata() {
        let tool = tool_for_test(shared_permission_for_test(), false);
        let result = tool
            .execute(
                serde_json::json!({"command": "exit 42"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("tool runs");
        assert!(result.is_error, "a non-zero exit is a failed call");
        let Some(crate::frontend::ToolOutputMetadata::CommandExit { exit_code, signal }) =
            result.frontend_metadata
        else {
            panic!(
                "expected CommandExit metadata; got {:?}",
                result.frontend_metadata
            );
        };
        assert_eq!(exit_code, Some(42));
        assert_eq!(signal, None, "a clean exit carries no signal");
    }

    /// A read can end partway through a multi-byte character. Relaying the raw bytes would show
    /// the client replacement characters that were never in the output, so the incomplete tail has
    /// to wait for the read that completes it. Feeds the reader one byte at a time to force the
    /// split on every character.
    #[tokio::test]
    async fn reader_relays_chunks_without_splitting_characters() {
        let recorder = Arc::new(ChunkRecorder::default());
        let relay = relay_into(&recorder);

        let source = "ünïcödé ✓ done\n";
        // `tokio::io::AsyncRead` over a byte slice yields whatever the caller's buffer allows, so
        // cap reads at one byte to guarantee every multi-byte character straddles a read.
        let reader = tokio::io::AsyncReadExt::take(source.as_bytes(), u64::MAX);
        let collected = drain_for_test(OneByteAtATime(reader), relay).await;

        assert_eq!(collected, source, "the full stream must survive intact");
        let chunks = recorder.chunks.lock().expect("lock").clone();
        assert_eq!(
            chunks.concat(),
            source,
            "the relayed chunks must reassemble to the same bytes",
        );
        assert!(
            !chunks.iter().any(|chunk| chunk.contains('\u{fffd}')),
            "no chunk may contain a replacement character; got {chunks:?}",
        );
    }

    /// The relay must reassemble to the source across a stream of many reads, each carrying a
    /// partial character into the next: `relayed` is a cursor into the drain's buffer, and a cursor
    /// that drifts by one byte loses a whole read from the live view while the result stays
    /// complete, which no single-read test can show.
    #[tokio::test]
    async fn the_relay_loses_nothing_across_a_stream_of_many_reads() {
        // A 3-byte character repeated, so no power-of-two read size can land on a boundary and
        // every read carries a partial character into the next.
        let unit = "日";
        let source = unit.repeat(crate::text::MIB / unit.len() + 4096);

        let recorder = Arc::new(ChunkRecorder::default());
        let relay = relay_into(&recorder);

        let collected =
            drain_for_test(std::io::Cursor::new(source.clone().into_bytes()), relay).await;

        assert_eq!(collected, source, "the result is the whole stream");
        let chunks = recorder.chunks.lock().expect("lock").clone();
        assert_eq!(
            chunks.concat(),
            source,
            "every byte printed must reach the client",
        );
        assert!(
            !chunks.iter().any(|chunk| chunk.contains('\u{fffd}')),
            "and no chunk may be cut mid-character",
        );
    }

    /// A drain stops reading once the budget is spent, keeps every byte it read including the
    /// chunk that crossed the line, and raises the signal the spawn kills the command on. Nothing
    /// is elided: what the model gets is a prefix of what the command printed.
    #[tokio::test]
    async fn a_drain_stops_at_the_output_bound_and_keeps_what_it_read() {
        let bound = 100 * crate::text::KIB;
        let mut source = Vec::with_capacity(3 * bound);
        source.extend_from_slice(b"FIRST-LINE\n");
        source.resize(3 * bound, b'x');
        let budget = Arc::new(OutputBudget::new(bound));

        let collected = drain_output(
            Some(std::io::Cursor::new(source.clone())),
            None,
            Arc::clone(&budget),
            tokio_util::sync::CancellationToken::new(),
        )
        .await;

        assert!(
            budget.exhausted.is_cancelled(),
            "spending the budget must raise the signal"
        );
        assert!(
            collected.len() >= bound && collected.len() < source.len(),
            "the drain stops at the bound, not before and not at EOF: {} bytes",
            collected.len()
        );
        assert_eq!(
            collected.as_bytes(),
            &source[..collected.len()],
            "what was kept is a prefix of the stream"
        );
    }

    /// Both of a command's drains draw on one budget, so output split across stdout and stderr is
    /// bounded as a whole rather than at twice the bound.
    #[tokio::test]
    async fn two_drains_share_one_budget() {
        let bound = 100 * crate::text::KIB;
        let budget = Arc::new(OutputBudget::new(bound));
        // Each stream alone fits; together they do not.
        let each = vec![b'y'; bound * 3 / 4];

        let (stdout, stderr) = tokio::join!(
            drain_output(
                Some(std::io::Cursor::new(each.clone())),
                None,
                Arc::clone(&budget),
                tokio_util::sync::CancellationToken::new(),
            ),
            drain_output(
                Some(std::io::Cursor::new(each.clone())),
                None,
                Arc::clone(&budget),
                tokio_util::sync::CancellationToken::new(),
            ),
        );

        assert!(budget.exhausted.is_cancelled(), "the pair spent the budget");
        assert!(
            stdout.len() + stderr.len() < 2 * each.len(),
            "one of the drains stopped short: {} + {}",
            stdout.len(),
            stderr.len()
        );
        assert!(
            stdout.len() + stderr.len() >= bound,
            "and not before the bound was reached: {} + {}",
            stdout.len(),
            stderr.len()
        );
    }

    /// A drain told to stop returns what it has read so far rather than waiting for a pipe that
    /// something else is holding open; that is what makes a grandchild cost only its later output.
    #[tokio::test]
    async fn a_drain_told_to_stop_returns_what_it_has() {
        use tokio::io::AsyncWriteExt;
        let (mut writer, reader) = tokio::io::duplex(4096);
        let stop = tokio_util::sync::CancellationToken::new();
        let recorder = Arc::new(ChunkRecorder::default());
        let drain = tokio::spawn({
            let stop = stop.clone();
            let relay = relay_into(&recorder);
            async move {
                drain_output(
                    Some(reader),
                    relay,
                    Arc::new(OutputBudget::new(MAX_OUTPUT_BYTES)),
                    stop,
                )
                .await
            }
        });

        writer.write_all(b"partial output\n").await.expect("write");
        writer.flush().await.expect("flush");
        // The writer stays open, so EOF never comes; only the signal ends the drain. Sent once the
        // relay shows the drain has taken the bytes, so the test never races the read.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        while recorder.relayed() != "partial output\n" {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the drain must read what was written"
            );
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        stop.cancel();

        let collected = tokio::time::timeout(std::time::Duration::from_secs(2), drain)
            .await
            .expect("the drain returns promptly once told to stop")
            .expect("the drain task completes");
        assert_eq!(collected, "partial output\n");
        drop(writer);
    }

    /// The common case must be untouched: byte-for-byte what was printed.
    #[tokio::test]
    async fn a_stream_within_the_bound_is_returned_whole() {
        let source = b"just a normal amount of output\n".to_vec();
        let collected = drain_for_test(std::io::Cursor::new(source.clone()), None).await;
        assert_eq!(collected.as_bytes(), source.as_slice());
    }

    /// Reader that hands out at most one byte per `poll_read`, so every multi-byte character in
    /// the source is guaranteed to span a read boundary.
    struct OneByteAtATime<R>(R);

    impl<R: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for OneByteAtATime<R> {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            context: &mut std::task::Context<'_>,
            buffer: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            let mut single = [0u8; 1];
            let mut limited = tokio::io::ReadBuf::new(&mut single);
            match std::pin::Pin::new(&mut self.0).poll_read(context, &mut limited) {
                std::task::Poll::Ready(Ok(())) => {
                    let filled = limited.filled().to_vec();
                    buffer.put_slice(&filled);
                    std::task::Poll::Ready(Ok(()))
                }
                other => other,
            }
        }
    }

    #[tokio::test]
    async fn execute_command_runs_a_command_and_returns_its_output() {
        let tool = tool_for_test(shared_permission_for_test(), true);
        let result = tool
            .execute(
                serde_json::json!({"command": "echo hello"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert_eq!(result.text_content().trim(), "hello");
    }

    /// A command that backgrounds a long-running helper (`(sleep 30 &)`) must have that helper
    /// killed when the tool times out, not outlive the agent. The child is placed in its own
    /// process group via `setsid` so the tool can signal the whole tree via `kill(-pgid, …)`.
    #[cfg(unix)]
    #[tokio::test]
    async fn execute_command_timeout_kills_grandchild() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let marker = temp_dir.path().join("marker");
        let marker_str = marker.to_str().expect("utf-8 path").to_string();

        let tool = tool_for_test(shared_permission_for_test(), false);

        // The grandchild sleeps 3s then touches `marker`. If it survived the timeout, the marker
        // file will appear. The timeout is 300ms and we wait 5s below for a definitive "did it
        // survive?" answer.
        let script = format!("( sleep 3 && : > '{marker_str}' ) & echo backgrounded; sleep 30");
        let result = tool
            .execute(
                serde_json::json!({ "command": script, "timeout_ms": 300u64 }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("execute should not error");

        // Tool reports timeout.
        assert!(result.is_error);
        let text = result.text_content();
        assert!(text.contains("timed out"), "got: {text:?}");

        // Wait well past the grandchild's sleep-3s. If the marker materializes, the grandchild
        // wasn't killed; the bug is back.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        assert!(
            !marker.exists(),
            "grandchild survived timeout and created marker at {marker:?}"
        );
    }

    /// A command that outgrows the bound is stopped, and everything it printed up to the stop is in
    /// the result: the record is complete, the model is told why the command ended and how much it
    /// printed, and the kill is reported the way the timeout's is.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_command_that_prints_past_the_bound_is_stopped_and_its_output_kept() {
        let tool = tool_for_test(shared_permission_for_test(), false);
        // Past the bound by a margin, with a recognizable first line, so the result can be checked
        // for both ends of what it must hold.
        let past_the_bound = MAX_OUTPUT_BYTES + 4 * crate::text::MIB;
        let result = tool
            .execute(
                serde_json::json!({
                    "command": format!(
                        "echo FIRST-LINE; head -c {past_the_bound} /dev/zero | tr '\\0' x; \
                         echo NEVER-REACHED"
                    ),
                    "timeout_ms": 60_000u64,
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a stopped command is a result");

        assert!(result.is_error, "a stopped command is a failed call");
        let text = result.text_content();
        assert!(
            text.starts_with(&format!("{}\nFIRST-LINE\n", output_bound_reason())),
            "the reason leads and what the command printed follows: {text:.120}"
        );
        assert!(
            text.len() >= MAX_OUTPUT_BYTES,
            "everything read up to the bound is kept, got {} bytes",
            text.len()
        );
        assert!(
            !text.contains("NEVER-REACHED"),
            "the command was stopped rather than run to its end"
        );
        let Some(crate::frontend::ToolOutputMetadata::CommandExit { exit_code, signal }) =
            result.frontend_metadata
        else {
            panic!("expected CommandExit metadata");
        };
        assert_eq!(exit_code, None, "a killed command has no exit code");
        assert_eq!(signal.as_deref(), Some("SIGKILL"));
    }

    /// A child exits once its last write fits in the pipe, so it can finish within a pipe's worth
    /// of the bound and leave the crossing to the drains' collection. That result must not read as
    /// a clean exit: it leads with the cut and is an error, whatever the exit status says.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_command_whose_drains_reach_the_bound_after_it_exits_is_not_reported_clean() {
        use std::os::unix::process::ExitStatusExt;
        let budget = OutputBudget::new(16);
        assert!(!budget.spend(32), "the budget is spent past its bound");
        let stdout_task = tokio::spawn(async { "printed before the cut\n".to_string() });
        let stderr_task = tokio::spawn(async { String::new() });

        let result = finish_command(
            Ok(std::process::ExitStatus::from_raw(0)),
            stdout_task,
            stderr_task,
            &tokio_util::sync::CancellationToken::new(),
            &budget,
        )
        .await
        .expect("a result");

        assert!(result.is_error, "a cut transcript is never a clean result");
        let text = result.text_content();
        assert!(
            text.starts_with(&format!("{}\nprinted before the cut\n", output_cut_note())),
            "the cut leads and the output follows: {text}"
        );
        let Some(crate::frontend::ToolOutputMetadata::CommandExit { exit_code, signal }) =
            result.frontend_metadata
        else {
            panic!("expected CommandExit metadata");
        };
        assert_eq!(exit_code, Some(0), "the real exit status is still reported");
        assert_eq!(signal, None);
    }

    /// End to end, a command whose output ends within a pipe's worth past the bound may exit
    /// before or after the drains cross it, so either the bound arm or the exit arm can win.
    /// Whichever does, the result is an error that leads with the bound.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_command_ending_just_past_the_bound_is_never_reported_clean() {
        let tool = tool_for_test(shared_permission_for_test(), false);
        let just_past = MAX_OUTPUT_BYTES + 100;
        let result = tool
            .execute(
                serde_json::json!({
                    "command": format!("head -c {just_past} /dev/zero | tr '\\0' x"),
                    "timeout_ms": 60_000u64,
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a result");

        assert!(
            result.is_error,
            "output past the bound is never a clean result"
        );
        let text = result.text_content();
        assert!(
            text.starts_with(&output_bound_reason()) || text.starts_with(&output_cut_note()),
            "the result leads with the bound either way: {text:.120}"
        );
        assert!(
            text.len() >= MAX_OUTPUT_BYTES,
            "everything read up to the bound is kept, got {} bytes",
            text.len()
        );
    }

    /// A command killed at its timeout keeps what it printed before the kill. A build that logged
    /// for twenty-nine seconds and died at thirty used to come back as the timeout line alone.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_timed_out_command_keeps_the_output_it_printed() {
        let tool = tool_for_test(shared_permission_for_test(), false);
        let result = tool
            .execute(
                serde_json::json!({
                    "command": "echo before-the-stall; echo on-stderr >&2; sleep 30",
                    "timeout_ms": 300u64,
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a timeout is a result");

        assert!(result.is_error);
        let text = result.text_content();
        let reason = timed_out_reason(
            std::time::Duration::from_millis(300),
            &OutputBudget::new(MAX_OUTPUT_BYTES),
        );
        assert!(
            text.starts_with(&format!("{reason}\nbefore-the-stall\n")),
            "the timeout is stated first and stdout printed before the kill follows: {text}"
        );
        assert!(
            text.ends_with("--- stderr ---\non-stderr\n"),
            "and so does stderr: {text}"
        );
    }

    /// A grandchild that keeps the pipe open past the command's exit delays the result by the drain
    /// timeout and costs only what it prints afterwards; what the command itself printed is kept,
    /// and the result says why the pipe was abandoned.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_pipe_held_open_by_a_grandchild_does_not_lose_the_output_already_read() {
        let tool = tool_for_test(shared_permission_for_test(), false);
        // The shell exits at once; the backgrounded `sleep` inherits stdout and holds it open for
        // longer than the drain timeout, then ends on its own.
        let hold = DRAIN_TIMEOUT.as_secs() + 3;
        let started = std::time::Instant::now();
        let result = tool
            .execute(
                serde_json::json!({
                    "command": format!("echo kept; sleep {hold} &"),
                    "timeout_ms": 60_000u64,
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("the command itself succeeds");

        assert!(
            started.elapsed() < std::time::Duration::from_secs(hold),
            "the result arrives at the drain timeout, not when the grandchild lets go"
        );
        assert!(!result.is_error, "the command exited 0");
        let text = result.text_content();
        assert!(
            text.starts_with("kept\n"),
            "the output read before the pipe was abandoned is kept: {text}"
        );
        assert!(
            text.contains("kept the output pipe open"),
            "and the result says the drain stopped early: {text}"
        );
    }

    #[tokio::test]
    async fn execute_command_failure() {
        let tool = tool_for_test(shared_permission_for_test(), true);
        let result = tool
            .execute(
                serde_json::json!({"command": "false"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        assert!(result.is_error);
    }

    /// `workspace` refuses the shell outright when `[shell].sandbox = false` leaves nothing to
    /// confine it with.
    ///
    /// The failure this guards is silent and one-sided: the config key unconfines the shell while
    /// the file tools stay fenced, so `workspace` keeps reporting a boundary it is only half
    /// holding. `read` never had the problem, because `Read.allows(Unrestricted)` is false and the
    /// tool simply disappears; `Workspace.allows` is true for everything, so the refusal has to be
    /// here.
    #[tokio::test]
    async fn workspace_refuses_the_shell_when_the_sandbox_is_disabled() {
        // Every level that promises confinement, not just `workspace`: narrowed to `workspace`
        // alone, `[tools.tool_permissions] shell_execute = "read"` plus `[shell].sandbox = false`
        // would run a plain `sh -c` at `read`, with the full parent environment since the scrub is
        // gated on the same flag.
        for level in [Permission::None, Permission::Read, Permission::Workspace] {
            refuses_at(level).await;
        }
    }

    async fn refuses_at(level: Permission) {
        let workspace_perm = crate::permission::SharedPermission::new(
            level,
            crate::permission::EnabledPermissions::ALL,
        );
        let tool = ExecuteCommandTool {
            #[cfg(windows)]
            windows_grants: std::sync::Arc::clone(crate::sandbox::windows::process_grants()),
            scope: crate::workspace::WriteScope::confined(vec![]),
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Bubblewrap,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "sandbox disabled in config".to_string(),
            },
            sandbox_enabled: false,
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_permission(workspace_perm),
        };
        let result = tool
            .execute(
                serde_json::json!({"command": "echo nope"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        match result {
            Err(MekaError::ToolExecution { tool_name, message }) => {
                assert_eq!(tool_name, "shell_execute");
                assert!(
                    message.contains("[shell].sandbox = false"),
                    "the refusal must name the key responsible: {message}"
                );
            }
            other => {
                panic!("expected a hard error at {level} with no sandbox, got {other:?}")
            }
        }
    }

    /// When the configured sandbox backend isn't usable, `shell_execute` at `read` must return
    /// `Err(MekaError::ToolExecution)`, *not* `Ok(ToolOutput { is_error: true })`. The hard error
    /// path is how the model is forced to surface the failure to the user rather than just retrying
    /// or describing it as a tool result.
    #[tokio::test]
    async fn execute_command_hard_errors_when_backend_unavailable() {
        let read_only_perm = crate::permission::SharedPermission::new(
            Permission::Read,
            crate::permission::EnabledPermissions::ALL,
        );
        let tool = ExecuteCommandTool {
            #[cfg(windows)]
            windows_grants: std::sync::Arc::new(crate::sandbox::windows::WindowsGrants::default()),
            scope: crate::workspace::WriteScope::unconfined(),
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Bubblewrap,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "bwrap not found on PATH".to_string(),
            },
            sandbox_enabled: true,
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_permission(read_only_perm),
        };
        let result = tool
            .execute(
                serde_json::json!({"command": "echo nope"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        match result {
            Err(MekaError::ToolExecution { tool_name, message }) => {
                assert_eq!(tool_name, "shell_execute");
                // The Linux error path splices in the configured backend's display name
                // (`Bubblewrap`); the non-Linux variant drops the Linux-specific config reference
                // and reads "sandbox is unavailable: ...". Both must include the probe reason
                // verbatim.
                #[cfg(target_os = "linux")]
                assert!(
                    message.contains("Bubblewrap"),
                    "expected backend display name in error: {message}"
                );
                assert!(
                    message.contains("bwrap not found on PATH"),
                    "expected probe reason in error: {message}"
                );
            }
            Err(other) => panic!("expected ToolExecution, got {other:?}"),
            Ok(output) => panic!("expected hard error, got Ok({:?})", output.text_content()),
        }
    }

    /// At `unrestricted` an unavailable sandbox backend must NOT short-circuit the spawn: that
    /// level promises no boundary, so there is nothing for a missing backend to fail to provide.
    /// `workspace` and `read` are the opposite case and are refused outright, which is what makes
    /// this arm worth pinning separately.
    #[tokio::test]
    async fn execute_command_runs_without_sandbox_when_unrestricted() {
        let unrestricted_perm = crate::permission::SharedPermission::new(
            Permission::Unrestricted,
            crate::permission::EnabledPermissions::ALL,
        );
        let tool = ExecuteCommandTool {
            #[cfg(windows)]
            windows_grants: std::sync::Arc::new(crate::sandbox::windows::WindowsGrants::default()),
            scope: crate::workspace::WriteScope::unconfined(),
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Bubblewrap,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "bwrap not found on PATH".to_string(),
            },
            sandbox_enabled: true,
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_permission(unrestricted_perm),
        };
        let result = tool
            .execute(
                serde_json::json!({"command": "echo hello"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should succeed at unrestricted");
        assert!(!result.is_error);
        assert_eq!(result.text_content().trim(), "hello");
    }

    #[tokio::test]
    async fn execute_command_large_output_not_truncated() {
        // Output well over any plausible inline cap: the tool must return it in full. The agent
        // layer handles oversize downstream.
        let tool = tool_for_test(shared_permission_for_test(), true);
        let result = tool
            .execute(
                // 50 000 "x" characters, in each host shell's own vocabulary. The Unix spelling
                // is POSIX-portable (`head` and `tr` rather than bash brace expansion, so it
                // works under `dash`) and the Windows one is PowerShell, which is the shell
                // `shell_execute` actually invokes there.
                serde_json::json!({
                    "command": if cfg!(windows) {
                        "Write-Output ('x' * 50000)"
                    } else {
                        "head -c 50000 /dev/zero | tr '\\0' x"
                    }
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        let text = result.text_content();
        assert!(
            !text.contains("(output truncated"),
            "no truncation marker expected, got: {text:.200}..."
        );
        assert!(
            text.trim().len() >= 50_000,
            "expected >= 50 000 chars, got {}",
            text.trim().len()
        );
    }

    /// A command writing far more than the OS pipe buffer (~64 KiB on Linux) must complete without
    /// blocking: unless stdout/stderr are drained on dedicated tasks that start before
    /// `child.wait()`, the child blocks in `write()`, `wait()` never returns, and the call hits a
    /// spurious timeout with truncated output.
    #[cfg(unix)]
    #[tokio::test]
    async fn execute_command_large_output_no_deadlock() {
        let tool = tool_for_test(shared_permission_for_test(), true);
        // 5 MiB of 'x', two orders of magnitude past any pipe buffer.
        let result = tool
            .execute(
                serde_json::json!({
                    "command": "head -c 5242880 /dev/zero | tr '\\0' x"
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error, "large output spuriously flagged as error");
        let text = result.text_content();
        assert!(
            !text.contains("kept the output pipe open"),
            "unexpected drain note: {text:.200}"
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

        /// Where the process-read probe's parent leaves the target for its child.
        ///
        /// A file in the child's working directory rather than an environment variable, because
        /// the sandboxed spawn path hands the child a *curated* environment on purpose and a
        /// probe-specific variable has no business being added to that allow-list.
        const PROBE_HANDOFF: &str = "probe-target.txt";

        /// Marks the probe child's one line of output, so the parent can tell a real verdict from
        /// a child that never ran. Without it a host where the re-entry silently failed would read
        /// as "no read happened", which is the same shape as success. It earned its keep on the
        /// first hardware run, catching a filter that selected zero tests.
        const PROBE_VERDICT_PREFIX: &str = "MEKA-PROBE-VERDICT ";

        /// The probe child's libtest name, which `--exact` needs in full.
        ///
        /// Derived from [`module_path!`] rather than written out, so moving the module cannot leave
        /// a filter that silently matches nothing. `module_path!` is crate-qualified and libtest's
        /// names are not, so the leading crate segment comes off.
        fn probe_child_test_name() -> String {
            let module = module_path!()
                .split_once("::")
                .map_or(module_path!(), |(_crate_name, rest)| rest);
            format!("{module}::windows_process_read_probe_child")
        }

        fn read_permission() -> crate::permission::SharedPermission {
            crate::permission::SharedPermission::new(
                Permission::Read,
                crate::permission::EnabledPermissions::ALL,
            )
        }

        /// Build an `ExecuteCommandTool` for the Low-integrity Windows path. Mirrors
        /// `super::tool_for_test` (which always calls `sandbox::detect()` and would resolve to
        /// `LowIntegrity` on Windows anyway) but constructs the fields explicitly so the tests
        /// document the intended state.
        fn windows_test_tool(
            shared_permission: crate::permission::SharedPermission,
        ) -> ExecuteCommandTool {
            let sandbox_capability = crate::sandbox::SandboxCapability::LowIntegrity;
            let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
            ExecuteCommandTool {
                scope: crate::workspace::WriteScope::unconfined(),
                windows_grants: std::sync::Arc::new(
                    crate::sandbox::windows::WindowsGrants::default(),
                ),
                sandbox_capability,
                // `sandbox_backend` is Linux-only metadata; on Windows the value is never read but
                // the field must still be populated. `Landlock` is the conventional placeholder.
                sandbox_backend: crate::config::SandboxBackend::Landlock,
                backend_probe,
                sandbox_enabled: true,
                site: crate::session::ToolSite::for_test()
                    .with_permission(shared_permission)
                    .with_cwd(crate::workspace::cwd_for_test()),
            }
        }

        /// The `workspace` restricted-token path, end to end, on real Windows.
        ///
        /// This is the whole Windows half of the level and nothing in the suite reached it: the two
        /// tests below drive the Low-integrity path, which is a different token, a different set of
        /// spawn flags, and no ACE at all. Everything specific to `workspace` -- the
        /// `WRITE_RESTRICTED` token, the synthesized capability SID, the inheritable ACE, the
        /// console a restricted child must inherit rather than create, and `lpCurrentDirectory` --
        /// only runs here.
        ///
        /// Both directions are checked in one command, because a token that denies *everything*
        /// would pass a refused-outside test on its own.
        #[tokio::test]
        async fn a_workspace_shell_writes_inside_the_root_and_is_refused_outside() {
            let temp = tempfile::tempdir().expect("tempdir");
            let base = crate::workspace::canonical_for_test(temp.path());
            let work = base.join("work");
            let outside = base.join("outside");
            std::fs::create_dir(&work).expect("work");
            std::fs::create_dir(&outside).expect("outside");

            let mut tool = windows_test_tool(crate::permission::SharedPermission::new(
                Permission::Workspace,
                crate::permission::EnabledPermissions::ALL,
            ));
            tool.site.cwd = crate::workspace::SharedCwd::new(work.clone());
            tool.scope = crate::workspace::WriteScope::confined(vec![work.clone()]);

            let result = tool
                .execute(
                    serde_json::json!({
                        "command": format!(
                            "Set-Content -Path '{}\\inside.txt' -Value 'in'; \
                             Set-Content -Path '{}\\escaped.txt' -Value 'out'",
                            work.display(),
                            outside.display()
                        ),
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("execute should not error");

            // Ground truth on disk, not the tool's narration: a shell that never started would
            // report failure just as convincingly as one the ACE confined.
            assert!(
                work.join("inside.txt").exists(),
                "a write inside the granted root must land, but did not: {result:?}"
            );
            assert!(
                !outside.join("escaped.txt").exists(),
                "a write outside every root must be refused by the token, not by meka: {result:?}"
            );

            tool.windows_grants.revoke_all();
        }

        /// The probe half of [`a_confined_child_reads_our_memory_at_workspace_and_cannot_at_read`].
        ///
        /// Re-entered as a child process rather than shelled out to, because the natural scripted
        /// probe cannot be trusted here: at `workspace` the `WRITE_RESTRICTED` token puts
        /// PowerShell into ConstrainedLanguage mode, where constructing the .NET types such a probe
        /// needs fails outright, and that failure is indistinguishable from a denied handle. This
        /// ran as a hand-cross-compiled C binary until the staging step became the only thing
        /// keeping the parent `#[ignore]`d.
        ///
        /// Stays ignored so a plain suite run never selects it, and no-ops when the handoff file is
        /// absent so selecting it by hand does nothing either. The parent passes the target through
        /// that file rather than the environment, because the spawn path hands the child a curated
        /// environment by design.
        #[tokio::test]
        #[ignore = "re-entered by its parent test; does nothing on its own"]
        async fn windows_process_read_probe_child() {
            let Ok(handoff) = std::fs::read_to_string(PROBE_HANDOFF) else {
                return;
            };
            let mut parts = handoff.split_whitespace();
            let (Some(pid), Some(address)) = (parts.next(), parts.next()) else {
                return;
            };
            let (Ok(pid), Ok(address)) = (pid.parse::<u32>(), usize::from_str_radix(address, 16))
            else {
                return;
            };

            use windows_sys::Win32::System::{
                Diagnostics::Debug::ReadProcessMemory,
                Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ},
            };

            // SAFETY: `pid` names a live process (the parent, which is blocked awaiting this
            // child), and the buffer is sized from itself. A denied handle comes back null and is
            // reported rather than dereferenced.
            let verdict = unsafe {
                let handle = OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, 0, pid);
                if handle.is_null() {
                    format!(
                        "OPEN_FAILED {}",
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                    )
                } else {
                    let mut buffer = [0u8; 64];
                    let mut read = 0usize;
                    let ok = ReadProcessMemory(
                        handle,
                        address as *const std::ffi::c_void,
                        buffer.as_mut_ptr().cast(),
                        buffer.len(),
                        &mut read,
                    );
                    windows_sys::Win32::Foundation::CloseHandle(handle);
                    if ok == 0 {
                        format!(
                            "READ_FAILED {}",
                            std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                        )
                    } else {
                        let text: String = buffer[..read]
                            .iter()
                            .copied()
                            .take_while(|byte| byte.is_ascii_graphic() || *byte == b' ')
                            .map(char::from)
                            .collect();
                        format!("READ_OK {text}")
                    }
                }
            };
            println!("{PROBE_VERDICT_PREFIX}{verdict}");
        }

        /// A `workspace` child can read meka's own process memory; a `read` child cannot.
        ///
        /// The two levels are compared in one test because either result alone is uninformative: a
        /// token that denied everything would pass a "refused" assertion on its own merits, and a
        /// host where the probe simply failed to run would look like confinement.
        ///
        /// **The `read` leg is the guard.** Low integrity has to deny `OpenProcess` against meka,
        /// and nothing else in the suite defends that. **The `workspace` leg is a tripwire on the
        /// documentation.** It asserts the measured weakness still exists, so hardening the spawn
        /// path fails this test and forces `docs/book/src/usage/permissions.md` to be corrected
        /// rather than left quietly wrong in the safe direction.
        ///
        /// Why the weakness exists: `WRITE_RESTRICTED` intersects the restricting SIDs for write
        /// access only, and the `workspace` path deliberately leaves the integrity label alone so
        /// ordinary tooling keeps working. Neither restricts a *read*, and meka's memory holds
        /// provider credentials. Measured on hardware 2026-08-24: `workspace` returned the canary,
        /// `read` returned `ERROR_ACCESS_DENIED`.
        ///
        /// What this does **not** show is that an attacker could locate those credentials unaided.
        /// The parent hands the child the exact address, so this measures the capability -- a
        /// readable handle plus a successful read -- and not a search.
        #[tokio::test]
        async fn a_confined_child_reads_our_memory_at_workspace_and_cannot_at_read() {
            // Leaked so the marker stays mapped for as long as the child needs to read it.
            let secret: &'static str =
                Box::leak("MEKA-CANARY-7f3a91d4c8e2".to_string().into_boxed_str());
            let temp = tempfile::tempdir().expect("tempdir");
            let work = crate::workspace::canonical_for_test(temp.path());
            let runner = std::env::current_exe().expect("the test binary's own path");

            for (label, permission) in [
                ("workspace", Permission::Workspace),
                ("read", Permission::Read),
            ] {
                // Written fresh per leg, and inside the child's working directory, which is the
                // one path both confinements agree the child can reach.
                std::fs::write(
                    work.join(PROBE_HANDOFF),
                    format!("{} {:x}", std::process::id(), secret.as_ptr() as usize),
                )
                .expect("hand the target to the child");

                let mut tool = windows_test_tool(crate::permission::SharedPermission::new(
                    permission,
                    crate::permission::EnabledPermissions::ALL,
                ));
                tool.site.cwd = crate::workspace::SharedCwd::new(work.clone());
                tool.scope = crate::workspace::WriteScope::confined(vec![work.clone()]);

                let result = tool
                    .execute(
                        serde_json::json!({
                            "command": format!(
                                "& '{}' {} --ignored --exact --nocapture",
                                runner.display(),
                                probe_child_test_name()
                            ),
                        }),
                        crate::tools::ToolContext::detached(CancellationToken::new()),
                    )
                    .await
                    .expect("execute should not error");

                let reported = format!("{result:?}");
                tool.windows_grants.revoke_all();
                assert!(
                    reported.contains(PROBE_VERDICT_PREFIX),
                    "the probe child did not report a verdict, so nothing below is meaningful: \
                     {reported}"
                );

                match permission {
                    // The canary itself, not just `READ_OK`: a read that succeeded against the
                    // wrong page and returned zeroes would satisfy the weaker check.
                    Permission::Workspace => assert!(
                        reported.contains("READ_OK") && reported.contains(secret),
                        "the `{label}` child could not read our memory. If the spawn path was \
                         hardened this is the good outcome, but permissions.md still documents \
                         the old one: {reported}"
                    ),
                    _ => assert!(
                        reported.contains("OPEN_FAILED"),
                        "`{label}` runs at Low integrity, which must refuse OpenProcess against \
                         meka: {reported}"
                    ),
                }
            }
        }

        /// Under Low integrity, writing to the user's profile directory must be denied by the OS.
        /// The test probes a path under `%USERPROFILE%` and asserts the file is never created.
        #[tokio::test]
        async fn windows_sandbox_blocks_write_to_userprofile() {
            let probe_path = format!(
                "{}\\meka-sandbox-probe.txt",
                std::env::var("USERPROFILE").expect("USERPROFILE must be set on Windows")
            );
            // Clean any stray file from an earlier failed run before starting.
            let _ = std::fs::remove_file(&probe_path);

            let tool = windows_test_tool(read_permission());
            let _ = tool
                .execute(
                    serde_json::json!({
                        "command": format!("echo hello > \"{probe_path}\""),
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("execute should not error");

            let existed = std::path::Path::new(&probe_path).exists();
            // Defensive cleanup even if the assertion below fails.
            let _ = std::fs::remove_file(&probe_path);
            assert!(
                !existed,
                "Low-integrity sandbox should have blocked write to {probe_path}"
            );
        }

        /// A command that produces well over the default Windows pipe buffer (~4 KB) of output must
        /// complete without deadlocking and without truncation. Before the concurrent-drain fix,
        /// the child would block in `WriteFile` past the buffer, the wait would never return, and
        /// the tool would report a spurious timeout.
        #[tokio::test]
        async fn windows_sandbox_large_output_under_sandbox() {
            let tool = windows_test_tool(read_permission());
            // PowerShell builds a 262144-char string in memory then emits it as one line. Total
            // output is ~256 KB, well past any plausible pipe buffer.
            let result = tool
                .execute(
                    serde_json::json!({
                        "command": "'x' * 262144",
                        "timeout_ms": 60000u64,
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("execute should not error");

            assert!(
                !result.is_error,
                "large-output command should not be flagged as an error"
            );
            let text = result.text_content();
            let x_count = text.matches('x').count();
            assert!(
                x_count >= 262144,
                "expected >= 262144 'x' characters in output, got {x_count}"
            );
        }

        /// The child's stdin must be connected to `NUL`, not inherited from the agent's TTY and not
        /// left as an invalid handle. `$input` enumerates pipeline input; piped from NUL it yields
        /// zero objects. The command must complete promptly rather than hanging on a dangling
        /// stdin.
        #[tokio::test]
        async fn windows_sandbox_stdin_is_null() {
            let tool = windows_test_tool(read_permission());
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                tool.execute(
                    serde_json::json!({
                        "command": "($input | Measure-Object).Count",
                        "timeout_ms": 5000u64,
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                ),
            )
            .await
            .expect("command must not hang waiting for stdin")
            .expect("execute should not error");

            assert!(!result.is_error);
            let text = result.text_content();
            assert!(
                text.trim().starts_with('0'),
                "expected stdin-object count of 0, got {text:?}"
            );
        }

        /// Round-trip a grab-bag of tricky marker strings through PowerShell to confirm
        /// `quote_command_arg` + PowerShell's argv parser are inverses. Uses PowerShell
        /// single-quote literals internally so the test exercises our command-line encoding, not PS
        /// string rules.
        #[tokio::test]
        async fn windows_sandbox_quoting_roundtrip() {
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
                        crate::tools::ToolContext::detached(CancellationToken::new()),
                    )
                    .await
                    .expect("execute should not error");
                assert!(!result.is_error, "command for marker {marker:?} errored");
                let text = result.text_content();
                assert!(
                    text.contains(marker),
                    "marker {marker:?} missing from output {text:?}"
                );
            }
        }

        /// Regression test for the parent-env-inheritance leak: secrets set in the parent (API
        /// keys, OAuth tokens) must not appear in the sandboxed child's environment, because a
        /// Low-integrity child can still open outbound sockets and exfiltrate them.
        #[tokio::test]
        async fn windows_sandbox_scrubs_provider_api_keys() {
            // SAFETY: tests run under `cargo test`, which is single-threaded per target by default
            // for integration tests, and this env var is scoped to the test's probe command.
            // Acceptable for a test.
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
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("execute should not error");

            unsafe {
                std::env::remove_var("ANTHROPIC_API_KEY");
            }

            let text = result.text_content();
            assert!(
                !text.contains("probe-12345-leaked"),
                "parent API key leaked into sandboxed child env: {text:?}"
            );
        }

        /// Reads must still succeed under Low integrity. The hosts file is readable by Everyone on
        /// stock Windows, so it's a good probe.
        #[tokio::test]
        async fn windows_sandbox_allows_read() {
            let tool = windows_test_tool(read_permission());
            let result = tool
                .execute(
                    serde_json::json!({
                        "command": "type C:\\Windows\\System32\\drivers\\etc\\hosts",
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("execute should not error");

            assert!(
                !result.is_error,
                "reading %WINDIR%\\System32\\drivers\\etc\\hosts should succeed under Low integrity"
            );
        }
    }

    /// The Bubblewrap workspace binding, exercised against a real `bwrap`.
    ///
    /// The Landlock dialect had a live confinement test and Bubblewrap had none, which left the
    /// ordering rule in `bwrap_args` unguarded: bind before the tmpfs masks and every workspace
    /// under `/tmp` silently comes out read-only, with bwrap reporting success. Both halves are
    /// checked here, the order in the argument list and the outcome on disk, because the first
    /// is what a future edit would break and the second is what a user would feel.
    #[cfg(target_os = "linux")]
    mod bubblewrap_boundary {
        use std::path::PathBuf;

        /// The level `shell_execute` needs depends on whether a sandbox can actually confine it.
        ///
        /// `read` only when the sandbox is both enabled *and* backed by a working backend;
        /// otherwise `unrestricted`, because a command meka cannot confine is a command only the
        /// boundary-free level may authorize. The conjunction is the whole rule: flipped to `||`,
        /// the tool would be offered at `read` with `[shell].sandbox = false`, or with the sandbox
        /// on but no usable backend. The runtime guard in `execute` still refuses the command in
        /// both cases, so this is a wrong catalog entry rather than an escape, but the catalog is
        /// what the model plans against.
        #[test]
        fn the_shell_needs_unrestricted_whenever_nothing_can_confine_it() {
            use crate::{permission::Permission, sandbox::SandboxCapability, tools::Tool};

            let check =
                |sandbox_enabled: bool, capability: SandboxCapability, expected: Permission| {
                    let mut tool = super::tool_for_test(
                        crate::permission::SharedPermission::new(
                            Permission::Read,
                            crate::permission::EnabledPermissions::ALL,
                        ),
                        sandbox_enabled,
                    );
                    tool.sandbox_capability = capability.clone();
                    assert_eq!(
                        tool.required_permission(),
                        expected,
                        "sandbox_enabled={sandbox_enabled}, capability={capability:?}"
                    );
                };

            // Nothing can confine: either meka was told not to, or the host offers no backend.
            check(
                true,
                SandboxCapability::Unavailable,
                Permission::Unrestricted,
            );
            check(
                false,
                SandboxCapability::Unavailable,
                Permission::Unrestricted,
            );

            // Which backend is available does not enter the rule, and the variants are
            // per-platform, so the positive leg uses whatever this host actually has.
            // Skipped rather than faked where there is none: an invented variant would
            // assert against a state meka cannot reach here.
            let available = crate::sandbox::detect();
            if matches!(available, SandboxCapability::Unavailable) {
                eprintln!("skipping the confined leg: no sandbox backend on this host");
                return;
            }
            check(true, available.clone(), Permission::Read);
            check(false, available, Permission::Unrestricted);
        }

        /// A cwd that *is* a masked directory must not be bound back over its own mask.
        ///
        /// The cwd bind and the tmpfs masks obey the same rule (last mount wins), so an
        /// unconditional bind hands a masked-directory session the host directory: a session at
        /// `/tmp` could `connect()` the tmux socket, one at `$XDG_RUNTIME_DIR` the session bus, and
        /// one at `/` would see the host's PIDs, defeating `--unshare-pid` as well. A read-only
        /// bind does not help, because `connect(2)` on a socket inode is not a write.
        ///
        /// `/` is in the table because it is systemd's default working directory for a daemon, so
        /// `meka serve` under a unit file lands there without anyone choosing it.
        ///
        /// The counterpart is `the_child_is_given_a_working_directory_it_can_reach`: a path merely
        /// *under* a mask still needs its bind, and still gets one.
        #[test]
        fn a_masked_working_directory_is_not_bound_back_over_its_own_mask() {
            let masked = [
                std::path::PathBuf::from("/tmp"),
                std::path::PathBuf::from("/var/tmp"),
                std::path::PathBuf::from("/run"),
                std::path::PathBuf::from("/"),
            ];
            for cwd in masked {
                let text: Vec<String> = super::bwrap_args(&[], &cwd, &[])
                    .iter()
                    .map(|arg| arg.to_string_lossy().into_owned())
                    .collect();

                // The bind is what undoes the mask, so its absence is the property. Checked as an
                // adjacent pair rather than by searching for the path alone, because `/tmp` also
                // appears as a `--tmpfs` operand and `/` as the `--ro-bind / /` operand.
                let rebound = text.windows(3).any(|window| {
                    window[0] == "--ro-bind-try"
                        && window[1] == cwd.to_string_lossy()
                        && window[2] == cwd.to_string_lossy()
                });
                assert!(
                    !rebound,
                    "binding {} back over its own mask restores the host directory the mask hides, \
                     which is the sandbox escape `is_system_root` exists to prevent: {text:?}",
                    cwd.display()
                );

                // And the child still has somewhere to stand: the mask leaves an empty tmpfs at
                // that path, so `--chdir` succeeds and nothing is silently relocated to `$HOME`.
                let chdir = text
                    .iter()
                    .position(|arg| arg == "--chdir")
                    .expect("the cwd must still be requested explicitly");
                assert_eq!(
                    text.get(chdir + 1).map(String::as_str),
                    Some(cwd.to_string_lossy().as_ref()),
                    "the masked cwd is still where the child starts"
                );
            }
        }

        /// The child is told which directory to start in, and can read it.
        ///
        /// bwrap's fallback when it cannot enter the pre-`execve` cwd is silent and lands the child
        /// in `$HOME`: without these two arguments `pwd` reports the user's home directory and the
        /// workspace is unreachable even by absolute path, with exit 0 and empty stderr; with them
        /// `pwd` is correct, the file reads, and a write is still refused read-only at `read`.
        ///
        /// The bind sits after the masks and before the writable binds, so a cwd under `/tmp` is
        /// restored, and a cwd that is also a writable root is upgraded to read-write by the loop
        /// that follows: last mount wins.
        #[test]
        #[cfg(target_os = "linux")]
        fn the_child_is_given_a_working_directory_it_can_reach() {
            let cwd = PathBuf::from("/tmp/session-cwd");
            let args = super::bwrap_args(&[], &cwd, &[]);
            let text: Vec<String> = args
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect();

            let chdir = text.iter().position(|arg| arg == "--chdir");
            assert!(
                chdir.is_some(),
                "the cwd must be requested explicitly: {text:?}"
            );
            assert_eq!(
                text.get(chdir.expect("checked above") + 1)
                    .map(String::as_str),
                Some("/tmp/session-cwd")
            );

            let bind = text
                .iter()
                .position(|arg| arg == "--ro-bind-try")
                .expect("the cwd must be bound back in, or `read` cannot see it");
            let last_mask = text
                .iter()
                .rposition(|arg| arg == "--tmpfs")
                .expect("the masks are always present");
            assert!(
                bind > last_mask,
                "a cwd bound before the masks is undone by them: {text:?}"
            );

            // A cwd that is also a writable root ends up read-write, because the rw bind comes
            // later.
            let rw = super::bwrap_args(std::slice::from_ref(&cwd), &cwd, &[]);
            let rw: Vec<String> = rw
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect();
            let ro_at = rw.iter().position(|arg| arg == "--ro-bind-try");
            let rw_at = rw.iter().position(|arg| arg == "--bind-try");
            assert!(
                ro_at < rw_at,
                "the writable bind must win over the read-only one: {rw:?}"
            );
        }

        #[test]
        fn every_workspace_bind_comes_after_every_mask() {
            let args = super::bwrap_args(
                &[PathBuf::from("/tmp/work")],
                std::path::Path::new("/tmp/work"),
                &[],
            );
            let last_mask = args
                .iter()
                .rposition(|arg| arg == "--tmpfs")
                .expect("the masks are part of the recipe");
            let bind = args
                .iter()
                .position(|arg| arg == "--bind-try")
                .expect("the workspace root is bound");
            assert!(
                bind > last_mask,
                "a bind before a mask is undone by it, silently: {args:?}"
            );
        }

        #[test]
        fn a_bubblewrapped_shell_writes_inside_the_root_and_is_refused_outside() {
            let Some(bwrap) = which_bwrap() else {
                // Not `#[ignore]`: this must run wherever bwrap exists, and skipping loudly beats a
                // test that silently never runs on the machines that have the backend.
                eprintln!("skipping: bwrap is not on PATH");
                return;
            };

            let temp = tempfile::tempdir().expect("tempdir");
            let base = crate::workspace::canonical_for_test(temp.path());
            let work = base.join("work");
            std::fs::create_dir(&work).expect("work");

            // The "outside" target lives in `$HOME`, not beside the workspace.
            //
            // A sibling under the tempdir is itself under `/tmp`, which the recipe masks with a
            // tmpfs, so a write there fails with ENOENT: the directory does not exist inside the
            // namespace at all. That is not the boundary refusing anything, and it would keep
            // passing with the boundary removed. `$HOME` is present and writable outside the
            // sandbox, so a refusal there is the ruleset's doing.
            let outside = crate::workspace::canonical_for_test(
                std::env::var_os("HOME")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir),
            )
            .join(format!("meka-bwrap-probe-{}", uuid::Uuid::new_v4()));
            assert!(!outside.exists(), "the probe target must start absent");

            // `base` is under `/tmp` on virtually every machine, so this is also the regression:
            // the root has to survive the `--tmpfs /tmp` that the recipe applies before
            // binding it.
            let script = format!(
                "echo in > {}/inside.txt 2>/dev/null || exit 3\n\
                 if echo out > {} 2>/dev/null; then exit 4; fi\n\
                 exit 0",
                work.display(),
                outside.display()
            );

            let status = std::process::Command::new(bwrap)
                .args(super::bwrap_args(std::slice::from_ref(&work), &work, &[]))
                .arg("--")
                .arg("sh")
                .arg("-c")
                .arg(&script)
                .status()
                .expect("spawn bwrap");

            match status.code() {
                Some(0) => {}
                Some(3) => panic!("the write inside the workspace root was refused"),
                Some(4) => panic!("the write outside every root was permitted"),
                other => panic!("bwrap did not run the command: exit {other:?}"),
            }
            // The bytes are visible outside the namespace, which is what makes the bind a bind
            // rather than a tmpfs the child happened to be able to write.
            assert_eq!(
                std::fs::read_to_string(work.join("inside.txt")).expect("read back"),
                "in\n"
            );
            assert!(
                !outside.exists(),
                "the write outside every root must not have landed in $HOME: {}",
                outside.display()
            );
            let _ = std::fs::remove_file(&outside);
        }

        /// The masks over meka's own directories come after every bind, so a writable root that
        /// contains the store does not hand it back: later mounts win.
        #[test]
        fn meka_s_own_directories_are_masked_after_every_writable_bind() {
            let root = std::path::PathBuf::from("/home/someone");
            let store = root.join(".local/share/meka");
            let text: Vec<String> = super::bwrap_args(
                std::slice::from_ref(&root),
                &root,
                std::slice::from_ref(&store),
            )
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
            let bind_at = text
                .windows(3)
                .position(|window| {
                    window[0] == "--bind-try"
                        && window[1] == root.to_string_lossy()
                        && window[2] == root.to_string_lossy()
                })
                .expect("the root is bound writable");
            let mask_at = text
                .windows(2)
                .position(|window| window[0] == "--tmpfs" && window[1] == store.to_string_lossy())
                .expect("the store is masked");
            let chdir_at = text
                .iter()
                .position(|arg| arg == "--chdir")
                .expect("the working directory is entered last");
            assert!(
                bind_at < mask_at && mask_at < chdir_at,
                "the mask must come after the bind it has to beat and before the chdir: {text:?}"
            );
        }

        /// The store stays hidden under a writable root that contains it, on real bubblewrap.
        #[test]
        fn a_confined_shell_cannot_read_meka_s_store_under_a_writable_root() {
            let Some(bwrap) = which_bwrap() else {
                eprintln!("bwrap not installed; skipping");
                return;
            };
            let temp = tempfile::tempdir().expect("tempdir");
            let root = crate::workspace::canonical_for_test(temp.path());
            let store = root.join("meka-store");
            std::fs::create_dir(&store).expect("store");
            std::fs::write(store.join("meka.db"), "secret").expect("seed the store");
            let work = root.join("work");
            std::fs::create_dir(&work).expect("work");

            let script = format!(
                "if cat {}/meka.db >/dev/null 2>&1; then exit 4; fi\n\
                 echo ok > {}/inside.txt 2>/dev/null || exit 3\n\
                 exit 0",
                store.display(),
                work.display()
            );
            let status = std::process::Command::new(bwrap)
                .args(super::bwrap_args(
                    std::slice::from_ref(&root),
                    &work,
                    std::slice::from_ref(&store),
                ))
                .arg("--")
                .arg("sh")
                .arg("-c")
                .arg(&script)
                .status()
                .expect("spawn bwrap");
            match status.code() {
                Some(0) => {}
                Some(4) => panic!("the store was readable inside a root that contains it"),
                Some(3) => panic!("the write inside the workspace root was refused"),
                other => panic!("bwrap did not run the command: exit {other:?}"),
            }
            assert_eq!(
                std::fs::read_to_string(work.join("inside.txt")).expect("read back"),
                "ok\n"
            );
        }

        /// The layer is told to keep writable exactly what bwrap made writable: the roots as
        /// roots, every mask and `/dev` as scratch, never the other way around, since a mask
        /// passed as a root would hand back the sockets a bind can put under it. Its argv ends in
        /// `--`, so the shell after it is never read as a flag.
        #[test]
        fn the_inner_layer_keeps_the_roots_the_masks_and_dev_writable() {
            let text = strings(&super::inner_confinement_argv(
                std::path::Path::new("/proc/self/fd/7"),
                &[PathBuf::from("/home/someone/work")],
            ));
            assert_eq!(&text[..2], ["/proc/self/fd/7", "confine"]);
            let after = |flag: &str| -> Vec<&str> {
                text.windows(2)
                    .filter(|window| window[0] == flag)
                    .map(|window| window[1].as_str())
                    .collect()
            };
            assert_eq!(after("--writable"), ["/home/someone/work"]);
            let scratch = after("--scratch");
            for expected in ["/tmp", "/run", "/var/tmp", "/dev"] {
                assert!(
                    scratch.contains(&expected),
                    "'{expected}' must stay writable inside, as scratch: {text:?}"
                );
            }
            assert_eq!(text.last().map(String::as_str), Some("--"));
        }

        /// The built `meka` beside the test binary (`target/debug/meka`), which `cargo test`
        /// builds for the integration tests. A test binary has no `confine` verb, so the layer's
        /// tests need the real one; absent (a bare `cargo test --bin meka`), they skip loudly.
        pub(super) fn meka_binary_for_test() -> Option<PathBuf> {
            let binary = std::env::current_exe()
                .ok()?
                .parent()?
                .parent()?
                .join("meka");
            binary.is_file().then_some(binary)
        }

        pub(super) fn on_path(name: &str) -> Option<PathBuf> {
            let path = std::env::var_os("PATH")?;
            std::env::split_paths(&path)
                .map(|dir| dir.join(name))
                .find(|candidate| candidate.is_file())
        }

        fn strings(args: &[std::ffi::OsString]) -> Vec<String> {
            args.iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect()
        }

        fn which_bwrap() -> Option<std::path::PathBuf> {
            let path = std::env::var_os("PATH")?;
            std::env::split_paths(&path)
                .map(|dir| dir.join("bwrap"))
                .find(|candidate| candidate.is_file())
        }
    }

    /// `shell_execute` at `workspace`, end to end, on a real Unix sandbox.
    ///
    /// The Linux dialects are each tested through their helpers (`bwrap_args`, `apply_landlock`)
    /// with a hand-built root list, so nothing else exercises the wire between `Confinement` and
    /// the backend: cut it (`bwrap_args(&[])`, `apply_landlock(abi, &[])`) and the workspace shell
    /// is silently read-only, which is the level's central promise.
    #[cfg(unix)]
    mod workspace_shell_boundary {

        use tokio_util::sync::CancellationToken;

        use super::*;

        #[tokio::test]
        async fn a_workspace_shell_writes_inside_the_root_and_is_refused_outside() {
            // Every backend this host can actually run, not just the one `detect()` names:
            // `detect()` on Linux only consults `probe_landlock`, so it never returns `Bubblewrap`,
            // while production resolves the backend through `resolve_sandbox_backend`, which
            // auto-prefers Bubblewrap whenever `bwrap` probes OK. Testing only what `detect()`
            // returns would leave `Confinement::writable() -> bwrap_args` unexercised end to end on
            // the backend most hosts actually use.
            let mut backends = Vec::new();
            let detected = crate::sandbox::detect();
            if !matches!(detected, crate::sandbox::SandboxCapability::Unavailable) {
                backends.push(detected);
            }
            backends.extend(a_backend_detect_does_not_name());
            // Skip rather than fail where no backend exists: this asserts what confinement does,
            // and a host without one has nothing to assert against. Loud, so it cannot
            // silently never run.
            if backends.is_empty() {
                eprintln!("skipping: no usable sandbox backend on this host");
                return;
            }

            for capability in backends {
                eprintln!("workspace boundary against {capability:?}");
                a_workspace_shell_boundary_holds_for(capability).await;
            }
        }

        /// A backend production would choose that [`crate::sandbox::detect`] does not name.
        ///
        /// Linux only. `detect()` there consults `probe_landlock` alone, so it never returns
        /// `Bubblewrap`, while production resolves through `resolve_sandbox_backend`, which
        /// auto-prefers Bubblewrap whenever `bwrap` probes OK.
        ///
        /// Split behind a `cfg` rather than pushed inline because `SandboxCapability::Bubblewrap`
        /// is itself `cfg(target_os = "linux")`: this module is `cfg(all(test, unix))`, so naming
        /// the variant unconditionally fails the macOS build.
        #[cfg(target_os = "linux")]
        fn a_backend_detect_does_not_name() -> Option<crate::sandbox::SandboxCapability> {
            // Deliberately not `sandbox::bwrap_on_path`, which demands a root-owned binary: this
            // only answers "can this test spawn it", and a developer with a local build in
            // `~/.local/bin` should still get the leg run rather than silently skipped.
            let path = std::env::var_os("PATH")?;
            let bwrap_path = std::env::split_paths(&path)
                .map(|dir| dir.join("bwrap"))
                .find(|candidate| candidate.is_file())?;
            // The layer inside needs the built `meka`, which a test binary is not; without it the
            // leg exercises bwrap alone.
            let landlock_abi = match super::bubblewrap_boundary::meka_binary_for_test()
                .and_then(|binary| std::fs::File::open(binary).ok())
            {
                Some(executable) => {
                    super::super::INNER_LAYER_EXECUTABLE
                        .with(|slot| *slot.borrow_mut() = Some(executable));
                    crate::sandbox::layer_landlock_abi()
                }
                None => None,
            };
            Some(crate::sandbox::SandboxCapability::Bubblewrap {
                bwrap_path,
                landlock_abi,
            })
        }

        /// One `read` call through the tool under `capability`, from `cwd`, and the text it
        /// returned.
        #[cfg(target_os = "linux")]
        async fn read_shell_text(
            capability: crate::sandbox::SandboxCapability,
            cwd: &std::path::Path,
            command: &str,
        ) -> String {
            let mut tool = super::tool_for_test(
                crate::permission::SharedPermission::new(
                    Permission::Read,
                    crate::permission::EnabledPermissions::ALL,
                ),
                true,
            );
            tool.backend_probe = crate::sandbox::BackendProbe::Ok(capability.clone());
            tool.sandbox_capability = capability;
            tool.site.cwd = crate::workspace::SharedCwd::new(cwd.to_path_buf());
            let result = tool
                .execute(
                    serde_json::json!({"command": command}),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("the shell itself must run");
            super::text_of(&result)
        }

        /// The layer reaches the tool's own spawn, not only the argv helpers: a `read` call under
        /// Bubblewrap with a usable Landlock refuses what bwrap alone permits. `/proc/self/comm`
        /// is the dependency-free observable: bwrap's fresh procfs lets a process rename itself,
        /// and the ruleset grants nothing under `/proc`.
        #[cfg(target_os = "linux")]
        #[tokio::test]
        async fn a_bubblewrapped_tool_call_runs_inside_the_landlock_layer() {
            let Some(crate::sandbox::SandboxCapability::Bubblewrap {
                bwrap_path,
                landlock_abi: Some(abi),
            }) = a_backend_detect_does_not_name()
            else {
                eprintln!(
                    "skipping: bwrap, the built meka binary and a usable Landlock are all needed"
                );
                return;
            };
            let temp = tempfile::tempdir().expect("tempdir");
            let cwd = crate::workspace::canonical_for_test(temp.path());
            let rename = "echo meka > /proc/self/comm 2>/dev/null && echo RENAMED || echo REFUSED";

            let layered = read_shell_text(
                crate::sandbox::SandboxCapability::Bubblewrap {
                    bwrap_path: bwrap_path.clone(),
                    landlock_abi: Some(abi),
                },
                &cwd,
                rename,
            )
            .await;
            assert!(
                layered.contains("REFUSED"),
                "the layer grants nothing under /proc: {layered}"
            );
            let alone = read_shell_text(
                crate::sandbox::SandboxCapability::Bubblewrap {
                    bwrap_path,
                    landlock_abi: None,
                },
                &cwd,
                rename,
            )
            .await;
            assert!(
                alone.contains("RENAMED"),
                "bwrap alone lets a process rename itself, so the refusal above is the layer's: \
                 {alone}"
            );
        }

        /// The property the layer exists for, through the tool: a pathname socket the masks leave
        /// reachable. The socket sits in the cwd, which the recipe binds back in read-only under
        /// the `/tmp` mask, and `connect(2)` on a socket inode is not a write, so bwrap alone
        /// reaches it, the same route a socket under `$HOME` takes on a real machine. It is also
        /// the case that decides how the masks are granted inside: as scratch, without socket
        /// resolution, or this very socket would be reachable through the grant on `/tmp`.
        #[cfg(target_os = "linux")]
        #[tokio::test]
        async fn a_bubblewrapped_tool_call_refuses_a_socket_the_masks_leave_reachable() {
            let Some(crate::sandbox::SandboxCapability::Bubblewrap {
                bwrap_path,
                landlock_abi: Some(abi),
            }) = a_backend_detect_does_not_name()
            else {
                eprintln!(
                    "skipping: bwrap, the built meka binary and a usable Landlock are all needed"
                );
                return;
            };
            if abi < 9 || super::bubblewrap_boundary::on_path("python3").is_none() {
                eprintln!("skipping: needs Landlock ABI 9 and python3 to connect with");
                return;
            }
            let temp = tempfile::tempdir().expect("tempdir");
            let cwd = crate::workspace::canonical_for_test(temp.path());
            let socket = cwd.join("outside.sock");
            let _listener = std::os::unix::net::UnixListener::bind(&socket).expect("listen");
            let connect = format!(
                "python3 -c \"import socket; socket.socket(socket.AF_UNIX).connect('{}')\" \
                 2>/dev/null && echo CONNECTED || echo REFUSED",
                socket.display()
            );

            let layered = read_shell_text(
                crate::sandbox::SandboxCapability::Bubblewrap {
                    bwrap_path: bwrap_path.clone(),
                    landlock_abi: Some(abi),
                },
                &cwd,
                &connect,
            )
            .await;
            assert!(
                layered.contains("REFUSED"),
                "the layer refuses the socket: {layered}"
            );
            let alone = read_shell_text(
                crate::sandbox::SandboxCapability::Bubblewrap {
                    bwrap_path,
                    landlock_abi: None,
                },
                &cwd,
                &connect,
            )
            .await;
            assert!(
                alone.contains("CONNECTED"),
                "bwrap alone reaches a socket outside its masks, so the refusal is the layer's: \
                 {alone}"
            );
        }

        /// The layer is reached through a descriptor of the running image, not a path, so the
        /// binary being replaced or deleted under a running meka changes nothing: an upgrade
        /// under `meka serve` must not break every sandboxed command until a restart.
        #[cfg(target_os = "linux")]
        #[tokio::test]
        async fn the_layer_survives_its_executable_being_replaced() {
            let Some(crate::sandbox::SandboxCapability::Bubblewrap {
                bwrap_path,
                landlock_abi: Some(abi),
            }) = a_backend_detect_does_not_name()
            else {
                eprintln!(
                    "skipping: bwrap, the built meka binary and a usable Landlock are all needed"
                );
                return;
            };
            let binary = super::bubblewrap_boundary::meka_binary_for_test().expect("checked above");
            let temp = tempfile::tempdir().expect("tempdir");
            let cwd = crate::workspace::canonical_for_test(temp.path());
            let copy = cwd.join("meka-that-will-be-gone");
            std::fs::copy(&binary, &copy).expect("copy the binary");
            let executable = std::fs::File::open(&copy).expect("open the copy");
            std::fs::remove_file(&copy).expect("delete the copy while it is open");
            super::super::INNER_LAYER_EXECUTABLE.with(|slot| *slot.borrow_mut() = Some(executable));

            let layered = read_shell_text(
                crate::sandbox::SandboxCapability::Bubblewrap {
                    bwrap_path,
                    landlock_abi: Some(abi),
                },
                &cwd,
                "echo meka > /proc/self/comm 2>/dev/null && echo RENAMED || echo REFUSED",
            )
            .await;
            assert!(
                layered.contains("REFUSED"),
                "the layer ran from an executable that no longer exists on disk: {layered}"
            );
        }

        /// A shell under Landlock alone has no temporary directory: `TMPDIR` is not set for it and
        /// `mktemp` is refused at `read`. Below `unrestricted` meka writes to nothing but its
        /// store, and a real directory under the real `/tmp` was the one thing the Landlock dialect
        /// wrote outside it.
        #[cfg(target_os = "linux")]
        #[tokio::test]
        async fn a_landlock_shell_has_no_temp_space() {
            let capability = crate::sandbox::detect();
            if !matches!(
                capability,
                crate::sandbox::SandboxCapability::Landlock { .. }
            ) {
                eprintln!("skipping: no usable Landlock on this host");
                return;
            }
            let mut tool = super::tool_for_test(
                crate::permission::SharedPermission::new(
                    Permission::Read,
                    crate::permission::EnabledPermissions::ALL,
                ),
                true,
            );
            tool.backend_probe = crate::sandbox::BackendProbe::Ok(capability.clone());
            tool.sandbox_capability = capability;

            let result = tool
                .execute(
                    serde_json::json!({
                        "command": "echo \"TMPDIR=${TMPDIR-unset}\"; mktemp && echo CREATED",
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("the shell itself must run");
            let text = super::text_of(&result);
            assert!(result.is_error, "mktemp is refused: {text}");
            // The scrubbed environment passes the parent's own `TMPDIR` through when there is one,
            // so the child sees exactly that and never a directory meka made for it.
            let inherited = std::env::var("TMPDIR").unwrap_or_else(|_| "unset".to_string());
            assert!(
                text.contains(&format!("TMPDIR={inherited}\n")),
                "no temporary directory of meka's is offered: {text}"
            );
            assert!(!text.contains("CREATED"), "{text}");
        }

        /// macOS has one `read`-level backend and `detect()` names it, so there is nothing to add.
        #[cfg(not(target_os = "linux"))]
        fn a_backend_detect_does_not_name() -> Option<crate::sandbox::SandboxCapability> {
            None
        }

        /// One backend's worth of the boundary check above.
        async fn a_workspace_shell_boundary_holds_for(
            capability: crate::sandbox::SandboxCapability,
        ) {
            let temp = tempfile::tempdir().expect("tempdir");
            let base = crate::workspace::canonical_for_test(temp.path());
            let work = base.join("work");
            let outside = base.join("outside");
            std::fs::create_dir(&work).expect("work");
            std::fs::create_dir(&outside).expect("outside");

            let mut tool = super::tool_for_test(
                crate::permission::SharedPermission::new(
                    Permission::Workspace,
                    crate::permission::EnabledPermissions::ALL,
                ),
                true,
            );
            // The backend under test, not whatever `detect()` picked.
            tool.backend_probe = crate::sandbox::BackendProbe::Ok(capability.clone());
            tool.sandbox_capability = capability;
            tool.site.cwd = crate::workspace::SharedCwd::new(work.clone());
            tool.scope = crate::workspace::WriteScope::confined(vec![work.clone()]);

            let result = tool
                .execute(
                    serde_json::json!({
                        "command": format!(
                            "echo in > {}/inside.txt 2>/dev/null; \
                             echo out > {}/escaped.txt 2>/dev/null; true",
                            work.display(),
                            outside.display()
                        ),
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("the shell itself must run");

            // Ground truth on disk, not the tool's narration: a shell that never started would
            // report failure just as convincingly as one the sandbox confined.
            assert!(
                work.join("inside.txt").exists(),
                "a write inside the workspace root must land: {result:?}"
            );
            assert!(
                !outside.join("escaped.txt").exists(),
                "a write outside every root must be refused by the backend: {result:?}"
            );
        }
    }
}
