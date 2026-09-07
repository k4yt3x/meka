//! Filesystem sandboxing for read-only command execution.
//!
//! On Linux there are two backends. Bubblewrap (`bwrap --ro-bind /` plus tmpfs masks) is preferred
//! whenever it is installed and its user-namespace smoke test passes; Landlock LSM is the fallback,
//! and requires ABI v3 (kernel 6.2+) because `truncate(2)` is unmediated below it (see
//! `MIN_LANDLOCK_ABI`, which is Linux-only and so deliberately not an intra-doc link: the link
//! would be unresolvable on the two targets where the item does not exist, and CI gates rustdoc on
//! all three). On macOS, uses `sandbox-exec`. On Windows there are two mechanisms rather than one:
//! `read` spawns the child with a duplicated primary token dropped to Low integrity via
//! `SetTokenInformation(TokenIntegrityLevel, …)`, which blocks writes to anything outside the
//! documented Low-integrity surface, while `workspace` uses a `WRITE_RESTRICTED` token plus a
//! per-root capability ACE and deliberately leaves the integrity label alone. Where no backend is
//! usable, sandboxing is unavailable and shell execution at `read` hard-errors rather than running
//! unconfined.
//!
//! **What every backend does not restrict**: reads. A sandboxed child can read any file the user
//! can, including credential files, and the network is deliberately left open on all of them. The
//! boundary these enforce is "this command cannot change the machine", not "this command cannot see
//! or send anything".

#[cfg(target_os = "linux")]
mod bubblewrap;
#[cfg(target_os = "linux")]
mod landlock;
pub(crate) mod seatbelt;
#[cfg(target_os = "windows")]
pub(crate) mod windows;

#[cfg(target_os = "linux")]
use self::bubblewrap::*;
#[cfg(target_os = "linux")]
pub(crate) use self::landlock::*;
use crate::config::SandboxBackend;

/// The capability a probe result grants: what the backend can do, or nothing when it is not usable.
pub(crate) fn capability_from_probe(probe: &BackendProbe) -> SandboxCapability {
    match probe {
        BackendProbe::Ok(capability) => capability.clone(),
        _ => SandboxCapability::Unavailable,
    }
}

/// What a single `execute_command` call runs under.
///
/// Three states rather than a boolean, because `workspace` is neither of the two a boolean could
/// express: it is not unconfined, and it is not read-only. Folding it into either
/// one fails in a direction that matters. Treated as unconfined, the shell ignores the boundary the
/// file tools enforce; treated as read-only, the level promises writes it never delivers.
///
/// A boolean also hides the asymmetry that makes this dangerous: the absence of confinement is the
/// permissive state, so any code path that forgets a case fails open. An enum makes each case one
/// the compiler asks about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Confinement {
    /// No sandbox. `unrestricted`, or sandboxing turned off in config.
    Unconfined,
    /// Reads everywhere, writes nowhere. `none` and `read`.
    ReadOnly,
    /// Reads everywhere, writes only beneath these canonical roots. `workspace`.
    ///
    /// An empty list is meaningful and behaves as [`Self::ReadOnly`]: it means no root resolved,
    /// which happens when the working directory was deleted under a running session.
    Workspace(Vec<std::path::PathBuf>),
}

impl Confinement {
    /// What this call runs under, given the config switch, the level and the live scope.
    ///
    /// `permission` is passed in rather than read back off `scope`, even though the scope holds a
    /// handle that would answer. The caller has already read the level once at its enforcement
    /// site, and re-reading it here would make the decision depend on two handles that are only
    /// the same one by convention: a tool wired with a scope built from a different handle would
    /// sandbox against one level while denying against another, and nothing would say so.
    pub(crate) fn resolve(
        sandbox_enabled: bool,
        permission: crate::permission::Permission,
        scope: &crate::workspace::WriteScope,
        cwd: &crate::workspace::SharedCwd,
    ) -> Self {
        if !sandbox_enabled {
            return Self::Unconfined;
        }
        match permission {
            // Only the level that promises no boundary runs the shell without one. An approved
            // command at any other level runs confined at that level, like an approved write.
            crate::permission::Permission::Unrestricted => Self::Unconfined,
            crate::permission::Permission::Workspace => {
                Self::Workspace(scope.confined_to(cwd).unwrap_or_default())
            }
            _ => Self::ReadOnly,
        }
    }

    /// Whether a sandbox is applied at all, i.e. anything other than [`Self::Unconfined`].
    pub(crate) fn is_sandboxed(&self) -> bool {
        !matches!(self, Self::Unconfined)
    }

    /// The roots this call may write beneath. Empty for every state but [`Self::Workspace`].
    pub(crate) fn writable(&self) -> &[std::path::PathBuf] {
        match self {
            Self::Workspace(roots) => roots,
            _ => &[],
        }
    }
}

/// Hand back any standing OS-level grant this process placed, before an exit that will not unwind.
///
/// A no-op on every platform but Windows. Landlock, bubblewrap and seatbelt confine a child process
/// and die with it, leaving nothing behind to clean up. The Windows boundary is instead an ACE
/// written onto the user's own directories, which outlives the process unless something takes it
/// back, so it needs a counterpart the others do not.
///
/// Call this immediately before each [`std::process::exit`], which skips destructors: see the call
/// sites in `main.rs` and `server.rs`. A crash or `SIGKILL` still leaves the ACE standing, which
/// `windows::WindowsGrants` documents. Unlinked deliberately: that module is `#[cfg(windows)]`, so
/// an intra-doc link to it fails the rustdoc gate everywhere else.
pub(crate) fn release_process_grants() {
    #[cfg(windows)]
    windows::process_grants().revoke_all();
}

/// What this process learned about the sandbox at startup: the backend it will use, whether it was
/// auto-picked, and the probe of that backend. Computed once by the host's bootstrap, because a
/// probe is a runtime fact about the machine rather than a setting, and it costs a smoke test.
#[derive(Debug, Clone)]
pub(crate) struct SandboxResolution {
    pub(crate) backend: crate::config::SandboxBackend,
    /// True when the backend was auto-resolved (the user pinned nothing). Gates the "stronger
    /// sandbox available; install bwrap" startup warn, which must not nag a user who chose
    /// landlock.
    pub(crate) auto_resolved: bool,
    pub(crate) probe: BackendProbe,
}

/// Resolve the backend and probe it, when sandboxing is on; a disabled sandbox gets a placeholder
/// probe nothing consults, so subcommands that never touch the shell do not pay the smoke test.
pub(crate) fn resolve_backend(
    configured: Option<crate::config::SandboxBackend>,
    enabled: bool,
) -> SandboxResolution {
    if !enabled {
        return SandboxResolution {
            backend: configured.unwrap_or(crate::config::SandboxBackend::Landlock),
            auto_resolved: false,
            probe: BackendProbe::Missing {
                reason: "sandbox disabled in config".to_string(),
            },
        };
    }
    let (backend, auto_resolved, probe) = resolve_sandbox_backend(configured);
    SandboxResolution {
        backend,
        auto_resolved,
        probe,
    }
}
/// Resolve the active Linux sandbox backend.
///
/// When the user pinned `[shell].sandbox_backend = "..."` in `config.toml`, that choice is binding,
/// no silent fallback at runtime; an unavailable explicit backend surfaces at use time via the
/// `BackendProbe::Missing` / `UserNamespaceDenied` variants.
///
/// When the value is unset (`None`), meka probes bubblewrap and picks it if available, falling back
/// to landlock otherwise. The `auto_resolved` flag is propagated so the startup warn helper can
/// nudge the user once toward installing bwrap (without nagging users who explicitly pinned
/// landlock).
#[cfg(target_os = "linux")]
pub(crate) fn resolve_sandbox_backend(
    configured: Option<SandboxBackend>,
) -> (SandboxBackend, bool, BackendProbe) {
    // Probe Bubblewrap only when its result is load-bearing for the resolution: either the user
    // pinned it explicitly, or no value was configured (so the probe decides whether to auto-pick
    // it). When the user pinned Landlock, the Bubblewrap smoke test would be pure waste
    // (~500 ms on every meka start).
    let (backend, auto_resolved, cached_bubblewrap_probe) = match configured {
        Some(explicit) => (explicit, false, None),
        None => {
            let probe = probe_backend(SandboxBackend::Bubblewrap);
            let picked = match &probe {
                BackendProbe::Ok(_) => SandboxBackend::Bubblewrap,
                _ => SandboxBackend::Landlock,
            };
            (picked, true, Some(probe))
        }
    };
    // The Landlock arm discards `cached_bubblewrap_probe` because the auto-resolve path that
    // populated it landed on Bubblewrap (it only falls through to Landlock when Bubblewrap probes
    // unavailable, and that probe isn't useful for the chosen backend's status).
    let backend_probe = match (backend, cached_bubblewrap_probe) {
        (SandboxBackend::Bubblewrap, Some(probe)) => probe,
        (SandboxBackend::Bubblewrap, None) => probe_backend(SandboxBackend::Bubblewrap),
        (SandboxBackend::Landlock, _) => probe_backend(SandboxBackend::Landlock),
    };
    (backend, auto_resolved, backend_probe)
}
/// Non-Linux platforms have a single platform-native sandbox (`sandbox-exec` on macOS,
/// Low-integrity on Windows, nothing elsewhere). `[shell].sandbox_backend` is documented as
/// Linux-only and is ignored here: the resolved capability comes from [`detect`] and is surfaced
/// through the same `BackendProbe::Ok` envelope so the downstream wiring needs no platform branch.
#[cfg(not(target_os = "linux"))]
pub(crate) fn resolve_sandbox_backend(
    _configured: Option<SandboxBackend>,
) -> (SandboxBackend, bool, BackendProbe) {
    let probe = match detect() {
        SandboxCapability::Unavailable => BackendProbe::Missing {
            reason: "no platform sandbox backend available".to_string(),
        },
        #[cfg(target_os = "macos")]
        capability @ SandboxCapability::SandboxExec => BackendProbe::Ok(capability),
        #[cfg(target_os = "windows")]
        capability @ SandboxCapability::LowIntegrity => BackendProbe::Ok(capability),
    };
    // `SandboxBackend::Landlock` is a stand-in here; the field exists for Linux config parity but
    // is never consulted on this platform.
    (SandboxBackend::Landlock, false, probe)
}
/// What kind of `read`-level sandbox is available on this platform. Resolved once at config time
/// and threaded into `tools::shell::ExecuteCommandTool` so the spawn path knows which argv shape
/// and `pre_exec` hook to use.
#[derive(Debug, Clone)]
pub(crate) enum SandboxCapability {
    /// Linux: filesystem-write restriction via Landlock LSM (kernel 5.13+). Below ABI v9 the kernel
    /// has no right governing `connect(2)` on a *pathname* Unix socket, so dbus and systemd-user
    /// stay reachable and a confined process can have them write on its behalf; from v9 that right
    /// is handled and granted nowhere, which also costs socket-based clients like `docker` and
    /// `psql`. Prefer Bubblewrap when available: its tmpfs masks remove the sockets outright, on
    /// every kernel.
    #[cfg(target_os = "linux")]
    Landlock { abi_version: i32 },
    /// Linux: read-only root bind via `bwrap --ro-bind /` plus tmpfs masks over `/tmp`, `/run`,
    /// `/var/tmp`, and `$XDG_RUNTIME_DIR`. Blocks both filesystem writes and IPC-socket mutation;
    /// network is unrestricted.
    #[cfg(target_os = "linux")]
    Bubblewrap { bwrap_path: std::path::PathBuf },
    /// macOS: `sandbox-exec` with the hardened SBPL profile defined in
    /// [`seatbelt::SANDBOX_PROFILE_READONLY`]. Blocks filesystem writes and IPC mutation (no
    /// launchd, pasteboard, LaunchServices, etc.); network is unrestricted.
    #[cfg(target_os = "macos")]
    SandboxExec,
    /// Windows: child runs with a duplicated primary token dropped to Low integrity. Blocks writes
    /// outside the Low-integrity surface (user home, AppData, Program Files); IPC mutation is
    /// constrained but not as tightly as Linux/macOS.
    #[cfg(target_os = "windows")]
    LowIntegrity,
    /// No sandbox available on this platform / configuration. Shell commands at `read` hard-error
    /// rather than silently bypass the sandbox.
    Unavailable,
}

/// Result of probing a specific sandbox backend at config-resolution time. The probe is run once
/// per meka launch (twice when the resolver needs to consider both Landlock and Bubblewrap for
/// auto-pick) and cached on `ResolvedConfig.backend_probe`.
#[derive(Debug, Clone)]
pub(crate) enum BackendProbe {
    Ok(SandboxCapability),
    /// The backend's prerequisite is missing: `bwrap` isn't on `$PATH`, the Landlock kernel ABI
    /// isn't supported, etc. The `reason` is plain text and is plumbed into user-facing
    /// warnings/errors verbatim.
    Missing {
        reason: String,
    },
    /// Linux + bubblewrap only: the user-namespace smoke test failed with stderr that matched the
    /// documented denial fingerprints. Stored stderr is truncated to a few KiB. The only
    /// constructor (`smoke_test_bwrap`) is Linux-only, so the variant is dead on other platforms;
    /// the explicit allow lets non-Linux clippy stay clean without hiding regressions on Linux.
    #[cfg_attr(
        not(target_os = "linux"),
        allow(dead_code, reason = "constructed only by the Linux bwrap smoke test")
    )]
    UserNamespaceDenied {
        stderr: String,
    },
    /// The asked-for backend doesn't apply on this platform.
    #[allow(
        dead_code,
        reason = "constructed only by tests::backend_unavailable_reason_maps_each_variant"
    )]
    UnsupportedPlatform,
}

/// Snapshot of the sandbox-relevant config slice. Carried by components that need to emit the
/// sandbox warns (`warn_if_sandbox_issues`) without depending on the whole `ResolvedConfig`.
///
/// The fields are read only on Linux (`warn_if_sandbox_issues` early-returns on other platforms
/// because the warnings reference Linux-only config keys), but the struct is constructed
/// unconditionally so the hosts need no platform branch.
#[derive(Clone)]
#[cfg_attr(
    not(target_os = "linux"),
    allow(
        dead_code,
        reason = "the fields are read only on the Linux warning path"
    )
)]
pub(crate) struct SandboxState {
    pub(crate) enabled: bool,
    pub(crate) backend: crate::config::SandboxBackend,
    pub(crate) auto_resolved: bool,
    pub(crate) probe: BackendProbe,
}

impl SandboxState {
    pub(crate) fn new(enabled: bool, resolution: &SandboxResolution) -> Self {
        Self {
            enabled,
            backend: resolution.backend,
            auto_resolved: resolution.auto_resolved,
            probe: resolution.probe.clone(),
        }
    }
}

/// Where in the meka lifecycle the sandbox-state check is happening. The "stronger sandbox
/// available" nudge (Warn 2) only fires at startup; "backend unavailable" (Warn 1) fires at every
/// relevant boundary because the user needs to know shell at `read` is broken right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WarnContext {
    /// Once-per-launch warn during `ResolvedConfig` construction or agent setup. Both Warn 1 and
    /// Warn 2 fire here.
    Startup,
    /// Initial permission level was `Read` at `meka --permission read` launch. Only Warn 1 fires.
    InitialReadLevel,
    /// User pressed Shift+Tab and cycled into `Read`. Only Warn 1 fires.
    ReadModeEntry,
}

/// Emit any relevant sandbox warnings for the configured backend state.
///
/// * **Warn 1** (backend unavailable): probe failed and `sandbox = true`. Shell commands at `read`
///   will hard-error at use time, so the user is told up front. Re-emitted at every lifecycle
///   boundary.
/// * **Warn 2** (could be stronger): the user has not pinned a backend and the backend
///   auto-resolved to landlock because bubblewrap was not usable. Nudges them once toward
///   installing bwrap, with an explicit escape hatch (pin landlock to suppress). Startup only.
pub(crate) fn warn_if_sandbox_issues(state: &SandboxState, context: WarnContext) {
    if !state.enabled {
        return;
    }

    // `sandbox_backend` is a Linux-only config knob; the warnings below name it directly and would
    // be misleading on macOS / Windows where the platform has a single fixed backend. On those
    // hosts an unusable platform sandbox is a near-impossible configuration and surfaces at use
    // time via the hard-error path in `src/tools/shell.rs` anyway.
    #[cfg(target_os = "macos")]
    {
        let _ = (state, context);
    }

    // Integrity levels stop a Low-integrity process writing up, not reading up, so the one
    // boundary Windows offers cannot hide meka's own store from a command at `read` the way the
    // bubblewrap and seatbelt masks do; and a workspace root's inheritable ACE grants writes to
    // everything under it, meka's directories included, with no way to subtract them afterwards.
    // Named at startup because nothing else would: the command succeeds, and the credential it
    // read leaves over the open network.
    #[cfg(windows)]
    if context == WarnContext::Startup {
        tracing::warn!(
            "the Windows sandbox cannot hide meka's config directory and credential store from a \
             confined command: a command at `read` can read them, and a workspace root that \
             contains them can write them"
        );
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(reason) = backend_unavailable_reason(&state.probe) {
            // No specific alternative backend is suggested: the other one may also be unavailable
            // on this host (a kernel without Landlock, bwrap not installed).
            let backend = state.backend;
            tracing::warn!(
                "read-level sandbox unavailable: {reason} (configured: {backend}); shell commands \
                 at `read` fail until `[shell].sandbox_backend` names a usable backend"
            );
            return;
        }

        if context == WarnContext::Startup
            && state.auto_resolved
            && matches!(state.backend, crate::config::SandboxBackend::Landlock)
        {
            tracing::warn!(
                "using Landlock for the sandbox; install Bubblewrap for stronger protection, or set \
                 `[shell].sandbox_backend = \"landlock\"` to silence this"
            );
        }

        // Landlock rules only ever add access, so there is no way to subtract meka's own
        // directories from the read grant on `/`, or from a workspace root that contains them.
        // Bubblewrap masks both. Said whether or not the backend was pinned: pinning accepts the
        // weaker backend, not necessarily this particular hole, which nothing else reports.
        if context == WarnContext::Startup
            && matches!(
                &state.probe,
                BackendProbe::Ok(SandboxCapability::Landlock { .. })
            )
        {
            tracing::warn!(
                "Landlock cannot hide meka's config directory and credential store from a confined \
                 command: a command at `read` can read them, and a workspace root that contains \
                 them can write them. Bubblewrap masks both"
            );
        }

        // Warn 3: the ABI clears `MIN_LANDLOCK_ABI` so the filesystem is genuinely write-protected,
        // but the mitigations added after v3 are absent. Each is a real hole a command at `read`
        // can walk through (a D-Bus or `systemd-run --user` call reaches a privileged
        // daemon that will happily write on its behalf), and none of them is visible to the
        // user otherwise, so the gap is named rather than left to the kernel version.
        if context == WarnContext::Startup
            && let BackendProbe::Ok(SandboxCapability::Landlock { abi_version }) = &state.probe
            && *abi_version < 9
        {
            let mut missing: Vec<&str> = Vec::new();
            if *abi_version < 5 {
                missing.push("device ioctls (v5)");
            }
            if *abi_version < 6 {
                missing.push("abstract Unix sockets and cross-domain signals (v6)");
            }
            missing.push("pathname Unix sockets (v9)");
            // Deliberately does not name Bubblewrap as the remedy. Measured: bwrap masks four
            // directories and unmounts nothing else, and it never unshares the network namespace,
            // so a socket in the abstract namespace or under `$HOME` stays reachable from inside
            // it. On this axis Landlock at v9 is the stronger backend, and sending a user to
            // install bwrap to close these channels would send them the wrong way.
            let missing = missing.join(", ");
            tracing::warn!(
                "Landlock ABI v{abi_version} write-protects the filesystem but does not restrict \
                 {missing}; a command at `read` can still reach a local service over those \
                 channels. A newer kernel closes them, and Bubblewrap does not"
            );
        }
    }
}

/// Human-readable reason a backend probe failed, or `None` when the probe is `Ok`. Used by both the
/// startup `warn!` path ([`warn_if_sandbox_issues`]) and the lazy hard-error path in
/// `src/tools/shell.rs` so the two surfaces stay in sync.
pub(crate) fn backend_unavailable_reason(probe: &BackendProbe) -> Option<String> {
    match probe {
        BackendProbe::Ok(_) => None,
        BackendProbe::Missing { reason } => Some(reason.clone()),
        BackendProbe::UserNamespaceDenied { stderr } => {
            let first_line = stderr.lines().next().unwrap_or("").trim();
            if first_line.is_empty() {
                Some("user namespaces are denied on this host".to_string())
            } else {
                Some(format!(
                    "user namespaces are denied on this host ({first_line})"
                ))
            }
        }
        BackendProbe::UnsupportedPlatform => {
            Some("backend is not supported on this platform".to_string())
        }
    }
}

/// Probe a specific sandbox backend. Linux-only: the `SandboxBackend` enum represents
/// Linux-specific backends, and non-Linux platforms route through `detect()` in the non-Linux
/// `resolve_sandbox_backend` instead.
#[cfg(target_os = "linux")]
pub(crate) fn probe_backend(backend: crate::config::SandboxBackend) -> BackendProbe {
    match backend {
        crate::config::SandboxBackend::Landlock => probe_landlock(),
        crate::config::SandboxBackend::Bubblewrap => probe_bubblewrap(),
    }
}

/// Test-only "what's the strongest sandbox available right now?" entry point. Production code
/// takes the backend `ResolvedConfig` settled from `[shell].sandbox_backend`, `--sandbox-backend`
/// and `MEKA_SANDBOX_BACKEND`; tests reach for whatever capability the host happens to support.
#[cfg(any(test, not(target_os = "linux")))]
pub(crate) fn detect() -> SandboxCapability {
    #[cfg(target_os = "linux")]
    {
        // Routed through the probe rather than the raw syscall so the `MIN_LANDLOCK_ABI` policy is
        // applied in exactly one place: a test asking "what sandbox does this host have?" must get
        // the same answer production would act on, or it would happily exercise an ABI meka
        // refuses.
        if let BackendProbe::Ok(capability) = probe_landlock() {
            return capability;
        }
    }

    #[cfg(target_os = "macos")]
    {
        if std::path::Path::new("/usr/bin/sandbox-exec").exists() {
            return SandboxCapability::SandboxExec;
        }
    }

    // Token-integrity APIs are available on every supported Windows version (7+). No runtime probe
    // is needed.
    #[cfg(target_os = "windows")]
    let detected = SandboxCapability::LowIntegrity;
    #[cfg(not(target_os = "windows"))]
    let detected = SandboxCapability::Unavailable;
    detected
}

/// PowerShell prelude that switches `$OutputEncoding` and `[Console]::OutputEncoding` to UTF-8. See
/// [`wrap_command_with_utf8_output`] for why this is necessary.
///
/// Guarded on the language mode, and belt-and-braces wrapped in `try`/`catch`, because a
/// `WRITE_RESTRICTED` token puts PowerShell into **ConstrainedLanguage** mode, where setting a
/// property on a non-core type is refused: "Property setting is supported only on core types in
/// this language mode." Unguarded, that error is printed to stderr ahead of every shell command at
/// `workspace` on Windows, which the model reads as the command having failed. `unrestricted` and
/// `read` both report `FullLanguage`, and only the restricted token constrains it, so this is a
/// cost of the workspace boundary rather than a property of the host.
///
/// Skipping it means output at `workspace` is decoded with the host's legacy code page, so
/// non-ASCII may be mangled there. A wrong character beats an error on every line, and there is no
/// other way to reach the encoding from inside ConstrainedLanguage.
#[cfg_attr(
    not(target_os = "windows"),
    allow(dead_code, reason = "called only on Windows; the tests run everywhere")
)]
const POWERSHELL_UTF8_PRELUDE: &str = "if($ExecutionContext.SessionState.LanguageMode -eq 'FullLanguage'){try{\
     [Console]::OutputEncoding=[System.Text.Encoding]::UTF8;\
     $OutputEncoding=[System.Text.Encoding]::UTF8}catch{}};";

/// Prepend the UTF-8 encoding prelude to a PowerShell command. Used by both the sandboxed and
/// non-sandboxed Windows `execute_command` paths so pipe output is always decoded as UTF-8 on the
/// Rust side regardless of the console's legacy code page.
#[cfg_attr(
    not(target_os = "windows"),
    allow(dead_code, reason = "called only on Windows; the tests run everywhere")
)]
pub(crate) fn wrap_command_with_utf8_output(command: &str) -> String {
    let mut wrapped = String::with_capacity(POWERSHELL_UTF8_PRELUDE.len() + command.len() + 1);
    wrapped.push_str(POWERSHELL_UTF8_PRELUDE);
    wrapped.push(' ');
    wrapped.push_str(command);
    wrapped
}

/// Quote a single command-line argument per Windows `CommandLineToArgvW` rules. Mirrors the
/// algorithm used by `std::process::Command` on Windows.
///
/// This is the correct encoding for any program that parses its command line with
/// `CommandLineToArgvW`, including `powershell.exe`, which is what the Low-integrity sandbox
/// invokes. It is **not** the correct encoding for `cmd.exe /C` (cmd treats `\` literally); don't
/// apply this to cmd command bodies.
///
/// Compiled on every platform even though the rules are Windows-specific: the implementation is
/// pure string manipulation, so unit tests run on Linux/macOS without an `#[cfg(target_os =
/// "windows")]` gate (the `cfg_attr` below just silences the dead-code warning off-Windows).
#[cfg_attr(
    not(target_os = "windows"),
    allow(dead_code, reason = "called only on Windows; the tests run everywhere")
)]
pub(crate) fn quote_command_arg(arg: &str) -> String {
    if !arg.is_empty()
        && !arg
            .chars()
            .any(|c| matches!(c, ' ' | '\t' | '\n' | '\u{000B}' | '"'))
    {
        return arg.to_string();
    }

    let mut quoted = String::from("\"");
    let mut pending_backslashes: usize = 0;
    for c in arg.chars() {
        match c {
            '\\' => {
                pending_backslashes += 1;
            }
            '"' => {
                // Double the run of backslashes, then emit an escaped quote.
                for _ in 0..(pending_backslashes * 2 + 1) {
                    quoted.push('\\');
                }
                pending_backslashes = 0;
                quoted.push('"');
            }
            _ => {
                for _ in 0..pending_backslashes {
                    quoted.push('\\');
                }
                pending_backslashes = 0;
                quoted.push(c);
            }
        }
    }
    // Any trailing backslashes must be doubled so the closing quote is not escaped by them.
    for _ in 0..(pending_backslashes * 2) {
        quoted.push('\\');
    }
    quoted.push('"');
    quoted
}

/// Curated env-var set for a sandboxed shell child. Applied via `Command::env_clear()` +
/// `Command::envs(...)` before spawn so it covers Bubblewrap, Landlock, Seatbelt, and the Windows
/// Low-integrity path uniformly without per-backend flag plumbing.
///
/// Sandboxes at `read` still allow outbound network (curl, dns, etc.), so a leaked secret in env
/// (`ANTHROPIC_API_KEY`, `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, …) is a live exfiltration vector
/// under prompt injection. Stripping the env at spawn time closes that gap without touching what
/// the sandbox itself enforces.
///
/// **Unix** uses an explicit allow-list (small, curated). Unknown vars are dropped; `EDITOR`,
/// `PAGER`, `BAT_THEME`, etc. don't survive into shells at `read`. Users who need a specific var
/// should switch to `unrestricted` (trusted-operation path; no scrubbing applies).
///
/// **Windows** uses a heuristic deny-list ([`is_sensitive_env_name`]) because PowerShell pulls in a
/// long tail of system vars (`PSModulePath`, `APPDATA`, `ProgramFiles`, etc.) that do not fit a
/// tidy allow-list; an allow-list breaks core cmdlets.
pub(crate) fn sandbox_child_env() -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
    std::env::vars_os()
        .filter(|(name_os, _)| match name_os.to_str() {
            Some(name) => keep_sandbox_env_var(name),
            // A name that is not UTF-8 is dropped on purpose, and only the name. Both filters below
            // decide by matching text, so a name they cannot read is one they cannot rule out, and
            // Windows' arm is a deny-list: passing an unexaminable name through would be the one
            // direction that fails open. Values are a different question and go through
            // `encode_wide` untouched, since the destination block is UTF-16 natively.
            None => false,
        })
        .collect()
}

#[cfg(unix)]
fn keep_sandbox_env_var(name: &str) -> bool {
    // Exact-match allow-list. Names that an empty-env `sh -c …` typically needs to function: `PATH`
    // so commands resolve, `HOME` for tools that read `~/.config`, locale so `grep`/`sort` don't
    // mangle non-ASCII, etc.
    const ALLOW_EXACT: &[&str] = &[
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "PWD",
        "TERM",
        "COLORTERM",
        "LANG",
        "TMPDIR",
        "TMP",
        "TEMP",
    ];
    // How the machine reaches the network at all, on a host that does not route directly. A child
    // that cannot see these connects to nothing and reports a TLS or DNS failure that names none of
    // the real cause, which for an MCP server means it starts, registers its tools, and then fails
    // every call. None of them grants authority: they say where to go and whom to trust, and the
    // child was going to make the request either way.
    //
    // Deliberately not extended to `SSH_AUTH_SOCK` (a live credential agent), `NODE_OPTIONS` (which
    // takes `--require`, i.e. arbitrary code), or the import-path family `PYTHONPATH` / `NODE_PATH`
    // / `VIRTUAL_ENV`, which redirect what a program loads. A server that needs one of those takes
    // it explicitly through `${VAR}` in its own `[[mcp.servers]] env` table, where the user has
    // said so.
    const ALLOW_NETWORK: &[&str] = &[
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "ALL_PROXY",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ];
    // Prefix allow-list keeps the LC_* / XDG_* families future-proof without enumerating each var.
    // Locale (`LC_ALL`, `LC_CTYPE`, `LC_MESSAGES`, …) and XDG paths (`XDG_RUNTIME_DIR`,
    // `XDG_CONFIG_HOME`, …) are both legitimately broad.
    const ALLOW_PREFIX: &[&str] = &["LC_", "XDG_"];

    if ALLOW_EXACT.contains(&name) || ALLOW_NETWORK.contains(&name) {
        return true;
    }
    if ALLOW_PREFIX.iter().any(|prefix| name.starts_with(prefix)) {
        return true;
    }
    // Apple frameworks (CFString, foundation, etc.) read this to pick a text encoding; dropping it
    // makes some CLIs misbehave with no useful error.
    #[cfg(target_os = "macos")]
    if name == "__CF_USER_TEXT_ENCODING" {
        return true;
    }
    false
}

#[cfg(windows)]
fn keep_sandbox_env_var(name: &str) -> bool {
    !is_sensitive_env_name(name)
}

#[cfg(not(any(unix, windows)))]
fn keep_sandbox_env_var(_name: &str) -> bool {
    // No sandbox is reachable on other platforms (SandboxCapability::Unavailable hard-errors at use
    // time), so this filter is never exercised. Pass through for completeness.
    true
}

/// Heuristic match for variable names that commonly carry credentials or point to
/// credential-bearing resources (SSH agent socket, kubeconfig, `.netrc`, GPG home, etc.).
/// Case-insensitive substring match on a list of credential-shaped markers plus prefix match on
/// known provider / service / database namespaces.
///
/// Tuned to be **aggressive on false positives** (a legitimate `GITHUB_ACTOR` is dropped alongside
/// `GITHUB_TOKEN`, `SLACK_CHANNEL` alongside `SLACK_WEBHOOK_URL`) because the downside of a
/// missing env var is a confusing tool error the user can recover from, while the downside of a
/// leaked secret is a live exfiltration channel.
///
/// Used by the Windows arm of [`sandbox_child_env`]; not consulted on Unix, where the curated
/// allow-list already drops every var by default. Lives at module scope (not inside `windows`)
/// so its tests exercise both platforms in CI; the function is pure string manipulation with no
/// Windows-specific dependency.
#[cfg_attr(
    unix,
    allow(dead_code, reason = "called only on Windows; the tests run everywhere")
)]
pub(crate) fn is_sensitive_env_name(name: &str) -> bool {
    const SENSITIVE_SUBSTRINGS: &[&str] = &[
        // Credential-shaped name fragments.
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "PASSPHRASE",
        "API_KEY",
        "APIKEY",
        "PRIVATE_KEY",
        "BEARER",
        "CREDENTIAL",
        "SESSION_KEY",
        "ACCESS_KEY",
        // Broader `_KEY` catches `SIGNING_KEY`, `ENCRYPTION_KEY`, `DEPLOY_KEY`, `MASTER_KEY`, etc.
        // without enumerating each.
        "_KEY",
        // Specific names that don't share a credential-shaped fragment but point to credentials,
        // sockets, or other exfil-relevant resources. Substring (not exact) match so derivatives
        // like `WSL_SSH_AUTH_SOCK` are also caught.
        "SSH_AUTH_SOCK",
        "SSH_ASKPASS",
        "GIT_ASKPASS",
        "GIT_SSH_COMMAND",
        "KUBECONFIG",
        "GNUPGHOME",
        "NETRC",
        // Code-execution vectors, which the Unix allow-list refuses by name, and the two arms
        // implement one policy. They are not credentials; they are ways to make an ordinary
        // command run something else: `NODE_OPTIONS` takes `--require`, `PYTHONPATH` /
        // `NODE_PATH` prepend an import path, and `PIP_INDEX_URL` redirects where a package is
        // fetched from. The values come from meka's own parent environment, so this is not an
        // escalation the agent can drive, but the sandboxed child is exactly the place where "the
        // same command, quietly doing something else" is worth refusing.
        "NODE_OPTIONS",
        "NODE_PATH",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "VIRTUAL_ENV",
        "PIP_INDEX_URL",
    ];
    const SENSITIVE_PREFIXES: &[&str] = &[
        // Agent / first-party.
        "ANTHROPIC_",
        "OPENAI_",
        "CLAUDE_",
        "MEKA_",
        // Major clouds.
        "AWS_",
        "GCP_",
        "GOOGLE_",
        "AZURE_",
        // Source control / CI.
        "GITHUB_",
        "GITLAB_",
        // Model hubs / AI APIs.
        "HF_",
        "HUGGINGFACE_",
        "OPENROUTER_",
        "GROQ_",
        "MISTRAL_",
        "COHERE_",
        "REPLICATE_",
        "TOGETHER_",
        "FIREWORKS_",
        // Package registries.
        "NPM_",
        "PYPI_",
        "CARGO_REGISTRY_",
        "DOCKER_",
        // Database connection strings often embed credentials.
        "DATABASE_",
        "POSTGRES_",
        "MYSQL_",
        "MONGO_",
        "REDIS_",
        // PaaS / hosting providers with API tokens.
        "STRIPE_",
        "CLOUDFLARE_",
        "HEROKU_",
        "VERCEL_",
        "NETLIFY_",
        "SUPABASE_",
        "RAILWAY_",
        // Identity / secret managers.
        "OKTA_",
        "AUTH0_",
        "VAULT_",
        "JWT_",
        "OAUTH_",
        // Observability tools with ingest keys.
        "SENTRY_",
        "DATADOG_",
        // Communication APIs with bot tokens / webhooks.
        "SLACK_",
        "DISCORD_",
    ];

    let upper = name.to_ascii_uppercase();
    SENSITIVE_SUBSTRINGS
        .iter()
        .any(|needle| upper.contains(needle))
        || SENSITIVE_PREFIXES
            .iter()
            .any(|prefix| upper.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::seatbelt::*;
    use crate::{permission::Permission, workspace};

    /// The Landlock masks are pinned per ABI, bit for bit.
    ///
    /// These two functions are the whole of what meka asks the kernel to police. The end-to-end
    /// boundary test only catches the subset that stops a write, and it cannot catch the rest,
    /// because `&` binds tighter than `|` in Rust: flipping
    /// a middle operator drops only the two adjacent bits and leaves the write flags standing.
    /// Losing `MAKE_SOCK` and `MAKE_FIFO` that way would let a confined shell create a socket or a
    /// fifo outside its roots while every existing assertion still passed.
    ///
    /// Spelled as literal sums rather than by rebuilding the expression, so the test states the
    /// intended mask instead of restating the code. The bit values are Landlock's, fixed by its
    /// ABI, and the per-version gating is the fact worth pinning: v2 adds `REFER`, v3 `TRUNCATE`,
    /// v5 `IOCTL_DEV`, v9 `RESOLVE_UNIX`, and v4 adds only network flags, which is why nothing
    /// changes there. Scoping arrives whole at v6 and must stay zero below it, because an unknown
    /// `scoped` bit makes `landlock_create_ruleset` fail with `EINVAL` and takes the sandbox down
    /// with it.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_landlock_masks_are_exactly_what_each_abi_supports() {
        // The thirteen filesystem rights every supported ABI handles: bits 0..=12.
        const BASE: u64 = (1 << 13) - 1;
        const REFER: u64 = 1 << 13;
        const TRUNCATE: u64 = 1 << 14;
        const IOCTL_DEV: u64 = 1 << 15;
        const RESOLVE_UNIX: u64 = 1 << 16;

        for (abi, expected) in [
            (1, BASE),
            (2, BASE | REFER),
            (3, BASE | REFER | TRUNCATE),
            // v4 is network-only, so the filesystem mask is unchanged from v3.
            (4, BASE | REFER | TRUNCATE),
            (5, BASE | REFER | TRUNCATE | IOCTL_DEV),
            (8, BASE | REFER | TRUNCATE | IOCTL_DEV),
            (9, BASE | REFER | TRUNCATE | IOCTL_DEV | RESOLVE_UNIX),
        ] {
            assert_eq!(
                handled_access_for_abi(abi),
                expected,
                "ABI {abi} handled-access mask: got {:#018b}, want {expected:#018b}",
                handled_access_for_abi(abi)
            );
        }

        const ABSTRACT_UNIX_SOCKET: u64 = 1 << 0;
        const SIGNAL: u64 = 1 << 1;
        for abi in 1..=5 {
            assert_eq!(
                scoped_for_abi(abi),
                0,
                "scoping arrived in v6; setting a bit below it fails ruleset creation outright"
            );
        }
        for abi in 6..=9 {
            assert_eq!(
                scoped_for_abi(abi),
                ABSTRACT_UNIX_SOCKET | SIGNAL,
                "v6 and up must scope both the abstract socket namespace and signals"
            );
        }
    }

    /// A `bwrap` the user can replace is not a sandbox, and must be refused.
    ///
    /// Every ordinary desktop has several user-writable directories ahead of `/usr/bin`
    /// (`~/.local/bin`, a cargo or go bin dir, a toolchain shim dir), so a `bwrap_on_path` that
    /// took the first executable named `bwrap` would let a six-line script that `exec`s its final
    /// argument turn every `read` and `workspace` shell command into an unconfined one. The smoke
    /// test cannot catch it, because `bwrap <flags> /bin/true` is exactly what such a shim
    /// satisfies.
    ///
    /// Asserted on the predicate rather than by planting a real shim, because the interesting half
    /// (a root-owned binary in a root-owned directory) cannot be constructed in a test without
    /// root. `/usr/bin` stands in for it, and the temp dir stands in for `~/.local/bin`.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_bwrap_in_a_user_writable_directory_is_not_trusted() {
        let temp = tempfile::tempdir().expect("tempdir");
        assert!(
            !super::only_root_can_write(temp.path()),
            "a directory this process just created is writable by this user, so anything in it \
             could be swapped for a shim"
        );

        // The control, and the reason this is a predicate rather than a hardcoded path list: a
        // distribution that ships `bwrap` somewhere unusual still works as long as root owns it.
        let system = std::path::Path::new("/usr/bin");
        if system.is_dir() {
            assert!(
                super::only_root_can_write(system),
                "/usr/bin must be trusted, or bubblewrap is unreachable on an ordinary host"
            );
        }
    }

    /// Trust needs *both* the binary and its directory, and the mixed cases are the whole point.
    ///
    /// Asserting the two ends only (both trusted, neither trusted) leaves the conjunction free:
    /// `&&` and `||` agree whenever their operands agree, and `||` re-admits a root-owned binary
    /// sitting in a directory the user can write.
    ///
    /// The mixed pairs are built from paths that exist on any Linux host rather than by planting
    /// files, because the trusted half cannot be created without root.
    #[test]
    #[cfg(target_os = "linux")]
    fn trust_requires_the_binary_and_its_directory_together() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mine = temp.path();
        let system = std::path::Path::new("/usr/bin");
        if !system.is_dir() {
            return;
        }

        assert!(
            super::trusted_to_confine(system, system),
            "a root-owned binary in a root-owned directory is the case that must be admitted"
        );
        assert!(
            !super::trusted_to_confine(mine, mine),
            "neither half trusted must be refused"
        );
        assert!(
            !super::trusted_to_confine(system, mine),
            "a root-owned binary in a user-writable directory is one `mv` from being the user's"
        );
        assert!(
            !super::trusted_to_confine(mine, system),
            "a user-writable binary is not redeemed by the directory around it"
        );
    }

    /// Only an executable regular file is a candidate.
    ///
    /// Both halves matter: dropping the `is_file` conjunct admits a directory named `bwrap`,
    /// and flipping the mode test to `== 0` admits only *non*-executable files, which quietly makes
    /// bubblewrap undiscoverable on every host and silently downgrades the backend.
    #[test]
    #[cfg(target_os = "linux")]
    fn only_an_executable_regular_file_is_a_bwrap_candidate() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().expect("tempdir");
        let plain = temp.path().join("plain");
        std::fs::write(&plain, "not executable").expect("write");
        let runnable = temp.path().join("runnable");
        std::fs::write(&runnable, "#!/bin/sh\n").expect("write");
        std::fs::set_permissions(&runnable, std::fs::Permissions::from_mode(0o755)).expect("chmod");

        let meta = |path: &std::path::Path| std::fs::metadata(path).expect("metadata");
        assert!(super::is_executable_file(&meta(&runnable)));
        assert!(
            !super::is_executable_file(&meta(&plain)),
            "a file with no execute bit cannot be the sandbox helper"
        );
        assert!(
            !super::is_executable_file(&meta(temp.path())),
            "a directory named `bwrap` is not a binary"
        );
    }
    /// Between ABI 3 and 9 the filesystem is genuinely write-protected but the later mitigations
    /// are absent, and the warning naming them is the only way a user learns which.
    ///
    /// Without the block a host believes `read` restricts more than the running kernel actually
    /// does. Driven through a subscriber pinned to `WARN` because that is the default floor, so
    /// this also fails if the level is dropped to `info` where `-v` would be needed to see it.
    ///
    /// Linux-gated because [`super::SandboxCapability::Landlock`] is.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_landlock_abi_below_9_names_the_mitigations_it_does_not_provide() {
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct Capture(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for Capture {
            fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
                crate::sync::lock(&self.0).extend_from_slice(buffer);
                Ok(buffer.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
            type Writer = Self;

            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        // v3 clears the floor, so this is the "protected, but not fully" band the warning owns.
        for (abi, expected) in [
            (3, vec![
                "device ioctls",
                "abstract Unix sockets",
                "pathname Unix sockets",
            ]),
            (6, vec!["pathname Unix sockets"]),
        ] {
            let capture = Capture(Arc::new(Mutex::new(Vec::new())));
            let buffer = Arc::clone(&capture.0);
            let subscriber = tracing_subscriber::fmt()
                .with_writer(capture)
                .with_max_level(tracing::Level::WARN)
                .finish();

            let state = super::SandboxState {
                enabled: true,
                backend: crate::config::SandboxBackend::Landlock,
                auto_resolved: false,
                probe: super::BackendProbe::Ok(super::SandboxCapability::Landlock {
                    abi_version: abi,
                }),
            };
            tracing::subscriber::with_default(subscriber, || {
                super::warn_if_sandbox_issues(&state, super::WarnContext::Startup);
            });

            let logged =
                String::from_utf8(crate::sync::lock(&buffer).clone()).expect("log output is utf-8");
            for gap in expected {
                assert!(
                    logged.contains(gap),
                    "ABI v{abi} must name '{gap}' as unrestricted: {logged:?}"
                );
            }
        }

        // v9 has them all, so there is nothing to warn about and a warning would be noise.
        let capture = Capture(Arc::new(Mutex::new(Vec::new())));
        let buffer = Arc::clone(&capture.0);
        let subscriber = tracing_subscriber::fmt()
            .with_writer(capture)
            .with_max_level(tracing::Level::WARN)
            .finish();
        let state = super::SandboxState {
            enabled: true,
            backend: crate::config::SandboxBackend::Landlock,
            auto_resolved: false,
            probe: super::BackendProbe::Ok(super::SandboxCapability::Landlock { abi_version: 9 }),
        };
        tracing::subscriber::with_default(subscriber, || {
            super::warn_if_sandbox_issues(&state, super::WarnContext::Startup);
        });
        let logged =
            String::from_utf8(crate::sync::lock(&buffer).clone()).expect("log output is utf-8");
        assert!(
            !logged.contains("does not restrict"),
            "v9 restricts all of them; warning anyway trains the user to ignore it: {logged:?}"
        );
    }

    /// A child that cannot see the machine's proxy or CA configuration reaches nothing, and says so
    /// in terms that name none of the cause. For an MCP server that means it connects, registers
    /// its tools, and then fails every call. Neither kind of variable grants authority: they say
    /// where to go and whom to trust, and the child was making the request either way.
    ///
    /// The three families below are refused on purpose, so a widening of the list has to argue with
    /// this test rather than slip past it.
    #[cfg(unix)]
    #[test]
    fn network_configuration_reaches_a_sandboxed_child_but_credentials_do_not() {
        for allowed in [
            "HTTPS_PROXY",
            "https_proxy",
            "NO_PROXY",
            "SSL_CERT_FILE",
            "NODE_EXTRA_CA_CERTS",
        ] {
            assert!(
                super::keep_sandbox_env_var(allowed),
                "{allowed} is how the child reaches the network",
            );
        }
        for refused in [
            "SSH_AUTH_SOCK",
            "NODE_OPTIONS",
            "PYTHONPATH",
            "NODE_PATH",
            "VIRTUAL_ENV",
            "AWS_SECRET_ACCESS_KEY",
        ] {
            assert!(
                !super::keep_sandbox_env_var(refused),
                "{refused} carries either a credential or a way to run code",
            );
        }
    }

    use super::*;

    #[test]
    fn is_sensitive_env_name_matches_known_secret_patterns() {
        // Provider API keys.
        assert!(is_sensitive_env_name("ANTHROPIC_API_KEY"));
        assert!(is_sensitive_env_name("OPENAI_API_KEY"));
        assert!(is_sensitive_env_name("anthropic_api_key"));
        // VCS / CI tokens.
        assert!(is_sensitive_env_name("GITHUB_TOKEN"));
        assert!(is_sensitive_env_name("GITLAB_PRIVATE_TOKEN"));
        // Cloud secrets.
        assert!(is_sensitive_env_name("AWS_SECRET_ACCESS_KEY"));
        assert!(is_sensitive_env_name("AWS_SESSION_TOKEN"));
        // Credential-shaped fragments.
        assert!(is_sensitive_env_name("my_bearer_auth"));
        assert!(is_sensitive_env_name("DATABASE_PASSWORD"));
        assert!(is_sensitive_env_name("GPG_PASSPHRASE"));
        // `_KEY` catches non-API-key creds that don't match the older
        // `API_KEY`/`PRIVATE_KEY`/`SESSION_KEY`/`ACCESS_KEY` patterns.
        assert!(is_sensitive_env_name("SIGNING_KEY"));
        assert!(is_sensitive_env_name("ENCRYPTION_KEY"));
        assert!(is_sensitive_env_name("DEPLOY_KEY"));
    }

    #[test]
    fn is_sensitive_env_name_catches_pointer_vars() {
        // Specific named variables that point to credentials, agent sockets, or other
        // exfil-relevant resources, caught by substring even when wrapped in a longer name.
        assert!(is_sensitive_env_name("SSH_AUTH_SOCK"));
        assert!(is_sensitive_env_name("WSL_SSH_AUTH_SOCK"));
        assert!(is_sensitive_env_name("KUBECONFIG"));
        assert!(is_sensitive_env_name("GNUPGHOME"));
        assert!(is_sensitive_env_name("NETRC"));
        assert!(is_sensitive_env_name("CURLOPT_NETRC"));
        assert!(is_sensitive_env_name("GIT_SSH_COMMAND"));
        assert!(is_sensitive_env_name("GIT_ASKPASS"));
        assert!(is_sensitive_env_name("SSH_ASKPASS"));
    }

    #[test]
    fn is_sensitive_env_name_catches_service_prefixes() {
        // AI provider namespaces beyond the original list.
        assert!(is_sensitive_env_name("OPENROUTER_API_KEY"));
        assert!(is_sensitive_env_name("GROQ_API_KEY"));
        assert!(is_sensitive_env_name("MISTRAL_API_KEY"));
        assert!(is_sensitive_env_name("COHERE_API_KEY"));
        // Database connection strings: DATABASE_URL embeds the password.
        assert!(is_sensitive_env_name("DATABASE_URL"));
        assert!(is_sensitive_env_name("POSTGRES_HOST"));
        assert!(is_sensitive_env_name("MONGO_URI"));
        assert!(is_sensitive_env_name("REDIS_PASSWORD"));
        // PaaS / hosting providers.
        assert!(is_sensitive_env_name("STRIPE_SECRET_KEY"));
        assert!(is_sensitive_env_name("CLOUDFLARE_API_TOKEN"));
        assert!(is_sensitive_env_name("VERCEL_TOKEN"));
        assert!(is_sensitive_env_name("SUPABASE_KEY"));
        // Identity / secret managers.
        assert!(is_sensitive_env_name("VAULT_TOKEN"));
        assert!(is_sensitive_env_name("OKTA_CLIENT_SECRET"));
        assert!(is_sensitive_env_name("AUTH0_CLIENT_ID"));
        // Generic auth tokens.
        assert!(is_sensitive_env_name("JWT_SECRET"));
        assert!(is_sensitive_env_name("OAUTH_CLIENT_SECRET"));
        // Observability and communications.
        assert!(is_sensitive_env_name("SENTRY_DSN"));
        assert!(is_sensitive_env_name("DATADOG_API_KEY"));
        assert!(is_sensitive_env_name("SLACK_WEBHOOK_URL"));
        assert!(is_sensitive_env_name("DISCORD_BOT_TOKEN"));
    }

    #[test]
    fn is_sensitive_env_name_allows_system_vars() {
        // Windows system vars PowerShell needs at startup must NOT be flagged sensitive; that's
        // the whole reason Windows uses deny-list instead of allow-list.
        assert!(!is_sensitive_env_name("SystemRoot"));
        assert!(!is_sensitive_env_name("PATH"));
        assert!(!is_sensitive_env_name("PSModulePath"));
        assert!(!is_sensitive_env_name("APPDATA"));
        assert!(!is_sensitive_env_name("LOCALAPPDATA"));
        assert!(!is_sensitive_env_name("ProgramFiles"));
        assert!(!is_sensitive_env_name("USERPROFILE"));
        assert!(!is_sensitive_env_name("TEMP"));
        // Unix basics also shouldn't flag (the function is used on Windows but compiles
        // cross-platform for testability).
        assert!(!is_sensitive_env_name("HOME"));
        assert!(!is_sensitive_env_name("USER"));
        assert!(!is_sensitive_env_name("LANG"));
        assert!(!is_sensitive_env_name("TERM"));
        // `KEYBOARD_LAYOUT` doesn't have `_KEY` as a substring (the pattern requires an underscore
        // before KEY), so it survives.
        assert!(!is_sensitive_env_name("KEYBOARD_LAYOUT"));
    }

    /// `cargo test` always runs with `PATH` set (the test binary needs it to invoke itself), so
    /// this is a no-mutation sanity check that the filter doesn't accidentally strip it. Windows
    /// env-var names are case-insensitive and typically stored as `Path`, so the match is
    /// case-insensitive.
    #[test]
    fn sandbox_child_env_keeps_path() {
        let env = sandbox_child_env();
        assert!(
            env.iter()
                .any(|(name, _)| name.to_string_lossy().eq_ignore_ascii_case("PATH")),
            "expected PATH to survive the sandbox env filter"
        );
    }

    /// Token-shaped sentinel: dropped by the Unix allow-list (not in the curated list) AND by the
    /// Windows deny-list (`TOKEN` substring match in `is_sensitive_env_name`). Verifies both arms
    /// strip it.
    #[test]
    fn sandbox_child_env_drops_token_sentinel() {
        const NAME: &str = "MEKA_TEST_SCRUB_TOKEN_PROBE";
        // SAFETY: `set_var`/`remove_var` are process-global and `cargo test` runs in-process tests
        // in parallel. The variable name is long and test-specific so it can't collide with another
        // test or the real environment.
        unsafe {
            std::env::set_var(NAME, "sentinel-should-be-dropped");
        }
        let env = sandbox_child_env();
        let leaked = env.iter().any(|(name, _)| name.to_string_lossy() == NAME);
        unsafe {
            std::env::remove_var(NAME);
        }
        assert!(
            !leaked,
            "token-shaped sentinel leaked through the sandbox env filter"
        );
    }

    /// Unix: any var not in the curated allow-list (and not matching `LC_*`/`XDG_*`) is dropped.
    /// The sentinel name has no special shape: pure "unknown var" test.
    #[cfg(unix)]
    #[test]
    fn sandbox_child_env_drops_unknown_var() {
        const NAME: &str = "MEKA_TEST_SCRUB_UNKNOWN_PROBE";
        unsafe {
            std::env::set_var(NAME, "should-be-dropped");
        }
        let env = sandbox_child_env();
        let leaked = env.iter().any(|(name, _)| name.to_string_lossy() == NAME);
        unsafe {
            std::env::remove_var(NAME);
        }
        assert!(!leaked, "unknown var leaked through the Unix allow-list");
    }

    /// Unix: `LC_*` prefix match keeps the locale family without enumerating each variant.
    #[cfg(unix)]
    #[test]
    fn sandbox_child_env_keeps_lc_prefix() {
        const NAME: &str = "LC_MEKA_TEST_PROBE";
        unsafe {
            std::env::set_var(NAME, "en_US.UTF-8");
        }
        let env = sandbox_child_env();
        let kept = env.iter().any(|(name, _)| name.to_string_lossy() == NAME);
        unsafe {
            std::env::remove_var(NAME);
        }
        assert!(kept, "LC_* prefix var was dropped from sandbox env");
    }

    /// Unix: `XDG_*` prefix match keeps the XDG basedir family without enumerating each variant.
    #[cfg(unix)]
    #[test]
    fn sandbox_child_env_keeps_xdg_prefix() {
        const NAME: &str = "XDG_MEKA_TEST_PROBE";
        unsafe {
            std::env::set_var(NAME, "/tmp/meka-probe");
        }
        let env = sandbox_child_env();
        let kept = env.iter().any(|(name, _)| name.to_string_lossy() == NAME);
        unsafe {
            std::env::remove_var(NAME);
        }
        assert!(kept, "XDG_* prefix var was dropped from sandbox env");
    }

    #[test]
    fn detect_sandbox_capability() {
        let capability = detect();
        // Should detect something on Linux/macOS/Windows, Unavailable on others
        match capability {
            #[cfg(target_os = "linux")]
            SandboxCapability::Landlock { abi_version } => {
                assert!(abi_version >= 1);
            }
            #[cfg(target_os = "linux")]
            SandboxCapability::Bubblewrap { .. } => {}
            #[cfg(target_os = "macos")]
            SandboxCapability::SandboxExec => {}
            #[cfg(target_os = "windows")]
            SandboxCapability::LowIntegrity => {}
            SandboxCapability::Unavailable => {}
        }
    }

    /// The seatbelt profile names each root as a `-D` parameter, never inside the profile text.
    ///
    /// This is the whole reason roots travel out of band: a directory called `it's "here"` or one
    /// ending in a backslash would otherwise have to survive SBPL string quoting, and getting that
    /// wrong does not fail loudly: it changes which subpath the rule matches. The assertion that
    /// no root's text appears in the profile body is the one that would catch a future rewrite
    /// deciding interpolation is simpler. The private denies come after the write allows: SBPL
    /// takes the last matching rule, so a root that contains meka's store must not win over the
    /// mask on it.
    #[test]
    fn the_profile_denies_meka_s_own_directories_after_the_write_allows() {
        let root = std::path::PathBuf::from("/Users/someone");
        let store = root.join("Library/Application Support/meka");
        let (profile, params) =
            sandbox_profile_for(std::slice::from_ref(&root), std::slice::from_ref(&store));
        let allow = profile
            .find("(allow file-write* (subpath (param \"MEKA_WRITABLE_0\")))")
            .expect("the root is allowed");
        let deny = profile
            .find(
                "(deny file-read* file-write* file-test-existence (subpath (param \
                 \"MEKA_PRIVATE_0\")))",
            )
            .expect("the store is denied");
        assert!(
            deny > allow,
            "the deny must follow the allow it has to beat:\n{profile}"
        );
        let mut expected = std::ffi::OsString::from("MEKA_PRIVATE_0=");
        expected.push(store.as_os_str());
        assert!(
            params
                .windows(2)
                .any(|window| window[0] == "-D" && window[1] == expected),
            "the store's path travels as a parameter: {params:?}"
        );
    }

    #[test]
    fn the_seatbelt_profile_passes_roots_as_parameters_not_profile_text() {
        let roots = vec![
            std::path::PathBuf::from(r#"/tmp/it's "quoted""#),
            std::path::PathBuf::from(r"/tmp/trailing\"),
        ];
        let (profile, params) = sandbox_profile_for(&roots, &[]);

        assert_eq!(params, vec![
            std::ffi::OsString::from("-D"),
            std::ffi::OsString::from(r#"MEKA_WRITABLE_0=/tmp/it's "quoted""#),
            std::ffi::OsString::from("-D"),
            std::ffi::OsString::from(r"MEKA_WRITABLE_1=/tmp/trailing\"),
        ]);
        for (index, _) in roots.iter().enumerate() {
            assert!(
                profile.contains(&format!(r#"(subpath (param "MEKA_WRITABLE_{index}"))"#)),
                "root {index} must be referenced by parameter name: {profile}"
            );
        }
        for root in &roots {
            assert!(
                !profile.contains(&root.display().to_string()),
                "no root's text may appear in the profile body: {profile}"
            );
        }
        assert!(
            profile.starts_with(SANDBOX_PROFILE_READONLY),
            "the writable rules must be appended to the read-only base, not replace it"
        );
    }

    /// A root that is not valid UTF-8 is still made writable.
    ///
    /// The parameter is built as an `OsString` and the path pushed whole, so there is no UTF-8
    /// requirement to fail. A `to_str()` with a `continue` drops such a root silently from the
    /// allow-list and runs the command with it still read-only, a boundary quietly narrower than
    /// the one meka reported.
    #[test]
    #[cfg(unix)]
    fn a_non_utf8_root_still_becomes_a_seatbelt_parameter() {
        use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

        let root = std::path::PathBuf::from(OsStr::from_bytes(b"/tmp/work\xff"));
        assert!(root.to_str().is_none(), "precondition: not valid UTF-8");

        let (profile, params) = sandbox_profile_for(std::slice::from_ref(&root), &[]);
        assert_eq!(params.len(), 2, "the root must not be dropped: {params:?}");

        let mut expected = std::ffi::OsString::from("MEKA_WRITABLE_0=");
        expected.push(root.as_os_str());
        assert_eq!(params[1], expected, "the path must survive byte-for-byte");
        assert!(profile.contains(r#"(param "MEKA_WRITABLE_0")"#));
    }

    #[test]
    fn wrap_command_with_utf8_output_prepends_prelude() {
        let wrapped = wrap_command_with_utf8_output("Write-Output '日本語'");
        assert!(wrapped.contains("[Console]::OutputEncoding="));
        assert!(wrapped.contains("$OutputEncoding=[System.Text.Encoding]::UTF8"));
        assert!(wrapped.ends_with("Write-Output '日本語'"));
        // A space must separate the prelude from the user command so PowerShell doesn't glue them
        // into one malformed statement.
        assert!(wrapped.contains("}; Write-Output"));
    }

    /// The prelude may not throw under ConstrainedLanguage, which is what a `WRITE_RESTRICTED`
    /// token puts PowerShell into.
    ///
    /// Both guards are asserted because either alone is thin: the language-mode test is what
    /// normally skips the encoding switch, and the `try`/`catch` is what stops a host that
    /// constrains something else from printing an error ahead of every command the agent runs. The
    /// symptom is not a failed command but a successful one that looks failed, which the model then
    /// reports to the user as an error.
    #[test]
    fn the_utf8_prelude_cannot_throw_under_constrained_language() {
        let wrapped = wrap_command_with_utf8_output("Get-Date");
        assert!(
            wrapped.contains("LanguageMode -eq 'FullLanguage'"),
            "the encoding switch must be skipped outright when the language mode forbids it: \
             {wrapped}"
        );
        assert!(
            wrapped.contains("try{") && wrapped.contains("}catch{}"),
            "and must still be caught if it runs and fails anyway: {wrapped}"
        );
    }

    /// Reference table covering the corners of the `CommandLineToArgvW` encoding. Cross-platform:
    /// `quote_command_arg` is pure string manipulation and has no Windows-specific runtime
    /// dependency.
    #[test]
    fn quote_command_arg_reference_table() {
        let cases: &[(&str, &str)] = &[
            ("cmd.exe", "cmd.exe"),
            ("", r#""""#),
            ("with space", r#""with space""#),
            ("with\ttab", "\"with\ttab\""),
            (r#"say "hi""#, r#""say \"hi\"""#),
            (r#"a\"b"#, r#""a\\\"b""#),
            (r"path with space\", r#""path with space\\""#),
            // A quote preceded by a single backslash: the backslash is doubled and the quote is
            // escaped.
            (r#"\""#, r#""\\\"""#),
            // Backslashes not adjacent to a quote pass through literally (no escaping needed, no
            // quoting needed, no special chars).
            (r"a\\b", r"a\\b"),
            // Unicode and newlines pass through. Newline counts as whitespace so the argument gets
            // quoted.
            ("日本語", "日本語"),
            ("hello world\n", "\"hello world\n\""),
        ];
        for (input, expected) in cases {
            assert_eq!(
                &quote_command_arg(input),
                expected,
                "input {input:?} produced wrong quoting"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn handled_access_abi_v1() {
        let access = handled_access_for_abi(1);
        assert!(access & LANDLOCK_ACCESS_FS_WRITE_FILE != 0);
        assert!(access & LANDLOCK_ACCESS_FS_READ_FILE != 0);
        assert!(access & LANDLOCK_ACCESS_FS_REFER == 0);
        assert!(access & LANDLOCK_ACCESS_FS_TRUNCATE == 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn handled_access_abi_v3() {
        let access = handled_access_for_abi(3);
        assert!(access & LANDLOCK_ACCESS_FS_REFER != 0);
        assert!(access & LANDLOCK_ACCESS_FS_TRUNCATE != 0);
        assert!(access & LANDLOCK_ACCESS_FS_IOCTL_DEV == 0);
    }

    /// The root rule grants only execute, read-file and read-dir, so handling `RESOLVE_UNIX` is
    /// what denies it: a sandboxed command cannot ask a daemon over the D-Bus or systemd socket to
    /// write on its behalf. Taking the bit below v9 would make `landlock_create_ruleset` fail with
    /// `EINVAL` and leave the process unconfined.
    #[cfg(target_os = "linux")]
    #[test]
    fn handled_access_takes_resolve_unix_only_from_abi_v9() {
        assert!(handled_access_for_abi(8) & LANDLOCK_ACCESS_FS_RESOLVE_UNIX == 0);
        assert!(handled_access_for_abi(9) & LANDLOCK_ACCESS_FS_RESOLVE_UNIX != 0);
    }

    /// Below ABI v3 the ruleset does not handle `LANDLOCK_ACCESS_FS_TRUNCATE`, so a sandboxed child
    /// can still empty an existing file even though every open-for-write is denied. meka documents
    /// the `read` level as write-protecting the filesystem, so the only honest answer on such a
    /// kernel is to report the backend unusable and let the shell tool hard-error, rather than
    /// to sandbox with a ruleset that does not enforce what was promised.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_landlock_abi_below_the_truncate_floor_is_reported_unusable() {
        for abi in [1, 2] {
            let probe = landlock_probe_from_abi(Some(abi));
            let reason = backend_unavailable_reason(&probe)
                .unwrap_or_else(|| panic!("ABI v{abi} must not be accepted"));
            assert!(
                reason.contains("truncate(2)"),
                "the reason must name what is unenforced, got: {reason}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_landlock_abi_at_or_above_the_floor_is_accepted() {
        for abi in [MIN_LANDLOCK_ABI, 6, 9] {
            assert!(
                matches!(
                    landlock_probe_from_abi(Some(abi)),
                    BackendProbe::Ok(SandboxCapability::Landlock { abi_version }) if abi_version == abi
                ),
                "ABI v{abi} should be accepted"
            );
        }
    }

    /// A kernel with no Landlock at all and one whose Landlock is too old are both unusable, but a
    /// user can only act on the difference: the first needs a newer kernel or Bubblewrap, the
    /// second is specifically about `truncate(2)`. Keep the two messages distinct.
    #[cfg(target_os = "linux")]
    #[test]
    fn absent_landlock_and_too_old_landlock_report_different_reasons() {
        let absent = backend_unavailable_reason(&landlock_probe_from_abi(None))
            .expect("no Landlock must be unusable");
        let too_old = backend_unavailable_reason(&landlock_probe_from_abi(Some(1)))
            .expect("ABI v1 must be unusable");
        assert_ne!(absent, too_old);
        assert!(absent.contains("5.13"), "got: {absent}");
        assert!(too_old.contains("6.2"), "got: {too_old}");
    }

    #[test]
    fn backend_unavailable_reason_maps_each_variant() {
        assert!(
            backend_unavailable_reason(&BackendProbe::Ok(SandboxCapability::Unavailable)).is_none()
        );
        let reason = backend_unavailable_reason(&BackendProbe::Missing {
            reason: "bwrap not found on PATH".to_string(),
        });
        assert_eq!(reason.as_deref(), Some("bwrap not found on PATH"));
        let reason = backend_unavailable_reason(&BackendProbe::UserNamespaceDenied {
            stderr: "bwrap: setting up uid map: Permission denied\n".to_string(),
        });
        assert!(
            reason
                .as_deref()
                .unwrap_or("")
                .contains("user namespaces are denied")
        );
        assert!(
            backend_unavailable_reason(&BackendProbe::UnsupportedPlatform)
                .as_deref()
                .unwrap_or("")
                .contains("not supported")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn probe_backend_landlock_returns_known_variant() {
        // Smoke test: confirms the probe runs without panicking on whatever kernel this build host
        // has. Which variant comes back cannot be asserted because CI may have an older kernel
        // where Landlock is unavailable.
        let probe = probe_backend(crate::config::SandboxBackend::Landlock);
        assert!(matches!(
            probe,
            BackendProbe::Ok(_) | BackendProbe::Missing { .. }
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore]
    fn probe_backend_bubblewrap_via_path_when_available() {
        // Opt-in: only runs when explicitly requested via `--ignored`. Skipped if `bwrap` isn't on
        // `$PATH` since the probe will report `Missing { reason: "bwrap not found on PATH" }` which
        // would fail the assertion below.
        if bwrap_on_path().is_none() {
            eprintln!("skipping: bwrap not on PATH");
            return;
        }
        let probe = probe_backend(crate::config::SandboxBackend::Bubblewrap);
        match probe {
            BackendProbe::Ok(SandboxCapability::Bubblewrap { bwrap_path }) => {
                assert!(bwrap_path.is_absolute());
            }
            BackendProbe::UserNamespaceDenied { .. } => {
                eprintln!("skipping: host doesn't support user namespaces");
            }
            other => panic!("unexpected probe result: {other:?}"),
        }
    }

    /// Every level maps to exactly one confinement, and only `unrestricted` maps to none.
    ///
    /// The table is spelled out rather than derived because this is the one decision in the change
    /// whose mistakes fail *open*: a level that should confine but resolves to `Unconfined` runs
    /// the shell with no sandbox at all, and nothing downstream would report it.
    #[test]
    fn every_level_resolves_to_exactly_one_confinement() {
        let temp = tempfile::tempdir().expect("tempdir");
        // Through `strip_verbatim`, because that is what `writable_roots` does and this asserts
        // equality against its output. Bare `canonicalize` returns a `\\?\`-prefixed path on
        // Windows, so the expectation would be the one spelling production never produces and this
        // test would fail on Windows alone.
        let base = crate::workspace::canonical_for_test(temp.path());
        let cwd = workspace::SharedCwd::new(base.clone());
        let scope = workspace::WriteScope::confined(vec![base.clone()]);

        for level in [Permission::None, Permission::Read] {
            assert_eq!(
                Confinement::resolve(true, level, &scope, &cwd),
                Confinement::ReadOnly,
                "{level} must confine the shell read-only"
            );
        }
        assert_eq!(
            Confinement::resolve(true, Permission::Workspace, &scope, &cwd),
            Confinement::Workspace(vec![base]),
            "workspace must hand the shell the same roots the file tools fence against"
        );
        assert_eq!(
            Confinement::resolve(true, Permission::Unrestricted, &scope, &cwd),
            Confinement::Unconfined,
            "unrestricted runs the shell unsandboxed"
        );
    }

    /// `[shell].sandbox = false` disables confinement at every level.
    #[test]
    fn disabling_the_sandbox_unconfines_every_level() {
        let cwd = workspace::cwd_for_test();
        let scope = workspace::WriteScope::confined(vec![]);
        for level in [
            Permission::None,
            Permission::Read,
            Permission::Workspace,
            Permission::Unrestricted,
        ] {
            assert_eq!(
                Confinement::resolve(false, level, &scope, &cwd),
                Confinement::Unconfined
            );
        }
    }

    /// A `workspace` whose roots all failed to resolve grants nothing, rather than everything.
    #[test]
    fn a_workspace_with_no_resolvable_root_writes_nowhere() {
        let temp = tempfile::tempdir().expect("tempdir");
        let missing = temp.path().join("deleted-under-us");
        let cwd = workspace::SharedCwd::new(missing);
        let scope = workspace::WriteScope::confined(vec![]);

        let confinement = Confinement::resolve(true, Permission::Workspace, &scope, &cwd);
        assert!(confinement.is_sandboxed(), "it must still be sandboxed");
        assert!(
            // Not `is_sandboxed()`, which holds by construction for any `Workspace(_)` and so
            // said nothing. What matters is that the root list came back empty, because that is
            // what every dialect turns into "no write may land anywhere".
            confinement.writable().is_empty(),
            "and must grant no writable root, which is read-only in effect"
        );
    }
}
