//! Seatbelt: the SBPL profile handed to `sandbox-exec`, built per write scope.

/// Path of the macOS `sandbox-exec` binary. Hardcoded (not PATH-searched) so a hostile `PATH` entry
/// can't shadow it with a wrapper that drops the sandbox.
#[cfg(target_os = "macos")]
pub(crate) const SANDBOX_EXEC_PATH: &str = "/usr/bin/sandbox-exec";
/// SBPL profile for the macOS sandbox at the `read` level. Modeled after Codex's hardened Seatbelt
/// profile (Apache 2.0; see attribution inside the policy), which is itself inspired by Chrome's
/// renderer sandbox.
///
/// Threat-model parity with Linux Bubblewrap:
/// - Filesystem read-only: `(deny default)` denies writes; only `/dev/null` and PTY device nodes
///   get write access for legitimate shell behavior.
/// - IPC mutation blocked: `mach-lookup` is denied by default; only a curated allow-list of safe
///   Mach services is whitelisted. Mutation services (`com.apple.launchd`,
///   `com.apple.pasteboard.1`, `com.apple.launchservicesd`, the cfprefsd *write* path via
///   `user-preference-write`) are NOT in the allow-list.
/// - Network allowed: outbound BSD sockets, DNS resolution, TLS trust evaluation, and proxy/network
///   configuration reads are explicitly permitted.
#[cfg_attr(
    not(target_os = "macos"),
    allow(dead_code, reason = "read only on macOS; the tests run everywhere")
)]
pub(crate) const SANDBOX_PROFILE_READONLY: &str = r#"
; Vendored from Codex (Apache 2.0 License):
;   github.com/openai/codex/blob/main/codex-rs/sandboxing/src/seatbelt_base_policy.sbpl
;   github.com/openai/codex/blob/main/codex-rs/sandboxing/src/seatbelt_network_policy.sbpl
; The base policy is itself inspired by Chrome's renderer sandbox:
;   https://source.chromium.org/chromium/chromium/src/+/main:sandbox/policy/mac/common.sb

(version 1)

; start with closed-by-default
(deny default)

; broad filesystem read: agent needs to read arbitrary files at the read level
(allow file-read*)
(allow file-test-existence)
(allow file-ioctl)
(allow file-map-executable)
(allow file-read-metadata)

; child processes inherit the policy of their parent
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))

; process-info
(allow process-info* (target same-sandbox))

; /dev/null writes are universally legitimate for shell redirects
(allow file-write-data
  (require-all
    (path "/dev/null")
    (vnode-type CHARACTER-DEVICE)))

; sysctls permitted (CPU / kernel info reads)
(allow sysctl-read
  (sysctl-name "hw.activecpu")
  (sysctl-name "hw.busfrequency_compat")
  (sysctl-name "hw.byteorder")
  (sysctl-name "hw.cacheconfig")
  (sysctl-name "hw.cachelinesize_compat")
  (sysctl-name "hw.cpufamily")
  (sysctl-name "hw.cpufrequency_compat")
  (sysctl-name "hw.cputype")
  (sysctl-name "hw.l1dcachesize_compat")
  (sysctl-name "hw.l1icachesize_compat")
  (sysctl-name "hw.l2cachesize_compat")
  (sysctl-name "hw.l3cachesize_compat")
  (sysctl-name "hw.logicalcpu_max")
  (sysctl-name "hw.machine")
  (sysctl-name "hw.model")
  (sysctl-name "hw.memsize")
  (sysctl-name "hw.ncpu")
  (sysctl-name "hw.nperflevels")
  (sysctl-name-prefix "hw.optional.arm.")
  (sysctl-name-prefix "hw.optional.armv8_")
  (sysctl-name "hw.packages")
  (sysctl-name "hw.pagesize_compat")
  (sysctl-name "hw.pagesize")
  (sysctl-name "hw.physicalcpu")
  (sysctl-name "hw.physicalcpu_max")
  (sysctl-name "hw.logicalcpu")
  (sysctl-name "hw.cpufrequency")
  (sysctl-name "hw.tbfrequency_compat")
  (sysctl-name "hw.vectorunit")
  (sysctl-name "machdep.cpu.brand_string")
  (sysctl-name "kern.argmax")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.maxfilesperproc")
  (sysctl-name "kern.maxproc")
  (sysctl-name "kern.osproductversion")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.osvariant_status")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.secure_kernel")
  (sysctl-name "kern.usrstack64")
  (sysctl-name "kern.version")
  (sysctl-name "sysctl.proc_cputype")
  (sysctl-name "vm.loadavg")
  (sysctl-name-prefix "hw.perflevel")
  (sysctl-name-prefix "kern.proc.pgrp.")
  (sysctl-name-prefix "kern.proc.pid.")
  (sysctl-name-prefix "net.routetable.")
)

; Java reads some CPU info via a misclassified "sysctl-write"
(allow sysctl-write
  (sysctl-name "kern.grade_cputype"))

; IOKit
(allow iokit-open
  (iokit-registry-entry-class "RootDomainUserClient"))

; Python multiprocessing
(allow ipc-posix-sem)

; PyTorch/libomp register OpenMP runtimes
(allow ipc-posix-shm-read-data
  ipc-posix-shm-write-create
  ipc-posix-shm-write-unlink
  (ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))

; power management queries
(allow mach-lookup
  (global-name "com.apple.PowerManagement.control"))

; PTYs (interactive shell behavior)
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write*
  (require-all
    (regex #"^/dev/ttys[0-9]+")
    (extension "com.apple.sandbox.pty")))
(allow file-ioctl (regex #"^/dev/ttys[0-9]+"))

; read-only user preferences (writes are blocked by deny default since
; we do NOT allow `user-preference-write`)
(allow ipc-posix-shm-read* (ipc-posix-name-prefix "apple.cfprefs."))
(allow mach-lookup
  (global-name "com.apple.cfprefsd.daemon")
  (global-name "com.apple.cfprefsd.agent")
  (local-name "com.apple.cfprefsd.agent"))
(allow user-preference-read)

; ====== network rules ======
; AF_SYSTEM control sockets used by some platform helpers.
(allow system-socket
  (require-all
    (socket-domain AF_SYSTEM)
    (socket-protocol 2)))

; Outbound BSD sockets (curl, http clients, etc.)
;
; Outbound only. This carried `(allow network-bind (local ip "*:0"))` and the matching
; `network-inbound`, vendored from Codex, and a current macOS rejects `"*:0"` outright:
; `sandbox-exec: invalid port in network address`. That is a *parse* failure, so the whole profile
; is refused and every read-level command exits 65 rather than running confined -- the level was
; entirely broken on macOS, not merely narrowed. Dropped rather than respelled because the right
; spelling cannot be confirmed without a macOS host, and because the read level's stated network need is
; outbound (`curl http://x | pdftotext`); nothing in it binds a listening socket.
(allow network-outbound)

; Services needed for hostname lookup, TLS trust evaluation, proxy config.
(allow mach-lookup
  (global-name "com.apple.bsd.dirhelper")
  (global-name "com.apple.system.opendirectoryd.membership")
  (global-name "com.apple.SecurityServer")
  (global-name "com.apple.networkd")
  (global-name "com.apple.ocspd")
  (global-name "com.apple.trustd.agent")
  (global-name "com.apple.SystemConfiguration.DNSConfiguration")
  (global-name "com.apple.SystemConfiguration.configd")
  (global-name "com.apple.mDNSResponder"))

(allow sysctl-read
  (sysctl-name-regex #"^net.routetable"))
"#;
/// Build the SBPL profile and the `-D` parameter arguments for one confinement.
///
/// The read-only profile is used verbatim when `writable` is empty. Otherwise one `(allow
/// file-write* (subpath (param "MEKA_WRITABLE_n")))` clause is appended per root. After those, one
/// `(deny ...)` clause per directory in `private` hides meka's own config and store from the
/// command; SBPL takes the last matching rule, so a deny appended last beats both the global read
/// allow and a write allow for a root that happens to contain it.
///
/// Roots travel as **parameters** rather than interpolated into the profile text, which is what
/// Codex does and for the same reason: SBPL string literals need escaping, and a path containing a
/// quote or a backslash would otherwise either break the profile or, worse, change what it matches.
/// A parameter is passed out of band and needs no quoting at all.
///
/// Compiled on every platform, and gated only at the call site in `tools::shell`. Nothing in here
/// touches a macOS API: it builds a string and a list of `OsString`s. Gating the function meant no
/// local target compiled it and no test ran it, so the one piece of macOS logic meka could check
/// anywhere was the one piece nothing checked. `sandbox-exec` itself stays macOS-only and stays
/// genuinely unverified; this at least shrinks that to the exec call. Same `cfg_attr` shape as
/// `POWERSHELL_UTF8_PRELUDE` above.
#[cfg_attr(
    not(target_os = "macos"),
    allow(dead_code, reason = "called only on macOS; the tests run everywhere")
)]
pub(crate) fn sandbox_profile_for(
    writable: &[std::path::PathBuf],
    private: &[std::path::PathBuf],
) -> (String, Vec<std::ffi::OsString>) {
    let mut profile = String::from(SANDBOX_PROFILE_READONLY);
    let mut params = Vec::new();
    for (index, root) in writable.iter().enumerate() {
        let key = format!("MEKA_WRITABLE_{index}");
        profile.push_str(&format!(
            "\n(allow file-write* (subpath (param \"{key}\")))\n"
        ));
        // The value is assembled as an `OsString` and the path pushed in whole, so a root that is
        // not valid UTF-8 is passed through byte-for-byte rather than being dropped. `Command::arg`
        // takes an `OsStr`, so nothing downstream needs it to be text. Only the key, which meka
        // authors itself, is built as a string.
        let mut param = std::ffi::OsString::from(&key);
        param.push("=");
        param.push(root.as_os_str());
        params.push(std::ffi::OsString::from("-D"));
        params.push(param);
    }
    // After the write allows, because SBPL takes the last matching rule: a deny appended here beats
    // the global read allow and a write allow for a root that happens to contain the directory.
    for (index, directory) in private.iter().enumerate() {
        let key = format!("MEKA_PRIVATE_{index}");
        profile.push_str(&format!(
            "\n(deny file-read* file-write* file-test-existence (subpath (param \"{key}\")))\n"
        ));
        let mut param = std::ffi::OsString::from(&key);
        param.push("=");
        param.push(directory.as_os_str());
        params.push(std::ffi::OsString::from("-D"));
        params.push(param);
    }
    (profile, params)
}
