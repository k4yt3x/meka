//! Filesystem sandboxing for read-only command execution.
//!
//! On Linux, uses Landlock LSM (kernel 5.13+) to restrict child processes
//! to read-only filesystem access. On macOS, uses `sandbox-exec`. On Windows,
//! spawns the child with a duplicated primary token dropped to Low integrity
//! via `SetTokenInformation(TokenIntegrityLevel, …)`; this blocks writes to
//! anything outside the documented Low-integrity surface (the user's home
//! directory, `%APPDATA%`, Program Files, system dirs, etc.). On unsupported
//! platforms, sandboxing is unavailable and shell execution is gated by the
//! permission level alone.

#[derive(Debug, Clone, Copy)]
pub enum SandboxCapability {
    #[cfg(target_os = "linux")]
    Landlock {
        abi_version: i32,
    },
    #[cfg(target_os = "macos")]
    SandboxExec,
    #[cfg(target_os = "windows")]
    LowIntegrity,
    Unavailable,
}

pub fn detect() -> SandboxCapability {
    #[cfg(target_os = "linux")]
    {
        if let Some(version) = detect_landlock() {
            return SandboxCapability::Landlock {
                abi_version: version,
            };
        }
    }

    #[cfg(target_os = "macos")]
    {
        if std::path::Path::new("/usr/bin/sandbox-exec").exists() {
            return SandboxCapability::SandboxExec;
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Token-integrity APIs are available on every supported Windows
        // version (7+). No runtime probe is needed.
        return SandboxCapability::LowIntegrity;
    }

    #[allow(unreachable_code)]
    SandboxCapability::Unavailable
}

#[cfg(target_os = "linux")]
fn detect_landlock() -> Option<i32> {
    let version = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<u8>(),
            0usize,
            LANDLOCK_CREATE_RULESET_VERSION,
        )
    };
    if version >= 1 {
        Some(version as i32)
    } else {
        None
    }
}

/// Apply Landlock read-only restrictions to the current process.
///
/// # Safety
///
/// This function uses raw syscalls and must only be called in a `pre_exec`
/// context (after fork, before exec) where the process is single-threaded.
/// All operations are async-signal-safe (syscalls only, no heap allocation).
#[cfg(target_os = "linux")]
pub unsafe fn apply_landlock_readonly(abi_version: i32) -> Result<(), i32> {
    unsafe {
        // PR_SET_NO_NEW_PRIVS is required for unprivileged Landlock usage
        if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
            return Err(*libc::__errno_location());
        }

        let attr = LandlockRulesetAttr {
            handled_access_fs: handled_access_for_abi(abi_version),
            handled_access_net: 0,
            scoped: 0,
        };

        let ruleset_fd = libc::syscall(
            libc::SYS_landlock_create_ruleset,
            &attr as *const LandlockRulesetAttr,
            std::mem::size_of::<LandlockRulesetAttr>(),
            0u32,
        ) as i32;
        if ruleset_fd < 0 {
            return Err(*libc::__errno_location());
        }

        // Allow read + execute for the entire filesystem
        let root_fd = libc::open(c"/".as_ptr(), libc::O_PATH | libc::O_CLOEXEC);
        if root_fd < 0 {
            libc::close(ruleset_fd);
            return Err(*libc::__errno_location());
        }

        let path_beneath = LandlockPathBeneathAttr {
            allowed_access: LANDLOCK_ACCESS_FS_EXECUTE
                | LANDLOCK_ACCESS_FS_READ_FILE
                | LANDLOCK_ACCESS_FS_READ_DIR,
            parent_fd: root_fd,
        };

        let ret = libc::syscall(
            libc::SYS_landlock_add_rule,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            &path_beneath as *const LandlockPathBeneathAttr,
            0u32,
        );
        libc::close(root_fd);
        if ret < 0 {
            libc::close(ruleset_fd);
            return Err(*libc::__errno_location());
        }

        let ret = libc::syscall(libc::SYS_landlock_restrict_self, ruleset_fd, 0u32);
        libc::close(ruleset_fd);
        if ret < 0 {
            return Err(*libc::__errno_location());
        }

        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn handled_access_for_abi(abi_version: i32) -> u64 {
    let mut access = LANDLOCK_ACCESS_FS_EXECUTE
        | LANDLOCK_ACCESS_FS_WRITE_FILE
        | LANDLOCK_ACCESS_FS_READ_FILE
        | LANDLOCK_ACCESS_FS_READ_DIR
        | LANDLOCK_ACCESS_FS_REMOVE_DIR
        | LANDLOCK_ACCESS_FS_REMOVE_FILE
        | LANDLOCK_ACCESS_FS_MAKE_CHAR
        | LANDLOCK_ACCESS_FS_MAKE_DIR
        | LANDLOCK_ACCESS_FS_MAKE_REG
        | LANDLOCK_ACCESS_FS_MAKE_SOCK
        | LANDLOCK_ACCESS_FS_MAKE_FIFO
        | LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | LANDLOCK_ACCESS_FS_MAKE_SYM;
    if abi_version >= 2 {
        access |= LANDLOCK_ACCESS_FS_REFER;
    }
    if abi_version >= 3 {
        access |= LANDLOCK_ACCESS_FS_TRUNCATE;
    }
    // ABI v4 added network access flags (BIND_TCP, CONNECT_TCP), not filesystem flags
    if abi_version >= 5 {
        access |= LANDLOCK_ACCESS_FS_IOCTL_DEV;
    }
    access
}

// Landlock constants
#[cfg(target_os = "linux")]
const LANDLOCK_CREATE_RULESET_VERSION: u32 = 1 << 0;
#[cfg(target_os = "linux")]
const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;

#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_EXECUTE: u64 = 1 << 0;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_WRITE_FILE: u64 = 1 << 1;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_READ_FILE: u64 = 1 << 2;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_READ_DIR: u64 = 1 << 3;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_REMOVE_DIR: u64 = 1 << 4;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_REMOVE_FILE: u64 = 1 << 5;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_CHAR: u64 = 1 << 6;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_DIR: u64 = 1 << 7;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_REG: u64 = 1 << 8;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_SOCK: u64 = 1 << 9;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_FIFO: u64 = 1 << 10;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_BLOCK: u64 = 1 << 11;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_MAKE_SYM: u64 = 1 << 12;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_REFER: u64 = 1 << 13;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_TRUNCATE: u64 = 1 << 14;
#[cfg(target_os = "linux")]
const LANDLOCK_ACCESS_FS_IOCTL_DEV: u64 = 1 << 15;

// Landlock kernel structs (stack-allocated, no heap)
#[cfg(target_os = "linux")]
#[repr(C)]
struct LandlockRulesetAttr {
    handled_access_fs: u64,
    handled_access_net: u64,
    scoped: u64,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct LandlockPathBeneathAttr {
    allowed_access: u64,
    parent_fd: i32,
}

// macOS sandbox profile
#[cfg(target_os = "macos")]
pub const SANDBOX_PROFILE_READONLY: &str =
    "(version 1)(allow default)(deny file-write*)(deny file-write-setugid)";

/// PowerShell prelude that switches `$OutputEncoding` and
/// `[Console]::OutputEncoding` to UTF-8. See
/// [`wrap_command_with_utf8_output`] for why this is necessary.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
const POWERSHELL_UTF8_PRELUDE: &str = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;\
     $OutputEncoding=[System.Text.Encoding]::UTF8;";

/// Prepend the UTF-8 encoding prelude to a PowerShell command. Used by
/// both the sandboxed and non-sandboxed Windows `execute_command` paths
/// so pipe output is always decoded as UTF-8 on the Rust side regardless
/// of the console's legacy code page.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
pub fn wrap_command_with_utf8_output(command: &str) -> String {
    let mut wrapped = String::with_capacity(POWERSHELL_UTF8_PRELUDE.len() + command.len() + 1);
    wrapped.push_str(POWERSHELL_UTF8_PRELUDE);
    wrapped.push(' ');
    wrapped.push_str(command);
    wrapped
}

/// Quote a single command-line argument per Windows `CommandLineToArgvW`
/// rules. Mirrors the algorithm used by `std::process::Command` on Windows.
///
/// This is the correct encoding for any program that parses its command line
/// with `CommandLineToArgvW` — including `powershell.exe`, which is what the
/// Low-integrity sandbox invokes. It is **not** the correct encoding for
/// `cmd.exe /C` (cmd treats `\` literally); don't apply this to cmd command
/// bodies.
///
/// Compiled on every platform even though the rules are Windows-specific:
/// the implementation is pure string manipulation, so unit tests run on
/// Linux/macOS without an `#[cfg(target_os = "windows")]` gate (the
/// `cfg_attr` below just silences the dead-code warning off-Windows).
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
pub fn quote_command_arg(arg: &str) -> String {
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
    // Any trailing backslashes must be doubled so the closing quote is
    // not escaped by them.
    for _ in 0..(pending_backslashes * 2) {
        quoted.push('\\');
    }
    quoted.push('"');
    quoted
}

#[cfg(target_os = "windows")]
pub use windows_impl::spawn_low_integrity_command;

#[cfg(target_os = "windows")]
mod windows_impl {
    use std::fs::File;
    use std::mem;
    use std::os::windows::io::FromRawHandle;
    use std::os::windows::process::ExitStatusExt;
    use std::process::ExitStatus;
    use std::ptr;

    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_PRIVILEGE_NOT_HELD, GENERIC_READ, HANDLE, HANDLE_FLAG_INHERIT,
        INVALID_HANDLE_VALUE, LocalFree, SetHandleInformation, TRUE, WAIT_OBJECT_0,
    };
    use windows_sys::Win32::Security::Authorization::ConvertStringSidToSidW;
    use windows_sys::Win32::Security::{
        AdjustTokenPrivileges, DuplicateTokenEx, SECURITY_ATTRIBUTES, SID_AND_ATTRIBUTES,
        SecurityAnonymous, SetTokenInformation, TOKEN_ADJUST_DEFAULT, TOKEN_ADJUST_PRIVILEGES,
        TOKEN_ASSIGN_PRIMARY, TOKEN_DUPLICATE, TOKEN_MANDATORY_LABEL, TOKEN_QUERY,
        TokenIntegrityLevel, TokenPrimary,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    use windows_sys::Win32::System::Pipes::CreatePipe;
    use windows_sys::Win32::System::Threading::{
        CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT, CreateProcessAsUserW,
        CreateProcessWithTokenW, DeleteProcThreadAttributeList, EXTENDED_STARTUPINFO_PRESENT,
        GetCurrentProcess, GetExitCodeProcess, INFINITE, InitializeProcThreadAttributeList,
        LPPROC_THREAD_ATTRIBUTE_LIST, OpenProcessToken, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
        PROCESS_INFORMATION, STARTF_USESTDHANDLES, STARTUPINFOEXW, STARTUPINFOW, TerminateProcess,
        UpdateProcThreadAttribute, WaitForSingleObject,
    };

    // SE_GROUP_INTEGRITY isn't exported by the `Win32_Security` feature in
    // windows-sys 0.59; define it locally. See
    // <https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-sid_and_attributes>.
    const SE_GROUP_INTEGRITY: u32 = 0x0000_0020;

    /// RAII wrapper for a Win32 `HANDLE`. Closes the handle on drop, unless
    /// ownership is transferred out via [`OwnedHandle::into_raw`], which
    /// invalidates the wrapper. `!Send`/`!Sync` for raw pointers is
    /// overridden here because the underlying kernel object is process-wide
    /// and thread-safe to close from any thread; we serialize usage through
    /// the owning struct.
    struct OwnedHandle(HANDLE);

    unsafe impl Send for OwnedHandle {}
    unsafe impl Sync for OwnedHandle {}

    impl OwnedHandle {
        fn as_raw(&self) -> HANDLE {
            self.0
        }

        /// Consume the wrapper and return the raw handle, suppressing the
        /// Drop-time `CloseHandle`. Use when the handle is being transferred
        /// into another owner (e.g. `File::from_raw_handle`, or into the
        /// `SandboxedChild` long-lived handles).
        fn into_raw(mut self) -> HANDLE {
            let h = self.0;
            self.0 = INVALID_HANDLE_VALUE;
            h
        }
    }

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            if !self.0.is_null() && self.0 != INVALID_HANDLE_VALUE {
                // SAFETY: We own this handle and haven't already closed it.
                // After Drop the struct is gone so no double-close is possible.
                unsafe {
                    CloseHandle(self.0);
                }
            }
        }
    }

    /// Child process spawned under a Low-integrity token. `stdout`/`stderr`
    /// are anonymous pipes wrapped in [`File`] (convertible to tokio async
    /// readers via `tokio::fs::File::from_std`). `wait_blocking` / `kill` run
    /// synchronous Win32 calls; call them from `tokio::task::spawn_blocking`.
    pub struct SandboxedChild {
        process: OwnedHandle,
        stdout: Option<File>,
        stderr: Option<File>,
    }

    impl SandboxedChild {
        pub fn take_stdout(&mut self) -> Option<File> {
            self.stdout.take()
        }

        pub fn take_stderr(&mut self) -> Option<File> {
            self.stderr.take()
        }

        /// Block the current thread until the child exits. Must be called
        /// from a blocking context (e.g. `tokio::task::spawn_blocking`).
        pub fn wait_blocking(&self) -> std::io::Result<ExitStatus> {
            // SAFETY: `process` is a valid open process HANDLE until Drop.
            unsafe {
                let rc = WaitForSingleObject(self.process.as_raw(), INFINITE);
                if rc != WAIT_OBJECT_0 {
                    return Err(std::io::Error::last_os_error());
                }
                let mut exit_code: u32 = 0;
                if GetExitCodeProcess(self.process.as_raw(), &mut exit_code) == 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(ExitStatus::from_raw(exit_code))
            }
        }

        /// Terminate the child process. Returns success even if the process
        /// had already exited; Win32 distinguishes these but the shell tool
        /// treats both as "gone".
        pub fn kill(&self) -> std::io::Result<()> {
            // SAFETY: `process` is a valid open process HANDLE until Drop.
            unsafe {
                if TerminateProcess(self.process.as_raw(), 1) == 0 {
                    let err = std::io::Error::last_os_error();
                    // ERROR_ACCESS_DENIED (5) is returned when the process
                    // already exited — treat as success.
                    if err.raw_os_error() == Some(5) {
                        return Ok(());
                    }
                    return Err(err);
                }
                Ok(())
            }
        }
    }

    /// Spawn `powershell.exe -NoProfile -NonInteractive -Command <command>`
    /// under a Low-integrity token. Stdout and stderr are captured via
    /// anonymous pipes; stdin is not connected.
    ///
    /// PowerShell parses its command line per `CommandLineToArgvW` rules, so
    /// the user command is encoded with the standard argv-escape helper —
    /// embedded `"`, `\`, spaces, and shell metacharacters all pass through
    /// unmangled. `-NoProfile` skips user profile scripts (fast startup, no
    /// unrelated side effects); `-NonInteractive` makes the child fail fast
    /// on any prompt instead of hanging on stdin.
    ///
    /// Returns [`std::io::Error`] mirroring the underlying Win32 call so the
    /// shell tool can surface a standard error message.
    pub fn spawn_low_integrity_command(command: &str) -> std::io::Result<SandboxedChild> {
        // Embedded NULs would silently truncate the CreateProcess command
        // line (Win32 treats the UTF-16 command-line buffer as a C string).
        // Agent-driven commands shouldn't contain these, but fail loudly
        // rather than silently execute a truncated prefix.
        if command.contains('\0') {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "command contains embedded NUL byte",
            ));
        }

        // Force UTF-8 output before running the user's command. PowerShell
        // 5.1 (the inbox version we invoke as `powershell.exe`) defaults
        // `[Console]::OutputEncoding` to the system's legacy OEM code
        // page — CP 437 / 1252 on most English installs — which mangles
        // non-ASCII output (日本語 → `???`) when the process writes to a
        // redirected pipe like ours. Prefixing every script with a UTF-8
        // encoding switch makes output round-trip losslessly regardless
        // of the host's console configuration.
        let wrapped_command = super::wrap_command_with_utf8_output(command);

        let mut cmd_line = String::from(r#""powershell.exe" -NoProfile -NonInteractive -Command "#);
        cmd_line.push_str(&super::quote_command_arg(&wrapped_command));

        // SAFETY: All Win32 calls below are documented and we check return
        // values. Handles are wrapped in `OwnedHandle` to close on drop.
        // Pipe handles transfer ownership into the spawned child (for the
        // write ends) or into the returned `File` (for the read ends).
        unsafe {
            // 1. Open our own process token and duplicate it as a primary
            //    token we can modify. The duplicate is what we'll drop to
            //    Low integrity — we must NOT mutate our own token.
            let mut self_token: HANDLE = ptr::null_mut();
            if OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_ADJUST_DEFAULT,
                &mut self_token,
            ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }
            let self_token = OwnedHandle(self_token);

            let mut low_token: HANDLE = ptr::null_mut();
            // `SecurityAnonymous` is the least-capable impersonation level and
            // is the correct "don't care" value when the target is a primary
            // token — per Win32 docs the parameter is only consulted for
            // impersonation tokens, but some kernel versions have historically
            // honored it, so pick the safest constant.
            if DuplicateTokenEx(
                self_token.as_raw(),
                TOKEN_ASSIGN_PRIMARY
                    | TOKEN_DUPLICATE
                    | TOKEN_QUERY
                    | TOKEN_ADJUST_DEFAULT
                    | TOKEN_ADJUST_PRIVILEGES,
                ptr::null(),
                SecurityAnonymous,
                TokenPrimary,
                &mut low_token,
            ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }
            let low_token = OwnedHandle(low_token);

            // 2. Strip all privileges from the duplicate before anything
            //    else touches it. Integrity-level enforcement already makes
            //    most privileges inert against Medium+ resources, but
            //    defense-in-depth: a Low-integrity token that still claims
            //    (say) `SeShutdownPrivilege` is a sharper edge than one
            //    that has none at all. Passing DisableAllPrivileges=TRUE
            //    with a NULL NewState disables every privilege on the token.
            if AdjustTokenPrivileges(
                low_token.as_raw(),
                TRUE,
                ptr::null(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }

            // 3. Build the Low-integrity SID via ConvertStringSidToSidW and
            //    point a TOKEN_MANDATORY_LABEL at it. The SID buffer is
            //    allocated by the OS and must be released via LocalFree.
            let sid_str: Vec<u16> = "S-1-16-4096\0".encode_utf16().collect();
            let mut low_sid: *mut core::ffi::c_void = ptr::null_mut();
            if ConvertStringSidToSidW(sid_str.as_ptr(), &mut low_sid) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            let _sid_guard = LocalFreeGuard(low_sid);

            let label = TOKEN_MANDATORY_LABEL {
                Label: SID_AND_ATTRIBUTES {
                    Sid: low_sid,
                    Attributes: SE_GROUP_INTEGRITY,
                },
            };

            if SetTokenInformation(
                low_token.as_raw(),
                TokenIntegrityLevel,
                &label as *const _ as *const core::ffi::c_void,
                mem::size_of::<TOKEN_MANDATORY_LABEL>() as u32,
            ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }

            // 4. Create two anonymous pipes with **non-inheritable** handles.
            //    We use `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` (step 6) to
            //    narrow inheritance to exactly the three handles our child
            //    needs — the inherit flag is only flipped to TRUE briefly
            //    on those three handles, not the read ends, which eliminates
            //    the classic CreatePipe→SetHandleInformation→CreateProcess
            //    race where a concurrent CreateProcess in the same process
            //    could leak the read ends to an unrelated child.
            let sa_noninherit = SECURITY_ATTRIBUTES {
                nLength: mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: ptr::null_mut(),
                bInheritHandle: 0,
            };

            let (stdout_read, stdout_write) = create_pipe(&sa_noninherit)?;
            let (stderr_read, stderr_write) = create_pipe(&sa_noninherit)?;

            // 5. Open NUL as the child's stdin. Non-inheritable; inherit
            //    flag flipped on just before CreateProcess.
            let nul_stdin = open_nul_read(&sa_noninherit)?;

            // 6. Promote the three child-bound handles to inheritable. The
            //    PROC_THREAD_ATTRIBUTE_HANDLE_LIST filter (step 7) requires
            //    each listed handle to have HANDLE_FLAG_INHERIT set.
            if SetHandleInformation(
                stdout_write.as_raw(),
                HANDLE_FLAG_INHERIT,
                HANDLE_FLAG_INHERIT,
            ) == 0
                || SetHandleInformation(
                    stderr_write.as_raw(),
                    HANDLE_FLAG_INHERIT,
                    HANDLE_FLAG_INHERIT,
                ) == 0
                || SetHandleInformation(
                    nul_stdin.as_raw(),
                    HANDLE_FLAG_INHERIT,
                    HANDLE_FLAG_INHERIT,
                ) == 0
            {
                return Err(std::io::Error::last_os_error());
            }

            // 7. Build a STARTUPINFOEXW with PROC_THREAD_ATTRIBUTE_HANDLE_LIST
            //    naming exactly the three handles we want the child to see.
            //    With bInheritHandles=TRUE and EXTENDED_STARTUPINFO_PRESENT,
            //    the child inherits *only* the listed handles even if
            //    other inheritable handles exist in this process.
            let child_handles: [HANDLE; 3] = [
                nul_stdin.as_raw(),
                stdout_write.as_raw(),
                stderr_write.as_raw(),
            ];
            let attr_list = ProcThreadAttributeList::new_with_handle_list(&child_handles)?;

            let mut startup: STARTUPINFOEXW = mem::zeroed();
            startup.StartupInfo.cb = mem::size_of::<STARTUPINFOEXW>() as u32;
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            startup.StartupInfo.hStdInput = nul_stdin.as_raw();
            startup.StartupInfo.hStdOutput = stdout_write.as_raw();
            startup.StartupInfo.hStdError = stderr_write.as_raw();
            startup.lpAttributeList = attr_list.as_raw();

            let mut proc_info: PROCESS_INFORMATION = mem::zeroed();

            // 8. Spawn. The child inherits only the three listed handles.
            let spawn_result = create_process_low_integrity(
                low_token.as_raw(),
                &cmd_line,
                &startup,
                &mut proc_info,
            );

            // Parent no longer needs the child-side write ends or the NUL
            // stdin handle regardless of success/failure. Dropping the
            // OwnedHandle wrappers closes them. On success, closing the
            // write ends ensures the parent's read end sees EOF when the
            // child exits. Drop before any early-return so the handles
            // aren't leaked if later steps fail.
            drop(stdout_write);
            drop(stderr_write);
            drop(nul_stdin);
            drop(attr_list);

            spawn_result?;

            // We don't need the main thread handle — close it immediately.
            if !proc_info.hThread.is_null() {
                CloseHandle(proc_info.hThread);
            }

            // Transfer pipe read ends into owned `File`s.
            // `File::from_raw_handle` takes ownership of the HANDLE;
            // `OwnedHandle::into_raw` suppresses the wrapper's Drop.
            let stdout_handle = stdout_read.into_raw();
            let stderr_handle = stderr_read.into_raw();

            Ok(SandboxedChild {
                process: OwnedHandle(proc_info.hProcess),
                stdout: Some(File::from_raw_handle(stdout_handle as _)),
                stderr: Some(File::from_raw_handle(stderr_handle as _)),
            })
        }
    }

    /// Create an anonymous pipe using the supplied SECURITY_ATTRIBUTES.
    ///
    /// A 1 MiB buffer hint is passed to `CreatePipe`. This is belt-and-
    /// braces with the concurrent draining in `run_windows_low_integrity`:
    /// even if the drain task is momentarily starved, the child has a MiB
    /// of slack before it blocks in `WriteFile`.
    unsafe fn create_pipe(sa: &SECURITY_ATTRIBUTES) -> std::io::Result<(OwnedHandle, OwnedHandle)> {
        const PIPE_BUFFER_SIZE: u32 = 1 << 20;
        let mut read: HANDLE = ptr::null_mut();
        let mut write: HANDLE = ptr::null_mut();
        // SAFETY: CreatePipe writes two HANDLEs through the provided pointers
        // on success. SECURITY_ATTRIBUTES is a valid initialized struct.
        if unsafe { CreatePipe(&mut read, &mut write, sa, PIPE_BUFFER_SIZE) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok((OwnedHandle(read), OwnedHandle(write)))
    }

    /// Open the `NUL` device for read. Inherit flag is left unset by the
    /// caller's `SECURITY_ATTRIBUTES`; promote via `SetHandleInformation`
    /// right before the handle is passed to `CreateProcess`. The child sees
    /// immediate EOF on any read — the correct "no stdin" primitive on
    /// Windows, equivalent to `/dev/null` on Unix.
    unsafe fn open_nul_read(sa: &SECURITY_ATTRIBUTES) -> std::io::Result<OwnedHandle> {
        let path: Vec<u16> = "NUL\0".encode_utf16().collect();
        // SAFETY: `path` is NUL-terminated; `sa` is a valid initialized
        // SECURITY_ATTRIBUTES owned by the caller for the duration of the call.
        let h = unsafe {
            CreateFileW(
                path.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                sa as *const SECURITY_ATTRIBUTES,
                OPEN_EXISTING,
                0,
                ptr::null_mut(),
            )
        };
        if h == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        Ok(OwnedHandle(h))
    }

    /// Create a process under the Low-integrity token. Tries
    /// `CreateProcessAsUserW` first (the usual path); on
    /// `ERROR_PRIVILEGE_NOT_HELD` — which happens when the current user
    /// lacks `SE_INCREASE_QUOTA_NAME` (common on locked-down corp-managed
    /// accounts) — falls back to `CreateProcessWithTokenW`, which requires
    /// the more broadly-granted `SE_IMPERSONATE_NAME` instead.
    ///
    /// The command line is re-encoded to UTF-16 for *each* attempt: Win32
    /// documents `lpCommandLine` as in/out, and the first call may mutate
    /// the buffer (typically inserting a NUL to split argv[0]) before
    /// failing, so re-using the same buffer between attempts could hand
    /// the fallback a corrupted string.
    ///
    /// Both invocations pass `EXTENDED_STARTUPINFO_PRESENT` together with
    /// `STARTUPINFOEXW`, so the handle-list filter in the attribute list
    /// applies uniformly across both paths.
    unsafe fn create_process_low_integrity(
        token: HANDLE,
        cmd_line_utf8: &str,
        startup: &STARTUPINFOEXW,
        proc_info: &mut PROCESS_INFORMATION,
    ) -> std::io::Result<()> {
        let creation_flags =
            CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT;
        let startup_ptr = startup as *const STARTUPINFOEXW as *const STARTUPINFOW;

        // Build a scrubbed UTF-16 environment block once. Passing this for
        // both `CreateProcessAsUserW` and the `CreateProcessWithTokenW`
        // fallback ensures the sandboxed child never sees the agent's
        // `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or any `*_TOKEN` /
        // `*_SECRET` variable — a Low-integrity child can still open
        // outbound sockets, so a leaked key in env is a live exfil vector.
        let mut env_block = build_scrubbed_env_block_utf16();
        let env_ptr = env_block.as_mut_ptr() as *const core::ffi::c_void;

        let mut cmd_line_utf16: Vec<u16> = cmd_line_utf8
            .encode_utf16()
            .chain(std::iter::once(0u16))
            .collect();

        // SAFETY: All pointers are valid for the duration of the call per the
        // caller's obligations. Win32 writes to `proc_info` on success.
        let ok = unsafe {
            CreateProcessAsUserW(
                token,
                ptr::null(),
                cmd_line_utf16.as_mut_ptr(),
                ptr::null(),
                ptr::null(),
                TRUE,
                creation_flags,
                env_ptr,
                ptr::null(),
                startup_ptr,
                proc_info,
            )
        };
        if ok != 0 {
            return Ok(());
        }

        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(ERROR_PRIVILEGE_NOT_HELD as i32) {
            return Err(err);
        }

        tracing::warn!(
            "CreateProcessAsUserW denied (SE_INCREASE_QUOTA_NAME not held); \
             falling back to CreateProcessWithTokenW. The child is still spawned \
             under the Low-integrity token with the same scrubbed environment \
             block; semantics differ slightly (no custom process/thread \
             SECURITY_ATTRIBUTES)."
        );

        // Rebuild the command-line buffer — the previous call may have
        // mutated it before failing (Win32 documents lpCommandLine as in/out).
        let mut cmd_line_utf16_retry: Vec<u16> = cmd_line_utf8
            .encode_utf16()
            .chain(std::iter::once(0u16))
            .collect();

        // SAFETY: same contract as CreateProcessAsUserW; the two APIs only
        // differ in their parameter list (no process/thread security attrs,
        // no bInheritHandles — inheritance is driven by the per-handle
        // `HANDLE_FLAG_INHERIT` flag plus the attribute-list filter). We
        // re-use the scrubbed environment block so the fallback path
        // doesn't accidentally regress to inheriting the parent's env.
        let ok = unsafe {
            CreateProcessWithTokenW(
                token,
                0, // dwLogonFlags: 0 means "use the token as-is"
                ptr::null(),
                cmd_line_utf16_retry.as_mut_ptr(),
                creation_flags,
                env_ptr,
                ptr::null(),
                startup_ptr,
                proc_info,
            )
        };
        // Keep `env_block` alive until after both calls complete — Win32
        // copies the contents but documents `lpEnvironment` as a pointer
        // that must be valid through the call.
        drop(env_block);
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Build a UTF-16 `NAME=VALUE\0NAME=VALUE\0\0` environment block for
    /// the sandboxed child. Starts from the parent environment and drops
    /// names that look sensitive — anything containing `TOKEN`, `SECRET`,
    /// `PASSWORD`, or `API_KEY`, plus names prefixed with known provider
    /// namespaces. Everything else passes through so PowerShell can load
    /// its built-in modules (it needs `PSModulePath`, `APPDATA`,
    /// `ProgramFiles`, and friends) without a brittle allowlist that
    /// breaks whenever Windows adds a new required variable.
    ///
    /// A Low-integrity child can still open outbound sockets, so a stray
    /// `ANTHROPIC_API_KEY` in the parent env is a live exfil vector —
    /// hence the denylist. An allowlist version was tried first but
    /// couldn't keep core cmdlets like `Write-Output` / `Measure-Object`
    /// working without enumerating half of the Windows environment.
    fn build_scrubbed_env_block_utf16() -> Vec<u16> {
        let mut block: Vec<u16> = Vec::new();
        for (name_os, value_os) in std::env::vars_os() {
            let Some(name) = name_os.to_str() else {
                continue;
            };
            let Some(value) = value_os.to_str() else {
                continue;
            };
            if is_sensitive_env_name(name) {
                continue;
            }
            append_env_entry(&mut block, name, value);
        }
        // Double-NUL terminator (each entry already ends with one NUL; we
        // need another to close the block).
        block.push(0);
        block
    }

    /// Heuristic match for variable names that commonly carry credentials.
    /// Case-insensitive substring match on a short list of markers plus
    /// a prefix match on known provider namespaces. Tuned to be
    /// aggressive on false positives (a legitimate `GITHUB_ACTOR` is
    /// dropped alongside `GITHUB_TOKEN`) because the downside of a
    /// missing env var is a confusing tool error; the downside of a
    /// leaked key is a live exfil channel.
    pub(super) fn is_sensitive_env_name(name: &str) -> bool {
        const SENSITIVE_SUBSTRINGS: &[&str] = &[
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "PASSWD",
            "API_KEY",
            "APIKEY",
            "PRIVATE_KEY",
            "BEARER",
            "CREDENTIAL",
            "SESSION_KEY",
            "ACCESS_KEY",
        ];
        const SENSITIVE_PREFIXES: &[&str] = &[
            "ANTHROPIC_",
            "OPENAI_",
            "CLAUDE_",
            "AGSH_",
            "AWS_",
            "GCP_",
            "GOOGLE_",
            "AZURE_",
            "GITHUB_",
            "GITLAB_",
            "HF_",
            "HUGGINGFACE_",
            "NPM_",
            "PYPI_",
            "CARGO_REGISTRY_",
            "DOCKER_",
        ];

        let upper = name.to_ascii_uppercase();
        SENSITIVE_SUBSTRINGS
            .iter()
            .any(|needle| upper.contains(needle))
            || SENSITIVE_PREFIXES
                .iter()
                .any(|prefix| upper.starts_with(prefix))
    }

    fn append_env_entry(block: &mut Vec<u16>, name: &str, value: &str) {
        block.extend(name.encode_utf16());
        block.push(u16::from(b'='));
        block.extend(value.encode_utf16());
        block.push(0);
    }

    /// RAII wrapper around `PROC_THREAD_ATTRIBUTE_LIST`. Owns both the
    /// attribute-list backing buffer and the HANDLE array it points into —
    /// Win32 stores the handle-list address (not a copy), so the array must
    /// outlive any `CreateProcess*` call that consumes the attribute list.
    struct ProcThreadAttributeList {
        // Both fields are kept alive for Drop. The Vec's heap buffer is the
        // attribute-list storage; `list_ptr` caches a stable mutable pointer
        // to it. The boxed handle slice is referenced (by pointer) from
        // inside the attribute-list buffer, so it must not move or drop
        // while the list is alive.
        _buffer: Vec<u8>,
        _handles: Box<[HANDLE]>,
        list_ptr: LPPROC_THREAD_ATTRIBUTE_LIST,
    }

    impl ProcThreadAttributeList {
        /// Build a one-attribute list containing a `HANDLE_LIST` attribute
        /// referencing the supplied handles. `UpdateProcThreadAttribute`
        /// stores the pointer to the handle array, not a copy — the array
        /// is boxed into the returned wrapper so it stays at a fixed
        /// address for the wrapper's lifetime.
        unsafe fn new_with_handle_list(handles: &[HANDLE]) -> std::io::Result<Self> {
            // First call: buffer=NULL, size=0 → fails with
            // ERROR_INSUFFICIENT_BUFFER but writes the required size.
            let mut size: usize = 0;
            unsafe {
                InitializeProcThreadAttributeList(ptr::null_mut(), 1, 0, &mut size);
            }
            if size == 0 {
                return Err(std::io::Error::last_os_error());
            }

            let mut buffer: Vec<u8> = vec![0; size];
            let list_ptr = buffer.as_mut_ptr() as LPPROC_THREAD_ATTRIBUTE_LIST;

            // SAFETY: `list_ptr` points to a correctly-sized buffer from the
            // previous size query, and `size` is that queried value.
            if unsafe { InitializeProcThreadAttributeList(list_ptr, 1, 0, &mut size) } == 0 {
                return Err(std::io::Error::last_os_error());
            }

            let boxed_handles: Box<[HANDLE]> = handles.to_vec().into_boxed_slice();
            let handles_bytes = std::mem::size_of_val(&*boxed_handles);

            // SAFETY: `list_ptr` was just initialized; `boxed_handles` lives
            // for 'self because it's stored in the returned wrapper; the
            // byte size passed matches the boxed array.
            if unsafe {
                UpdateProcThreadAttribute(
                    list_ptr,
                    0,
                    PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                    boxed_handles.as_ptr() as *const core::ffi::c_void,
                    handles_bytes,
                    ptr::null_mut(),
                    ptr::null(),
                )
            } == 0
            {
                let err = std::io::Error::last_os_error();
                // SAFETY: Initialize succeeded; must be paired with Delete
                // regardless of subsequent failures.
                unsafe { DeleteProcThreadAttributeList(list_ptr) };
                return Err(err);
            }

            Ok(Self {
                _buffer: buffer,
                _handles: boxed_handles,
                list_ptr,
            })
        }

        fn as_raw(&self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
            self.list_ptr
        }
    }

    impl Drop for ProcThreadAttributeList {
        fn drop(&mut self) {
            // SAFETY: constructor either fully initialized the list (and
            // stored its pointer in `list_ptr`) or returned Err (in which
            // case this Drop doesn't run).
            unsafe {
                DeleteProcThreadAttributeList(self.list_ptr);
            }
        }
    }

    /// RAII guard for a pointer allocated by the OS and freed via `LocalFree`.
    struct LocalFreeGuard(*mut core::ffi::c_void);

    impl Drop for LocalFreeGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: pointer was returned by a Win32 API that documents
                // `LocalFree` as the correct release call.
                unsafe {
                    LocalFree(self.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "windows")]
    #[test]
    fn test_is_sensitive_env_name_matches_known_secret_patterns() {
        use windows_impl::is_sensitive_env_name;
        assert!(is_sensitive_env_name("ANTHROPIC_API_KEY"));
        assert!(is_sensitive_env_name("OPENAI_API_KEY"));
        assert!(is_sensitive_env_name("GITHUB_TOKEN"));
        assert!(is_sensitive_env_name("AWS_SECRET_ACCESS_KEY"));
        assert!(is_sensitive_env_name("my_bearer_auth"));
        assert!(is_sensitive_env_name("DATABASE_PASSWORD"));
        assert!(is_sensitive_env_name("anthropic_api_key"));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_is_sensitive_env_name_allows_system_vars() {
        use windows_impl::is_sensitive_env_name;
        assert!(!is_sensitive_env_name("SystemRoot"));
        assert!(!is_sensitive_env_name("PATH"));
        assert!(!is_sensitive_env_name("PSModulePath"));
        assert!(!is_sensitive_env_name("APPDATA"));
        assert!(!is_sensitive_env_name("LOCALAPPDATA"));
        assert!(!is_sensitive_env_name("ProgramFiles"));
        assert!(!is_sensitive_env_name("USERPROFILE"));
        assert!(!is_sensitive_env_name("TEMP"));
    }

    #[test]
    fn test_detect_sandbox_capability() {
        let capability = detect();
        // Should detect something on Linux/macOS/Windows, Unavailable on others
        match capability {
            #[cfg(target_os = "linux")]
            SandboxCapability::Landlock { abi_version } => {
                assert!(abi_version >= 1);
            }
            #[cfg(target_os = "macos")]
            SandboxCapability::SandboxExec => {}
            #[cfg(target_os = "windows")]
            SandboxCapability::LowIntegrity => {}
            SandboxCapability::Unavailable => {}
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    #[test]
    fn test_wrap_command_with_utf8_output_prepends_prelude() {
        let wrapped = wrap_command_with_utf8_output("Write-Output '日本語'");
        assert!(wrapped.starts_with("[Console]::OutputEncoding="));
        assert!(wrapped.contains("$OutputEncoding=[System.Text.Encoding]::UTF8"));
        assert!(wrapped.ends_with("Write-Output '日本語'"));
        // A space must separate the prelude from the user command so
        // PowerShell doesn't glue them into one malformed statement.
        assert!(wrapped.contains("UTF8; Write-Output"));
    }

    /// Reference table covering the corners of the `CommandLineToArgvW`
    /// encoding. Cross-platform — `quote_command_arg` is pure string
    /// manipulation and has no Windows-specific runtime dependency.
    #[test]
    fn test_quote_command_arg_reference_table() {
        let cases: &[(&str, &str)] = &[
            ("cmd.exe", "cmd.exe"),
            ("", r#""""#),
            ("with space", r#""with space""#),
            ("with\ttab", "\"with\ttab\""),
            (r#"say "hi""#, r#""say \"hi\"""#),
            (r#"a\"b"#, r#""a\\\"b""#),
            (r"path with space\", r#""path with space\\""#),
            // A quote preceded by a single backslash: the backslash is
            // doubled and the quote is escaped.
            (r#"\""#, r#""\\\"""#),
            // Backslashes not adjacent to a quote pass through literally
            // (no escaping needed, no quoting needed — no special chars).
            (r"a\\b", r"a\\b"),
            // Unicode and newlines pass through. Newline counts as
            // whitespace so the argument gets quoted.
            ("日本語", "日本語"),
            ("hello world\n", "\"hello world\n\""),
        ];
        for (input, expected) in cases {
            assert_eq!(
                &quote_command_arg(input),
                expected,
                "input {:?} produced wrong quoting",
                input
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_handled_access_abi_v1() {
        let access = handled_access_for_abi(1);
        assert!(access & LANDLOCK_ACCESS_FS_WRITE_FILE != 0);
        assert!(access & LANDLOCK_ACCESS_FS_READ_FILE != 0);
        assert!(access & LANDLOCK_ACCESS_FS_REFER == 0);
        assert!(access & LANDLOCK_ACCESS_FS_TRUNCATE == 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_handled_access_abi_v3() {
        let access = handled_access_for_abi(3);
        assert!(access & LANDLOCK_ACCESS_FS_REFER != 0);
        assert!(access & LANDLOCK_ACCESS_FS_TRUNCATE != 0);
        assert!(access & LANDLOCK_ACCESS_FS_IOCTL_DEV == 0);
    }
}
