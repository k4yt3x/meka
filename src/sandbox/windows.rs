//! Windows: a restricted token, a job object and an ACL fence around the workspace, which together
//! are what confinement means here.

use std::{
    fs::File,
    mem,
    os::windows::{ffi::OsStrExt, io::FromRawHandle, process::ExitStatusExt},
    process::ExitStatus,
    ptr,
};

use windows_sys::Win32::{
    Foundation::{
        CloseHandle, ERROR_PRIVILEGE_NOT_HELD, GENERIC_READ, GENERIC_WRITE, HANDLE,
        HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE, LocalFree, SetHandleInformation, TRUE,
        WAIT_OBJECT_0,
    },
    Security::{
        ACCESS_ALLOWED_ACE, ACE_HEADER, ACL, AdjustTokenPrivileges,
        Authorization::{
            ConvertStringSidToSidW, EXPLICIT_ACCESS_W, GRANT_ACCESS, GetNamedSecurityInfoW,
            NO_MULTIPLE_TRUSTEE, REVOKE_ACCESS, SE_FILE_OBJECT, SetEntriesInAclW,
            SetNamedSecurityInfoW, TRUSTEE_IS_SID, TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
        },
        CreateRestrictedToken, DACL_SECURITY_INFORMATION, DuplicateTokenEx, EqualSid, GetAce,
        GetLengthSid, GetTokenInformation, SECURITY_ATTRIBUTES, SID_AND_ATTRIBUTES,
        SecurityAnonymous, SetTokenInformation, TOKEN_ADJUST_DEFAULT, TOKEN_ADJUST_PRIVILEGES,
        TOKEN_ASSIGN_PRIMARY, TOKEN_DUPLICATE, TOKEN_GROUPS, TOKEN_MANDATORY_LABEL, TOKEN_QUERY,
        TokenGroups, TokenIntegrityLevel, TokenPrimary,
    },
    Storage::FileSystem::{CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING},
    System::{
        Console::{
            AllocConsole, GetConsoleWindow, GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE,
            STD_OUTPUT_HANDLE, SetStdHandle,
        },
        JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_BASIC_LIMIT_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JobObjectExtendedLimitInformation, SetInformationJobObject, TerminateJobObject,
        },
        Pipes::CreatePipe,
        Threading::{
            CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT, CreateProcessAsUserW,
            CreateProcessWithTokenW, DeleteProcThreadAttributeList, EXTENDED_STARTUPINFO_PRESENT,
            GetCurrentProcess, GetExitCodeProcess, INFINITE, InitializeProcThreadAttributeList,
            LPPROC_THREAD_ATTRIBUTE_LIST, OpenProcessToken, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            PROCESS_INFORMATION, ResumeThread, STARTF_USESTDHANDLES, STARTUPINFOEXW, STARTUPINFOW,
            TerminateProcess, UpdateProcThreadAttribute, WaitForSingleObject,
        },
    },
    UI::WindowsAndMessaging::{SW_HIDE, ShowWindow},
};

// SE_GROUP_INTEGRITY isn't exported by the `Win32_Security` feature in windows-sys 0.61; define
// it locally. See
// <https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-sid_and_attributes>.
const SE_GROUP_INTEGRITY: u32 = 0x0000_0020;

/// RAII wrapper for a Win32 `HANDLE`. Closes the handle on drop, unless ownership is transferred
/// out via [`OwnedHandle::into_raw`], which invalidates the wrapper. `!Send`/`!Sync` for raw
/// pointers is overridden here because the underlying kernel object is process-wide and thread-safe
/// to close from any thread; we serialize usage through the owning struct.
struct OwnedHandle(HANDLE);

unsafe impl Send for OwnedHandle {}
unsafe impl Sync for OwnedHandle {}

impl OwnedHandle {
    fn as_raw(&self) -> HANDLE {
        self.0
    }

    /// Consume the wrapper and return the raw handle, suppressing the Drop-time `CloseHandle`. Use
    /// when the handle is being transferred into another owner (e.g. `File::from_raw_handle`, or
    /// into the `SandboxedChild` long-lived handles).
    fn into_raw(mut self) -> HANDLE {
        let h = self.0;
        self.0 = INVALID_HANDLE_VALUE;
        h
    }
}

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        if !self.0.is_null() && self.0 != INVALID_HANDLE_VALUE {
            // SAFETY: We own this handle and haven't already closed it. After Drop the struct
            // is gone so no double-close is possible.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

/// Child process spawned under a Low-integrity token. `stdout`/`stderr` are anonymous pipes
/// wrapped in [`File`] (convertible to tokio async readers via `tokio::fs::File::from_std`).
/// `wait_blocking` / `kill` run synchronous Win32 calls; call them from
/// `tokio::task::spawn_blocking`.
///
/// The child is wrapped in a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so any
/// grandchildren spawned by the user command are atomically killed when the job handle drops
/// (matching Unix's `setsid()` + `kill(-pgid, …)` semantics). `kill()` terminates the entire
/// job, not just the direct child.
pub(crate) struct SandboxedChild {
    process: OwnedHandle,
    job: OwnedHandle,
    stdout: Option<File>,
    stderr: Option<File>,
}

impl SandboxedChild {
    pub(crate) fn take_stdout(&mut self) -> Option<File> {
        self.stdout.take()
    }

    pub(crate) fn take_stderr(&mut self) -> Option<File> {
        self.stderr.take()
    }

    /// Block the current thread until the child exits. Must be called from a blocking context
    /// (e.g. `tokio::task::spawn_blocking`).
    pub(crate) fn wait_blocking(&self) -> std::io::Result<ExitStatus> {
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

    /// Terminate the child process and every grandchild via the Job Object. Returns success
    /// even if the job was already empty; Win32 distinguishes these but the shell tool treats
    /// both as "gone".
    pub(crate) fn kill(&self) -> std::io::Result<()> {
        // SAFETY: `job` is a valid open Job HANDLE until Drop. Terminating the job cascades to
        // every process assigned to it, including any grandchildren the user command spawned.
        unsafe {
            if TerminateJobObject(self.job.as_raw(), 1) == 0 {
                let error = std::io::Error::last_os_error();
                // ERROR_ACCESS_DENIED (5) is returned when the job is already gone; treat as
                // success.
                if error.raw_os_error() == Some(5) {
                    return Ok(());
                }
                return Err(error);
            }
            Ok(())
        }
    }
}

/// Spawn `powershell.exe -NoProfile -NonInteractive -Command <command>` under a Low-integrity
/// token. Stdout and stderr are captured via anonymous pipes; stdin is not connected.
///
/// PowerShell parses its command line per `CommandLineToArgvW` rules, so the user command is
/// encoded with the standard argv-escape helper; embedded `"`, `\`, spaces, and shell
/// metacharacters all pass through unmangled. `-NoProfile` skips user profile scripts (fast
/// startup, no unrelated side effects); `-NonInteractive` makes the child fail fast on any
/// prompt instead of hanging on stdin.
///
/// Returns [`std::io::Error`] mirroring the underlying Win32 call so the shell tool can surface
/// a standard error message. Which token a sandboxed Windows child should run under.
///
/// The two are different mechanisms, not two settings of one. `LowIntegrity` drops the token's
/// integrity label so it cannot touch anything above Low; `WriteRestricted` leaves integrity
/// alone and instead intersects every *write* access against a restricting-SID list, granting
/// back exactly the workspace roots. Neither can express the other's boundary.
pub(crate) enum WindowsConfinement {
    /// Reads everywhere, writes nowhere outside the Low-integrity surface.
    LowIntegrity,
    /// Reads everywhere, writes only beneath these canonical roots.
    WriteRestricted(Vec<std::path::PathBuf>),
}

/// `WRITE_RESTRICTED`: the token's restricting SIDs are intersected for write accesses only.
const WRITE_RESTRICTED: u32 = 0x8;
/// `DISABLE_MAX_PRIVILEGE`: strip every privilege from the restricted token except
/// `SeChangeNotifyPrivilege`, which has to stay or traverse checks fail on every path.
const DISABLE_MAX_PRIVILEGE: u32 = 0x1;
// Not re-exported by the `Win32_Foundation` feature in windows-sys 0.61, and defined here for
// the same reason as the constants around it. `DELETE` is a standard access right, fixed at
// this value since NT and documented under ACCESS_MASK.
const DELETE: u32 = 0x0001_0000;
/// Inheritable by both files and subdirectories, so one ACE on the root covers the tree.
///
/// Belongs on this constant, not on `DELETE` above.
const SUB_CONTAINERS_AND_OBJECTS_INHERIT: u32 = 0x3;
const SE_GROUP_LOGON_ID: u32 = 0xC000_0000;
const ACCESS_ALLOWED_ACE_TYPE: u8 = 0x0;

/// The per-workspace write identity: a deterministic SID in the non-unique-authority space
/// (`S-1-4-x-y`), derived from the canonical root path.
///
/// `S-1-4` is the point of the design. It is a SID space that corresponds to no real principal,
/// so meka can mint one per workspace without creating an account and without any elevation.
/// Its power is defined entirely by the ACEs that name it, which exist only on that workspace's
/// own tree; the string itself is not a secret.
///
/// Deterministic so the same workspace derives the same identity across sessions, which is what
/// lets a grant be recognized and revoked rather than accumulating a fresh one per run.
pub(crate) fn workspace_write_sid(root: &std::path::Path) -> String {
    // FNV-1a over the path's own UTF-16 units, which is how Windows stores it, rather than over
    // a `to_string_lossy` rendering. Lossy conversion turns every unpaired surrogate into
    // U+FFFD, so two directories that Windows considers distinct could hash alike and share one
    // capability. Reading the units the OS actually holds removes the question.
    //
    // A cryptographic digest would buy nothing: the input is not secret, and a collision
    // between two of a user's own workspaces costs a shared grant rather than an escape.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut absorb = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    };
    for unit in root.as_os_str().encode_wide() {
        // Windows matches paths case-insensitively, so `C:\Work` and `c:\work` name one
        // directory and must reach one capability. Folded over ASCII only: that covers drive
        // letters and nearly every real path, and a non-ASCII case difference yields a second
        // SID, which is a second grant rather than a wrong one.
        let folded = match u8::try_from(unit) {
            Ok(ascii) => u16::from(ascii.to_ascii_lowercase()),
            Err(_) => unit,
        };
        absorb((folded & 0xff) as u8);
        absorb((folded >> 8) as u8);
    }
    let first = ((hash & 0x3fff_ffff) as u32).max(1);
    let second = (((hash >> 32) & 0x3fff_ffff) as u32).max(1);
    format!("S-1-4-{first}-{second}")
}

unsafe fn has_workspace_ace(acl: *const ACL, sid: *mut core::ffi::c_void) -> bool {
    if acl.is_null() {
        return false;
    }
    let count = unsafe { (*acl).AceCount };
    for index in 0..u32::from(count) {
        let mut ace: *mut core::ffi::c_void = ptr::null_mut();
        if unsafe { GetAce(acl, index, &mut ace) } == 0 {
            continue;
        }
        let header = ace as *const ACE_HEADER;
        if unsafe { (*header).AceType } != ACCESS_ALLOWED_ACE_TYPE {
            continue;
        }
        // An ACE that does not carry both inherit flags covers the root only, so the tree below
        // it is unwritable and the grant has to be re-placed. Measured on a real ACL, the flags
        // meka's own ACE comes back with are `0xb`: both inherit bits plus `INHERIT_ONLY_ACE`,
        // which `SetEntriesInAclW` adds because a mask holding generic bits means nothing when
        // applied to the object itself. So this tests for the two bits it needs rather than for
        // equality, which would never match.
        let flags = u32::from(unsafe { (*header).AceFlags });
        if flags & SUB_CONTAINERS_AND_OBJECTS_INHERIT != SUB_CONTAINERS_AND_OBJECTS_INHERIT {
            continue;
        }
        let allowed = ace as *const ACCESS_ALLOWED_ACE;
        // The generic bits survive the round trip: measured on a real ACL, the mask comes back
        // as `0x40010000`, exactly the `GENERIC_WRITE | DELETE` that was written. Nothing maps
        // it to `FILE_GENERIC_WRITE` on the way in, so reading it back is a plain comparison.
        let mask = unsafe { (*allowed).Mask };
        if mask & (GENERIC_WRITE | DELETE) != GENERIC_WRITE | DELETE {
            continue;
        }
        let ace_sid = unsafe { &raw const (*allowed).SidStart } as *const core::ffi::c_void;
        if unsafe { EqualSid(ace_sid as *mut core::ffi::c_void, sid) } != 0 {
            return true;
        }
    }
    false
}

/// Add or remove the workspace capability's inheritable write ACE on `root`.
///
/// Granting requires only that the caller *own* the directory, which supplies `WRITE_DAC`
/// implicitly. No elevation, and no account creation.
///
/// The ACE is real, standing state on the user's filesystem, visible to `icacls`. meka takes it
/// back when the process exits; see [`WindowsGrants`] for what that does and does not cover.
/// Whether `acl` already carries meka's inheritable write ACE for `sid`.
///
/// Conservative in the safe direction: anything it cannot positively identify reads as absent,
/// which re-places an ACE that was already correct. That costs time, never reach.
///
/// Identifying by SID alone is sound here because `workspace_write_sid` mints an `S-1-4`
/// identity that corresponds to no real principal and is derived from the root path, so nothing
/// but meka ever names it in an ACE.
unsafe fn set_workspace_ace(root: &std::path::Path, grant: bool) -> std::io::Result<()> {
    let sid_text: Vec<u16> = workspace_write_sid(root)
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let mut sid: *mut core::ffi::c_void = ptr::null_mut();
    if unsafe { ConvertStringSidToSidW(sid_text.as_ptr(), &mut sid) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let _sid_guard = LocalFreeGuard(sid);

    let target: Vec<u16> = root
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let mut existing: *mut ACL = ptr::null_mut();
    let mut descriptor: *mut core::ffi::c_void = ptr::null_mut();
    let status = unsafe {
        GetNamedSecurityInfoW(
            target.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut existing,
            ptr::null_mut(),
            &mut descriptor,
        )
    };
    if status != 0 {
        return Err(std::io::Error::from_raw_os_error(status as i32));
    }
    let _descriptor_guard = LocalFreeGuard(descriptor);

    // Placing the ACE re-propagates it over the whole tree, and `ensure` runs before *every*
    // `execute_command`. Measured on the test box against a 5000-file workspace: 282ms per
    // command, and every file's USN bumped each time.
    //
    // So skip the write when the ACE is already there. The read above has happened either way,
    // and this only reads its result, so the check costs nothing and does not walk the tree.
    // Same workspace with the skip in place: 282ms for the first command, 21us for each one
    // after it.
    //
    // This is a short-circuit, not a cache: it asks the filesystem, not a ledger. If another
    // meka's `revoke_all` took the ACE off the root while this one was running, it reads as
    // absent here and gets re-placed.
    //
    // What it does *not* preserve is repair below the root. Re-propagating on every command
    // also fixed any object beneath the root that had lost the inherited ACE -- a directory
    // with inheritance disabled, a tree restored with `robocopy /COPYALL`, a folder moved in
    // from another volume. Those now stay unwritable, with a bare access-denied and nothing
    // explaining it, until something takes the root's ACE off and back on. Judged worth it
    // against 282ms on every single command, but it is a real narrowing rather than a pure
    // optimization.
    if grant && unsafe { has_workspace_ace(existing, sid) } {
        return Ok(());
    }

    // `GENERIC_WRITE | DELETE`, not `GENERIC_ALL`. The capability is granted so a confined
    // child can *write* inside the workspace, and nothing in that job needs the rest of full
    // control. Granting it anyway meant `icacls` showed `(F)` while the code and the docs both
    // called this a write ACE, so a reader auditing the ACL and a reader auditing the source
    // came away with different answers.
    //
    // `GENERIC_WRITE` covers what a workspace write actually needs: on a file, write and append
    // data plus attributes; on a directory, the same two bits mean add-file and add-subdirectory.
    // `DELETE` is separate and is required for replacing a file, which is how every atomic write
    // lands -- `write_file` renames a temp file over the target, and the rename needs delete rights
    // on what it displaces.
    let access = EXPLICIT_ACCESS_W {
        grfAccessPermissions: GENERIC_WRITE | DELETE,
        grfAccessMode: if grant { GRANT_ACCESS } else { REVOKE_ACCESS },
        grfInheritance: SUB_CONTAINERS_AND_OBJECTS_INHERIT,
        Trustee: TRUSTEE_W {
            pMultipleTrustee: ptr::null_mut(),
            MultipleTrusteeOperation: NO_MULTIPLE_TRUSTEE,
            TrusteeForm: TRUSTEE_IS_SID,
            TrusteeType: TRUSTEE_IS_UNKNOWN,
            ptstrName: sid as *mut u16,
        },
    };

    let mut updated: *mut ACL = ptr::null_mut();
    let status = unsafe { SetEntriesInAclW(1, &access, existing, &mut updated) };
    if status != 0 {
        return Err(std::io::Error::from_raw_os_error(status as i32));
    }
    let _updated_guard = LocalFreeGuard(updated as *mut core::ffi::c_void);

    let status = unsafe {
        SetNamedSecurityInfoW(
            target.as_ptr() as *mut u16,
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            updated,
            ptr::null_mut(),
        )
    };
    if status != 0 {
        return Err(std::io::Error::from_raw_os_error(status as i32));
    }
    Ok(())
}

/// The workspace ACEs this process has placed, so they can be taken back.
///
/// The grant is standing state on the user's own directories, and meka is the only thing that knows
/// it put it there. Revoking is what keeps `workspace` from leaving a permission trail behind on
/// every folder the agent was ever run in. The cost is that the next run re-propagates the ACE,
/// which is one pass over the tree.
///
/// Reach the process's ledger through [`process_grants`] rather than constructing one, for the
/// reason given there. [`Drop`] revokes as a backstop for the instances tests build; the shared
/// one is a `static` and never drops, so [`crate::sandbox::release_process_grants`] is what
/// actually closes an ordinary run.
///
/// A `SIGKILL` or a hard crash still strands the ACE. That is untidy rather than unsafe: the
/// capability SID names no real principal, so the residue grants nothing to anyone, and the
/// next run in the same directory recognizes and reuses it instead of adding a second.
#[derive(Default)]
pub(crate) struct WindowsGrants {
    granted: std::sync::Mutex<std::collections::BTreeSet<std::path::PathBuf>>,
}

/// The one ledger for this process.
///
/// An ACE is filesystem state, not session state. Two sessions confining the same root place a
/// single ACE between them, so a per-session ledger has whichever session ends first revoke the
/// grant out from under the other. A sub-agent made that concrete rather than theoretical: its
/// `ToolRegistry` is dropped the moment its task finishes, which under a per-registry ledger
/// took the parent's still-live grants with it.
pub(crate) fn process_grants() -> &'static std::sync::Arc<WindowsGrants> {
    static GRANTS: std::sync::OnceLock<std::sync::Arc<WindowsGrants>> = std::sync::OnceLock::new();
    GRANTS.get_or_init(|| std::sync::Arc::new(WindowsGrants::default()))
}

impl WindowsGrants {
    /// Ensure `root` carries the capability's write ACE.
    ///
    /// Placed every time rather than skipped when the ledger already lists it. The ACE is machine
    /// state, not process state, and `process_grants` reasoned only about the two registries inside
    /// one process. Two mekas at `workspace` in the same directory both grant -- `SetEntriesInAcl`
    /// merges, so there is one ACE -- and whichever exits first revokes it. The survivor's ledger
    /// still says "granted", so a cached `ensure` short-circuits and never re-places it, and every
    /// later shell write in that session fails with a bare access-denied that nothing explains.
    /// Re-placing is idempotent and costs one pass over the tree, which is why the caller runs this
    /// off the async executor.
    pub(crate) fn ensure(&self, root: &std::path::Path) -> std::io::Result<()> {
        unsafe { set_workspace_ace(root, true)? };
        let mut granted = crate::sync::lock(&self.granted);
        granted.insert(root.to_path_buf());
        Ok(())
    }

    /// The roots this ledger currently believes carry an ACE.
    ///
    /// Test-only. Production never reads the set, it only adds to it and drains it at exit;
    /// this exists so a test can ask which ledger a tool built by the production path is
    /// actually writing into.
    #[cfg(test)]
    pub(crate) fn granted_roots(&self) -> std::collections::BTreeSet<std::path::PathBuf> {
        crate::sync::lock(&self.granted).clone()
    }

    /// Take every ACE back. Best-effort per root: one failure must not strand the others.
    ///
    /// A root whose revoke failed stays in the ledger, so a later attempt (or the next process
    /// running against the same root) still knows the ACE is out there. Clearing the whole set
    /// unconditionally meant the one case worth remembering -- the ACE meka placed and could
    /// not take back -- was the case it forgot.
    pub(crate) fn revoke_all(&self) {
        let mut granted = crate::sync::lock(&self.granted);
        let mut stranded = std::collections::BTreeSet::new();
        for root in granted.iter() {
            if let Err(error) = unsafe { set_workspace_ace(root, false) } {
                stranded.insert(root.clone());
                let path = root.display();
                let sid = workspace_write_sid(root);
                tracing::warn!(
                    "failed to revoke the workspace write ACE on {path}: {error}. Remove it with \
                     `icacls \"{path}\" /remove:g *{sid}` if it is unwanted"
                );
            }
        }
        *granted = stranded;
    }
}

impl Drop for WindowsGrants {
    fn drop(&mut self) {
        self.revoke_all();
    }
}

/// The token's logon-session SID (`S-1-5-5-x-y`), or `None` if it carries none.
///
/// Needed in the restricting list so the child keeps reach to the per-logon objects a shell
/// expects: the window station, the desktop, and the named pipes under them. Leaving it out
/// does not tighten the filesystem boundary, it just breaks the process.
///
/// Returns an owned copy of the SID's bytes. The caller must keep it alive across the
/// `CreateRestrictedToken` call, which reads through the pointer into it; see the note in the
/// body for why this copies rather than leaking the whole `TOKEN_GROUPS` buffer.
unsafe fn logon_session_sid(token: HANDLE) -> Option<Vec<u8>> {
    let mut needed = 0u32;
    unsafe { GetTokenInformation(token, TokenGroups, ptr::null_mut(), 0, &mut needed) };
    if needed == 0 {
        return None;
    }
    // `u64` elements for alignment; the length is rounded up to whole units.
    let mut buffer: Vec<u64> = vec![0u64; needed.div_ceil(8) as usize];
    if unsafe {
        GetTokenInformation(
            token,
            TokenGroups,
            buffer.as_mut_ptr() as *mut core::ffi::c_void,
            needed,
            &mut needed,
        )
    } == 0
    {
        return None;
    }
    // Aligned storage, because `TOKEN_GROUPS` needs 8-byte alignment and `Vec<u8>` guarantees
    // only 1. Windows' allocator happens to satisfy it, so the `u8` version worked in practice
    // and was UB by the letter.
    //
    // Read through raw pointers rather than through a `&TOKEN_GROUPS`. Forming the reference
    // asserted that a whole `TOKEN_GROUPS` (24 bytes) is dereferenceable, which a token
    // reporting zero groups would not satisfy, and `Groups` is a flexible array member whose
    // declared length is 1 -- so building a longer slice from a reference to it is the C idiom
    // that Rust's aliasing model rejects. Neither is reachable with a real token, and the
    // comment above claimed the letter-of-the-law problem had been dealt with, so this makes
    // that true instead of nearly true.
    let groups = buffer.as_ptr() as *const TOKEN_GROUPS;
    let group_count = unsafe { (*groups).GroupCount } as usize;
    let entries = unsafe {
        std::slice::from_raw_parts(
            (&raw const (*groups).Groups) as *const SID_AND_ATTRIBUTES,
            group_count,
        )
    };
    let found = entries
        .iter()
        .find(|entry| entry.Attributes & SE_GROUP_LOGON_ID != 0)?;

    // Copied out rather than leaked. The SID has to outlive `CreateRestrictedToken`, and the
    // previous version bought that with `Box::leak` on the whole `TOKEN_GROUPS` buffer -- one
    // permanent leak per spawn, 1-4 KB on a domain-joined account. Invisible for a one-shot CLI
    // run and unbounded for `meka serve`. A SID is self-describing and fixed-size, so the
    // caller can own just those bytes.
    let length = unsafe { GetLengthSid(found.Sid) } as usize;
    if length == 0 {
        return None;
    }
    let mut sid = vec![0u8; length];
    unsafe {
        std::ptr::copy_nonoverlapping(found.Sid as *const u8, sid.as_mut_ptr(), length);
    }
    Some(sid)
}

/// Make sure this process has a console, allocating a hidden one if it does not.
///
/// A `WRITE_RESTRICTED` child cannot *create* a console, only inherit one. That is the measured
/// rule, not the documented one: `CREATE_NO_WINDOW` and `CREATE_NEW_CONSOLE` both ask for a new
/// console and both die with `STATUS_DLL_INIT_FAILED` (0xC0000142) before `main` runs, and so
/// does a child of a parent that has no console to pass on. Allocating one here and hiding its
/// window makes the headless cases (ACP under an editor, `meka serve` as a service) behave like
/// the terminal case.
///
/// Runs at most once, and only when a restricted spawn is actually about to happen, so a meka
/// that never reaches `workspace` never allocates anything.
fn ensure_console() {
    // Not a `Once`: a failed allocation must be retried.
    //
    // `Once` is consumed whether the closure succeeded or not, so a single transient
    // `AllocConsole` failure disabled `workspace`'s shell for the life of the process -- every
    // later command dying with `STATUS_DLL_INIT_FAILED` before `main`, with nothing to retry
    // it. The `GetConsoleWindow` check is itself the idempotence guard: once a console exists
    // this returns immediately, so the only repeated work is on the path that has not
    // succeeded yet.
    if unsafe { !GetConsoleWindow().is_null() } {
        return;
    }
    unsafe { allocate_hidden_console() };
}

/// The body of [`ensure_console`], minus the `Once` and the already-have-one check.
///
/// Split out so a test can reach it: with a console present -- which is every `cargo test` run
/// -- `ensure_console` returns at its first line, so a test calling it exercises nothing. The
/// one interaction that needed proving was the only one it could not reach.
unsafe fn allocate_hidden_console() {
    // Snapshot the three standard handles across `AllocConsole`.
    //
    // Windows documents it as *initializing* the process's standard handles to the new
    // console's buffers. That is exactly wrong for the cases this function exists for: under
    // ACP meka speaks JSON-RPC over stdin/stdout, and `meka serve` may be piped. Rebinding them
    // mid-session on the first `workspace` shell command would break the transport silently,
    // with no error anywhere. Restoring unconditionally costs nothing when Windows leaves them
    // alone.
    let saved: [(u32, HANDLE); 3] = [
        (STD_INPUT_HANDLE, unsafe { GetStdHandle(STD_INPUT_HANDLE) }),
        (STD_OUTPUT_HANDLE, unsafe {
            GetStdHandle(STD_OUTPUT_HANDLE)
        }),
        (STD_ERROR_HANDLE, unsafe { GetStdHandle(STD_ERROR_HANDLE) }),
    ];

    if unsafe { AllocConsole() } == 0 {
        // `warn!`, because the consequence is total rather than cosmetic: a restricted child
        // cannot *create* a console, only inherit one, so without this every subsequent
        // `workspace` shell command dies with `STATUS_DLL_INIT_FAILED` before reaching `main`.
        // At `debug!` the user saw nothing at all at default verbosity and a shell that simply
        // did not work.
        let error = std::io::Error::last_os_error();
        tracing::warn!(
            "failed to allocate a console ({error}); shell commands at `workspace` will fail to \
             start. Running meka from a terminal avoids this"
        );
        return;
    }

    for (which, handle) in saved {
        if !handle.is_null() && handle != INVALID_HANDLE_VALUE {
            unsafe { SetStdHandle(which, handle) };
        }
    }

    let window = unsafe { GetConsoleWindow() };
    if !window.is_null() {
        unsafe { ShowWindow(window, SW_HIDE) };
    }
}

/// Spawn one sandboxed child under `confinement`.
///
/// Both variants share every step after the token: the pipe setup with its handle-inheritance
/// narrowing, the job object, the suspended spawn. Only the token differs, and only the
/// restricted path needs a console.
pub(crate) fn spawn_sandboxed_command(
    command: &str,
    confinement: &WindowsConfinement,
    cwd: &std::path::Path,
) -> std::io::Result<SandboxedChild> {
    // Embedded NULs would silently truncate the CreateProcess command line (Win32 treats the
    // UTF-16 command-line buffer as a C string). Agent-driven commands shouldn't contain these,
    // but fail loudly rather than silently execute a truncated prefix.
    if command.contains('\0') {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "command contains embedded NUL byte",
        ));
    }

    // Force UTF-8 output before running the user's command. PowerShell 5.1 (the inbox version we
    // invoke as `powershell.exe`) defaults `[Console]::OutputEncoding` to the system's legacy OEM
    // code page, CP 437 / 1252 on most English installs, which mangles non-ASCII output (日本語 →
    // `???`) when the process writes to a redirected pipe like ours. Prefixing every script with a
    // UTF-8 encoding switch makes output round-trip losslessly regardless of the host's console
    // configuration. Before anything is spawned, and only for the path that needs it.
    if matches!(confinement, WindowsConfinement::WriteRestricted(_)) {
        ensure_console();
    }

    let wrapped_command = super::wrap_command_with_utf8_output(command);

    // The session working directory, NUL-terminated for `lpCurrentDirectory`.
    //
    // Both `CreateProcess*` calls passed null here, which means "inherit the *process* cwd" --
    // and meka deliberately never mutates that (`main.rs` says so outright), so the sandboxed
    // child ran somewhere else entirely from every other path. Under ACP or `meka serve` the
    // session cwd is client-supplied and the process cwd is wherever the editor or unit
    // started, so at `workspace` the ACE was granted on one directory and the command ran in
    // another: relative writes hit a directory with no capability and were denied, and relative
    // reads silently read the wrong tree. Invisible in a plain terminal REPL, where the two are
    // the same path.
    let cwd_utf16: Vec<u16> = cwd
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let mut cmd_line = String::from(r#""powershell.exe" -NoProfile -NonInteractive -Command "#);
    cmd_line.push_str(&super::quote_command_arg(&wrapped_command));

    // SAFETY: All Win32 calls below are documented and we check return values. Handles are
    // wrapped in `OwnedHandle` to close on drop. Pipe handles transfer ownership into the
    // spawned child (for the write ends) or into the returned `File` (for the read ends).
    unsafe {
        // 1. Open our own process token and duplicate it as a primary token we can modify. The
        //    duplicate is what we'll confine; we must NOT mutate our own token.
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
        // `SecurityAnonymous` is the least-capable impersonation level and is the correct "don't
        // care" value when the target is a primary token; per Win32 docs the parameter is only
        // consulted for impersonation tokens, but some kernel versions have historically honored
        // it, so pick the safest constant.
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

        // 2. Strip all privileges from the duplicate before anything else touches it.
        //    Integrity-level enforcement already makes most privileges inert against Medium+
        //    resources, but defense-in-depth: a Low-integrity token that still claims (say)
        //    `SeShutdownPrivilege` is a sharper edge than one that has none at all. Passing
        //    DisableAllPrivileges=TRUE with a NULL NewState disables every privilege on the token.
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

        // 3. Build the Low-integrity SID via ConvertStringSidToSidW and point a
        //    TOKEN_MANDATORY_LABEL at it. The SID buffer is allocated by the OS and must be
        //    released via LocalFree.
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

        // Applied only for the Low-integrity confinement. The restricted path deliberately
        // leaves the integrity label alone: dropping to Low there would confine the child to
        // the Low surface *as well*, which would take away the workspace the ACEs just granted.
        if matches!(confinement, WindowsConfinement::LowIntegrity)
            && SetTokenInformation(
                low_token.as_raw(),
                TokenIntegrityLevel,
                &label as *const _ as *const core::ffi::c_void,
                mem::size_of::<TOKEN_MANDATORY_LABEL>() as u32,
            ) == 0
        {
            return Err(std::io::Error::last_os_error());
        }

        // 3b. For a workspace confinement, replace the token with a WRITE_RESTRICTED one whose
        //     restricting SIDs are the logon session, Everyone, and one capability per root.
        //
        //     `WRITE_RESTRICTED` intersects the restricting list against the DACL for *write*
        //     accesses only, so reads and execution are untouched and a write succeeds exactly
        //     where one of those SIDs has a write ACE. The capability SIDs have one only on
        //     their own workspace root, which is what makes the boundary.
        let low_token = match confinement {
            WindowsConfinement::LowIntegrity => low_token,
            WindowsConfinement::WriteRestricted(roots) => {
                let mut restricting: Vec<SID_AND_ATTRIBUTES> = Vec::new();
                let mut sid_guards: Vec<LocalFreeGuard> = Vec::new();

                // Held in this scope so the pointer handed to `CreateRestrictedToken` below
                // stays valid; it is dropped with the rest of the arm, after the call.
                let logon_sid = logon_session_sid(low_token.as_raw());
                if let Some(bytes) = logon_sid.as_ref() {
                    restricting.push(SID_AND_ATTRIBUTES {
                        Sid: bytes.as_ptr() as *mut core::ffi::c_void,
                        Attributes: 0,
                    });
                }
                // Everyone, plus one capability per root.
                //
                // Everyone is load-bearing, and not for the reason a filesystem test suggests.
                // Dropping it still let a child create a file under a granted root, overwrite one
                // already there, and write to a capture pipe: by that measure it looks free, and
                // removing it looks like a pure tightening. What it actually costs is the shell
                // itself. meka spawns `powershell.exe`, and with the restricting list cut to the
                // logon SID and the capability, PowerShell dies before evaluating anything with
                // `Starting the CLR failed with HRESULT 80070005` (E_ACCESSDENIED). Not one command
                // in five ran. Measured both ways on Windows 11 10.0.26200; the .NET runtime
                // reaches something on startup that only Everyone admits.
                //
                // The price is the one hole the docs name: a file carrying an explicit
                // `Everyone: Write` ACE stays writable from outside the workspace. That is the
                // only case the two configurations differed on, apart from the shell not
                // starting. Paying it buys a workspace level that can run commands at all.
                //
                // Probe the shell, not just the filesystem, before touching this list. A
                // `cmd.exe` probe passes every case here and tells you nothing about the
                // process meka actually spawns.
                let mut sid_texts: Vec<String> = vec!["S-1-1-0".to_string()];
                sid_texts.extend(roots.iter().map(|root| workspace_write_sid(root)));
                for text in &sid_texts {
                    let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
                    let mut sid: *mut core::ffi::c_void = ptr::null_mut();
                    if ConvertStringSidToSidW(wide.as_ptr(), &mut sid) == 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    sid_guards.push(LocalFreeGuard(sid));
                    restricting.push(SID_AND_ATTRIBUTES {
                        Sid: sid,
                        Attributes: 0,
                    });
                }

                let mut restricted: HANDLE = ptr::null_mut();
                // Derived from our *own* token, not from the privilege-stripped duplicate.
                //
                // `CreateProcessAsUserW` normally needs `SE_ASSIGNPRIMARYTOKEN` and
                // `SE_INCREASE_QUOTA`, and waives them only when the token is recognizably a
                // restricted version of the caller's. Deriving from the duplicate that step 2 had
                // already emptied of privileges broke that lineage: the kernel refused with
                // "SE_INCREASE_QUOTA_NAME not held", fell through to `CreateProcessWithTokenW`, and
                // that rejected the restricted token outright with ERROR_INVALID_PARAMETER. Both
                // commands then failed to spawn at all. `DISABLE_MAX_PRIVILEGE` alongside
                // `WRITE_RESTRICTED`.
                //
                // The restricted token is derived from `self_token`, which never went through
                // the privilege strip that step 2 applies to the Low-integrity duplicate -- so
                // without this the `read` child ran with zero privileges while the `workspace`
                // child kept meka's entire set, and the stronger-sounding level was the less
                // hardened one. Restricting SIDs bound *write access checks*; they do nothing
                // about privileges, which are checked separately and several of which exist
                // precisely to bypass a DACL. Launched from an elevated shell the child kept
                // `SeImpersonatePrivilege`, enabled by default, which is the whole "Potato"
                // family and takes the boundary with it; unelevated it still kept
                // `SeShutdownPrivilege`, so `shutdown /r` worked from inside a sandbox that
                // promises the command cannot change the machine.
                //
                // This does not touch the "restricted version of the caller's token" lineage
                // that `CreateProcessAsUserW`'s privilege waiver depends on -- the derivation
                // is still from `self_token` -- so the fallback bug that cost a live debugging
                // round should not return. It is verified on the box regardless.
                if CreateRestrictedToken(
                    self_token.as_raw(),
                    WRITE_RESTRICTED | DISABLE_MAX_PRIVILEGE,
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                    restricting.len() as u32,
                    restricting.as_ptr(),
                    &mut restricted,
                ) == 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                OwnedHandle(restricted)
            }
        };

        // 4. Create two anonymous pipes with **non-inheritable** handles. We use
        //    `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` (step 6) to narrow inheritance to exactly the
        //    three handles our child needs; the inherit flag is only flipped to TRUE briefly on
        //    those three handles, not the read ends, which eliminates the classic
        //    CreatePipe→SetHandleInformation→CreateProcess race where a concurrent CreateProcess in
        //    the same process could leak the read ends to an unrelated child.
        let sa_noninherit = SECURITY_ATTRIBUTES {
            nLength: mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: ptr::null_mut(),
            bInheritHandle: 0,
        };

        let (stdout_read, stdout_write) = create_pipe(&sa_noninherit)?;
        let (stderr_read, stderr_write) = create_pipe(&sa_noninherit)?;

        // 5. Open NUL as the child's stdin. Non-inheritable; inherit flag flipped on just before
        //    CreateProcess.
        let nul_stdin = open_nul_read(&sa_noninherit)?;

        // 6. Promote the three child-bound handles to inheritable. The
        //    PROC_THREAD_ATTRIBUTE_HANDLE_LIST filter (step 7) requires each listed handle to have
        //    HANDLE_FLAG_INHERIT set.
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
            || SetHandleInformation(nul_stdin.as_raw(), HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)
                == 0
        {
            return Err(std::io::Error::last_os_error());
        }

        // 7. Build a STARTUPINFOEXW with PROC_THREAD_ATTRIBUTE_HANDLE_LIST naming exactly the three
        //    handles we want the child to see. With bInheritHandles=TRUE and
        //    EXTENDED_STARTUPINFO_PRESENT, the child inherits *only* the listed handles even if
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

        // 8. Create a Job Object with `KILL_ON_JOB_CLOSE` BEFORE spawning, so the child can be
        //    assigned to it while still suspended. When `SandboxedChild` drops, the job handle
        //    drops too; that automatic close cascades to every assigned process, eliminating
        //    grandchild leaks on normal exit, kill, or panic.
        let job = create_kill_on_close_job()?;

        // 9. Spawn SUSPENDED. We assign the child to the job before any of its code runs; otherwise
        //    the child could spawn a grandchild outside the job in the gap between create and
        //    assign. With `CREATE_SUSPENDED` set, the main thread is created suspended and we
        //    manually resume it after assignment.
        let spawn_result = create_process_confined(
            low_token.as_raw(),
            &cwd_utf16,
            &cmd_line,
            &startup,
            &mut proc_info,
            // The restricted token cannot make a console, so it must inherit the one
            // `ensure_console` guaranteed rather than ask for a fresh one.
            !matches!(confinement, WindowsConfinement::WriteRestricted(_)),
        );

        // Parent no longer needs the child-side write ends or the NUL stdin handle regardless
        // of success/failure. Dropping the OwnedHandle wrappers closes them. On success,
        // closing the write ends ensures the parent's read end sees EOF when the child exits.
        // Drop before any early-return so the handles aren't leaked if later steps fail.
        drop(stdout_write);
        drop(stderr_write);
        drop(nul_stdin);
        drop(attr_list);

        spawn_result?;

        // 10. Assign the suspended child to the job, then resume.
        if AssignProcessToJobObject(job.as_raw(), proc_info.hProcess) == 0 {
            let error = std::io::Error::last_os_error();
            // Best-effort kill of the suspended child before bailing, so the orphan doesn't sit
            // around if AssignProcess failed.
            TerminateProcess(proc_info.hProcess, 1);
            CloseHandle(proc_info.hProcess);
            if !proc_info.hThread.is_null() {
                CloseHandle(proc_info.hThread);
            }
            return Err(error);
        }

        // Resume the main thread. ResumeThread returns the previous suspend count, or u32::MAX
        // on failure.
        if ResumeThread(proc_info.hThread) == u32::MAX {
            let error = std::io::Error::last_os_error();
            // Kill via job since assignment already succeeded.
            TerminateJobObject(job.as_raw(), 1);
            CloseHandle(proc_info.hProcess);
            if !proc_info.hThread.is_null() {
                CloseHandle(proc_info.hThread);
            }
            return Err(error);
        }

        // We don't need the main thread handle; close it immediately.
        if !proc_info.hThread.is_null() {
            CloseHandle(proc_info.hThread);
        }

        // Transfer pipe read ends into owned `File`s. `File::from_raw_handle` takes ownership
        // of the HANDLE; `OwnedHandle::into_raw` suppresses the wrapper's Drop.
        let stdout_handle = stdout_read.into_raw();
        let stderr_handle = stderr_read.into_raw();

        Ok(SandboxedChild {
            process: OwnedHandle(proc_info.hProcess),
            job,
            stdout: Some(File::from_raw_handle(stdout_handle as _)),
            stderr: Some(File::from_raw_handle(stderr_handle as _)),
        })
    }
}

/// Create an empty Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` set. Any process later
/// assigned to the job is killed when the job's last handle closes, the Windows analogue to
/// Unix process groups teardown via `kill(-pgid, SIGKILL)`. Grandchildren inherit job
/// membership automatically.
unsafe fn create_kill_on_close_job() -> std::io::Result<OwnedHandle> {
    // SAFETY: CreateJobObjectW with null name and null SECURITY_ATTRIBUTES returns an unnamed
    // Job HANDLE the current process owns.
    let job = unsafe { CreateJobObjectW(ptr::null(), ptr::null()) };
    if job.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    let job = OwnedHandle(job);

    let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { mem::zeroed() };
    info.BasicLimitInformation = JOBOBJECT_BASIC_LIMIT_INFORMATION {
        LimitFlags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        ..unsafe { mem::zeroed() }
    };

    if unsafe {
        SetInformationJobObject(
            job.as_raw(),
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const core::ffi::c_void,
            mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error());
    }

    Ok(job)
}

/// Create an anonymous pipe using the supplied SECURITY_ATTRIBUTES.
///
/// A 1 MiB buffer hint is passed to `CreatePipe`. This is belt-and- braces with the concurrent
/// draining in the Windows spawn path: even if the drain task is momentarily starved, the child has
/// a MiB of slack before it blocks in `WriteFile`.
unsafe fn create_pipe(sa: &SECURITY_ATTRIBUTES) -> std::io::Result<(OwnedHandle, OwnedHandle)> {
    const PIPE_BUFFER_SIZE: u32 = crate::text::MIB as u32;
    let mut read: HANDLE = ptr::null_mut();
    let mut write: HANDLE = ptr::null_mut();
    // SAFETY: CreatePipe writes two HANDLEs through the provided pointers on success.
    // SECURITY_ATTRIBUTES is a valid initialized struct.
    if unsafe { CreatePipe(&mut read, &mut write, sa, PIPE_BUFFER_SIZE) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok((OwnedHandle(read), OwnedHandle(write)))
}

/// Open the `NUL` device for read. Inherit flag is left unset by the caller's
/// `SECURITY_ATTRIBUTES`; promote via `SetHandleInformation` right before the handle is passed
/// to `CreateProcess`. The child sees immediate EOF on any read, the correct "no stdin"
/// primitive on Windows, equivalent to `/dev/null` on Unix.
unsafe fn open_nul_read(sa: &SECURITY_ATTRIBUTES) -> std::io::Result<OwnedHandle> {
    let path: Vec<u16> = "NUL\0".encode_utf16().collect();
    // SAFETY: `path` is NUL-terminated; `sa` is a valid initialized SECURITY_ATTRIBUTES owned
    // by the caller for the duration of the call.
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

/// Create a process under the Low-integrity token. Tries `CreateProcessAsUserW` first (the
/// usual path); on `ERROR_PRIVILEGE_NOT_HELD`, which happens when the current user lacks
/// `SE_INCREASE_QUOTA_NAME` (common on locked-down corp-managed accounts), falls back to
/// `CreateProcessWithTokenW`, which requires the more broadly-granted `SE_IMPERSONATE_NAME`
/// instead.
///
/// The command line is re-encoded to UTF-16 for *each* attempt: Win32 documents `lpCommandLine`
/// as in/out, and the first call may mutate the buffer (typically inserting a NUL to split
/// `argv[0]`) before failing, so re-using the same buffer between attempts could hand the
/// fallback a corrupted string.
///
/// Both invocations pass `EXTENDED_STARTUPINFO_PRESENT` together with `STARTUPINFOEXW`, so the
/// handle-list filter in the attribute list applies uniformly across both paths.
unsafe fn create_process_confined(
    token: HANDLE,
    cwd_utf16: &[u16],
    cmd_line_utf8: &str,
    startup: &STARTUPINFOEXW,
    proc_info: &mut PROCESS_INFORMATION,
    // `low_integrity`: true for the Low-integrity path, false for `WRITE_RESTRICTED`. It
    // governs two things that happen to share an answer, both measured rather than documented:
    // whether `CREATE_NO_WINDOW` may be asked for (a restricted child cannot *create* a
    // console, only inherit one), and whether the `CreateProcessWithTokenW` fallback below is
    // worth attempting at all.
    low_integrity: bool,
) -> std::io::Result<()> {
    // CREATE_SUSPENDED so the child sits at its entry point until we've assigned it to the Job
    // Object; otherwise the child could spawn a grandchild before assignment, and that grandchild
    // would never be bound to the job. `CREATE_NO_WINDOW` asks for a *new* console, which a
    // WRITE_RESTRICTED child cannot create: it dies with STATUS_DLL_INIT_FAILED before `main` runs.
    // Measured, not documented. The restricted path inherits the console `ensure_console`
    // guaranteed instead.
    let no_window = if low_integrity { CREATE_NO_WINDOW } else { 0 };
    let creation_flags =
        no_window | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED;
    let startup_ptr = startup as *const STARTUPINFOEXW as *const STARTUPINFOW;

    // Build a scrubbed UTF-16 environment block once. Passing this for both `CreateProcessAsUserW`
    // and the `CreateProcessWithTokenW` fallback ensures the sandboxed child never sees the agent's
    // `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or any `*_TOKEN` / `*_SECRET` variable; a
    // Low-integrity child can still open outbound sockets, so a leaked key in env is a live exfil
    // vector.
    let mut env_block = build_scrubbed_env_block_utf16();
    let env_ptr = env_block.as_mut_ptr() as *const core::ffi::c_void;

    let mut cmd_line_utf16: Vec<u16> = cmd_line_utf8
        .encode_utf16()
        .chain(std::iter::once(0u16))
        .collect();

    // SAFETY: All pointers are valid for the duration of the call per the caller's obligations.
    // Win32 writes to `proc_info` on success.
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
            cwd_utf16.as_ptr(),
            startup_ptr,
            proc_info,
        )
    };
    if ok != 0 {
        return Ok(());
    }

    let error = std::io::Error::last_os_error();
    if error.raw_os_error() != Some(ERROR_PRIVILEGE_NOT_HELD as i32) {
        return Err(error);
    }

    // The fallback is Low-integrity only, and saying so is the point.
    //
    // `CreateProcessWithTokenW` rejects a restricted token outright with `ERROR_INVALID_PARAMETER`
    // -- measured during the feasibility probe, and recorded at the token-construction site.
    // Running it anyway on the `workspace` path replaced the real diagnosis
    // (`ERROR_PRIVILEGE_NOT_HELD`: this account does not hold `SE_INCREASE_QUOTA_NAME`) with "The
    // parameter is incorrect. (os error 87)", and logged a warning describing a fallback that could
    // not have happened. A corp-managed account switching to `workspace` got error 87 on every
    // command and nothing pointing at why.
    if !low_integrity {
        return Err(std::io::Error::new(
            error.kind(),
            format!(
                "{error}. `workspace` spawns through CreateProcessAsUserW, which needs \
                 SE_INCREASE_QUOTA_NAME; this account does not hold it. `unrestricted` \
                 avoids that path, or run meka from an account that holds it."
            ),
        ));
    }

    tracing::warn!(
        "CreateProcessAsUserW denied (SE_INCREASE_QUOTA_NAME not held); falling back to \
         CreateProcessWithTokenW, which still spawns the child at Low integrity with the \
         same scrubbed environment"
    );

    // Rebuild the command-line buffer; the previous call may have mutated it before failing
    // (Win32 documents lpCommandLine as in/out).
    let mut cmd_line_utf16_retry: Vec<u16> = cmd_line_utf8
        .encode_utf16()
        .chain(std::iter::once(0u16))
        .collect();

    // SAFETY: same contract as CreateProcessAsUserW; the two APIs only differ in their
    // parameter list (no process/thread security attrs, no bInheritHandles; inheritance is
    // driven by the per-handle `HANDLE_FLAG_INHERIT` flag plus the attribute-list filter). We
    // re-use the scrubbed environment block so the fallback path doesn't accidentally regress
    // to inheriting the parent's env.
    let ok = unsafe {
        CreateProcessWithTokenW(
            token,
            0, // dwLogonFlags: 0 means "use the token as-is"
            ptr::null(),
            cmd_line_utf16_retry.as_mut_ptr(),
            creation_flags,
            env_ptr,
            cwd_utf16.as_ptr(),
            startup_ptr,
            proc_info,
        )
    };
    // Keep `env_block` alive until after both calls complete; Win32 copies the contents but
    // documents `lpEnvironment` as a pointer that must be valid through the call.
    drop(env_block);
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Build a UTF-16 `NAME=VALUE\0NAME=VALUE\0\0` environment block for the sandboxed child.
/// Delegates to [`super::sandbox_child_env`] for the filter (Windows uses the deny-list arm) so
/// the Low-integrity spawn path stays in sync with the Unix sandbox paths.
fn build_scrubbed_env_block_utf16() -> Vec<u16> {
    let mut block: Vec<u16> = Vec::new();
    for (name_os, value_os) in super::sandbox_child_env() {
        append_env_entry(&mut block, &name_os, &value_os);
    }
    // Double-NUL terminator (each entry already ends with one NUL; we need another to close the
    // block).
    block.push(0);
    block
}

/// Append one `NAME=VALUE\0` entry, in the environment block's own encoding.
///
/// Takes `OsStr` and walks it with `encode_wide` rather than going via `&str`. The block is
/// UTF-16 either way, so a `to_str()` round-trip would exist only to *discard* values the
/// destination represents natively: a `PATH` or `TMP` holding an unpaired surrogate was
/// dropped from the child's environment with no diagnostic, and a shell with no `PATH`
/// resolves no commands.
fn append_env_entry(block: &mut Vec<u16>, name: &std::ffi::OsStr, value: &std::ffi::OsStr) {
    block.extend(name.encode_wide());
    block.push(u16::from(b'='));
    block.extend(value.encode_wide());
    block.push(0);
}

/// RAII wrapper around `PROC_THREAD_ATTRIBUTE_LIST`. Owns both the attribute-list backing
/// buffer and the HANDLE array it points into; Win32 stores the handle-list address (not a
/// copy), so the array must outlive any `CreateProcess*` call that consumes the attribute list.
struct ProcThreadAttributeList {
    // Both fields are kept alive for Drop. The Vec's heap buffer is the attribute-list storage;
    // `list_ptr` caches a stable mutable pointer to it. The boxed handle slice is referenced (by
    // pointer) from inside the attribute-list buffer, so it must not move or drop while the list
    // is alive.
    _buffer: Vec<u8>,
    _handles: Box<[HANDLE]>,
    list_ptr: LPPROC_THREAD_ATTRIBUTE_LIST,
}

impl ProcThreadAttributeList {
    /// Build a one-attribute list containing a `HANDLE_LIST` attribute referencing the supplied
    /// handles. `UpdateProcThreadAttribute` stores the pointer to the handle array, not a copy;
    /// the array is boxed into the returned wrapper so it stays at a fixed address for the
    /// wrapper's lifetime.
    unsafe fn new_with_handle_list(handles: &[HANDLE]) -> std::io::Result<Self> {
        // First call: buffer=NULL, size=0 → fails with ERROR_INSUFFICIENT_BUFFER but writes the
        // required size.
        let mut size: usize = 0;
        unsafe {
            InitializeProcThreadAttributeList(ptr::null_mut(), 1, 0, &mut size);
        }
        if size == 0 {
            return Err(std::io::Error::last_os_error());
        }

        let mut buffer: Vec<u8> = vec![0; size];
        let list_ptr = buffer.as_mut_ptr() as LPPROC_THREAD_ATTRIBUTE_LIST;

        // SAFETY: `list_ptr` points to a correctly-sized buffer from the previous size query,
        // and `size` is that queried value.
        if unsafe { InitializeProcThreadAttributeList(list_ptr, 1, 0, &mut size) } == 0 {
            return Err(std::io::Error::last_os_error());
        }

        let boxed_handles: Box<[HANDLE]> = handles.to_vec().into_boxed_slice();
        let handles_bytes = std::mem::size_of_val(&*boxed_handles);

        // SAFETY: `list_ptr` was just initialized; `boxed_handles` lives for 'self because it's
        // stored in the returned wrapper; the byte size passed matches the boxed array.
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
            let error = std::io::Error::last_os_error();
            // SAFETY: Initialize succeeded; must be paired with Delete regardless of subsequent
            // failures.
            unsafe { DeleteProcThreadAttributeList(list_ptr) };
            return Err(error);
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
        // SAFETY: constructor either fully initialized the list (and stored its pointer in
        // `list_ptr`) or returned Err (in which case this Drop doesn't run).
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
            // SAFETY: pointer was returned by a Win32 API that documents `LocalFree` as the
            // correct release call.
            unsafe {
                LocalFree(self.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a directory's DACL and ask whether meka's write ACE is on it.
    ///
    /// Test-only, because production only ever asks this from inside `set_workspace_ace`, which
    /// already holds the descriptor it read.
    fn carries_workspace_ace(root: &std::path::Path) -> bool {
        let sid_text: Vec<u16> = workspace_write_sid(root)
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let mut sid: *mut core::ffi::c_void = ptr::null_mut();
        assert_ne!(
            unsafe { ConvertStringSidToSidW(sid_text.as_ptr(), &mut sid) },
            0,
            "the capability SID must parse"
        );
        let _sid_guard = LocalFreeGuard(sid);
        let target: Vec<u16> = root
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let mut acl: *mut ACL = ptr::null_mut();
        let mut descriptor: *mut core::ffi::c_void = ptr::null_mut();
        let status = unsafe {
            GetNamedSecurityInfoW(
                target.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut acl,
                ptr::null_mut(),
                &mut descriptor,
            )
        };
        assert_eq!(status, 0, "reading the DACL must succeed");
        let _descriptor_guard = LocalFreeGuard(descriptor);
        unsafe { has_workspace_ace(acl, sid) }
    }

    /// The short-circuit that keeps `ensure` from re-propagating the DACL on every command must
    /// answer both ways against a real ACL.
    ///
    /// Both failure directions are silent, which is why this is asserted rather than reasoned
    /// about. Always-false costs only time, so nothing would ever surface it: the measured
    /// 280ms-per-command would simply stay. Always-true is worse -- `ensure` would stop placing
    /// the ACE at all and every `workspace` write would be denied on a tree meka believed it
    /// had granted.
    ///
    /// The specific traps are the mask and the flags, and both were settled by dumping a real
    /// ACL rather than by reasoning: the mask comes back as the `GENERIC_WRITE | DELETE` that
    /// was written and is *not* mapped to `FILE_GENERIC_WRITE`, while the flags come back as
    /// `0xb`, carrying an `INHERIT_ONLY_ACE` bit nothing in meka asked for. A check written
    /// from the documentation alone would have tested for flag equality and never matched.
    #[test]
    fn the_workspace_ace_is_recognized_once_placed_and_not_before() {
        let workspace = tempfile::tempdir().expect("temp dir");
        let root = workspace.path();

        assert!(
            !carries_workspace_ace(root),
            "a fresh directory carries no workspace ACE"
        );

        unsafe { set_workspace_ace(root, true) }.expect("granting must succeed");
        assert!(
            carries_workspace_ace(root),
            "the ACE just placed must be recognized, or `ensure` re-propagates the whole tree \
             on every command"
        );

        unsafe { set_workspace_ace(root, false) }.expect("revoking must succeed");
        assert!(
            !carries_workspace_ace(root),
            "a revoked ACE must read as absent, or `ensure` would skip re-placing a grant that \
             another meka took back"
        );
    }

    /// The capability SID is stable for a path and distinct between paths.
    ///
    /// Stability is what lets a grant be recognized and revoked instead of accumulating a new
    /// ACE per run; distinctness is what stops two of the user's workspaces sharing one.
    #[test]
    fn the_workspace_sid_is_deterministic_and_per_path() {
        let a = std::path::Path::new(r"C:\Users\x\project");
        let b = std::path::Path::new(r"C:\Users\x\other");
        assert_eq!(workspace_write_sid(a), workspace_write_sid(a));
        assert_ne!(workspace_write_sid(a), workspace_write_sid(b));
        assert!(workspace_write_sid(a).starts_with("S-1-4-"));
        // Windows paths are case-insensitive, so two spellings of one directory must not mint
        // two identities and leave half the ACEs unrevoked.
        assert_eq!(
            workspace_write_sid(a),
            workspace_write_sid(std::path::Path::new(r"C:\USERS\X\PROJECT"))
        );
    }

    /// Every caller shares one ledger, so no teardown can revoke another's grant.
    ///
    /// The failure this guards is quiet: with a ledger per `ToolRegistry`, a sub-agent finishing
    /// its task dropped its registry and took the parent's ACEs with it, leaving a parent still at
    /// `workspace` unable to write to its own root.
    #[test]
    fn the_process_shares_one_grant_ledger() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = crate::workspace::canonical_for_test(temp.path());

        // A sub-agent's handle, taken and dropped the way its `ToolRegistry` is dropped the
        // moment its task finishes. Identity alone (`Arc::ptr_eq` on two calls) was the whole
        // test and could not fail: a `OnceLock` returns itself by construction, so it held
        // equally well for the per-registry ledger this exists to rule out.
        {
            let borrowed = std::sync::Arc::clone(process_grants());
            borrowed.ensure(&root).expect("grant");
        }
        assert!(
            icacls(&root).contains(&workspace_write_sid(&root)),
            "one holder going away must not take the grant with it"
        );

        // And the grant placed through that handle is the one the process-wide release takes
        // back, which is the other half of "one ledger".
        process_grants().revoke_all();
        assert!(
            !icacls(&root).contains(&workspace_write_sid(&root)),
            "the shared ledger must still own what was granted through a clone of it"
        );
    }

    /// A granted ACE is visible and a revoked one leaves nothing behind.
    #[test]
    fn a_workspace_grant_round_trips() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = crate::workspace::canonical_for_test(temp.path());
        let grants = WindowsGrants::default();

        grants.ensure(&root).expect("grant");
        let after_grant = icacls(&root);
        assert!(
            after_grant.contains(&workspace_write_sid(&root)),
            "the capability must appear in the DACL: {after_grant}"
        );

        grants.revoke_all();
        let after_revoke = icacls(&root);
        assert!(
            !after_revoke.contains(&workspace_write_sid(&root)),
            "revoking must leave no residue: {after_revoke}"
        );
    }

    fn icacls(path: &std::path::Path) -> String {
        let output = std::process::Command::new("icacls")
            .arg(path)
            .output()
            .expect("icacls");
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    /// Allocating a console must not disturb stdin/stdout.
    ///
    /// Under ACP meka speaks JSON-RPC over those handles, so a console allocation that
    /// redirected them would break the session silently, with no error anywhere.
    ///
    /// **What this covers and what it cannot.** It calls `allocate_hidden_console` rather than
    /// `ensure_console`, because a `cargo test` process always has a console and `ensure_console`
    /// returns early: the previous version of this test never reached the code it was named after.
    /// Calling the inner function does run the snapshot-and-restore, but `AllocConsole` itself
    /// fails with `ERROR_ACCESS_DENIED` in a process that already has one, so the branch where the
    /// allocation *succeeds* and rebinds the handles is still not reachable from a test binary.
    /// That branch is verified by running meka headless on Windows hardware, which is where the
    /// feasibility probe measured it in the first place.
    #[test]
    fn allocating_a_console_leaves_the_standard_handles_alone() {
        use windows_sys::Win32::System::Console::{
            GetStdHandle, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
        };
        let (before_in, before_out) = unsafe {
            (
                GetStdHandle(STD_INPUT_HANDLE),
                GetStdHandle(STD_OUTPUT_HANDLE),
            )
        };
        // Safe: the body is a snapshot-and-restore around one `AllocConsole`, and a test
        // process already holds a console, so that call is a no-op here.
        unsafe { allocate_hidden_console() };
        let (after_in, after_out) = unsafe {
            (
                GetStdHandle(STD_INPUT_HANDLE),
                GetStdHandle(STD_OUTPUT_HANDLE),
            )
        };
        assert_eq!(before_in, after_in, "stdin was rebound");
        assert_eq!(before_out, after_out, "stdout was rebound");
    }
}
