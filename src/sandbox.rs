/// Filesystem sandboxing for read-only command execution.
///
/// On Linux, uses Landlock LSM (kernel 5.13+) to restrict child processes
/// to read-only filesystem access. On macOS, uses sandbox-exec.

#[derive(Debug, Clone, Copy)]
pub enum SandboxCapability {
    #[cfg(target_os = "linux")]
    Landlock {
        abi_version: i32,
    },
    #[cfg(target_os = "macos")]
    SandboxExec,
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

    SandboxCapability::Unavailable
}

// ---------------------------------------------------------------------------
// Linux Landlock
// ---------------------------------------------------------------------------

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
        let root_fd = libc::open(
            b"/\0".as_ptr() as *const libc::c_char,
            libc::O_PATH | libc::O_CLOEXEC,
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_sandbox_capability() {
        let capability = detect();
        // Should detect something on Linux/macOS, Unavailable on others
        match capability {
            #[cfg(target_os = "linux")]
            SandboxCapability::Landlock { abi_version } => {
                assert!(abi_version >= 1);
            }
            #[cfg(target_os = "macos")]
            SandboxCapability::SandboxExec => {}
            SandboxCapability::Unavailable => {}
            #[allow(unreachable_patterns)]
            _ => {}
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
