//! Landlock: the ABI probe, the ruleset meka builds for a write scope, and the raw syscall bindings
//! it needs.

use super::*;

pub(super) fn probe_landlock() -> BackendProbe {
    landlock_probe_from_abi(landlock_abi())
}
/// The [`MIN_LANDLOCK_ABI`] policy, split from the syscall so it can be exercised at ABI values
/// this host does not have. Kernels below v3 are the ones that matter and are exactly the ones a
/// developer machine running a current kernel cannot reproduce.
pub(super) fn landlock_probe_from_abi(abi: Option<i32>) -> BackendProbe {
    match abi {
        Some(abi_version) if abi_version >= MIN_LANDLOCK_ABI => {
            BackendProbe::Ok(SandboxCapability::Landlock { abi_version })
        }
        Some(abi_version) => BackendProbe::Missing {
            reason: format!(
                "Landlock ABI v{abi_version} is too old to write-protect the filesystem: truncate(2) is \
                 unmediated below v{MIN_LANDLOCK_ABI} (needs Linux 6.2+), so a command at `read` could still empty \
                 an existing file",
            ),
        },
        None => BackendProbe::Missing {
            reason: "Landlock LSM not supported by this kernel (needs Linux 5.13+)".to_string(),
        },
    }
}
/// Lowest Landlock ABI meka will sandbox with.
///
/// v3 (Linux 6.2) is where `LANDLOCK_ACCESS_FS_TRUNCATE` arrives. Below it `truncate(2)` is
/// unmediated, so a "read-only" child can still open an existing file for truncation and empty it:
/// `os.truncate(path, 0)` succeeds at v1 even though `open(O_WRONLY)` is denied. That is a write,
/// and meka documents the `read` level as write-protecting the filesystem, so accepting v1/v2 would
/// be promising a boundary the kernel is not enforcing.
///
/// Failing closed costs shell at `read` on kernels 5.13-6.1 (Ubuntu 22.04, Debian 12) that lack
/// Bubblewrap. That is the intended trade: Bubblewrap is auto-preferred whenever `bwrap` is on
/// `PATH` and is unaffected, and a refusal the user can act on beats a guarantee that quietly does
/// not hold.
pub(super) const MIN_LANDLOCK_ABI: i32 = 3;
/// Raw kernel ABI probe. Reports what the kernel supports, not what meka will accept: the
/// [`MIN_LANDLOCK_ABI`] policy lives in [`probe_landlock`] so the "too old" case can be reported
/// differently from "no Landlock at all".
pub(super) fn landlock_abi() -> Option<i32> {
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
/// Apply Landlock restrictions to the current process: read and execute everywhere, plus full
/// access beneath each path in `writable`.
///
/// `writable` is empty for a read-only confinement and holds the canonical workspace roots
/// otherwise. The paths arrive as [`std::ffi::CString`] because this runs after `fork`: building
/// them here would allocate, which the safety contract below forbids, so the caller prepares them
/// in the parent.
///
/// Landlock rules are additive grants with no deny form, so a writable root cannot have a subtree
/// carved back out of it. Nothing in meka asks for that today; if something ever does, this backend
/// cannot express it and the caller must say so rather than silently granting the whole root.
///
/// # Safety
///
/// This function uses raw syscalls and must only be called in a `pre_exec` context (after fork,
/// before exec) where the process is single-threaded. All operations are async-signal-safe
/// (syscalls only, no heap allocation).
pub(crate) unsafe fn apply_landlock(
    abi_version: i32,
    writable: &[std::ffi::CString],
) -> Result<(), i32> {
    unsafe {
        // PR_SET_NO_NEW_PRIVS is required for unprivileged Landlock usage
        if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
            return Err(*libc::__errno_location());
        }

        let attr = LandlockRulesetAttr {
            handled_access_fs: handled_access_for_abi(abi_version),
            handled_access_net: 0,
            scoped: scoped_for_abi(abi_version),
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
            // `close(2)` is permitted to set `errno` even on success, so read the failure reason
            // before releasing the ruleset.
            let error = *libc::__errno_location();
            libc::close(ruleset_fd);
            return Err(error);
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
        if ret < 0 {
            let error = *libc::__errno_location();
            libc::close(root_fd);
            libc::close(ruleset_fd);
            return Err(error);
        }
        libc::close(root_fd);

        // `/dev/null` is writable in every confinement, including read-only.
        //
        // The other two Unix backends already do this and say why: the macOS profile calls
        // `/dev/null` writes "universally legitimate for shell redirects", and Bubblewrap's
        // `--dev /dev` supplies a writable one. Landlock granted neither, so `cmd 2>/dev/null`
        // failed with a bare "Permission denied" under this backend alone. That is a redirect
        // discards output; it is not a write to the machine, and refusing it confines nothing.
        let dev_null_fd = libc::open(c"/dev/null".as_ptr(), libc::O_PATH | libc::O_CLOEXEC);
        if dev_null_fd >= 0 {
            let path_beneath = LandlockPathBeneathAttr {
                allowed_access: LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_READ_FILE,
                parent_fd: dev_null_fd,
            };
            let ret = libc::syscall(
                libc::SYS_landlock_add_rule,
                ruleset_fd,
                LANDLOCK_RULE_PATH_BENEATH,
                &path_beneath as *const LandlockPathBeneathAttr,
                0u32,
            );
            libc::close(dev_null_fd);
            if ret < 0 {
                let error = *libc::__errno_location();
                libc::close(ruleset_fd);
                return Err(error);
            }
        }

        // Grant every right the ruleset handles beneath each workspace root. "Writable" here means
        // the full set rather than just `WRITE_FILE`: creating, removing, renaming and truncating
        // are all separate Landlock rights, and a shell that can write bytes but not create a file
        // would fail on the first `>` redirect.
        for path in writable {
            let root_fd = libc::open(path.as_ptr(), libc::O_PATH | libc::O_CLOEXEC);
            if root_fd < 0 {
                // A root that cannot be opened grants nothing. Skipping rather than failing keeps
                // a deleted directory from turning every command into a spawn error, and the
                // effect is restrictive: the confinement stays as tight as it was.
                continue;
            }
            let path_beneath = LandlockPathBeneathAttr {
                allowed_access: handled_access_for_abi(abi_version),
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
                let error = *libc::__errno_location();
                libc::close(ruleset_fd);
                return Err(error);
            }
        }

        let ret = libc::syscall(libc::SYS_landlock_restrict_self, ruleset_fd, 0u32);
        if ret < 0 {
            let error = *libc::__errno_location();
            libc::close(ruleset_fd);
            return Err(error);
        }
        libc::close(ruleset_fd);

        Ok(())
    }
}
pub(super) fn handled_access_for_abi(abi_version: i32) -> u64 {
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
    if abi_version >= 9 {
        access |= LANDLOCK_ACCESS_FS_RESOLVE_UNIX;
    }
    access
}
/// IPC scoping flags for the ruleset. ABI v6 (kernel 6.12) added scoping; restricting it blocks the
/// sandboxed child from reaching abstract Unix sockets (D-Bus and similar) and from signaling
/// processes outside its own Landlock domain. Setting an unknown `scoped` bit makes
/// `landlock_create_ruleset` fail with `EINVAL`, so this stays zero below v6.
pub(super) fn scoped_for_abi(abi_version: i32) -> u64 {
    if abi_version >= 6 {
        LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET | LANDLOCK_SCOPE_SIGNAL
    } else {
        0
    }
}
// Landlock constants
pub(super) const LANDLOCK_CREATE_RULESET_VERSION: u32 = 1 << 0;
pub(super) const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;
pub(super) const LANDLOCK_ACCESS_FS_EXECUTE: u64 = 1 << 0;
pub(super) const LANDLOCK_ACCESS_FS_WRITE_FILE: u64 = 1 << 1;
pub(super) const LANDLOCK_ACCESS_FS_READ_FILE: u64 = 1 << 2;
pub(super) const LANDLOCK_ACCESS_FS_READ_DIR: u64 = 1 << 3;
pub(super) const LANDLOCK_ACCESS_FS_REMOVE_DIR: u64 = 1 << 4;
pub(super) const LANDLOCK_ACCESS_FS_REMOVE_FILE: u64 = 1 << 5;
pub(super) const LANDLOCK_ACCESS_FS_MAKE_CHAR: u64 = 1 << 6;
pub(super) const LANDLOCK_ACCESS_FS_MAKE_DIR: u64 = 1 << 7;
pub(super) const LANDLOCK_ACCESS_FS_MAKE_REG: u64 = 1 << 8;
pub(super) const LANDLOCK_ACCESS_FS_MAKE_SOCK: u64 = 1 << 9;
pub(super) const LANDLOCK_ACCESS_FS_MAKE_FIFO: u64 = 1 << 10;
pub(super) const LANDLOCK_ACCESS_FS_MAKE_BLOCK: u64 = 1 << 11;
pub(super) const LANDLOCK_ACCESS_FS_MAKE_SYM: u64 = 1 << 12;
pub(super) const LANDLOCK_ACCESS_FS_REFER: u64 = 1 << 13;
pub(super) const LANDLOCK_ACCESS_FS_TRUNCATE: u64 = 1 << 14;
pub(super) const LANDLOCK_ACCESS_FS_IOCTL_DEV: u64 = 1 << 15;
/// ABI v9 (kernel 7.1). Mediates `connect(2)` and addressed `sendmsg(2)` on *pathname* Unix
/// sockets, the class Landlock left entirely unmediated before it: the D-Bus system and session
/// buses, and `/run/systemd/private`. Without this bit in `handled_access_fs` a confined process
/// can hand work to a privileged daemon and have it done on its behalf, which is a complete bypass
/// of the filesystem boundary rather than a gap in it. `scoped` covers only *abstract* sockets, so
/// it does not reach these.
pub(super) const LANDLOCK_ACCESS_FS_RESOLVE_UNIX: u64 = 1 << 16;
pub(super) const LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET: u64 = 1 << 0;
pub(super) const LANDLOCK_SCOPE_SIGNAL: u64 = 1 << 1;
// Landlock kernel structs (stack-allocated, no heap)
#[repr(C)]
pub(super) struct LandlockRulesetAttr {
    pub(super) handled_access_fs: u64,
    pub(super) handled_access_net: u64,
    pub(super) scoped: u64,
}
#[repr(C)]
pub(super) struct LandlockPathBeneathAttr {
    pub(super) allowed_access: u64,
    pub(super) parent_fd: i32,
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, os::unix::process::CommandExt};

    /// End-to-end proof that the kernel honors the workspace boundary, not just that meka computed
    /// it: a real `sh` under a real Landlock ruleset, checked by what lands on disk. Skips itself
    /// when the host has no usable Landlock, since CI runs this matrix on macOS and Windows too.
    #[test]
    fn a_confined_shell_writes_inside_the_root_and_is_refused_outside() {
        let Some(abi) = super::landlock_abi().filter(|abi| *abi >= super::MIN_LANDLOCK_ABI) else {
            eprintln!("skipping: no usable Landlock on this host");
            return;
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        let outside = base.join("outside");
        std::fs::create_dir(&work).expect("work");
        std::fs::create_dir(&outside).expect("outside");

        // `OsStrExt::as_bytes`, matching the production path in `shell.rs`. `as_encoded_bytes` is
        // documented as an unspecified encoding and explicitly not for FFI, so a test using it
        // would be exercising a different conversion than the one that ships.
        let writable = vec![
            CString::new(std::os::unix::ffi::OsStrExt::as_bytes(work.as_os_str()))
                .expect("cstring"),
        ];
        // The exit code carries the result, so `status.success()` is an assertion rather than a
        // formality. The script must not end in `; true`, which makes "the confined shell itself
        // must run" pass whatever happened inside it: a ruleset denying every write and one
        // allowing every write produce the same success. Only the two `exists()` checks below were
        // doing any work.
        let script = format!(
            "echo in > {}/inside.txt 2>/dev/null || exit 3\n\
             if echo out > {}/escaped.txt 2>/dev/null; then exit 4; fi\n\
             exit 0",
            work.display(),
            outside.display()
        );

        let mut command = std::process::Command::new("/bin/sh");
        command.arg("-c").arg(&script);
        unsafe {
            command.pre_exec(move || {
                super::apply_landlock(abi, &writable).map_err(std::io::Error::from_raw_os_error)
            });
        }
        let status = command.status().expect("spawn");
        match status.code() {
            Some(0) => {}
            Some(3) => panic!("the write inside the workspace root was refused"),
            Some(4) => panic!("the write outside every root was permitted"),
            other => panic!("the confined shell did not run: exit {other:?}"),
        }

        assert!(
            work.join("inside.txt").exists(),
            "a write inside the workspace root must land"
        );
        assert!(
            !outside.join("escaped.txt").exists(),
            "a write outside every root must be refused by the kernel, not merely by meka"
        );
    }

    /// With no writable root, the same ruleset refuses both. This is the read-only case, and it
    /// proves the grant above came from the root rather than from Landlock not being applied.
    #[test]
    fn an_unwritable_confinement_refuses_even_the_workspace() {
        let Some(abi) = super::landlock_abi().filter(|abi| *abi >= super::MIN_LANDLOCK_ABI) else {
            eprintln!("skipping: no usable Landlock on this host");
            return;
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let script = format!("echo x > {}/nope.txt 2>/dev/null; true", base.display());

        let mut command = std::process::Command::new("/bin/sh");
        command.arg("-c").arg(&script);
        unsafe {
            command.pre_exec(move || {
                super::apply_landlock(abi, &[]).map_err(std::io::Error::from_raw_os_error)
            });
        }
        command.status().expect("spawn");

        assert!(
            !base.join("nope.txt").exists(),
            "read-only means read-only: granting nothing must write nothing"
        );
    }

    /// `2>/dev/null` works under Landlock, as it already did under Bubblewrap and Seatbelt.
    ///
    /// It did not before: Landlock granted only read and execute on `/`, so the redirect failed
    /// with a bare "Permission denied" and every `cmd 2>/dev/null` in an agent's shell broke on
    /// this backend alone. Discarding output is not a write to the machine.
    #[test]
    fn discarding_output_to_dev_null_is_permitted_in_every_confinement() {
        let Some(abi) = super::landlock_abi().filter(|abi| *abi >= super::MIN_LANDLOCK_ABI) else {
            eprintln!("skipping: no usable Landlock on this host");
            return;
        };

        // Both root lists, since the name says "every confinement" and the body tested one.
        // `&[]` is the `read` level; a real root is `workspace`. `/dev/null` is granted by its own
        // rule rather than by the roots, so it has to hold under both -- and a rule that
        // only worked when the root list happened to be empty would pass with a single root
        // list.
        let temp = tempfile::tempdir().expect("tempdir");
        let root = crate::workspace::canonical_for_test(temp.path());
        let workspace_root = [
            std::ffi::CString::new(std::os::unix::ffi::OsStrExt::as_bytes(root.as_os_str()))
                .expect("cstring"),
        ];

        for (label, writable) in [
            ("read level", &[] as &[std::ffi::CString]),
            ("workspace", &workspace_root[..]),
        ] {
            let writable = writable.to_vec();
            let mut command = std::process::Command::new("/bin/sh");
            command.arg("-c").arg("echo discarded > /dev/null");
            unsafe {
                command.pre_exec(move || {
                    super::apply_landlock(abi, &writable).map_err(std::io::Error::from_raw_os_error)
                });
            }
            let status = command.status().expect("spawn");
            assert!(
                status.success(),
                "a redirect to /dev/null must succeed under {label}, where nothing else outside \
                 the roots is writable"
            );
        }
    }
}
