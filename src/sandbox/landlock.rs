//! Landlock: the ABI probe, the ruleset meka builds for a write scope, and the raw syscall bindings
//! it needs.

use std::path::{Path, PathBuf};

use super::*;

pub(super) fn probe_landlock() -> BackendProbe {
    landlock_probe_from_abi(landlock_abi_or_errno())
}
/// The [`MIN_LANDLOCK_ABI`] policy, split from the syscall so it can be exercised at ABI values
/// this host does not have. Kernels below v3 are the ones that matter and are exactly the ones a
/// developer machine running a current kernel cannot reproduce.
///
/// A kernel without Landlock answers the probe with `ENOSYS`; one that has it but left it out of
/// the boot-time `lsm=` list answers `EOPNOTSUPP`. The remedies differ, a newer kernel against a
/// boot parameter, so the two are told apart rather than folded into "not supported".
pub(super) fn landlock_probe_from_abi(abi: Result<i32, i32>) -> BackendProbe {
    match abi {
        Ok(abi_version) if abi_version >= MIN_LANDLOCK_ABI => {
            BackendProbe::Ok(SandboxCapability::Landlock { abi_version })
        }
        Ok(abi_version) => BackendProbe::Missing {
            reason: format!(
                "Landlock ABI v{abi_version} leaves Unix sockets reachable; v{MIN_LANDLOCK_ABI} \
                 (Linux 7.1+) is required",
            ),
        },
        Err(libc::EOPNOTSUPP) => BackendProbe::Missing {
            reason: "Landlock is built into this kernel but disabled at boot; add `landlock` to \
                     the kernel's `lsm=` list"
                .to_string(),
        },
        Err(_) => BackendProbe::Missing {
            reason: "Landlock LSM not supported by this kernel (needs Linux 5.13+)".to_string(),
        },
    }
}
/// Lowest Landlock ABI meka will sandbox with on its own.
///
/// v9 (Linux 7.1) is where `LANDLOCK_ACCESS_FS_RESOLVE_UNIX` arrives. Below it no right governs
/// `connect(2)` on a pathname Unix socket, so a "read-only" child can still reach the D-Bus buses
/// and `systemd-run --user`, and have a privileged daemon write on its behalf: not a gap in the
/// filesystem boundary but a way around the whole of it. meka documents `read` as "this command
/// cannot change the machine", so accepting an older ABI would be promising a boundary the kernel
/// is not enforcing. The truncate right of v3 and the ioctl and scope rights of v5 and v6 come
/// with it.
///
/// Failing closed costs shell at `read` on kernels below 7.1 (Debian 13, RHEL 10, Ubuntu 24.04
/// and 26.04 LTS) that lack Bubblewrap. That is the intended trade: Bubblewrap is auto-preferred
/// whenever `bwrap` is on `PATH`, works on any kernel with user namespaces, and a refusal the user
/// can act on beats a guarantee that quietly does not hold.
pub(super) const MIN_LANDLOCK_ABI: i32 = 9;
/// Lowest Landlock ABI worth enacting inside a Bubblewrap sandbox.
///
/// Lower than [`MIN_LANDLOCK_ABI`] because the layer only ever adds restrictions to a boundary
/// bwrap already holds, so an incomplete one costs nothing. v6 (Linux 6.12) is where it first adds
/// anything the masks do not: the abstract socket namespace and signals to outside processes.
/// Below that the layer would only re-deny writes the read-only bind refuses, for the price of
/// an extra exec.
pub(super) const MIN_LAYER_LANDLOCK_ABI: i32 = 6;
const _: () = assert!(
    MIN_LAYER_LANDLOCK_ABI < MIN_LANDLOCK_ABI,
    "the layer only adds to a boundary bwrap holds, so its floor sits below the standalone one"
);
/// The kernel's Landlock ABI when it clears [`MIN_LAYER_LANDLOCK_ABI`]: what the layer inside
/// Bubblewrap runs at, read by the bwrap probe and by `meka confine` alike.
pub(crate) fn layer_landlock_abi() -> Option<i32> {
    landlock_abi_or_errno()
        .ok()
        .filter(|abi| *abi >= MIN_LAYER_LANDLOCK_ABI)
}
/// Raw kernel ABI probe. Reports what the kernel supports, not what meka will accept: the
/// [`MIN_LANDLOCK_ABI`] policy lives in [`probe_landlock`] so the "too old" case can be reported
/// differently from "no Landlock at all". `Err` carries the `errno` the kernel answered with.
pub(super) fn landlock_abi_or_errno() -> Result<i32, i32> {
    let version = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<u8>(),
            0usize,
            LANDLOCK_CREATE_RULESET_VERSION,
        )
    };
    if version >= 1 {
        Ok(version as i32)
    } else {
        Err(unsafe { *libc::__errno_location() })
    }
}
/// One rule of a ruleset: a path and the rights granted beneath it.
///
/// Planned in the parent by [`landlock_grants`], which may allocate, and applied by
/// [`apply_landlock`] after `fork`, which may not. The split is what lets the ruleset be computed
/// from the filesystem (walking siblings, telling files from directories) and still be applied in
/// a `pre_exec` closure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LandlockGrant {
    pub(crate) path: std::ffi::CString,
    pub(crate) allowed: u64,
}
/// The rights a rule on anything but a directory may carry. The kernel answers a rule whose parent
/// is a regular file and whose rights include a directory-class one with `EINVAL`, and a failed
/// `add_rule` takes the whole spawn down.
pub(super) const ACCESS_FILE: u64 = LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_TRUNCATE
    | LANDLOCK_ACCESS_FS_IOCTL_DEV
    | LANDLOCK_ACCESS_FS_RESOLVE_UNIX;
/// The rules for a confinement: read and execute everywhere but inside meka's own directories,
/// every handled right beneath each writable root, every right but socket resolution beneath each
/// scratch directory, and `/dev/null` writable in every confinement.
///
/// `writable` holds the canonical workspace roots (empty for a read-only confinement), `scratch`
/// the throwaway space a command may fill (its temporary directory, or Bubblewrap's masks), and
/// `private` the canonical directories to hide. Scratch space gets no `RESOLVE_UNIX`: a socket a
/// command creates there is reachable by the domain rule regardless, so the right would only ever
/// reach a socket from outside that a bind put there, such as one in a working directory under
/// `/tmp` bound back through Bubblewrap's mask. Landlock rules only ever add access, so a directory
/// is hidden by never being covered by a rule: the read grant on `/` keeps only listing and
/// execution, and reading files is granted per sibling along the way down to each private
/// directory, on everything except the private directory and the directories leading to it. A
/// workspace root above a private directory is split the same way, or the root's own rule would
/// hand the store back; the cost is that a file can then be created directly in such a root only by
/// a rule on the root itself, which is the rule being withheld, so new files land in its
/// subdirectories while its existing files stay writable. Listing stays global because `READ_DIR`
/// on a directory covers everything beneath it, so granting it per sibling would refuse `ls ~` at
/// `read`; the store's file names and sizes are visible either way, since `stat(2)` is unmediated,
/// and its bytes are what the walk protects.
///
/// Each grant is a path rather than a descriptor because the roots can go away: a concurrent
/// `rm -rf` on one between planning and the spawn makes [`apply_landlock`] skip it, which is the
/// restrictive direction, where a descriptor would keep a deleted directory reachable.
pub(crate) fn landlock_grants(
    abi_version: i32,
    writable: &[PathBuf],
    scratch: &[PathBuf],
    private: &[PathBuf],
) -> Vec<LandlockGrant> {
    let handled = handled_access_for_abi(abi_version);
    let mut grants = Vec::new();
    for directory in scratch {
        push_grant(
            &mut grants,
            directory,
            handled & !LANDLOCK_ACCESS_FS_RESOLVE_UNIX,
        );
    }
    // `/dev/null` is writable in every confinement, including read-only, as it is under the macOS
    // profile and Bubblewrap's `--dev /dev`: `cmd 2>/dev/null` discards output rather than writing
    // to the machine, and refusing it confines nothing. Resolved first because the rule lands on
    // the inode the descriptor names, and `apply_landlock` opens without following a link: a
    // container that makes `/dev/null` a symlink would otherwise get a rule on the link itself.
    let dev_null = Path::new("/dev/null");
    push_grant(
        &mut grants,
        &std::fs::canonicalize(dev_null).unwrap_or_else(|_| dev_null.to_path_buf()),
        LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_READ_FILE,
    );
    if private.is_empty() {
        push_grant(
            &mut grants,
            Path::new("/"),
            LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR,
        );
    } else {
        push_grant(
            &mut grants,
            Path::new("/"),
            LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_DIR,
        );
        sibling_grants(
            &mut grants,
            Path::new("/"),
            private,
            LANDLOCK_ACCESS_FS_READ_FILE,
        );
    }
    // Every right the ruleset handles beneath each root. "Writable" means the full set rather than
    // just `WRITE_FILE`: creating, removing, renaming and truncating are all separate rights, and
    // a shell that can write bytes but not create a file would fail on the first `>` redirect.
    for root in writable {
        if private.iter().any(|directory| directory.starts_with(root)) {
            sibling_grants(&mut grants, root, private, handled);
        } else {
            push_grant(&mut grants, root, handled);
        }
    }
    grants
}
/// Grant `rights` on every entry beneath `top` that is neither a private directory nor on the way
/// to one, walking only the directories on those ways. `top` itself gets no rule.
///
/// A symlink entry gets nothing: a rule lands on the inode the descriptor names, and following the
/// link would put the rule on its target, which may be the very directory being hidden. Reading
/// through the link still works wherever the target's own ancestors grant it.
fn sibling_grants(grants: &mut Vec<LandlockGrant>, top: &Path, private: &[PathBuf], rights: u64) {
    let mut excluded: Vec<&Path> = Vec::new();
    for directory in private
        .iter()
        .filter(|directory| directory.starts_with(top))
    {
        for ancestor in directory.ancestors() {
            if !ancestor.starts_with(top) {
                break;
            }
            if !excluded.contains(&ancestor) {
                excluded.push(ancestor);
            }
        }
    }
    for directory in &excluded {
        if private.iter().any(|hidden| hidden == directory) {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if excluded.contains(&path.as_path()) {
                continue;
            }
            let Ok(metadata) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.file_type().is_symlink() {
                continue;
            }
            let allowed = if metadata.is_dir() {
                rights
            } else {
                rights & ACCESS_FILE
            };
            push_grant(grants, &path, allowed);
        }
    }
}
/// A path whose bytes hold a NUL cannot be named to the kernel; it grants nothing, which is the
/// restrictive direction.
fn push_grant(grants: &mut Vec<LandlockGrant>, path: &Path, allowed: u64) {
    if let Ok(path) =
        std::ffi::CString::new(std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str()))
    {
        grants.push(LandlockGrant { path, allowed });
    }
}
/// Apply Landlock restrictions to the current process: the ruleset for `abi_version`, with
/// `grants` as its rules.
///
/// The grants arrive planned, as [`std::ffi::CString`] paths, because this runs after `fork`:
/// planning them here would read the filesystem and allocate, which the safety contract below
/// forbids, so [`landlock_grants`] does that in the parent.
///
/// Landlock rules are additive grants with no deny form, so a writable root cannot have a subtree
/// carved back out of it; the planner expresses a hidden directory by never covering it.
///
/// # Safety
///
/// This function uses raw syscalls and must only be called where the process is single-threaded:
/// in a `pre_exec` context (after fork, before exec), or at the top of `main` before any thread
/// starts, as [`run_confined`] does. All operations are async-signal-safe (syscalls only, no heap
/// allocation).
pub(crate) unsafe fn apply_landlock(abi_version: i32, grants: &[LandlockGrant]) -> Result<(), i32> {
    unsafe {
        // `PR_SET_NO_NEW_PRIVS` is required for unprivileged Landlock use.
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

        for grant in grants {
            // `O_NOFOLLOW`: the planner skipped symlink entries, and a path swapped for a link
            // since then must not put the rule on the link's target.
            let fd = libc::open(
                grant.path.as_ptr(),
                libc::O_PATH | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            );
            if fd < 0 {
                // A path that cannot be opened grants nothing. Skipping rather than failing keeps
                // a deleted root or a vanished sibling from turning every command into a spawn
                // error, and the effect is restrictive: the confinement stays as tight as it was.
                continue;
            }
            let path_beneath = LandlockPathBeneathAttr {
                allowed_access: grant.allowed,
                parent_fd: fd,
            };
            let ret = libc::syscall(
                libc::SYS_landlock_add_rule,
                ruleset_fd,
                LANDLOCK_RULE_PATH_BENEATH,
                &path_beneath as *const LandlockPathBeneathAttr,
                0u32,
            );
            // `close(2)` is permitted to set `errno` even on success, so read the failure reason
            // before releasing the descriptor.
            let error = *libc::__errno_location();
            libc::close(fd);
            if ret < 0 {
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
/// The body of `meka confine`: enact the ruleset inside a Bubblewrap sandbox, then become the
/// command. Returns only on failure, with what went wrong; on success the process is replaced.
///
/// Bubblewrap has no Landlock support of its own, and a domain that handles any filesystem right
/// refuses `mount(2)`, so the ruleset cannot be applied ahead of bwrap's mounts. The one place left
/// is a program bwrap execs, and that program is meka itself, reached through a descriptor the
/// spawn lets the sandbox inherit.
///
/// The ABI is probed here rather than passed in: it is the same kernel the parent probed, and
/// carrying the answer through argv would only be a second place for the floor to be decided. No
/// private directories either, since bwrap's masks already hide them and the walk would only find
/// empty masks.
pub(crate) fn run_confined(
    writable: &[PathBuf],
    scratch: &[PathBuf],
    command: &[std::ffi::OsString],
) -> std::io::Error {
    let Some(abi_version) = layer_landlock_abi() else {
        return std::io::Error::other(format!(
            "Landlock ABI v{MIN_LAYER_LANDLOCK_ABI} (Linux 6.12+) is needed inside Bubblewrap"
        ));
    };
    let Some((program, arguments)) = command.split_first() else {
        return std::io::Error::other("no command to run");
    };
    let grants = landlock_grants(abi_version, writable, scratch, &[]);
    // SAFETY: called from the top of `main`, before tracing or the runtime has started a thread,
    // so the process is single-threaded; the no-allocation half of the contract exists for the
    // `pre_exec` caller and is not what this call site relies on.
    if let Err(errno) = unsafe { apply_landlock(abi_version, &grants) } {
        return std::io::Error::from_raw_os_error(errno);
    }
    // Everything above stdio was inherited, the descriptor this image was exec'd from included,
    // and none of it is the command's: a descriptor reaches past every rule, since Landlock judges
    // opens rather than what a process already holds.
    //
    // SAFETY: `close_range(2)` takes plain integers and touches nothing else.
    if unsafe { libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, 0u32) } == -1 {
        return std::io::Error::last_os_error();
    }
    std::os::unix::process::CommandExt::exec(std::process::Command::new(program).args(arguments))
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
    // ABI v4 added only network flags.
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
    use std::{collections::HashMap, os::unix::process::CommandExt, path::PathBuf};

    /// The host's usable ABI, or a loud skip: CI runs this matrix on macOS and Windows too, and
    /// on a Linux runner whose kernel is too old.
    fn usable_abi() -> Option<i32> {
        let abi = super::landlock_abi_or_errno()
            .ok()
            .filter(|abi| *abi >= super::MIN_LANDLOCK_ABI);
        if abi.is_none() {
            eprintln!("skipping: no usable Landlock on this host");
        }
        abi
    }

    /// Run `script` under the ruleset and hand back its exit code, which carries the result: a
    /// script ending in `; true` would make a ruleset denying every write and one allowing every
    /// write produce the same success.
    fn confined_exit(abi: i32, grants: Vec<super::LandlockGrant>, script: &str) -> Option<i32> {
        let mut command = std::process::Command::new("/bin/sh");
        command.arg("-c").arg(script);
        unsafe {
            command.pre_exec(move || {
                super::apply_landlock(abi, &grants).map_err(std::io::Error::from_raw_os_error)
            });
        }
        command.status().expect("spawn").code()
    }

    fn by_path(grants: &[super::LandlockGrant]) -> HashMap<PathBuf, u64> {
        grants
            .iter()
            .map(|grant| {
                let path: &std::ffi::OsStr =
                    std::os::unix::ffi::OsStrExt::from_bytes(grant.path.as_bytes());
                (PathBuf::from(path), grant.allowed)
            })
            .collect()
    }

    /// End-to-end proof that the kernel honors the workspace boundary, not just that meka computed
    /// it: a real `sh` under a real Landlock ruleset, checked by what lands on disk.
    #[test]
    fn a_confined_shell_writes_inside_the_root_and_is_refused_outside() {
        let Some(abi) = usable_abi() else {
            return;
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        let outside = base.join("outside");
        std::fs::create_dir(&work).expect("work");
        std::fs::create_dir(&outside).expect("outside");

        let grants = super::landlock_grants(abi, std::slice::from_ref(&work), &[], &[]);
        let script = format!(
            "echo in > {}/inside.txt 2>/dev/null || exit 3\n\
             if echo out > {}/escaped.txt 2>/dev/null; then exit 4; fi\n\
             exit 0",
            work.display(),
            outside.display()
        );
        match confined_exit(abi, grants, &script) {
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
        let Some(abi) = usable_abi() else {
            return;
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let script = format!("echo x > {}/nope.txt 2>/dev/null; true", base.display());
        confined_exit(abi, super::landlock_grants(abi, &[], &[], &[]), &script);

        assert!(
            !base.join("nope.txt").exists(),
            "read-only means read-only: granting nothing must write nothing"
        );
    }

    /// `2>/dev/null` works under Landlock, as it does under Bubblewrap and Seatbelt: discarding
    /// output is not a write to the machine.
    #[test]
    fn discarding_output_to_dev_null_is_permitted_in_every_confinement() {
        let Some(abi) = usable_abi() else {
            return;
        };

        // Both root lists: `&[]` is the `read` level, a real root is `workspace`. `/dev/null` is
        // granted by its own rule rather than by the roots, so it has to hold under both.
        let temp = tempfile::tempdir().expect("tempdir");
        let root = crate::workspace::canonical_for_test(temp.path());
        for (label, writable) in [("read level", vec![]), ("workspace", vec![root])] {
            let grants = super::landlock_grants(abi, &writable, &[], &[]);
            assert_eq!(
                confined_exit(abi, grants, "echo discarded > /dev/null"),
                Some(0),
                "a redirect to /dev/null must succeed under {label}, where nothing else outside \
                 the roots is writable"
            );
        }
    }

    /// A kernel that has Landlock but left it out of `lsm=` is told the boot parameter, not sent
    /// after a newer kernel it already has; one without it is sent after the kernel.
    #[test]
    fn a_kernel_that_left_landlock_out_of_its_boot_list_is_told_the_remedy() {
        let disabled = super::backend_unavailable_reason(&super::landlock_probe_from_abi(Err(
            libc::EOPNOTSUPP,
        )))
        .expect("unusable");
        assert!(
            disabled.contains("disabled at boot") && disabled.contains("`lsm=`"),
            "the boot list is the remedy: {disabled}"
        );
        let absent =
            super::backend_unavailable_reason(&super::landlock_probe_from_abi(Err(libc::ENOSYS)))
                .expect("unusable");
        assert!(
            absent.contains("5.13") && !absent.contains("lsm="),
            "a kernel without Landlock needs a newer kernel: {absent}"
        );
    }

    /// The planner never names a private directory or a directory on the way to it: a rule on any
    /// of those would cover the store beneath. Siblings get the rights, a file only the file-class
    /// subset the kernel accepts on one, and a symlink entry nothing at all.
    #[test]
    fn grants_step_around_a_private_directory_and_its_ancestors() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let private = base.join("a/b/meka");
        std::fs::create_dir_all(&private).expect("private");
        std::fs::create_dir(base.join("a/b/other")).expect("other");
        std::fs::write(base.join("a/file"), "x").expect("file");
        std::os::unix::fs::symlink("b", base.join("a/link")).expect("link");
        std::fs::create_dir(base.join("open")).expect("open");
        let handled = super::handled_access_for_abi(9);
        let read = super::LANDLOCK_ACCESS_FS_EXECUTE | super::LANDLOCK_ACCESS_FS_READ_DIR;

        let grants = by_path(&super::landlock_grants(
            9,
            std::slice::from_ref(&base.join("a")),
            &[],
            std::slice::from_ref(&private),
        ));
        for excluded in [
            base.clone(),
            base.join("a"),
            base.join("a/b"),
            private.clone(),
            base.join("a/link"),
        ] {
            assert!(
                !grants.contains_key(&excluded),
                "'{}' must carry no rule of its own",
                excluded.display()
            );
        }
        assert_eq!(
            grants[&PathBuf::from("/")],
            read,
            "reading files is not global"
        );
        assert_eq!(
            grants[&base.join("open")],
            super::LANDLOCK_ACCESS_FS_READ_FILE
        );
        assert_eq!(grants[&base.join("a/b/other")], handled);
        assert_eq!(grants[&base.join("a/file")], handled & super::ACCESS_FILE);

        // Nothing to hide: one rule on `/` that reads files, one on the root.
        let plain = by_path(&super::landlock_grants(
            9,
            std::slice::from_ref(&base.join("a")),
            &[],
            &[],
        ));
        assert_eq!(
            plain[&PathBuf::from("/")],
            read | super::LANDLOCK_ACCESS_FS_READ_FILE
        );
        assert_eq!(plain[&base.join("a")], handled);
        assert_eq!(plain.len(), 3, "`/dev/null`, `/` and the root: {plain:?}");

        // Scratch space: everything but socket resolution.
        let scratch = by_path(&super::landlock_grants(
            9,
            &[],
            std::slice::from_ref(&base.join("open")),
            &[],
        ));
        assert_eq!(
            scratch[&base.join("open")],
            handled & !super::LANDLOCK_ACCESS_FS_RESOLVE_UNIX
        );
    }

    /// A confined shell cannot read a file inside a private directory, can read its sibling, and
    /// can still list the private directory: the bytes are the secret, the names are not.
    #[test]
    fn a_private_directory_is_unreadable_and_its_siblings_are_not() {
        let Some(abi) = usable_abi() else {
            return;
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let private = base.join("private");
        let open = base.join("open");
        std::fs::create_dir(&private).expect("private");
        std::fs::create_dir(&open).expect("open");
        std::fs::write(private.join("secret.txt"), "s").expect("secret");
        std::fs::write(open.join("file.txt"), "f").expect("file");

        let grants = super::landlock_grants(abi, &[], &[], std::slice::from_ref(&private));
        let script = format!(
            "cat {open}/file.txt >/dev/null 2>&1 || exit 3\n\
             ls {private} >/dev/null 2>&1 || exit 5\n\
             if cat {private}/secret.txt >/dev/null 2>&1; then exit 4; fi\n\
             exit 0",
            open = open.display(),
            private = private.display()
        );
        match confined_exit(abi, grants, &script) {
            Some(0) => {}
            Some(3) => panic!("a file beside the private directory was refused"),
            Some(4) => panic!("a file inside the private directory was readable"),
            Some(5) => panic!("listing the private directory was refused"),
            other => panic!("the confined shell did not run: exit {other:?}"),
        }
    }

    /// A workspace root above a private directory: writes land beside it and are refused inside.
    /// This is the case a root at `$HOME` produces, and a rule on the root itself would hand the
    /// store back. The cost is pinned too: creating a file directly in the root is refused, because
    /// the only rule that could grant it is the withheld one, while an existing file there stays
    /// writable and a sibling directory takes new files.
    #[test]
    fn a_root_above_a_private_directory_writes_beside_it_and_not_inside_it() {
        let Some(abi) = usable_abi() else {
            return;
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let private = base.join("private");
        let other = base.join("other");
        std::fs::create_dir(&private).expect("private");
        std::fs::create_dir(&other).expect("other");
        std::fs::write(base.join("existing.txt"), "old\n").expect("existing");

        let grants = super::landlock_grants(
            abi,
            std::slice::from_ref(&base),
            &[],
            std::slice::from_ref(&private),
        );
        let script = format!(
            "echo x > {base}/existing.txt 2>/dev/null || exit 3\n\
             echo x > {other}/in.txt 2>/dev/null || exit 5\n\
             if echo x > {private}/inside.txt 2>/dev/null; then exit 4; fi\n\
             if echo x > {base}/fresh.txt 2>/dev/null; then exit 6; fi\n\
             exit 0",
            base = base.display(),
            other = other.display(),
            private = private.display()
        );
        match confined_exit(abi, grants, &script) {
            Some(0) => {}
            Some(3) => panic!("an existing file directly under the root was not writable"),
            Some(4) => panic!("a write inside the private directory was permitted"),
            Some(5) => panic!("a write in a sibling directory under the root was refused"),
            Some(6) => panic!(
                "a file was created directly in the root, which only a rule covering the private \
                 directory could grant"
            ),
            other => panic!("the confined shell did not run: exit {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(base.join("existing.txt")).expect("read back"),
            "x\n"
        );
        assert!(other.join("in.txt").exists());
        assert!(!private.join("inside.txt").exists());
        assert!(!base.join("fresh.txt").exists());
    }
}
