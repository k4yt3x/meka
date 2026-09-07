//! Filesystem helpers with a security property: private directories born at the right mode,
//! files replaced atomically, file identity for lock retakes, and the file locks the config
//! directory, the skill store and each session hold.
//!
//! Three lock shapes live here side by side because three callers grew them separately; they are
//! one primitive with different waiting policies.

use std::{
    fs::{File, OpenOptions},
    path::Path,
};

use crate::error::{MekaError, Result};

/// Resolve `path` through a final-component symlink, leaving anything else untouched.
///
/// Returns `path` unchanged when it is not a link, when the link cannot be read, or when the target
/// is itself a link that leads nowhere useful: every failure mode falls back to writing where the
/// caller asked. A relative link target is joined onto the link's own directory, as the OS
/// resolves it.
///
/// Chains are followed to a small depth so a link-to-a-link still lands on the real file, with the
/// bound there to stop a cycle (`a -> b -> a`) spinning.
pub(crate) fn resolve_symlink_target(path: &Path) -> std::path::PathBuf {
    const MAX_LINK_DEPTH: usize = 8;

    let mut current = path.to_path_buf();
    for _ in 0..MAX_LINK_DEPTH {
        let is_link = std::fs::symlink_metadata(&current)
            .map(|metadata| metadata.file_type().is_symlink())
            .unwrap_or(false);
        if !is_link {
            return current;
        }
        let Ok(target) = std::fs::read_link(&current) else {
            return current;
        };
        current = if target.is_absolute() {
            target
        } else {
            match current.parent() {
                Some(parent) => parent.join(target),
                None => return current,
            }
        };
    }
    tracing::warn!(
        "'{path}' is a symlink chain deeper than {MAX_LINK_DEPTH} links; writing to the link itself",
        path = path.display()
    );
    path.to_path_buf()
}
/// `(device, inode)` on Unix, `(volume serial, file index)` on Windows: what makes "the thing I
/// locked" and "the thing at this path" the same question.
pub(crate) type FileIdentity = (u64, u64);
/// Read the identity of an already-open file, from the descriptor rather than the path.
///
/// A platform that is neither Unix nor Windows gets a compile error here rather than a lock nothing
/// revalidates.
#[cfg(unix)]
pub(crate) fn file_identity(file: &std::fs::File) -> std::io::Result<FileIdentity> {
    use std::os::unix::fs::MetadataExt;

    let metadata = file.metadata()?;
    Ok((metadata.dev(), metadata.ino()))
}
#[cfg(windows)]
pub(crate) fn file_identity(file: &std::fs::File) -> std::io::Result<FileIdentity> {
    use std::os::windows::io::AsRawHandle;

    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };

    // SAFETY: `BY_HANDLE_FILE_INFORMATION` is plain data with no niche, so an all-zero value is a
    // valid one to hand out for filling.
    let mut information: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: the handle is borrowed from a live `File` for the duration of the call, and the
    // pointer is to a stack local of exactly the type the call expects.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle() as _, &mut information) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok((
        u64::from(information.dwVolumeSerialNumber),
        (u64::from(information.nFileIndexHigh) << 32) | u64::from(information.nFileIndexLow),
    ))
}
/// How many times [`lock_path`] will re-take a lock whose resource was replaced underneath it.
///
/// A retake costs one extra wait, and each one needs the resource to have been replaced while this
/// process was blocked on the previous inode. Bounded so a caller cannot wait forever on something
/// replacing the resource in a loop.
pub(crate) const MAX_LOCK_RETAKES: usize = 32;
/// An exclusive `flock` keyed to the inode currently at a path, so a lock can be taken on the
/// resource itself rather than on a file beside it.
///
/// Holding the open file is holding the lock: closing the descriptor releases it, on drop or when
/// the kernel reaps the process.
pub(crate) struct PathLock {
    _file: std::fs::File,
}
/// Take [`PathLock`] on `path`, opened by `open`, and prove afterwards that it is still the live
/// resource.
///
/// A waiter blocks on the inode it opened, and that inode can be removed and something else
/// recreated under the same name before it wakes. Re-reading the identity is what stops it holding
/// an unlinked inode nobody else can reach while the next arrival locks the replacement, each blind
/// to the other.
///
/// The window this closes is the wait, not the whole critical section: a replacement landing after
/// the check leaves the holder on the old inode with no way to notice, exactly as it would have
/// left a lock file that the same `rm -rf` deleted. Nothing here defends against that.
///
/// Each iteration blocks in the kernel rather than polling, and only repeats when the check fails.
pub(crate) fn lock_path(
    path: &Path,
    open: fn(&Path) -> std::io::Result<std::fs::File>,
) -> std::io::Result<PathLock> {
    for _ in 0..MAX_LOCK_RETAKES {
        let file = open(path)?;
        let held = file_identity(&file)?;
        file.lock()?;

        let live = open(path).and_then(|current| file_identity(&current));
        match live {
            Ok(live) if live == held => return Ok(PathLock { _file: file }),
            _ => drop(file),
        }
    }
    // Not `WouldBlock`, which promises the caller that nothing was attempted and a retry may work.
    // This call did block, repeatedly, and gave up.
    Err(std::io::Error::other(format!(
        "gave up locking '{}' after {} retakes; something is replacing it continuously",
        path.display(),
        MAX_LOCK_RETAKES
    )))
}
/// Open whichever inode carries the config lock.
///
/// The directory, not `config.toml`. Editors publish the file by `rename`, so a lock on the file
/// stops excluding anyone the moment its holder writes: the holder is left on an unlinked inode
/// while an arrival locks the replacement and enters a section the holder has not left.
/// [`crate::cli::mcp`]'s `purge_server` would notice, revoking a credential over the network after
/// its write. A directory's inode survives every write beneath it, and locking one adds no file to
/// a tree people keep under version control.
#[cfg(unix)]
pub(crate) fn open_config_lock_target(directory: &Path) -> std::io::Result<std::fs::File> {
    std::fs::File::open(directory)
}
/// A file, because Windows locks are mandatory rather than advisory and `File::lock` takes the
/// whole byte range, so a lock on `config.toml` makes it unreadable to the read-modify-write
/// holding it (`ERROR_LOCK_VIOLATION` from `read_to_string` on the owning thread). `LockFileEx`
/// also rejects a directory handle with `ERROR_INVALID_PARAMETER`, leaving nowhere else to put it.
#[cfg(windows)]
pub(crate) fn open_config_lock_target(directory: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(directory.join(".config.toml.lock"))
}
/// Remove the temporary file a failed atomic write left behind; the write's own error is the one
/// the caller sees, so a failure here is only logged.
fn remove_temporary(temporary_path: &Path) {
    if let Err(error) = std::fs::remove_file(temporary_path) {
        tracing::debug!(
            "failed to remove the temporary file {temporary_path}: {error}",
            temporary_path = temporary_path.display()
        );
    }
}

/// Write `content` to `path` atomically: serialize to a `<name>.<pid>.<seq>.tmp` beside it,
/// `sync_all` the fd, then `rename` over the target. Also creates the parent directory (0700 on
/// Unix, unless a symlink redirected the write out of meka's own tree) and holds the final file to
/// at most 0600 on Unix. meka's own secrets live in the database, but an MCP server's `headers` and
/// `env` are the user's to fill and may hold one verbatim, so the file is kept off-limits to group
/// and other regardless of the user's umask or of what mode the target already had.
///
/// Not config-specific despite living here: the same durability and permission guarantees are what
/// the skill store under the config dir wants, so `meka skill` bodies go through it too.
///
/// A symlinked target is followed (see [`resolve_symlink_target`]), so writing through a
/// dotfile-manager link updates the tracked file instead of replacing the link.
pub(crate) fn write_file_atomic(path: &Path, content: &str) -> std::io::Result<()> {
    use std::io::Write as _;

    // Follow a symlink to its target before choosing where to write.
    //
    // `rename(2)` replaces the directory entry, so renaming over a symlink destroys the link itself
    // rather than updating what it points at. Dotfile managers (stow, chezmoi, yadm) leave
    // `~/.config/meka/config.toml` as a link into a tracked repo, and one `meka mcp add` would
    // turn it into a plain file: the tracked copy goes stale, every later edit diverges from it,
    // and the next `stow --restow` silently reverts the lot. Resolving first keeps the write on
    // the file the user actually manages, and the atomicity below is unaffected because the temp
    // file is then created beside the *target*.
    //
    // Only the link itself is resolved, not the whole path: `canonicalize` would also resolve
    // symlinked parent directories, which changes where the temp file lands for no benefit here.
    let resolved = resolve_symlink_target(path);
    // Whether the write is still landing inside meka's own tree. The 0700 tightening below is a
    // statement about a directory meka owns, and following a link takes the write somewhere it does
    // not: with a dotfile manager's `~/dotfiles/config.toml`, one `meka mcp disable` would re-mode
    // the whole repository directory to 0700, leaving every unrelated file in it newly hidden, a
    // permission fact meka never established and cannot restore. Consulted only by the Unix
    // mode-tightening branch below; there is no mode to tighten elsewhere, so computing it on
    // those targets is dead work the compiler rightly flags.
    #[cfg(unix)]
    let redirected = resolved.as_path() != path;
    let path = resolved.as_path();

    if let Some(parent) = path.parent() {
        // Create newly-missing parents already at 0700 to avoid the umask window left by
        // `create_dir_all` followed by `set_permissions`. `DirBuilderExt::mode` passes the mode
        // straight to `mkdir(2)`.
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new()
                .mode(0o700)
                .recursive(true)
                .create(parent)?;
        }
        #[cfg(not(unix))]
        std::fs::create_dir_all(parent)?;

        #[cfg(unix)]
        if !redirected {
            use std::os::unix::fs::PermissionsExt;
            // Pre-existing dirs may have a different mode (e.g. user pre-created `~/.config` at
            // 0755). Best-effort tighten to 0700; failure here gets a warning rather than aborting
            // the write.
            if let Ok(metadata) = std::fs::metadata(parent) {
                let mut permissions = metadata.permissions();
                if permissions.mode() & 0o777 != 0o700 {
                    permissions.set_mode(0o700);
                    if let Err(error) = std::fs::set_permissions(parent, permissions) {
                        tracing::warn!(
                            "failed to tighten '{parent}' to 0700: {error}",
                            parent = parent.display()
                        );
                    }
                }
            }
        }
    }

    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no file name")
    })?;
    // Unique per writer, and created exclusively. A fixed `<file>.tmp` is shared by every writer of
    // the same path: both `open(O_TRUNC)` the same inode, both write from offset zero, and whoever
    // renames last publishes a byte-level splice of two documents as a success (4 MiB of one body
    // with 64 KiB of another laid over its front, renamed into place, `Ok(())` returned to both).
    // The pid separates processes and the counter separates threads within one; `create_new` is
    // what makes the pair a guarantee rather than a strong hope. `write_file_bytes` in
    // `crate::tools::file` carries the same pair for the same reason.
    static TEMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let (temporary_path, mut file) = loop {
        let mut tmp_name = file_name.to_os_string();
        tmp_name.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let candidate = path.with_file_name(tmp_name);
        match options.open(&candidate) {
            Ok(file) => break (candidate, file),
            // Only a name collision is worth retrying; anything else is the real error.
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    // Every failure arm removes the temp file. A unique name means a leftover is never reused, so
    // without this each interrupted write leaves one behind for ever.
    let write = (|| -> std::io::Result<()> {
        file.write_all(content.as_bytes())?;
        file.sync_all()
    })();
    drop(file);
    if let Err(error) = write {
        remove_temporary(&temporary_path);
        return Err(error);
    }

    // An existing file keeps whatever mode it had, *narrowed* to at most 0600.
    //
    // Two failures meet here and only this rule avoids both. Handing the target the temp's fresh
    // 0600 unconditionally re-modes a file meka did not create, which is wrong for a `config.toml`
    // a dotfile manager owns. But simply carrying the existing mode across is worse: it leaves a
    // hand-written `headers = { Authorization = "Bearer sk-..." }` sitting in a world-readable file
    // after any ordinary `meka mcp` edit, because the write follows the symlink out of meka's 0700
    // directory and the group and other bits came along. Narrowing keeps a deliberately *tighter*
    // mode like 0400 and refuses to publish a looser one.
    //
    // Said out loud when it bites, because it is a change to something the user set.
    #[cfg(unix)]
    if let Ok(existing) = std::fs::metadata(path) {
        use std::os::unix::fs::PermissionsExt;
        let existing_mode = existing.permissions().mode() & 0o777;
        let narrowed = existing_mode & 0o600;
        if narrowed != existing_mode {
            tracing::warn!(
                "tightening '{path}' from {existing_mode:o} to {narrowed:o}; it may hold credentials",
                path = path.display()
            );
        }
        if let Err(error) =
            std::fs::set_permissions(&temporary_path, std::fs::Permissions::from_mode(narrowed))
        {
            tracing::warn!(
                "failed to set the mode of '{path}' on the replacement: {error}",
                path = path.display()
            );
        }
    }

    std::fs::rename(&temporary_path, path).inspect_err(|_| remove_temporary(&temporary_path))?;
    sync_parent_directory(path);
    Ok(())
}

/// Sync the directory that names `path`, after a rename installed it.
///
/// The data was synced before the rename; the directory entry that now names it was not, so a
/// power loss here could leave the name pointing at neither file. Best effort: the write itself is
/// done, and a directory that cannot be synced is not a reason to report it as failed. One
/// definition for both atomic writers, so neither is the one that forgets.
pub(crate) fn sync_parent_directory(path: &Path) {
    #[cfg(unix)]
    if let Some(parent) = path.parent()
        && let Err(error) = std::fs::File::open(parent).and_then(|directory| directory.sync_all())
    {
        let path = path.display();
        tracing::warn!("failed to sync the directory holding {path}: {error}");
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}
/// Create a directory (and any missing parents) born at mode 0700 on Unix. Avoids the umask window
/// that `create_dir_all` + later `set_permissions` would open: between `mkdir(2)` and `chmod(2)`,
/// the directory would be readable by other local users on a permissive umask.
/// `DirBuilderExt::mode` passes the mode straight to `mkdir`. Pre-existing directories keep their
/// mode; callers that need to tighten an already-existing dir should still follow up with
/// `restrict_permissions`.
#[cfg(unix)]
pub(crate) fn create_private_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .recursive(true)
        .create(path)
}
#[cfg(not(unix))]
pub(crate) fn create_private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)
}
/// Restrict a path's permissions on Unix. Best-effort: a failure is logged, because on some mounts
/// (`/tmp` under some overlay setups, NFS without proper support) `chmod` returns `EPERM` or
/// `EROFS`, and refusing to open the store is a strictly worse failure than leaving the file at
/// the umask-derived mode.
#[cfg(unix)]
pub(crate) fn restrict_permissions(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(metadata) => {
            let mut permissions = metadata.permissions();
            permissions.set_mode(mode);
            if let Err(error) = std::fs::set_permissions(path, permissions) {
                tracing::warn!(
                    "failed to restrict '{path}' to mode {mode:o}: {error}",
                    path = path.display()
                );
            }
        }
        Err(error) => {
            tracing::warn!(
                "failed to stat '{path}' while restricting its permissions: {error}",
                path = path.display()
            );
        }
    }
}
#[cfg(not(unix))]
pub(crate) fn restrict_permissions(_path: &Path, _mode: u32) {
    // Windows ACLs inherit from the parent directory; leave alone.
}
/// Name of the sidecar lock file in a store root, which Windows needs because `LockFileEx` refuses
/// a directory handle. Ignored by discovery, which walks directories.
#[cfg(windows)]
pub(crate) const STORE_LOCK_FILE: &str = ".meka-store.lock";
/// An exclusive `flock` on one store root, held until dropped.
///
/// Taken on the root directory itself wherever the platform allows it, for the reason
/// `config::open_config_lock_target` gives; Windows alone still needs a file.
///
/// A skill write is read-modify-write: read `SKILL.md`, compose the new contents from what was
/// read, write it back. Nothing else serializes them across processes: `config.toml` has
/// [`crate::config::lock_config_file`] and sessions have `FileLock`, and without this two
/// `meka skill add` runs, or `meka serve` racing a CLI edit, each read the same file and the
/// loser's change vanishes with both reporting success. Memory needs none of this: its write is
/// one statement in one transaction, and SQLite serializes writers across processes.
///
/// Unique temp names in [`crate::fs::write_file_atomic`] stop the *splice*, where the published
/// file is a mixture of two documents. They cannot stop a lost update, because both writers are
/// behaving correctly at the file level and simply disagree about what was there.
pub(crate) struct StoreLock {
    _lock: crate::fs::PathLock,
}
/// Take [`StoreLock`] on `root`. Blocks until any other holder releases it.
///
/// Blocking rather than failing, for the reason [`crate::config::lock_config_file`] blocks: the
/// contended window is one small file write, and failing a `skill_write` because a `meka skill add`
/// happened to be in flight would trade a rare lost update for a common spurious error.
///
/// Must not nest. No path takes this twice, so there is no ordering to get wrong; a future caller
/// that wants to nest needs the depth counting [`crate::config::ConfigFileLock`] does.
pub(crate) fn lock_store(root: &std::path::Path) -> std::io::Result<StoreLock> {
    // 0700 straight from `mkdir(2)`, matching [`crate::fs::write_file_atomic`]. A plain
    // `create_dir_all` takes the umask, and this runs *before* that function on every write path
    // (and is the only thing that runs at all on a delete-only one, such as `skill_delete` or
    // `DELETE /v1/skills/{name}` against a store that does not exist yet), so a first-ever write
    // would leave `<config>/skills` at 0755 permanently, with the store's own entry names listable
    // to every local user.
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .mode(0o700)
            .recursive(true)
            .create(root)?;
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(root)?;

    // The root directory, for the reason `config::open_config_lock_target` gives: entries beneath
    // it are published by rename, so a lock on one stops excluding when its holder writes.
    #[cfg(unix)]
    let lock = crate::fs::lock_path(root, |path| std::fs::File::open(path))?;

    // `LockFileEx` rejects a directory handle with `ERROR_INVALID_PARAMETER` even when
    // `FILE_FLAG_BACKUP_SEMANTICS` opened it, so Windows locks a file inside the root instead.
    #[cfg(windows)]
    let lock = crate::fs::lock_path(&root.join(STORE_LOCK_FILE), |path| {
        std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
    })?;

    Ok(StoreLock { _lock: lock })
}
/// Refuse a store path that is a symlink, so a write stays inside the store it was aimed at.
///
/// [`crate::entry::validate_entry_name`] keeps a *name* from escaping the root, but it cannot see
/// what is already on disk under that name: a symlink planted at `<root>/<entry>` redirects the
/// write wherever it points, while the path meka checked still looks local. Archives preserve
/// symlinks, so unpacking a downloaded skill bundle is enough to plant one, with no code execution
/// involved.
///
/// This matters because the skill store is writable at [`crate::permission::Permission::Read`],
/// whose whole contract is that nothing outside meka's own directory changes. Following a symlink
/// out of the store breaks exactly that. Checked with `symlink_metadata`, which does not follow the
/// link.
pub(crate) fn reject_symlinked_path(path: &Path, noun: &str) -> std::result::Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            // Warned as well as returned. The error reaches the model, which will recover by
            // picking another name and say nothing more about it; but a symlink inside meka's own
            // config directory is something the person running it should hear about once, since
            // they did not put it there by using meka.
            tracing::warn!(
                "refusing to write through the symlinked {noun} path {path}",
                path = path.display()
            );
            Err(format!(
                "{} path {} is a symlink; refusing to write through it",
                noun,
                path.display()
            ))
        }
        // Absent is fine: the caller is about to create it. Any other stat error is left to the
        // write itself, which reports it with more context than a bare "could not stat".
        _ => Ok(()),
    }
}
/// RAII handle for an exclusive OS file lock in the store's lock directory. Holding this value
/// keeps the underlying descriptor open; dropping it (including when the process exits or panics)
/// closes the FD, which causes the kernel to release the `flock`/`LockFileEx` lock automatically.
/// There is no "stale lock" failure mode; even `SIGKILL` is safe.
///
/// Two things are locked this way: a session, so only one meka is ever attached to a conversation,
/// and an account's credential, so only one meka at a time is rotating its refresh token.
///
/// Closing the descriptor is what releases the lock, so holding the file is holding the claim.
pub(crate) struct FileLock {
    pub(super) _file: File,
}
/// Open `path` (creating it) and take its exclusive `flock`.
///
/// `Ok(None)` means somebody else holds it; an `Err` means the question could not be asked at all
/// (an unwritable lock directory, descriptors exhausted), which is a different answer and callers
/// treat it differently.
///
/// A free function rather than a method because two stores need it: [`crate::store::Store`] locks
/// conversations, [`crate::store::TokenStore`] locks accounts, and both live in the same
/// directory.
pub(super) fn try_lock_file(path: &std::path::Path) -> Result<Option<FileLock>> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(|error| {
            MekaError::Database(format!(
                "failed to open lock file '{}': {}",
                path.display(),
                error
            ))
        })?;

    match file.try_lock() {
        Ok(()) => Ok(Some(FileLock { _file: file })),
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(error)) => Err(MekaError::Database(format!(
            "failed to acquire lock '{}': {}",
            path.display(),
            error
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mode that could not be tightened is warned about: the file stays at the umask's mode,
    /// which the user can act on and would otherwise never hear of.
    #[cfg(unix)]
    #[test]
    fn a_failed_permission_restriction_is_warned_about() {
        #[derive(Clone)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
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

        let captured = Capture(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(captured.clone())
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let temp = tempfile::tempdir().expect("tempdir");
        restrict_permissions(&temp.path().join("absent"), 0o600);

        let logged = String::from_utf8_lossy(&crate::sync::lock(&captured.0)).to_string();
        assert!(
            logged.contains("failed to stat") && logged.contains("absent"),
            "the failure is warned about by path: {logged:?}"
        );
    }

    /// The store root is created at 0700.
    ///
    /// This is the only thing that runs on a delete-only path, and it runs before
    /// [`crate::fs::write_file_atomic`] on every write path, so a `create_dir_all` taking the
    /// umask left `<config>/skills` at 0755 permanently on a fresh install -- and a store's entry
    /// names are exactly what it should not be publishing to every local user.
    #[cfg(unix)]
    #[test]
    fn a_store_root_is_private() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("store");
        let guard = lock_store(&root).expect("lock a store that does not exist yet");

        let mode = |path: &std::path::Path| {
            std::fs::metadata(path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
                .permissions()
                .mode()
                & 0o777
        };
        assert_eq!(mode(&root), 0o700, "store root created world-listable");
        drop(guard);

        // Idempotent: locking an existing root neither fails nor loosens it.
        drop(lock_store(&root).expect("lock again"));
        assert_eq!(mode(&root), 0o700);
    }
    /// Claiming a store must not put anything inside it. Users keep skill stores in a git
    /// repository, and a lock file there is a working-tree change meka had no reason to make.
    #[cfg(unix)]
    #[test]
    fn locking_a_store_leaves_nothing_behind_in_it() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("store");
        let guard = lock_store(&root).expect("lock");

        let entries: Vec<_> = std::fs::read_dir(&root)
            .expect("read store root")
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.file_name())
            .collect();
        assert!(
            entries.is_empty(),
            "locking the store created {entries:?} inside it"
        );
        drop(guard);
    }
    /// The claim is real, not merely file-free: a second acquisition waits for the first.
    #[test]
    fn a_store_lock_excludes_a_second_holder() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("store");
        let guard = lock_store(&root).expect("lock");

        let contender = {
            std::thread::spawn(move || {
                let taken = lock_store(&root).expect("second lock");
                drop(taken);
            })
        };

        // The contender cannot finish while the first lock is held.
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(
            !contender.is_finished(),
            "a second holder took the store lock while it was held"
        );
        drop(guard);
        contender.join().expect("contender");
    }
    #[test]
    fn write_file_atomic_writes_content_and_no_tmp_left() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sub").join("config.toml");
        write_file_atomic(&path, "[x]\nk = 1\n").expect("atomic write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read back"),
            "[x]\nk = 1\n"
        );
        // The temporary file must not be left behind after a successful write.
        let tmp = dir.path().join("sub").join("config.toml.tmp");
        assert!(!tmp.exists(), "temp file should not remain: {tmp:?}");
    }
    #[test]
    fn write_file_atomic_overwrites_existing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "old contents that are LONGER than the new ones").expect("seed file");
        write_file_atomic(&path, "new\n").expect("atomic overwrite");
        assert_eq!(std::fs::read_to_string(&path).expect("read back"), "new\n");
    }
    /// A waiter must end up on whatever the path names when it wakes, not on what it queued on.
    ///
    /// Replacing the target while a waiter is blocked on it leaves that waiter holding an unlinked
    /// inode, free for the next arrival to lock the live one and enter alongside it. The identity
    /// re-check in [`lock_path`] is what closes that window.
    ///
    /// Pinned from the outside: while the woken waiter holds the lock, a third acquisition must
    /// block. It does not if the waiter is holding the dead inode. The waiter also has to have
    /// *waited*, or it reached `open` after the rename and never took the retake path at all.
    #[test]
    fn a_waiter_ends_up_holding_whatever_replaced_what_it_waited_for() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };

        fn open(path: &Path) -> std::io::Result<std::fs::File> {
            std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(path)
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("target");
        std::fs::write(&path, b"first").expect("seed");

        let first = lock_path(&path, open).expect("first");

        // Queues on the inode that is about to be replaced.
        let waiter_holds = Arc::new(AtomicBool::new(false));
        let waiter_may_release = Arc::new(AtomicBool::new(false));
        let waiter = std::thread::spawn({
            let (path, holds, may_release) = (
                path.clone(),
                waiter_holds.clone(),
                waiter_may_release.clone(),
            );
            move || {
                let queued_at = std::time::Instant::now();
                let held = lock_path(&path, open).expect("waiter");
                let waited = queued_at.elapsed();
                holds.store(true, Ordering::SeqCst);
                while !may_release.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                drop(held);
                waited
            }
        });

        // Let it block, then publish a new inode over the one it is waiting on and release.
        std::thread::sleep(std::time::Duration::from_millis(150));
        let replacement = dir.path().join("replacement");
        std::fs::write(&replacement, b"second").expect("replacement");
        std::fs::rename(&replacement, &path).expect("publish a new inode");
        drop(first);

        while !waiter_holds.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // The waiter now holds what it believes is the lock. Prove it is the live one.
        let contender =
            std::thread::spawn(move || drop(lock_path(&path, open).expect("contender")));
        std::thread::sleep(std::time::Duration::from_millis(250));
        let contender_got_in = contender.is_finished();

        waiter_may_release.store(true, Ordering::SeqCst);
        let waited = waiter.join().expect("waiter");
        contender.join().expect("contender");

        // Without this the test can pass for the wrong reason: a waiter slow enough to reach `open`
        // after the rename locks the live inode first time and never exercises the retake at all.
        assert!(
            waited >= std::time::Duration::from_millis(100),
            "the waiter took {waited:?}, so it never queued on the inode that was replaced"
        );
        assert!(
            !contender_got_in,
            "a third acquisition took the lock while the waiter held it, so the waiter is holding \
             an inode that is no longer at the path"
        );
    }
    /// Concurrent writers of one path must not splice their contents together.
    ///
    /// A fixed `<file>.tmp` is shared by every writer: both `open(O_TRUNC)` the same inode, both
    /// write from offset zero, and whoever renames last publishes a mixture of the two as a
    /// success. Reproduced before the fix at 4 MiB against 64 KiB; the sizes here are the same
    /// shape, small enough to stay quick.
    #[test]
    fn write_file_atomic_never_publishes_two_writers_spliced_together() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("contended.txt");
        let long = "A".repeat(512 * 1024);
        let short = "B".repeat(4 * 1024);

        for _ in 0..40 {
            std::thread::scope(|scope| {
                for content in [&long, &short] {
                    let path = &path;
                    scope.spawn(move || {
                        write_file_atomic(path, content).expect("write");
                    });
                }
            });
            let published = std::fs::read_to_string(&path).expect("read back");
            assert!(
                published == long || published == short,
                "published a splice: {} bytes, {} A and {} B",
                published.len(),
                published.matches('A').count(),
                published.matches('B').count(),
            );
        }
        // And no temp files are left behind by a run that succeeded.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .expect("read dir")
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "leftover temp files: {leftovers:?}");
    }
    /// An existing file's mode is preserved, but never left looser than 0600.
    ///
    /// Two failures meet here. Handing the target the temp's fresh 0600 unconditionally re-modes a
    /// file meka did not create. Carrying the existing mode across verbatim is worse: it leaves a
    /// hand-written `Authorization` header in a world-readable `config.toml` after any ordinary
    /// `meka mcp` edit, because the write follows a dotfile manager's symlink out of meka's 0700
    /// directory and the group and other bits came with it. Narrowing keeps a deliberately tighter
    /// mode and refuses a looser one.
    #[cfg(unix)]
    #[test]
    fn write_file_atomic_never_leaves_a_rewritten_file_looser_than_0600() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        for (seeded, expected) in [
            (0o644, 0o600),
            (0o400, 0o400),
            (0o600, 0o600),
            (0o660, 0o600),
        ] {
            let path = dir.path().join(format!("f{seeded:o}.toml"));
            std::fs::write(&path, "old\n").expect("seed");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(seeded))
                .expect("chmod");

            write_file_atomic(&path, "new\n").expect("rewrite");

            assert_eq!(
                std::fs::metadata(&path).expect("stat").permissions().mode() & 0o777,
                expected,
                "seeded {seeded:o}"
            );
            assert_eq!(std::fs::read_to_string(&path).expect("read"), "new\n");
        }

        // A file meka creates still gets the restrictive mode.
        let fresh = dir.path().join("fresh.toml");
        write_file_atomic(&fresh, "new\n").expect("create");
        assert_eq!(
            std::fs::metadata(&fresh)
                .expect("stat")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    /// Following a link out of meka's tree must not re-mode the directory it lands in.
    ///
    /// One `meka mcp disable` against a `config.toml` symlinked into `~/dotfiles` set that whole
    /// directory to 0700, hiding every unrelated file in it. Observed.
    #[cfg(unix)]
    #[test]
    fn write_file_atomic_does_not_tighten_a_directory_it_was_redirected_into() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let dotfiles = dir.path().join("dotfiles");
        std::fs::create_dir_all(&dotfiles).expect("dotfiles");
        std::fs::set_permissions(&dotfiles, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let target = dotfiles.join("config.toml");
        std::fs::write(&target, "old\n").expect("seed");
        let link = dir.path().join("config.toml");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        write_file_atomic(&link, "new\n").expect("write through the link");

        assert_eq!(
            std::fs::read_to_string(&target).expect("read"),
            "new\n",
            "the write must still land on the target"
        );
        assert_eq!(
            std::fs::metadata(&dotfiles)
                .expect("stat")
                .permissions()
                .mode()
                & 0o777,
            0o755,
            "a directory meka was redirected into is not one it owns"
        );
    }
    // Unix-only: the body calls `std::os::unix::fs::symlink`, so without this the file does not
    // compile on Windows at all -- taking out the `lint` and `test` jobs that CI now runs on the
    // three-OS matrix, which is exactly the platform-only break that matrix was added to catch and
    // which cannot be reproduced from a Linux workstation.
    #[cfg(unix)]
    #[test]
    fn write_file_atomic_writes_through_a_symlink_instead_of_replacing_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("tracked.toml");
        let link = dir.path().join("config.toml");
        std::fs::write(&target, "old\n").expect("seed target");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        write_file_atomic(&link, "new\n").expect("atomic write through link");

        assert!(
            std::fs::symlink_metadata(&link)
                .expect("stat link")
                .file_type()
                .is_symlink(),
            "the link itself must survive the write"
        );
        assert_eq!(
            std::fs::read_to_string(&target).expect("read target"),
            "new\n",
            "the tracked file is what should have changed"
        );
    }
    /// A chain still resolves to the real file, and a cycle must not spin.
    #[cfg(unix)]
    #[test]
    fn write_file_atomic_follows_a_symlink_chain_and_survives_a_cycle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("real.toml");
        let middle = dir.path().join("middle.toml");
        let entry = dir.path().join("entry.toml");
        std::fs::write(&target, "old\n").expect("seed");
        std::os::unix::fs::symlink(&target, &middle).expect("link 1");
        std::os::unix::fs::symlink(&middle, &entry).expect("link 2");

        write_file_atomic(&entry, "new\n").expect("write through chain");
        assert_eq!(std::fs::read_to_string(&target).expect("read"), "new\n");

        // `a -> b -> a`: resolution gives up and the call still returns rather than hanging.
        let loop_a = dir.path().join("a.toml");
        let loop_b = dir.path().join("b.toml");
        std::os::unix::fs::symlink(&loop_b, &loop_a).expect("cycle a");
        std::os::unix::fs::symlink(&loop_a, &loop_b).expect("cycle b");
        let _ = write_file_atomic(&loop_a, "whatever\n");
    }
    #[cfg(unix)]
    #[test]
    fn write_file_atomic_sets_unix_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let parent = dir.path().join("meka");
        let path = parent.join("config.toml");
        write_file_atomic(&path, "x = 1\n").expect("atomic write");

        let file_mode = std::fs::metadata(&path)
            .expect("stat file")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            file_mode, 0o600,
            "config file should be 0600, got {file_mode:o}"
        );

        let dir_mode = std::fs::metadata(&parent)
            .expect("stat dir")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            dir_mode, 0o700,
            "config dir should be 0700, got {dir_mode:o}"
        );
    }
}
