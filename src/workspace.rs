//! The per-session workspace: where relative paths resolve from, and which roots a search sweeps.
//!
//! Its own module rather than a corner of `agent.rs` because every file-touching tool needs it and
//! nothing here needs an `Agent`: in `agent.rs` it would make `tools` depend on `agent` for a path
//! join, a cycle between the two largest modules in the tree.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

/// Per-session working directory, shared by reference between the agent, every file-touching tool,
/// the REPL prompt, the `/cd` slash command, and the per-turn environment-context block.
/// `std::sync::RwLock` (rather than `tokio::sync::RwLock`) so the synchronous REPL prompt can read
/// it without entering an async context; reads/writes are microseconds (a `PathBuf` clone or
/// replace), never held across `.await`. Poisoning is recovered through [`crate::sync`]: meka never
/// panics with this lock held, so a poisoned lock is a separate bug that already fired, and serving
/// the stored path beats failing every later tool call.
#[derive(Clone)]
pub(crate) struct SharedCwd(Arc<RwLock<PathBuf>>);

impl SharedCwd {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self(Arc::new(RwLock::new(path)))
    }

    pub(crate) fn get(&self) -> PathBuf {
        crate::sync::read(&self.0).clone()
    }

    pub(crate) fn set(&self, path: PathBuf) {
        *crate::sync::write(&self.0) = path;
    }
}

/// Resolve a tool-input path against the per-session [`SharedCwd`]. Absolute paths pass through
/// unchanged; relative paths are joined to the current cwd value. Tools use this at the top of
/// their `execute` methods to decouple from process `cwd`.
pub(crate) fn resolve_against_cwd(cwd: &SharedCwd, input: impl AsRef<std::path::Path>) -> PathBuf {
    let input = input.as_ref();
    if input.is_absolute() {
        input.to_path_buf()
    } else {
        cwd.get().join(input)
    }
}

/// Workspace roots beyond [`SharedCwd`]: an ACP client's `additionalDirectories`, and each
/// `--writable-root` on the command line.
///
/// A separate handle rather than a field on `SharedCwd` because only the search tools, the
/// environment-context block and the write boundary care: widening [`resolve_against_cwd`] would
/// touch every file tool's constructor to serve them. `cwd` remains the base for relative paths,
/// per the ACP spec.
///
/// These expand discovery scope **and**, at [`crate::permission::Permission::Workspace`], the set
/// of roots a write may land under ([`writable_roots`]). A client that hands meka a folder is
/// naming part of the workspace rather than somewhere to search, and a level that could read those
/// folders but never write them would impose a boundary the client never asked for.
///
/// Empty unless one of those named a root, which is the common case for the HTTP API and for an ACP
/// client that sends none. Locked and recovered like [`SharedCwd`], for the same reasons.
#[derive(Clone, Default)]
pub(crate) struct SharedRoots(Arc<RwLock<Vec<PathBuf>>>);

impl SharedRoots {
    pub(crate) fn new(roots: Vec<PathBuf>) -> Self {
        Self(Arc::new(RwLock::new(roots)))
    }

    pub(crate) fn get(&self) -> Vec<PathBuf> {
        crate::sync::read(&self.0).clone()
    }
}

/// The ordered set of roots a **recursive** search should sweep when the caller named no explicit
/// path: `cwd` first, then each additional root, with anything already covered by another root
/// dropped.
///
/// Only correct for a walker that descends, which today means `search_contents`. A tool that
/// anchors a pattern at each root instead wants [`glob_roots`]; dropping a contained root would
/// drop the files under it.
///
/// A root is dropped when some other root *contains* it, which subsumes exact duplicates. Both
/// shapes are things a client legitimately sends: Zed may repeat `cwd` inside
/// `additionalDirectories`, and nothing stops a client naming a folder nested inside another. Left
/// in, the overlapping tree is walked twice, so every file under it is reported twice, consumes two
/// slots of the result cap, and spends the shared walk budget twice.
///
/// Containment is checked in both directions, so a root that is an *ancestor* of `cwd` wins and
/// `cwd` drops out of the search set. A descending walk from the ancestor still reaches everything
/// under `cwd`, and this does not affect `cwd`'s real job: it remains the base for relative paths
/// and the shell's working directory regardless of what this returns.
///
/// Paths are compared as given. A symlink pointing at another root, or a path containing `..`, is
/// not detected; canonicalizing to catch those would resolve symlinked roots to targets the client
/// never named, which is a worse trade than an occasional duplicate.
pub(crate) fn search_roots(cwd: &SharedCwd, roots: &SharedRoots) -> Vec<PathBuf> {
    retain_broadest(std::iter::once(cwd.get()).chain(roots.get()))
}

/// Keep only the broadest roots, in first-seen order: a path contained by one already kept is
/// dropped, and a path that *contains* ones already kept replaces them.
///
/// Shared by [`search_roots`] and [`writable_roots`] because both answer questions where a
/// contained root is genuinely redundant. A descending walk from the ancestor reaches everything
/// beneath it, and a write permitted under the ancestor is permitted under its children, so in both
/// cases keeping the narrower path would only duplicate work. [`glob_roots`] deliberately does not
/// use this; see its doc comment for the failure that caused.
fn retain_broadest(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut kept: Vec<PathBuf> = Vec::new();
    for path in paths {
        if kept.iter().any(|existing| path.starts_with(existing)) {
            continue;
        }
        // This root is broader than ones already kept, so those become redundant.
        kept.retain(|existing| !existing.starts_with(&path));
        kept.push(path);
    }
    kept
}

/// The roots a write may land under at [`crate::permission::Permission::Workspace`]: the working
/// directory plus [`SharedRoots`].
///
/// All three sources the user can name feed [`SharedRoots`] rather than traveling separately: an
/// ACP client's `additionalDirectories`, and each `--writable-root` on the command line. They mean
/// the same thing (this folder is part of my workspace) and get the same treatment, searchable and
/// writable, so there is no second list to keep in step with this one.
///
/// **This is the one definition of the boundary.** The in-process fence on `write_file` /
/// `edit_file` / `scratchpad_save_file` and every sandbox dialect (Landlock, Bubblewrap, Seatbelt,
/// the Windows restricted token) derive their allow-list from here and nowhere else, so the file
/// tools and the shell cannot disagree about where a write may land.
///
/// Roots come back **canonical**, with symlinks resolved, which is what makes the containment check
/// meaningful: the target of a write is canonicalized too, so a symlink planted inside the
/// workspace and pointing out of it (`<root>/escape -> /etc`) resolves to `/etc/...` and fails the
/// prefix test. It also states the boundary in the filesystem's own terms, which is what the kernel
/// backends match on.
///
/// A root that does not resolve is **dropped**, not passed through. A path that cannot be
/// canonicalized does not exist yet, and a boundary naming a directory that is not there should
/// permit nothing rather than permit a name that something could later be created at.
///
/// Empty is a meaningful answer and means "no write may land anywhere", which is what every caller
/// must do with it. It happens when the cwd has been deleted out from under a running session.
///
/// Synchronous because both kinds of caller need it: the async fence and the `pre_exec` sandbox
/// setup, where an `.await` is not available. The cost is a handful of `stat` calls per write.
pub(crate) fn writable_roots(cwd: &SharedCwd, roots: &SharedRoots) -> Vec<PathBuf> {
    usable_roots(std::iter::once(cwd.get()).chain(roots.get()))
}

/// The boundary [`writable_roots`] would compute from an already-taken snapshot of the same paths.
///
/// Exists so the `[Environment context]` block can name the roots that will actually hold, rather
/// than the roots the session was *asked* for. The two differ: a root that no longer resolves, one
/// that is a file, one that is a masked system directory, and one already contained by another are
/// each dropped here. Listing the request told the model it could write to paths where the very
/// next write would be refused, and hid the merge when two roots collapsed into one.
pub(crate) fn usable_roots(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    retain_broadest(
        paths
            .into_iter()
            .filter_map(|path| std::fs::canonicalize(&path).ok().map(strip_verbatim))
            .filter(|path| is_usable_root(path)),
    )
}

/// Whether a canonical path can serve as a workspace root at all.
///
/// Two refusals, each because a backend cannot express the boundary otherwise.
///
/// **Not a directory.** Landlock's `PATH_BENEATH` rule is rejected with `EINVAL` when the parent is
/// a regular file and the rule carries directory-class rights, which meka's full handled-access
/// mask does. `apply_landlock` then fails inside `pre_exec`, so every shell command in the
/// session dies with `Invalid argument`, not just a write to that root. The other three backends
/// accept a file root and quietly do something different with it.
///
/// **A system directory that Bubblewrap masks.** The bwrap backend binds each root after its tmpfs
/// masks so a root under `/tmp` survives, and later mounts win, so a root at or above a masked path
/// un-masks it. Those masks exist to put the D-Bus and systemd-user sockets out of reach:
/// `--writable-root /run/user/1000` would hand back the session bus, and a root of `/` would
/// un-mask `/proc` and `/dev`, defeating the PID namespace. None of these are workspaces, and every
/// backend is degraded by them, so they are refused rather than special-cased per backend.
fn is_usable_root(path: &Path) -> bool {
    let displayed = path.display();
    if !path.is_dir() {
        tracing::warn!("workspace root '{displayed}' is not a directory; ignoring it");
        return false;
    }
    if is_system_root(path) {
        tracing::warn!(
            "refusing workspace root '{displayed}': the sandbox masks system directories; name a \
             project directory instead"
        );
        return false;
    }
    true
}

/// Paths that must never become workspace roots. Compared against the canonical form.
///
/// A root is refused when it *is* one of these, and when it is an **ancestor** of one. Both
/// directions un-mask: bwrap binds each root after its tmpfs masks and the later mount wins, so
/// `--writable-root /var` restores the host's world-writable `/var/tmp` inside the sandbox just as
/// surely as `--writable-root /var/tmp` would. The ancestor case is the one that bites in practice,
/// because `$XDG_RUNTIME_DIR` lives under `$HOME` on WSL, on minimal window managers, and wherever
/// someone set `XDG_RUNTIME_DIR=$HOME/.xdg`; a root at `$HOME` then hands the session bus socket
/// back to a confined shell, which is exactly what the masks exist to prevent.
///
/// A root under a masked path is fine and is deliberately allowed (the bind restores only that
/// subdirectory, not the masked directory itself), except for the two socket trees below, which
/// are refused as whole subtrees because everything in them is the kind of socket being hidden.
///
/// The list below is the authority. `/tmp` and `/var/tmp` are in it: "they hold no IPC socket
/// meka's masks care about" is false on any ordinary desktop.
pub(crate) fn is_system_root(path: &Path) -> bool {
    is_masked_root(path, &private_directories())
}

/// meka's own directories: the config directory, the data directory holding the credential store,
/// and the command-output captures.
///
/// Every sandbox dialect that can hides these from a confined shell, and the write fence refuses a
/// target under them whatever roots the session holds. The shell's environment is already scrubbed
/// because a leaked secret plus the open network is a live exfiltration vector under prompt
/// injection; the same secrets sit in `meka.db` at a path any shell can guess, so leaving the disk
/// door open while guarding the environment one guarded nothing. Canonicalized where they exist,
/// so a symlinked home matches the path the kernel reports; deduplicated because the capture
/// directory usually sits under the data directory.
pub(crate) fn private_directories() -> Vec<PathBuf> {
    let candidates = [
        crate::paths::meka_config_dir(),
        crate::paths::meka_data_dir(),
        Some(crate::paths::command_output_dir()),
    ];
    let mut directories: Vec<PathBuf> = Vec::new();
    for directory in candidates.into_iter().flatten() {
        // Only a directory that exists: bubblewrap mounts its mask over a path in the bound root,
        // and there is nothing to hide where nothing is. And never one the system masks already
        // cover, which happens when captures fall back to the temp directory: taking `/tmp` as
        // private would make every working directory under it read as masked, and a `read`-level
        // session there would lose sight of its own files.
        let Ok(directory) = std::fs::canonicalize(&directory).map(strip_verbatim) else {
            continue;
        };
        if !directory.is_dir() || is_masked_root(&directory, &[]) {
            continue;
        }
        if !directories.contains(&directory) {
            directories.push(directory);
        }
    }
    directories
}

/// The refusal for reading `target` at `permission`, when it lies inside meka's own directories.
///
/// The sandbox hides these from a confined shell and the write fence refuses them at every level
/// below `unrestricted`; the in-process read tools had no such door, so `read_file` and
/// `search_contents` at `read` returned `config.toml` and byte runs out of `meka.db`, and
/// `fetch_url` at the same level could carry them out. Same rule and the same exception:
/// `unrestricted` reads them as it writes them. `target` must be canonical, or a symlink into the
/// store walks past it.
pub(crate) fn private_read_refusal(
    permission: crate::permission::Permission,
    target: &Path,
) -> Option<String> {
    private_read_refusal_with(permission, target, &private_directories())
}

/// [`private_read_refusal`] with the private directories taken as a parameter, for the tests.
fn private_read_refusal_with(
    permission: crate::permission::Permission,
    target: &Path,
    private: &[PathBuf],
) -> Option<String> {
    if permission == crate::permission::Permission::Unrestricted {
        return None;
    }
    let directory = private
        .iter()
        .find(|directory| target.starts_with(directory))?;
    Some(format!(
        "'{}' is inside meka's own directory {}, which only `unrestricted` reads.",
        target.display(),
        directory.display()
    ))
}

/// The private directories a walk at `permission` has to step around: none at `unrestricted`, so
/// the walk resolves nothing.
pub(crate) fn private_directories_hidden_at(
    permission: crate::permission::Permission,
) -> Vec<PathBuf> {
    if permission == crate::permission::Permission::Unrestricted {
        Vec::new()
    } else {
        private_directories()
    }
}

/// Whether `path`, resolved through any symlinks, lies inside one of `private`. A path that cannot
/// be resolved is not inside anything meka owns, and an empty list costs no resolution at all.
pub(crate) fn resolves_into_private(path: &Path, private: &[PathBuf]) -> bool {
    if private.is_empty() {
        return false;
    }
    std::fs::canonicalize(path)
        .map(strip_verbatim)
        .is_ok_and(|real| private.iter().any(|directory| real.starts_with(directory)))
}

/// [`is_system_root`] with meka's private directories taken as a parameter, so it can be tested
/// without the environment deciding where those are.
///
/// A private directory refuses a root **at or under** it, not an ancestor of it. The system masks
/// refuse ancestors because a bind over one un-masks it; the private masks are applied after every
/// bind, so a workspace root at `$HOME` still cannot reach `~/.config/meka` and need not be refused
/// for containing it.
fn is_masked_root(path: &Path, private: &[PathBuf]) -> bool {
    if private.iter().any(|directory| path.starts_with(directory)) {
        return true;
    }
    if path.parent().is_none() {
        // The filesystem root itself, and on Windows a bare drive prefix.
        return true;
    }
    #[cfg(unix)]
    {
        // A root **at** one of these is refused; a root **under** one is fine, and the difference
        // is the whole point.
        //
        // Bubblewrap masks each of these with a tmpfs and then binds the workspace back afterwards,
        // last-mount-wins. Binding `/tmp/work` restores exactly the workspace. Binding `/tmp`
        // restores the entire host `/tmp`, including every X11, D-Bus and tmux socket living there,
        // which is a hole straight back out of the sandbox: measured, by reaching a tmux server
        // over a socket under `/tmp` from inside a confined shell and having it create a file in
        // `$HOME`, outside every workspace root. `/tmp` and `/var/tmp` are not admitted: "they hold
        // no IPC socket meka's masks care about" is false on any ordinary desktop.
        //
        // Refusing `/tmp` costs a session started with `cd /tmp` its write boundary, which is the
        // safe direction: the alternative is a boundary that reports itself as holding while the
        // shell can reach the session bus.
        const MASKED: &[&str] = &[
            "/proc",
            "/dev",
            "/sys",
            "/run",
            "/tmp",
            "/var/tmp",
            // The same two directories as macOS canonicalizes them. `/tmp` and `/var` are symlinks
            // into `/private` there, and a root is canonicalized before it reaches here, so the
            // literals above are matched against a path that never has that spelling. Harmless on
            // Linux, where nothing resolves to either.
            "/private/tmp",
            "/private/var/tmp",
        ];
        // Equal to a masked path, or an ancestor of one. The equality test alone left the ancestor
        // case open: `--writable-root /var` was accepted and then un-masked `/var/tmp` inside the
        // sandbox, measured by writing a file there from a confined shell and finding it on the
        // host. `starts_with` is component-wise, so `/vary` is not an ancestor of `/var/tmp`.
        if MASKED
            .iter()
            .map(Path::new)
            .any(|masked| path == masked || masked.starts_with(path))
        {
            return true;
        }
        // The session's socket tree, refused as a *subtree* because everything in it is the kind of
        // socket the masks exist to hide. Checked by path as well as by variable so an unset or
        // stale `XDG_RUNTIME_DIR` does not open it.
        if path.starts_with("/run/user") || Path::new("/run/user").starts_with(path) {
            return true;
        }
        if let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR")
            && !runtime.is_empty()
            && let Ok(runtime) = std::fs::canonicalize(&runtime)
            // Both directions, and the ancestor half is the one that matters: this directory is
            // often under `$HOME`, so a root at `$HOME` (or the far more ordinary `cd ~`) would
            // put the session bus back within reach of a confined shell.
            && (path.starts_with(&runtime) || runtime.starts_with(path))
        {
            return true;
        }
    }
    false
}

/// Serializes the tests that point `XDG_RUNTIME_DIR` at a directory of their own.
///
/// Same shape and same reason as [`crate::config::CONFIG_DIR_ENV_LOCK`]: the variable is
/// process-global, `cargo test` runs these in parallel, and [`is_system_root`] reads it.
#[cfg(all(test, unix))]
pub(crate) static RUNTIME_DIR_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Drop Windows' `\\?\` verbatim prefix, which only `canonicalize` speaks.
///
/// Invisible on Unix and load-bearing on Windows. `canonicalize` returns `\\?\C:\ws` while a cwd
/// from `current_dir` and a path the model wrote are both spelled `C:\ws`, and the former never
/// prefix-matches the latter, so without this the fence would refuse every write inside the
/// workspace while still refusing the ones outside it.
///
/// Verbatim UNC paths are deliberately left alone: a network share cannot be a workspace root
/// anyway, because meka has to own the directory to grant on it.
///
/// `pub(crate)` because the tools canonicalize too, for the file they are about to open. This is
/// the tree's one answer to what shape a path is in, so everything that produces one goes through
/// it; [`accept_cwd`] is where a working directory does.
pub(crate) fn strip_verbatim(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        use std::path::{Component, Prefix};
        let mut components = path.components();
        if let Some(Component::Prefix(prefix)) = components.next()
            && let Prefix::VerbatimDisk(letter) = prefix.kind()
        {
            // Rebuilt from the remaining components rather than by trimming the rendered string. A
            // path is a sequence of `OsStr`s, not text: `to_string_lossy` would replace any
            // unpaired surrogate in a Windows filename with U+FFFD, and comparing a boundary
            // against a path that has silently changed is exactly the class of bug this function
            // exists to fix. `VerbatimDisk` carries an ASCII drive letter by construction, so the
            // designator is the one part that can safely be built as text.
            let mut rebuilt = PathBuf::from(format!("{}:", letter as char));
            rebuilt.extend(components);
            return rebuilt;
        }
    }
    path
}

/// Admit a working directory named from outside the process: an HTTP `cwd`, an ACP `cwd`, a `/cd`
/// argument. Returns the canonical spelling, which is what the session row records and what the
/// write fence, the prompt and every relative tool path are then derived from.
///
/// One definition because the working directory is the writable boundary at `workspace` and the
/// directory a scheduled gate is re-checked in, and each door once judged it with its own subset of
/// these checks and recorded its own spelling.
pub(crate) fn accept_cwd(path: &Path) -> crate::error::Result<PathBuf> {
    if !path.is_absolute() {
        return Err(refuse_cwd(path, "is not an absolute path"));
    }
    // The kernel reads a path up to its first NUL, so the directory the OS would open is not the
    // one the caller spelled.
    if path.as_os_str().to_string_lossy().contains('\0') {
        return Err(refuse_cwd(path, "contains a NUL byte"));
    }
    let canonical = match std::fs::canonicalize(path) {
        Ok(canonical) => strip_verbatim(canonical),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(refuse_cwd(path, "does not exist"));
        }
        Err(error) => return Err(refuse_cwd(path, &format!("cannot be resolved: {error}"))),
    };
    if !canonical.is_dir() {
        return Err(refuse_cwd(path, "is not a directory"));
    }
    Ok(canonical)
}

fn refuse_cwd(path: &Path, reason: &str) -> crate::error::MekaError {
    crate::error::MekaError::Usage(format!("working directory '{}' {reason}", path.display()))
}

/// The spelling to match a recorded working directory against when a listing is filtered by one:
/// the canonical spelling while the directory resolves, since that is what every door recorded, and
/// the given one otherwise, since a session whose directory has since been deleted still has a row
/// to find.
pub(crate) fn cwd_filter(path: &Path) -> PathBuf {
    accept_cwd(path).unwrap_or_else(|_| path.to_path_buf())
}

/// A sub-agent's write boundary, as `agent_spawn`'s `writable_roots` named it and
/// [`accept_writable_roots`] accepted it: the worker's cells are built from these two fields and
/// from nothing of the parent's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundedWorkspace {
    /// The first entry, canonical: the worker's working directory, where its relative paths
    /// resolve and the first root its writes may land under.
    pub(crate) cwd: PathBuf,
    /// Every later entry, canonical, in the order given.
    pub(crate) additional_roots: Vec<PathBuf>,
}

impl BoundedWorkspace {
    /// The whole boundary, working directory first: the list the spawn terms record, so that a
    /// follow-up hands it back through [`accept_writable_roots`] unchanged.
    pub(crate) fn roots(&self) -> Vec<PathBuf> {
        std::iter::once(self.cwd.clone())
            .chain(self.additional_roots.iter().cloned())
            .collect()
    }
}

/// Admit the directories a parent agent names as a sub-agent's write boundary.
///
/// One definition for both doors a bounded worker comes through, `agent_spawn` and
/// `agent_followup`, so the boundary a follow-up rebuilds is judged by the rule the spawn was.
/// Every refusal happens here, before either door has written anything.
///
/// The parent must hold a writing level: below `workspace` it has no write reach to hand down, and
/// the list would grant what the parent itself lacks. Each entry is resolved against the parent's
/// working directory when relative and then admitted as a working directory is ([`accept_cwd`]),
/// since the first one becomes exactly that, and refused when it is a directory no root may name
/// ([`is_system_root`]), since a boundary the sandbox masks holds nothing. At `workspace` every
/// entry must also lie inside the parent's own boundary, [`writable_roots`], compared canonical to
/// canonical and component-wise, so a sub-agent's reach is never wider than its parent's; at
/// `unrestricted` the parent may write anywhere, and so may name anywhere.
pub(crate) fn accept_writable_roots(
    parent_level: crate::permission::Permission,
    parent_cwd: &SharedCwd,
    parent_roots: &SharedRoots,
    requested: &[PathBuf],
) -> crate::error::Result<BoundedWorkspace> {
    use crate::{error::MekaError, permission::Permission};

    if !matches!(
        parent_level,
        Permission::Workspace | Permission::Unrestricted
    ) {
        return Err(MekaError::Usage(format!(
            "writable_roots needs a parent at `workspace` or `unrestricted`; this one is at \
             `{parent_level}`"
        )));
    }
    // Computed once, and only at the level where it constrains: at `unrestricted` the parent's
    // boundary is the whole filesystem and there is nothing to compare against.
    let parent_reach =
        (parent_level == Permission::Workspace).then(|| writable_roots(parent_cwd, parent_roots));

    let mut accepted = Vec::with_capacity(requested.len());
    for entry in requested {
        let canonical = accept_cwd(&resolve_against_cwd(parent_cwd, entry))
            .map_err(|error| MekaError::Usage(format!("writable_roots: {error}")))?;
        if is_system_root(&canonical) {
            return Err(MekaError::Usage(format!(
                "writable_roots: '{}' is a system directory the sandbox masks, so it cannot be a \
                 workspace root",
                canonical.display()
            )));
        }
        if let Some(reach) = &parent_reach
            && !is_within_roots(&canonical, reach)
        {
            let where_to = if reach.is_empty() {
                "no root of this session's workspace currently resolves".to_string()
            } else {
                format!(
                    "this session's writes land under {}",
                    reach
                        .iter()
                        .map(|root| root.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            return Err(MekaError::Usage(format!(
                "writable_roots: '{}' is outside this session's workspace: {where_to}, and a \
                 sub-agent's reach cannot exceed its parent's",
                canonical.display()
            )));
        }
        accepted.push(canonical);
    }
    // The one place an empty list is refused: a bounded worker with no working directory cannot
    // be built, and a list that bounds nothing must not fall through to the parent's workspace.
    let mut accepted = accepted.into_iter();
    let cwd = accepted.next().ok_or_else(|| {
        MekaError::Usage(
            "writable_roots names no directory; leave it out to share the parent's workspace"
                .to_string(),
        )
    })?;
    Ok(BoundedWorkspace {
        cwd,
        additional_roots: accepted.collect(),
    })
}

/// Whether `path` lies within one of `roots`.
///
/// Split out so the fence and its tests agree on what containment means, and so a root that equals
/// the path is a hit: writing *to* a workspace root's own path is a write inside it.
///
/// Both sides are put in the same normal form here rather than at each call site. Callers arrive
/// from two directions, `canonicalize` (verbatim on Windows) and lexical normalization of a path
/// that does not exist yet (never verbatim), and a comparison that assumed either one would be
/// wrong half the time.
pub(crate) fn is_within_roots(path: &std::path::Path, roots: &[PathBuf]) -> bool {
    let path = strip_verbatim(path.to_path_buf());
    roots
        .iter()
        .any(|root| path.starts_with(strip_verbatim(root.clone())))
}

/// Normalize `.` and `..` textually, without consulting the filesystem.
///
/// The filesystem cannot help for the case this exists for: a `write_file` naming a path that does
/// not exist yet, which is most of them. `canonicalize` fails on a missing path, so the only way to
/// judge `<root>/../../etc/passwd` *before* creating anything is to resolve the components as
/// text. It is not a substitute for canonicalization, which still runs afterwards to catch the
/// symlinked ancestor this pass cannot see.
fn normalize_lexically(path: &std::path::Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Per-path write locks, so two tool calls that mutate the same file cannot interleave.
///
/// One per host rather than one per registry, because the hazard belongs to the file, not to a
/// registry: a sub-agent and its parent hold different `ToolRegistry` instances but write the
/// same disk, and a per-registry lock would let exactly the pair that shares a workspace race.
/// The session materials carry the host's handle to every registry built for it. Keyed on the
/// canonicalized path so two spellings of one file take the same lock.
///
/// The map is a `std::sync::Mutex` holding only `Arc` clones (no `.await` happens inside it, so
/// it never blocks the runtime), while the per-path lock is a `tokio::sync::Mutex`, since it is
/// held across the read/modify/write awaits.
#[derive(Clone, Default)]
pub(crate) struct WriteLocks {
    by_path: Arc<std::sync::Mutex<std::collections::HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>>,
}

impl WriteLocks {
    /// Acquire the write lock for `canonical`, creating it on first use.
    ///
    /// Entries whose only remaining owner is the map are dropped on the way past, which keeps it
    /// bounded by the number of files being written *concurrently* rather than by every file the
    /// session has ever touched.
    pub(crate) async fn lock_path(&self, canonical: &Path) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut by_path = crate::sync::lock(&self.by_path);
            by_path.retain(|_, held| Arc::strong_count(held) > 1);
            Arc::clone(
                by_path
                    .entry(canonical.to_path_buf())
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
            )
        };
        lock.lock_owned().await
    }
}

/// The write boundary as a tool sees it: whether one applies right now, and what it admits.
///
/// Carries the *live* permission handle rather than a snapshot, so a `/permission` change mid-turn
/// reaches the next write rather than the next session. Cheap to clone; every field is shared.
#[derive(Clone)]
pub(crate) struct WriteScope {
    permission: crate::permission::SharedPermission,
    roots: SharedRoots,
    locks: WriteLocks,
    /// Set only by [`WriteScope::deny_all`]. A separate flag rather than an empty root list,
    /// because an empty list does not mean "nothing": [`writable_roots`] always folds in the cwd,
    /// so the closed fallback was in fact a cwd-wide grant. There has to be a state that means
    /// *no* root, and the root list cannot express it.
    denied: bool,
}

impl WriteScope {
    pub(crate) fn new(
        permission: crate::permission::SharedPermission,
        roots: SharedRoots,
        locks: WriteLocks,
    ) -> Self {
        Self {
            permission,
            roots,
            denied: false,
            locks,
        }
    }

    /// The roots a write must land under right now, or `None` when this level imposes no boundary.
    ///
    /// Only `unrestricted` disclaims a boundary, by definition; an approved call runs at its own
    /// level and reaches no further. Every other level, `none` and `read` included, is confined
    /// here: a level that should never reach a write door must fail closed if one is ever wired to
    /// it. The write lock for `canonical`, from the map every scope built for this host shares.
    pub(crate) async fn lock_path(&self, canonical: &Path) -> tokio::sync::OwnedMutexGuard<()> {
        self.locks.lock_path(canonical).await
    }

    pub(crate) fn confined_to(&self, cwd: &SharedCwd) -> Option<Vec<PathBuf>> {
        if self.denied {
            return Some(Vec::new());
        }
        match self.permission.get() {
            // Only the level that disclaims a boundary is exempt. Written as an allow-list rather
            // than `Workspace => Some(..), _ => None`, because that catch-all fails open: `none`
            // and `read` normally never reach a write door, but `[tools.tool_permissions]`
            // overrides a tool's required level with no floor, so `write_file = "read"` dispatches
            // the tool at `read` and the fence must still confine it.
            //
            // At `none` and `read` this yields the workspace roots rather than nothing, which is a
            // deliberate difference from `Confinement::resolve`, whose catch-all is `ReadOnly` and
            // grants no write at all. The two are *not* the same decision:
            //
            // - `Confinement::resolve` answers "what may a command meka did not write do", and the
            //   honest answer at `read` is nothing.
            // - This answers "where may a tool the operator deliberately lowered write", and a
            //   `write_file = "read"` override is a statement that the tool should be usable at
            //   that level. Refusing it outright would make the override a no-op with no
            //   diagnostic; confining it to the workspace roots is the narrowest reading that still
            //   honors what was configured.
            //
            // So the override can only ever narrow *reach*: it never escapes the roots, and it
            // cannot touch `execute_command`, which stays closed at `read` regardless.
            crate::permission::Permission::Unrestricted => None,
            _ => Some(writable_roots(cwd, &self.roots)),
        }
    }

    /// Judge one write target, returning the refusal sentence when it falls outside the boundary.
    ///
    /// `target` should be canonical where the caller can manage it and lexically normalized where
    /// it cannot (a path being created). Both are checked the same way; the difference is only in
    /// how much the caller has been able to resolve.
    pub(crate) fn admit(&self, cwd: &SharedCwd, target: &std::path::Path) -> Result<(), String> {
        self.admit_with_private(cwd, target, &private_directories())
    }

    /// [`Self::admit`] with meka's private directories taken as a parameter, so the refusal can be
    /// tested without the environment deciding where those are.
    fn admit_with_private(
        &self,
        cwd: &SharedCwd,
        target: &std::path::Path,
        private: &[PathBuf],
    ) -> Result<(), String> {
        let Some(roots) = self.confined_to(cwd) else {
            return Ok(());
        };
        let candidate = normalize_lexically(target);
        // meka's own store is never a workspace, whatever roots the session holds: a root at
        // `$HOME` contains `~/.config/meka`, and a confined agent that can rewrite `config.toml` or
        // `meka.db` has left every boundary that file describes. The sandbox masks the same
        // directories from the shell; this is the same rule at the in-process door.
        if let Some(directory) = private
            .iter()
            .find(|directory| candidate.starts_with(directory))
        {
            return Err(format!(
                "'{}' is inside meka's own directory {}, which no workspace root reaches. Only \
                 `unrestricted` writes there.",
                target.display(),
                directory.display()
            ));
        }
        if is_within_roots(&candidate, &roots) {
            return Ok(());
        }
        // Names the roots rather than just refusing: the model cannot see the boundary from the
        // tool schema, and without them its next attempt is a guess. Empty is a real state (the
        // working directory was deleted), and saying so beats reporting an empty list.
        let where_to = if roots.is_empty() {
            "no workspace root currently resolves, so no write can land anywhere".to_string()
        } else {
            format!(
                "writes must land under {}",
                roots
                    .iter()
                    .map(|root| root.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        Err(format!(
            "'{}' is outside the workspace: at `workspace` permission {}. Pick a path inside, or \
             ask the user for `unrestricted` if it genuinely belongs elsewhere.",
            target.display(),
            where_to
        ))
    }

    /// A scope that admits nothing, for a registry that never registered the core tools and so
    /// holds no permission handle to judge against.
    ///
    /// Unreachable in production: both `build_default` and `build_for_subagent` call
    /// `register_core_tools` before the session-scoped pass. It exists so the fallback is
    /// *closed*. A registry that cannot tell whether a boundary applies must refuse rather than
    /// assume there is none, which is the direction a missing scope would otherwise fail in.
    pub(crate) fn deny_all() -> Self {
        Self {
            permission: crate::permission::SharedPermission::new(
                crate::permission::Permission::None,
                crate::permission::EnabledPermissions::DEFAULT,
            ),
            roots: SharedRoots::default(),
            locks: WriteLocks::default(),
            denied: true,
        }
    }

    /// A scope that confines nothing, for tools under test that are not exercising the boundary.
    #[cfg(test)]
    pub(crate) fn unconfined() -> Self {
        Self::new(
            crate::permission::SharedPermission::new(
                crate::permission::Permission::Unrestricted,
                crate::permission::EnabledPermissions::ALL,
            ),
            roots_for_test(),
            WriteLocks::default(),
        )
    }

    /// A scope confined to `roots`, for tests that *are* exercising the boundary.
    #[cfg(test)]
    pub(crate) fn confined(roots: Vec<PathBuf>) -> Self {
        Self::new(
            crate::permission::SharedPermission::new(
                crate::permission::Permission::Workspace,
                crate::permission::EnabledPermissions::ALL,
            ),
            SharedRoots::new(roots),
            WriteLocks::default(),
        )
    }
}

/// The ordered set of roots to anchor a glob at when the caller named no explicit path: `cwd`
/// first, then each additional root, with only *exact* repeats dropped.
///
/// The counterpart to [`search_roots`] for a tool that builds one rooted pattern per root rather
/// than descending from it. Containment must not drop anything here: `find_files` turns each root
/// into `<root>/<pattern>`, and a glob's `*` does not cross `/`, so a workspace of `/work` plus
/// `cwd = /work/main` would answer `*.md` from `/work/*.md` alone and miss `/work/main/README.md`
/// entirely. That is the exact "the agent says a file you can see doesn't exist" failure multi-root
/// support was added to prevent, so nested roots are all kept and the caller deduplicates the
/// matches instead.
pub(crate) fn glob_roots(cwd: &SharedCwd, roots: &SharedRoots) -> Vec<PathBuf> {
    let mut kept: Vec<PathBuf> = Vec::new();
    for path in std::iter::once(cwd.get()).chain(roots.get()) {
        if !kept.contains(&path) {
            kept.push(path);
        }
    }
    kept
}

/// Canonicalize a path the way meka does, for tests that compare against meka's own output.
///
/// `std::fs::canonicalize` hands back a `\\?\`-prefixed path on Windows, and every production
/// caller runs the result through [`strip_verbatim`]. A test that skips that step asserts against
/// the one spelling meka never produces, and passes everywhere `strip_verbatim` is the identity, so
/// the failure is Windows-only. A named helper rather than a `.map(strip_verbatim)` a test can
/// forget.
#[cfg(test)]
pub(crate) fn canonical_for_test(path: impl AsRef<std::path::Path>) -> PathBuf {
    std::fs::canonicalize(path.as_ref())
        .map(strip_verbatim)
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", path.as_ref().display()))
}

/// Construct a fresh [`SharedCwd`] pointing at the process cwd, for use in tests that need to
/// instantiate a tool but don't exercise the per-session cwd resolution path. Tests using absolute
/// paths or `tempdir()` are unaffected by the value here.
#[cfg(test)]
pub(crate) fn cwd_for_test() -> SharedCwd {
    SharedCwd::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Construct an empty [`SharedRoots`], for tools under test that don't exercise multi-root search.
#[cfg(test)]
pub(crate) fn roots_for_test() -> SharedRoots {
    SharedRoots::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A holder that panics poisons the lock; the next reader still gets the value, because the
    /// alternative is every later tool call failing for a bug that already reported itself.
    #[test]
    fn a_poisoned_cwd_or_roots_handle_still_serves_its_value() {
        let cwd = SharedCwd::new(PathBuf::from("/before"));
        let roots = SharedRoots::new(vec![PathBuf::from("/root")]);
        let poisoned = std::panic::catch_unwind({
            let cwd = cwd.clone();
            let roots = roots.clone();
            move || {
                let _cwd_guard = cwd.0.write().expect("not poisoned yet");
                let _roots_guard = roots.0.write().expect("not poisoned yet");
                panic!("poison both locks");
            }
        });
        assert!(poisoned.is_err(), "the holder must have panicked");

        assert_eq!(cwd.get(), PathBuf::from("/before"));
        assert_eq!(roots.get(), vec![PathBuf::from("/root")]);
        cwd.set(PathBuf::from("/after"));
        assert_eq!(cwd.get(), PathBuf::from("/after"));
    }

    /// One acceptor for every door a working directory comes through from outside: an existing
    /// directory, in the one spelling the row records, or a refusal that says which rule it broke.
    #[test]
    fn a_working_directory_is_accepted_only_as_an_existing_directory_spelled_canonically() {
        let temp = tempfile::tempdir().expect("tempdir");
        let directory = temp.path().join("project");
        std::fs::create_dir(&directory).expect("mkdir");
        let file = temp.path().join("file");
        std::fs::write(&file, b"x").expect("write");

        assert_eq!(
            accept_cwd(&directory).expect("a directory is accepted"),
            canonical_for_test(&directory)
        );
        assert_eq!(
            accept_cwd(&directory.join("..").join("project").join("."))
                .expect("a directory reached through `..` is the same directory"),
            canonical_for_test(&directory),
            "the row must never record a spelling the fence would compare against another"
        );

        let refused = |path: &Path| accept_cwd(path).expect_err("refused").to_string();
        assert!(refused(Path::new("relative/path")).contains("is not an absolute path"));
        assert!(refused(&temp.path().join("missing")).contains("does not exist"));
        assert!(refused(&file).contains("is not a directory"));
        assert!(refused(&temp.path().join("with\0nul")).contains("contains a NUL byte"));
    }

    /// A directory named through a symlink is recorded as the directory itself, and a listing
    /// filter spelled either way finds the rows; one for a directory that is gone is matched as
    /// given, so the sessions it recorded stay findable.
    #[cfg(unix)]
    #[test]
    fn a_working_directory_named_through_a_symlink_is_recorded_as_its_target() {
        let temp = tempfile::tempdir().expect("tempdir");
        let target = temp.path().join("real");
        std::fs::create_dir(&target).expect("mkdir");
        let link = temp.path().join("link");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        assert_eq!(
            accept_cwd(&link).expect("a symlink to a directory is accepted"),
            canonical_for_test(&target)
        );
        assert_eq!(cwd_filter(&link), canonical_for_test(&target));
        let gone = temp.path().join("gone");
        assert_eq!(cwd_filter(&gone), gone);
    }

    /// One acceptor for both doors a bounded sub-agent comes through. A parent below `workspace`
    /// has no write reach to hand down, and an empty list would bound nothing, so both are refused
    /// before any entry is resolved.
    #[test]
    fn a_sub_agent_boundary_needs_a_writing_parent_and_at_least_one_directory() {
        use crate::permission::Permission;
        let temp = tempfile::tempdir().expect("tempdir");
        let work = canonical_for_test(temp.path()).join("work");
        std::fs::create_dir_all(work.join("sub")).expect("dirs");
        let parent_cwd = SharedCwd::new(work.clone());
        let parent_roots = SharedRoots::default();
        let requested = vec![work.join("sub")];

        for level in [Permission::None, Permission::Read] {
            let refusal = accept_writable_roots(level, &parent_cwd, &parent_roots, &requested)
                .expect_err("a parent that cannot write cannot delegate writing")
                .to_string();
            assert!(
                refusal.contains("needs a parent at `workspace` or `unrestricted`"),
                "{refusal}"
            );
        }
        let refusal = accept_writable_roots(Permission::Workspace, &parent_cwd, &parent_roots, &[])
            .expect_err("an empty list bounds nothing")
            .to_string();
        assert!(refusal.contains("names no directory"), "{refusal}");
    }

    /// At `workspace` every entry must lie inside the parent's own boundary, judged canonical
    /// against canonical and by path component, so a sibling that merely shares a prefix is
    /// outside. At `unrestricted` the parent may write anywhere, and so may name anywhere.
    #[test]
    fn a_sub_agent_boundary_lies_within_the_parents_reach_unless_the_parent_is_unrestricted() {
        use crate::permission::Permission;
        let temp = tempfile::tempdir().expect("tempdir");
        let base = canonical_for_test(temp.path());
        let work = base.join("work");
        let docs = base.join("shared").join("docs");
        std::fs::create_dir_all(work.join("sub")).expect("dirs");
        std::fs::create_dir_all(base.join("work2")).expect("sibling");
        std::fs::create_dir_all(&docs).expect("shared");
        let parent_cwd = SharedCwd::new(work.clone());
        let parent_roots = SharedRoots::new(vec![base.join("shared")]);

        let accepted = accept_writable_roots(Permission::Workspace, &parent_cwd, &parent_roots, &[
            work.join("sub"),
            docs.clone(),
        ])
        .expect("a directory under the cwd and one under a named root are both inside");
        assert_eq!(accepted, BoundedWorkspace {
            cwd: work.join("sub"),
            additional_roots: vec![docs.clone()],
        });
        assert_eq!(accepted.roots(), vec![work.join("sub"), docs]);

        let refusal = accept_writable_roots(Permission::Workspace, &parent_cwd, &parent_roots, &[
            work.join("sub"),
            base.join("work2"),
        ])
        .expect_err("`work2` shares a prefix with `work` and is outside it")
        .to_string();
        assert!(
            refusal.contains(&base.join("work2").display().to_string())
                && refusal.contains("outside this session's workspace"),
            "the refusal must name the entry and the rule: {refusal}"
        );

        let accepted =
            accept_writable_roots(Permission::Unrestricted, &parent_cwd, &parent_roots, &[
                base.join("work2"),
            ])
            .expect("an unrestricted parent may name a directory outside its own");
        assert_eq!(accepted.cwd, base.join("work2"));
    }

    /// Entries are admitted as a working directory is: a relative one resolves against the
    /// parent's directory, and a file or a missing directory is refused by name. A directory the
    /// sandbox masks is refused at any level, since a boundary with no usable root holds nothing.
    #[test]
    fn a_sub_agent_boundary_is_resolved_and_admitted_like_a_working_directory() {
        use crate::permission::Permission;
        let temp = tempfile::tempdir().expect("tempdir");
        let work = canonical_for_test(temp.path()).join("work");
        std::fs::create_dir_all(work.join("sub")).expect("dirs");
        let file = work.join("notes.md");
        std::fs::write(&file, b"x").expect("write");
        let parent_cwd = SharedCwd::new(work.clone());
        let parent_roots = SharedRoots::default();

        let accepted = accept_writable_roots(Permission::Workspace, &parent_cwd, &parent_roots, &[
            PathBuf::from("sub"),
        ])
        .expect("a relative entry resolves against the parent's directory");
        assert_eq!(accepted.cwd, work.join("sub"));

        let refused = |requested: &[PathBuf]| {
            accept_writable_roots(Permission::Workspace, &parent_cwd, &parent_roots, requested)
                .expect_err("refused")
                .to_string()
        };
        assert!(refused(&[work.join("missing")]).contains("does not exist"));
        assert!(refused(&[file]).contains("is not a directory"));

        #[cfg(unix)]
        {
            let refusal =
                accept_writable_roots(Permission::Unrestricted, &parent_cwd, &parent_roots, &[
                    PathBuf::from("/tmp"),
                ])
                .expect_err("a masked system directory cannot be a root at any level")
                .to_string();
            assert!(refusal.contains("system directory"), "{refusal}");
        }
    }

    /// A private directory refuses a root at or under it and nothing above it: the masks are
    /// applied after every bind, so a root that merely contains the store cannot reach it.
    #[cfg(unix)]
    #[test]
    fn a_private_directory_refuses_roots_at_or_under_it_but_not_above() {
        let private = vec![PathBuf::from("/home/someone/.config/meka")];
        assert!(is_masked_root(
            Path::new("/home/someone/.config/meka"),
            &private
        ));
        assert!(is_masked_root(
            Path::new("/home/someone/.config/meka/skills"),
            &private
        ));
        assert!(!is_masked_root(Path::new("/home/someone"), &private));
        assert!(!is_masked_root(
            Path::new("/home/someone/.config"),
            &private
        ));
        assert!(!is_masked_root(
            Path::new("/home/someone/project"),
            &private
        ));
    }

    /// The read door is the write fence's twin: below `unrestricted` nothing under meka's own
    /// directories is readable, whatever roots the session holds. Without it a session at `read`,
    /// the level whose contract is "cannot change the machine", read the credential store.
    #[test]
    fn a_private_directory_is_not_readable_below_unrestricted() {
        use crate::permission::Permission;
        let private = vec![PathBuf::from("/home/someone/.local/share/meka")];
        let store = Path::new("/home/someone/.local/share/meka/meka.db");
        for permission in [Permission::Read, Permission::Workspace, Permission::None] {
            let refusal = private_read_refusal_with(permission, store, &private)
                .unwrap_or_else(|| panic!("{permission:?} must refuse the store"));
            assert!(refusal.contains("meka's own directory"), "{refusal}");
        }
        assert!(
            private_read_refusal_with(Permission::Unrestricted, store, &private).is_none(),
            "unrestricted reads it, as it writes it"
        );
        assert!(
            private_read_refusal_with(
                Permission::Read,
                Path::new("/home/someone/.local/share/other"),
                &private
            )
            .is_none()
        );
    }

    /// The in-process write door refuses meka's own directories even when a workspace root
    /// contains them, the same rule the sandbox masks enforce for the shell.
    #[test]
    fn the_write_fence_refuses_meka_s_own_directories_under_a_containing_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = canonical_for_test(temp.path());
        let store = root.join("meka-store");
        std::fs::create_dir(&store).expect("store");
        let cwd = SharedCwd::new(root.clone());
        let scope = WriteScope::new(
            crate::permission::SharedPermission::new(
                crate::permission::Permission::Workspace,
                crate::permission::EnabledPermissions::DEFAULT,
            ),
            SharedRoots::default(),
            WriteLocks::default(),
        );
        let private = vec![store.clone()];
        assert!(
            scope
                .admit_with_private(&cwd, &root.join("notes.txt"), &private)
                .is_ok(),
            "a path beside the store is still inside the workspace"
        );
        let refusal = scope
            .admit_with_private(&cwd, &store.join("config.toml"), &private)
            .expect_err("a path inside the store is refused although the root contains it");
        assert!(refusal.contains("meka's own directory"), "{refusal}");
        assert!(
            scope
                .admit_with_private(&cwd, &store.join("config.toml"), &[])
                .is_ok(),
            "the refusal is the private list's doing, not the root's"
        );
    }

    fn shared(path: &std::path::Path) -> SharedCwd {
        SharedCwd::new(path.to_path_buf())
    }

    /// The three sources compose into one canonical, containment-deduplicated set.
    #[test]
    fn writable_roots_combines_the_cwd_with_every_named_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        for name in ["work", "shared", "extra"] {
            std::fs::create_dir(base.join(name)).expect("create dir");
        }

        let roots = writable_roots(
            &shared(&base.join("work")),
            &SharedRoots::new(vec![base.join("shared"), base.join("extra")]),
        );

        assert_eq!(roots, vec![
            base.join("work"),
            base.join("shared"),
            base.join("extra"),
        ]);
    }

    /// Containment is judged on the *resolved* path, so a link that leads out of the workspace is
    /// not inside it once resolved.
    ///
    /// Named for what it actually checks. `is_within_roots` is a component-wise prefix test and
    /// deliberately does no symlink resolution of its own (`normalize_lexically` is documented as
    /// not touching them), so the resolution this depends on happens in the caller
    /// (`resolve_write_target`, via `resolve_existing_prefix`). The fixture stands in for that
    /// caller by canonicalizing before it asks.
    ///
    /// This layer does not resolve the link: feeding it the *spelled* path returns `true`, and the
    /// reason that is safe is that no caller ever does.
    #[test]
    #[cfg(unix)]
    fn a_resolved_path_leading_out_of_the_workspace_is_not_within_it() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        let outside = base.join("outside");
        std::fs::create_dir(&work).expect("work");
        std::fs::create_dir(&outside).expect("outside");
        std::os::unix::fs::symlink(&outside, work.join("escape")).expect("symlink");

        let roots = writable_roots(&shared(&work), &roots_for_test());
        let spelled = work.join("escape").join("passwd");
        let escaped = crate::workspace::canonical_for_test(work.join("escape")).join("passwd");

        assert!(is_within_roots(&work.join("src"), &roots));
        // The spelled form *is* admitted, and stating that is the point: it is what makes
        // resolving-before-asking load-bearing rather than decorative. An implementation that
        // compared as written would take this path and land the bytes in `outside`.
        assert!(
            is_within_roots(&spelled, &roots),
            "this layer does not resolve links, so the spelled path passes -- which is exactly why \
             the caller must resolve first"
        );
        assert!(
            !is_within_roots(&escaped, &roots),
            "a link resolving outside the root must not be inside it"
        );
    }

    /// The boundary tracks the cwd rather than freezing at the value it was built with.
    ///
    /// Documented behavior, not an accident of the implementation: `/cd` is meant to move the
    /// workspace, and a boundary that stayed put would refuse writes to the directory the user is
    /// plainly now working in. Named roots are unaffected by the move, which is what keeps a
    /// client-supplied folder the client's rather than the cwd's.
    #[test]
    fn writable_roots_follow_the_cwd_when_it_moves() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        for name in ["before", "after", "named"] {
            std::fs::create_dir(base.join(name)).expect("create dir");
        }
        let cwd = shared(&base.join("before"));
        let named = SharedRoots::new(vec![base.join("named")]);

        assert_eq!(writable_roots(&cwd, &named), vec![
            base.join("before"),
            base.join("named"),
        ]);

        cwd.set(base.join("after"));

        assert_eq!(
            writable_roots(&cwd, &named),
            vec![base.join("after"), base.join("named")],
            "the boundary must move with the cwd and leave named roots alone"
        );
    }

    /// A canonical root and an as-spelled target inside it agree, despite Windows' verbatim prefix.
    ///
    /// Invisible to every Linux test: `canonicalize` hands back `\\?\C:\ws`, the fence checks a
    /// target spelled `C:\ws\f.txt` against it, and `starts_with` says no. Writes outside would
    /// still be refused, so the boundary would look correct while refusing everything the level
    /// exists to allow.
    #[test]
    #[cfg(windows)]
    fn a_verbatim_root_admits_an_as_spelled_path_inside_it() {
        use std::path::{Component, Prefix};

        let temp = tempfile::tempdir().expect("tempdir");
        // Deliberately *not* `canonical_for_test`: this test is about the verbatim prefix, so it
        // needs the raw spelling the helper exists to remove.
        let canonical = std::fs::canonicalize(temp.path()).expect("canonicalize");
        assert!(
            matches!(
                canonical.components().next(),
                Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::VerbatimDisk(_))
            ),
            "precondition: Windows canonicalize returns a verbatim path, got {}",
            canonical.display()
        );
        // The as-spelled form is derived from `canonical`, never from `temp.path()`. The two differ
        // by more than the prefix wherever `TEMP` resolves through an 8.3 short name, which is
        // exactly what the GitHub Windows runner does: `canonicalize` expands `RUNNER~1` to
        // `runneradmin`, so comparing against the raw handle failed on a difference this test is
        // not about.
        let as_spelled = strip_verbatim(canonical.clone());

        // The rebuild must preserve every component, not merely drop the prefix.
        assert_eq!(
            strip_verbatim(canonical.join("a").join("b")),
            as_spelled.join("a").join("b")
        );

        let roots = writable_roots(&shared(temp.path()), &roots_for_test());
        assert!(
            is_within_roots(&as_spelled.join("f.txt"), &roots),
            "an as-spelled path inside the root must be admitted: roots {roots:?}"
        );
        assert!(
            is_within_roots(&canonical.join("f.txt"), &roots),
            "a verbatim path inside the root must be admitted too: roots {roots:?}"
        );
        assert!(!is_within_roots(
            std::path::Path::new(r"C:\Windows\x"),
            &roots
        ));
    }

    /// A file is not a workspace root, and neither is a masked system directory.
    ///
    /// Both refusals exist because a backend cannot express the boundary otherwise, and both are
    /// reachable from `--writable-root` or an ACP client. A file root makes Landlock reject its own
    /// rule with `EINVAL` inside `pre_exec`, which kills every shell command in the session rather
    /// than just a write to that root. A root at or above one of Bubblewrap's tmpfs masks un-masks
    /// it, and those masks are what keep the D-Bus and systemd sockets out of reach, so
    /// `--writable-root /run/user/1000` would turn a workspace grant into `systemd-run --user`.
    #[test]
    fn writable_roots_refuses_a_file_and_a_masked_system_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        std::fs::create_dir(&work).expect("work");
        let not_a_directory = base.join("notes.md");
        std::fs::write(&not_a_directory, b"x").expect("write");

        let roots = writable_roots(&shared(&work), &SharedRoots::new(vec![not_a_directory]));
        assert_eq!(roots, vec![work.clone()], "a file must not become a root");

        #[cfg(unix)]
        {
            let roots = writable_roots(
                &shared(&work),
                &SharedRoots::new(vec![
                    PathBuf::from("/run"),
                    PathBuf::from("/proc"),
                    PathBuf::from("/"),
                ]),
            );
            assert_eq!(
                roots,
                vec![work],
                "no masked system directory may become a root: {roots:?}"
            );
        }
    }

    /// A runtime directory named only by `$XDG_RUNTIME_DIR` is refused in both directions.
    ///
    /// The ancestor direction is the half that matters. `$XDG_RUNTIME_DIR` sits under `$HOME` on
    /// WSL and on minimal window managers, so an ordinary `cd ~` names an ancestor of the session
    /// bus, and a boundary that only refused the directory itself would hand the bus back to a
    /// confined shell.
    #[test]
    #[cfg(unix)]
    fn a_runtime_directory_named_by_the_environment_is_refused_from_either_side() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let runtime = base.join("runtime");
        let socket = runtime.join("bus");
        std::fs::create_dir_all(&socket).expect("dirs");
        let sibling = base.join("project");
        std::fs::create_dir(&sibling).expect("sibling");

        let _guard = crate::sync::lock(&RUNTIME_DIR_ENV_LOCK);
        let previous = std::env::var_os("XDG_RUNTIME_DIR");
        // SAFETY: `XDG_RUNTIME_DIR` is process-global; `RUNTIME_DIR_ENV_LOCK` serializes every test
        // that touches it and the guard is held across the whole set/read/restore cycle.
        unsafe { std::env::set_var("XDG_RUNTIME_DIR", &runtime) };

        let at = is_system_root(&runtime);
        let under = is_system_root(&socket);
        let above = is_system_root(&base);
        let beside = is_system_root(&sibling);

        // Restored before asserting: a panic here must not leak the variable into every test that
        // runs after it.
        match previous {
            Some(value) => unsafe { std::env::set_var("XDG_RUNTIME_DIR", value) },
            None => unsafe { std::env::remove_var("XDG_RUNTIME_DIR") },
        }

        assert!(at, "the runtime directory itself must be refused");
        assert!(under, "a socket inside it must be refused");
        assert!(
            above,
            "an ancestor un-masks the whole tree, which is what `cd ~` does on WSL"
        );
        assert!(
            !beside,
            "a sibling holds no session socket and must stay usable as a workspace root"
        );
    }

    /// A root *above* a masked directory un-masks it just as surely as a root *at* it.
    ///
    /// bwrap binds each root after its tmpfs masks and the later mount wins, so `--writable-root
    /// /var` would restore the host's world-writable `/var/tmp` inside the sandbox. The gap is
    /// reachable without any exotic setup: `$XDG_RUNTIME_DIR` sits under `$HOME` on WSL and on
    /// minimal window managers, so an ordinary `cd ~` would hand the session bus back.
    #[test]
    #[cfg(unix)]
    fn a_root_above_a_masked_directory_is_refused_too() {
        for ancestor in ["/var", "/run"] {
            assert!(
                is_system_root(std::path::Path::new(ancestor)),
                "{ancestor} is an ancestor of a masked directory and must be refused"
            );
        }
        // The control: an ordinary directory that merely shares a textual prefix with one.
        assert!(
            !is_system_root(std::path::Path::new("/vary")),
            "`starts_with` is component-wise, so /vary is not an ancestor of /var/tmp"
        );
    }

    /// A root **at** a masked directory is refused; a root **under** one is not.
    ///
    /// Both halves are load-bearing and they pull in opposite directions, which is why they are
    /// asserted together. Admitting `/tmp` itself is a sandbox escape: bubblewrap masks it with a
    /// tmpfs and then binds each workspace root back afterwards, so `--bind-try /tmp /tmp` restores
    /// the whole host `/tmp` and with it every X11, D-Bus and tmux socket living there. Refusing
    /// everything *under* a masked directory is the opposite failure: `/run/media/<user>/<disk>` is
    /// where udisks2 mounts removable drives, so a project on an external drive resolved fine and
    /// was then discarded, leaving a boundary that permitted nothing anywhere.
    #[test]
    #[cfg(unix)]
    fn a_root_at_a_masked_directory_is_refused_but_one_under_it_is_not() {
        for refused in ["/tmp", "/var/tmp", "/run", "/proc", "/dev", "/sys"] {
            assert!(
                is_system_root(Path::new(refused)),
                "{refused} is masked, so binding it back would un-mask what the mask hides"
            );
        }
        for admitted in [
            "/run/media/someone/disk/project",
            "/tmp/work",
            "/var/tmp/build",
            "/dev/shm/scratch",
        ] {
            assert!(
                !is_system_root(Path::new(admitted)),
                "{admitted} is under a masked directory, not at it, and binding it back restores \
                 only itself"
            );
        }
        // The session's socket tree stays refused as a whole, with or without the variable set,
        // because every path in it is the kind of socket the masks exist to hide. The uid is one
        // that cannot be this process's own: asserting on the developer's real `XDG_RUNTIME_DIR`
        // would pass through the environment branch below and leave this literal one unguarded.
        assert!(is_system_root(Path::new("/run/user/99999")));
        assert!(is_system_root(Path::new("/run/user/99999/bus")));

        // And the real pipeline agrees: a tempdir (which lives under `/tmp` on this host) is a
        // usable root, while `/tmp` itself is not.
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        assert_eq!(usable_roots([base.clone()]), vec![base]);
        assert!(usable_roots([PathBuf::from("/tmp")]).is_empty());
    }

    /// The closed fallback admits nothing, including under the working directory: `writable_roots`
    /// unconditionally folds in the cwd, so a `deny_all` built at `Workspace` with an empty root
    /// list would grant the entire working tree. Unreachable in production, which is exactly why
    /// nothing would notice.
    #[test]
    fn deny_all_admits_nothing_not_even_the_cwd() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let cwd = shared(&base);
        let scope = WriteScope::deny_all();

        assert_eq!(
            scope.confined_to(&cwd),
            Some(Vec::new()),
            "the closed fallback must name no writable root at all"
        );
        assert!(
            scope.admit(&cwd, &base.join("f.txt")).is_err(),
            "a write in the working directory must be refused by the closed fallback"
        );
        assert!(scope.admit(&cwd, &base).is_err());
    }

    /// Which levels confine, spelled out for all four.
    ///
    /// Spelled out because a `_ => None` arm would exempt `none` and `read` along with the level it
    /// meant, and `[tools.tool_permissions]` can dispatch a write tool at either of those.
    #[test]
    fn only_unrestricted_disclaims_a_write_boundary() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let cwd = shared(&base);

        for level in [
            crate::permission::Permission::None,
            crate::permission::Permission::Read,
            crate::permission::Permission::Workspace,
            crate::permission::Permission::Unrestricted,
        ] {
            let scope = WriteScope::new(
                crate::permission::SharedPermission::new(
                    level,
                    crate::permission::EnabledPermissions::ALL,
                ),
                roots_for_test(),
                WriteLocks::default(),
            );
            let outside = std::env::temp_dir().join("meka-outside-every-root.txt");
            match level {
                crate::permission::Permission::Unrestricted => {
                    assert_eq!(
                        scope.confined_to(&cwd),
                        None,
                        "{level} disclaims a boundary: a write reaches anywhere"
                    );
                    scope
                        .admit(&cwd, &outside)
                        .expect("an unconfined level admits a path outside every root");
                }
                _ => {
                    assert_eq!(
                        scope.confined_to(&cwd),
                        Some(vec![base.clone()]),
                        "{level} must be confined to the working directory"
                    );
                    scope
                        .admit(&cwd, &base.join("f.txt"))
                        .expect("a write inside the workspace is admitted");
                    assert!(
                        scope.admit(&cwd, &outside).is_err(),
                        "{level} must refuse a write outside every root"
                    );
                }
            }
        }
    }

    /// A `..` that climbs out of the workspace is refused even though the path is never created.
    ///
    /// `admit` is reached with a path that does not exist yet on every `write_file` to a new file,
    /// so `canonicalize` cannot answer and the only defense is resolving the components as text
    /// first. Without that, `starts_with` is component-wise and answers yes for
    /// `<root>/../../etc/passwd`, because the literal path really does begin with `<root>`, so
    /// the escape is admitted by the check written to refuse it.
    ///
    /// Both halves are asserted. The refusal alone would still pass against an implementation that
    /// refused everything containing `..`, which would break `<root>/a/../b`: a legitimate write
    /// that normalizes back inside.
    #[test]
    fn a_parent_traversal_out_of_the_workspace_is_refused_before_the_file_exists() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        std::fs::create_dir(&work).expect("work");
        let cwd = shared(&work);
        let scope = WriteScope::new(
            crate::permission::SharedPermission::new(
                crate::permission::Permission::Workspace,
                crate::permission::EnabledPermissions::ALL,
            ),
            roots_for_test(),
            WriteLocks::default(),
        );

        let escape = work.join("..").join("..").join("etc").join("passwd");
        assert!(
            !escape.exists(),
            "the point of this test is a target `canonicalize` cannot resolve"
        );
        let refusal = scope
            .admit(&cwd, &escape)
            .expect_err("a `..` leaving the workspace must be refused");
        assert!(
            refusal.contains("outside the workspace"),
            "the refusal must name the boundary: {refusal}"
        );

        scope
            .admit(&cwd, &work.join("a").join("..").join("b.txt"))
            .expect("a `..` that normalizes back inside the workspace is an ordinary write");
    }

    /// A root that does not resolve permits nothing rather than permitting its name.
    ///
    /// Empty is the honest answer when the cwd has been deleted under a running session, and every
    /// caller has to treat it as "no write may land anywhere".
    #[test]
    fn an_unresolvable_root_is_dropped_rather_than_passed_through() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let missing = base.join("was-deleted");

        assert!(writable_roots(&shared(&missing), &roots_for_test()).is_empty());

        // Against a populated root set, which is what makes this an independent check: run
        // against the roots this same call has just proved empty, `is_within_roots` is `.any()`
        // over `&[]`, false for every input no matter what the function does.
        let real = base.join("real");
        std::fs::create_dir(&real).expect("real root");
        let roots = writable_roots(&shared(&real), &roots_for_test());
        assert!(!roots.is_empty(), "the control root must resolve");
        assert!(
            is_within_roots(&real.join("f.txt"), &roots),
            "a path under a resolvable root is inside it"
        );
        assert!(
            !is_within_roots(&missing.join("f.txt"), &roots),
            "a path under an unresolvable root is inside nothing, even when other roots exist"
        );
    }

    /// A nested root is redundant: anything under it is already permitted by its ancestor.
    #[test]
    fn writable_roots_drops_roots_contained_by_another() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        std::fs::create_dir_all(base.join("work/nested")).expect("dirs");

        let roots = writable_roots(
            &shared(&base.join("work")),
            &SharedRoots::new(vec![base.join("work/nested")]),
        );
        assert_eq!(roots, vec![base.join("work")]);
        assert!(is_within_roots(&base.join("work/nested/f.txt"), &roots));
    }

    /// `cwd` leads and duplicates are dropped: a client is free to repeat `cwd` inside
    /// `additionalDirectories`, and a repeated root would double every search result and spend the
    /// shared walk budget twice on the same tree.
    #[test]
    fn search_roots_puts_cwd_first_and_dedupes() {
        let cwd = SharedCwd::new(PathBuf::from("/work/main"));
        let roots = SharedRoots::new(vec![
            PathBuf::from("/work/shared"),
            PathBuf::from("/work/main"),
            PathBuf::from("/work/shared"),
            PathBuf::from("/work/docs"),
        ]);

        assert_eq!(search_roots(&cwd, &roots), vec![
            PathBuf::from("/work/main"),
            PathBuf::from("/work/shared"),
            PathBuf::from("/work/docs"),
        ]);
    }

    /// A root nested inside another is covered by it, so keeping both walks that tree twice and
    /// reports every file in it twice.
    #[test]
    fn search_roots_drops_roots_nested_in_another() {
        let cwd = SharedCwd::new(PathBuf::from("/work/main"));
        let roots = SharedRoots::new(vec![
            PathBuf::from("/work/main/nested"),
            PathBuf::from("/work/other"),
        ]);

        assert_eq!(search_roots(&cwd, &roots), vec![
            PathBuf::from("/work/main"),
            PathBuf::from("/work/other"),
        ]);
    }

    /// And the inverse: a root that *contains* `cwd` wins, because its walk already reaches
    /// everything under `cwd`. Dropping `cwd` from the search set is safe; it stays the base for
    /// relative paths and the shell either way.
    #[test]
    fn search_roots_lets_an_ancestor_root_subsume_cwd() {
        let cwd = SharedCwd::new(PathBuf::from("/work/main"));
        let roots = SharedRoots::new(vec![PathBuf::from("/work")]);
        assert_eq!(search_roots(&cwd, &roots), vec![PathBuf::from("/work")]);
    }

    /// A shared prefix is not containment: `/work/main2` is not inside `/work/main`.
    #[test]
    fn search_roots_keeps_sibling_with_shared_prefix() {
        let cwd = SharedCwd::new(PathBuf::from("/work/main"));
        let roots = SharedRoots::new(vec![PathBuf::from("/work/main2")]);
        assert_eq!(search_roots(&cwd, &roots), vec![
            PathBuf::from("/work/main"),
            PathBuf::from("/work/main2"),
        ]);
    }

    /// The single-root case has to stay exactly one path: that is every REPL and HTTP session, and
    /// every ACP client that sends no extra roots.
    #[test]
    fn search_roots_without_extras_is_just_cwd() {
        let cwd = SharedCwd::new(PathBuf::from("/work/main"));
        let roots = SharedRoots::default();
        assert_eq!(search_roots(&cwd, &roots), vec![PathBuf::from(
            "/work/main"
        )]);
    }

    #[test]
    fn resolve_against_cwd_passes_absolute_paths_through() {
        let cwd = SharedCwd::new(PathBuf::from("/home/agent"));
        let absolute = std::path::Path::new("/etc/hosts");
        let resolved = resolve_against_cwd(&cwd, absolute);
        assert_eq!(resolved, PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn resolve_against_cwd_joins_relative_paths_to_session_cwd() {
        let cwd = SharedCwd::new(PathBuf::from("/home/agent/project"));
        let resolved = resolve_against_cwd(&cwd, "src/main.rs");
        assert_eq!(resolved, PathBuf::from("/home/agent/project/src/main.rs"));
    }

    #[test]
    fn resolve_against_cwd_follows_subsequent_writes() {
        // Confirms multiple sessions in one process would observe their own cwds: a write to the
        // shared lock is visible on the next resolve, without touching process cwd.
        let cwd = SharedCwd::new(PathBuf::from("/tmp/a"));
        let first = resolve_against_cwd(&cwd, "foo.txt");
        cwd.set(PathBuf::from("/tmp/b"));
        let second = resolve_against_cwd(&cwd, "foo.txt");
        assert_eq!(first, PathBuf::from("/tmp/a/foo.txt"));
        assert_eq!(second, PathBuf::from("/tmp/b/foo.txt"));
    }
}
