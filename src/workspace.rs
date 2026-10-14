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
/// the screen prompt, the `/cd` slash command, and the per-turn environment-context block.
/// `std::sync::RwLock` (rather than `tokio::sync::RwLock`) so the synchronous screen prompt can
/// read it without entering an async context; reads/writes are microseconds (a `PathBuf` clone or
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
/// Only correct for a walker that descends, which today means `file_search`. A tool that
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
/// **This is the one definition of the boundary.** The in-process fence on `file_write` /
/// `file_edit` / `scratchpad_save_file` and every sandbox dialect (Landlock, Bubblewrap, Seatbelt,
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

/// meka's own directories: the config directory and the data directory holding the credential
/// store.
///
/// Every sandbox dialect that can hides these from a confined shell, and the write fence refuses a
/// target under them whatever roots the session holds. The shell's environment is already scrubbed
/// because a leaked secret plus the open network is a live exfiltration vector under prompt
/// injection; the same secrets sit in `meka.db` at a path any shell can guess, so leaving the disk
/// door open while guarding the environment one guarded nothing. Canonicalized where they exist,
/// so a symlinked home matches the path the kernel reports; deduplicated because an override can
/// point both at one place.
pub(crate) fn private_directories() -> Vec<PathBuf> {
    let candidates = [
        crate::paths::meka_config_dir(),
        crate::paths::meka_data_dir(),
    ];
    private_directories_among(candidates.into_iter().flatten())
}

/// [`private_directories`] over the given candidates, so the rule can be tested without the
/// environment deciding where meka's directories are.
fn private_directories_among(candidates: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut directories: Vec<PathBuf> = Vec::new();
    for directory in candidates {
        // Only a directory that exists: bubblewrap mounts its mask over a path in the bound root,
        // and there is nothing to hide where nothing is. And never one the system masks already
        // cover, which an override pointed at `/tmp` itself would produce: taking `/tmp` as
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

/// The part of the private directories that stays readable below `unrestricted`: meka's own
/// skills store. A skill's bundled files are what its body tells the model to read, and the
/// store sits under the config directory only because that is where meka keeps what the user
/// authored; nothing secret lives there. The shell sandbox still masks it, so the header a skill
/// is served with says to read them with `file_read`.
fn readable_within_private() -> Vec<PathBuf> {
    private_directories_among(crate::paths::skills_dir())
}

/// The refusal for reading a canonical `target` at `permission` when it lies inside meka's own
/// directories, with the private and readable directories taken as parameters: the rule
/// [`ReadFence`] applies, exposed so the tests can state it without the environment deciding
/// where meka's directories are.
#[cfg(test)]
fn private_read_refusal_with(
    permission: crate::permission::Permission,
    target: &Path,
    private: &[PathBuf],
    readable: &[PathBuf],
) -> Option<String> {
    if permission == crate::permission::Permission::Unrestricted {
        return None;
    }
    ReadFence {
        private: private.to_vec(),
        readable: readable.to_vec(),
        confined: None,
        named: Vec::new(),
    }
    .refusal_shown(target, target)
}

/// Whether `target` lies under a process's own directory in `/proc`: `/proc/<pid>`, or `self` and
/// `thread-self`, which canonicalization has usually already turned into a pid.
///
/// The system-wide files (`meminfo`, `cpuinfo`, `mounts`, `net/*`) stay readable: they carry no
/// secret, and at `read` the shell is closed, so `file_read` is the only door to them. Every entry
/// that does carry one is per process: `environ` holds what a process was given, which for an MCP
/// server meka spawned is its credential and for meka itself every `${VAR}` `config.toml` names;
/// `cmdline`, `fd`, `maps` and `mem` (readable for a child under the default ptrace scope) hold
/// the rest. The sandboxed shell already sees no other process; this is the same line for the
/// in-process readers.
fn is_process_proc_entry(target: &Path) -> bool {
    use std::path::Component;
    let mut components = target.components();
    if components.next() != Some(Component::RootDir) {
        return false;
    }
    if components.next() != Some(Component::Normal("proc".as_ref())) {
        return false;
    }
    let Some(Component::Normal(process)) = components.next() else {
        return false;
    };
    process.to_str().is_some_and(|name| {
        name == "self" || name == "thread-self" || name.bytes().all(|byte| byte.is_ascii_digit())
    })
}

/// What a read below `unrestricted` may not reach: meka's own directories, less the part of them
/// that stays readable, and, when reads are confined, anything outside the readable roots. None
/// of it at `unrestricted`, so a walk there resolves nothing.
///
/// Built once per tool call by [`ReadScope::fence`] and asked once per path, so the roots are
/// resolved once rather than per directory entry.
pub(crate) struct ReadFence {
    private: Vec<PathBuf>,
    readable: Vec<PathBuf>,
    /// The roots a read must stay under, or `None` when reads reach everywhere.
    confined: Option<Vec<PathBuf>>,
    /// The session's own roots among them, which a refusal names; the system directories are
    /// named in a word, since listing them would bury the one the model has to keep to.
    named: Vec<PathBuf>,
}

impl ReadFence {
    /// The fence at `permission` with reads unconfined: meka's own directories alone.
    pub(crate) fn at(permission: crate::permission::Permission) -> Self {
        if permission == crate::permission::Permission::Unrestricted {
            return Self {
                private: Vec::new(),
                readable: Vec::new(),
                confined: None,
                named: Vec::new(),
            };
        }
        Self {
            private: private_directories(),
            readable: readable_within_private(),
            confined: None,
            named: Vec::new(),
        }
    }

    /// Whether `path` is one this fence refuses; see [`Self::refusal`].
    pub(crate) fn hides(&self, path: &Path) -> bool {
        self.refusal(path).is_some()
    }

    /// Why `path` may not be read, or `None` when it may: it lies inside a private directory and
    /// outside the readable part of it, it is a process's own `/proc` entry, or reads are confined
    /// and it lies outside the readable roots. The path is judged with every component that exists
    /// resolved and the rest as spelled ([`resolve_existing_prefix`]), so one outside the roots is
    /// refused the same way whether or not it exists, and the refusal names the path as spelled
    /// rather than where a link leads: the boundary keeps names as well as contents. A path that
    /// does not exist inside the roots is admitted, and the read of it fails on its own. An empty
    /// fence costs no resolution at all.
    pub(crate) fn refusal(&self, path: &Path) -> Option<String> {
        if self.private.is_empty() && self.confined.is_none() {
            return None;
        }
        self.refusal_shown(path, &resolve_existing_prefix(path))
    }

    /// [`Self::refusal`] for a path the caller has already resolved to `judged`, worded with
    /// `shown`, the spelling the caller used.
    pub(crate) fn refusal_shown(&self, shown: &Path, judged: &Path) -> Option<String> {
        if self
            .readable
            .iter()
            .any(|directory| judged.starts_with(directory))
        {
            return None;
        }
        if is_process_proc_entry(judged) {
            return Some(format!(
                "'{}' is a process's `/proc` entry, which only `unrestricted` reads: `environ`, \
                 `cmdline`, `fd` and `mem` carry what the process was given.",
                shown.display()
            ));
        }
        if let Some(directory) = self
            .private
            .iter()
            .find(|directory| judged.starts_with(directory))
        {
            return Some(format!(
                "'{}' is inside meka's own directory {}, which only `unrestricted` reads.",
                shown.display(),
                directory.display()
            ));
        }
        let roots = self.confined.as_deref()?;
        if is_within_roots(judged, roots) {
            return None;
        }
        // Names the roots, as the write fence does: the model cannot see the boundary from the
        // tool schema, and without them its next attempt is a guess.
        let where_to = if self.named.is_empty() {
            "no readable root currently resolves, and reads reach only the system directories"
                .to_string()
        } else {
            format!(
                "reads stay under {} and the system directories",
                self.named
                    .iter()
                    .map(|root| root.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        Some(format!(
            "'{}' is outside the readable roots: below `unrestricted` {where_to}. Ask the user to \
             add the directory as a root if the task needs it.",
            shown.display()
        ))
    }
}

/// The directories every confined command may read whatever roots the session holds: where the
/// platform keeps the programs, libraries and configuration a command needs to run at all.
/// System-level paths only, so the list widens nothing of the user's; the home directory is the
/// point of the boundary and is deliberately absent. Each is taken only where it exists.
///
/// The Linux list is the one Codex's restricted-read sandbox binds, plus `/proc` and `/dev`,
/// which Bubblewrap mounts on its own and Landlock alone has to be told about. The macOS list is
/// Codex's `:minimal` seatbelt defaults widened to the two package prefixes whole and to Apple's
/// developer tools, since nearly every command a developer runs comes from one of them and the
/// boundary exists to stop reads of the user's data, not the system's programs. Windows confines
/// no command's reads, so it lists nothing: the file tools are fenced by the roots alone there.
#[cfg(target_os = "linux")]
pub(crate) const SYSTEM_READ_PATHS: &[&str] = &[
    "/bin",
    "/sbin",
    "/usr",
    "/lib",
    "/lib32",
    "/lib64",
    "/etc",
    "/proc",
    "/dev",
    "/nix/store",
    "/run/current-system/sw",
];
#[cfg(target_os = "macos")]
pub(crate) const SYSTEM_READ_PATHS: &[&str] = &[
    "/bin",
    "/sbin",
    "/usr",
    "/etc",
    "/private/etc",
    "/System",
    "/Library/Apple",
    "/Library/Preferences",
    "/Library/Filesystems/NetFSPlugins",
    "/var/db",
    "/private/var/db",
    "/opt/homebrew",
    "/Library/Developer",
    "/dev",
];
#[cfg(windows)]
pub(crate) const SYSTEM_READ_PATHS: &[&str] = &[];

/// Where macOS records the active developer directory: a symlink `xcode-select` maintains,
/// pointing at the Command Line Tools or at an Xcode's `Contents/Developer`. The developer
/// commands in `/usr/bin` are stubs that exec the real binary from wherever it points, so its
/// target is readable whatever it is; asked of the system rather than guessed, since only
/// `xcode-select` knows which.
#[cfg(target_os = "macos")]
const DEVELOPER_DIRECTORY_LINK: &str = "/var/db/xcode_select_link";

/// What `[permissions].confine_reads`, `[shell].readable_paths` and `[skills].extra_paths` resolve
/// to for a process: the value every root session's [`ReadScope`] is built from, and the one a
/// sub-agent's is derived from.
#[derive(Debug, Clone, Default)]
pub(crate) struct ReadBoundary {
    /// Whether reads below `unrestricted` are confined at all.
    pub(crate) confined: bool,
    /// The paths every confined read may reach beyond the session's roots: the system directories,
    /// the configured extras and the skill roots, only where they exist, each as written and in
    /// canonical form. Both spellings because the consumers differ: a bind mount wants the name,
    /// since `/lib64` is a symlink on a merged-`/usr` system and the interpreter path every
    /// binary hardcodes runs through it, so a sandbox holding only `/usr/lib` can start nothing;
    /// the fence and a Landlock rule judge the resolved path, which is the target. Resolved
    /// whether or not `confined` is set, so a sub-agent whose spawn turns confinement on has the
    /// platform's paths to run with.
    pub(crate) base: Vec<PathBuf>,
}

impl ReadBoundary {
    /// The boundary the configuration asks for; see [`Self::new`].
    pub(crate) fn from_config(config: &crate::config::ResolvedConfig) -> Self {
        Self::new(
            config.confine_reads,
            &config.readable_paths,
            &config.skills_extra_paths,
        )
    }

    /// The boundary over the platform's paths, `readable_paths` and `skill_roots`.
    ///
    /// A `readable_paths` entry is dropped with a warning when it is relative, since a relative
    /// spelling would be bound and granted against whatever directory the child is in; when it
    /// does not resolve, as a workspace root is, since a boundary naming a directory that is not
    /// there should permit nothing rather than a name something could later be created at; and
    /// when it is a directory the sandbox masks or one of meka's own ([`is_system_root`]), since
    /// Bubblewrap binds the readable paths after its masks and a later mount wins, so `/run`
    /// named here would hand the session bus back to a confined shell through a read-only bind.
    /// A skill root is judged the same way, silently when absent, since a configured root that is
    /// not there is already promised to leave no trace: a skill's bundled files are what its body
    /// tells the model to read, which is why meka's own skills store stays readable too.
    pub(crate) fn new(confined: bool, readable_paths: &[PathBuf], skill_roots: &[PathBuf]) -> Self {
        let mut base: Vec<PathBuf> = Vec::new();
        // `key` names the configuration key a dropped entry is reported under, `None` for the
        // system directories, which are never reported and never judged against the masks,
        // since `/proc` and `/dev` are among them for Landlock's sake. `report_absent` is off for
        // a skill root, whose absence is promised to leave no trace.
        let mut take = |path: PathBuf, key: Option<&str>, report_absent: bool| {
            let displayed = path.display().to_string();
            if !path.is_absolute() {
                if let Some(key) = key
                    && report_absent
                {
                    tracing::warn!(
                        "`{key}` entry '{displayed}' is not an absolute path; ignoring it"
                    );
                }
                return;
            }
            let Ok(canonical) = std::fs::canonicalize(&path).map(strip_verbatim) else {
                if let Some(key) = key
                    && report_absent
                {
                    tracing::warn!("`{key}` entry '{displayed}' does not resolve; ignoring it");
                }
                return;
            };
            if let Some(key) = key
                && is_system_root(&canonical)
            {
                tracing::warn!(
                    "`{key}` entry '{displayed}' is a directory the sandbox masks; ignoring it"
                );
                return;
            }
            for spelling in [strip_verbatim(path), canonical] {
                if !base.contains(&spelling) {
                    base.push(spelling);
                }
            }
        };
        for path in SYSTEM_READ_PATHS {
            take(PathBuf::from(path), None, false);
        }
        #[cfg(target_os = "macos")]
        if let Ok(target) = std::fs::canonicalize(DEVELOPER_DIRECTORY_LINK) {
            take(target, None, false);
        }
        for path in readable_paths {
            take(path.clone(), Some("[shell].readable_paths"), true);
        }
        for path in skill_roots {
            take(path.clone(), Some("[skills].extra_paths"), false);
        }
        Self { confined, base }
    }

    /// No boundary: reads reach everywhere but meka's own directories.
    pub(crate) fn unconfined() -> Self {
        Self::default()
    }

    /// The boundary a sub-agent holds under this one: this one, or confined when its spawn
    /// `granted` readable roots. One rule for both doors a worker comes through, so a spawn turns
    /// confinement on and never off: a parent whose reads are confined cannot spawn a reader
    /// that is not, and a parent reading everywhere can hand a worker a boundary the way it hands
    /// one a write boundary with `writable_roots`.
    pub(crate) fn for_worker(&self, granted: bool) -> Self {
        Self {
            confined: self.confined || granted,
            base: self.base.clone(),
        }
    }
}

/// The roots a confined read may land under: the writable roots, the read-only roots and the
/// base set, canonical, with anything already covered by another dropped. Writable roots are
/// readable by definition, so the one list [`writable_roots`] computes is the first part of this
/// one, and the two cannot disagree about a root.
pub(crate) fn readable_roots(
    cwd: &SharedCwd,
    roots: &SharedRoots,
    read_only_roots: &SharedRoots,
    base: &[PathBuf],
) -> Vec<PathBuf> {
    retain_broadest(
        usable_roots(
            std::iter::once(cwd.get())
                .chain(roots.get())
                .chain(read_only_roots.get()),
        )
        .into_iter()
        .chain(base.iter().cloned()),
    )
}

/// The read boundary of a session, the counterpart of [`WriteScope`]: the one definition of where
/// a read may land below `unrestricted`, for the in-process read tools and for every sandbox
/// dialect alike.
#[derive(Clone)]
pub(crate) struct ReadScope {
    permission: crate::permission::SharedPermission,
    roots: SharedRoots,
    read_only_roots: SharedRoots,
    boundary: ReadBoundary,
}

impl ReadScope {
    pub(crate) fn new(
        permission: crate::permission::SharedPermission,
        roots: SharedRoots,
        read_only_roots: SharedRoots,
        boundary: ReadBoundary,
    ) -> Self {
        Self {
            permission,
            roots,
            read_only_roots,
            boundary,
        }
    }

    /// The roots a read must land under at `level`, or `None` when reads reach everywhere: with
    /// the boundary off, and at `unrestricted`, the one level that disclaims every boundary. Like
    /// [`WriteScope::confined_to_at`], the level is the caller's, read once at its enforcement
    /// site.
    pub(crate) fn confined_to_at(
        &self,
        level: crate::permission::Permission,
        cwd: &SharedCwd,
    ) -> Option<Vec<PathBuf>> {
        if !self.boundary.confined || level == crate::permission::Permission::Unrestricted {
            return None;
        }
        Some(readable_roots(
            cwd,
            &self.roots,
            &self.read_only_roots,
            &self.boundary.base,
        ))
    }

    /// The session's own readable roots at `level`, without the system directories, or `None`
    /// when reads reach everywhere: what the environment block names, since the system directories
    /// are the same on every turn and are stated in a word.
    pub(crate) fn session_roots_at(
        &self,
        level: crate::permission::Permission,
        cwd: &SharedCwd,
    ) -> Option<Vec<PathBuf>> {
        if !self.boundary.confined || level == crate::permission::Permission::Unrestricted {
            return None;
        }
        Some(usable_roots(
            std::iter::once(cwd.get())
                .chain(self.roots.get())
                .chain(self.read_only_roots.get()),
        ))
    }

    /// The fence for one tool call at the level the call was admitted at.
    pub(crate) fn fence_at(
        &self,
        level: crate::permission::Permission,
        cwd: &SharedCwd,
    ) -> ReadFence {
        let mut fence = ReadFence::at(level);
        // The session's roots resolved once, then the whole set over them, so a refusal can name
        // the former without resolving them a second time.
        if let Some(session) = self.session_roots_at(level, cwd) {
            fence.confined = Some(retain_broadest(
                session
                    .iter()
                    .cloned()
                    .chain(self.boundary.base.iter().cloned()),
            ));
            fence.named = session;
        }
        fence
    }

    /// [`Self::fence_at`] at the level the handle reads now.
    pub(crate) fn fence(&self, cwd: &SharedCwd) -> ReadFence {
        self.fence_at(self.permission.get(), cwd)
    }

    /// Judge a directory a sub-agent would hold against this scope's reach at `level`: inside
    /// the readable roots when reads are confined, anywhere when they are not. The one rule for a
    /// worker's directory, whichever door hands it over, so a sub-agent's reach never exceeds its
    /// parent's. `directory` is judged with every component that exists resolved and the rest as
    /// spelled, so one outside the reach is refused the same way whether or not it exists, and the
    /// refusal names `shown`, the spelling the caller used, rather than where a link leads.
    pub(crate) fn admit_within_reach(
        &self,
        level: crate::permission::Permission,
        cwd: &SharedCwd,
        shown: &Path,
        directory: &Path,
    ) -> Result<(), String> {
        let Some(reach) = self.confined_to_at(level, cwd) else {
            return Ok(());
        };
        if is_within_roots(&resolve_existing_prefix(directory), &reach) {
            return Ok(());
        }
        let where_to = if reach.is_empty() {
            "no readable root of this session currently resolves".to_string()
        } else {
            format!(
                "this session's reads stay under {}",
                reach
                    .iter()
                    .map(|root| root.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        Err(format!(
            "'{}' is outside this session's readable roots: {where_to}, and a sub-agent's reach \
             cannot exceed its parent's",
            shown.display()
        ))
    }

    /// Judge one read of `path`, as [`ReadFence::refusal`] judges it, returning the refusal
    /// sentence when it is refused. The tools ask the fence directly; this is the tests' shorthand.
    #[cfg(test)]
    pub(crate) fn admit(&self, cwd: &SharedCwd, path: &Path) -> Result<(), String> {
        match self.fence(cwd).refusal(path) {
            Some(refusal) => Err(refusal),
            None => Ok(()),
        }
    }

    /// A scope that confines nothing but meka's own directories, for tools under test that are
    /// not exercising the boundary.
    #[cfg(test)]
    pub(crate) fn unconfined(permission: crate::permission::SharedPermission) -> Self {
        Self::new(
            permission,
            SharedRoots::default(),
            SharedRoots::default(),
            ReadBoundary::unconfined(),
        )
    }

    /// A scope confined to `roots` plus `base`, for tests that *are* exercising the boundary.
    #[cfg(test)]
    pub(crate) fn confined(
        permission: crate::permission::SharedPermission,
        roots: Vec<PathBuf>,
        base: Vec<PathBuf>,
    ) -> Self {
        Self::new(
            permission,
            SharedRoots::new(roots),
            SharedRoots::default(),
            ReadBoundary {
                confined: true,
                base,
            },
        )
    }
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
/// entry must also lie inside the parent's own boundary, [`writable_roots`], judged component-wise
/// with what exists resolved before its existence is checked and again once canonical, so a
/// sub-agent's reach is never wider than its parent's and a refusal does not say whether the
/// directory is there; at `unrestricted` the parent may write anywhere, and so may name anywhere.
pub(crate) fn accept_writable_roots(
    parent_level: crate::permission::Permission,
    parent_cwd: &SharedCwd,
    parent_roots: &SharedRoots,
    parent_reads: &ReadScope,
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
    let mut accepted = Vec::with_capacity(requested.len());
    for entry in requested {
        let resolved = resolve_against_cwd(parent_cwd, entry);
        // Reach before existence, so an entry outside it is refused the same way whether or not
        // it is there; then once more on the canonical directory, which is what the worker
        // holds, so a link swapped between the two resolutions lands where it is judged.
        let within_reach = |directory: &Path| {
            admit_within_parent_reach(
                parent_level,
                parent_cwd,
                parent_roots,
                parent_reads,
                &resolved,
                directory,
            )
            .map_err(|error| MekaError::Usage(format!("writable_roots: {error}")))
        };
        within_reach(&resolved)?;
        let canonical = accept_cwd(&resolved)
            .map_err(|error| MekaError::Usage(format!("writable_roots: {error}")))?;
        within_reach(&canonical)?;
        if is_system_root(&canonical) {
            return Err(MekaError::Usage(format!(
                "writable_roots: '{}' is a system directory the sandbox masks, so it cannot be a \
                 workspace root",
                resolved.display()
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

/// Admit the directories a parent agent names as a sub-agent's read-only roots: what the worker
/// may read beyond what it may write.
///
/// One definition for both doors a worker comes through, `agent_spawn` and `agent_followup`, so
/// the roots a follow-up replays are judged by the rule the spawn was. Every refusal happens here,
/// before either door has written anything. Each entry is resolved against the parent's working
/// directory when relative and admitted as a working directory is ([`accept_cwd`]), and refused
/// when it is a directory no root may name ([`is_system_root`]). When the parent's own reads are
/// confined, every entry must lie inside the parent's readable roots, judged component-wise with
/// what exists resolved before its existence is checked and again once canonical, so a sub-agent's
/// reach is never wider than its parent's and a refusal does not say whether the directory is
/// there; a parent
/// that reads everywhere may name anywhere, and the list then confines a worker whose reads were
/// confined by nothing ([`ReadBoundary::for_worker`]). An empty list is refused, as an empty
/// `writable_roots` is: a present list is accepted whole or refused whole, so `[]` never carries
/// a meaning of its own beside "omitted", which a model emits for "nothing to add".
///
/// Every entry must also lie outside `workspace`, the worker's own workspace as [`writable_roots`]
/// computes it, whether shared with the parent or set by `writable_roots`. The workspace is
/// readable at every level, so an entry inside it confines nothing, and a parent that names a
/// subdirectory of its workspace wants the worker confined to it, which only `writable_roots` does.
/// The refusal names that remedy rather than accepting a list that reads as a boundary it is not.
pub(crate) fn accept_readable_roots(
    parent_level: crate::permission::Permission,
    parent_cwd: &SharedCwd,
    parent_reads: &ReadScope,
    workspace: &[PathBuf],
    requested: &[PathBuf],
) -> crate::error::Result<Vec<PathBuf>> {
    use crate::error::MekaError;

    if requested.is_empty() {
        return Err(MekaError::Usage(
            "readable_roots names no directory; leave it out to read what this session reads"
                .to_string(),
        ));
    }
    let mut accepted = Vec::with_capacity(requested.len());
    for entry in requested {
        let resolved = resolve_against_cwd(parent_cwd, entry);
        // Reach before existence, so an entry outside it is refused the same way whether or not
        // it is there; then once more on the canonical directory, which is what the worker
        // holds, so a link swapped between the two resolutions lands where it is judged.
        let within_reach = |directory: &Path| {
            parent_reads
                .admit_within_reach(parent_level, parent_cwd, &resolved, directory)
                .map_err(|error| MekaError::Usage(format!("readable_roots: {error}")))
        };
        within_reach(&resolved)?;
        let canonical = accept_cwd(&resolved)
            .map_err(|error| MekaError::Usage(format!("readable_roots: {error}")))?;
        within_reach(&canonical)?;
        if is_system_root(&canonical) {
            return Err(MekaError::Usage(format!(
                "readable_roots: '{}' is a system directory the sandbox masks, so it cannot be a \
                 root",
                resolved.display()
            )));
        }
        if let Some(root) = workspace.iter().find(|root| canonical.starts_with(root)) {
            return Err(MekaError::Usage(format!(
                "readable_roots: '{}' is inside the sub-agent's workspace at '{}', which it reads \
                 already; to confine the sub-agent to it, name the directory in `writable_roots` \
                 instead, with `permission: \"read\"` for one that writes nothing",
                resolved.display(),
                root.display()
            )));
        }
        if !accepted.contains(&canonical) {
            accepted.push(canonical);
        }
    }
    Ok(accepted)
}

/// Admit a directory a sub-agent works in against its parent's reach *now*.
///
/// One rule for what a parent's state means for the directory a worker holds, whichever door gave
/// it: an entry of `writable_roots`, judged by [`accept_writable_roots`] at spawn and again on a
/// follow-up, or the parent's own working directory at spawn, which a plain worker keeps across a
/// follow-up and which is judged here once the parent may have moved. Two doors deciding this
/// separately is how one of them came to resume a writer under a directory the other refused.
///
/// At `workspace` the directory must lie within [`writable_roots`], judged component-wise, so a
/// sub-agent's reach is never wider than its parent's. `unrestricted`
/// reaches everywhere and bounds nothing. Below `workspace` the parent has no write reach and a
/// worker under it is narrowed to none, so there is nothing to compare. When the parent's reads
/// are confined the directory must also lie within its readable roots, at every level below
/// `unrestricted`: a worker reads its own directory, so one the parent can no longer read is a
/// reader the parent could not spawn today. The directory is judged with every component that
/// exists resolved and the rest as spelled ([`resolve_existing_prefix`]), so one outside the reach
/// is refused the same way whether or not it exists, and a refusal names `shown`, the spelling the
/// caller used, rather than where a link leads.
pub(crate) fn admit_within_parent_reach(
    parent_level: crate::permission::Permission,
    parent_cwd: &SharedCwd,
    parent_roots: &SharedRoots,
    parent_reads: &ReadScope,
    shown: &Path,
    directory: &Path,
) -> Result<(), String> {
    if parent_level == crate::permission::Permission::Workspace {
        let reach = writable_roots(parent_cwd, parent_roots);
        if !is_within_roots(&resolve_existing_prefix(directory), &reach) {
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
            return Err(format!(
                "'{}' is outside this session's workspace: {where_to}, and a sub-agent's reach \
                 cannot exceed its parent's",
                shown.display()
            ));
        }
    }
    parent_reads.admit_within_reach(parent_level, parent_cwd, shown, directory)
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

/// Resolve every component of `path` that already exists, leaving the rest as spelled.
///
/// This is what the kernel does when the path is finally opened, computed ahead of time so a
/// boundary check and the syscall agree about which file is meant. Walking forward rather than
/// canonicalizing the whole path matters because the target often does not exist, and
/// `canonicalize` fails outright on a missing tail. Every boundary judges a path through this, the
/// read fence and the spawn acceptors included, so a path outside a boundary is refused the same
/// way whether or not it exists: `canonicalize` alone answers "not found" for a missing one and
/// "outside" for one that is there, and a confined session could learn from the difference which
/// paths exist.
///
/// `..` pops the *resolved* accumulator, not the spelling, which is the whole point: `<root>/L/..`
/// where `L` is a symlink lands where the kernel lands, not where the text suggests.
///
/// What this closes is the deterministic escape: a symlink already planted on the path resolves
/// here and is caught by the boundary check that follows. What it does not close is the race, since
/// a component verified here can be swapped for a symlink before the open; closing that needs the
/// walk to hold a directory descriptor and write through `openat`/`renameat`. The sandbox is
/// documented as defense against an agent damaging user data by accident rather than as an
/// adversarial boundary, and winning this race requires a concurrent writer deliberately planting
/// the symlink mid-call, so the residual is accepted.
///
/// `path` is absolute, as every caller's is after [`resolve_against_cwd`]: `canonicalize` resolves
/// a relative one against the process's directory, which is not the session's.
pub(crate) fn resolve_existing_prefix(path: &Path) -> PathBuf {
    use std::path::Component;

    // A path that exists in full resolves in one call, and that is what a walk asks for every
    // entry it visits; the component walk below is for a path with a missing tail.
    if let Ok(resolved) = std::fs::canonicalize(path) {
        return strip_verbatim(resolved);
    }
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => out.push(component),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(part) => {
                out.push(part);
                // A component that does not exist cannot be a symlink, so a failure here means the
                // rest of the path is a pure spelling and needs no further resolution.
                if let Ok(resolved) = std::fs::canonicalize(&out) {
                    // Stripped, or the accumulator switches spelling mid-walk: the first existing
                    // component turns `C:\ws` into `\\?\C:\ws` and every component after it is
                    // pushed onto the verbatim form.
                    out = strip_verbatim(resolved);
                }
            }
        }
    }
    out
}

/// Normalize `.` and `..` textually, without consulting the filesystem.
///
/// The filesystem cannot help for the case this exists for: a `file_write` naming a path that does
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

    /// The write lock for `canonical`, from the map every scope built for this host shares.
    pub(crate) async fn lock_path(&self, canonical: &Path) -> tokio::sync::OwnedMutexGuard<()> {
        self.locks.lock_path(canonical).await
    }

    /// The roots a write must land under at `level`, or `None` when that level imposes no
    /// boundary.
    ///
    /// Only `unrestricted` disclaims a boundary, by definition; an approved call runs at its own
    /// level and reaches no further. Every other level, `none` and `read` included, is confined
    /// here: a level that should never reach a write door must fail closed if one is ever wired to
    /// it. The level is the caller's, read once at its enforcement site, never re-read off the
    /// handle here: a door that judged the call at one level must not confine it at another.
    pub(crate) fn confined_to_at(
        &self,
        level: crate::permission::Permission,
        cwd: &SharedCwd,
    ) -> Option<Vec<PathBuf>> {
        if self.denied {
            return Some(Vec::new());
        }
        match level {
            // Only the level that disclaims a boundary is exempt. Written as an allow-list rather
            // than `Workspace => Some(..), _ => None`, because that catch-all fails open: `none`
            // and `read` normally never reach a write door, but `[tools.tool_permissions]`
            // overrides a tool's required level with no floor, so `file_write = "read"` dispatches
            // the tool at `read` and the fence must still confine it.
            //
            // At `none` and `read` this yields the workspace roots rather than nothing, which is a
            // deliberate difference from `Confinement::resolve`, whose catch-all is `ReadOnly` and
            // grants no write at all. The two are *not* the same decision:
            //
            // - `Confinement::resolve` answers "what may a command meka did not write do", and the
            //   honest answer at `read` is nothing.
            // - This answers "where may a tool the operator deliberately lowered write", and a
            //   `file_write = "read"` override is a statement that the tool should be usable at
            //   that level. Refusing it outright would make the override a no-op with no
            //   diagnostic; confining it to the workspace roots is the narrowest reading that still
            //   honors what was configured.
            //
            // So the override can only ever narrow *reach*: it never escapes the roots, and it
            // cannot touch `shell_execute`, which stays closed at `read` regardless.
            crate::permission::Permission::Unrestricted => None,
            _ => Some(writable_roots(cwd, &self.roots)),
        }
    }

    /// Judge one write target, returning the refusal sentence when it falls outside the boundary.
    ///
    /// `target` should be canonical where the caller can manage it and lexically normalized where
    /// it cannot (a path being created). Both are checked the same way; the difference is only in
    /// how much the caller has been able to resolve. The refusal names `target`; every door
    /// words it with the spelling its caller used through [`Self::admit_shown`], so this is the
    /// tests' shorthand.
    #[cfg(test)]
    pub(crate) fn admit(&self, cwd: &SharedCwd, target: &std::path::Path) -> Result<(), String> {
        self.admit_shown(cwd, target, target)
    }

    /// Judge one write target, worded with `shown`, the spelling the caller used, rather than
    /// with the resolved `target`: a refusal that named where a link leads would give away a name
    /// the boundary keeps.
    pub(crate) fn admit_shown(
        &self,
        cwd: &SharedCwd,
        shown: &std::path::Path,
        target: &std::path::Path,
    ) -> Result<(), String> {
        self.admit_shown_at(self.permission.get(), cwd, shown, target)
    }

    /// [`Self::admit_shown`] at a level the caller has already read. The approval door asks this
    /// with the level it admitted the call at, so the refusal it reports ahead of the prompt is
    /// the one the write would meet, not one read off the handle a moment later.
    pub(crate) fn admit_shown_at(
        &self,
        level: crate::permission::Permission,
        cwd: &SharedCwd,
        shown: &std::path::Path,
        target: &std::path::Path,
    ) -> Result<(), String> {
        self.admit_with_private(level, cwd, shown, target, &private_directories())
    }

    /// [`Self::admit_shown_at`] with meka's private directories taken as a parameter, so the
    /// refusal can be tested without the environment deciding where those are.
    fn admit_with_private(
        &self,
        level: crate::permission::Permission,
        cwd: &SharedCwd,
        shown: &std::path::Path,
        target: &std::path::Path,
        private: &[PathBuf],
    ) -> Result<(), String> {
        let Some(roots) = self.confined_to_at(level, cwd) else {
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
                shown.display(),
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
            "'{}' is outside the workspace: below `unrestricted` {}. Pick a path inside, or \
             ask the user for `unrestricted` if it genuinely belongs elsewhere.",
            shown.display(),
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
/// than descending from it. Containment must not drop anything here: `file_find` turns each root
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

    /// A scope confining nothing, for the tests of the write side.
    fn reads_for_test() -> ReadScope {
        ReadScope::unconfined(crate::permission::SharedPermission::new(
            crate::permission::Permission::Workspace,
            crate::permission::EnabledPermissions::ALL,
        ))
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
            let refusal = accept_writable_roots(
                level,
                &parent_cwd,
                &parent_roots,
                &reads_for_test(),
                &requested,
            )
            .expect_err("a parent that cannot write cannot delegate writing")
            .to_string();
            assert!(
                refusal.contains("needs a parent at `workspace` or `unrestricted`"),
                "{refusal}"
            );
        }
        let refusal = accept_writable_roots(
            Permission::Workspace,
            &parent_cwd,
            &parent_roots,
            &reads_for_test(),
            &[],
        )
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

        let accepted = accept_writable_roots(
            Permission::Workspace,
            &parent_cwd,
            &parent_roots,
            &reads_for_test(),
            &[work.join("sub"), docs.clone()],
        )
        .expect("a directory under the cwd and one under a named root are both inside");
        assert_eq!(accepted, BoundedWorkspace {
            cwd: work.join("sub"),
            additional_roots: vec![docs.clone()],
        });
        assert_eq!(accepted.roots(), vec![work.join("sub"), docs]);

        let refusal = accept_writable_roots(
            Permission::Workspace,
            &parent_cwd,
            &parent_roots,
            &reads_for_test(),
            &[work.join("sub"), base.join("work2")],
        )
        .expect_err("`work2` shares a prefix with `work` and is outside it")
        .to_string();
        assert!(
            refusal.contains(&base.join("work2").display().to_string())
                && refusal.contains("outside this session's workspace"),
            "the refusal must name the entry and the rule: {refusal}"
        );
        assert_eq!(
            accept_writable_roots(
                Permission::Workspace,
                &parent_cwd,
                &parent_roots,
                &reads_for_test(),
                &[work.join("sub"), base.join("missing")],
            )
            .expect_err("outside the workspace, whether or not it exists")
            .to_string(),
            refusal.replace(
                &base.join("work2").display().to_string(),
                &base.join("missing").display().to_string()
            ),
            "a missing directory outside the workspace is refused in the same words as one that \
             is there, so the refusal does not say which it is"
        );
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(base.join("work2"), work.join("link")).expect("symlink");
            let through_link = accept_writable_roots(
                Permission::Workspace,
                &parent_cwd,
                &parent_roots,
                &reads_for_test(),
                &[PathBuf::from("link/missing")],
            )
            .expect_err("a missing name behind a link out of the workspace is outside it")
            .to_string();
            assert!(
                through_link.contains("outside this session's workspace")
                    && through_link.contains(&work.join("link/missing").display().to_string())
                    && !through_link.contains(&base.join("work2").display().to_string()),
                "judged where the link leads, ahead of whether it exists, and named as spelled: \
                 {through_link}"
            );
        }

        let accepted = accept_writable_roots(
            Permission::Unrestricted,
            &parent_cwd,
            &parent_roots,
            &reads_for_test(),
            &[base.join("work2")],
        )
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

        let accepted = accept_writable_roots(
            Permission::Workspace,
            &parent_cwd,
            &parent_roots,
            &reads_for_test(),
            &[PathBuf::from("sub")],
        )
        .expect("a relative entry resolves against the parent's directory");
        assert_eq!(accepted.cwd, work.join("sub"));

        let refused = |requested: &[PathBuf]| {
            accept_writable_roots(
                Permission::Workspace,
                &parent_cwd,
                &parent_roots,
                &reads_for_test(),
                requested,
            )
            .expect_err("refused")
            .to_string()
        };
        assert!(refused(&[work.join("missing")]).contains("does not exist"));
        assert!(refused(&[file]).contains("is not a directory"));

        #[cfg(unix)]
        {
            let refusal = accept_writable_roots(
                Permission::Unrestricted,
                &parent_cwd,
                &parent_roots,
                &reads_for_test(),
                &[PathBuf::from("/tmp")],
            )
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

    /// A private directory is one that exists, is a directory, and is not already a system mask.
    /// The last case is what an override pointed at `/tmp` itself would produce: taking `/tmp` as
    /// private would mask every working directory under it.
    #[test]
    fn a_private_directory_is_an_existing_directory_that_is_not_a_system_mask() {
        let temp = tempfile::tempdir().expect("tempdir");
        let kept = canonical_for_test(temp.path());
        let file = temp.path().join("file");
        std::fs::write(&file, b"x").expect("write");
        let mut candidates = vec![temp.path().to_path_buf(), temp.path().join("missing"), file];
        // `cfg!` rather than `#[cfg]`, so the binding is mutated on every platform and Windows
        // does not see an unused `mut`.
        if cfg!(unix) {
            candidates.push(PathBuf::from("/tmp"));
        }

        assert_eq!(private_directories_among(candidates), vec![kept]);
    }

    /// A released lock is pruned on the next acquisition and a held one is not, so the map is
    /// bounded by the files being written concurrently rather than by every file ever touched.
    #[tokio::test]
    async fn a_released_write_lock_is_pruned_and_a_held_one_is_kept() {
        let locks = WriteLocks::default();
        let held = locks.lock_path(Path::new("/held")).await;
        let released = locks.lock_path(Path::new("/released")).await;
        drop(released);
        let _next = locks.lock_path(Path::new("/next")).await;

        let mut keys: Vec<PathBuf> = crate::sync::lock(&locks.by_path).keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, [PathBuf::from("/held"), PathBuf::from("/next")]);
        drop(held);
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
            let refusal = private_read_refusal_with(permission, store, &private, &[])
                .unwrap_or_else(|| panic!("{permission:?} must refuse the store"));
            assert!(refusal.contains("meka's own directory"), "{refusal}");
        }
        assert!(
            private_read_refusal_with(Permission::Unrestricted, store, &private, &[]).is_none(),
            "unrestricted reads it, as it writes it"
        );
        assert!(
            private_read_refusal_with(
                Permission::Read,
                Path::new("/home/someone/.local/share/other"),
                &private,
                &[]
            )
            .is_none()
        );
    }

    /// The skills store stays readable inside the private config directory: a skill's bundled
    /// files are what its body tells the model to read. The config file beside it does not.
    #[test]
    fn the_skills_store_is_readable_inside_the_private_config_directory() {
        use crate::permission::Permission;
        let config = PathBuf::from("/home/someone/.config/meka");
        let private = vec![config.clone()];
        let readable = vec![config.join("skills")];
        assert!(
            private_read_refusal_with(
                Permission::Read,
                &config.join("skills/deploy/scripts/helper.sh"),
                &private,
                &readable
            )
            .is_none(),
            "a bundled file is the model's to read"
        );
        assert!(
            private_read_refusal_with(
                Permission::Read,
                &config.join("config.toml"),
                &private,
                &readable
            )
            .is_some(),
            "the file beside the store stays private"
        );
    }

    /// A process's own `/proc` directory is refused below `unrestricted` by the same rule that
    /// refuses meka's directories, whether named by pid, `self` or `thread-self`; the system-wide
    /// files beside them are not, since they carry no secret and are the one way to read them at
    /// `read`.
    #[test]
    fn a_process_s_proc_entries_are_not_readable_below_unrestricted() {
        use crate::permission::Permission;
        let private = vec![PathBuf::from("/home/someone/.local/share/meka")];
        for entry in [
            "/proc/1234/environ",
            "/proc/1234/mem",
            "/proc/self/environ",
            "/proc/thread-self/cmdline",
            "/proc/42/fd/3",
        ] {
            let refusal =
                private_read_refusal_with(Permission::Read, Path::new(entry), &private, &[])
                    .unwrap_or_else(|| panic!("{entry} must be refused"));
            assert!(refusal.contains("`/proc` entry"), "{refusal}");
            assert!(
                private_read_refusal_with(Permission::Unrestricted, Path::new(entry), &private, &[
                ])
                .is_none(),
                "unrestricted reads {entry}"
            );
        }
        for system_wide in [
            "/proc/meminfo",
            "/proc/cpuinfo",
            "/proc/net/tcp",
            "/proc/mounts",
        ] {
            assert!(
                private_read_refusal_with(Permission::Read, Path::new(system_wide), &private, &[])
                    .is_none(),
                "{system_wide} carries no secret and stays readable"
            );
        }
        assert!(!is_process_proc_entry(Path::new("/procfs/1234/environ")));
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
        let level = crate::permission::Permission::Workspace;
        assert!(
            scope
                .admit_with_private(
                    level,
                    &cwd,
                    &root.join("notes.txt"),
                    &root.join("notes.txt"),
                    &private
                )
                .is_ok(),
            "a path beside the store is still inside the workspace"
        );
        let config = store.join("config.toml");
        let refusal = scope
            .admit_with_private(level, &cwd, &config, &config, &private)
            .expect_err("a path inside the store is refused although the root contains it");
        assert!(refusal.contains("meka's own directory"), "{refusal}");
        assert!(
            scope
                .admit_with_private(level, &cwd, &config, &config, &[])
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
            scope.confined_to_at(crate::permission::Permission::Workspace, &cwd),
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
                        scope.confined_to_at(level, &cwd),
                        None,
                        "{level} disclaims a boundary: a write reaches anywhere"
                    );
                    scope
                        .admit(&cwd, &outside)
                        .expect("an unconfined level admits a path outside every root");
                }
                _ => {
                    assert_eq!(
                        scope.confined_to_at(level, &cwd),
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
    /// `admit` is reached with a path that does not exist yet on every `file_write` to a new file,
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

    /// The single-root case has to stay exactly one path: that is every screen and HTTP session,
    /// and every ACP client that sends no extra roots.
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

    /// The read boundary: inside a root or a base path is admitted, outside is refused with the
    /// roots named, `unrestricted` disclaims it, and a scope without one admits everything but
    /// meka's own directories.
    #[test]
    fn a_confined_read_stays_within_the_roots_and_the_base_paths() {
        use crate::permission::{EnabledPermissions, Permission, SharedPermission};
        let temp = tempfile::tempdir().expect("tempdir");
        let work = canonical_for_test(temp.path()).join("work");
        let system = canonical_for_test(temp.path()).join("system");
        let elsewhere = canonical_for_test(temp.path()).join("elsewhere");
        for directory in [&work, &system, &elsewhere] {
            std::fs::create_dir(directory).expect("mkdir");
            std::fs::write(directory.join("file.txt"), "x").expect("write");
        }
        let cwd = SharedCwd::new(work.clone());
        let at = |level| SharedPermission::new(level, EnabledPermissions::ALL);

        let confined = ReadScope::confined(at(Permission::Read), vec![], vec![system.clone()]);
        assert!(confined.admit(&cwd, &work.join("file.txt")).is_ok());
        assert!(confined.admit(&cwd, &system.join("file.txt")).is_ok());
        let refusal = confined
            .admit(&cwd, &elsewhere.join("file.txt"))
            .expect_err("outside the roots");
        assert!(
            refusal.contains("outside the readable roots")
                && refusal.contains(&work.display().to_string())
                && refusal.contains("the system directories")
                && !refusal.contains(&system.display().to_string()),
            "the refusal names the session's root and the system directories in a word: {refusal}"
        );
        assert_eq!(
            confined.confined_to_at(Permission::Read, &cwd),
            Some(vec![work.clone(), system.clone()])
        );
        assert_eq!(
            confined.session_roots_at(Permission::Read, &cwd),
            Some(vec![work.clone()]),
            "the block names the session's roots alone"
        );

        let unrestricted =
            ReadScope::confined(at(Permission::Unrestricted), vec![], vec![system.clone()]);
        assert!(
            unrestricted
                .admit(&cwd, &elsewhere.join("file.txt"))
                .is_ok()
        );
        assert_eq!(
            unrestricted.confined_to_at(Permission::Unrestricted, &cwd),
            None
        );

        let unconfined = ReadScope::unconfined(at(Permission::Read));
        assert!(unconfined.admit(&cwd, &elsewhere.join("file.txt")).is_ok());
        assert_eq!(unconfined.confined_to_at(Permission::Read, &cwd), None);
    }

    /// A symlink inside a root that points out of it resolves to where it lands, and is refused
    /// there; a walk that reaches it through the fence steps around it.
    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_a_readable_root_is_refused_by_the_fence() {
        use crate::permission::{EnabledPermissions, Permission, SharedPermission};
        let temp = tempfile::tempdir().expect("tempdir");
        let work = canonical_for_test(temp.path()).join("work");
        let elsewhere = canonical_for_test(temp.path()).join("elsewhere");
        std::fs::create_dir_all(&work).expect("mkdir");
        std::fs::create_dir_all(&elsewhere).expect("mkdir");
        std::fs::write(elsewhere.join("secret"), "x").expect("write");
        std::os::unix::fs::symlink(elsewhere.join("secret"), work.join("link")).expect("link");
        let cwd = SharedCwd::new(work.clone());
        let scope = ReadScope::confined(
            SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            vec![],
            vec![],
        );
        let fence = scope.fence(&cwd);
        let through_link = fence
            .refusal(&work.join("link"))
            .expect("the link lands outside");
        assert!(
            through_link.contains(&work.join("link").display().to_string())
                && !through_link.contains(&elsewhere.display().to_string()),
            "the refusal names the link as spelled, not where it leads: {through_link}"
        );
        assert!(!fence.hides(&work), "the root itself is inside");
        assert!(
            !fence.hides(&work.join("missing")),
            "a path that does not exist inside the roots is admitted; the read of it fails on \
             its own"
        );
        assert_eq!(
            fence.refusal(&elsewhere.join("missing")),
            fence
                .refusal(&elsewhere.join("secret"))
                .map(|refusal| refusal.replace(
                    &elsewhere.join("secret").display().to_string(),
                    &elsewhere.join("missing").display().to_string()
                )),
            "a path outside the roots is refused in the same words whether or not it exists"
        );
        std::os::unix::fs::symlink(&elsewhere, work.join("door")).expect("link");
        assert!(
            fence.hides(&work.join("door").join("missing")),
            "a missing name behind a link out of the roots is judged where the link leads"
        );
    }

    /// The boundary keeps the paths that exist, in both spellings, and drops a configured entry
    /// that is missing, relative, or a directory the sandbox masks; a skill root comes along the
    /// same way; and the base is resolved whether or not the switch is on, so a sub-agent a spawn
    /// confines has the platform's paths.
    #[test]
    fn the_read_boundary_takes_only_paths_that_resolve() {
        let temp = tempfile::tempdir().expect("tempdir");
        let present = canonical_for_test(temp.path()).join("present");
        let skills = canonical_for_test(temp.path()).join("skills");
        std::fs::create_dir_all(&present).expect("mkdir");
        std::fs::create_dir_all(&skills).expect("mkdir");
        let missing = present.join("not-there");
        // `.` resolves against the process's directory, so only the relative rule drops it; `/tmp`
        // is a masked directory on Unix alone.
        let readable: Vec<PathBuf> = [present.clone(), missing.clone(), PathBuf::from(".")]
            .into_iter()
            .chain(cfg!(unix).then(|| PathBuf::from("/tmp")))
            .collect();
        let boundary = ReadBoundary::new(true, &readable, &[skills.clone(), skills.join("absent")]);
        assert!(boundary.confined);
        assert!(boundary.base.contains(&present));
        assert!(boundary.base.contains(&skills), "a skill root is readable");
        assert!(!boundary.base.contains(&missing));
        assert!(!boundary.base.contains(&skills.join("absent")));
        let process_directory = std::env::current_dir()
            .and_then(std::fs::canonicalize)
            .map(strip_verbatim)
            .expect("current directory");
        assert!(
            !boundary.base.iter().any(|path| path.is_relative())
                && !boundary.base.contains(&process_directory),
            "a relative entry is dropped in both spellings: {:?}",
            boundary.base
        );
        #[cfg(unix)]
        assert!(
            !boundary.base.contains(&PathBuf::from("/tmp"))
                && !boundary.base.contains(&PathBuf::from("/private/tmp")),
            "a masked directory is dropped, or the bind would hand its sockets back: {:?}",
            boundary.base
        );
        for path in SYSTEM_READ_PATHS {
            let path = Path::new(path);
            // Both spellings of an existing path: the name a bind mount wants and the target a
            // rule wants.
            assert_eq!(
                boundary.base.contains(&path.to_path_buf()),
                path.exists(),
                "{}",
                path.display()
            );
            assert_eq!(
                boundary.base.contains(
                    &std::fs::canonicalize(path)
                        .map(strip_verbatim)
                        .unwrap_or_default()
                ),
                path.exists(),
                "{}",
                path.display()
            );
        }
        let off = ReadBoundary::new(false, &readable, std::slice::from_ref(&skills));
        assert!(!off.confined);
        assert_eq!(
            off.base, boundary.base,
            "the base is the same with the switch off, for a sub-agent a spawn confines"
        );
    }

    /// A spawn turns read confinement on and never off: a confined parent's worker is confined
    /// whatever the call says, an unconfined parent's only when the call grants readable roots.
    #[test]
    fn a_spawn_turns_read_confinement_on_and_never_off() {
        let base = vec![PathBuf::from("/usr")];
        let on = ReadBoundary {
            confined: true,
            base: base.clone(),
        };
        let off = ReadBoundary {
            confined: false,
            base: base.clone(),
        };
        for granted in [false, true] {
            assert!(on.for_worker(granted).confined, "granted {granted}");
            assert_eq!(on.for_worker(granted).base, base);
        }
        assert!(!off.for_worker(false).confined);
        assert!(off.for_worker(true).confined);
        assert_eq!(off.for_worker(true).base, base);
    }

    /// A worker's directory is judged against the parent's readable reach at every level below
    /// `unrestricted`, not only against its write reach at `workspace`: a parent at `read` whose
    /// reads are confined cannot resume a worker in a directory it can no longer read.
    #[test]
    fn a_workers_directory_is_admitted_within_the_parents_readable_reach() {
        use crate::permission::{EnabledPermissions, Permission, SharedPermission};
        let temp = tempfile::tempdir().expect("tempdir");
        let work = canonical_for_test(temp.path()).join("work");
        let elsewhere = canonical_for_test(temp.path()).join("elsewhere");
        std::fs::create_dir_all(work.join("sub")).expect("mkdir");
        std::fs::create_dir_all(&elsewhere).expect("mkdir");
        let cwd = SharedCwd::new(work.clone());
        let roots = SharedRoots::default();
        let at = |level| SharedPermission::new(level, EnabledPermissions::ALL);

        let admit = |level, scope: &ReadScope, directory: &Path| {
            admit_within_parent_reach(level, &cwd, &roots, scope, directory, directory)
        };

        let confined = ReadScope::confined(at(Permission::Read), vec![], vec![]);
        assert!(admit(Permission::Read, &confined, &work.join("sub")).is_ok());
        let refusal =
            admit(Permission::Read, &confined, &elsewhere).expect_err("outside the readable reach");
        assert!(
            refusal.contains("outside this session's readable roots")
                && refusal.contains(&work.display().to_string()),
            "{refusal}"
        );
        assert_eq!(
            admit(Permission::Read, &confined, &elsewhere.join("missing"))
                .expect_err("outside the readable reach, whether or not it exists"),
            refusal.replace(
                &elsewhere.display().to_string(),
                &elsewhere.join("missing").display().to_string()
            ),
            "a missing directory outside the reach is refused in the same words as one that is \
             there, so the refusal does not say which it is"
        );
        assert!(
            admit(Permission::Unrestricted, &confined, &elsewhere).is_ok(),
            "`unrestricted` reaches everywhere"
        );

        let open = ReadScope::unconfined(at(Permission::Read));
        assert!(
            admit(Permission::Read, &open, &elsewhere).is_ok(),
            "a parent reading everywhere bounds nothing at `read`"
        );
        let refusal =
            admit(Permission::Workspace, &open, &elsewhere).expect_err("outside the write reach");
        assert!(
            refusal.contains("outside this session's workspace"),
            "{refusal}"
        );
        assert_eq!(
            admit(Permission::Workspace, &open, &elsewhere.join("missing"))
                .expect_err("outside the write reach, whether or not it exists"),
            refusal.replace(
                &elsewhere.display().to_string(),
                &elsewhere.join("missing").display().to_string()
            )
        );
    }

    /// The readable roots fold in the writable roots, the read-only roots and the base set, each
    /// canonical, with a root already covered by another dropped.
    #[test]
    fn readable_roots_fold_every_kind_of_root_in() {
        let temp = tempfile::tempdir().expect("tempdir");
        let top = canonical_for_test(temp.path());
        let work = top.join("work");
        let nested = work.join("nested");
        let docs = top.join("docs");
        for directory in [&work, &nested, &docs] {
            std::fs::create_dir_all(directory).expect("mkdir");
        }
        let roots = readable_roots(
            &SharedCwd::new(work.clone()),
            &SharedRoots::new(vec![nested]),
            &SharedRoots::new(vec![docs.clone()]),
            &[top.join("system")],
        );
        assert_eq!(roots, vec![work, docs, top.join("system")]);
    }

    /// A parent's grant of read-only roots: inside its own readable reach when its reads are
    /// confined, anywhere that exists when they are not, never a masked system directory, and
    /// never an empty list.
    #[test]
    fn readable_roots_are_admitted_within_the_parents_reach() {
        use crate::permission::{EnabledPermissions, Permission, SharedPermission};
        let temp = tempfile::tempdir().expect("tempdir");
        let base = canonical_for_test(temp.path());
        let work = base.join("work");
        let docs = work.join("docs");
        let inner = docs.join("inner");
        let elsewhere = base.join("elsewhere");
        let own = base.join("own");
        for directory in [&inner, &elsewhere, &own] {
            std::fs::create_dir_all(directory).expect("mkdir");
        }
        let cwd = SharedCwd::new(work.clone());
        let at = |level| SharedPermission::new(level, EnabledPermissions::ALL);
        // A bounded worker's workspace, beside everything the list names.
        let bounded = vec![own.clone()];

        let confined = ReadScope::confined(at(Permission::Read), vec![], vec![]);
        assert_eq!(
            accept_readable_roots(Permission::Read, &cwd, &confined, &bounded, &[
                PathBuf::from("docs"),
                docs.clone()
            ])
            .expect("inside the reach"),
            vec![docs.clone()],
            "relative entries resolve against the parent's directory, and a repeat is one root"
        );
        let refusal = accept_readable_roots(
            Permission::Read,
            &cwd,
            &confined,
            &bounded,
            std::slice::from_ref(&elsewhere),
        )
        .expect_err("outside the reach");
        assert!(
            refusal
                .to_string()
                .contains("outside this session's readable roots"),
            "{refusal}"
        );
        assert_eq!(
            accept_readable_roots(Permission::Read, &cwd, &confined, &bounded, &[
                elsewhere.join("missing")
            ])
            .expect_err("outside the reach, whether or not it exists")
            .to_string(),
            refusal.to_string().replace(
                &elsewhere.display().to_string(),
                &elsewhere.join("missing").display().to_string()
            ),
            "a missing directory outside the reach is refused in the same words as one that is \
             there, so the refusal does not say which it is"
        );
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&elsewhere, work.join("link")).expect("symlink");
            let through_link =
                accept_readable_roots(Permission::Read, &cwd, &confined, &bounded, &[
                    PathBuf::from("link"),
                ])
                .expect_err("the link lands outside the reach")
                .to_string();
            assert!(
                through_link.contains(&work.join("link").display().to_string())
                    && !through_link.contains(&elsewhere.display().to_string()),
                "the refusal names the link as spelled, not where it leads: {through_link}"
            );
            assert!(
                accept_readable_roots(Permission::Read, &cwd, &confined, &bounded, &[
                    PathBuf::from("link/missing")
                ])
                .expect_err("a missing name behind the link is outside the reach")
                .to_string()
                .contains("outside this session's readable roots"),
                "judged where the link leads, ahead of whether it exists"
            );
        }
        assert!(
            accept_readable_roots(Permission::Read, &cwd, &confined, &bounded, &[])
                .expect_err("an empty list is refused, as an empty writable_roots is")
                .to_string()
                .contains("names no directory")
        );
        assert!(
            accept_readable_roots(Permission::Read, &cwd, &confined, &bounded, &[
                work.join("missing")
            ])
            .is_err(),
            "a directory that is not there grants nothing"
        );

        let open = ReadScope::unconfined(at(Permission::Read));
        assert_eq!(
            accept_readable_roots(
                Permission::Read,
                &cwd,
                &open,
                &bounded,
                std::slice::from_ref(&elsewhere)
            )
            .expect("a parent reading everywhere may name anywhere"),
            vec![elsewhere]
        );
        #[cfg(unix)]
        assert!(
            accept_readable_roots(Permission::Read, &cwd, &open, &bounded, &[PathBuf::from(
                "/tmp"
            )])
            .expect_err("a masked system directory")
            .to_string()
            .contains("system directory")
        );

        // The worker's workspace is readable already, so an entry inside it, the workspace
        // itself included, is refused with the spelling that confines a worker to a directory;
        // one that contains the workspace widens the worker's reads and is accepted.
        let shared = vec![work.clone()];
        let refusal = accept_readable_roots(
            Permission::Read,
            &cwd,
            &open,
            &shared,
            std::slice::from_ref(&docs),
        )
        .expect_err("inside the shared workspace");
        assert!(
            refusal
                .to_string()
                .contains("inside the sub-agent's workspace at")
                && refusal.to_string().contains("`writable_roots`"),
            "{refusal}"
        );
        assert!(
            accept_readable_roots(
                Permission::Read,
                &cwd,
                &open,
                &shared,
                std::slice::from_ref(&work)
            )
            .is_err(),
            "the workspace itself"
        );
        assert!(
            accept_readable_roots(
                Permission::Read,
                &cwd,
                &open,
                std::slice::from_ref(&docs),
                std::slice::from_ref(&inner)
            )
            .is_err(),
            "inside a bounded worker's own root"
        );
        assert_eq!(
            accept_readable_roots(
                Permission::Read,
                &cwd,
                &open,
                &shared,
                std::slice::from_ref(&base)
            )
            .expect("a directory containing the workspace widens it"),
            vec![base.clone()]
        );
    }
}
