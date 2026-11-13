//! The one door for every process Meka starts, and what each is handed on the way out: a place
//! behind Meka in the line for memory.
//!
//! A command the model runs (an `ffmpeg`, a build, a script with a leak) can take every byte the
//! machine or the service's cgroup has, and which process the operating system then takes is
//! decided by scores and caps the parent sets before the child runs. Set at one door, a new spawn
//! site cannot forget them; `tests/layering.rs` lists every file that starts a process and how
//! this door reaches it. What the door sets is best effort on every platform: a child that could
//! not be placed runs anyway, since refusing the command would not save Meka from the next one.
//!
//! - Linux: the child's `oom_score_adj` is raised to the ceiling between `fork` and `exec`. The
//!   value is inherited, so the shell, the sandbox around it and everything either starts carry it,
//!   and the kernel's OOM killer takes any of them ahead of Meka in a cgroup and host-wide. A
//!   `pre_exec` step costs std's `posix_spawn` fast path, so every spawn forks: `posix_spawn` has
//!   no attribute for the write, and a write from the parent after the spawn would miss a
//!   grandchild forked before it.
//! - Windows: the child is placed in a job object whose committed memory stops at the machine's
//!   physical memory, so past it the command's own allocations fail and Meka's never do, the only
//!   way to lose first on a platform without an OOM killer. A job the door hands back ends with its
//!   handle, so dropping the child ends every process it started.
//! - macOS: no lever exists for one process over another; the system takes the largest footprint
//!   first, which is the runaway command.
//!
//! A command Meka runs and waits for (`shell_execute`, the screen's `!command`, a schedule gate's
//! shell probe, an MCP server's `headers_helper`) is started as the root of a [`Tree`] of its own
//! and run by [`run`]: its output read under a [`Keep`] rule per stream, and the tree killed on a
//! stop, a timeout, an exhausted budget or a drop.

use std::{
    io,
    ops::{Deref, DerefMut},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

/// A command the door can start: tokio's and the standard library's.
pub(crate) trait Spawnable {
    /// What the spawn hands back.
    type Child;

    /// The standard-library command under this one, where the child-side steps are registered.
    fn standard(&mut self) -> &mut std::process::Command;

    /// Start the child.
    fn spawn(&mut self) -> io::Result<Self::Child>;

    /// The child's process handle, `None` once the child has been reaped.
    #[cfg(windows)]
    fn handle(child: &Self::Child) -> Option<std::os::windows::io::RawHandle>;
}

impl Spawnable for tokio::process::Command {
    type Child = tokio::process::Child;

    fn standard(&mut self) -> &mut std::process::Command {
        self.as_std_mut()
    }

    fn spawn(&mut self) -> io::Result<Self::Child> {
        tokio::process::Command::spawn(self)
    }

    #[cfg(windows)]
    fn handle(child: &Self::Child) -> Option<std::os::windows::io::RawHandle> {
        child.raw_handle()
    }
}

impl Spawnable for std::process::Command {
    type Child = std::process::Child;

    fn standard(&mut self) -> &mut std::process::Command {
        self
    }

    fn spawn(&mut self) -> io::Result<Self::Child> {
        std::process::Command::spawn(self)
    }

    #[cfg(windows)]
    fn handle(child: &Self::Child) -> Option<std::os::windows::io::RawHandle> {
        Some(std::os::windows::io::AsRawHandle::as_raw_handle(child))
    }
}

/// A child the door started. Dereferences to the child itself; on Windows it also owns the job
/// the child runs in, which ends the child and everything it started when this value is dropped.
pub(crate) struct Spawned<Child> {
    inner: Child,
    #[cfg(windows)]
    job: Option<std::os::windows::io::OwnedHandle>,
}

impl<Child> Deref for Spawned<Child> {
    type Target = Child;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Child> DerefMut for Spawned<Child> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(windows)]
impl<Child> Spawned<Child> {
    /// End the child and every process it started, through the job. `Err` when the child runs in
    /// no job, which the spawn logged, or when the job refuses.
    pub(crate) fn terminate_tree(&self) -> io::Result<()> {
        use std::os::windows::io::AsRawHandle;

        let Some(job) = &self.job else {
            return Err(io::Error::other("the child runs in no job"));
        };
        // SAFETY: `job` is an open job handle this value owns.
        if unsafe {
            windows_sys::Win32::System::JobObjects::TerminateJobObject(job.as_raw_handle(), 1)
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Start `command` with the door's placement applied. Windows only, where a tree's root needs no
/// step in the child; on Unix every door goes through `spawn_with_pre_exec`.
#[cfg(windows)]
pub(crate) fn spawn<C: Spawnable>(command: &mut C) -> io::Result<Spawned<C::Child>> {
    prepare(command);
    finish(command)
}

/// Start `command` with the door's placement applied and `pre_exec` run in the child after it,
/// so a step that restricts the child (`setsid`, a Landlock domain, which refuses the `/proc`
/// write the placement makes) comes second.
///
/// # Safety
///
/// `pre_exec`'s contract is [`std::os::unix::process::CommandExt::pre_exec`]'s: it runs after
/// `fork` in a child that may hold a lock another thread had, so it may only make
/// async-signal-safe calls and must allocate nothing.
#[cfg(unix)]
pub(crate) unsafe fn spawn_with_pre_exec<C, F>(
    command: &mut C,
    pre_exec: F,
) -> io::Result<Spawned<C::Child>>
where
    C: Spawnable,
    F: FnMut() -> io::Result<()> + Send + Sync + 'static,
{
    prepare(command);
    // SAFETY: the caller's obligation, stated above, is the one `pre_exec` requires.
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(command.standard(), pre_exec);
    }
    finish(command)
}

/// The half of the placement that is set on the command before it starts, for a door where
/// another crate makes the spawn; [`adopt`] is the other half. A door this crate spawns for itself
/// goes through [`spawn_tree`] or `spawn_with_pre_exec`, which do both.
pub(crate) fn prepare<C: Spawnable>(command: &mut C) {
    prepare_standard(command.standard());
}

/// The half of the placement that needs the running child, by its process id. On Windows the
/// child is capped at the machine's memory but not tied to a handle, since this door holds none:
/// the crate that spawned it ends it.
#[cfg(windows)]
pub(crate) fn adopt(process_id: u32) {
    drop(cap_by_process_id(process_id));
}

/// See the Windows definition: nothing is left to do once the child runs.
#[cfg(not(windows))]
pub(crate) fn adopt(_process_id: u32) {}

fn finish<C: Spawnable>(command: &mut C) -> io::Result<Spawned<C::Child>> {
    let inner = command.spawn()?;
    // Placed once the child runs: a grandchild it starts in the microseconds before the
    // assignment escapes the job, which the confined command's suspended spawn avoids and tokio
    // gives no thread handle to repeat here.
    #[cfg(windows)]
    let job = C::handle(&inner).and_then(|handle| place_in_job(handle, JobEnd::WithHandle));
    Ok(Spawned {
        inner,
        #[cfg(windows)]
        job,
    })
}

/// The kernel's `oom_score_adj` ceiling, as the file takes it.
#[cfg(target_os = "linux")]
const OOM_SCORE_ADJ_MAX: &[u8] = b"1000";

#[cfg(target_os = "linux")]
fn prepare_standard(command: &mut std::process::Command) {
    // SAFETY: the closure runs after `fork` in the child and makes three async-signal-safe calls
    // with static arguments, allocating nothing. Its failure is not reported: nothing in the child
    // can report one, and a spawn must not fail for it.
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(command, || {
            let descriptor = libc::open(
                c"/proc/self/oom_score_adj".as_ptr(),
                libc::O_WRONLY | libc::O_CLOEXEC,
            );
            if descriptor >= 0 {
                libc::write(
                    descriptor,
                    OOM_SCORE_ADJ_MAX.as_ptr().cast(),
                    OOM_SCORE_ADJ_MAX.len(),
                );
                libc::close(descriptor);
            }
            Ok(())
        });
    }
}

/// See the Linux definition: nothing is set on the command elsewhere.
#[cfg(not(target_os = "linux"))]
fn prepare_standard(_command: &mut std::process::Command) {}

/// What ends a job: the last handle to it closing, which is how a child the door hands back is
/// torn down with everything it started, or its last process exiting, for a job nobody holds.
#[cfg(windows)]
#[derive(Clone, Copy)]
pub(crate) enum JobEnd {
    WithHandle,
    WithLastProcess,
}

/// A job for one child, capped at the machine's physical memory. The cap is left off when the
/// machine's memory cannot be read, which is logged.
#[cfg(windows)]
pub(crate) fn open_job(end: JobEnd) -> io::Result<std::os::windows::io::OwnedHandle> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};

    use windows_sys::Win32::System::JobObjects::{
        CreateJobObjectW, JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    // SAFETY: a null name and null attributes make an unnamed job this process owns.
    let raw = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if raw.is_null() {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the handle was just returned to this process and nothing else owns it.
    let job = unsafe { OwnedHandle::from_raw_handle(raw) };

    // SAFETY: an all-zero limit block is a valid one with no limit set.
    let mut information: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
    if matches!(end, JobEnd::WithHandle) {
        information.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    }
    match physical_memory_bytes() {
        Ok(bytes) => {
            information.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
            information.JobMemoryLimit = bytes;
        }
        Err(error) => {
            static ANNOUNCED: AtomicBool = AtomicBool::new(false);
            announce_once(
                &ANNOUNCED,
                &format!(
                    "failed to read the machine's memory: {error}; a command is not capped at it"
                ),
            );
        }
    }
    // SAFETY: `information` is a complete, correctly sized block that outlives the call.
    if unsafe {
        SetInformationJobObject(
            job.as_raw_handle(),
            JobObjectExtendedLimitInformation,
            (&raw const information).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(job)
}

/// The machine's physical memory.
#[cfg(windows)]
fn physical_memory_bytes() -> io::Result<usize> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    // SAFETY: an all-zero status block is valid once its length is set, which is what the call
    // reads first.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is a sized block that outlives the call.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Err(io::Error::last_os_error());
    }
    usize::try_from(status.ullTotalPhys).map_err(io::Error::other)
}

/// Place the process behind `handle` in a job of its own, logging a refusal rather than failing:
/// the command runs either way.
#[cfg(windows)]
fn place_in_job(
    handle: std::os::windows::io::RawHandle,
    end: JobEnd,
) -> Option<std::os::windows::io::OwnedHandle> {
    use std::os::windows::io::AsRawHandle;

    let assigned = open_job(end).and_then(|job| {
        // SAFETY: both handles are open; the process handle carries the quota and terminate
        // rights, which is what the job needs of it.
        if unsafe {
            windows_sys::Win32::System::JobObjects::AssignProcessToJobObject(
                job.as_raw_handle(),
                handle,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(job)
    });
    match assigned {
        Ok(job) => Some(job),
        Err(error) => {
            static ANNOUNCED: AtomicBool = AtomicBool::new(false);
            announce_once(
                &ANNOUNCED,
                &format!(
                    "failed to place a child in a job: {error}; a command that takes every byte \
                     can take Meka with it"
                ),
            );
            None
        }
    }
}

/// Cap the process with `process_id` at the machine's memory, in a job that is not tied to the
/// handle returned. The id names that child and no other: the crate that spawned it holds its
/// process handle for as long as the transport lives, and an open handle keeps a pid from being
/// reused.
#[cfg(windows)]
fn cap_by_process_id(process_id: u32) -> Option<std::os::windows::io::OwnedHandle> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};

    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    // SAFETY: a plain lookup by id; a null return is the error.
    let raw = unsafe { OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, process_id) };
    if raw.is_null() {
        static ANNOUNCED: AtomicBool = AtomicBool::new(false);
        let error = io::Error::last_os_error();
        announce_once(
            &ANNOUNCED,
            &format!(
                "failed to open child {process_id} to place it in a job: {error}; a command that \
                 takes every byte can take Meka with it"
            ),
        );
        return None;
    }
    // SAFETY: the handle was just opened by this process and nothing else owns it.
    let process = unsafe { OwnedHandle::from_raw_handle(raw) };
    place_in_job(process.as_raw_handle(), JobEnd::WithLastProcess)
}

/// Say `message` once per process, then only under `-vv`: a host where every spawn is refused
/// would otherwise warn on every command. `announced` is the site's own flag, so one refusal does
/// not silence a different one.
#[cfg(windows)]
fn announce_once(announced: &AtomicBool, message: &str) {
    if announced.swap(true, Ordering::Relaxed) {
        tracing::debug!("{message}");
    } else {
        tracing::warn!("{message}");
    }
}

/// How long a tree's processes are given to end on `SIGTERM` before `SIGKILL` lands.
#[cfg(unix)]
const KILL_GRACE: Duration = Duration::from_millis(200);

/// A child that leads a tree of its own, which ends with it: everything the child starts, and
/// everything they start. On Unix the child leads a session of its own (`setsid`), a process group
/// with no controlling terminal, so `-pid` names the tree, backgrounded descendants such as
/// `(sleep 3600 &)` included; on Windows the job the door placed it in ends the tree with its
/// handle.
///
/// [`run`] kills the tree on a stop, a timeout and an exhausted budget, but a caller that drops the
/// future never reaches those arms: a tool gate's `tokio::time::timeout` drops a `shell_execute`
/// call and only then fires its token, so without the kill on drop every evaluation of such a gate
/// would leave its command running. `Drop` is synchronous, so there is no grace signal there; a
/// child already reaped has no id and is not signaled.
pub(crate) struct Tree(Spawned<tokio::process::Child>);

impl Tree {
    /// The child's process id, `None` once it has been reaped.
    pub(crate) fn id(&self) -> Option<u32> {
        self.0.id()
    }

    /// End the tree: on Unix `SIGTERM` to the group, a moment for well-behaved processes to shut
    /// down cleanly, then `SIGKILL`; on Windows the job is terminated.
    async fn kill(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.0.id() {
            let pgid = pid as libc::pid_t;
            // SAFETY: `kill(2)` is always safe to call; it just returns an error if the target is
            // gone. Sending to `-pgid` targets the whole process group. Errors here usually mean
            // the group already exited; log at debug so an unkillable group still leaves a trail
            // without spamming default verbosity.
            let term_result = unsafe { libc::kill(-pgid, libc::SIGTERM) };
            if term_result != 0 {
                let error = io::Error::last_os_error();
                tracing::debug!("failed to send SIGTERM to process group {pgid}: {error}");
            }
            tokio::time::sleep(KILL_GRACE).await;
        }
        self.kill_at_once().await;
    }

    /// End the tree with no grace: `SIGKILL` to the group on Unix, the job terminated on Windows.
    /// The closing `kill` is a no-op once the tree is gone and reaps the child either way.
    async fn kill_at_once(&mut self) {
        #[cfg(windows)]
        if let Err(error) = self.0.terminate_tree() {
            tracing::debug!("failed to terminate the command's job: {error}");
        }
        #[cfg(unix)]
        if let Some(pid) = self.0.id() {
            kill_tree_now(pid);
        }
        if let Err(error) = self.0.kill().await {
            tracing::debug!("failed to kill child process: {error}");
        }
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let Some(pid) = self.0.id() else {
            return;
        };
        kill_tree_now(pid);
        if let Err(error) = self.0.start_kill() {
            tracing::debug!("failed to kill child {pid} on drop: {error}");
        }
    }
}

/// Kill the tree whose root has process id `process_id` with `SIGKILL`, at once: what a dropped
/// [`Tree`] does, and what a holder that cannot wait on the task owning the `Tree` calls, such as a
/// screen on its way out, which may not poll that task again.
#[cfg(unix)]
pub(crate) fn kill_tree_now(process_id: u32) {
    // SAFETY: `kill(2)` has no preconditions; the root leads its own session, so `-pid` names its
    // tree and nothing else.
    let killed = unsafe { libc::kill(-(process_id as libc::pid_t), libc::SIGKILL) };
    if killed != 0 {
        let error = io::Error::last_os_error();
        tracing::debug!("failed to kill process group {process_id}: {error}");
    }
}

/// See the Unix definition: on Windows the job ends the tree when the [`Tree`] holding it drops,
/// as it does when Meka's own process ends.
#[cfg(windows)]
pub(crate) fn kill_tree_now(_process_id: u32) {}

/// Start `command` with the door's placement applied, as the root of a [`Tree`].
pub(crate) fn spawn_tree(command: &mut tokio::process::Command) -> io::Result<Tree> {
    // SAFETY: the caller's step does nothing, so the child makes the door's own calls alone.
    #[cfg(unix)]
    let spawned = unsafe { spawn_tree_with_pre_exec(command, || Ok(())) };
    #[cfg(windows)]
    let spawned = spawn(command).map(Tree);
    spawned
}

/// [`spawn_tree`], with `pre_exec` run in the child once the door's own steps have: after the
/// placement and the `setsid` that gives the child its tree, so a step that restricts the child (a
/// Landlock domain) comes last.
///
/// # Safety
///
/// As for [`spawn_with_pre_exec`].
#[cfg(unix)]
pub(crate) unsafe fn spawn_tree_with_pre_exec<F>(
    command: &mut tokio::process::Command,
    mut pre_exec: F,
) -> io::Result<Tree>
where
    F: FnMut() -> io::Result<()> + Send + Sync + 'static,
{
    // SAFETY: `setsid(2)` is async-signal-safe and has no precondition beyond the caller not
    // leading a process group already, which a freshly forked child never does; `pre_exec` carries
    // the caller's obligation.
    let spawned = unsafe {
        spawn_with_pre_exec(command, move || {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            pre_exec()
        })
    };
    spawned.map(Tree)
}

/// Which of a command's two output streams a chunk was read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stream {
    Stdout,
    Stderr,
}

/// Where a drain sends each chunk as soon as it has read it, beside what it keeps. The drain waits
/// for the sink before reading on, so a slow sink slows the command rather than queueing its
/// output.
#[async_trait]
pub(crate) trait Sink: Send + Sync {
    /// Take `chunk`, the next bytes read from `stream`.
    async fn receive(&self, stream: Stream, chunk: &[u8]);
}

/// How much a command may print before it is stopped, spent by the drains that draw on it.
///
/// Shared rather than split per stream when both streams draw on it, so a command that prints
/// everything on one stream gets the whole bound, and one that splits its output cannot double it.
pub(crate) struct OutputBudget {
    remaining: AtomicUsize,
    exhausted: AtomicBool,
}

impl OutputBudget {
    /// A budget of `bytes`.
    pub(crate) fn new(bytes: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(bytes),
            exhausted: AtomicBool::new(false),
        }
    }

    /// Spend `bytes`. `false` when they did not fit, at which point the budget is exhausted and
    /// stays so; the chunk that crossed the line is kept, so the record ends on a whole read.
    pub(crate) fn spend(&self, bytes: usize) -> bool {
        let before = self
            .remaining
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                Some(remaining.saturating_sub(bytes))
            })
            .unwrap_or(0);
        if before >= bytes {
            return true;
        }
        self.exhausted.store(true, Ordering::SeqCst);
        false
    }

    /// Whether a spend did not fit.
    pub(crate) fn is_exhausted(&self) -> bool {
        self.exhausted.load(Ordering::SeqCst)
    }
}

/// What a drain keeps of what it reads from its pipe.
pub(crate) enum Keep {
    /// Every byte, spent from a budget a command's streams may share. The read that crosses it is
    /// kept, so the capture ends on a whole read, and ends the command: the drain stops reading
    /// and [`run`] kills the tree.
    Budgeted(Arc<OutputBudget>),
    /// The first this many bytes. The rest is read to the pipe's end and let go, since closing the
    /// pipe at the cap would have the child's next write take `SIGPIPE`.
    Head(usize),
    /// The last this many bytes, with a count of what was let go ahead of them.
    Tail(usize),
}

/// What a drain kept of one stream.
#[derive(Default)]
pub(crate) struct Captured {
    pub(crate) bytes: Vec<u8>,
    /// Bytes read and let go: past a [`Keep::Head`], or ahead of a [`Keep::Tail`].
    pub(crate) dropped: u64,
}

/// What [`run`] keeps of each of a command's streams, and where it sends each chunk as it reads it.
pub(crate) struct Capture {
    pub(crate) stdout: Keep,
    pub(crate) stderr: Keep,
    pub(crate) sink: Option<Arc<dyn Sink>>,
}

/// A command's two drains, reading from the moment they start.
///
/// Started before the wait, because waiting on a child does not read its pipes: a child writing
/// past a pipe's buffer (about 64 KiB) blocks in `write`, never exits, and the wait sits out its
/// timeout. Once every process holding a pipe's write end has exited, the drain reads to its end.
pub(crate) struct Drains {
    stdout: tokio::task::JoinHandle<Captured>,
    stderr: tokio::task::JoinHandle<Captured>,
    /// Tells a drain to stop reading and return what it has.
    stop: CancellationToken,
    /// Stops both drains when they are let go unjoined, so a run whose future is dropped leaves no
    /// drain reading a pipe that something outside the tree still holds.
    stop_on_drop: tokio_util::sync::DropGuard,
    /// Fired by a drain whose budget ran out.
    bound: CancellationToken,
}

impl Drains {
    /// Start a drain on each pipe, keeping what `capture` says.
    pub(crate) fn start<StdoutPipe, StderrPipe>(
        stdout: Option<StdoutPipe>,
        stderr: Option<StderrPipe>,
        capture: Capture,
    ) -> Self
    where
        StdoutPipe: tokio::io::AsyncRead + Unpin + Send + 'static,
        StderrPipe: tokio::io::AsyncRead + Unpin + Send + 'static,
    {
        let stop = CancellationToken::new();
        let bound = CancellationToken::new();
        let stdout = tokio::spawn(drain(
            stdout,
            Stream::Stdout,
            capture.stdout,
            capture.sink.clone(),
            stop.clone(),
            bound.clone(),
        ));
        let stderr = tokio::spawn(drain(
            stderr,
            Stream::Stderr,
            capture.stderr,
            capture.sink,
            stop.clone(),
            bound.clone(),
        ));
        Self {
            stdout,
            stderr,
            stop_on_drop: stop.clone().drop_guard(),
            stop,
            bound,
        }
    }

    /// Wait until a drain's budget has run out.
    pub(crate) async fn bound_reached(&self) {
        self.bound.cancelled().await;
    }

    /// Both drains' captures, waiting at most `drain_timeout` in all for the pipes to close.
    ///
    /// A drain that has not reached its pipe's end by then is told to stop and returns what it
    /// has, so a process the command left behind holding the pipe delays the result by the timeout
    /// and costs the output it prints afterwards, never the output already read.
    pub(crate) async fn collect(self, drain_timeout: Duration) -> Drained {
        let Self {
            mut stdout,
            mut stderr,
            stop,
            stop_on_drop,
            bound: _,
        } = self;
        let deadline = tokio::time::Instant::now() + drain_timeout;
        let mut stopped_early = false;
        let stdout = join_drain_by(&mut stdout, deadline, &stop, &mut stopped_early).await;
        let stderr = join_drain_by(&mut stderr, deadline, &stop, &mut stopped_early).await;
        drop(stop_on_drop);
        Drained {
            stdout,
            stderr,
            stopped_early,
        }
    }
}

/// What a command's two drains kept.
pub(crate) struct Drained {
    pub(crate) stdout: Captured,
    pub(crate) stderr: Captured,
    /// Something kept a pipe open past the drain timeout, so the drains were told to stop and
    /// anything printed after that is not in `stdout` or `stderr`.
    pub(crate) stopped_early: bool,
}

/// One drain's capture by `deadline`. Past it, `stop` is fired and the drain is joined for what it
/// has, which it returns as soon as it sees the signal.
async fn join_drain_by(
    task: &mut tokio::task::JoinHandle<Captured>,
    deadline: tokio::time::Instant,
    stop: &CancellationToken,
    stopped_early: &mut bool,
) -> Captured {
    let joined = match tokio::time::timeout_at(deadline, &mut *task).await {
        Ok(joined) => joined,
        Err(_elapsed) => {
            *stopped_early = true;
            stop.cancel();
            task.await
        }
    };
    joined.unwrap_or_else(|error| {
        tracing::debug!("drain task failed: {error}");
        Captured::default()
    })
}

/// Read `reader` until its end, `stop`, or its budget running out, sending each chunk to `sink` as
/// it arrives and keeping what `keep` says. A budget that runs out fires `bound`.
///
/// Whichever way the read ends, what `keep` kept is returned: the caller decides what to say about
/// a stream that ended early, and this function never drops a byte its rule keeps.
async fn drain<R>(
    reader: Option<R>,
    stream: Stream,
    keep: Keep,
    sink: Option<Arc<dyn Sink>>,
    stop: CancellationToken,
    bound: CancellationToken,
) -> Captured
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt as _;

    let mut captured = Captured::default();
    let Some(mut reader) = reader else {
        return captured;
    };
    let mut buffer = [0u8; 8192];
    loop {
        // `read` is cancel-safe on a pipe: a read that has not completed has copied nothing, so
        // stopping here loses nothing already produced.
        let read = tokio::select! {
            read = reader.read(&mut buffer) => read,
            () = stop.cancelled() => break,
        };
        let read = match read {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) => {
                tracing::debug!("failed to read child output: {error}");
                break;
            }
        };
        let chunk = buffer.get(..read).unwrap_or_default();
        if let Some(sink) = &sink {
            sink.receive(stream, chunk).await;
        }
        match &keep {
            Keep::Budgeted(budget) => {
                captured.bytes.extend_from_slice(chunk);
                // After the chunk is kept, so the capture ends on what was actually read; the run
                // kills the command, and this drain has nothing more to take from it.
                if !budget.spend(read) {
                    bound.cancel();
                    break;
                }
            }
            Keep::Head(limit) => {
                let room = limit.saturating_sub(captured.bytes.len());
                let kept = chunk.get(..room).unwrap_or(chunk);
                captured.bytes.extend_from_slice(kept);
                captured.dropped += (chunk.len() - kept.len()) as u64;
            }
            Keep::Tail(limit) => {
                captured.bytes.extend_from_slice(chunk);
                // Trimmed once the kept part has doubled, so a long output is moved a few times
                // rather than once per read.
                if captured.bytes.len() > limit.saturating_mul(2) {
                    keep_tail(&mut captured, *limit);
                }
            }
        }
    }
    if let Keep::Tail(limit) = keep {
        keep_tail(&mut captured, limit);
    }
    captured
}

/// Cut `captured` to its last `limit` bytes, counting what goes.
fn keep_tail(captured: &mut Captured, limit: usize) {
    let cut = captured.bytes.len().saturating_sub(limit);
    captured.bytes.drain(..cut);
    captured.dropped += cut as u64;
}

/// How a command [`run`] waited on ended.
#[derive(Debug)]
pub(crate) enum Ended {
    /// It exited on its own, with this status.
    Exited(std::process::ExitStatus),
    /// Its timeout passed first, and the tree was killed.
    TimedOut,
    /// The stop token fired first, and the tree was killed.
    Stopped,
    /// A stream's budget ran out first, and the tree was killed.
    OutputBound,
}

/// How a command ended, and what its drains kept.
pub(crate) struct Finished {
    pub(crate) ended: Ended,
    pub(crate) drained: Drained,
}

/// What ends a run from outside, and how its tree is ended then.
pub(crate) enum Stop<'a> {
    /// As a timeout ends it: `SIGTERM`, a grace, then `SIGKILL`, so a command interrupted under a
    /// turn has its moment to clean up.
    Graceful(&'a CancellationToken),
    /// With `SIGKILL` and no grace: the holder has decided the command is over, as the screen
    /// has when its user stops a `!command`, and waiting out a grace would only hold the answer.
    AtOnce(&'a CancellationToken),
}

impl Stop<'_> {
    fn token(&self) -> &CancellationToken {
        match self {
            Self::Graceful(token) | Self::AtOnce(token) => token,
        }
    }
}

/// Run `tree` to its end: drain its pipes as `capture` says, wait for the first of its exit,
/// `stop`, `timeout` and a budget running out, kill the tree on any but the exit, and collect the
/// drains within `drain_timeout`, or on a stop at once with what they have read, since an
/// interrupt does not wait on a pipe something outside the tree still holds. `None` for `timeout`
/// or `stop` is a run nothing ends but the command, a budget or a drop.
///
/// `Err` only when waiting on the child fails, and the tree is killed with the dropped value then.
pub(crate) async fn run(
    mut tree: Tree,
    capture: Capture,
    timeout: Option<Duration>,
    stop: Option<Stop<'_>>,
    drain_timeout: Duration,
) -> io::Result<Finished> {
    let drains = Drains::start(tree.0.stdout.take(), tree.0.stderr.take(), capture);
    // `biased`, so a bound reached in the same instant the child exits is reported as the bound:
    // the random pick would otherwise call the command clean while its last reads were never
    // taken.
    let ended = tokio::select! {
        biased;
        () = until_stopped(stop.as_ref()) => {
            match stop {
                Some(Stop::AtOnce(_)) => tree.kill_at_once().await,
                _ => tree.kill().await,
            }
            drains.stop.cancel();
            Ended::Stopped
        }
        () = drains.bound_reached() => {
            tree.kill().await;
            Ended::OutputBound
        }
        () = until_elapsed(timeout) => {
            tree.kill().await;
            Ended::TimedOut
        }
        status = tree.0.wait() => Ended::Exited(status?),
    };
    Ok(Finished {
        ended,
        drained: drains.collect(drain_timeout).await,
    })
}

/// Until `stop` fires; never, when there is none.
async fn until_stopped(stop: Option<&Stop<'_>>) {
    match stop {
        Some(stop) => stop.token().cancelled().await,
        None => std::future::pending().await,
    }
}

/// Until `timeout` has passed; never, when there is none.
async fn until_elapsed(timeout: Option<Duration>) {
    match timeout {
        Some(timeout) => tokio::time::sleep(timeout).await,
        None => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quiet child that lives until it is killed, where a test spawns one.
    #[cfg(any(target_os = "linux", windows))]
    fn sleeping_command() -> tokio::process::Command {
        #[cfg(windows)]
        let mut command = {
            let mut command = tokio::process::Command::new("cmd");
            command.args(["/c", "ping", "-n", "30", "127.0.0.1"]);
            command
        };
        #[cfg(not(windows))]
        let mut command = {
            let mut command = tokio::process::Command::new("sleep");
            command.arg("30");
            command
        };
        command
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        command
    }

    /// A child the door started reads the kernel's ceiling, so an OOM kill in its cgroup or on
    /// the host takes it ahead of Meka whatever Meka's own score is.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_child_runs_at_the_highest_oom_score() {
        let mut command = sleeping_command();
        let mut tree = spawn_tree(&mut command).expect("spawn sleep");
        let id = tree.id().expect("a running child");
        let score =
            std::fs::read_to_string(format!("/proc/{id}/oom_score_adj")).expect("read the score");
        tree.kill().await;
        assert_eq!(score.trim(), "1000");
    }

    /// The door's own child-side step runs once the score is raised: the step here fails the
    /// spawn unless it reads the ceiling, so a placement registered after it would fail this
    /// spawn the way a Landlock domain would fail the placement.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_door_s_own_step_in_the_child_runs_after_the_score_is_raised() {
        let mut command = tokio::process::Command::new("true");
        // SAFETY: the closure reads one proc file through raw syscalls and allocates nothing.
        let spawned = unsafe {
            spawn_with_pre_exec(&mut command, || {
                let descriptor = libc::open(
                    c"/proc/self/oom_score_adj".as_ptr(),
                    libc::O_RDONLY | libc::O_CLOEXEC,
                );
                if descriptor < 0 {
                    return Err(io::Error::last_os_error());
                }
                let mut read = [0u8; 8];
                let count = libc::read(descriptor, read.as_mut_ptr().cast(), read.len());
                libc::close(descriptor);
                if count >= 4 && &read[..4] == b"1000" {
                    Ok(())
                } else {
                    Err(io::Error::from_raw_os_error(libc::EPERM))
                }
            })
        };
        let mut child = spawned.expect("the step read the raised score");
        child.wait().await.expect("wait for the child");
    }

    /// A child the door started runs in a job that ends with the handle and whose committed
    /// memory stops at the machine's.
    #[cfg(windows)]
    #[tokio::test]
    async fn a_child_runs_in_a_job_that_ends_with_it_and_stops_at_the_machine_s_memory() {
        use std::os::windows::io::AsRawHandle;

        use windows_sys::Win32::System::JobObjects::{
            IsProcessInJob, JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            QueryInformationJobObject,
        };

        let mut command = sleeping_command();
        let mut child = spawn(&mut command).expect("spawn the child");
        let job = child.job.as_ref().expect("a job");
        let handle = child.raw_handle().expect("a running child");

        let mut inside = 0;
        // SAFETY: both handles are open; `inside` receives the answer.
        assert_ne!(
            unsafe { IsProcessInJob(handle, job.as_raw_handle(), &mut inside) },
            0
        );
        assert_ne!(inside, 0, "the child is in the job");

        // SAFETY: an all-zero block is a valid one to fill.
        let mut information: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `information` is a complete, correctly sized block that outlives the call.
        assert_ne!(
            unsafe {
                QueryInformationJobObject(
                    job.as_raw_handle(),
                    JobObjectExtendedLimitInformation,
                    (&raw mut information).cast(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                    std::ptr::null_mut(),
                )
            },
            0
        );
        let flags = information.BasicLimitInformation.LimitFlags;
        assert_ne!(flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, 0);
        assert_ne!(flags & JOB_OBJECT_LIMIT_JOB_MEMORY, 0);
        assert_eq!(
            information.JobMemoryLimit,
            physical_memory_bytes().expect("the machine's memory")
        );
        child.kill().await.expect("kill the child");
    }

    /// A child another crate spawned is capped through its id alone, in a job that is not tied
    /// to the handle.
    #[cfg(windows)]
    #[tokio::test]
    async fn a_child_another_crate_spawned_is_capped_by_its_id() {
        use std::os::windows::io::AsRawHandle;

        use windows_sys::Win32::System::JobObjects::{
            IsProcessInJob, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            QueryInformationJobObject,
        };

        let mut command = sleeping_command();
        let mut child = command.spawn().expect("spawn the child");
        let job = cap_by_process_id(child.id().expect("a running child")).expect("a job");
        let handle = child.raw_handle().expect("a running child");

        let mut inside = 0;
        // SAFETY: both handles are open; `inside` receives the answer.
        assert_ne!(
            unsafe { IsProcessInJob(handle, job.as_raw_handle(), &mut inside) },
            0
        );
        assert_ne!(inside, 0, "the child is in the job");

        // SAFETY: an all-zero block is a valid one to fill.
        let mut information: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `information` is a complete, correctly sized block that outlives the call.
        assert_ne!(
            unsafe {
                QueryInformationJobObject(
                    job.as_raw_handle(),
                    JobObjectExtendedLimitInformation,
                    (&raw mut information).cast(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                    std::ptr::null_mut(),
                )
            },
            0
        );
        assert_eq!(
            information.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            0,
            "a job this door does not hold must not end with its handle"
        );
        drop(job);
        assert!(
            child.try_wait().expect("poll the child").is_none(),
            "the child outlives the dropped handle"
        );
        child.kill().await.expect("kill the child");
    }

    /// A sink that hands every chunk it receives to a channel, so a test can wait for a drain to
    /// have read something.
    struct ChannelSink(tokio::sync::mpsc::UnboundedSender<Vec<u8>>);

    #[async_trait]
    impl Sink for ChannelSink {
        async fn receive(&self, _stream: Stream, chunk: &[u8]) {
            self.0.send(chunk.to_vec()).expect("the test still listens");
        }
    }

    /// A drain over `reader` keeping what `keep` says, with nothing telling it to stop; also
    /// whether it reported its bound.
    async fn drain_for_test<R>(reader: R, keep: Keep) -> (Captured, bool)
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        let bound = CancellationToken::new();
        let captured = drain(
            Some(reader),
            Stream::Stdout,
            keep,
            None,
            CancellationToken::new(),
            bound.clone(),
        )
        .await;
        (captured, bound.is_cancelled())
    }

    /// A drain stops reading once the budget is spent, keeps every byte it read including the
    /// chunk that crossed the line, and raises the signal the run kills the command on. Nothing
    /// is elided: what the caller gets is a prefix of what the command printed.
    #[tokio::test]
    async fn a_drain_stops_at_the_output_bound_and_keeps_what_it_read() {
        let bound = 100 * crate::text::KIB;
        let mut source = Vec::with_capacity(3 * bound);
        source.extend_from_slice(b"FIRST-LINE\n");
        source.resize(3 * bound, b'x');
        let budget = Arc::new(OutputBudget::new(bound));

        let (collected, reported) = drain_for_test(
            std::io::Cursor::new(source.clone()),
            Keep::Budgeted(Arc::clone(&budget)),
        )
        .await;

        assert!(
            budget.is_exhausted() && reported,
            "spending the budget must raise the signal"
        );
        let collected = collected.bytes;
        assert!(
            collected.len() >= bound && collected.len() < source.len(),
            "the drain stops at the bound, not before and not at EOF: {} bytes",
            collected.len()
        );
        assert_eq!(
            collected.as_slice(),
            &source[..collected.len()],
            "what was kept is a prefix of the stream"
        );
    }

    /// Both of a command's drains draw on one budget, so output split across stdout and stderr is
    /// bounded as a whole rather than at twice the bound.
    #[tokio::test]
    async fn two_drains_share_one_budget() {
        let bound = 100 * crate::text::KIB;
        let budget = Arc::new(OutputBudget::new(bound));
        // Each stream alone fits; together they do not.
        let each = vec![b'y'; bound * 3 / 4];

        let ((stdout, _), (stderr, _)) = tokio::join!(
            drain_for_test(
                std::io::Cursor::new(each.clone()),
                Keep::Budgeted(Arc::clone(&budget))
            ),
            drain_for_test(
                std::io::Cursor::new(each.clone()),
                Keep::Budgeted(Arc::clone(&budget))
            ),
        );

        assert!(budget.is_exhausted(), "the pair spent the budget");
        let printed = stdout.bytes.len() + stderr.bytes.len();
        assert!(
            printed < 2 * each.len(),
            "one of the drains stopped short: {printed}"
        );
        assert!(
            printed >= bound,
            "and not before the bound was reached: {printed}"
        );
    }

    /// A drain told to stop returns what it has read so far rather than waiting for a pipe that
    /// something else is holding open; that is what makes a grandchild cost only its later output.
    #[tokio::test]
    async fn a_drain_told_to_stop_returns_what_it_has() {
        use tokio::io::AsyncWriteExt;
        let (mut writer, reader) = tokio::io::duplex(4096);
        let stop = CancellationToken::new();
        let (sink, mut received) = tokio::sync::mpsc::unbounded_channel();
        let draining = tokio::spawn(drain(
            Some(reader),
            Stream::Stdout,
            Keep::Budgeted(Arc::new(OutputBudget::new(usize::MAX))),
            Some(Arc::new(ChannelSink(sink))),
            stop.clone(),
            CancellationToken::new(),
        ));

        writer.write_all(b"partial output\n").await.expect("write");
        writer.flush().await.expect("flush");
        // The writer stays open, so EOF never comes; only the signal ends the drain. Sent once the
        // sink shows the drain has taken the bytes, so the test never races the read.
        assert_eq!(
            received
                .recv()
                .await
                .expect("the drain reads what was written"),
            b"partial output\n"
        );
        stop.cancel();

        let collected = tokio::time::timeout(Duration::from_secs(2), draining)
            .await
            .expect("the drain returns promptly once told to stop")
            .expect("the drain task completes");
        assert_eq!(collected.bytes, b"partial output\n");
        drop(writer);
    }

    /// A stop is answered at once, with what was read: the run does not wait out its drain timeout
    /// for a pipe that something outside the tree still holds.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_stop_does_not_wait_on_a_pipe_held_outside_the_tree() {
        let dir = tempfile::tempdir().expect("tempdir");
        let escaped = dir.path().join("escaped");
        // `setsid` takes its shell out of the tree, so the kill leaves it holding the pipes; it
        // says so once it is out, which is what the stop waits for.
        let mut command = tokio::process::Command::new("sh");
        command
            .arg("-c")
            .arg(format!(
                "setsid sh -c 'echo $$ > \"{}\"; echo escaped; exec sleep 30' & exec sleep 30",
                escaped.display()
            ))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let tree = spawn_tree(&mut command).expect("the command starts");
        let (sink, mut received) = tokio::sync::mpsc::unbounded_channel();
        let capture = Capture {
            stdout: Keep::Tail(crate::text::KIB),
            stderr: Keep::Tail(crate::text::KIB),
            sink: Some(Arc::new(ChannelSink(sink))),
        };
        let stop = CancellationToken::new();
        let running = tokio::spawn({
            let stop = stop.clone();
            async move {
                let stop = Stop::Graceful(&stop);
                run(tree, capture, None, Some(stop), Duration::from_secs(3600)).await
            }
        });
        assert_eq!(
            received.recv().await.expect("the command prints"),
            b"escaped\n"
        );
        stop.cancel();

        let finished = tokio::time::timeout(Duration::from_secs(10), running).await;
        // The escaped `sleep` is ended here, by the pid the command wrote, rather than left for
        // its 30 seconds.
        let pid: libc::pid_t = std::fs::read_to_string(&escaped)
            .expect("the command wrote the pid")
            .trim()
            .parse()
            .expect("a pid");
        // SAFETY: `kill(2)` has no preconditions.
        unsafe { libc::kill(pid, libc::SIGKILL) };

        let finished = finished
            .expect("a stop does not wait for the pipe to close")
            .expect("the run completes")
            .expect("the command ran");
        assert!(
            matches!(finished.ended, Ended::Stopped),
            "{:?}",
            finished.ended
        );
        assert_eq!(finished.drained.stdout.bytes, b"escaped\n");
    }

    /// A stop that kills at once sends no `SIGTERM` first: the command gets no moment to clean
    /// up, and the run does not wait one out.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_stop_at_once_gives_the_tree_no_grace() {
        let dir = tempfile::tempdir().expect("tempdir");
        let termed = dir.path().join("termed");
        // The trap is set before the command says it is ready, so a `SIGTERM` after that would
        // leave the file behind.
        let mut command = tokio::process::Command::new("sh");
        command
            .arg("-c")
            .arg(format!(
                "trap 'echo termed > \"{}\"; exit 0' TERM; echo ready; sleep 30 & wait",
                termed.display()
            ))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let tree = spawn_tree(&mut command).expect("the command starts");
        let (sink, mut received) = tokio::sync::mpsc::unbounded_channel();
        let capture = Capture {
            stdout: Keep::Tail(crate::text::KIB),
            stderr: Keep::Tail(crate::text::KIB),
            sink: Some(Arc::new(ChannelSink(sink))),
        };
        let stop = CancellationToken::new();
        let running = tokio::spawn({
            let stop = stop.clone();
            async move {
                let stop = Stop::AtOnce(&stop);
                run(tree, capture, None, Some(stop), Duration::from_secs(10)).await
            }
        });
        assert_eq!(
            received.recv().await.expect("the command prints"),
            b"ready\n"
        );
        stop.cancel();

        let finished = running
            .await
            .expect("the run completes")
            .expect("the command ran");
        assert!(
            matches!(finished.ended, Ended::Stopped),
            "{:?}",
            finished.ended
        );
        assert!(
            !termed.exists(),
            "the command was killed outright, with no signal it could answer"
        );
    }

    /// A command that prints without end cannot grow its reader's memory with it: a tail keeps
    /// the stream's last bytes and counts what it let go.
    #[tokio::test]
    async fn a_tail_keeps_the_last_bytes_and_counts_the_rest() {
        let limit = 64 * crate::text::KIB;
        let (mut writer, reader) = tokio::io::duplex(64 * 1024);
        let writing = tokio::spawn(async move {
            use tokio::io::AsyncWriteExt as _;
            for byte in *b"abc" {
                writer.write_all(&vec![byte; limit]).await.expect("write");
            }
        });
        let (kept, _) = drain_for_test(reader, Keep::Tail(limit)).await;
        writing.await.expect("the writer");
        assert_eq!(kept.bytes.len(), limit);
        assert_eq!(kept.dropped, 2 * limit as u64);
        assert!(kept.bytes.iter().all(|&byte| byte == b'c'));
    }
}
