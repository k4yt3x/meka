//! The one door for every process meka starts, and what each is handed on the way out: a place
//! behind meka in the line for memory.
//!
//! A command the model runs (an `ffmpeg`, a build, a script with a leak) can take every byte the
//! machine or the service's cgroup has, and which process the operating system then takes is
//! decided by scores and caps the parent sets before the child runs. Set at one door, a new spawn
//! site cannot forget them; `tests/layering.rs` lists every file that starts a process and how
//! this door reaches it. What the door sets is best effort on every platform: a child that could
//! not be placed runs anyway, since refusing the command would not save meka from the next one.
//!
//! - Linux: the child's `oom_score_adj` is raised to the ceiling between `fork` and `exec`. The
//!   value is inherited, so the shell, the sandbox around it and everything either starts carry it,
//!   and the kernel's OOM killer takes any of them ahead of meka in a cgroup and host-wide. A
//!   `pre_exec` step costs std's `posix_spawn` fast path, so every spawn forks: `posix_spawn` has
//!   no attribute for the write, and a write from the parent after the spawn would miss a
//!   grandchild forked before it.
//! - Windows: the child is placed in a job object whose committed memory stops at the machine's
//!   physical memory, so past it the command's own allocations fail and meka's never do, the only
//!   way to lose first on a platform without an OOM killer. A job the door hands back ends with its
//!   handle, so dropping the child ends every process it started.
//! - macOS: no lever exists for one process over another; the system takes the largest footprint
//!   first, which is the runaway command.

#[cfg(windows)]
use std::sync::atomic::AtomicBool;
use std::{
    io,
    ops::{Deref, DerefMut},
};

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

/// Start `command` with the door's placement applied.
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
/// goes through [`spawn`] or [`spawn_with_pre_exec`], which do both.
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
                     can take meka with it"
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
                 takes every byte can take meka with it"
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
    use std::sync::atomic::Ordering;

    if announced.swap(true, Ordering::Relaxed) {
        tracing::debug!("{message}");
    } else {
        tracing::warn!("{message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quiet child that lives until it is killed, on every platform.
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
    /// the host takes it ahead of meka whatever meka's own score is.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_child_runs_at_the_highest_oom_score() {
        let mut command = sleeping_command();
        let mut child = spawn(&mut command).expect("spawn sleep");
        let id = child.id().expect("a running child");
        let score =
            std::fs::read_to_string(format!("/proc/{id}/oom_score_adj")).expect("read the score");
        child.kill().await.expect("kill the child");
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
}
