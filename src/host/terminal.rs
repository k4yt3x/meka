//! What the two terminal hosts share: the interrupt relay behind Ctrl+C, the console episode
//! helpers, and the turn wrappers that race a turn against an interrupt.

use super::*;

/// The process's Ctrl+C handling: one long-lived listener, not one per turn.
///
/// tokio installs its SIGINT handler on first use and never removes it, so a per-turn listener that
/// is aborted when the turn ends leaves that handler in place with nothing awaiting it. Every later
/// press is then captured and dropped, and a turn whose tool ignores cancellation -- a stuck child,
/// an MCP call that never returns -- became unkillable from its own terminal. A task that never
/// stops awaiting cannot drop one.
///
/// Escalation counts per *turn*, not per process: publishing a turn's token resets the count, so
/// the second press of the fifth turn means what the second press of the first one did.
///
/// Nothing here competes with the prompt. reedline reads Ctrl+C as a key event in raw mode, where
/// the terminal generates no SIGINT at all, so this listener only ever sees a press made while a
/// turn is running -- which is the only window it is about.
pub(crate) struct InterruptRelay {
    pub(crate) presses: std::sync::atomic::AtomicUsize,
    /// Woken on every press, for the one caller that waits outside a turn.
    ///
    /// A second `tokio::signal::ctrl_c()` elsewhere in the process would be a second *handler*:
    /// tokio delivers each press to every awaiter, so one keystroke ran the escalation ladder here
    /// and printed an unrelated message there, racing each other's output and the outcome
    /// collection between them. Waiters listen to this instead, so the ladder stays the only
    /// reader of the signal.
    pub(crate) pressed: tokio::sync::Notify,
}
pub(crate) static INTERRUPT_RELAY: std::sync::LazyLock<InterruptRelay> =
    std::sync::LazyLock::new(|| InterruptRelay {
        presses: std::sync::atomic::AtomicUsize::new(0),
        pressed: tokio::sync::Notify::new(),
    });
/// Grace given to background tasks on the press that leaves. Long enough for a child to die and its
/// row to be written, short enough that a user who has pressed Ctrl+C three times is not made to
/// wait: `exit` on the spot orphans the process group and leaves the row reading `running`.
pub(crate) const INTERRUPT_DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(2);
/// What the nth Ctrl+C means.
///
/// Split from the handler because the handler cannot be tested: it is a `tokio::spawn` around
/// `tokio::signal::ctrl_c()` whose last arm calls `std::process::exit`, so driving it needs real
/// signals and survives none of them. The escalation itself is a decision about a number, and every
/// mutation of it -- deleting the second arm, or moving its boundary -- is a real behavior change:
/// dropping [`Escalation::CancelBackgroundTasks`] makes the second press exit the process, which is
/// the data-loss shape the three-press ladder exists to avoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Escalation {
    /// The shell's contract: the first SIGINT reaches the foreground job only. Background work
    /// survives, because losing a twenty-minute build to a Ctrl+C aimed at the answer on screen is
    /// unrecoverable and is not what the keystroke meant.
    CancelTurn,
    CancelBackgroundTasks,
    /// Leave -- but let what was already canceled finish unwinding first.
    Leave,
}
pub(crate) fn escalation_for(press: usize) -> Escalation {
    match press {
        1 => Escalation::CancelTurn,
        2 => Escalation::CancelBackgroundTasks,
        _ => Escalation::Leave,
    }
}
/// What to tell the user about the tasks a second Ctrl+C stopped, or `None` when it stopped none.
///
/// `None` rather than an empty string, because the difference is whether the console is written to
/// at all: announcing "stopping 0 background tasks" would be both false and an episode's worth of
/// spacing spent on nothing.
pub(crate) fn background_cancellation_notice(signaled: usize) -> Option<String> {
    (signaled > 0).then(|| {
        format!(
            "stopping {} background task{}",
            signaled,
            if signaled == 1 { "" } else { "s" }
        )
    })
}
/// Start the process's single SIGINT listener. Idempotent; every turn path calls it, and only the
/// first call spawns.
pub(crate) fn install_interrupt_handler(
    cancel: crate::host::CancelCell,
    agent: &Agent,
    console: Arc<std::sync::Mutex<crate::console::Console>>,
) {
    static INSTALLED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return;
    }

    let tasks = agent.background_tasks();
    let store = agent.store();
    tokio::spawn(async move {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                return;
            }
            INTERRUPT_RELAY.pressed.notify_waiters();
            let press = INTERRUPT_RELAY
                .presses
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;

            match escalation_for(press) {
                Escalation::CancelTurn => {
                    cancel.cancel();
                }
                Escalation::CancelBackgroundTasks => {
                    // Recorded before signaling, so what the agent hears is "you stopped it"
                    // rather than the `failed` its own interruption would otherwise write.
                    for id in tasks.task_ids().await {
                        if let Err(error) = store
                            .background_store()
                            .finish_background_task(
                                &id,
                                crate::store::background::TaskStatus::Canceled,
                                None,
                                None,
                            )
                            .await
                        {
                            tracing::warn!("failed to record task {id} as canceled: {error}");
                        }
                    }
                    let signaled = tasks.cancel_all().await;
                    if let Some(notice) = background_cancellation_notice(signaled) {
                        with_console(&console, |console| console.annotation(&notice));
                    }
                }
                Escalation::Leave => {
                    with_console(&console, |console| console.annotation("interrupted"));
                    if tokio::time::timeout(INTERRUPT_DRAIN_GRACE, tasks.wait_all())
                        .await
                        .is_err()
                    {
                        tracing::warn!(
                            "background tasks did not unwind within {INTERRUPT_DRAIN_GRACE:?}; exiting anyway"
                        );
                    }
                    // This arm exits from inside a spawned task, so it never reaches the funnel in
                    // `main` and has to release the grants itself.
                    crate::sandbox::release_process_grants();
                    std::process::exit(130);
                }
            }
        }
    });
}
/// Publish a turn's token where Ctrl+C reaches it, and start the escalation count over: the
/// first press of a new turn cancels that turn, whatever happened during the last one.
///
/// `admission` is the caller's, sampled where the turn was decided on rather than here: a press
/// between the two would otherwise bump an epoch nothing had read yet, and the turn would run as
/// if it had not happened.
fn publish_for_interrupts(
    cancel: &crate::host::CancelCell,
    admission: crate::host::Admission,
    cancellation: CancellationToken,
) -> crate::host::Published {
    let published = cancel.publish(cancellation, admission);
    reset_interrupt_escalation();
    published
}

/// Start the Ctrl+C escalation count over: the first press of a new turn cancels that turn,
/// whatever happened during the last one. The scheduler driver's turns publish through the resident
/// session's cell rather than [`publish_for_interrupts`], so they reset it here.
pub(crate) fn reset_interrupt_escalation() {
    INTERRUPT_RELAY
        .presses
        .store(0, std::sync::atomic::Ordering::SeqCst);
}

/// Run a `/compact` with Ctrl+C wired to a fresh cancellation token.
///
/// Compaction is not a turn, but it makes provider calls and can block on an approval prompt, so
/// it needs a signal source for the reason [`run_turn_interruptible`] gives: a bare token silently
/// swallows Ctrl+C.
pub(crate) async fn compact_interruptible(
    cancel: &crate::host::CancelCell,
    admission: crate::host::Admission,
    agent: &Agent,
    session_id: &mut Option<uuid::Uuid>,
    messages: &mut crate::conversation::Conversation,
    request: crate::session::CompactRequest,
) -> crate::error::Result<crate::agent::CompactOutcome> {
    let cancellation = CancellationToken::new();
    let _published = publish_for_interrupts(cancel, admission, cancellation.clone());
    let outcome = agent.compact_session(messages, request, cancellation).await;
    *session_id = agent.session_id();
    outcome
}
/// Run one agent turn with Ctrl+C wired to a fresh cancellation token. Hands the token to
/// [`InterruptRelay`] for the turn's duration, so a SIGINT during the turn cancels it and every
/// tool and sub-agent it spawned. Every `run_turn` callsite in the REPL / CLI path must go through
/// here; a bare `CancellationToken` with no signal source silently swallows Ctrl+C.
pub(crate) async fn run_turn_interruptible(
    cancel: &crate::host::CancelCell,
    admission: crate::host::Admission,
    agent: &Agent,
    session_id: &mut Option<uuid::Uuid>,
    messages: &mut crate::conversation::Conversation,
    input: crate::agent::TurnInput,
) -> crate::error::Result<crate::agent::TurnOutcome> {
    let cancellation = CancellationToken::new();
    let _published = publish_for_interrupts(cancel, admission, cancellation.clone());
    // The REPL does not surface a stop reason; `--oneshot --format json` reports it.
    let outcome = agent.run_turn(messages, input, cancellation).await;
    // The REPL keeps its own copy for the loop that owns the console; the agent's cell is the
    // authority, and a first turn has just filled it.
    *session_id = agent.session_id();
    outcome
}
/// How long leaving the REPL waits for its canceled background tasks to unwind.
///
/// Long enough for the work a canceled task actually has left -- signal its process group, write
/// one row -- and short enough that a task ignoring its token cannot hold the terminal. Whatever
/// overruns it is swept to `interrupted` when the session is next opened.
pub(crate) const BACKGROUND_EXIT_GRACE: std::time::Duration = std::time::Duration::from_secs(5);
/// Translate the live-REPL display config into the options that
/// [`crate::render::render_message_history`] consumes. Keeps the spacing / styling rules between
/// live output and history rendering in sync from a single source of truth.
pub(crate) fn history_render_options(
    config: &ResolvedConfig,
) -> crate::render::HistoryRenderOptions {
    crate::render::HistoryRenderOptions {
        render_mode: config.render_mode,
        show_thinking: config.thinking_show_content,
        tool_params: config.tool_params,
        input_style: config.input_style,
        newline_before_prompt: config.newline_before_prompt,
        newline_after_prompt: config.newline_after_prompt,
        // `/history` is separated from the command line by the episode's own blank; only the resume
        // path asks for one, because the episode it prints in opens against the shell's prompt and
        // so has no blank of its own. It passes `newline_after_prompt` rather than `true` so tight
        // spacing stays tight.
        leading_blank: false,
    }
}
/// `leading_blank` is emitted only once there is something to print under it, so a last message
/// that renders to nothing (a tool-call-only turn) leaves the banner unbracketed rather than
/// trailing a blank into the prompt. Callers pass `newline_after_prompt`: on a resume the
/// `Continuing session:` banner stands in for the line you typed, and this is the separator that
/// would have followed one.
pub(crate) fn reprint_last_message(
    messages: &[crate::conversation::Message],
    render_mode: crate::config::RenderMode,
    leading_blank: bool,
) -> bool {
    let Some(last) = messages.last() else {
        return false;
    };

    // The words for either role: a user turn's context block is not a `Text` block.
    let text = last.text_content();
    if text.is_empty() {
        return false;
    }

    if leading_blank {
        crate::streams::write_stderr_line("");
    }
    let mut renderer = crate::render::StreamingRenderer::new(render_mode);
    if let Err(error) = renderer.push_delta(&text) {
        crate::render::report_lost_output("a replayed message did not reach stdout", &error);
    }
    if let Err(error) = renderer.finish() {
        crate::render::report_lost_output("a replayed message did not reach stdout", &error);
    }
    true
}
/// Borrow the shared console for one synchronous run of writes.
///
/// A poisoned lock is recovered from rather than propagated: the console holds the terminal's
/// layout, and losing that is a worse outcome than continuing from a state one panicking writer may
/// have left mid-transition. No `.await` may appear inside `act` -- `clippy::await_holding_lock` is
/// deny-level and would catch it, but the reason is that the agent's frontend writes through the
/// same lock.
pub(crate) fn with_console<T>(
    console: &std::sync::Mutex<crate::console::Console>,
    act: impl FnOnce(&mut crate::console::Console) -> T,
) -> T {
    act(&mut crate::sync::lock(console))
}
/// Close an episode that no meka prompt follows.
///
/// Every caller is on the way out of the process: a one-shot turn that has finished or failed, and
/// the REPL after its loop has broken. The prompt below is the shell's own, which the shell has
/// already spaced, so this closes without the `newline_before_prompt` blank.
///
/// "No prompt below" rather than "no output below": a caller can still have a line to print after
/// it -- the shutdown notice, an outcome a one-shot run waited for, a `warn!` the relay routes
/// here. A closed episode arms nothing, so each prints flush against the line above. `repl` closes
/// the episodes a meka prompt really does follow.
pub(crate) fn close_console_episode(console: &std::sync::Mutex<crate::console::Console>) {
    with_console(console, |console| {
        console.close_episode(crate::console::Neighbor::Shell)
    });
}
/// Closes the run's last episode however its host leaves, on the failing paths as much as the
/// ordinary one.
///
/// A `Drop` where the early exits converge rather than a close at each of them, because a close
/// written at a door is a close the next `return Err` forgets. The path that needs it today is
/// `repl_handle.await?`, which returns only when the REPL thread panicked, possibly mid-wake with
/// the prompt it broke out of still drawn and a partial answer still buffered.
///
/// Closing twice is closing once, so this composes with the explicit closes rather than replacing
/// them: those sit ahead of the session lock being released, after the shutdown notice, and ahead
/// of everything a one-shot run still prints once its turn is over.
pub(crate) struct LastEpisode(pub(crate) Arc<std::sync::Mutex<crate::console::Console>>);
impl Drop for LastEpisode {
    fn drop(&mut self) {
        close_console_episode(&self.0);
    }
}
/// Move a lock into the slot the agent and its host share, replacing whatever was there.
///
/// The replacement is what `/fork` depends on: the new lock is already held by the time this is
/// called, and the old guard is dropped only once the new one is in place, so the session is never
/// momentarily unheld. Passing `None` releases outright, which is how the REPL lets go on the way
/// out.
///
/// A poisoned mutex is recovered from rather than propagated. The slot holds one value and every
/// writer replaces it whole, so a panic cannot have left it half-updated -- and refusing to release
/// a lock because some unrelated thread panicked would be the worse failure.
pub(crate) fn hold_session_lock(
    slot: &crate::store::SessionLockSlot,
    lock: Option<crate::fs::FileLock>,
) {
    *crate::sync::lock(slot) = lock;
}
