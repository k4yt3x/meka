//! What the screen and the one-shot host share: the interrupt relay behind Ctrl+C and the turn
//! wrappers that race a turn against an interrupt.

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
/// the second press of the fifth turn means what the second press of the first one did, and a
/// press between turns starts a ladder of its own; see [`InterruptRelay::count_press`].
///
/// Nothing here competes with the screen. Its raw mode delivers Ctrl+C as a key the screen
/// answers itself, and the terminal generates no SIGINT, so while the screen is up this listener
/// only ever sees a signal sent from outside; in a one-shot run it is the whole of Ctrl+C.
pub(crate) struct InterruptRelay {
    pub(crate) presses: std::sync::atomic::AtomicUsize,
    /// Whether the ladder being climbed found a turn on its first press. A ladder a turn started
    /// ends with the turn: the next press between turns starts a new one instead of being that
    /// turn's second press.
    pub(crate) ladder_had_turn: std::sync::atomic::AtomicBool,
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
        ladder_had_turn: std::sync::atomic::AtomicBool::new(false),
        pressed: tokio::sync::Notify::new(),
    });
impl InterruptRelay {
    /// Fire the live turn's token, if any, and count the press on the ladder.
    ///
    /// A ladder climbed during a turn ends with the turn. Carrying its count into the wait after
    /// made a Ctrl+C at `/usage` or a stuck `/mcp reconnect` the second press of a turn already
    /// over, stopping every background task with nothing on screen having asked. Presses between
    /// turns still climb a ladder of their own, so a hung wait can be left the same way as a hung
    /// turn: nothing, then the background tasks, then the process.
    pub(crate) fn count_press(&self, cancel: &crate::host::CancelCell) -> usize {
        use std::sync::atomic::Ordering::SeqCst;
        let found_turn = cancel.cancel(crate::store::turns::CancelReason::Client);
        if !found_turn && self.ladder_had_turn.swap(false, SeqCst) {
            self.presses.store(0, SeqCst);
        }
        if found_turn {
            self.ladder_had_turn.store(true, SeqCst);
        }
        self.presses.fetch_add(1, SeqCst) + 1
    }

    fn reset(&self) {
        self.presses.store(0, std::sync::atomic::Ordering::SeqCst);
        self.ladder_had_turn
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }
}
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
/// `None` rather than an empty string, because the difference is whether anything is said at all:
/// announcing "Stopping 0 background tasks" would be false.
pub(crate) fn background_cancellation_notice(signaled: usize) -> Option<String> {
    (signaled > 0).then(|| {
        format!(
            "Stopping {} background task{}",
            signaled,
            if signaled == 1 { "" } else { "s" }
        )
    })
}
/// Start the process's single SIGINT listener, and on Unix the SIGHUP and SIGTERM listeners with
/// it. Idempotent; every turn path calls it, and only the first call spawns. `announce` says what a
/// press did where the host shows such things, and `leave` is how the host leaves on the ladder's
/// last press, once the canceled work has had its moment: the one-shot exits on the spot, while the
/// screen has a terminal to give back first and goes through its own exit.
pub(crate) fn install_interrupt_handler(
    cancel: crate::host::CancelCell,
    agent: &Agent,
    announce: impl Fn(&str) + Send + Sync + 'static,
    leave: impl Fn() + Send + Sync + 'static,
) {
    static INSTALLED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return;
    }

    let leave: std::sync::Arc<dyn Fn() + Send + Sync> = std::sync::Arc::new(leave);
    // A hangup or a termination has no ladder: the terminal is going or gone, so the turn is
    // stopped, and the background tasks with it, whose process groups would otherwise outlive the
    // exit untracked; then the host leaves through its own exit, which kills a `!command` still
    // running and gives the terminal back where there is one.
    #[cfg(unix)]
    tokio::spawn({
        let leave = std::sync::Arc::clone(&leave);
        let cancel = cancel.clone();
        let tasks = agent.background_tasks();
        let store = agent.store();
        async move {
            use tokio::signal::unix::{SignalKind, signal};
            let (Ok(mut hangup), Ok(mut terminate)) = (
                signal(SignalKind::hangup()),
                signal(SignalKind::terminate()),
            ) else {
                tracing::warn!("failed to listen for SIGHUP and SIGTERM");
                return;
            };
            tokio::select! {
                _ = hangup.recv() => {}
                _ = terminate.recv() => {}
            }
            cancel.cancel(crate::store::turns::CancelReason::ServerShutdown);
            // The process is going away under them, which is what `interrupted` says.
            tasks
                .stop_all(
                    &store.background_store(),
                    crate::store::background::TaskStatus::Interrupted,
                )
                .await;
            if tokio::time::timeout(INTERRUPT_DRAIN_GRACE, tasks.wait_all())
                .await
                .is_err()
            {
                tracing::warn!(
                    "background tasks did not unwind within {INTERRUPT_DRAIN_GRACE:?}; exiting anyway"
                );
            }
            leave();
        }
    });
    let tasks = agent.background_tasks();
    let store = agent.store();
    tokio::spawn(async move {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                return;
            }
            INTERRUPT_RELAY.pressed.notify_waiters();
            let press = INTERRUPT_RELAY.count_press(&cancel);

            match escalation_for(press) {
                // `count_press` has already fired the turn's token, when there was one.
                Escalation::CancelTurn => {}
                Escalation::CancelBackgroundTasks => {
                    let signaled = tasks
                        .stop_all(
                            &store.background_store(),
                            crate::store::background::TaskStatus::Canceled,
                        )
                        .await;
                    if let Some(notice) = background_cancellation_notice(signaled) {
                        announce(&notice);
                    }
                }
                Escalation::Leave => {
                    announce("Interrupted");
                    if tokio::time::timeout(INTERRUPT_DRAIN_GRACE, tasks.wait_all())
                        .await
                        .is_err()
                    {
                        tracing::warn!(
                            "background tasks did not unwind within {INTERRUPT_DRAIN_GRACE:?}; exiting anyway"
                        );
                    }
                    leave();
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
    cancellation: crate::agent::TurnCancellation,
) -> crate::host::Published {
    let published = cancel.publish(cancellation, admission);
    reset_interrupt_escalation();
    published
}

/// Start the Ctrl+C escalation count over: the first press of a new turn cancels that turn,
/// whatever happened during the last one. The scheduler driver's turns publish through the resident
/// session's cell rather than [`publish_for_interrupts`], so they reset it here.
pub(crate) fn reset_interrupt_escalation() {
    INTERRUPT_RELAY.reset();
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
    let cancellation = crate::agent::TurnCancellation::new();
    let _published = publish_for_interrupts(cancel, admission, cancellation.clone());
    let outcome = agent
        .compact_session(messages, request, cancellation.token().clone())
        .await;
    *session_id = agent.session_id();
    outcome
}
/// Run one agent turn with Ctrl+C wired to a fresh cancellation token. Hands the token to
/// [`InterruptRelay`] for the turn's duration, so a SIGINT during the turn cancels it and every
/// tool and sub-agent it spawned. Every `run_turn` callsite on the screen / CLI path must go
/// through here; a bare `CancellationToken` with no signal source silently swallows Ctrl+C.
pub(crate) async fn run_turn_interruptible(
    shared: &crate::host::SharedDeps,
    cancel: &crate::host::CancelCell,
    admission: crate::host::Admission,
    agent: &Agent,
    session_id: &mut Option<uuid::Uuid>,
    messages: &mut crate::conversation::Conversation,
    input: crate::agent::TurnInput,
) -> crate::error::Result<crate::agent::TurnOutcome> {
    // The row is the carrier, as on every server's turn doors: a rename another process made
    // moves the row, and the turn runs on what the row names now rather than on what this
    // process last read.
    if let Some(id) = *session_id
        && let Err(error) = crate::host::apply_recorded_bindings(shared, agent, id).await
    {
        return Err(match error.downcast::<crate::error::MekaError>() {
            Ok(error) => error,
            Err(error) => crate::error::MekaError::Internal(format!("{error:#}")),
        });
    }
    let cancellation = crate::agent::TurnCancellation::new();
    let _published = publish_for_interrupts(cancel, admission, cancellation.clone());
    // The screen does not surface a stop reason; `--oneshot --format json` reports it.
    let outcome = agent.run_turn(messages, input, cancellation).await;
    // The screen keeps its own copy for its loop; the agent's cell is the authority, and a first
    // turn has just filled it.
    *session_id = agent.session_id();
    outcome
}
/// How long leaving the screen waits for its canceled background tasks to unwind.
///
/// Long enough for the work a canceled task actually has left -- signal its process group, write
/// one row -- and short enough that a task ignoring its token cannot hold the terminal. Whatever
/// overruns it is swept to `interrupted` when the session is next opened.
pub(crate) const BACKGROUND_EXIT_GRACE: std::time::Duration = std::time::Duration::from_secs(5);
/// Move a lock into the slot the agent and its host share, replacing whatever was there.
///
/// The replacement is what `/fork` depends on: the new lock is already held by the time this is
/// called, and the old guard is dropped only once the new one is in place, so the session is never
/// momentarily unheld. Passing `None` releases outright, which is how the screen lets go on the way
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The handler's count, driven the way SIGINT drives it, against a relay of the test's own: the
    /// process-wide one is shared with every other test in the binary.
    #[test]
    fn a_press_between_turns_starts_a_ladder_of_its_own() {
        let relay = InterruptRelay {
            presses: std::sync::atomic::AtomicUsize::new(0),
            ladder_had_turn: std::sync::atomic::AtomicBool::new(false),
            pressed: tokio::sync::Notify::new(),
        };
        let cancel = crate::host::CancelCell::default();

        let token = crate::agent::TurnCancellation::new();
        let published = cancel.publish(token.clone(), cancel.admit());
        assert_eq!(relay.count_press(&cancel), 1);
        assert!(token.is_canceled(), "the first press cancels the turn");
        assert_eq!(
            relay.count_press(&cancel),
            2,
            "a turn still unwinding takes the second press"
        );
        drop(published);

        assert_eq!(
            relay.count_press(&cancel),
            1,
            "a press between turns starts a ladder of its own instead of being the third"
        );
        assert_eq!(
            relay.count_press(&cancel),
            2,
            "a second press at a hung wait still reaches the background tasks"
        );

        let token = crate::agent::TurnCancellation::new();
        let _published = cancel.publish(token, cancel.admit());
        relay.reset();
        assert_eq!(
            relay.count_press(&cancel),
            1,
            "publishing a turn starts the count over, whatever the wait before it counted"
        );
    }

    /// The Ctrl+C ladder, which nothing else can reach.
    ///
    /// `install_interrupt_handler` is a spawned listener on `tokio::signal::ctrl_c()` ending in
    /// `std::process::exit(130)`, so no test drives it; the four mutants that survived the sweep
    /// all lived in these two decisions. Both are behavior: collapsing the second press into the
    /// third makes Ctrl+C Ctrl+C kill the process instead of the background tasks, which is the
    /// unrecoverable outcome the ladder exists to put one more keystroke in front of.
    #[test]
    fn the_interrupt_ladder_escalates_one_press_at_a_time() {
        assert_eq!(
            escalation_for(1),
            Escalation::CancelTurn,
            "the first press is the shell's contract: the foreground job, and nothing else"
        );
        assert_eq!(
            escalation_for(2),
            Escalation::CancelBackgroundTasks,
            "the second stops the background work, and is the rung whose absence would make the \
             second press fatal"
        );
        for press in [3, 4, 99] {
            assert_eq!(
                escalation_for(press),
                Escalation::Leave,
                "press {press} is past the ladder and leaves"
            );
        }
        // Zero is unreachable -- the counter is incremented before this is asked -- so what matters
        // is that it does not land on a rung, not which one it picks.
        assert_eq!(escalation_for(0), Escalation::Leave);
    }

    /// Nothing is announced when nothing was stopped, and the plural agrees with the count.
    #[test]
    fn the_background_cancellation_notice_counts_what_it_stopped() {
        assert_eq!(
            background_cancellation_notice(0),
            None,
            "a second press with nothing running must not claim to have stopped anything"
        );
        assert_eq!(
            background_cancellation_notice(1).as_deref(),
            Some("Stopping 1 background task")
        );
        assert_eq!(
            background_cancellation_notice(4).as_deref(),
            Some("Stopping 4 background tasks")
        );
    }
}
