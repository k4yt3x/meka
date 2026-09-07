//! Background tool calls: work the agent starts and does not wait for.
//!
//! An ordinary tool call blocks the turn until it returns. That is right for a file read and wrong
//! for a twenty-minute build, where the agent's only alternatives today are to hold the turn open
//! or to approximate detachment with `nohup … &` and then poll for a result nothing will announce.
//!
//! A backgrounded call returns a handle immediately and delivers its outcome later, as a turn, over
//! the same path [`crate::schedule`] already uses to inject one from outside the conversation.
//!
//! The invariant that shapes everything here: **a task always ends in a delivered outcome.**
//! [`crate::agent::Agent::run_turn`] is not resumable mid-tool-loop, so work in flight when the
//! process dies cannot be recovered. Silence would be worse than never having offered, because the
//! agent has usually already told someone it would report back. So a task that cannot finish is
//! [`crate::store::background::TaskStatus::Interrupted`] and is delivered as such, and the lease
//! that decides *when* a task counts as dead is the session lock itself
//! ([`crate::store::Store::lock_session`]): holding it means nothing else can still be running that
//! session's tasks.

use chrono::Utc;
// Reached through `humantime_serde`'s re-export rather than a direct dependency, matching
// `crate::schedule`.
use humantime_serde::re::humantime;
use uuid::Uuid;

use crate::store::background::BackgroundTask;

/// Live handles for the tasks *this process* started.
///
/// The database row is the durable record; this is the control surface. Deliberately not persisted
/// and deliberately not global: a `CancellationToken` cannot outlive the process holding it, and a
/// task started elsewhere is not this process's to stop. That asymmetry is exactly why the
/// session-load sweep exists, since a row with no handle behind it is a task nobody can finish.
#[derive(Clone, Default)]
pub(crate) struct BackgroundTasks {
    inner: std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<String, TaskHandle>>>,
}

struct TaskHandle {
    session_id: Uuid,
    cancellation: tokio_util::sync::CancellationToken,
    /// `None` between reserving the slot and spawning the work. Canceling in that window still
    /// works, because the token exists from the start; only joining has to wait.
    join: Option<tokio::task::JoinHandle<()>>,
}

impl BackgroundTasks {
    /// Claim a slot for `session_id`, or refuse because the ceiling is full.
    ///
    /// Check and insert happen under one lock, which is the whole point. Several background calls
    /// in a single assistant message are dispatched concurrently by `execute_tool_calls`, so a
    /// separate count-then-register would let every one of them read the same pre-registration
    /// count and sail past a ceiling of one.
    pub(crate) async fn try_reserve(
        &self,
        id: String,
        session_id: Uuid,
        cancellation: tokio_util::sync::CancellationToken,
        max_tasks: usize,
    ) -> bool {
        let mut guard = self.inner.lock().await;
        let running = guard
            .values()
            .filter(|handle| handle.session_id == session_id)
            .count();
        if running >= max_tasks {
            return false;
        }
        guard.insert(id, TaskHandle {
            session_id,
            cancellation,
            join: None,
        });
        true
    }

    /// Attach the spawned work to a slot already reserved by [`Self::try_reserve`].
    pub(crate) async fn attach(&self, id: &str, join: tokio::task::JoinHandle<()>) {
        if let Some(handle) = self.inner.lock().await.get_mut(id) {
            handle.join = Some(join);
        }
    }

    /// Drop a task's handle: on the way out of the work itself, and on the failure path between
    /// reserving a slot and spawning into it, so a refused start cannot leak the reservation and
    /// shrink the ceiling for the rest of the session.
    pub(crate) async fn forget(&self, id: &str) {
        self.inner.lock().await.remove(id);
    }

    /// How many tasks this process is running, across sessions. For the REPL's survivor line, which
    /// cannot name a session for the same reason [`Self::cancel_all`] cannot.
    pub(crate) async fn running_count_all(&self) -> usize {
        self.inner.lock().await.len()
    }

    /// How many of `session_id`'s tasks are running here. Backs the `max_tasks` ceiling.
    pub(crate) async fn running_count(&self, session_id: Uuid) -> usize {
        self.inner
            .lock()
            .await
            .values()
            .filter(|handle| handle.session_id == session_id)
            .count()
    }

    /// Signal one task to stop. `false` when it isn't ours, which is the honest answer for a task
    /// whose process is gone: the row will be swept, not canceled.
    pub(crate) async fn cancel(&self, id: &str) -> bool {
        match self.inner.lock().await.get(id) {
            Some(handle) => {
                handle.cancellation.cancel();
                true
            }
            None => false,
        }
    }

    /// Every task id this process is running, without touching them.
    ///
    /// Callers canceling in bulk need this *before* signaling: a `canceled` row has to be
    /// written first, because `finish_background_task` only overwrites a `running` row and the
    /// work reacting to its token would otherwise land `failed` there first. "Your build
    /// failed" and "you stopped your build" are exactly the distinction the four terminal
    /// states exist to keep.
    pub(crate) async fn task_ids(&self) -> Vec<String> {
        self.inner.lock().await.keys().cloned().collect()
    }

    /// Every task id this process is running for one session, without touching them.
    pub(crate) async fn session_task_ids(&self, session_id: Uuid) -> Vec<String> {
        self.inner
            .lock()
            .await
            .iter()
            .filter(|(_, handle)| handle.session_id == session_id)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Signal every task this process is running, returning how many were signaled.
    ///
    /// For the REPL's second Ctrl+C, which cannot name a session: on the first turn the id does not
    /// exist yet (`run_turn` creates it), and a REPL process has exactly one conversation anyway,
    /// since sub-agents never get background calls.
    pub(crate) async fn cancel_all(&self) -> usize {
        let guard = self.inner.lock().await;
        for handle in guard.values() {
            handle.cancellation.cancel();
        }
        guard.len()
    }

    /// Signal every one of this session's tasks to stop, returning how many were signaled. Backs
    /// `task_cancel --all`.
    pub(crate) async fn cancel_session(&self, session_id: Uuid) -> usize {
        let guard = self.inner.lock().await;
        let mut signaled = 0;
        for handle in guard.values().filter(|h| h.session_id == session_id) {
            handle.cancellation.cancel();
            signaled += 1;
        }
        drop(guard);
        signaled
    }

    /// Wait for every task this process is running, for a host that is about to exit.
    ///
    /// [`Self::cancel_all`] only fires the tokens, and firing a token is not the same as the task
    /// acting on it. A task parked at an await is dropped without ever being polled again when the
    /// runtime goes, so it reaches neither `kill_child_tree` (leaving the `setsid()`-ed child
    /// running with no meka process tracking it) nor `finish_background_task` (leaving the row
    /// `running` for the next session open to sweep to `interrupted`).
    ///
    /// Callers should bound this: canceling asks, and a task that does not answer must not hold
    /// the terminal.
    pub(crate) async fn wait_all(&self) {
        let joins: Vec<tokio::task::JoinHandle<()>> = {
            let mut guard = self.inner.lock().await;
            let ids: Vec<String> = guard.keys().cloned().collect();
            let joins: Vec<tokio::task::JoinHandle<()>> = ids
                .into_iter()
                .filter_map(|id| guard.remove(&id).and_then(|handle| handle.join))
                .collect();
            drop(guard);
            joins
        };
        for join in joins {
            if let Err(error) = join.await {
                tracing::warn!("background task ended abnormally: {error}");
            }
        }
    }

    /// Wait for this session's tasks to finish, for a host with nowhere to deliver an outcome
    /// later. `--oneshot` is the case: the process exits with the turn, so a background call there
    /// has to degrade into a slow synchronous one rather than a promise nothing will keep.
    pub(crate) async fn wait_for_session(&self, session_id: Uuid) {
        let joins: Vec<tokio::task::JoinHandle<()>> = {
            let mut guard = self.inner.lock().await;
            let ids: Vec<String> = guard
                .iter()
                .filter(|(_, handle)| handle.session_id == session_id)
                .map(|(id, _)| id.clone())
                .collect();
            let joins: Vec<tokio::task::JoinHandle<()>> = ids
                .into_iter()
                .filter_map(|id| guard.remove(&id).and_then(|handle| handle.join))
                .collect();
            drop(guard);
            joins
        };
        for join in joins {
            // A panicking task has already written its own `failed` outcome via the dispatch
            // wrapper, so there is nothing to report here beyond not hanging on it.
            if let Err(error) = join.await {
                tracing::warn!("background task ended abnormally: {error}");
            }
        }
    }
}

/// Ceiling on outcome text carried inline in the delivered turn. Past this the full output goes to
/// a scratchpad entry and the turn carries the head plus the entry name: a twenty-minute build log
/// would otherwise land in the conversation permanently, for a result that mattered once.
///
/// Coupled to `tools::shell`'s `OUTPUT_WINDOW_BYTES`, which is eight times larger and keeps both
/// ends of an overflowing stream; `split_outcome` keeps only the head, because an outcome arrives
/// unbidden and should cost less window than a result the model asked for. Raising this one past
/// that one would deliver a turn wider than the tool's own result. Nothing is lost either way: the
/// scratchpad holds all of it.
pub(crate) const OUTCOME_INLINE_LIMIT: usize = 4 * crate::text::KIB;

/// Longest task label shown in the `[Background]` index and in delivered headers.
pub(crate) const LABEL_MAX_CHARS: usize = 80;

/// Keep only the outcomes a stamp actually won.
///
/// The stamps are compare-and-swaps that return the rows they took, and every caller must report
/// against that rather than against the snapshot it chose from: two claimers reading the same row
/// as unclaimed is the ordinary case, and acting on the read is how the same outcome reaches the
/// model twice.
pub(crate) fn only_what_was_won(
    ready: Vec<BackgroundTask>,
    claimed: &[String],
) -> Vec<BackgroundTask> {
    ready
        .into_iter()
        .filter(|task| claimed.contains(&task.id))
        .collect()
}

/// The retention a carrier prompt deserves once an outcome may be riding on it.
///
/// A recurring job asks for [`crate::conversation::PromptRetention::WithdrawOnFailure`] because its
/// next occurrence regenerates the prompt, so a failed copy carries nothing worth keeping. That
/// stops being true the moment an outcome joins it: the row is stamped delivered before the turn
/// starts and is never handed out again, so withdrawing the message loses the only copy. Three
/// hosts fold an outcome into a fired job's prompt and all three ask this, rather than each
/// remembering.
pub(crate) fn retention_carrying(
    riding: &[BackgroundTask],
    job: crate::conversation::PromptRetention,
) -> crate::conversation::PromptRetention {
    if riding.is_empty() {
        job
    } else {
        crate::conversation::PromptRetention::Keep
    }
}

/// [`render_outcomes`] for a report that rides a prompt somebody else wrote, in that turn's
/// context block.
///
/// The standalone form ends "Pick the work back up from here", which is right when it *is* the
/// prompt and wrong above an unrelated question. Riding the prompt's own message rather than
/// being appended as a message of its own is what keeps this off the conversation as a turn
/// boundary: a lone user message opens a turn (`crate::conversation::opens_turn`), and one nobody
/// answers would be rewound in place of the user's last exchange and sent as a second consecutive
/// user turn.
pub(crate) fn render_outcomes_riding(tasks: &[BackgroundTask]) -> String {
    render_outcomes_with_trailer(tasks, PREAMBLE_TRAILER)
}

/// What the standalone form tells the model to do next.
const STANDALONE_TRAILER: &str = "\nYou started these earlier and did not wait for them. Pick the \
    work back up from here; do not restate this header.\n";

/// The same, for a report that precedes a prompt of its own.
const PREAMBLE_TRAILER: &str = "\nYou started these earlier and did not wait for them. Read them, \
    then answer what follows; do not restate this header.\n";

/// Render one or more finished tasks as the user-turn text that delivers them.
///
/// The header is not decoration, for the same reason [`crate::scheduler::Wakeup::render_prompt`]
/// carries one: without it the model reads a bare result as though a human had just typed it, and
/// answers conversationally to nobody. It also has to be unambiguous about *who* is speaking,
/// because a backgrounded `agent_spawn` reports in a sub-agent's words, and a sub-agent is
/// permission-clamped by `resolve_subagent_permission` while its words are not.
///
/// Several outcomes ready at once coalesce into one turn rather than one turn each, as the
/// scheduler already does for a backlog.
pub(crate) fn render_outcomes(tasks: &[BackgroundTask]) -> String {
    render_outcomes_with_trailer(tasks, STANDALONE_TRAILER)
}

fn render_outcomes_with_trailer(tasks: &[BackgroundTask], trailer: &str) -> String {
    let mut rendered = format!(
        "[Background {} reporting at {}]",
        if tasks.len() == 1 {
            "task".to_string()
        } else {
            format!("tasks ({})", tasks.len())
        },
        crate::text::format_timestamp(Utc::now(), crate::text::Precision::Minutes),
    );
    rendered.push_str(trailer);

    for task in tasks {
        rendered.push_str(&format!(
            "\n---\n\n**{}** ({}) {} after {}.\n",
            task.short_id(),
            // Sanitized like the outcome below. The label is derived from the tool's primary
            // argument, which for `execute_command` is a shell command line the *model* wrote, so
            // it is no more trusted than the output it names.
            elide(&crate::text::sanitize_text(&task.label), LABEL_MAX_CHARS),
            task.status.headline(),
            format_elapsed(task.elapsed()),
        ));
        if let Some(name) = &task.scratchpad_name {
            rendered.push_str(&format!(
                "\nFull output is in scratchpad entry `{name}`; the beginning follows.\n"
            ));
        }
        if let Some(outcome) = &task.outcome
            && !outcome.trim().is_empty()
        {
            rendered.push('\n');
            // Sanitized for the same reason MCP text is: this is content from a shell command or a
            // sub-agent, and a terminal-control or bidi-override sequence in it would be rendered
            // to the user and fed to the model verbatim.
            rendered.push_str(&crate::text::sanitize_text(outcome));
            rendered.push('\n');
        }
    }
    rendered
}

/// Split a tool's output into what the turn carries inline and what, if anything, needs a
/// scratchpad entry. `None` in the second slot means it fit.
///
/// Measured in bytes throughout, including the cut. Deciding in bytes and then cutting in
/// characters would make the head of a non-ASCII log as much as four times the budget, and could
/// spill an output whose "head" is then the whole of it: a delivered turn announcing a scratchpad
/// entry and quoting the entire text it was supposed to spare the conversation.
pub(crate) fn split_outcome(output: &str) -> (String, Option<String>) {
    if output.len() <= OUTCOME_INLINE_LIMIT {
        return (output.to_string(), None);
    }
    let mut end = OUTCOME_INLINE_LIMIT;
    while end > 0 && !output.is_char_boundary(end) {
        end -= 1;
    }
    (output[..end].to_string(), Some(output.to_string()))
}

/// Scratchpad entry name for a spilled outcome. Namespaced by the short id so two tasks of the same
/// tool cannot collide, and recognizable so the agent can find it without being told twice.
pub(crate) fn spill_entry_name(task_id: &str, tool_name: &str) -> String {
    let short = task_id.get(..8).unwrap_or(task_id);
    format!("task_{short}_{tool_name}")
}

/// Human-readable duration, coarse on purpose: nobody needs milliseconds on a twenty-minute build.
fn format_elapsed(elapsed: chrono::Duration) -> String {
    let seconds = elapsed.num_seconds().max(0) as u64;
    if seconds == 0 {
        return "less than a second".to_string();
    }
    humantime::format_duration(std::time::Duration::from_secs(seconds)).to_string()
}

/// A single-line excerpt of `text`, whitespace collapsed and clipped to `limit`. For listing a
/// finished task's result without reprinting it.
pub(crate) fn excerpt(text: &str, limit: usize) -> String {
    elide(text, limit)
}

/// Shorten `text` to `limit` characters on a whitespace boundary.
fn elide(text: &str, limit: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= limit {
        return collapsed;
    }
    let clipped: String = collapsed.chars().take(limit).collect();
    let trimmed = match clipped.rfind(char::is_whitespace) {
        Some(space) => &clipped[..space],
        None => clipped.as_str(),
    };
    format!("{}…", trimmed.trim_end())
}

/// Name of the tool the `[Background]` index exists to drive. Without it the index would be a menu
/// with nothing to order from.
pub(crate) const TASK_INDEX_TOOL: &str = "task_list";
#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::background::TaskStatus;

    fn task(status: TaskStatus, outcome: Option<&str>) -> BackgroundTask {
        BackgroundTask {
            id: "7f3a1c22-0000-0000-0000-000000000000".to_string(),
            session_id: Uuid::nil(),
            tool_name: "execute_command".to_string(),
            label: "cargo test --all".to_string(),
            status,
            outcome: outcome.map(str::to_string),
            scratchpad_name: None,
            started_at: Utc::now() - chrono::Duration::seconds(90),
            finished_at: Some(Utc::now()),
            announced_at: None,
            delivered_at: None,
        }
    }

    /// The ceiling has to hold against the sibling calls in one assistant message, which
    /// `execute_tool_calls` dispatches concurrently: counting and then registering separately lets
    /// every call read "zero running" and start.
    #[tokio::test]
    async fn the_ceiling_holds_against_concurrent_reservations() {
        let tasks = BackgroundTasks::default();
        let session_id = Uuid::new_v4();

        let attempts: Vec<_> = (0..8)
            .map(|index| {
                let tasks = tasks.clone();
                tokio::spawn(async move {
                    tasks
                        .try_reserve(
                            format!("task-{index}"),
                            session_id,
                            tokio_util::sync::CancellationToken::new(),
                            3,
                        )
                        .await
                })
            })
            .collect();

        let mut granted = 0;
        for attempt in attempts {
            if attempt.await.expect("join") {
                granted += 1;
            }
        }
        assert_eq!(
            granted, 3,
            "exactly the ceiling, no matter the interleaving"
        );
        assert_eq!(tasks.running_count(session_id).await, 3);
    }

    /// A start that fails after reserving must hand the slot back, or the ceiling shrinks for the
    /// rest of the session.
    #[tokio::test]
    async fn a_released_reservation_frees_its_slot() {
        let tasks = BackgroundTasks::default();
        let session_id = Uuid::new_v4();
        let token = tokio_util::sync::CancellationToken::new;

        assert!(
            tasks
                .try_reserve("a".to_string(), session_id, token(), 1)
                .await
        );
        assert!(
            !tasks
                .try_reserve("b".to_string(), session_id, token(), 1)
                .await
        );

        tasks.forget("a").await;
        assert!(
            tasks
                .try_reserve("b".to_string(), session_id, token(), 1)
                .await
        );
    }

    /// Canceling and leaving is not enough: the task has to be given the chance to act on it.
    ///
    /// `/exit` returns straight into `Runtime::shutdown_background`, which drops every task where
    /// it stands. A task parked at an await is then never polled again, so the cleanup that follows
    /// its cancellation check (killing its process group, writing its terminal row) never happens.
    /// The task here records the same way: it observes the token, then does one more await before
    /// setting the flag.
    #[tokio::test]
    async fn canceling_every_task_and_waiting_lets_them_run_their_cleanup() {
        let tasks = BackgroundTasks::default();
        let session_id = Uuid::new_v4();
        let token = tokio_util::sync::CancellationToken::new();
        let cleaned_up = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        assert!(
            tasks
                .try_reserve("a".to_string(), session_id, token.clone(), 4)
                .await
        );
        let join = tokio::spawn({
            let cleaned_up = std::sync::Arc::clone(&cleaned_up);
            async move {
                token.cancelled().await;
                tokio::task::yield_now().await;
                cleaned_up.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        tasks.attach("a", join).await;

        assert_eq!(tasks.cancel_all().await, 1);
        tasks.wait_all().await;

        assert!(
            cleaned_up.load(std::sync::atomic::Ordering::Acquire),
            "the task must have reached its cleanup before the wait returned",
        );
        assert_eq!(
            tasks.running_count_all().await,
            0,
            "and the registry must be empty afterwards",
        );
    }

    /// The ceiling is per session, so one session filling it must not starve another.
    #[tokio::test]
    async fn the_ceiling_is_per_session() {
        let tasks = BackgroundTasks::default();
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let token = tokio_util::sync::CancellationToken::new;

        assert!(tasks.try_reserve("a".to_string(), first, token(), 1).await);
        assert!(!tasks.try_reserve("b".to_string(), first, token(), 1).await);
        assert!(tasks.try_reserve("c".to_string(), second, token(), 1).await);
    }

    #[test]
    fn status_round_trips_through_its_string() {
        for status in [
            TaskStatus::Running,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Canceled,
            TaskStatus::Interrupted,
        ] {
            assert_eq!(status.name().parse::<TaskStatus>(), Ok(status));
        }
        assert!("wedged".parse::<TaskStatus>().is_err());
    }

    #[test]
    fn render_names_the_task_and_what_happened() {
        let rendered = render_outcomes(&[task(TaskStatus::Completed, Some("42 passed"))]);
        assert!(rendered.contains("7f3a1c22"), "{rendered}");
        assert!(rendered.contains("cargo test --all"), "{rendered}");
        assert!(rendered.contains("finished"), "{rendered}");
        assert!(rendered.contains("42 passed"), "{rendered}");
    }

    /// Each terminal state has to read differently: "your build failed" and "your build never ran"
    /// call for different next moves.
    #[test]
    fn each_terminal_status_reads_differently() {
        let headlines: Vec<String> = [
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Canceled,
            TaskStatus::Interrupted,
        ]
        .into_iter()
        .map(|status| render_outcomes(&[task(status, None)]))
        .collect();
        for (index, first) in headlines.iter().enumerate() {
            for second in headlines.iter().skip(index + 1) {
                assert_ne!(first, second);
            }
        }
    }

    #[test]
    fn several_outcomes_coalesce_into_one_turn() {
        let rendered = render_outcomes(&[
            task(TaskStatus::Completed, Some("one")),
            task(TaskStatus::Failed, Some("two")),
        ]);
        assert!(rendered.contains("tasks (2)"), "{rendered}");
        assert!(
            rendered.contains("one") && rendered.contains("two"),
            "{rendered}"
        );
    }

    /// Outcome text is a shell command's stdout or a sub-agent's prose. Either can carry a bidi
    /// override or a terminal escape, and both are rendered to the user on the way past.
    #[test]
    fn outcome_text_is_sanitized() {
        let rendered = render_outcomes(&[task(
            TaskStatus::Completed,
            Some("done\u{202E}gnihtemos rehto"),
        )]);
        assert!(!rendered.contains('\u{202E}'), "{rendered}");
    }

    #[test]
    fn split_outcome_spills_only_when_oversized() {
        let (inline, spilled) = split_outcome("short");
        assert_eq!(inline, "short");
        assert!(spilled.is_none());

        let long = "x".repeat(OUTCOME_INLINE_LIMIT + 100);
        let (inline, spilled) = split_outcome(&long);
        assert_eq!(inline.len(), OUTCOME_INLINE_LIMIT);
        assert_eq!(spilled.as_deref(), Some(long.as_str()));
    }

    /// The limit bounds what lands in the conversation, which is measured in bytes. Cutting in
    /// characters would let a multi-byte log carry several times the budget inline, and just past
    /// the threshold the "head" would be the whole output.
    #[test]
    fn split_outcome_bounds_multibyte_output_by_bytes() {
        // Over the limit in bytes (three each), comfortably under it in characters.
        let log = "√".repeat(OUTCOME_INLINE_LIMIT / 2);
        assert!(log.len() > OUTCOME_INLINE_LIMIT);
        assert!(log.chars().count() < OUTCOME_INLINE_LIMIT);

        let (inline, spilled) = split_outcome(&log);
        assert!(inline.len() <= OUTCOME_INLINE_LIMIT, "{}", inline.len());
        assert!(
            inline.len() < log.len(),
            "a spilled outcome must not also carry the whole log inline",
        );
        assert_eq!(spilled.as_deref(), Some(log.as_str()));
    }

    #[test]
    fn spill_entry_name_is_unique_per_task() {
        assert_ne!(
            spill_entry_name("7f3a1c22-aaaa", "execute_command"),
            spill_entry_name("91bd0e44-bbbb", "execute_command"),
        );
        assert!(spill_entry_name("7f3a1c22-aaaa", "execute_command").contains("7f3a1c22"));
    }

    #[test]
    fn spilled_outcome_names_its_entry() {
        let mut spilled = task(TaskStatus::Completed, Some("head of the log"));
        spilled.scratchpad_name = Some("task_7f3a1c22_execute_command".to_string());
        let rendered = render_outcomes(&[spilled]);
        assert!(
            rendered.contains("task_7f3a1c22_execute_command"),
            "{rendered}"
        );
    }

    /// Only a cancellation is somebody's deliberate act, so only a cancellation waits.
    ///
    /// Every terminal outcome still reaches the model; this decides whether reaching it is worth a
    /// turn nobody asked for. Written over the whole enum rather than the two cases that motivated
    /// it, so a status added later has to answer the question rather than inherit an answer.
    #[test]
    fn only_a_cancellation_waits_for_a_turn_that_was_happening_anyway() {
        for status in [
            TaskStatus::Running,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Interrupted,
        ] {
            assert!(
                status.wakes_a_host(),
                "{} is nobody's decision, so the agent has to be told when it happens",
                status.name()
            );
        }
        assert!(
            !TaskStatus::Canceled.wakes_a_host(),
            "whoever canceled it already knows, and a stop command must not start a turn"
        );
    }

    /// A caller reports what the stamp gave it, not what it read a moment earlier.
    ///
    /// Two claimers reading the same row as unclaimed is ordinary, the compare-and-swap picks one,
    /// and a loser that reports its snapshot anyway delivers the outcome a second time.
    #[test]
    fn a_report_carries_only_the_outcomes_the_stamp_won() {
        // Distinct ids: the shared fixture hands out a fixed one, and a filter keyed on id cannot
        // be tested with two rows that carry the same one.
        let mut won = task(TaskStatus::Completed, Some("built"));
        won.id = "11111111-0000-0000-0000-000000000000".to_string();
        let mut lost = task(TaskStatus::Canceled, None);
        lost.id = "22222222-0000-0000-0000-000000000000".to_string();
        let read = vec![won.clone(), lost.clone()];

        let reported = only_what_was_won(read.clone(), &[won.id.clone()]);
        assert_eq!(
            reported.iter().map(|task| &task.id).collect::<Vec<_>>(),
            vec![&won.id],
            "the row this caller lost must not be reported by it"
        );
        assert!(
            only_what_was_won(read.clone(), &[]).is_empty(),
            "a caller that won nothing reports nothing"
        );
        assert_eq!(
            only_what_was_won(read, &[won.id, lost.id]).len(),
            2,
            "and winning everything reports everything"
        );
    }

    /// A prompt carrying an outcome is never withdrawn, whatever the job asked for.
    ///
    /// The branch only matters when a turn fails, so nothing that drives a successful turn can see
    /// it. Withdrawing a carrier prompt destroys the only copy of the outcome: the row was stamped
    /// delivered before the turn began and `list_undelivered_background_tasks` never returns it
    /// again.
    #[test]
    fn a_prompt_carrying_an_outcome_is_never_withdrawn() {
        let carried = [task(TaskStatus::Canceled, None)];
        for asked in [
            crate::conversation::PromptRetention::Keep,
            crate::conversation::PromptRetention::WithdrawOnFailure,
        ] {
            assert_eq!(
                retention_carrying(&carried, asked),
                crate::conversation::PromptRetention::Keep,
                "an outcome rides on this prompt, so a failure must not take it with it"
            );
        }
        assert_eq!(
            retention_carrying(&[], crate::conversation::PromptRetention::WithdrawOnFailure),
            crate::conversation::PromptRetention::WithdrawOnFailure,
            "and a job carrying only its own prompt keeps the job's answer: the next occurrence \
             regenerates it"
        );
        assert_eq!(
            retention_carrying(&[], crate::conversation::PromptRetention::Keep),
            crate::conversation::PromptRetention::Keep
        );
    }

    /// The notice rides on a prompt rather than standing as a message of its own.
    ///
    /// A lone user message opens a turn (`crate::conversation::opens_turn`), so one nobody answers
    /// is a boundary with no turn behind it: `/rewind 1` cuts there instead of at the user's last
    /// exchange, compaction reads the conversation as ending on an unanswered prompt, and the next
    /// real prompt goes out as a second consecutive user turn.
    #[test]
    fn an_outcome_joins_a_prompt_rather_than_becoming_one() {
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::new_v4(),
            tool_name: "execute_command".to_string(),
            label: "sleep 900".to_string(),
            status: TaskStatus::Canceled,
            outcome: None,
            scratchpad_name: None,
            announced_at: None,
            started_at: chrono::Utc::now(),
            finished_at: Some(chrono::Utc::now()),
            delivered_at: None,
        };
        let joined = render_outcomes_riding(std::slice::from_ref(&task));

        assert!(
            joined.contains("was canceled"),
            "the outcome has to be in it: {joined}"
        );
        assert!(
            !joined.contains("Pick the work back up from here"),
            "the standalone trailer instructs the model to resume the canceled work, which is \
             wrong above somebody else's question: {joined}"
        );
        assert!(
            render_outcomes(std::slice::from_ref(&task))
                .contains("Pick the work back up from here"),
            "while the standalone form, which is the whole prompt, still says it"
        );
    }
}
