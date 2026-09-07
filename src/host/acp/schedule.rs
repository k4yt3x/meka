//! Scheduled-job execution for `meka acp`.
//!
//! An editor is a live client, so unlike `meka serve` this host has a human on the far side. Two
//! things follow. Approval requests genuinely round-trip (`session/request_permission` reaches
//! the editor), rather than resolving to deny the way an unattended turn must. And the job's prompt
//! is pushed as a `UserMessageChunk` before the turn runs, so the transcript shows what triggered
//! the reply instead of an answer to a question nobody asked.
//!
//! ACP fires only jobs whose session the editor currently has open. It cannot revive an arbitrary
//! session the way the server can: a session's cwd, permission and capabilities come from the
//! client's `session/new`, and inventing them here would run a job against a workspace the editor
//! never opened. Jobs for other sessions are simply not this host's to run, and the scope predicate
//! filters them out before their gates are ever evaluated.

use std::sync::Arc;

use agent_client_protocol::schema::v1::{ContentBlock, ContentChunk, SessionUpdate, TextContent};

use crate::{frontend::Frontend, scheduler::SchedulerScope};

/// Start the scheduler for a running ACP process. Returns the handle so the caller can abort it on
/// shutdown, matching how `meka serve` treats its own.
pub(super) fn spawn(state: Arc<super::ServerState>) -> tokio::task::JoinHandle<()> {
    let config = state.shared.config.schedule.clone();
    if !config.enabled {
        tracing::info!("scheduler disabled ([schedule] enabled = false)");
        return tokio::spawn(async {});
    }

    // Re-asked every sweep, so a session opened or closed since the last tick is accounted for
    // without the scheduler holding a stale snapshot. The map is keyed by the session's uuid in
    // string form (every insertion site derives the key from `session_uuid.to_string()`), so this
    // needs no lock on any session's runtime.
    let sessions = state.sessions.clone();
    let scope = SchedulerScope::Jobs(Arc::new(move |job: &crate::schedule::ScheduledJob| {
        // `try_read` rather than `read`: the map is briefly write-locked on every `session/new` and
        // `session/close`, and blocking the sweep behind session setup would be worse than skipping
        // a tick. The job stays due either way.
        sessions
            .try_read()
            .map(|open| open.contains_key(&job.session_id.to_string()))
            .unwrap_or(false)
    }));

    tracing::info!(
        "scheduler enabled for ACP: poll_interval={poll_interval:?}, open sessions only",
        poll_interval = config.poll_interval
    );
    let store = Arc::new(state.shared.store.clone());
    let hooks = Arc::new(AcpHooks {
        state: Arc::clone(&state),
    });
    crate::scheduler::spawn(
        store,
        config,
        state.shared.gate_tools.clone(),
        Arc::clone(&hooks) as Arc<dyn crate::scheduler::ResidentPermissions>,
        scope,
        move |wakeup| {
            let hooks = Arc::clone(&hooks);
            async move { crate::host::scheduler::run_wakeup(&*hooks, wakeup).await }
        },
    )
}

/// Start the background-outcome poller for a running ACP process.
///
/// Same limit as the scheduler above: only sessions the editor currently has open. A session it
/// closed is not this host's to revive, and the outcome keeps its `delivered_at IS NULL` until
/// something opens it again.
pub(super) fn spawn_background_poller(
    state: Arc<super::ServerState>,
) -> tokio::task::JoinHandle<()> {
    if !state.shared.config.background.enabled {
        return tokio::spawn(async {});
    }
    let poll_interval = state.shared.config.schedule.poll_interval;
    tracing::info!("background tasks enabled for ACP: open sessions only");
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(poll_interval);
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let hooks = AcpHooks {
                state: Arc::clone(&state),
            };
            let sweep = std::panic::AssertUnwindSafe(
                crate::host::scheduler::deliver_ready_outcomes(&hooks, &state.sessions),
            );
            if let Err(panic) = futures::FutureExt::catch_unwind(sweep).await {
                let panic = crate::error::panic_message(&*panic);
                tracing::warn!("background outcome delivery panicked ({panic}); continuing");
            }
        }
    })
}

/// Build the `session/update` that shows a job's prompt as the user turn that triggered the reply.
///
/// Zed skips an echoed `UserMessageChunk` only when it matches a message it optimistically added
/// itself before calling `session/prompt`. A scheduled turn has no such message, so this renders --
/// and without it the editor would show an answer with nothing above it explaining the question.
pub(super) fn out_of_band_prompt_update(prompt: &str) -> SessionUpdate {
    SessionUpdate::UserMessageChunk(ContentChunk::new(ContentBlock::Text(TextContent::new(
        prompt.to_string(),
    ))))
}

/// `meka acp` around an out-of-band turn: only a session the editor has open can run one, the
/// agent is moved onto the profile the row records first, and the prompt is pushed to the editor
/// as a user-message chunk so the transcript shows what the turn answered.
struct AcpHooks {
    state: Arc<super::ServerState>,
}

#[async_trait::async_trait]
impl crate::scheduler::ResidentPermissions for AcpHooks {
    async fn live_permission_of(
        &self,
        session_id: uuid::Uuid,
    ) -> Option<crate::permission::Permission> {
        self.state
            .sessions
            .read()
            .await
            .get(&session_id.to_string())
            .map(|entry| entry.agent.cells().permission.get())
    }
}

#[async_trait::async_trait]
impl crate::host::scheduler::HostHooks for AcpHooks {
    type Entry = super::SessionEntry;

    async fn resident(&self, session_id: uuid::Uuid) -> anyhow::Result<Option<Self::Entry>> {
        Ok(self
            .state
            .sessions
            .read()
            .await
            .get(&session_id.to_string())
            .cloned())
    }

    async fn still_resident(&self, entry: &Self::Entry) -> bool {
        self.state
            .sessions
            .read()
            .await
            .get(&entry.id.to_string())
            .is_some_and(|held| std::sync::Arc::ptr_eq(&held.agent, &entry.agent))
    }

    async fn prepare(&self, entry: &Self::Entry) -> anyhow::Result<()> {
        super::apply_recorded_profile(&self.state, &entry.agent, entry.id).await
    }

    fn show_prompt(
        &self,
        entry: &Self::Entry,
        prompt: crate::host::scheduler::OutOfBandPrompt<'_>,
    ) {
        match prompt {
            crate::host::scheduler::OutOfBandPrompt::Outcomes(text) => {
                entry.frontend.push_out_of_band_prompt(text);
            }
            crate::host::scheduler::OutOfBandPrompt::Scheduled(wakeup) => {
                entry.frontend.push_scheduled_prompt(wakeup);
            }
        }
    }

    /// An out-of-band turn has no `session/prompt` response to carry its outcome, so a failure
    /// would otherwise leave the editor with a prompt and nothing under it. Said as the notice the
    /// REPL prints and the webhook `meka serve` posts; a clean finish needs no announcement, since
    /// the reply is one.
    async fn finished(
        &self,
        entry: &Self::Entry,
        job: Option<&crate::schedule::ScheduledJob>,
        outcome: &Result<(), crate::error::MekaError>,
    ) {
        let what = match job {
            Some(job) => format!("scheduled job '{}'", job.short_id()),
            None => "the background outcome report".to_string(),
        };
        let notice = match outcome {
            Ok(()) => return,
            Err(crate::error::MekaError::Interrupted) => {
                crate::frontend::Notice::info(format!("{what} was interrupted"))
            }
            Err(error) => crate::frontend::Notice::warn(format!("{what} failed: {error}")),
        };
        entry
            .frontend
            .emit(crate::frontend::FrontendEvent::Notice(notice))
            .await;
    }

    fn background_enabled(&self) -> bool {
        self.state.shared.config.background.enabled
    }

    fn store(&self) -> &crate::store::Store {
        &self.state.shared.store
    }
}
