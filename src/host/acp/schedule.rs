//! Schedule execution for `meka acp`.
//!
//! An editor is a live client, so unlike `meka serve` this host has a human on the far side. Two
//! things follow. Approval requests genuinely round-trip (`session/request_permission` reaches
//! the editor), rather than resolving to deny the way an unattended turn must. And the schedule's
//! prompt is pushed as a `UserMessageChunk` before the turn runs, so the transcript shows what
//! triggered the reply instead of an answer to a question nobody asked.
//!
//! ACP fires only schedules whose session the editor currently has open. It cannot revive an
//! arbitrary session the way the server can: a session's cwd, permission and capabilities come from
//! the client's `session/new`, and inventing them here would run a schedule against a workspace the
//! editor never opened. Schedules for other sessions are simply not this host's to run, and the
//! scope predicate filters them out before their gates are ever evaluated.

use std::sync::Arc;

use agent_client_protocol::v1::schema::{ContentBlock, ContentChunk, SessionUpdate, TextContent};

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
    // string form (every insertion site derives the key from `session_uuid.to_string()`), so
    // finding the session needs no lock on any session's runtime.
    let sessions = state.sessions.clone();
    let scope = SchedulerScope::Schedules(Arc::new(move |schedule: &crate::schedule::Schedule| {
        // `try_read` rather than `read`: the map is briefly write-locked on every `session/new` and
        // `session/close`, and blocking the sweep behind session setup would be worse than skipping
        // a tick. The schedule stays due either way.
        sessions
            .try_read()
            .is_ok_and(|open| open.contains_key(&schedule.session_id.to_string()))
    }));

    tracing::info!(
        "scheduler enabled for ACP: poll_interval={poll_interval:?}, open sessions only",
        poll_interval = config.poll_interval
    );
    let store = Arc::new(state.shared.store.clone());
    let inbox_wake = Arc::clone(&state.inbox_wake);
    crate::scheduler::spawn(
        store,
        config,
        state.shared.gate_tools.clone(),
        Arc::new(AcpHooks {
            state: Arc::clone(&state),
        }) as Arc<dyn crate::scheduler::ResidentPermissions>,
        scope,
        // The inbox driver opens the fire's turn when the session is idle; a running turn reads it
        // at its next pause.
        move |_schedule, _queued| {
            inbox_wake.notify_one();
            std::future::ready(())
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
    let sessions = state.sessions.clone();
    // A token nothing fires: the handle is aborted with the connection, which is what ends this
    // poller.
    crate::host::scheduler::spawn_outcome_poller(
        AcpHooks { state },
        sessions,
        poll_interval,
        tokio_util::sync::CancellationToken::new(),
    )
}

/// Start the inbox driver for a running ACP process, over the sessions the editor has open, for the
/// reason the scheduler above keeps to them: a session it closed is not this host's to revive, and
/// its items wait for whichever host opens it next.
pub(super) fn spawn_inbox_driver(state: Arc<super::ServerState>) -> tokio::task::JoinHandle<()> {
    let sessions = state.sessions.clone();
    crate::host::scheduler::spawn_inbox_driver(
        Arc::new(AcpHooks {
            state: Arc::clone(&state),
        }),
        Arc::clone(&state.inbox_wake),
        state.shared.config.schedule.poll_interval,
        // A token nothing fires: the handle is aborted with the connection, which is what ends
        // this driver.
        tokio_util::sync::CancellationToken::new(),
        // `try_read`, as the scheduler's scope reads the map: a pass that finds it write-locked by
        // a `session/new` or a `session/close` leaves the session to the next.
        move |session_id| {
            sessions
                .try_read()
                .is_ok_and(|open| open.contains_key(&session_id.to_string()))
        },
    )
}

/// Build the `session/update` that shows a schedule's prompt as the user turn that triggered the
/// reply.
///
/// Zed skips an echoed `UserMessageChunk` only when it matches a message it optimistically added
/// itself before calling `session/prompt`. A scheduled turn has no such message, so this renders --
/// and without it the editor would show an answer with nothing above it explaining the question.
pub(super) fn out_of_band_prompt_update(prompt: &str) -> SessionUpdate {
    SessionUpdate::UserMessageChunk(ContentChunk::new(ContentBlock::Text(TextContent::new(
        prompt.to_string(),
    ))))
}

/// `meka acp` around an out-of-band turn: only a session the editor has open can run one, and the
/// agent is moved onto the profile the row records first. The prompt reaches the editor as a
/// user-message chunk through the frontend, as every turn's opening does.
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
        crate::host::apply_recorded_bindings(&self.state.shared, &entry.agent, entry.id).await
    }

    fn shutting_down(&self) -> bool {
        self.state.leaving.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn inbox_deferred(
        &self,
        entry: &Self::Entry,
        items: &[crate::store::inbox::InboxItem],
        not_before: chrono::DateTime<chrono::Utc>,
        _error: &crate::error::MekaError,
    ) {
        let notice = crate::host::scheduler::deferral_notice(items, not_before);
        entry
            .frontend
            .emit(crate::frontend::FrontendEvent::Notice(
                crate::frontend::Notice::info(notice),
            ))
            .await;
    }

    async fn inbox_given_up(
        &self,
        entry: &Self::Entry,
        item: &crate::store::inbox::InboxItem,
        reason: &str,
        _error: Option<&crate::error::MekaError>,
    ) {
        let notice = crate::host::scheduler::give_up_notice(item, reason);
        entry
            .frontend
            .emit(crate::frontend::FrontendEvent::Notice(
                crate::frontend::Notice::warn(notice),
            ))
            .await;
    }

    /// An out-of-band turn has no `session/prompt` response to carry its outcome, so a failure
    /// would otherwise leave the editor with a prompt and nothing under it. Said as the notice the
    /// screen prints and the webhook `meka serve` posts; a clean finish needs no announcement,
    /// since the reply is one.
    async fn finished(
        &self,
        entry: &Self::Entry,
        origin: &crate::host::scheduler::TurnStarter,
        outcome: &Result<(), crate::error::MekaError>,
    ) {
        use crate::host::scheduler::TurnStarter;
        let what = match origin {
            TurnStarter::Schedule { schedule_ids, .. } => match schedule_ids.as_slice() {
                [schedule_id] => format!(
                    "Schedule '{}'",
                    schedule_id
                        .get(..crate::text::ID_PREFIX)
                        .unwrap_or(schedule_id)
                ),
                _ => "The scheduled turn".to_string(),
            },
            TurnStarter::Inbox { .. } => "The inbox turn".to_string(),
            _ => "The background outcome report".to_string(),
        };
        let notice = match outcome {
            Ok(()) => return,
            Err(crate::error::MekaError::Interrupted) => {
                crate::frontend::Notice::info(format!("{what} was interrupted."))
            }
            // The words the request door would have given the editor: a provider's body or a
            // store path travels only as `acp_error_for` allows, on this wire as on that one.
            Err(error) => crate::frontend::Notice::warn(format!(
                "{what} failed: {}",
                super::acp_error_text(error, self.state.shared.relay_provider_errors())
            )),
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
