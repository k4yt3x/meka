//! Process-wide shared state for `meka serve`. Owns [`SharedDeps`] (provider, MCP, session DB,
//! skill cache; identical to the ACP path), the auth registry, and the per-session map. Held
//! behind an `Arc` and cloned into every axum handler via the `State` extractor.

use std::sync::{Arc, atomic::AtomicUsize};

use uuid::Uuid;

use super::{
    errors::{ErrorKind, ProblemDetail},
    http_frontend::HttpFrontend,
    idempotency::IdempotencyCache,
};
use crate::host::{ResidentSession, Sessions, SharedDeps};

/// Top-level server state. Cloned by `Arc` reference into handlers; mutation goes through inner
/// `RwLock`s on the fields that need it.
#[derive(Clone)]
pub(crate) struct ServerState {
    pub(crate) shared: Arc<SharedDeps>,
    pub(crate) sessions: Sessions<Uuid, SessionEntry>,
    /// Configured serve settings, post-resolve (defaults filled, env vars substituted).
    pub(crate) config: Arc<crate::host::http::config::ResolvedServeConfig>,
    /// Stripe-style `Idempotency-Key` cache; spans the whole process. `POST /turn` consults it
    /// before doing any real work.
    pub(crate) idempotency: IdempotencyCache,
    /// Process-wide count of in-flight turns. Inspected by `submit_turn` for the
    /// `max_concurrent_turns` cap; incremented + decremented via [`crate::host::TurnGuard`].
    pub(crate) concurrent_turns: Arc<AtomicUsize>,
    /// Cancellation token fired when the process receives SIGTERM / SIGINT.
    ///
    /// What actually stops an in-flight turn is `host::http::drain_active_sessions`, which fires
    /// every per-session `cancellation` token. A streaming turn's task reads *this* one only
    /// to label its terminal event `turn.cancelled{reason:"server_shutdown"}` rather than
    /// `client`.
    pub(crate) shutdown: tokio_util::sync::CancellationToken,
    /// Outbound webhook fan-out. Empty unless `[[serve.webhooks]]` is configured, in which case
    /// every `send` is a no-op, so call sites need no `if configured` guard of their own.
    pub(crate) webhooks: super::webhook::WebhookDispatcher,
    /// One reconstruction of an unloaded session at a time, per id.
    pub(crate) reconstruction_locks: super::reattach::ReconstructionLocks,
}

/// Per-session map entry. Most mutable state lives behind nested locks so cancel / permission /
/// close handlers can act without waiting on the runtime mutex an in-flight turn holds.
/// One session `meka serve` holds open: the shared [`ResidentSession`] plus what only the HTTP
/// host tracks. Derefs to the resident part.
#[derive(Clone)]
pub(crate) struct SessionEntry {
    pub(crate) resident: ResidentSession,
    #[allow(
        dead_code,
        reason = "persisted at create time and restored on re-attach for observability"
    )]
    pub(crate) token_id: Option<String>,
    pub(crate) created_at: chrono::DateTime<chrono::Utc>,
    pub(crate) updated_at: Arc<std::sync::RwLock<chrono::DateTime<chrono::Utc>>>,
    /// Wall-clock twin of the resident's `last_activity`, for the API's `last_turn_at`.
    pub(crate) last_turn_at_wall: Arc<std::sync::RwLock<Option<chrono::DateTime<chrono::Utc>>>>,
    pub(crate) capabilities: super::http_frontend::SessionCapabilities,
    pub(crate) frontend: Arc<HttpFrontend>,
}

impl std::ops::Deref for SessionEntry {
    type Target = ResidentSession;

    fn deref(&self) -> &Self::Target {
        &self.resident
    }
}

impl ServerState {
    pub(crate) fn new(
        shared: Arc<SharedDeps>,
        config: Arc<crate::host::http::config::ResolvedServeConfig>,
        idempotency: IdempotencyCache,
    ) -> Self {
        let config_webhooks = config.webhooks.clone();
        Self {
            shared,
            sessions: Sessions::new(),
            config,
            idempotency,
            concurrent_turns: Arc::new(AtomicUsize::new(0)),
            shutdown: tokio_util::sync::CancellationToken::new(),
            webhooks: super::webhook::WebhookDispatcher::new(config_webhooks),
            reconstruction_locks: super::reattach::ReconstructionLocks::default(),
        }
    }
}

impl SessionEntry {
    /// Record activity on both clocks: the resident's, which the idle sweep reads, and the
    /// wall-clock twins the API reports.
    pub(crate) fn touch(&self) {
        self.resident.touch();
        let now_wall = chrono::Utc::now();
        *crate::sync::write(&self.last_turn_at_wall) = Some(now_wall);
        *crate::sync::write(&self.updated_at) = now_wall;
    }
}

/// Admit a turn on `entry` under the server's process-wide cap, as the problem the client reads
/// when it is refused.
pub(crate) fn admit_turn(
    state: &ServerState,
    entry: &SessionEntry,
) -> Result<crate::host::TurnGuard, ProblemDetail> {
    entry
        .admit_turn(Some((
            &state.concurrent_turns,
            state.config.max_concurrent_turns,
        )))
        .map_err(|refused| {
            ProblemDetail::new(
                ErrorKind::ConcurrencyLimit,
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "process-wide concurrent-turn limit of {} reached; retry shortly",
                    refused.cap
                ),
            )
            .with_retry_after(1)
        })
}
