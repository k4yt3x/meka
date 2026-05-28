//! Process-wide shared state for `agsh serve`. Owns [`SharedDeps`] (provider, MCP, session DB,
//! skill cache — identical to the ACP path), the auth registry, and the per-session map. Held
//! behind an `Arc` and cloned into every axum handler via the `State` extractor.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::sync::RwLock;
use uuid::Uuid;

use super::{
    errors::{ErrorKind, ProblemDetail},
    http_frontend::HttpFrontend,
    idempotency::IdempotencyCache,
};
use crate::{
    SharedDeps, agent::SharedCwd, conversation::Conversation, permission::SharedPermission,
};

/// Top-level server state. Cloned by `Arc` reference into handlers; mutation goes through inner
/// `RwLock`s on the fields that need it.
#[derive(Clone)]
pub struct ServerState {
    pub shared: Arc<SharedDeps>,
    pub sessions: Arc<RwLock<HashMap<Uuid, SessionEntry>>>,
    /// Configured serve settings, post-resolve (defaults filled, env vars substituted).
    pub config: Arc<crate::config::ResolvedServeConfig>,
    /// Stripe-style `Idempotency-Key` cache; spans the whole process. `POST /turn` consults it
    /// before doing any real work.
    pub idempotency: IdempotencyCache,
    /// Process-wide count of in-flight turns. Inspected by `submit_turn` for the
    /// `max_concurrent_turns` cap; incremented + decremented via [`TurnGuard`].
    pub concurrent_turns: Arc<AtomicUsize>,
    /// Cancellation token fired when the process receives SIGTERM / SIGINT. Streaming
    /// turn handlers watch it via `tokio::select!` and emit a final
    /// `turn.cancelled{reason:"server_shutdown"}` SSE event before closing. Per-session
    /// `cancellation` tokens are *also* fired during shutdown so the agent loop unwinds.
    pub shutdown: tokio_util::sync::CancellationToken,
}

/// Per-session map entry. Most mutable state lives behind nested locks so cancel / mode /
/// close handlers can act without waiting on the runtime mutex an in-flight turn holds.
#[derive(Clone)]
pub struct SessionEntry {
    pub session_uuid: Uuid,
    /// Fingerprint of the bearer token that created this session. Used for per-token
    /// idempotency cache keying and observability. Not the raw token — the SHA-256 fingerprint
    /// already on [`crate::server::auth::Principal::token_id`], safe to log.
    ///
    /// Persisted to the `sessions` row at create time and restored on re-attach. `None` only
    /// on legacy rows written before the `token_id` column existed.
    #[allow(
        dead_code,
        reason = "persisted at create time and restored on re-attach for observability"
    )]
    pub token_id: Option<String>,
    /// Session-level runtime mutex. Held for the duration of a turn via `try_lock` rejection on
    /// the `turn-in-flight` path.
    pub runtime: Arc<tokio::sync::Mutex<SessionRuntime>>,
    /// Permission cell, hoisted out of the runtime mutex so `PATCH /sessions/{id}` can
    /// flip it without contending with a long-running turn.
    pub permission: SharedPermission,
    /// Per-session working directory, hoisted for the same reason.
    pub cwd: SharedCwd,
    /// Wall-clock creation time, captured at the start of `POST /v1/sessions`. Surfaced in
    /// session-record responses so clients can sort / display ages without a separate query
    /// to the DB row.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Wall-clock time of the last touch (end of a successful turn). Drives the
    /// `updated_at` field on session-record responses. The monotonic `last_turn_at` is
    /// retained alongside because the GC scanner needs a monotonic clock to dodge wall-clock
    /// adjustments.
    pub updated_at: Arc<std::sync::RwLock<chrono::DateTime<chrono::Utc>>>,
    /// `std::sync::RwLock` over `parking_lot` to keep a consistent vocabulary across the
    /// codebase; the guard is never held across an `.await`.
    pub last_turn_at: Arc<std::sync::RwLock<Instant>>,
    /// Wall-clock companion to `last_turn_at`. The GC scanner uses the monotonic `Instant`
    /// (immune to wall-clock jumps), but API responses need a representable timestamp.
    pub last_turn_at_wall: Arc<std::sync::RwLock<Option<chrono::DateTime<chrono::Utc>>>>,
    /// Per-session capability flags resolved at creation (or re-attach). Surfaced on
    /// `SessionResponse` so clients can introspect their session's wire-shape settings.
    pub capabilities: super::http_frontend::SessionCapabilities,
    /// The session's `HttpFrontend`, kept as a typed `Arc` so the turn handler can call
    /// `drain()` after `run_turn` returns. The same `Arc` (cast to `Arc<dyn Frontend>`) is
    /// also held by `runtime.agent` — both point at the same instance.
    pub frontend: Arc<HttpFrontend>,
    /// In-flight turn's cancellation token. Written by the turn handler at the start of every
    /// turn, read by `POST /cancel`. The handler cancels this token to interrupt the running
    /// turn; if no turn is in flight the cancel is a no-op (the next turn that starts will
    /// install a fresh token).
    pub cancellation: Arc<std::sync::RwLock<tokio_util::sync::CancellationToken>>,
    /// Held only for its Drop side-effect (releasing the per-session OS file lock on session
    /// eviction). See `SessionLock` at `src/session.rs:76`.
    #[allow(dead_code, reason = "RAII guard: held for Drop, never read")]
    pub session_lock: Arc<crate::session::SessionLock>,
    /// Number of turns currently executing on this session. Bumped + decremented via
    /// [`TurnGuard`]. The GC scanner consults this so a long-running turn whose previous
    /// `last_turn_at` is older than the idle timeout can't be evicted out from under itself.
    pub in_flight: Arc<AtomicUsize>,
}

/// Per-session state held under the runtime mutex. Everything that needs out-of-band access
/// lives on [`SessionEntry`] directly.
pub struct SessionRuntime {
    pub session_uuid: Uuid,
    pub messages: Conversation,
    pub agent: crate::agent::Agent,
    /// Held so `delete_session` / `gc::evict_idle` can call
    /// `McpClientManager::detach_registry` and stop `tools/list_changed` notifications from
    /// targeting this dead session. Not otherwise read.
    pub tool_registry: crate::tools::ToolRegistry,
}

impl ServerState {
    pub fn new(
        shared: Arc<SharedDeps>,
        config: Arc<crate::config::ResolvedServeConfig>,
        idempotency: IdempotencyCache,
    ) -> Self {
        Self {
            shared,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            idempotency,
            concurrent_turns: Arc::new(AtomicUsize::new(0)),
            shutdown: tokio_util::sync::CancellationToken::new(),
        }
    }
}

impl SessionEntry {
    /// Touch `last_turn_at` so the GC scanner doesn't evict this session, plus `updated_at`
    /// for the user-facing wall-clock timestamp. Called at the end of each successful turn.
    pub fn touch(&self) {
        let now_wall = chrono::Utc::now();
        *super::poisoned::write(&self.last_turn_at, "session::touch::last_turn_at") =
            Instant::now();
        *super::poisoned::write(&self.last_turn_at_wall, "session::touch::last_turn_at_wall") =
            Some(now_wall);
        *super::poisoned::write(&self.updated_at, "session::touch::updated_at") = now_wall;
    }

    /// True iff `last_turn_at` is older than the configured idle timeout *and* no turn is
    /// currently in flight. Used by the GC scanner. `0` timeout disables eviction (always
    /// returns false). The in-flight check ensures a long-running turn whose previous
    /// `last_turn_at` is stale never gets evicted out from under itself.
    pub fn is_idle(&self, timeout: Duration) -> bool {
        if timeout.is_zero() {
            return false;
        }
        if self.in_flight.load(Ordering::Acquire) > 0 {
            return false;
        }
        let last = *super::poisoned::read(&self.last_turn_at, "session::is_idle");
        last.elapsed() >= timeout
    }
}

/// RAII guard tracking one in-flight turn. Construction bumps both the process-wide and the
/// per-session counters (after enforcing `max_concurrent_turns`); `Drop` decrements them.
/// Hold across the whole `run_turn` invocation so that the SSE response stream and the GC
/// scanner both see a consistent picture.
///
/// Per spec: exceeding the cap returns `429` + `https://agsh.dev/errors/concurrency-limit`
/// with a `Retry-After` header.
#[must_use = "dropping the guard immediately defeats the in-flight tracking"]
pub struct TurnGuard {
    process: Arc<AtomicUsize>,
    session: Arc<AtomicUsize>,
}

impl TurnGuard {
    /// Acquire a guard, enforcing the optional process-wide cap. On overflow, returns a
    /// `ProblemDetail` carrying the suggested `Retry-After` value (set to 1 second — the
    /// in-flight tracker decreases as soon as any other turn finishes).
    // `ProblemDetail` is ~144 bytes; for a path that fires at most once per request the
    // extra stack space is fine and boxing would just shuffle the allocation to the heap.
    #[allow(clippy::result_large_err)]
    pub fn acquire(
        process_counter: Arc<AtomicUsize>,
        session_counter: Arc<AtomicUsize>,
        max_concurrent: Option<usize>,
    ) -> Result<Self, ProblemDetail> {
        // Two-phase admission: fetch_add unconditionally, then re-check. Two callers racing
        // both seeing `current == cap - 1` would both pass a plain load+check, but the post-
        // increment re-check catches the overshoot and rolls back the second one. This avoids
        // a compare_exchange loop without sacrificing correctness.
        let prior = process_counter.fetch_add(1, Ordering::AcqRel);
        if let Some(cap) = max_concurrent
            && prior >= cap
        {
            process_counter.fetch_sub(1, Ordering::AcqRel);
            return Err(ProblemDetail::new(
                ErrorKind::ConcurrencyLimit,
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "process-wide concurrent-turn limit of {} reached; retry shortly",
                    cap
                ),
            )
            .with_retry_after(1));
        }
        session_counter.fetch_add(1, Ordering::AcqRel);
        Ok(Self {
            process: process_counter,
            session: session_counter,
        })
    }
}

impl Drop for TurnGuard {
    fn drop(&mut self) {
        self.process.fetch_sub(1, Ordering::AcqRel);
        self.session.fetch_sub(1, Ordering::AcqRel);
    }
}
