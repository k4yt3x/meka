//! The `[serve]` section: the HTTP server's bind address, tokens, webhooks and limits, as
//! `config.toml` states them. Resolved by `crate::host::http::config`.

use super::*;

/// `[serve]` table: HTTP server config for `meka serve`. All fields optional with sensible
/// defaults, but at least one `[[serve.tokens]]` entry is required; the server refuses to start
/// without one.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServeConfig {
    /// Listen address. Default `127.0.0.1:8080`: bind to loopback so a fresh deploy isn't
    /// accidentally world-reachable. Operators front with a reverse proxy (nginx, caddy) for
    /// TLS termination and put a public address there.
    pub(crate) bind: Option<String>,
    /// Browser origins allowed to call the API cross-origin. Omitted or empty (the default) means
    /// no CORS headers at all; `["*"]` allows any origin; otherwise each entry is one exact
    /// origin, `scheme://host[:port]`, normalized at startup.
    ///
    /// Safe to relax because the API authenticates with a bearer header the page sets itself and
    /// never with a cookie, so a page that lacks the token gets a 401 from any origin, and a page
    /// that holds it can use it from anywhere regardless. The allowlist guards only what needs no
    /// token: the health probes, the opt-in OpenAPI document and the body of a 401.
    pub(crate) cors_allowed_origins: Option<Vec<String>>,
    /// Idle-timeout for session eviction. Sessions with no turn activity for this long are dropped
    /// from the in-memory map by the GC scanner. Accepts humantime strings like `"24h"`, `"30m"`,
    /// `"86400s"`. Default `"24h"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) idle_timeout: Option<std::time::Duration>,
    /// How often the GC scanner sweeps the session map. Accepts humantime strings like `"5m"`,
    /// `"300s"`. Default `"5m"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) gc_scan_interval: Option<std::time::Duration>,
    /// When true, GC also deletes the SQLite row for idle sessions; default false (keep the row so
    /// a future request with the same session ID can re-attach, mirroring ACP's `session/load`).
    pub(crate) delete_on_idle: Option<bool>,
    /// On SIGTERM / SIGINT, wait at most this long for in-flight turns to finish before forcibly
    /// aborting. Accepts humantime strings like `"30s"`, `"1m"`. Default `"30s"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) shutdown_drain_timeout: Option<std::time::Duration>,
    /// Process-wide cap on concurrent in-flight turns across all sessions. None = unbounded
    /// (default). Returns 429 with `concurrency-limit` when exceeded.
    pub(crate) max_concurrent_turns: Option<usize>,
    /// Request body size limit (bytes). Default 10 MiB.
    pub(crate) max_body_bytes: Option<usize>,
    /// Whether a 502 carries the failing provider call's error text, as a `provider_response`
    /// extension member. Default `true`. `detail` is meka's own sentence and is identical either
    /// way.
    ///
    /// On, because the upstream's error type is the actionable part of a failed turn and "consult
    /// the server log" is no answer to anyone running against a meka they do not operate. `meka
    /// acp` honors this key too: the same policy decides what a failed turn's `error.data`
    /// carries.
    ///
    /// **What it can expose.** Usually the upstream's response body, which can name the
    /// *operator's* provider account and its rate-limit posture. Not always, though: the member
    /// carries the failing call's error message, and for some failures that is meka's own sentence
    /// about the call rather than anything the provider sent.
    ///
    /// **Who can read it.** `sessions:r`, not just `sessions:w`. Submitting a turn takes the write
    /// scope, but the failure also rides the terminal `turn.failed` event, and `GET
    /// /v1/sessions/{id}/stream` replays that to any reader.
    ///
    /// Turn it off where read-only tokens go to people who may watch a session but are not
    /// entitled to the account behind it. `/errors/mcp-unavailable` is not covered either way: it
    /// reports server names only, and that reason is meka's own subprocess text.
    pub(crate) relay_provider_errors: Option<bool>,
    /// Whether to serve the Swagger UI and the OpenAPI document at `/v1/docs` and
    /// `/v1/openapi.json`. Default `false`.
    ///
    /// Off by default because they are unauthenticated (as are the two health probes, which
    /// publish nothing) and what they publish is the shape of every endpoint the deployment
    /// exposes. That is useful while building a client and pure reconnaissance value once the
    /// deployment is real. Turn it on deliberately, on a deployment where anyone who can reach the
    /// port is entitled to the map.
    pub(crate) docs: Option<bool>,
    /// How many SSE events per turn to retain so a client reconnecting with `Last-Event-ID` can
    /// replay what it missed. Default 256, matching the live broadcast channel's capacity.
    ///
    /// Raising it buys a longer reconnect window at the cost of holding more per-session memory
    /// during a turn; `0` switches replay off, so a reconnect gets only what happens from then on.
    pub(crate) stream_replay_events: Option<usize>,
    /// How long a streaming turn keeps running after its SSE consumer disconnects, waiting for a
    /// reconnect. Accepts humantime strings like `"30s"`. Default `"30s"`.
    ///
    /// `"0s"` cancels the turn the moment the stream drops. That spends fewer provider tokens on
    /// abandoned work, and makes re-attach useful only for turns that already finished.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) stream_reattach_grace: Option<std::time::Duration>,
    /// Bearer tokens configured for this deployment. An empty list is refused at startup: `meka
    /// serve` exits rather than binding a port nothing can authenticate against.
    pub(crate) tokens: Option<Vec<ServeTokenConfig>>,
    /// Outbound webhook endpoints. Empty (the default) means meka never makes an outbound request.
    pub(crate) webhooks: Option<Vec<WebhookConfig>>,
}
/// One entry in `[[serve.webhooks]]`.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct WebhookConfig {
    /// Where to POST. `http://` is accepted for loopback development but logged as a warning:
    /// deliveries are signed, not encrypted, so anything on the path can read them.
    pub(crate) url: String,
    /// Shared secret for the `X-Meka-Signature` HMAC. Supports `${ENV_VAR}` substitution.
    /// Mutually exclusive with `secret_file`.
    pub(crate) secret: Option<String>,
    /// Path to a file whose contents (trimmed) are the secret. chmod 0600 recommended.
    pub(crate) secret_file: Option<std::path::PathBuf>,
    /// Which events to deliver. Required and non-empty: an endpoint subscribed to nothing is
    /// almost certainly a mistake, and silently never firing is the worst way to find out.
    pub(crate) events: Vec<String>,
    /// Per-attempt request timeout. Accepts humantime strings like `"10s"`. Default `"10s"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) timeout: Option<std::time::Duration>,
    /// Retries after the first attempt, with exponential backoff. Default 3.
    pub(crate) max_retries: Option<u32>,
}
// Manual `Debug` so a secret cannot reach a log through the *raw* struct either. Nothing prints
// this today, but the derived one would have been the single unredacted path in the webhook chain,
// and that is exactly the kind of thing a later `dbg!` finds.
impl std::fmt::Debug for WebhookConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebhookConfig")
            .field("url", &self.url)
            .field("secret", &self.secret.as_ref().map(|_| "[REDACTED]"))
            .field("secret_file", &self.secret_file)
            .field("events", &self.events)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}
/// One entry in `[serve.tokens]`. Tokens identify callers; scopes gate what they can do. See the
/// Auth section of the HTTP API docs for the full scope catalog.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServeTokenConfig {
    /// Inline token value. Supports `${ENV_VAR}` substitution at config-load time. Mutually
    /// exclusive with `token_file`.
    pub(crate) token: Option<String>,
    /// Path to a file whose contents (trimmed) are the token. chmod 0600 recommended.
    pub(crate) token_file: Option<std::path::PathBuf>,
    /// Free-form description, surfaced in startup logs. Operators use it to remember which caller
    /// a token belongs to (e.g. "telegram bridge", "ci debug").
    pub(crate) description: Option<String>,
    /// Scopes granted to this token. See the HTTP API docs for the catalog.
    pub(crate) scopes: Vec<String>,
}
// Manual `Debug` for the same reason as [`WebhookConfig`]'s, and more urgently: `ServeConfig` and
// `ResolvedConfig` both derive `Debug` and `ResolvedConfig` owns the whole `[serve]` table, so the
// derived impl made every bearer token reachable from a single `{:?}` on the config. Redacting only
// `ResolvedServeToken` left that path open, because the raw form survives resolution.
impl std::fmt::Debug for ServeTokenConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServeTokenConfig")
            .field(
                "token",
                &self
                    .token
                    .as_ref()
                    .map(|token| format_args!("[REDACTED len={}]", token.len()).to_string()),
            )
            .field("token_file", &self.token_file)
            .field("description", &self.description)
            .field("scopes", &self.scopes)
            .finish()
    }
}
