//! The `[serve]` table: what the HTTP host listens on, how it authenticates, and where it announces
//! outcomes.

use std::time::Duration;

use crate::config::{ServeConfig, ServeTokenConfig, WebhookConfig, substitute_env};
pub(crate) use crate::host::session::{DEFAULT_GC_SCAN_INTERVAL, DEFAULT_IDLE_TIMEOUT};
/// `[serve].shutdown_drain_timeout` when unset.
const DEFAULT_SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// `[serve].stream_reattach_grace` when unset.
const DEFAULT_STREAM_REATTACH_GRACE: Duration = Duration::from_secs(30);
/// `[serve].max_body_bytes` when unset.
const DEFAULT_MAX_BODY_BYTES: usize = 10 * crate::text::MIB;
/// `[[serve.webhooks]].timeout` when unset.
const DEFAULT_WEBHOOK_TIMEOUT: Duration = Duration::from_secs(10);

/// Scheme + host of a webhook URL, for log lines that run at the default verbosity.
///
/// The path is dropped because it frequently *is* the secret (Slack, Discord, and every other
/// "unguessable URL" webhook). Enough to tell two endpoints apart; not enough to call one.
///
/// Parsed rather than split on `"://"` and `'/'`: a URL has three other places to keep a secret (a
/// query, a fragment, and userinfo), and reconstructing from the parsed components keeps only what
/// is named here, so a component nobody thought of cannot ride along.
pub(crate) fn webhook_host(url: &str) -> String {
    let Ok(parsed) = url::Url::parse(url) else {
        return "<malformed>".to_string();
    };
    let Some(host) = parsed.host_str() else {
        return "<malformed>".to_string();
    };
    match parsed.port() {
        Some(port) => format!("{}://{}:{}", parsed.scheme(), host, port),
        None => format!("{}://{}", parsed.scheme(), host),
    }
}
/// Upper bound on `[[serve.webhooks]] max_retries`. See where it is applied for the reasoning.
const MAX_WEBHOOK_RETRIES: u32 = 10;
/// Validated webhook endpoint, with `${ENV}` substitution applied and `secret_file` loaded.
#[derive(Clone)]
pub(crate) struct ResolvedWebhook {
    pub(crate) url: String,
    pub(crate) secret: Option<String>,
    pub(crate) events: Vec<String>,
    pub(crate) timeout: std::time::Duration,
    pub(crate) max_retries: u32,
}
// Manual `Debug` so a configured secret never reaches a log line, matching `ResolvedServeToken`.
impl std::fmt::Debug for ResolvedWebhook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedWebhook")
            .field("url", &self.url)
            .field(
                "secret",
                &self
                    .secret
                    .as_ref()
                    .map(|secret| format_args!("[REDACTED len={}]", secret.len()).to_string()),
            )
            .field("events", &self.events)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}
impl ResolvedWebhook {
    fn resolve(raw: WebhookConfig) -> Result<Self, String> {
        if raw.url.trim().is_empty() {
            return Err("[serve.webhooks] entry has an empty `url`".into());
        }
        let (url, _) = substitute_env(&raw.url)?;
        if !url.starts_with("https://") && !url.starts_with("http://") {
            return Err(format!(
                "[serve.webhooks] url '{url}' must start with https:// or http://"
            ));
        }
        let secret = match (raw.secret, raw.secret_file) {
            (Some(_), Some(_)) => {
                return Err(
                    "[serve.webhooks] entry has both `secret` and `secret_file`; pick one".into(),
                );
            }
            (Some(inline), None) => {
                let (resolved, _) = substitute_env(&inline)?;
                Some(resolved)
            }
            (None, Some(path)) => {
                let resolved = std::fs::read_to_string(&path)
                    .map(|contents| contents.trim().to_string())
                    .map_err(|error| {
                        format!("failed to read `secret_file` {}: {}", path.display(), error)
                    })?;
                warn_if_world_readable(&path);
                Some(resolved)
            }
            (None, None) => None,
        };
        if raw.events.is_empty() {
            #[cfg(feature = "serve")]
            let known = crate::host::http::webhook::WebhookEvent::ALL.join(", ");
            #[cfg(not(feature = "serve"))]
            let known = String::from("the events a `serve` build lists");
            return Err(format!(
                "[serve.webhooks] entry for '{url}' subscribes to no events; set `events` to one \
                 or more of: {known}"
            ));
        }
        // Validated only when the server is built: the event names are the server's own table,
        // and a build without `serve` parses the section so one config serves every build.
        #[cfg(feature = "serve")]
        for event in &raw.events {
            if !crate::host::http::webhook::WebhookEvent::ALL.contains(&event.as_str()) {
                // Rejected rather than warned, unlike an unknown token scope: a scope that grants
                // nothing still leaves the token working for whatever else it holds, whereas an
                // endpoint whose only subscription is a typo is silently never called at all.
                return Err(format!(
                    "[serve.webhooks] {}",
                    crate::text::unknown_name(
                        "event",
                        event,
                        crate::host::http::webhook::WebhookEvent::ALL
                    )
                ));
            }
        }
        if secret.as_deref().is_some_and(str::is_empty) {
            // An empty key still produces a well-formed HMAC, so the receiver would see a valid
            // `X-Meka-Signature` computed over nothing and the "unsigned deliveries" warning would
            // not fire. Rejected rather than treated as absent: reaching here means the operator
            // meant to sign, and an env var that resolved to empty is the likeliest cause.
            return Err(format!(
                "[serve.webhooks] entry for '{url}' has an empty `secret`; set it to a key"
            ));
        }
        if raw.timeout == Some(std::time::Duration::ZERO) {
            // reqwest treats a zero timeout as "already elapsed", so every delivery fails its
            // whole retry schedule and the only symptom is a warn per event with no hint why.
            return Err(format!(
                "[serve.webhooks] entry for '{url}' has `timeout = \"0s\"`, which fails every \
                 delivery; omit the field for the default ({})",
                humantime_serde::re::humantime::format_duration(DEFAULT_WEBHOOK_TIMEOUT)
            ));
        }
        Ok(Self {
            url,
            secret,
            events: raw.events,
            timeout: raw.timeout.unwrap_or(DEFAULT_WEBHOOK_TIMEOUT),
            // Clamped rather than trusted: each retry holds a task and sleeps, so a config typo of
            // `max_retries = 100000` would keep one alive for days against an endpoint that is
            // plainly not coming back. Ten attempts spans about two minutes of backoff.
            max_retries: raw.max_retries.unwrap_or(3).min(MAX_WEBHOOK_RETRIES),
        })
    }
}
/// `[serve].cors_allowed_origins` once validated: which browser origins are answered with a CORS
/// grant. Absent (the key omitted or empty) means no CORS middleware at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CorsOrigins {
    /// `["*"]`: every origin, sent as the literal wildcard.
    Any,
    /// Exact origins in their ASCII serialization, deduplicated and sorted.
    Exact(Vec<String>),
}
impl CorsOrigins {
    fn resolve(raw: Option<Vec<String>>) -> Result<Option<Self>, String> {
        let raw = raw.unwrap_or_default();
        if raw.is_empty() {
            return Ok(None);
        }
        if raw.iter().any(|entry| entry.trim() == "*") {
            // The named entries beside a `*` would be dead, and a list written that way suggests
            // the operator expected something narrower than "any origin".
            if raw.iter().any(|entry| entry.trim() != "*") {
                return Err(
                    "[serve] `cors_allowed_origins` lists `*` beside other origins; `*` \
                            must be the only entry"
                        .into(),
                );
            }
            return Ok(Some(Self::Any));
        }
        let mut origins = raw
            .iter()
            .map(|entry| resolve_cors_origin(entry))
            .collect::<Result<Vec<_>, _>>()?;
        origins.sort();
        origins.dedup();
        Ok(Some(Self::Exact(origins)))
    }
}
/// One `cors_allowed_origins` entry as a browser sends it in `Origin`.
///
/// The URL parser accepts more than an origin: a path, a query, userinfo and a `*` inside the host
/// all parse, and each would normalize into an entry no browser ever presents. Refused by name so
/// the operator learns at startup that the entry is dead, rather than from a browser's opaque
/// failure. The ASCII serialization is what browsers send: lowercase host, default port dropped,
/// IPv6 compacted, IDNA punycoded.
fn resolve_cors_origin(raw: &str) -> Result<String, String> {
    let entry = raw.trim();
    if entry.is_empty() {
        return Err("[serve] `cors_allowed_origins` has an empty entry".into());
    }
    if entry == "null" {
        return Err(
            "[serve] `cors_allowed_origins` entry 'null' is the opaque origin and cannot be granted"
                .into(),
        );
    }
    let parsed = url::Url::parse(entry).map_err(|error| {
        format!("[serve] `cors_allowed_origins` entry '{entry}' is not an origin: {error}")
    })?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!(
            "[serve] `cors_allowed_origins` entry '{entry}' must start with https:// or http://"
        ));
    }
    if parsed.host_str().is_some_and(|host| host.contains('*')) {
        return Err(format!(
            "[serve] `cors_allowed_origins` entry '{entry}' is a pattern; list each origin in full"
        ));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!(
            "[serve] `cors_allowed_origins` entry '{entry}' carries credentials; an origin is a \
             scheme, host and port"
        ));
    }
    let extra = if parsed.path() != "/" {
        Some("a path")
    } else if parsed.query().is_some() {
        Some("a query")
    } else if parsed.fragment().is_some() {
        Some("a fragment")
    } else {
        None
    };
    if let Some(extra) = extra {
        return Err(format!(
            "[serve] `cors_allowed_origins` entry '{entry}' has {extra}; an origin is a scheme, \
             host and port"
        ));
    }
    Ok(parsed.origin().ascii_serialization())
}
/// Validated, defaults-filled view of [`ServeConfig`]. Constructed at config-load time by
/// [`ResolvedServeConfig::resolve`].
// Individual fields mirror [`ServeConfig`]; see its per-field documentation.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedServeConfig {
    pub(crate) bind: String,
    pub(crate) cors_allowed_origins: Option<CorsOrigins>,
    pub(crate) idle_timeout: std::time::Duration,
    pub(crate) gc_scan_interval: std::time::Duration,
    pub(crate) delete_on_idle: bool,
    pub(crate) shutdown_drain_timeout: std::time::Duration,
    pub(crate) max_concurrent_turns: Option<usize>,
    pub(crate) max_body_bytes: usize,
    /// Whether a 502 carries the provider's own response as a `provider_response` member. On
    /// unless turned off.
    pub(crate) relay_provider_errors: bool,
    /// Whether `/v1/docs` and `/v1/openapi.json` are served. Off unless asked for.
    pub(crate) docs: bool,
    /// How many SSE events per turn to retain for `Last-Event-ID` replay.
    pub(crate) stream_replay_events: usize,
    /// How long a streaming turn keeps running after its SSE consumer disconnects, waiting for a
    /// reconnect. Zero cancels the turn the moment the stream drops.
    pub(crate) stream_reattach_grace: std::time::Duration,
    pub(crate) tokens: Vec<ResolvedServeToken>,
    pub(crate) webhooks: Vec<ResolvedWebhook>,
}
/// Validated token entry. The `token` field carries the final secret value with `${ENV}`
/// substitution applied and `token_file` contents loaded; call sites compare against this
/// directly via constant-time equality.
#[derive(Clone)]
pub(crate) struct ResolvedServeToken {
    pub(crate) token: String,
    pub(crate) description: Option<String>,
    pub(crate) scopes: Vec<String>,
    /// Where the token value originated, used at startup to nudge operators away from inline
    /// plaintext. See [`TokenSource`]. Not security-sensitive itself.
    pub(crate) source: TokenSource,
}
impl std::fmt::Debug for ResolvedServeToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedServeToken")
            .field(
                "token",
                &format_args!("[REDACTED len={}]", self.token.len()),
            )
            .field("description", &self.description)
            .field("scopes", &self.scopes)
            .field("source", &self.source)
            .finish()
    }
}
/// Provenance of a configured token. Determines whether `meka serve` emits a `warn!` at startup
/// (inline plaintext) or file-backed.
#[derive(Debug, Clone)]
pub(crate) enum TokenSource {
    /// Literal value in `token = "..."` with no `${ENV}` markers, discouraged outside
    /// development.
    Inline,
    /// `token = "${ENV_VAR}"` substituted at config-load time.
    EnvVar,
    /// `token_file = "/path/to/token"` read at config-load time.
    File,
}
impl ResolvedServeConfig {
    /// Resolve a [`ServeConfig`] (or its absence) into a [`ResolvedServeConfig`] with all
    /// defaults filled and tokens read from disk / env. Errors are returned for caller
    /// configuration problems (both `token` and `token_file` set, env var unset, file missing).
    pub(crate) fn resolve(raw: Option<ServeConfig>) -> Result<Self, String> {
        let raw = raw.unwrap_or_default();
        // Read before the fields below consume `raw`, and through the shared reading so `meka acp`
        // and `meka serve` cannot come to disagree about what a deployment configured to withhold
        // withholds.
        let relay_provider_errors = crate::host::relay_provider_errors(Some(&raw));
        // A zero `max_*` is refused at config time: it would 429 every turn or 413 every request,
        // and "unbounded" is spelled by omitting the field.
        if let Some(0) = raw.max_concurrent_turns {
            return Err(
                "[serve] `max_concurrent_turns = 0` would refuse every turn; omit the field to \
                 disable the cap"
                    .into(),
            );
        }
        if let Some(0) = raw.max_body_bytes {
            return Err(format!(
                "[serve] `max_body_bytes = 0` would refuse every request body; omit the field for \
                 the default ({})",
                crate::text::format_size(DEFAULT_MAX_BODY_BYTES)
            ));
        }
        if raw.gc_scan_interval == Some(std::time::Duration::ZERO) {
            // `tokio::time::interval(ZERO)` panics, and the GC handle is only ever aborted, never
            // joined, so the panic is swallowed: eviction silently never runs and every session
            // lock is held until the process exits. `[schedule].poll_interval` guards the same
            // shape for the same reason.
            return Err(format!(
                "[serve] `gc_scan_interval = \"0s\"` would stop the session GC from ever running; \
                 omit the field for the default ({})",
                humantime_serde::re::humantime::format_duration(DEFAULT_GC_SCAN_INTERVAL)
            ));
        }
        let tokens = raw
            .tokens
            .unwrap_or_default()
            .into_iter()
            .map(ResolvedServeToken::resolve)
            .collect::<Result<Vec<_>, _>>()?;
        let webhooks = raw
            .webhooks
            .unwrap_or_default()
            .into_iter()
            .map(ResolvedWebhook::resolve)
            .collect::<Result<Vec<_>, _>>()?;
        let cors_allowed_origins = CorsOrigins::resolve(raw.cors_allowed_origins)?;
        for webhook in &webhooks {
            if webhook.secret.is_none() {
                tracing::warn!(
                    // Host only. `warn` is the default level, so this line lands in logs an
                    // operator may well paste elsewhere, and for a Slack- or Discord-style
                    // endpoint the URL path *is* the credential. The full URL is at `info`.
                    endpoint = %webhook_host(&webhook.url),
                    "webhook has no `secret`, so its deliveries are unsigned",
                );
            }
            if webhook.url.starts_with("http://") {
                tracing::warn!(
                    endpoint = %webhook_host(&webhook.url),
                    "webhook uses plaintext http; deliveries are signed but not encrypted",
                );
            }
        }
        Ok(Self {
            bind: raw.bind.unwrap_or_else(|| "127.0.0.1:8080".to_string()),
            cors_allowed_origins,
            idle_timeout: raw.idle_timeout.unwrap_or(DEFAULT_IDLE_TIMEOUT),
            gc_scan_interval: raw.gc_scan_interval.unwrap_or(DEFAULT_GC_SCAN_INTERVAL),
            delete_on_idle: raw.delete_on_idle.unwrap_or(false),
            shutdown_drain_timeout: raw
                .shutdown_drain_timeout
                .unwrap_or(DEFAULT_SHUTDOWN_DRAIN_TIMEOUT),
            max_concurrent_turns: raw.max_concurrent_turns,
            max_body_bytes: raw.max_body_bytes.unwrap_or(DEFAULT_MAX_BODY_BYTES),
            relay_provider_errors,
            docs: raw.docs.unwrap_or(false),
            // Matches the broadcast channel's capacity: retaining more than the live channel can
            // buffer would let a client replay events a *connected* consumer would have been
            // dropped for missing.
            stream_replay_events: raw.stream_replay_events.unwrap_or(256),
            stream_reattach_grace: raw
                .stream_reattach_grace
                .unwrap_or(DEFAULT_STREAM_REATTACH_GRACE),
            tokens,
            webhooks,
        })
    }
}
impl ResolvedServeToken {
    fn resolve(raw: ServeTokenConfig) -> Result<Self, String> {
        let (token, source) = match (raw.token, raw.token_file) {
            (Some(_), Some(_)) => {
                return Err(
                    "[serve.tokens] entry has both `token` and `token_file`; pick one".into(),
                );
            }
            (Some(inline), None) => {
                let (resolved, substituted) = substitute_env(&inline)?;
                let source = if substituted {
                    TokenSource::EnvVar
                } else {
                    TokenSource::Inline
                };
                (resolved, source)
            }
            (None, Some(path)) => {
                let resolved = std::fs::read_to_string(&path)
                    .map(|s| s.trim().to_string())
                    .map_err(|error| {
                        format!("failed to read `token_file` {}: {}", path.display(), error)
                    })?;
                warn_if_world_readable(&path);
                (resolved, TokenSource::File)
            }
            (None, None) => {
                return Err("[serve.tokens] entry must set either `token` or `token_file`".into());
            }
        };
        if token.is_empty() {
            return Err("[serve.tokens] resolved token is empty".into());
        }
        #[cfg(feature = "serve")]
        crate::host::http::scope::warn_unknown(&raw.scopes, raw.description.as_deref());
        Ok(Self {
            token,
            description: raw.description,
            scopes: raw.scopes,
            source,
        })
    }
}
/// Warn when `path` is readable by group or others on Unix; a no-op elsewhere. A warning rather
/// than a refusal, since the file has already been read.
fn warn_if_world_readable(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(path) {
            let mode = metadata.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                tracing::warn!(
                    "`[serve.tokens].token_file` '{path}' has mode {mode:04o}; run `chmod 0600` \
                     on it",
                    path = path.display(),
                );
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each configured origin ends up as the string a browser sends in `Origin`, so the exact
    /// match at the router compares like with like.
    #[test]
    fn a_cors_origin_is_normalized_to_what_a_browser_sends() {
        for (raw, expected) in [
            ("https://owner.github.io/", "https://owner.github.io"),
            ("HTTPS://MekaWeb.Example:443", "https://mekaweb.example"),
            ("http://localhost:80", "http://localhost"),
            ("http://localhost:5173", "http://localhost:5173"),
            ("https://[0:0:0:0:0:0:0:1]:8443/", "https://[::1]:8443"),
            ("https://bücher.example", "https://xn--bcher-kva.example"),
            ("  https://x.example  ", "https://x.example"),
        ] {
            assert_eq!(
                resolve_cors_origin(raw).as_deref(),
                Ok(expected),
                "normalizing {raw}"
            );
        }
    }

    /// What the URL parser accepts and a browser never sends is refused, and the message names
    /// what is wrong with the entry.
    #[test]
    fn a_cors_entry_that_is_not_an_exact_origin_is_refused() {
        for (raw, probe) in [
            ("", "empty"),
            ("   ", "empty"),
            ("null", "opaque"),
            ("not an origin at all", "not an origin"),
            ("https://x.example:65536", "not an origin"),
            ("ws://x.example", "https://"),
            ("file:///tmp", "https://"),
            ("https://*.example.com", "pattern"),
            ("https://user:pw@x.example", "credentials"),
            ("https://x.example/app", "a path"),
            ("https://x.example/?q", "a query"),
            ("https://x.example/#f", "a fragment"),
        ] {
            let error = resolve_cors_origin(raw).expect_err(raw);
            assert!(error.contains(probe), "{raw:?}: {error}");
        }
    }

    /// The list resolves to off, any, or an exact set; `*` hiding inside a list is refused.
    #[test]
    fn the_cors_list_resolves_to_off_any_or_exact() {
        assert_eq!(CorsOrigins::resolve(None), Ok(None));
        assert_eq!(CorsOrigins::resolve(Some(Vec::new())), Ok(None));
        assert_eq!(
            CorsOrigins::resolve(Some(vec!["*".to_string()])),
            Ok(Some(CorsOrigins::Any))
        );
        assert_eq!(
            CorsOrigins::resolve(Some(vec![
                "https://b.example/".to_string(),
                "https://a.example".to_string(),
                "https://B.example".to_string(),
            ])),
            Ok(Some(CorsOrigins::Exact(vec![
                "https://a.example".to_string(),
                "https://b.example".to_string(),
            ])))
        );
        let error =
            CorsOrigins::resolve(Some(vec!["*".to_string(), "https://a.example".to_string()]))
                .expect_err("mixed");
        assert!(error.contains("only entry"), "{error}");
    }

    /// The webhook redactor keeps the scheme, host and port, and nothing else.
    ///
    /// Every case below is one a splitting implementation lets through, and each reaches a `warn!`
    /// at default verbosity. The query one is the sharpest: a webhook URL with no path has no `/`
    /// to split on.
    #[test]
    fn webhook_host_keeps_only_the_origin() {
        for (url, expected) in [
            (
                "https://hooks.example.com/services/T00/B00/XXXX",
                "https://hooks.example.com",
            ),
            // No path, so nothing for a `'/'` split to cut.
            (
                "https://hooks.example.com?token=SECRET",
                "https://hooks.example.com",
            ),
            (
                "https://hooks.example.com/path?token=SECRET",
                "https://hooks.example.com",
            ),
            (
                "https://hooks.example.com#SECRET",
                "https://hooks.example.com",
            ),
            // Userinfo is part of the authority, and must not survive as part of the host.
            (
                "https://user:pass@hooks.example.com/x",
                "https://hooks.example.com",
            ),
            // A port distinguishes two endpoints and is not a secret, so it is kept.
            ("http://127.0.0.1:8080/hook", "http://127.0.0.1:8080"),
            ("not a url", "<malformed>"),
        ] {
            let rendered = webhook_host(url);
            assert_eq!(rendered, expected, "redacting {url}");
            for secret in ["SECRET", "pass", "T00", "XXXX"] {
                assert!(
                    !rendered.contains(secret),
                    "{rendered} still carries `{secret}` from {url}"
                );
            }
        }
    }
}
