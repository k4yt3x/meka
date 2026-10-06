//! RFC 9457 Problem Details for HTTP APIs. Every error response from `meka serve` uses this
//! shape, with content type `application/problem+json`. Stable `type` URIs under
//! `https://meka.run/errors/` act as machine-readable error codes that survive HTTP-status
//! collisions (multiple 404 meanings, multiple 409 meanings); see the HTTP API docs for the full
//! catalog.
//!
//! Mid-stream failures (after the SSE response has started) are emitted as an in-band
//! `turn.failed` SSE event carrying the same JSON shape; this module owns the wire type and
//! `axum` integration for the HTTP-level path.

use std::collections::BTreeMap;

use axum::{
    Json,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Serialize;
use serde_json::Value;
use utoipa::ToSchema;

use crate::error::{INTERNAL_SENTENCE, MekaError, bounded_upstream_body};

/// RFC 9457 Problem Details body. The five core members (`type`, `title`, `status`, `detail`,
/// `instance`) are first-class; meka-specific extension members ride in `extensions` and get
/// flattened into the top-level JSON object on serialization.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub(crate) struct ProblemDetail {
    /// Stable URI identifying the error class. Always set; opaque to clients beyond exact
    /// comparison. URIs are documented (not dereferenced); clients should never fetch them.
    #[serde(rename = "type")]
    pub(crate) type_uri: String,
    /// Short, human-readable summary. Stable for a given `type_uri`.
    pub(crate) title: String,
    /// HTTP status code that accompanied this response, mirrored into the body for clients that
    /// only see the body.
    pub(crate) status: u16,
    /// Instance-specific message. May vary between occurrences of the same `type_uri`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detail: Option<String>,
    /// URI (typically a request path) identifying the specific occurrence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) instance: Option<String>,
    /// Extension members (e.g. `session_id`, `request_id`, `retry_after`). Serialized as
    /// top-level JSON fields via `#[serde(flatten)]`.
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub(crate) extensions: BTreeMap<String, Value>,
    /// When present, surfaced as an HTTP `Retry-After: <n>` response header (seconds). Not
    /// serialized into the body; [`Self::with_retry_after`] mirrors it into the `retry_after` body
    /// extension for clients that only parse JSON.
    #[serde(skip)]
    #[schema(value_type = Option<u32>)]
    pub(crate) retry_after_seconds: Option<u32>,
}

impl ProblemDetail {
    /// Whether this problem is of `kind`. The kind travels as its type URI, so this is the one
    /// place that comparison is written.
    pub(crate) fn is(&self, kind: ErrorKind) -> bool {
        self.type_uri == kind.type_uri()
    }

    pub(crate) fn new(error: ErrorKind, status: StatusCode, detail: impl Into<String>) -> Self {
        Self {
            type_uri: error.type_uri().to_string(),
            title: error.title().to_string(),
            status: status.as_u16(),
            detail: Some(detail.into()),
            instance: None,
            extensions: BTreeMap::new(),
            retry_after_seconds: None,
        }
    }

    /// Attach the request path as `instance` (RFC 9457's "URI reference that identifies the
    /// specific occurrence").
    #[must_use]
    pub(crate) fn instance(mut self, instance: impl Into<String>) -> Self {
        self.instance = Some(instance.into());
        self
    }

    /// Attach an extension member. Common keys: `session_id`, `turn_id`, `request_id`,
    /// `retry_after`. Caller is responsible for the value's JSON shape.
    #[must_use]
    pub(crate) fn with(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extensions.insert(key.into(), value.into());
        self
    }

    /// Attach the delay as both the `Retry-After: <seconds>` response header and the `retry_after`
    /// body extension. The spec requires the header on every 429 (concurrency limit, idempotency
    /// cache cap); the extension is for clients that only parse JSON. One method rather than one
    /// per half, so no caller can ship a response the two kinds of client disagree about.
    #[must_use]
    pub(crate) fn with_retry_after(mut self, seconds: u32) -> Self {
        self.retry_after_seconds = Some(seconds);
        self.with("retry_after", Value::from(seconds))
    }

    /// Build a 500 Problem Detail whose body carries a generic message while the full error
    /// detail is logged server-side, so the wire response doesn't leak internal details.
    ///
    /// `context` is a short operator-readable description logged alongside the error.
    pub(crate) fn internal_sanitized(context: &str, error: impl std::fmt::Display) -> Self {
        tracing::error!("{context}: {error}");
        Self::new(
            ErrorKind::Internal,
            StatusCode::INTERNAL_SERVER_ERROR,
            INTERNAL_SENTENCE,
        )
    }

    /// The 422 for a request body that did not parse, from every door that parses one.
    ///
    /// One definition so the doors agree on what a parse failure says. The parser's diagnostic
    /// travels because it names the field at fault, and those names are the wire schema rather than
    /// anything internal: a client sending a retired field is told which one, and the OpenAPI
    /// document says what was expected instead. `what` names the endpoint's body, as in
    /// `"turn"` or `"session patch"`.
    pub(crate) fn invalid_body(what: &str, error: impl std::fmt::Display) -> Self {
        Self::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("invalid {what} request body: {error}"),
        )
    }
}

/// The longest `Retry-After` meka will relay from an upstream: one hour.
///
/// Not a judgment about how long a client should wait, which is the upstream's to make, but a
/// bound on what meka will repeat: `parse_retry_after` returns whatever the header said, and a
/// header saying a year is a broken or hostile upstream steering every client of this server into a
/// stall. An hour is far past any real rate-limit window and far short of that.
const RELAYED_RETRY_AFTER_CAP: std::time::Duration = std::time::Duration::from_secs(60 * 60);

pub(crate) use crate::error::ErrorKind;

/// The status each catalog entry answers with where the kind alone decides it. Two vary by site
/// and take their common one here: `Idempotency` is 429 at the cache cap, and `InvalidBody` is
/// 400 for a path or query the router refused; those sites say so themselves.
pub(crate) fn status_of(kind: ErrorKind) -> StatusCode {
    match kind {
        ErrorKind::Auth => StatusCode::UNAUTHORIZED,
        ErrorKind::AuthScope | ErrorKind::SessionPermission => StatusCode::FORBIDDEN,
        ErrorKind::SessionNotFound | ErrorKind::NotFound | ErrorKind::RequestNotFound => {
            StatusCode::NOT_FOUND
        }
        ErrorKind::SessionLocked
        | ErrorKind::SessionNotLoaded
        | ErrorKind::SubagentNotRunning
        | ErrorKind::TurnInFlight
        | ErrorKind::TurnCanceled
        | ErrorKind::Idempotency
        | ErrorKind::InboxAppended
        | ErrorKind::TurnMismatch
        | ErrorKind::StoreReadOnly => StatusCode::CONFLICT,
        ErrorKind::PreconditionFailed => StatusCode::PRECONDITION_FAILED,
        ErrorKind::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
        ErrorKind::InvalidBody | ErrorKind::SessionNotDrivable | ErrorKind::RequestTooLarge => {
            StatusCode::UNPROCESSABLE_ENTITY
        }
        ErrorKind::ConcurrencyLimit => StatusCode::TOO_MANY_REQUESTS,
        ErrorKind::Provider | ErrorKind::ProviderUnavailable | ErrorKind::ContextOverflow => {
            StatusCode::BAD_GATEWAY
        }
        ErrorKind::McpUnavailable => StatusCode::SERVICE_UNAVAILABLE,
        ErrorKind::StreamDetached | ErrorKind::SseLag | ErrorKind::Internal => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

impl IntoResponse for ProblemDetail {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let retry_after = self.retry_after_seconds;
        let mut response = (status, Json(&self)).into_response();
        // RFC 9457 mandates `application/problem+json` instead of axum's default
        // `application/json`.
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/problem+json"),
        );
        if let Some(seconds) = retry_after
            && let Ok(value) = header::HeaderValue::from_str(&seconds.to_string())
        {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        // RFC 9110 §15.5.2: 401 responses MUST include WWW-Authenticate.
        if status == StatusCode::UNAUTHORIZED {
            response.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                header::HeaderValue::from_static(r#"Bearer realm="meka""#),
            );
        }
        response
    }
}

impl ProblemDetail {
    /// Best-effort mapping from internal `MekaError` to a Problem Detail. Used by handlers that
    /// propagate agent-layer errors back to the client. Variants without a dedicated HTTP shape
    /// land on `internal` (500). Refine on demand as new error paths surface.
    ///
    /// `relay_provider_errors` is `[serve] relay_provider_errors`, and it decides whether the four
    /// upstream arms attach the provider's own response as a `provider_response` extension member.
    /// It does not touch `detail`, which is meka's own sentence and identical in both states. A
    /// parameter rather than a `From` impl precisely so it cannot be forgotten: a `.into()` that
    /// silently picked one policy is how a deployment configured to withhold would have gone on
    /// publishing from whichever call site missed the memo.
    ///
    /// Deliberately not extended to [`MekaError::McpTurnGated`], whose reasons are meka's own
    /// subprocess text rather than a provider's response; see the key's own documentation.
    pub(crate) fn for_error(error: &MekaError, relay_provider_errors: bool) -> Self {
        let kind = error.kind();
        // Logged here, once, whatever the surface shows: the upstream's text goes to the log
        // unconditionally, and a fault that reaches a caller as a 500 is the operator's to read
        // about there.
        match error {
            MekaError::Provider(message)
            | MekaError::InvalidRequest(message)
            | MekaError::StreamError(message)
            | MekaError::RetryableProvider { message, .. } => {
                tracing::warn!("provider error: {message}");
            }
            MekaError::ContextOverflow(message) => tracing::warn!("context overflow: {message}"),
            MekaError::McpTurnGated { .. } => tracing::warn!("mcp gate declined a turn: {error}"),
            MekaError::Installation(message) => tracing::error!("installation error: {message}"),
            other if kind == ErrorKind::Internal => {
                tracing::error!("unhandled agent error mapped to 500: {other}");
            }
            _ => {}
        }
        let mut problem = Self::new(kind, status_of(kind), error.sentence());
        // An extension member rather than a replacement for `detail`, because the two carry
        // different things and neither substitutes for the other. `detail` is meka's own sentence
        // and for a context overflow it is the entire remedy ("compact it before retrying"), which
        // relaying by overwrite would have deleted in exchange for a JSON blob. A client wanting to
        // branch on the upstream's error type also wants one well-known field, not prose to parse.
        if relay_provider_errors && let Some(body) = error.upstream_body() {
            problem = problem.with(
                "provider_response",
                Value::from(bounded_upstream_body(body)),
            );
        }
        match error {
            MekaError::RetryableProvider {
                retry_after: Some(delay),
                ..
            // Worth relaying because it is fresh rather than spent: read from the headers of the
            // final attempt, after which the agent loop gives up rather than sleeping again, so
            // nothing has elapsed against it. Clamped, because `parse_retry_after` relays
            // whatever the header said and a broken or hostile upstream can say a year; `u32`
            // seconds is what `ProblemDetail` carries, so an unclamped `u64` would wrap rather
            // than saturate.
            } => problem.with_retry_after(
                u32::try_from(delay.min(&RELAYED_RETRY_AFTER_CAP).as_secs()).unwrap_or(u32::MAX),
            ),
            MekaError::McpTurnGated { servers } => {
                let names: Vec<&str> = servers.iter().map(|(name, _)| name.as_str()).collect();
                problem.with("servers", Value::from(names))
            }
            MekaError::SessionLocked(id) | MekaError::SessionNotFound(id) => {
                problem.with("session_id", id.to_string())
            }
            _ => problem,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn problem_detail_serializes_extensions_at_top_level() {
        let problem = ProblemDetail::new(
            ErrorKind::SessionNotFound,
            StatusCode::NOT_FOUND,
            "session 's_abc' does not exist",
        )
        .instance("/v1/sessions/s_abc/turn")
        .with("session_id", "s_abc");
        let body = serde_json::to_value(&problem).expect("serializable");
        assert_eq!(
            body["type"], "https://meka.run/errors/session-not-found",
            "type URI must match the catalog entry exactly",
        );
        assert_eq!(body["status"], 404);
        assert_eq!(body["instance"], "/v1/sessions/s_abc/turn");
        assert_eq!(
            body["session_id"], "s_abc",
            "extension members flatten to the top level"
        );
    }

    #[test]
    fn meka_error_provider_maps_to_502() {
        let error = MekaError::Provider("upstream 529".into());
        let problem = ProblemDetail::for_error(&error, false);
        assert_eq!(problem.status, 502);
        assert_eq!(problem.type_uri, "https://meka.run/errors/provider");
    }

    /// The operator's switch decides whether a 502 carries the provider's own response text.
    ///
    /// That body can hold an account identifier, a rate-limit posture and a fragment of the
    /// request; whoever holds a `sessions:w` token is not necessarily whoever holds the provider
    /// account.
    #[test]
    fn the_upstream_body_travels_only_when_the_operator_asked_for_it() {
        let leaky = "{\"error\":{\"account_uuid\":\"acct-0f3c\",\"type\":\"rate_limit_error\",\
                     \"message\":\"organization has exceeded its quota\"}}";
        for error in [
            MekaError::Provider(leaky.to_string()),
            MekaError::InvalidRequest(leaky.to_string()),
            MekaError::RetryableProvider {
                message: leaky.to_string(),
                retry_after: None,
                server_error_on_completion: false,
            },
            MekaError::StreamError(leaky.to_string()),
            // In the loop, not merely documented. `http-api.md` promises this arm relays like the
            // others, and nothing else in the suite calls it with relaying on: deleting its
            // `attach` left a documented member silently absent with everything green.
            MekaError::ContextOverflow(leaky.to_string()),
        ] {
            let withheld = ProblemDetail::for_error(&error, false);
            // Kept because a deleted arm falls to the catch-all, whose detail is "internal server
            // error; consult server logs" -- it contains neither secret and does contain "server
            // log" as a substring of "server logs", so the redaction checks alone would pass it.
            assert_eq!(withheld.status, 502, "{error}");
            let body = serde_json::to_string(&withheld).expect("serialize");
            assert!(
                !body.contains("acct-0f3c") && !body.contains("exceeded its quota"),
                "the upstream body reached the caller: {body}",
            );
            // The three provider arms point at the log; the overflow arm spends its `detail` on the
            // remedy instead ("compact it before retrying") and names no log, which is right and is
            // why this is not asserted across the loop.
            let detail = withheld.detail.as_deref().unwrap_or_default().to_string();
            if !matches!(error, MekaError::ContextOverflow(_)) {
                assert!(
                    detail.contains("server log"),
                    "and the caller must be told where the detail went: {error}",
                );
            }
            assert!(
                !detail.is_empty(),
                "every arm owes the caller a sentence of meka's own: {error}",
            );

            // The whole body, not just `detail`: an arm that moved the text into an extension
            // would satisfy a `detail`-only assertion while publishing exactly what the operator
            // turned off. `provider_response` is flattened to the top level like every other
            // extension, so one serialization covers both places it could hide.
            let relayed = ProblemDetail::for_error(&error, true);
            assert_eq!(
                relayed.extensions.get("provider_response"),
                Some(&Value::from(leaky)),
                "with relaying on, the upstream's own words must reach the caller: {error}",
            );
            assert_eq!(
                relayed.detail, withheld.detail,
                "relaying adds a member; it must not rewrite meka's own sentence, which for a \
                 context overflow is the entire remedy: {error}",
            );
        }
    }

    /// A rejection the *upstream* issued is an upstream failure (502), not a complaint about the
    /// caller's own body (4xx). The name says "invalid request" and the status deliberately does
    /// not agree with it: what was invalid is the conversation meka assembled, which the caller
    /// neither sent nor can fix by correcting its own payload. Whether the agent loop tried to
    /// repair it first is not part of the mapping: `/compact`'s summarizer has no repair path at
    /// all, so an `InvalidRequest` from there arrives having been tried exactly once.
    #[test]
    fn meka_error_invalid_request_maps_to_502() {
        let error = MekaError::InvalidRequest("400 invalid_request_error".into());
        let problem = ProblemDetail::for_error(&error, false);
        assert_eq!(problem.status, 502);
        assert_eq!(problem.type_uri, "https://meka.run/errors/provider");
    }

    /// An upstream failure reaches the caller as one rather than as an internal fault, and says it
    /// was the transient kind.
    ///
    /// Falling through to the catch-all makes an exhausted 429 answer 500 and log itself as an
    /// unhandled internal fault, sending an operator to look in the wrong process for a failure
    /// meka classified correctly.
    ///
    /// The type URI is asserted alongside the status because the two travel together: one error
    /// type answering with two different statuses is what a client keying on the type cannot
    /// handle.
    ///
    /// `StreamError` is in here rather than with `Provider` because every producer is
    /// transport-shaped -- an idle timeout, an `Err` from the SSE stream, a stream that ended
    /// before its terminal event -- and a malformed payload is skipped rather than raised, so
    /// nothing here invites a resend into a body the provider rejects identically every time.
    #[test]
    fn an_exhausted_upstream_failure_maps_to_502_as_unavailable() {
        for error in [
            MekaError::RetryableProvider {
                message: "529 overloaded, four attempts".into(),
                retry_after: None,
                server_error_on_completion: false,
            },
            MekaError::StreamError("connection closed mid-stream".into()),
        ] {
            let problem = ProblemDetail::for_error(&error, false);
            assert_eq!(problem.status, 502, "{error}");
            assert_eq!(
                problem.type_uri, "https://meka.run/errors/provider-unavailable",
                "{error}"
            );
        }
    }

    /// A transient failure carrying no `Retry-After` is still distinguishable from a permanent one.
    ///
    /// A relayed `Retry-After` is absent from most transient failures: a transport error has no
    /// response to read a header from, a mid-stream `overloaded` event has no headers at all, and
    /// `parse_retry_after` understands only delta-seconds. Sharing one `type` would leave a bridge
    /// choosing between retrying a dead token forever and discarding turns a second attempt would
    /// have completed.
    ///
    /// `retry_after: None` is the load-bearing part of the setup: with a header present the two
    /// are distinguishable anyway, so a version of this test that supplied one would pass against
    /// the shape it exists to reject.
    #[test]
    fn a_transient_failure_without_a_retry_after_is_still_distinguishable() {
        let transient = ProblemDetail::for_error(
            &MekaError::RetryableProvider {
                message: "529 overloaded".into(),
                retry_after: None,
                server_error_on_completion: false,
            },
            false,
        );
        let permanent =
            ProblemDetail::for_error(&MekaError::Provider("401 invalid x-api-key".into()), false);

        assert_eq!(
            transient.retry_after_seconds, None,
            "the setup must be the case with no header, or this proves nothing"
        );
        assert_eq!(transient.status, permanent.status, "both are 502");
        assert_ne!(
            transient.type_uri, permanent.type_uri,
            "with no `Retry-After` to tell them apart, `type` is all a client has left"
        );
        assert_eq!(
            transient.type_uri,
            "https://meka.run/errors/provider-unavailable"
        );
        assert_eq!(permanent.type_uri, "https://meka.run/errors/provider");

        // The titles too, for the reason `a_context_overflow_is_502_under_its_own_type` asserts
        // its own: `type` is what a client switches on, but `title` is what a UI renders, and
        // nothing else in the suite would notice the two variants being given one string.
        assert_eq!(transient.title, "Provider temporarily unavailable");
        assert_eq!(permanent.title, "Provider call failed");
        assert_ne!(transient.title, permanent.title);
    }

    /// A required MCP server that never came up answers 503 under its own type, and its reasons
    /// stay in the log.
    ///
    /// Both halves matter. Falling through to the `other` arm reports a subprocess that failed to
    /// start as `/errors/internal` 500, logged as an unhandled internal fault, which is the one
    /// classification that sends an operator looking in the wrong process. And a reason string is
    /// the connector's own text: it has carried a spawn failure complete with the command line and
    /// its path, which is the same argument that keeps a provider's response body out of the arm
    /// above.
    #[test]
    fn a_required_mcp_server_that_is_down_is_503_under_its_own_type() {
        let error = MekaError::McpTurnGated {
            servers: vec![
                (
                    "ida".to_string(),
                    "failed to spawn process: /opt/private/ida-mcp: No such file".to_string(),
                ),
                ("exa".to_string(), "handshake timed out".to_string()),
            ],
        };
        // `true`, which is the case that can actually fail. `[serve] relay_provider_errors` governs
        // the *provider's* response and is documented in three places as not reaching this arm, but
        // asserting it with relaying off proved only that the off switch works. Adding `attach()`
        // here would have published a spawn failure's command line and path to every caller with
        // nothing going red.
        let problem = ProblemDetail::for_error(&error, true);

        assert_eq!(problem.status, 503);
        assert_eq!(problem.type_uri, "https://meka.run/errors/mcp-unavailable");
        let detail = problem.detail.clone().expect("a detail naming the servers");
        assert!(detail.contains("ida") && detail.contains("exa"), "{detail}");
        assert_eq!(
            problem.extensions.get("servers"),
            Some(&serde_json::json!(["ida", "exa"])),
            "a client should branch on the names without parsing the sentence"
        );

        let body = serde_json::to_string(&problem).expect("serialize the problem");
        assert!(!body.contains("/opt/private"), "{body}");
        assert!(!body.contains("handshake timed out"), "{body}");
    }

    /// The upstream's own `Retry-After` reaches the caller, clamped.
    ///
    /// Both halves, because `ProblemDetail` carries the header and the body extension separately
    /// and setting one without the other is a wire-shape bug a client hits silently. The clamp is
    /// not cosmetic: `parse_retry_after` relays whatever the header said, `Duration::as_secs` is a
    /// `u64`, and the field is a `u32`, so an upstream saying a year would wrap to a small number
    /// rather than saturate if the cast were unguarded.
    #[test]
    fn a_retryable_failure_relays_the_upstream_retry_after() {
        let problem = ProblemDetail::for_error(
            &MekaError::RetryableProvider {
                message: "429 rate limited".into(),
                retry_after: Some(std::time::Duration::from_secs(30)),
                server_error_on_completion: false,
            },
            false,
        );
        assert_eq!(problem.retry_after_seconds, Some(30));
        assert_eq!(
            problem.extensions.get("retry_after"),
            Some(&Value::from(30))
        );

        let absurd = ProblemDetail::for_error(
            &MekaError::RetryableProvider {
                message: "529 overloaded".into(),
                retry_after: Some(std::time::Duration::from_secs(31_536_000)),
                server_error_on_completion: false,
            },
            false,
        );
        assert_eq!(
            absurd.retry_after_seconds,
            u32::try_from(RELAYED_RETRY_AFTER_CAP.as_secs()).ok(),
            "a header saying a year must be clamped, not relayed and not wrapped"
        );

        let silent = ProblemDetail::for_error(
            &MekaError::RetryableProvider {
                message: "connection reset".into(),
                retry_after: None,
                server_error_on_completion: false,
            },
            false,
        );
        assert_eq!(
            silent.retry_after_seconds, None,
            "a transport failure has no header to relay, so meka must not invent one"
        );
    }

    /// A conversation that will not fit is 502 like its neighbors but says so under its own type.
    ///
    /// The status is shared because the upstream is what refused the turn. The type is not, because
    /// the remedies are opposites: `/errors/provider` means "the upstream is unwell, send it
    /// again", and a client reading this as that retries an oversized conversation until it gives
    /// up on wall-clock. Nothing else on the wire distinguishes them, so if this assertion is ever
    /// relaxed to make a match arm simpler, that loop comes back.
    #[test]
    fn a_context_overflow_is_502_under_its_own_type() {
        let problem = ProblemDetail::for_error(
            &MekaError::ContextOverflow(
                "API returned status 400: {\"error\":{\"account_uuid\":\"acct-0f3c\",\"message\":\
             \"prompt is too long: 250000 tokens > 200000 maximum\"}}"
                    .into(),
            ),
            false,
        );
        assert_eq!(problem.status, 502);
        assert_eq!(
            problem.type_uri, "https://meka.run/errors/context-overflow",
            "sharing `/errors/provider` sends a correct client into a retry loop"
        );
        // The title is a wire field too, and RFC 9457 asks it to be stable for a given type, so a
        // client may render it verbatim. Asserting one is also what stops `ErrorKind::title` being
        // gutted wholesale without a test noticing.
        assert_eq!(
            problem.title,
            "Conversation exceeds the model's context window"
        );
        let detail = problem.detail.unwrap_or_default();
        assert!(
            detail.contains("compact it before retrying"),
            "the detail has to name the remedy, since the type alone is opaque: {detail}"
        );
        // Built from a provider response like every other arm, so it carries that body and must not
        // relay it. Unlike the 502 arm it does not point at the log, because what the caller needs
        // to know is stated outright rather than withheld -- which is why this lives here rather
        // than in `a_provider_failure_does_not_relay_the_upstream_body`, whose loop asserts that
        // pointer for every entry.
        assert!(
            !detail.contains("250000"),
            "the upstream body reached the caller: {detail}"
        );
    }

    #[test]
    fn meka_error_session_locked_carries_session_id() {
        let id = uuid::Uuid::nil();
        let problem = ProblemDetail::for_error(&MekaError::SessionLocked(id), false);
        assert_eq!(problem.status, 409);
        assert_eq!(problem.type_uri, "https://meka.run/errors/session-locked");
        assert_eq!(
            problem.extensions.get("session_id"),
            Some(&Value::String(id.to_string()))
        );
    }

    /// An installation fault is the operator's, so the caller gets the sanitized 500 and the
    /// sentence goes to the log.
    ///
    /// It is meka's own sentence, which is what made it look like a `Config` refusal and land in
    /// the verbatim 422 arm: a `[web] ca_cert_file` that is not there answered `invalid-body` with
    /// the operator's filesystem path in `detail`, to a caller who cannot fix it and should not
    /// read it. Relaying is asked for here, so the switch cannot be what is hiding it.
    #[test]
    fn an_installation_fault_is_sanitized_rather_than_relayed_as_a_refusal() {
        let problem = ProblemDetail::for_error(
            &MekaError::Installation(
                "[web].ca_cert_file '/etc/meka/private/ca.pem': No such file or directory"
                    .to_string(),
            ),
            true,
        );
        assert_eq!(problem.status, 500);
        assert_eq!(problem.type_uri, ErrorKind::Internal.type_uri());
        let body = serde_json::to_string(&problem).expect("serialize");
        assert!(
            !body.contains("/etc/meka/private"),
            "the operator's path reached the caller: {body}"
        );
    }

    /// meka's own request ceiling is the caller's to act on and has no upstream behind it, so it
    /// must not be published as a 502 `provider` pointing at a provider response that never
    /// existed. The remedy is in `detail`, so `detail` is asserted rather than merely present.
    #[test]
    fn the_request_ceiling_is_mekas_own_refusal_rather_than_a_provider_failure() {
        let error = MekaError::RequestTooLarge(
            "request body is 31.4 MiB after redacting old tool-result images; this profile's              ceiling is 30.0 MiB (`max_request_bytes`). Run /compact, remove large attachments              from the most recent turn, or split the work across smaller turns."
                .to_string(),
        );
        let problem = ProblemDetail::for_error(&error, true);
        assert_eq!(problem.status, 422);
        assert_eq!(
            problem.type_uri,
            "https://meka.run/errors/request-too-large"
        );
        assert_eq!(problem.detail.as_deref(), Some(error.to_string().as_str()));
        assert!(
            !problem.extensions.contains_key("provider_response"),
            "nothing upstream judged this request, so there is no response to relay"
        );
    }

    /// The refusals every door raises by variant land on the status and `type` the API documents,
    /// rather than on the 500 the fallthrough arm gives an error nobody classified.
    #[test]
    fn the_refusal_variants_map_to_their_documented_types() {
        let id = uuid::Uuid::nil();
        let missing = ProblemDetail::for_error(&MekaError::SessionNotFound(id), false);
        assert_eq!(missing.status, 404);
        assert_eq!(
            missing.type_uri,
            "https://meka.run/errors/session-not-found"
        );
        assert_eq!(
            missing.extensions.get("session_id"),
            Some(&Value::String(id.to_string()))
        );
        assert_eq!(
            missing.detail.as_deref(),
            Some(MekaError::SessionNotFound(id).to_string().as_str())
        );

        let busy = ProblemDetail::for_error(&MekaError::TurnInFlight { doing: "patch" }, false);
        assert_eq!(busy.status, 409);
        assert_eq!(busy.type_uri, "https://meka.run/errors/turn-in-flight");
        let detail = busy.detail.clone().unwrap_or_default();
        assert!(detail.contains("cannot patch"), "{detail}");

        for error in [
            MekaError::EmptyPrompt,
            MekaError::DisabledLevel {
                level: "unrestricted".to_string(),
                enabled: vec!["read".to_string()],
            },
            MekaError::ProfileNotConfigured {
                name: "ghost".to_string(),
                known: vec!["work".to_string()],
            },
        ] {
            let problem = ProblemDetail::for_error(&error, false);
            assert_eq!(problem.status, 422, "{error}");
            assert_eq!(
                problem.type_uri, "https://meka.run/errors/invalid-body",
                "{error}"
            );
            assert_eq!(problem.detail.as_deref(), Some(error.to_string().as_str()));
        }
    }
}
