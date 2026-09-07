//! Crate-wide [`MekaError`] enum and [`Result`] alias. All non-binary code paths return `Result<T,
//! MekaError>`; the `main` binary wraps these in `anyhow::Result` for top-level reporting.

use std::time::Duration;

use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum MekaError {
    #[error("configuration error: {0}")]
    Config(String),

    /// The caller asked for something that cannot be done as asked, in words meant for them: a
    /// malformed archive, an id that names nothing, an option that contradicts another.
    #[error("{0}")]
    Usage(String),

    #[error("database error: {0}")]
    Database(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("provider error: {0}")]
    Provider(String),

    /// The provider rejected the request because the prompt exceeded the model's context window
    /// (e.g. HTTP 400 "prompt is too long" / 413 / `context_length_exceeded`). Distinct from
    /// [`Self::Provider`] so the agent loop can catch it by type and compact-and-retry once instead
    /// of matching error strings at the call site.
    #[error("context window exceeded: {0}")]
    ContextOverflow(String),

    /// A session exists, but this door may not drive it. Carries the whole refusal, which names
    /// both why and the door that can, so every surface can relay it verbatim.
    ///
    /// Its own variant rather than [`Self::Config`] because nothing about the installation is
    /// wrong: `Config`'s `Display` opens with "configuration error", which is what a CLI user would
    /// read, and it is the wrong diagnosis. Nor is it [`Self::InvalidRequest`], which means the
    /// *provider* refused a body and arms the turn's degrade path.
    ///
    /// Raised by the two agent builders as a backstop, and by the doors that have to refuse before
    /// they write anything. Mapped to 422 by [`crate::host::http::reattach::agent_build_problem`].
    #[error("{0}")]
    SessionNotDrivable(String),

    /// The operator's installation is wrong in a way no caller can remedy: a `[web]` client that
    /// cannot be built from its proxy URL or CA file, or a `base_url` whose shape a backend
    /// refuses.
    ///
    /// Its own variant rather than [`Self::Config`] because the two travel differently. `Config`
    /// is a refusal the caller can act on and every host relays it verbatim: a profile with no
    /// stored credential names the login that fixes it. This one names paths and endpoints out of
    /// `config.toml`, which are for the operator's terminal and not for a remote caller, so HTTP
    /// answers a sanitized 500, ACP an `InternalError` without the text, and the REPL and CLI
    /// print it. `crate::host::build_shared_deps` raises the web client's at startup, before any
    /// session exists.
    #[error("{0}")]
    Installation(String),

    /// The provider rejected the request as malformed (HTTP 400 / 422 that isn't an overflow).
    /// Deterministic on the request body, so retrying it unchanged is pointless; distinct from
    /// [`Self::Provider`] so the agent loop can instead degrade the content it most recently
    /// appended and retry once (see `Agent::run_turn`). Without that path a single rejected block
    /// is permanent: it is already committed to the session, so every later request carries it and
    /// fails the same way.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// meka's own refusal to send a body still over the profile's `max_request_bytes` once every
    /// older image has been redacted.
    ///
    /// Not [`Self::InvalidRequest`], which means the *provider* refused a body and is published
    /// with the provider's response beside it: this request never left the process and the
    /// sentence is meka's own, naming the ceiling and the remedy, so every host relays it verbatim
    /// as the caller's to act on. The agent loop answers it as it answers `InvalidRequest`, by
    /// degrading what the turn appended, because once the older images are gone the newest content
    /// is what is left to blame.
    #[error("{0}")]
    RequestTooLarge(String),

    /// A transient provider failure that is safe to retry with backoff. Distinct from
    /// [`Self::Provider`] so the agent loop can retry by type instead of matching error strings.
    /// Carries the provider's `Retry-After` hint when present.
    ///
    /// Two sources, and they are the same fact arriving by different routes. A response that says
    /// so (HTTP 429, any 5xx including Anthropic's 529 "overloaded", or a mid-stream
    /// `overloaded_error`/`rate_limit_error`/`api_error` SSE event), classified by
    /// [`provider_http_error`]. Or no response at all, classified by [`provider_transport_error`]:
    /// the connection failed, the request could not be delivered, or the body could not be read
    /// back.
    #[error("provider temporarily unavailable: {message}")]
    RetryableProvider {
        message: String,
        retry_after: Option<Duration>,
        /// Whether a *completion* reached the provider and the provider then failed handling it,
        /// which is exactly a 5xx from [`ProviderRequest::Completion`] and nothing else.
        ///
        /// Recorded because it is the only shape of this variant that the turn's own content could
        /// explain, and the agent loop's degrade-and-retry keys on it. A transport failure never
        /// delivered the body, so nothing judged it. A 429 is the provider declining to process
        /// the request at all, which is a statement about rate and not about what was
        /// sent. A token endpoint's 5xx concerns a request that carried no conversation.
        /// Degrading on any of those would answer an outage by deleting the user's
        /// content: the degraded retry can succeed simply because the network came back,
        /// and `TurnRecovery::persist_vindicated_repair` would then write that loss to the
        /// store as proven-good.
        server_error_on_completion: bool,
    },

    #[error("tool execution error: {tool_name}: {message}")]
    ToolExecution { tool_name: String, message: String },

    #[error("tool registration error: {message}")]
    ToolRegistration { message: String },

    #[error("session already attached by another process: {0}")]
    SessionLocked(uuid::Uuid),

    /// The id names no session in the store. Raised by every door that looks a session up on a
    /// caller's behalf, so each host maps it once: 404 on HTTP, `InvalidParams` on ACP, a non-zero
    /// exit on the CLI.
    #[error("session '{0}' not found")]
    SessionNotFound(uuid::Uuid),

    /// A turn holds the session's conversation and what the caller asked for cannot share it.
    /// `doing` is the caller's verb phrase, so the sentence names the thing refused rather than
    /// the thing running.
    #[error("cannot {doing} while a turn is in flight; cancel it first")]
    TurnInFlight { doing: &'static str },

    /// A prompt with nothing in it: no text and no image. Refused before admission, which is why
    /// it is not [`Self::InvalidRequest`]; that one means the provider rejected a body.
    #[error("the prompt must contain non-empty text, or at least one image")]
    EmptyPrompt,

    /// A level `[permissions].enabled` does not admit, named beside the ones it does.
    ///
    /// Strings rather than `Permission` and `EnabledPermissions`: this module is a leaf and may
    /// not name `crate::permission`, so the doors render both before raising it.
    #[error("permission level '{level}' is not enabled (enabled: {})", enabled.join(", "))]
    DisabledLevel { level: String, enabled: Vec<String> },

    /// A profile name `config.toml` does not have, with the names it does.
    #[error("{}", crate::text::unknown_name("profile", name, known))]
    ProfileNotConfigured { name: String, known: Vec<String> },

    #[error("agent interrupted by user")]
    Interrupted,

    /// A logic invariant in meka itself was violated. Used in place of `.expect()` for cases where
    /// a bug in our own code (not user input or I/O) is the only path to the error.
    #[error("internal error: {0}")]
    Internal(String),

    #[error("SSE stream error: {0}")]
    StreamError(String),

    #[error("MCP connection error: {server_name}: {message}")]
    McpConnection {
        server_name: String,
        message: String,
    },

    #[error("MCP tool error: {server_name}: {tool_name}: {message}")]
    McpToolExecution {
        server_name: String,
        tool_name: String,
        message: String,
    },

    #[error("MCP authentication error: {server_name}: {message}")]
    McpAuth {
        server_name: String,
        message: String,
    },

    /// MCP readiness gate rejected the turn: at least one server marked
    /// [`crate::config::McpServerConfig::required`] wasn't `Connected` within the configured
    /// grace period. Servers that aren't required never appear here. Turn contents haven't been
    /// sent to the provider. The REPL catches this and loops back to the prompt; one-shot mode
    /// propagates to a non-zero process exit.
    #[error("mcp: {} server(s) not ready: {}", .servers.len(), .servers.iter().map(|(n, s)| format!("{n} ({s})")).collect::<Vec<_>>().join(", "))]
    McpTurnGated { servers: Vec<(String, String)> },
}

pub(crate) type Result<T> = std::result::Result<T, MekaError>;

/// The most of an upstream's response a host will repeat to a caller, in bytes.
///
/// Not a redaction measure: a length bound keeps the *start*, which is where an identifier sits in
/// a JSON error object, so whether the text travels at all is `relay_provider_errors`'s decision
/// and this only bounds what does. Bounded at all because the text is attacker-influenced and
/// repeated to more than one reader: it arrives from `response.text()` with no cap of its own, it
/// is copied into every failure a host reports, and the HTTP host retains the terminal event
/// carrying it for reconnects. A `base_url` is user-supplied, so a misconfigured endpoint answering
/// a multi-megabyte error page should cost a truncated string rather than a copy per reader.
///
/// Far above any real provider error, which run to a few hundred bytes of JSON.
pub(crate) const RELAYED_BODY_CAP: usize = 4 * crate::text::KIB;

/// Truncation marker, charged against [`RELAYED_BODY_CAP`] rather than added on top of it.
pub(crate) const TRUNCATION_MARKER: &str = "… (truncated; the full text is in the log)";

/// The two constants above are subtracted from each other, so their relationship is load-bearing
/// rather than incidental. Asserted at compile time because the failure is otherwise a runtime
/// panic inside a handler, on attacker-influenced input: the subtraction underflows, and a wrapped
/// `usize` then indexes far past the end of the string.
const _: () = assert!(
    RELAYED_BODY_CAP > TRUNCATION_MARKER.len() + 4,
    "RELAYED_BODY_CAP must exceed the marker by at least one UTF-8 character, or \
     `bounded_upstream_body` underflows its budget"
);

/// Repeat at most [`RELAYED_BODY_CAP`] bytes of `message`, marking the cut when one happens.
///
/// Shared by the HTTP host's `provider_response` member and the ACP host's `error.data`, so the
/// two cannot disagree about how much of an upstream a caller sees. The result never exceeds the
/// cap and is never longer than the input, because the marker comes out of the budget instead of
/// being appended to it. Added on top, a body a few bytes over the cap came back larger than it
/// went in, while telling the reader it had been shortened.
///
/// Cuts on a character boundary. Multi-byte text is ordinary here (a provider error in Japanese,
/// an emoji in a proxy's HTML page) and slicing mid-codepoint panics.
pub(crate) fn bounded_upstream_body(message: &str) -> String {
    if message.len() <= RELAYED_BODY_CAP {
        return message.to_string();
    }
    let mut end = RELAYED_BODY_CAP - TRUNCATION_MARKER.len();
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{}", &message[..end], TRUNCATION_MARKER)
}

/// What a response is answering, which decides whether a 400 or 422 is worth
/// [`MekaError::InvalidRequest`].
///
/// A parameter rather than two functions, so the choice cannot be made by omission. Every caller of
/// [`provider_http_error`] has to say which it is, and a probe added later cannot inherit the
/// completion semantics just because that was the obvious function to reach for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderRequest {
    /// A completion: the turn itself, streaming or not.
    ///
    /// Only this one lets [`provider_http_error`] answer [`MekaError::InvalidRequest`], because
    /// that variant does not mean "the request was malformed". It means "degrade the content this
    /// turn appended and try again", which is a coherent instruction only when the request carries
    /// turn content. (`crate::store::memory` constructs the variant too, for a different reason;
    /// its errors are absorbed into a tool result and never reach the agent loop's repair arm.)
    Completion,
    /// Anything else meka asks a provider: a usage or identity probe, a history fetch.
    ///
    /// Not an OAuth token exchange, which looks like it belongs here and does not:
    /// [`oauth_refresh_error`] judges those, because a spent grant needs a remedy attached and this
    /// function's overflow sniff is meaningless coming from a token endpoint.
    ///
    /// A 400 here is permanent and there is nothing to degrade, so it stays
    /// [`MekaError::Provider`]. Classifying it as `InvalidRequest` would arm the agent loop's
    /// image-degrading repair with a failure that has nothing to do with images, and the visible
    /// result would be a turn quietly stripping the user's attachments because a *usage* endpoint
    /// returned 400.
    Auxiliary,
}

/// Classify a provider HTTP failure response: map context-window overflows to
/// [`MekaError::ContextOverflow`] (so the agent loop can compact-and-retry once), transient
/// failures (429, any 5xx including Anthropic's 529 "overloaded") to
/// [`MekaError::RetryableProvider`] (so the agent loop can retry with backoff), malformed
/// completions (400 / 422) to [`MekaError::InvalidRequest`] (so the agent loop can
/// degrade-and-retry once), and everything else to [`MekaError::Provider`]. Anthropic returns HTTP
/// 400 `invalid_request_error` with "prompt is too long"; OpenAI returns 400
/// `context_length_exceeded` (or 413). The overflow check is matched on the body, because a bare
/// 400 is shared with many unrelated errors, but only on a status that could *be* an overflow:
/// never a 5xx, never a 429, and unconditionally on a 413. See the comment on `overflow` below for
/// why the status is the half worth trusting when the two disagree.
///
/// The 400 / 422 bucket deliberately makes no attempt to tell a content problem from a parameter
/// problem: a `max_tokens` above the model's ceiling, an unknown beta header and a mislabeled
/// image all arrive in the same shape. The agent loop restores what it degraded when the retry also
/// fails, so classifying too broadly here costs one round trip and destroys nothing.
///
/// **That tolerance is why `request` exists.** It holds only where there is turn content to degrade
/// and a loop that will put it back. A usage probe answering 400 has neither, so `Auxiliary` keeps
/// it out of that bucket; see [`ProviderRequest`].
pub(crate) fn provider_http_error(
    status: reqwest::StatusCode,
    body: &str,
    retry_after: Option<Duration>,
    request: ProviderRequest,
) -> MekaError {
    let lower = body.to_ascii_lowercase();
    // The body is consulted only on a 4xx that is not a 429, because a provider may echo the
    // request back in an error. Matched against any status, a turn whose own text discusses
    // `context_length_exceeded` reclassifies a blip or a rate limit as an overflow, skipping the
    // retry and running an emergency compaction. `PAYLOAD_TOO_LARGE` is a status rather than a
    // guess, so it stays unconditional.
    //
    // The cost, since the two failures are not symmetric: a gateway that re-wraps an upstream 400
    // as a 5xx turns a real overflow into a retry and then a degrade, losing the turn's attachments
    // where it should have compacted. That ends in a legible error the user can answer with
    // `/compact`, where misreading a blip destroys context silently on the common path.
    let overflow = status == reqwest::StatusCode::PAYLOAD_TOO_LARGE
        || (status.is_client_error()
            && status != reqwest::StatusCode::TOO_MANY_REQUESTS
            && (lower.contains("prompt is too long")
                || lower.contains("context_length_exceeded")
                || lower.contains("context length exceeded")
                || lower.contains("maximum context length")
                || lower.contains("exceeds the maximum context")));
    let message = format!("API returned status {status}: {}", render_error_body(body));
    if overflow {
        MekaError::ContextOverflow(message)
    } else if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        MekaError::RetryableProvider {
            message,
            retry_after,
            server_error_on_completion: request == ProviderRequest::Completion
                && status.is_server_error(),
        }
    } else if request == ProviderRequest::Completion
        && (status == reqwest::StatusCode::BAD_REQUEST
            || status == reqwest::StatusCode::UNPROCESSABLE_ENTITY)
    {
        MekaError::InvalidRequest(message)
    } else {
        MekaError::Provider(message)
    }
}

/// Classify a provider call that produced no *usable* response: the request could not be delivered,
/// the connection failed, or the response body could not be read back.
///
/// The companion to [`provider_http_error`]: a call either yields a response to judge by status,
/// which that function does, or leaves the caller nothing to act on, which this one judges. The
/// default is [`MekaError::RetryableProvider`], and an unrecognized reqwest error kind inherits it
/// rather than becoming terminal.
///
/// Retrying is not free at every site. A body read that fails on a completion was already generated
/// and charged, so the retry pays for it again. That is accepted over losing the turn, and
/// `should_retry_provider_error` bounds it with [`crate::provider::retry::RETRY_BUDGET`], which is
/// the layer that knows how long the sequence has been running. Classification cannot.
///
/// Two exceptions, both properties of the request and its destination rather than of the network,
/// so both fail identically however many times they are tried. `is_builder` means the request was
/// never constructed and nothing was sent. `is_redirect` means reqwest followed real 3xx answers
/// until its policy ran out, so the endpoint replied every time.
///
/// A timeout is not a third exception, tempting as it is: reqwest cannot tell a request that may
/// still be generating from one that delivered nothing. `is_timeout` scans the source chain for any
/// `io::ErrorKind::TimedOut` and `is_connect` matches only a failure inside the connector, so a
/// write that times out on an already-pooled connection reads as `is_timeout && !is_connect` while
/// having sent nothing at all. The same timeout mid-SSE is retried as [`MekaError::StreamError`].
///
/// `retry_after` is what the response said, when one arrived. A send failure has no headers and
/// passes `None`; a body-read failure parsed the hint before reading and passes it on.
pub(crate) fn provider_transport_error(
    context: &str,
    error: &reqwest::Error,
    retry_after: Option<Duration>,
) -> MekaError {
    let message = format!("{}: {}", context, format_reqwest_error(error));
    if error.is_builder() || error.is_redirect() {
        MekaError::Provider(message)
    } else {
        MekaError::RetryableProvider {
            message,
            retry_after,
            // No response arrived, so whatever was sent was never judged.
            server_error_on_completion: false,
        }
    }
}

/// A mid-stream error event, classified by the code the backend put on it. The codes each backend
/// documents as transient are retryable; anything else, including a code this build does not know,
/// is permanent, so a real problem surfaces at once instead of burning the retry budget first.
///
/// Never `server_error_on_completion`, even though a completion was in flight: the event arrives
/// on the stream, where an overload cannot be told from a failure to handle the body, so it does
/// not license deleting content.
pub(crate) fn provider_stream_error(code: &str, message: String) -> MekaError {
    match code {
        // Anthropic `error.type`.
        "overloaded_error" | "rate_limit_error" | "api_error"
        // OpenAI Responses `error.code`.
        | "server_error" | "rate_limit_exceeded" | "overloaded" => MekaError::RetryableProvider {
            message,
            retry_after: None,
            server_error_on_completion: false,
        },
        _ => MekaError::Provider(message),
    }
}

/// [`provider_stream_error`] for an OpenAI-shaped error object: `{"code", "type", "message"}`.
///
/// `code` is read as a string when it is one, and as an HTTP status when it is a number, which is
/// how OpenRouter and other gateways relay the upstream status inside a 200 stream; a numeric 429
/// or 5xx is then retried as the same status arriving as a response would be. Read as a string
/// only, every numeric code was "unknown" and a mid-stream rate limit ended the turn where the
/// same limit on the headers would have backed off. `type` is the fallback when there is no string
/// code, and `fallback_message` stands in for a missing `message`.
pub(crate) fn provider_stream_error_object(
    error: &serde_json::Value,
    fallback_message: &str,
) -> MekaError {
    let message = error
        .get("message")
        .and_then(|value| value.as_str())
        .unwrap_or(fallback_message)
        .to_string();
    if let Some(status) = error.get("code").and_then(|value| value.as_u64()) {
        return if status == 429 || (500..600).contains(&status) {
            MekaError::RetryableProvider {
                message,
                retry_after: None,
                server_error_on_completion: false,
            }
        } else {
            MekaError::Provider(message)
        };
    }
    let code = error
        .get("code")
        .and_then(|value| value.as_str())
        .or_else(|| error.get("type").and_then(|value| value.as_str()))
        .unwrap_or("unknown");
    provider_stream_error(code, message)
}

/// Classify an OAuth token exchange the authorization server *answered*, whether by rejecting it or
/// by returning a success meka failed to read back.
///
/// A refresh consumes its request: under RFC 9700 rotation the refresh token is single-use, so once
/// the server has read it, it is spent and its replacement is in a response meka may not be
/// holding. Sending it again is a replay, and §4.14.2 has the server revoke the whole token family
/// on one, which costs a browser login rather than a round trip. So the split is by what the answer
/// implies about the token, not by whether one arrived: a 429 was refused before the grant was
/// read and a 5xx usually means the server is unwell rather than that the grant is bad, so both
/// retry in the ordinary way; that accepts the issuer that rotates and *then* fails. Everything
/// else is terminal and names the login that recovers it. A success whose body cannot be decoded
/// is terminal too: the server accepted the token, so it is certainly spent.
///
/// A transport failure stays with [`provider_transport_error`]: the token's fate is unknown, and a
/// token spent without meka seeing the replacement is already dead, so retrying is right when the
/// request never landed and no worse than inaction when it did. Not [`provider_http_error`] with
/// [`ProviderRequest::Auxiliary`] either: that one sniffs the body for context-window phrases,
/// which is meaningless from a token endpoint, and has no remedy to attach.
pub(crate) fn oauth_refresh_error(
    context: &str,
    status: reqwest::StatusCode,
    detail: &str,
    retry_after: Option<Duration>,
    account: &str,
) -> MekaError {
    let message = format!("{} ({}): {}", context, status, render_error_body(detail));
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        MekaError::RetryableProvider {
            message,
            retry_after,
            // A token exchange carries no conversation, so there is nothing to degrade.
            server_error_on_completion: false,
        }
    } else {
        MekaError::Provider(format!(
            "{message}; run `meka account login {account}` to sign in again"
        ))
    }
}

/// Prepare a server's error response body for display at the tail of an error message.
///
/// Bodies arrive with a trailing newline more often than not, and the body is the last thing in
/// every message that carries one, so an untrimmed one ends the console line in whitespace and
/// prints a blank line before the next prompt. A body that is only whitespace, or that failed to
/// read at all, is named as absent rather than left as a dangling colon.
pub(crate) fn render_error_body(body: &str) -> &str {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        "no response body"
    } else {
        trimmed
    }
}

/// Parse the `Retry-After` response header as a whole number of seconds. Only the delta-seconds
/// form is read; the HTTP-date form yields `None`, and so the computed backoff, rather than
/// pulling in a date parser for a form no provider has been seen to send.
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Read a caught panic's payload as text.
///
/// `catch_unwind` hands back a `Box<dyn Any>` whose only useful shapes are the two the standard
/// panic machinery boxes: a `&'static str` for a literal message and a `String` for a formatted
/// one. Everything else is a payload from a hand-rolled `panic_any`, which nothing here does.
///
/// Shared because every supervised loop needs the same three lines, and a loop that catches a panic
/// and then cannot say what it was is barely better than one that dies.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Format a [`reqwest::Error`] together with its full source chain.
///
/// reqwest's outer Display string ("error sending request for url …") hides the cause (TCP reset,
/// HTTP/2 GOAWAY, TLS handshake failure, connect timeout, DNS failure). Walking
/// [`std::error::Error::source`] puts that cause inline, so the user sees what broke instead of
/// reqwest's wrapper.
///
/// A provider call that failed in transit reaches it through [`provider_transport_error`] rather
/// than directly, so that formatting the cause and deciding whether the failure is worth retrying
/// stay one step. The other callers are the web tools, MCP auth, and `exchange_refresh_token`,
/// which formats the cause itself and hands it to [`oauth_refresh_error`] because a token exchange
/// is judged by a different rule. One provider-side site still formats a `reqwest::Error` without
/// this function and loses the chain: `build_http_client`, which fails before any call exists.
pub(crate) fn format_reqwest_error(error: &reqwest::Error) -> String {
    use std::error::Error as _;
    let mut out = error.to_string();
    let mut source: Option<&dyn std::error::Error> = error.source();
    while let Some(cause) = source {
        out.push_str(": ");
        out.push_str(&cause.to_string());
        source = cause.source();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A missing profile is refused in the one "no such name" sentence, with what exists beside it.
    #[test]
    fn a_missing_profile_is_refused_beside_the_configured_ones() {
        assert_eq!(
            MekaError::ProfileNotConfigured {
                name: "ghost".to_string(),
                known: vec!["work".to_string(), "personal".to_string()],
            }
            .to_string(),
            "no profile named 'ghost' (configured: work, personal)"
        );
        assert_eq!(
            MekaError::ProfileNotConfigured {
                name: "ghost".to_string(),
                known: Vec::new(),
            }
            .to_string(),
            "no profile named 'ghost' (none configured)"
        );
    }

    /// The size bound never exceeds the cap, never grows the input, and cuts on a character
    /// boundary.
    ///
    /// The four-byte case carries a one-byte prefix on purpose: `RELAYED_BODY_CAP` is 4096, which
    /// is itself divisible by four, so an unprefixed run of four-byte characters lands the cut on a
    /// boundary already and a naive slice would pass. Only a misaligned input exercises the walk.
    #[test]
    fn the_relayed_body_is_bounded_and_cut_on_a_character_boundary() {
        for (label, body) in [
            ("three-byte", "\u{3042}".repeat(RELAYED_BODY_CAP)),
            (
                "four-byte misaligned",
                format!("a{}", "\u{1F642}".repeat(RELAYED_BODY_CAP)),
            ),
            (
                "mixed",
                format!(
                    "{}{}",
                    "a".repeat(RELAYED_BODY_CAP - 1),
                    "\u{3042}".repeat(16)
                ),
            ),
            // Just over the cap, which is the one size a bound must not grow: adding a marker on
            // top would return more bytes than it was given.
            ("one byte over", "a".repeat(RELAYED_BODY_CAP + 1)),
        ] {
            let out = bounded_upstream_body(&body);
            assert!(
                out.len() <= RELAYED_BODY_CAP,
                "{label}: a bound that can be exceeded is not one; got {} bytes",
                out.len()
            );
            assert!(
                out.len() <= body.len(),
                "{label}: truncation must not grow the input; {} -> {}",
                body.len(),
                out.len()
            );
            assert!(out.contains("truncated"), "{label} must mark the cut");
            assert!(
                body.starts_with(&out[..out.len() - TRUNCATION_MARKER.len()]),
                "{label}: the kept part must be the input's own prefix, which is where a \
                 provider's error type sits"
            );
        }
        let exact = "x".repeat(RELAYED_BODY_CAP);
        assert_eq!(
            bounded_upstream_body(&exact),
            exact,
            "a body exactly at the cap is relayed whole, not marked truncated"
        );
    }

    /// A gateway relaying the upstream status as a numeric `code` inside a 200 stream is read like
    /// the status itself: 429 and 5xx retry, anything else is permanent. Read as a string only,
    /// every number was "unknown" and a mid-stream rate limit ended the turn.
    #[test]
    fn a_numeric_stream_error_code_is_read_as_a_status() {
        for status in [429, 500, 502, 529] {
            let error = provider_stream_error_object(
                &serde_json::json!({"code": status, "message": "later"}),
                "fallback",
            );
            assert!(
                matches!(error, MekaError::RetryableProvider { ref message, .. } if message == "later"),
                "{status}: {error:?}"
            );
        }
        for status in [400, 401, 404] {
            let error = provider_stream_error_object(
                &serde_json::json!({"code": status, "message": "no"}),
                "fallback",
            );
            assert!(
                matches!(error, MekaError::Provider(ref message) if message == "no"),
                "{status}: {error:?}"
            );
        }
        // A string code goes through the documented list; `type` stands in when there is none, and
        // the fallback message when the object names none.
        assert!(matches!(
            provider_stream_error_object(&serde_json::json!({"code": "server_error"}), "fallback"),
            MekaError::RetryableProvider { ref message, .. } if message == "fallback"
        ));
        assert!(matches!(
            provider_stream_error_object(&serde_json::json!({"type": "overloaded_error"}), "x"),
            MekaError::RetryableProvider { .. }
        ));
        assert!(matches!(
            provider_stream_error_object(
                &serde_json::json!({"type": "invalid_request_error"}),
                "x"
            ),
            MekaError::Provider(_)
        ));
    }

    #[test]
    fn a_stream_error_is_retryable_only_for_a_documented_transient_code() {
        for code in [
            "overloaded_error",
            "rate_limit_error",
            "api_error",
            "server_error",
            "rate_limit_exceeded",
            "overloaded",
        ] {
            assert!(
                matches!(
                    provider_stream_error(code, "x".to_string()),
                    MekaError::RetryableProvider {
                        server_error_on_completion: false,
                        ..
                    }
                ),
                "{code} is documented as transient"
            );
        }
        for code in [
            "invalid_request_error",
            "authentication_error",
            "permission_error",
            "not_found_error",
            "invalid_prompt",
            "unknown",
            "",
        ] {
            assert!(
                matches!(
                    provider_stream_error(code, "x".to_string()),
                    MekaError::Provider(_)
                ),
                "{code} must not be retried"
            );
        }
    }

    #[test]
    fn provider_http_error_maps_overflow() {
        // Anthropic: 400 invalid_request_error / "prompt is too long".
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}"#,
                None,
                ProviderRequest::Completion,
            ),
            MekaError::ContextOverflow(_)
        ));
        // OpenAI: context_length_exceeded.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"code":"context_length_exceeded"}}"#,
                None,
                ProviderRequest::Completion,
            ),
            MekaError::ContextOverflow(_)
        ));
        // 413 Payload Too Large is an overflow regardless of body.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::PAYLOAD_TOO_LARGE,
                "Request Entity Too Large",
                None,
                ProviderRequest::Completion,
            ),
            MekaError::ContextOverflow(_)
        ));
    }

    /// The one flag that decides whether the agent loop may answer a failure by deleting the
    /// turn's content, pinned at the only place that can ever set it true.
    ///
    /// Nothing else covered this. `MockProvider` hard-codes `true`, so every agent-level test
    /// asserts what the mock was told rather than what the classifier decides, and both halves of
    /// this expression could be replaced by a constant with the whole suite still green. A `true`
    /// here is not a cosmetic bug: it is the data-loss shape, where a rate limit or an auxiliary
    /// call's outage licenses a degrade whose successful retry is then written to the store as
    /// proven-good.
    #[test]
    fn only_a_server_error_answering_a_completion_may_blame_the_content() {
        let flag = |status, request| match provider_http_error(
            status,
            "upstream said no",
            None,
            request,
        ) {
            MekaError::RetryableProvider {
                server_error_on_completion,
                ..
            } => Some(server_error_on_completion),
            _ => None,
        };

        assert_eq!(
            flag(
                reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                ProviderRequest::Completion
            ),
            Some(true),
            "a 5xx answering a completion is the one shape the turn's own content could explain"
        );
        assert_eq!(
            flag(
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
                ProviderRequest::Completion
            ),
            Some(true),
            "and every 5xx status, not just 500"
        );
        assert_eq!(
            flag(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                ProviderRequest::Completion
            ),
            Some(false),
            "a rate limit is a statement about rate, not about what was sent"
        );
        assert_eq!(
            flag(
                reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                ProviderRequest::Auxiliary
            ),
            Some(false),
            "an auxiliary call carried no conversation, so there is nothing in it to degrade"
        );
    }

    #[test]
    fn provider_http_error_maps_other_as_provider() {
        // Nothing the agent loop can repair: the credentials or the endpoint are wrong.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::UNAUTHORIZED,
                r#"{"error":{"type":"authentication_error"}}"#,
                None,
                ProviderRequest::Completion,
            ),
            MekaError::Provider(_)
        ));
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::NOT_FOUND,
                "not found",
                None,
                ProviderRequest::Completion
            ),
            MekaError::Provider(_)
        ));
    }

    #[test]
    fn provider_http_error_maps_malformed_request() {
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"type":"invalid_request_error","message":"messages.34.content.0.tool_result.content.1.image.source.base64: The image was specified using the image/png media type, but the image appears to be a image/jpeg image"}}"#,
                None,
                ProviderRequest::Completion,
            ),
            MekaError::InvalidRequest(_)
        ));
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::UNPROCESSABLE_ENTITY,
                "invalid",
                None,
                ProviderRequest::Completion
            ),
            MekaError::InvalidRequest(_)
        ));
    }

    /// An overflow also arrives as a 400 `invalid_request_error`, and compacting is the right
    /// response to it rather than degrading content.
    #[test]
    fn provider_http_error_overflow_takes_priority_over_invalid_request() {
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"type":"invalid_request_error","message":"prompt is too long"}}"#,
                None,
                ProviderRequest::Completion,
            ),
            MekaError::ContextOverflow(_)
        ));
    }

    /// A 400 from a probe must never become the error that strips images from a turn.
    ///
    /// [`MekaError::InvalidRequest`] does not mean "malformed". It means "degrade the content this
    /// turn appended and try again", and the agent loop acts on it by deleting image blocks from
    /// the conversation. That is coherent for a completion and nonsense for a usage or identity
    /// probe, which carries no turn content at all.
    ///
    /// Without the `request` parameter every probe answered 400 with `InvalidRequest`, and only
    /// the absence of a probe inside `run_turn` kept a usage endpoint's 400 from silently deleting
    /// a user's attachments. A pre-flight quota check or context-window lookup added later would
    /// have done exactly that.
    #[test]
    fn only_a_completion_can_ask_the_agent_loop_to_degrade_its_content() {
        for status in [
            reqwest::StatusCode::BAD_REQUEST,
            reqwest::StatusCode::UNPROCESSABLE_ENTITY,
        ] {
            assert!(
                matches!(
                    provider_http_error(status, "malformed", None, ProviderRequest::Completion),
                    MekaError::InvalidRequest(_)
                ),
                "{status} on a completion is the repair path's whole trigger"
            );
            assert!(
                matches!(
                    provider_http_error(status, "malformed", None, ProviderRequest::Auxiliary),
                    MekaError::Provider(_)
                ),
                "{status} on a probe has nothing to degrade and must stay terminal"
            );
        }

        // The other classifications are about the response rather than the request, so the kind
        // must not disturb them. An overflow is an overflow and a 529 is retryable either way.
        for request in [ProviderRequest::Completion, ProviderRequest::Auxiliary] {
            assert!(matches!(
                provider_http_error(
                    reqwest::StatusCode::BAD_REQUEST,
                    "prompt is too long",
                    None,
                    request,
                ),
                MekaError::ContextOverflow(_)
            ));
            assert!(matches!(
                provider_http_error(
                    reqwest::StatusCode::from_u16(529).expect("valid status"),
                    "Overloaded",
                    None,
                    request,
                ),
                MekaError::RetryableProvider { .. }
            ));
        }
    }

    /// A client whose only job is to fail fast, for the transport tests below.
    fn probe_client() -> reqwest::Client {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(500))
            .build()
            .expect("a client with no exotic configuration builds")
    }

    /// A TCP port with nothing behind it, so connecting to it is refused rather than hung.
    ///
    /// Bound and dropped rather than hardcoded: a fixed port is a test that passes until the day
    /// something else on the machine happens to be listening on it.
    fn dead_port() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);
        port
    }

    /// A call that got no answer is retryable, which is the whole point of the function.
    ///
    /// The case that is terminal if `.send()` failures are classified as plain `Provider`: a
    /// connection reset while sending ends the turn, while the identical reset *after* the response
    /// started gets three retries. Nothing reached the caller and the request is unchanged, so
    /// asking again costs another round trip and nothing else. (Only for a send. The function's doc
    /// explains why the same is not true of the body reads it also classifies.)
    #[tokio::test]
    async fn a_provider_call_that_never_answered_is_retryable() {
        let url = format!("http://127.0.0.1:{}/v1/messages", dead_port());
        let error = probe_client()
            .get(&url)
            .send()
            .await
            .expect_err("nothing is listening there");
        assert!(
            !error.is_builder(),
            "the request itself was fine; it is the network that refused: {error}"
        );

        match provider_transport_error("HTTP request failed (body 2.0 MiB)", &error, None) {
            MekaError::RetryableProvider {
                message,
                retry_after,
                server_error_on_completion,
            } => {
                assert_eq!(
                    retry_after, None,
                    "the hint is a response header, and the premise here is that no response came"
                );
                // The same premise decides this: nothing judged the body, so the agent loop's
                // degrade-and-retry must not treat the failure as something the content explains.
                // Otherwise a dropped connection answers itself by deleting the turn's work.
                assert!(
                    !server_error_on_completion,
                    "a call that never reached the provider cannot blame what it carried"
                );
                assert!(
                    message.starts_with("HTTP request failed (body 2.0 MiB): "),
                    "the call site's own description leads the message: {message}"
                );
                assert!(
                    message.contains(&url),
                    "and the cause chain follows it, naming what was attempted: {message}"
                );
            }
            other => panic!("expected a retryable failure, got {other:?}"),
        }
    }

    /// A request meka could not build is not retried, because it will not build next time either.
    ///
    /// Three shapes of the same fault, all of which reqwest reports as a builder error before a
    /// socket is ever opened: no host, no scheme, and a scheme reqwest does not speak.
    #[tokio::test]
    async fn a_request_that_could_not_be_built_is_not_retried() {
        for url in ["http://", "not a url", "ftp://example.invalid/"] {
            // `let ... else` so the panic message names which of the three URLs failed. The earlier
            // `expect_err("'{url}' …")` passed a literal, so the braces printed verbatim.
            let Err(error) = probe_client().get(url).send().await else {
                panic!("'{url}' is not a request that can be made");
            };
            assert!(
                error.is_builder(),
                "'{url}' should fail before the network is involved: {error}"
            );
            assert!(
                matches!(
                    provider_transport_error("HTTP request failed", &error, None),
                    MekaError::Provider(_)
                ),
                "'{url}' is meka's fault and retrying it changes nothing"
            );
        }
    }

    /// A timeout is retryable, and the predicate that once excluded it could not tell it apart.
    ///
    /// The exclusion was `is_timeout() && !is_connect()`, read as "delivered, so possibly already
    /// generating". A server that accepts and then says nothing does produce that pattern, but
    /// `is_timeout` matches any `io::ErrorKind::TimedOut` anywhere in the source chain, so a write
    /// that times out on a pooled connection matches it too, having delivered nothing. Whether
    /// meka retries cannot rest on it.
    ///
    /// Loopback only: a blackholed TEST-NET-3 address needs a default route that silently drops
    /// packets, and offline CI or `--network=none` answers `ENETUNREACH` at once instead.
    #[tokio::test]
    async fn a_timeout_is_retryable_whichever_half_of_the_call_it_lands_in() {
        // Accepts the connection, then never answers.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let address = listener.local_addr().expect("the bound address");
        let silent = tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((socket, _)) = listener.accept().await {
                held.push(socket);
            }
        });

        let error = reqwest::Client::builder()
            .read_timeout(Duration::from_millis(400))
            .build()
            .expect("client")
            .get(format!("http://{address}/"))
            .send()
            .await
            .expect_err("the server never answers");
        silent.abort();

        assert!(
            error.is_timeout() && !error.is_connect(),
            "the shape the old exclusion keyed on: {error}"
        );
        assert!(
            matches!(
                provider_transport_error("HTTP request failed", &error, None),
                MekaError::RetryableProvider { .. }
            ),
            "a timeout is a transient failure like any other; the wall-clock budget in \
             `should_retry_provider_error` is what bounds what it costs"
        );
    }

    /// A hint the caller already read is carried through, and one it never had is not invented.
    ///
    /// This is the classifier's half of the contract only: it asserts that whatever the third
    /// argument holds survives into the error, whichever `reqwest::Error` it is paired with. The
    /// read sites' half (that each of them passes the `Retry-After` it parsed rather than `None`)
    /// is asserted at a real site by
    /// `provider::anthropic::messages::tests::a_truncated_body_keeps_the_rate_limit_hint`, because
    /// nothing here can see whether a site called this function with the hint or without it.
    ///
    /// Dropping the hint at the read sites sends the agent loop back at a rate-limited endpoint on
    /// plain 1s/2s backoff after a 429 saying "wait 60".
    #[tokio::test]
    async fn a_hint_the_response_gave_survives_a_failed_body_read() {
        let port = dead_port();
        let error = probe_client()
            .get(format!("http://127.0.0.1:{port}/"))
            .send()
            .await
            .expect_err("nothing is listening there");

        match provider_transport_error(
            "failed to read response",
            &error,
            Some(Duration::from_secs(60)),
        ) {
            MekaError::RetryableProvider { retry_after, .. } => assert_eq!(
                retry_after,
                Some(Duration::from_secs(60)),
                "the hint the site parsed before reading must reach the retry loop"
            ),
            other => panic!("expected a retryable failure, got {other:?}"),
        }
        match provider_transport_error("HTTP request failed", &error, None) {
            MekaError::RetryableProvider { retry_after, .. } => assert_eq!(
                retry_after, None,
                "a send failure saw no headers, so there is no hint to pass on"
            ),
            other => panic!("expected a retryable failure, got {other:?}"),
        }
    }

    /// A rejected refresh is terminal, and says what to do about it.
    ///
    /// Two failures in one, and they are separate. Treating a dead grant as retryable sends the
    /// agent loop back at the authorization server three more times with a token it has already
    /// refused, and a terminal error with no remedy leaves the user reading `invalid_grant` with
    /// nothing to act on. The profile has to be named because two accounts of one backend can
    /// coexist, so "log in again" alone does not say where.
    #[test]
    fn a_rejected_refresh_is_terminal_and_names_the_remedy() {
        let error = oauth_refresh_error(
            "OAuth token refresh failed",
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":"invalid_grant"}"#,
            None,
            "work",
        );
        let MekaError::Provider(message) = error else {
            panic!("a refused grant must not be retried, got: {error:?}");
        };
        assert!(message.contains("invalid_grant"), "{message}");
        assert!(message.contains("meka account login work"), "{message}");
    }

    /// A token endpoint that is merely unwell is retried, hint and all.
    ///
    /// The bug the split exists for: without it a 503 here ends the turn while a 503 from the
    /// completions endpoint two lines later is retried, and since the refresh runs inside
    /// `complete`/`stream` it takes the turn down with it. Neither status usually means the grant
    /// was read, which is the judgment `oauth_refresh_error` documents and hedges; this asserts
    /// the classification that follows from it, not the judgment itself.
    #[test]
    fn an_unwell_token_endpoint_is_retried() {
        for status in [
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
        ] {
            let error = oauth_refresh_error(
                "OAuth token refresh failed",
                status,
                "slow down",
                Some(Duration::from_secs(30)),
                "work",
            );
            assert!(
                matches!(
                    error,
                    MekaError::RetryableProvider {
                        retry_after: Some(delay),
                        ..
                    } if delay == Duration::from_secs(30)
                ),
                "{status} should be retried carrying its hint, got: {error:?}"
            );
        }
    }

    /// A success whose body could not be read is the case where retrying is certainly useless.
    ///
    /// The server accepted the grant, so under rotation it is spent and its replacement was in the
    /// response that did not arrive. This is the one place the transport rule's premise inverts
    /// completely: there is nothing to ask again for, only a login to redo.
    #[test]
    fn a_success_that_could_not_be_read_back_is_terminal() {
        let error = oauth_refresh_error(
            "OAuth token refresh failed to read the response",
            reqwest::StatusCode::OK,
            "error decoding response body",
            None,
            "personal",
        );
        let MekaError::Provider(message) = error else {
            panic!("the grant was accepted, so nothing is left to retry: {error:?}");
        };
        assert!(message.contains("meka account login personal"), "{message}");
    }

    /// A route that loops is not retried either, for the same reason: it is a property of the
    /// destination rather than of the network, so the next attempt loops identically.
    ///
    /// Served locally rather than mocked, because `is_redirect` is only reachable by exhausting
    /// reqwest's redirect policy, and a hand-built error would be testing the test.
    #[tokio::test]
    async fn a_redirect_loop_is_not_retried() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let address = listener.local_addr().expect("the bound address");
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut discard = [0u8; 1024];
                if socket.read(&mut discard).await.is_err() {
                    continue;
                }
                if socket
                    .write_all(b"HTTP/1.1 302 Found\r\nLocation: /\r\nContent-Length: 0\r\n\r\n")
                    .await
                    .is_err()
                {
                    // reqwest gives up after its redirect cap and drops the connection, so the last
                    // write fails. Nothing to do but stop serving this one and wait for the next
                    // `accept`, which is what `break` does here.
                    break;
                }
            }
        });

        // Not `probe_client`, whose 500ms connect timeout is there for the dead-port tests beside
        // this one; here the server is local and the error under test is only reached by walking
        // reqwest's whole redirect chain.
        //
        // Pooling is off because the server above answers one request per `accept()` and then drops
        // the socket. A later hop that reuses a pooled connection therefore writes into a closed
        // one and fails as a transport error, and the assertion below sees that instead of the
        // redirect. Raising the connect timeout did not fix this: it was never a connect timeout.
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(0)
            .build()
            .expect("a client with no exotic configuration builds");
        let error = client
            .get(format!("http://{address}/"))
            .send()
            .await
            .expect_err("a server that only ever redirects to itself");
        server.abort();

        assert!(error.is_redirect(), "{error}");
        assert!(matches!(
            provider_transport_error("HTTP request failed", &error, None),
            MekaError::Provider(_)
        ));
    }

    #[test]
    fn provider_http_error_maps_retryable() {
        // 429 rate limit.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                "rate limit exceeded",
                None,
                ProviderRequest::Completion,
            ),
            MekaError::RetryableProvider { .. }
        ));
        // 5xx server errors, including Anthropic's non-standard 529 "overloaded".
        for status in [500u16, 502, 503, 504, 529] {
            let status = reqwest::StatusCode::from_u16(status).expect("valid status");
            assert!(
                matches!(
                    provider_http_error(status, "Overloaded", None, ProviderRequest::Completion),
                    MekaError::RetryableProvider { .. }
                ),
                "status {status} should be retryable"
            );
        }
    }

    #[test]
    fn provider_http_error_retryable_carries_retry_after() {
        let delay = Duration::from_secs(7);
        match provider_http_error(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            "rate limited",
            Some(delay),
            ProviderRequest::Completion,
        ) {
            MekaError::RetryableProvider { retry_after, .. } => {
                assert_eq!(retry_after, Some(delay));
            }
            other => panic!("expected RetryableProvider, got {other:?}"),
        }
    }

    /// The overflow phrases are read on a 4xx and nowhere else.
    ///
    /// Consulting the body before the status classifies a 500 mentioning the context window as an
    /// overflow. The body is not meka's to trust that far. A server that fails while echoing the
    /// request back (which providers do) turns a transient 500 into an emergency compaction, so a
    /// turn whose text merely discussed `context_length_exceeded` has its context destroyed to
    /// answer a blip, and skips both the retry and the outage reprieve on the way.
    ///
    /// Both arms, because a rule that stopped reading the body at all would silently break the
    /// 400-shaped overflow every backend actually sends.
    ///
    /// `429` is asserted alongside the 5xx rather than left to the sibling test, because it is the
    /// same rule and was fixed one release later than the 5xx was: a rate limit says how often meka
    /// is asking, never how large the request is. It is also the likeliest status to carry an
    /// echoed body, since a busy session meets it first. The existing `429` tests use bodies with
    /// no overflow phrase in them, so they pass whichever way this goes and cover nothing here.
    #[test]
    fn the_overflow_phrases_are_read_on_a_4xx_and_not_on_a_5xx() {
        assert!(
            matches!(
                provider_http_error(
                    reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                    "context_length_exceeded",
                    None,
                    ProviderRequest::Completion,
                ),
                MekaError::RetryableProvider {
                    server_error_on_completion: true,
                    ..
                }
            ),
            "a 5xx is the server saying it failed, never that the request was too big"
        );
        assert!(
            matches!(
                provider_http_error(
                    reqwest::StatusCode::TOO_MANY_REQUESTS,
                    "context_length_exceeded",
                    None,
                    ProviderRequest::Completion,
                ),
                MekaError::RetryableProvider { .. }
            ),
            "a 429 is a rate limit whatever its body says, so it waits rather than compacting"
        );
        // Every phrase, not one: they are an `||` chain, and a chain is the shape where one
        // wrong operator still passes a single-value test while quietly requiring all of them.
        for phrase in [
            "prompt is too long",
            "context_length_exceeded",
            "context length exceeded",
            "maximum context length",
            "exceeds the maximum context",
        ] {
            assert!(
                matches!(
                    provider_http_error(
                        reqwest::StatusCode::BAD_REQUEST,
                        phrase,
                        None,
                        ProviderRequest::Completion,
                    ),
                    MekaError::ContextOverflow(_)
                ),
                "'{phrase}' is a shape a backend actually sends and has to be recognized alone"
            );
        }
        assert!(
            matches!(
                provider_http_error(
                    reqwest::StatusCode::PAYLOAD_TOO_LARGE,
                    "no useful body",
                    None,
                    ProviderRequest::Completion,
                ),
                MekaError::ContextOverflow(_)
            ),
            "413 is a status rather than a guess, so it needs no phrase"
        );
    }

    /// The body is the tail of the message, so whitespace on it is whitespace on the console line.
    #[test]
    fn a_servers_error_body_reaches_the_message_without_its_surrounding_whitespace() {
        let error = provider_http_error(
            reqwest::StatusCode::NOT_FOUND,
            "{\"error\":\"model not found\"}\n\n",
            None,
            ProviderRequest::Completion,
        );
        let rendered = error.to_string();
        assert!(
            rendered.ends_with(r#"{"error":"model not found"}"#),
            "trailing newlines survived into the message: {rendered:?}"
        );
    }

    /// A body that is only whitespace would otherwise leave the message ending in a dangling colon.
    #[test]
    fn an_absent_error_body_is_named_rather_than_left_blank() {
        let error = provider_http_error(
            reqwest::StatusCode::BAD_GATEWAY,
            "\n",
            None,
            ProviderRequest::Completion,
        );
        assert!(
            error
                .to_string()
                .ends_with("502 Bad Gateway: no response body"),
            "an empty body should be named: {:?}",
            error.to_string()
        );
    }

    #[test]
    fn parse_retry_after_present_integer_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "30".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(30)));
    }

    #[test]
    fn parse_retry_after_absent() {
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after(&headers), None);
    }

    #[test]
    fn parse_retry_after_ignores_http_date_form() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        assert_eq!(parse_retry_after(&headers), None);
    }

    #[test]
    fn parse_retry_after_ignores_malformed_value() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "not-a-number".parse().unwrap(),
        );
        assert_eq!(parse_retry_after(&headers), None);
    }
}
