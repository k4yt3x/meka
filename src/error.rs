//! Crate-wide [`MekaError`] enum and [`Result`] alias. All non-binary code paths return `Result<T,
//! MekaError>`; the `main` binary wraps these in `anyhow::Result` for top-level reporting.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MekaError {
    #[error("configuration error: {0}")]
    Config(String),

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

    #[error("tool execution error: {tool_name}: {message}")]
    ToolExecution { tool_name: String, message: String },

    #[error("tool registration error: {message}")]
    ToolRegistration { message: String },

    #[error("session already attached by another process: {0}")]
    SessionLocked(uuid::Uuid),

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

    /// Strict MCP gate rejected the turn: at least one enabled server wasn't `Connected` within the
    /// configured grace period. Turn contents haven't been sent to the provider. The REPL catches
    /// this and loops back to the prompt; one-shot mode propagates to a non-zero process exit.
    #[error("mcp: {} server(s) not ready: {}", .servers.len(), .servers.iter().map(|(n, s)| format!("{} ({})", n, s)).collect::<Vec<_>>().join(", "))]
    McpTurnGated { servers: Vec<(String, String)> },
}

pub type Result<T> = std::result::Result<T, MekaError>;

/// Classify a provider HTTP failure response: map context-window overflows to
/// [`MekaError::ContextOverflow`] (so the agent loop can compact-and-retry once) and everything
/// else to [`MekaError::Provider`]. Anthropic returns HTTP 400 `invalid_request_error` with "prompt
/// is too long"; OpenAI returns 400 `context_length_exceeded` (or 413). Matched on the body because
/// a bare 400 is shared with many unrelated errors.
pub(crate) fn provider_http_error(status: reqwest::StatusCode, body: &str) -> MekaError {
    let lower = body.to_ascii_lowercase();
    let overflow = status == reqwest::StatusCode::PAYLOAD_TOO_LARGE
        || lower.contains("prompt is too long")
        || lower.contains("context_length_exceeded")
        || lower.contains("context length exceeded")
        || lower.contains("maximum context length")
        || lower.contains("exceeds the maximum context");
    let message = format!("API returned status {status}: {body}");
    if overflow {
        MekaError::ContextOverflow(message)
    } else {
        MekaError::Provider(message)
    }
}

/// Format a [`reqwest::Error`] together with its full source chain.
///
/// reqwest's outer Display string ("error sending request for url …") usually hides the actual
/// cause (TCP reset, HTTP/2 GOAWAY, TLS handshake failure, connect timeout, DNS resolution failure,
/// …). Walking [`std::error::Error::source`] surfaces the underlying reason inline, so users (and
/// bug reports) see what actually broke instead of reqwest's generic wrapper.
///
/// Used at every site that wraps a `reqwest::Error` in an `MekaError` via Display formatting.
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

    #[test]
    fn test_provider_http_error_maps_overflow() {
        // Anthropic: 400 invalid_request_error / "prompt is too long".
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}"#,
            ),
            MekaError::ContextOverflow(_)
        ));
        // OpenAI: context_length_exceeded.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"code":"context_length_exceeded"}}"#,
            ),
            MekaError::ContextOverflow(_)
        ));
        // 413 Payload Too Large is an overflow regardless of body.
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::PAYLOAD_TOO_LARGE,
                "Request Entity Too Large"
            ),
            MekaError::ContextOverflow(_)
        ));
    }

    #[test]
    fn test_provider_http_error_maps_other_as_provider() {
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::UNAUTHORIZED,
                r#"{"error":{"type":"authentication_error"}}"#,
            ),
            MekaError::Provider(_)
        ));
        assert!(matches!(
            provider_http_error(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                "rate limit exceeded"
            ),
            MekaError::Provider(_)
        ));
    }
}
