//! Crate-wide [`AgshError`] enum and [`Result`] alias. All non-binary code paths return `Result<T,
//! AgshError>`; the `main` binary wraps these in `anyhow::Result` for top-level reporting.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgshError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("database error: {0}")]
    Database(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("tool execution error: {tool_name}: {message}")]
    ToolExecution { tool_name: String, message: String },

    #[error("tool registration error: {message}")]
    ToolRegistration { message: String },

    #[error("session already attached by another process: {0}")]
    SessionLocked(uuid::Uuid),

    #[error("agent interrupted by user")]
    Interrupted,

    /// A logic invariant in agsh itself was violated. Used in place of `.expect()` for cases where
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

pub type Result<T> = std::result::Result<T, AgshError>;

/// Format a [`reqwest::Error`] together with its full source chain.
///
/// reqwest's outer Display string ("error sending request for url …") usually hides the actual
/// cause (TCP reset, HTTP/2 GOAWAY, TLS handshake failure, connect timeout, DNS resolution failure,
/// …). Walking [`std::error::Error::source`] surfaces the underlying reason inline, so users (and
/// bug reports) see what actually broke instead of reqwest's generic wrapper.
///
/// Used at every site that wraps a `reqwest::Error` in an `AgshError` via Display formatting.
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
