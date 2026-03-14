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

    #[error("session not found: {0}")]
    SessionNotFound(uuid::Uuid),

    #[error("session already attached by another process: {0}")]
    SessionLocked(uuid::Uuid),

    #[error("agent interrupted by user")]
    Interrupted,

    #[error("SSE stream error: {0}")]
    StreamError(String),
}

pub type Result<T> = std::result::Result<T, AgshError>;
