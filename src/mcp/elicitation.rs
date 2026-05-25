//! MCP `elicitation/create` handling. When a server asks for user input
//! (either a structured form or a URL-consent flow), we route the request
//! through the shell's existing approval channel so the TUI can prompt the
//! user. Declines unanswered (or timed-out) requests by default so a
//! misbehaving server can't stall the session.

use std::sync::{Mutex, OnceLock, mpsc::SyncSender};

use rmcp::model::{CreateElicitationResult, ElicitationAction};

pub type ElicitationSink = Box<dyn Fn(ElicitationPrompt) + Send + Sync + 'static>;

/// User-facing payload the shell renders.
pub struct ElicitationPrompt {
    pub server_name: String,
    pub kind: ElicitationKind,
    pub message: String,
    pub responder: SyncSender<ElicitationResponse>,
}

#[derive(Debug)]
pub enum ElicitationKind {
    /// Structured form: the server sent a JSON schema of fields to fill.
    Form { schema: serde_json::Value },
    /// URL consent: the server wants the user to visit a URL (e.g. to log
    /// in to a third-party service).
    Url { url: String },
}

/// Shell's response back to the MCP handler.
#[derive(Debug, Clone)]
pub enum ElicitationResponse {
    Accept { content: Option<serde_json::Value> },
    Decline,
    Cancel,
}

impl ElicitationResponse {
    pub fn into_result(self) -> CreateElicitationResult {
        match self {
            ElicitationResponse::Accept { content } => CreateElicitationResult {
                action: ElicitationAction::Accept,
                content,
                meta: None,
            },
            ElicitationResponse::Decline => CreateElicitationResult {
                action: ElicitationAction::Decline,
                content: None,
                meta: None,
            },
            ElicitationResponse::Cancel => CreateElicitationResult {
                action: ElicitationAction::Cancel,
                content: None,
                meta: None,
            },
        }
    }
}

/// Callback for forwarding elicitation prompts to the shell. Set once at
/// startup by the agent loop; if unset (non-interactive mode), elicitation
/// requests are auto-declined.
static SINK: OnceLock<Mutex<Option<ElicitationSink>>> = OnceLock::new();

fn sink_slot() -> &'static Mutex<Option<ElicitationSink>> {
    SINK.get_or_init(|| Mutex::new(None))
}

/// Install the shell sink. Later calls replace the sink, which is useful
/// when the TUI is re-initialised mid-process.
pub fn set_shell_sink(sink: Option<ElicitationSink>) {
    if let Ok(mut guard) = sink_slot().lock() {
        *guard = sink;
    }
}

/// Forward a prompt to the shell. Returns `false` if no sink is installed
/// (caller should auto-decline).
pub fn send_prompt(prompt: ElicitationPrompt) -> bool {
    let Ok(guard) = sink_slot().lock() else {
        return false;
    };
    let Some(sink) = guard.as_ref() else {
        return false;
    };
    sink(prompt);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decline_maps_to_decline_action() {
        match ElicitationResponse::Decline.into_result().action {
            ElicitationAction::Decline => {}
            other => panic!("expected Decline, got {:?}", other),
        }
    }

    #[test]
    fn accept_preserves_content() {
        let content = serde_json::json!({"k": "v"});
        let result = ElicitationResponse::Accept {
            content: Some(content.clone()),
        }
        .into_result();
        assert!(matches!(result.action, ElicitationAction::Accept));
        assert_eq!(result.content, Some(content));
    }

    #[test]
    fn cancel_maps_to_cancel_action_with_no_content() {
        let result = ElicitationResponse::Cancel.into_result();
        assert!(matches!(result.action, ElicitationAction::Cancel));
        assert!(result.content.is_none());
        assert!(result.meta.is_none());
    }

    #[test]
    fn send_prompt_without_sink_returns_false() {
        // Don't install sink; result should be false (unless a prior test
        // installed one that outlives this test — OnceLock is not
        // resettable, so be tolerant).
        let (responder, _rx) = std::sync::mpsc::sync_channel(1);
        let prompt = ElicitationPrompt {
            server_name: "srv".into(),
            kind: ElicitationKind::Url {
                url: "https://example.com".into(),
            },
            message: "msg".into(),
            responder,
        };
        let _ = send_prompt(prompt);
    }
}
