//! The frontend of a one-shot run: it keeps the agent's last text block for the host to print when
//! the turn ends, and says nothing while the turn runs.
//!
//! The answer a one-shot prints is the last text block the model wrote, the one after its last tool
//! call, which is what `claude -p` and `codex exec` print of theirs. The thinking, the tool calls,
//! the checklist and the progress are the process, and the process is a log: `-v` shows its
//! signposts. Nothing else reaches either stream, so a pipe gets the answer alone and a log gets
//! the lines it asked for.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::frontend::{
    ElicitationPrompt, ElicitationResponse, Frontend, FrontendEvent, Notice, NoticeLevel,
    PermissionOutcome, PermissionRequest,
};

pub(crate) struct PlainFrontend {
    answer: Mutex<Answer>,
}

/// The turn's text blocks as they arrive, kept to the last one with words in it.
#[derive(Default)]
struct Answer {
    /// The block being written: every delta since the last tool call or nudge.
    current: String,
    /// The last block that ended with words in it, for a turn whose final block has none.
    settled: String,
}

impl Answer {
    /// A tool call or a nudge ends the block being written; what comes after is a new one.
    fn boundary(&mut self) {
        if self.current.trim().is_empty() {
            self.current.clear();
        } else {
            self.settled = std::mem::take(&mut self.current);
        }
    }

    fn last_block(&self) -> &str {
        if self.current.trim().is_empty() {
            &self.settled
        } else {
            &self.current
        }
    }
}

impl PlainFrontend {
    pub(crate) fn new() -> Self {
        Self {
            answer: Mutex::new(Answer::default()),
        }
    }

    /// The last text block the model wrote, empty for a turn that wrote none.
    pub(crate) fn answer(&self) -> String {
        crate::sync::lock(&self.answer).last_block().to_string()
    }
}

#[async_trait]
impl Frontend for PlainFrontend {
    async fn emit(&self, event: FrontendEvent) {
        match event {
            FrontendEvent::AssistantTextDelta(text) => {
                crate::sync::lock(&self.answer).current.push_str(&text);
            }
            FrontendEvent::ToolCallStarted { name, .. } => {
                tracing::info!("calling tool '{name}'");
                crate::sync::lock(&self.answer).boundary();
            }
            FrontendEvent::Nudged { .. } | FrontendEvent::BackgroundReported { .. } => {
                crate::sync::lock(&self.answer).boundary();
            }
            FrontendEvent::SessionStarted { id } => tracing::info!("created session {id}"),
            FrontendEvent::Notice(notice) => log_notice(&notice),
            _ => {}
        }
    }

    /// There is no human to ask. Denied rather than canceled, because nothing stopped the turn: the
    /// call was refused, and the tool result should say so.
    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        self.emit(FrontendEvent::Notice(
            Notice::approval_refused_without_asking(&request.tool_name),
        ))
        .await;
        PermissionOutcome::Deny
    }

    /// No human to ask, as for an approval: declined, and said, because the tool call that needed
    /// the answer then fails with nothing else naming the cause.
    async fn handle_elicitation(&self, prompt: ElicitationPrompt) -> ElicitationResponse {
        self.emit(FrontendEvent::Notice(Notice::elicitation_declined(
            &prompt.server_name,
            "no interactive prompt is available here",
        )))
        .await;
        ElicitationResponse::Decline
    }
}

/// A notice as a log line at its own level: what meka has to say about the turn is not the answer,
/// and the default level shows a warning.
fn log_notice(notice: &Notice) {
    let text = &notice.text;
    match notice.level {
        NoticeLevel::Warn => tracing::warn!("{text}"),
        NoticeLevel::Info => tracing::info!("{text}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(words: &str) -> FrontendEvent {
        FrontendEvent::AssistantTextDelta(words.to_string())
    }

    fn call() -> FrontendEvent {
        FrontendEvent::ToolCallStarted {
            id: "t1".into(),
            name: "schedule_list".into(),
            input: serde_json::json!({}),
            display_summary: None,
        }
    }

    async fn answer_after(events: Vec<FrontendEvent>) -> String {
        let frontend = PlainFrontend::new();
        for event in events {
            frontend.emit(event).await;
        }
        frontend.answer()
    }

    /// The answer is the block after the last tool call, whole, and nothing before it.
    #[tokio::test]
    async fn the_answer_is_the_last_text_block() {
        let answer = answer_after(vec![
            text("Let me "),
            text("check."),
            call(),
            text("Nothing "),
            text("is due."),
        ])
        .await;
        assert_eq!(answer, "Nothing is due.");
    }

    /// A turn that ends on a tool call keeps the block before it, so an answer is never lost to a
    /// call that said nothing after itself.
    #[tokio::test]
    async fn a_turn_ending_on_a_call_keeps_the_block_before_it() {
        assert_eq!(answer_after(vec![text("Done."), call()]).await, "Done.");
        assert_eq!(answer_after(vec![call()]).await, "");
    }

    /// A nudge ends a block as a call does: what the model writes after it is the answer.
    #[tokio::test]
    async fn a_nudge_starts_a_new_block() {
        let nudge = FrontendEvent::Nudged {
            kind: crate::conversation::NudgeKind::Checklist,
            text: String::new(),
        };
        assert_eq!(
            answer_after(vec![text("Planned."), nudge, text("Finished.")]).await,
            "Finished."
        );
    }

    /// There is nobody to ask, so a call that needs approval is denied, and the refusal is not
    /// mistaken for text the model wrote.
    #[tokio::test]
    async fn a_call_nobody_can_approve_is_denied() {
        let frontend = PlainFrontend::new();
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "shell_execute".to_string(),
                primary_param: Some("rm -rf /".to_string()),
                input: serde_json::json!({"command": "rm -rf /"}),
                cancellation: tokio_util::sync::CancellationToken::new(),
                subagent_id: None,
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
        assert_eq!(frontend.answer(), "");
    }

    /// Same for a server's question: declined, and no part of the answer.
    #[tokio::test]
    async fn an_elicitation_nobody_can_answer_is_declined() {
        let frontend = PlainFrontend::new();
        let response = frontend
            .handle_elicitation(ElicitationPrompt {
                server_name: "notion".to_string(),
                message: "authorize?".to_string(),
                kind: crate::frontend::ElicitationKind::Url {
                    url: "https://example.com/".to_string(),
                },
            })
            .await;
        assert!(matches!(response, ElicitationResponse::Decline));
        assert_eq!(frontend.answer(), "");
    }
}
