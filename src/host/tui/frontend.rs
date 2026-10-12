//! What reaches the screen: the events the agent emits, the questions it asks, and what the host
//! loop has to say, all as one stream the UI task drains. The [`TuiFrontend`] is the agent's end
//! of it; [`Ui`] is the host's.

use async_trait::async_trait;

use super::model::Item;
use crate::frontend::{
    Frontend, FrontendEvent, Notice, PermissionOutcome, PermissionRequest, StickyApprovals,
};

/// What the user decided at an approval prompt.
///
/// The two sticky variants answer for the tool rather than the call, so the frontend records them
/// before mapping the decision to a [`PermissionOutcome`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApprovalDecision {
    Allow,
    /// Allow this call and every later call to the tool this session.
    AllowAlways,
    Deny,
    /// Deny this call and every later call to the tool this session.
    DenyAlways,
}

impl ApprovalDecision {
    pub(crate) fn allows(self) -> bool {
        matches!(self, Self::Allow | Self::AllowAlways)
    }
}

/// A tool call waiting on the user's answer.
pub(crate) struct ApprovalRequest {
    pub(crate) tool_name: String,
    /// Every argument the call was made with, shown in full: a prompt that hides the write is not
    /// doing its job.
    pub(crate) input: serde_json::Value,
    /// Fires when the turn is stopped, which withdraws the question.
    pub(crate) cancellation: tokio_util::sync::CancellationToken,
    pub(crate) responder: tokio::sync::oneshot::Sender<ApprovalDecision>,
}

/// One configured MCP server as the picker lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct McpServerRow {
    pub(crate) name: String,
    /// `connected`, `pending`, `disabled`, or `failed: <why>`.
    pub(crate) state: String,
}

/// What the UI task receives.
pub(crate) enum UiEvent {
    Agent(FrontendEvent),
    Approval(ApprovalRequest),
    Elicitation {
        prompt: crate::frontend::ElicitationPrompt,
        responder: tokio::sync::oneshot::Sender<crate::frontend::ElicitationResponse>,
    },
    /// Something the host has to say, as the item it is.
    Item(Item),
    /// The host has finished answering the last thing the UI sent it.
    Done,
    /// A scheduled job came due or a background task has an outcome to report; the UI hands the
    /// host a wake when it is idle.
    Wake,
    /// The host loop has ended.
    Leave,
    /// The log changed under the screen (a rewind, a fork): what is shown is to be read again
    /// from the log's end.
    Reload,
    /// The log's end and its last page, read for a reload.
    Reloaded {
        log_end: Option<i64>,
        page: crate::store::LogPage,
        /// The window it was read for; the screen drops a read for a window it has let go of.
        generation: u64,
    },
    /// A page of the log the screen asked for, above the window or below it.
    HistoryPage {
        above: bool,
        page: crate::store::LogPage,
        /// The window it was read for, as on [`Self::Reloaded`].
        generation: u64,
    },
    /// Each configured MCP server with its state, for the picker a bare `/mcp` opens.
    McpServers(Vec<McpServerRow>),
    /// What a command the host answered has to say: its answer, a refusal, a failure, an empty
    /// list. A panel until Esc, and nothing on the transcript.
    Panel(String),
    /// A `!command` the screen ran has ended: what it printed and how it ended, or why it could
    /// not be started. `stopped` marks one the user killed.
    ShellFinished {
        run: u64,
        output: Result<ShellOutput, String>,
        stopped: bool,
    },
}

/// What a `!command` printed, each stream cut to its tail, and how it ended.
#[derive(Debug)]
pub(crate) struct ShellOutput {
    pub(crate) status: std::process::ExitStatus,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    /// Bytes let go ahead of what was kept, both streams together.
    pub(crate) dropped_bytes: u64,
}

pub(crate) type UiSender = tokio::sync::mpsc::UnboundedSender<UiEvent>;

/// What the screen sends the host loop.
pub(crate) enum HostEvent {
    /// A prompt from the composer, with the images attached to it.
    UserInput {
        text: String,
        images: Vec<crate::image::ImageSource>,
    },
    Command(super::commands::SlashCommand),
    /// Something out-of-band wants a turn: a scheduled job came due, or a background task has an
    /// outcome to report. Carries no payload; the host re-reads both, because between the watcher
    /// noticing and this arriving the job could have been canceled.
    Wake,
    /// The user moved this session's permission level, to be recorded on the row: a scheduled
    /// gate is re-checked at fire time against the row, so the withdrawal has to reach the
    /// database without waiting for a turn.
    PermissionChanged(crate::permission::Permission),
    /// The user toggled whether calls above the level are submitted for approval.
    ApprovalsChanged(bool),
    /// The user moved this session's working directory, which is where the next resume opens
    /// and where a scheduled tool gate is re-checked.
    CwdChanged(std::path::PathBuf),
    /// Leave. `interrupted` marks the ladder's third press, which exits 130 as an interrupted run
    /// does, after the same teardown as any other exit.
    Exit {
        interrupted: bool,
    },
}

/// The host's handle on the screen: every row, panel, question and answer the host wants shown
/// goes through it, as an event the screen applies between frames.
#[derive(Clone)]
pub(crate) struct Ui {
    sender: UiSender,
}

impl Ui {
    pub(crate) fn new(sender: UiSender) -> Self {
        Self { sender }
    }

    pub(crate) fn sender(&self) -> &UiSender {
        &self.sender
    }

    /// Hand the screen an event; a screen that has gone is not an error for the host, which is
    /// about to find out on its own channel.
    pub(crate) fn send(&self, event: UiEvent) {
        if self.sender.send(event).is_err() {
            tracing::trace!("the screen has gone");
        }
    }

    pub(crate) fn item(&self, item: Item) {
        self.send(UiEvent::Item(item));
    }

    /// A note beside the conversation, in the dim register: a command's outcome, a stage
    /// direction, a fact of the session said in passing.
    pub(crate) fn note(&self, note: impl Into<String>) {
        self.item(Item::note(note));
    }

    pub(crate) fn error(&self, error: &dyn std::fmt::Display) {
        self.item(Item::Error(error.to_string()));
    }

    /// The log changed under the screen; it reads the end again.
    pub(crate) fn reload(&self) {
        self.send(UiEvent::Reload);
    }

    /// What a command has to say: its answer, a refusal, a failure, an empty list. A panel until
    /// Esc, and nothing on the transcript.
    pub(crate) fn reply(&self, message: impl Into<String>) {
        self.send(UiEvent::Panel(message.into()));
    }

    /// The configured MCP servers with their states, for the picker.
    pub(crate) fn mcp_servers(&self, servers: Vec<McpServerRow>) {
        self.send(UiEvent::McpServers(servers));
    }

    pub(crate) fn session_id(&self, label: &str, id: &str) {
        self.item(Item::SessionId {
            label: label.to_string(),
            id: id.to_string(),
        });
    }

    /// Say how a turn, or a `/compact`, ended when it did not succeed. An interrupt is the user's
    /// own act, marked rather than reported as an error.
    pub(crate) fn report_failure(&self, error: &crate::error::MekaError) {
        match error {
            crate::error::MekaError::Interrupted => self.item(Item::interrupted()),
            error => self.error(error),
        }
    }

    pub(crate) fn done(&self) {
        self.send(UiEvent::Done);
    }
}

/// The agent's [`Frontend`] for the terminal UI: every event goes to the screen, and every
/// question comes back answered from it.
pub(crate) struct TuiFrontend {
    ui: Ui,
    /// The `always` and `never` answers given at the approval prompt. One frontend serves every
    /// session a run moves through, so these are cleared when a new session starts rather than
    /// with the frontend.
    sticky: StickyApprovals,
}

impl TuiFrontend {
    pub(crate) fn new(ui: Ui) -> Self {
        Self {
            ui,
            sticky: StickyApprovals::default(),
        }
    }

    /// Forget the `always` and `never` answers given so far, for a host that has moved this
    /// frontend onto another session.
    pub(crate) fn forget_session_answers(&self) {
        self.sticky.clear();
    }
}

#[async_trait]
impl Frontend for TuiFrontend {
    /// Reasoning is kept on the transcript whether or not it is shown, so a retried attempt would
    /// deliver it twice.
    fn retains_reasoning(&self) -> bool {
        true
    }

    async fn emit(&self, event: FrontendEvent) {
        if let FrontendEvent::SessionStarted { .. } = &event {
            // A sticky answer is "for the rest of the session", and a first turn creating one
            // arrives here before any call it could be asked about.
            self.forget_session_answers();
        }
        self.ui.send(UiEvent::Agent(event));
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        if let Some(remembered) = self.sticky.remembered(&request.tool_name) {
            return remembered;
        }
        let (responder, receiver) = tokio::sync::oneshot::channel::<ApprovalDecision>();
        let tool_name = request.tool_name.clone();
        let cancellation = request.cancellation.clone();
        if self
            .ui
            .sender()
            .send(UiEvent::Approval(ApprovalRequest {
                tool_name: request.tool_name,
                input: request.input,
                cancellation: cancellation.clone(),
                responder,
            }))
            .is_err()
        {
            // The screen is gone, so there is no human to ask. Denied rather than canceled,
            // because nothing stopped the turn: the call was refused, and the result should say so.
            self.emit(FrontendEvent::Notice(
                Notice::approval_refused_without_asking(&tool_name),
            ))
            .await;
            return PermissionOutcome::Deny;
        }
        // Raced against the turn's own stop: otherwise a Ctrl+C at the prompt does nothing
        // visible, and the next keystroke approves the call the user has just tried to stop.
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => PermissionOutcome::Canceled,
            response = receiver => match response {
                Ok(decision) => {
                    match decision {
                        ApprovalDecision::AllowAlways => self.sticky.remember_allow(&tool_name),
                        ApprovalDecision::DenyAlways => self.sticky.remember_deny(&tool_name),
                        ApprovalDecision::Allow | ApprovalDecision::Deny => {}
                    }
                    if decision.allows() {
                        PermissionOutcome::Allow
                    } else {
                        PermissionOutcome::Deny
                    }
                }
                Err(_) => PermissionOutcome::Canceled,
            },
        }
    }

    async fn handle_elicitation(
        &self,
        prompt: crate::frontend::ElicitationPrompt,
    ) -> crate::frontend::ElicitationResponse {
        let (responder, receiver) =
            tokio::sync::oneshot::channel::<crate::frontend::ElicitationResponse>();
        let server_name = prompt.server_name.clone();
        if self
            .ui
            .sender()
            .send(UiEvent::Elicitation { prompt, responder })
            .is_err()
        {
            self.emit(FrontendEvent::Notice(Notice::elicitation_declined(
                &server_name,
                "no interactive prompt is available here",
            )))
            .await;
            return crate::frontend::ElicitationResponse::Decline;
        }
        receiver
            .await
            .unwrap_or(crate::frontend::ElicitationResponse::Decline)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// A stop during an approval prompt ends the prompt as canceled rather than waiting for the
    /// answer the user is no longer giving.
    #[tokio::test]
    async fn a_stopped_turn_withdraws_its_approval_prompt() {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let frontend = TuiFrontend::new(Ui::new(sender));
        let cancellation = tokio_util::sync::CancellationToken::new();
        cancellation.cancel();
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            frontend.request_permission(PermissionRequest {
                tool_name: "file_write".to_string(),
                primary_param: Some("/tmp/x".to_string()),
                input: serde_json::json!({"path": "/tmp/x"}),
                cancellation,
                subagent_id: None,
            }),
        )
        .await
        .expect("a canceled turn must not wait on the prompt");
        assert_eq!(outcome, PermissionOutcome::Canceled);
        // The question was still put to the screen, which has one to withdraw.
        assert!(matches!(receiver.try_recv(), Ok(UiEvent::Approval(_))));
    }

    /// An `always` answer covers the next call to the tool without asking again, and a session
    /// change forgets it.
    #[tokio::test]
    async fn an_always_answer_covers_the_next_call_until_the_session_changes() {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let frontend = Arc::new(TuiFrontend::new(Ui::new(sender)));
        let request = || PermissionRequest {
            tool_name: "shell_execute".to_string(),
            primary_param: None,
            input: serde_json::json!({"command": "ls"}),
            cancellation: tokio_util::sync::CancellationToken::new(),
            subagent_id: None,
        };
        let asking = tokio::spawn({
            let frontend = Arc::clone(&frontend);
            async move { frontend.request_permission(request()).await }
        });
        let Some(UiEvent::Approval(question)) = receiver.recv().await else {
            panic!("the screen is asked");
        };
        question
            .responder
            .send(ApprovalDecision::AllowAlways)
            .expect("answered");
        assert_eq!(asking.await.expect("joined"), PermissionOutcome::Allow);
        assert_eq!(
            frontend.request_permission(request()).await,
            PermissionOutcome::Allow,
            "the second call is not asked"
        );
        assert!(receiver.try_recv().is_err(), "nothing reached the screen");
        frontend
            .emit(FrontendEvent::SessionStarted {
                id: uuid::Uuid::new_v4(),
            })
            .await;
        let asking = tokio::spawn({
            let frontend = Arc::clone(&frontend);
            async move { frontend.request_permission(request()).await }
        });
        receiver.recv().await.expect("the session start");
        assert!(
            matches!(receiver.recv().await, Some(UiEvent::Approval(_))),
            "a new session asks again"
        );
        asking.abort();
    }
}
