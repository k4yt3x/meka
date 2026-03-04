use std::borrow::Cow;

use crossterm::style::{Color, Stylize};
use reedline::{
    EditCommand, Emacs, KeyCode, KeyModifiers, Prompt, PromptEditMode, PromptHistorySearch,
    PromptHistorySearchStatus, Reedline, ReedlineEvent, Signal, default_emacs_keybindings,
};

use crate::permission::SharedPermission;

const CYCLE_PERMISSION_SENTINEL: &str = "__cycle_permission__";

struct AgshPrompt {
    shared_permission: SharedPermission,
}

impl Prompt for AgshPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("agsh ")
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> Cow<'_, str> {
        let permission = self.shared_permission.get();
        let colored_indicator =
            format!("[{}]", permission.indicator()).with(permission.indicator_color());
        Cow::Owned(format!("{} > ", colored_indicator))
    }

    fn get_prompt_color(&self) -> Color {
        Color::White
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("  ... ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        Cow::Owned(format!(
            "({}reverse-i-search `{}')",
            prefix, history_search.term
        ))
    }

    fn get_indicator_color(&self) -> Color {
        Color::Reset
    }
}

fn build_reedline_editor() -> Reedline {
    let mut keybindings = default_emacs_keybindings();

    keybindings.add_binding(
        KeyModifiers::SHIFT,
        KeyCode::BackTab,
        ReedlineEvent::ExecuteHostCommand(CYCLE_PERMISSION_SENTINEL.to_string()),
    );

    keybindings.add_binding(
        KeyModifiers::ALT,
        KeyCode::Enter,
        ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
    );

    let emacs_mode = Emacs::new(keybindings);
    Reedline::create().with_edit_mode(Box::new(emacs_mode))
}

pub enum ShellEvent {
    UserInput(String),
    Exit,
}

pub fn run_repl(
    shared_permission: SharedPermission,
    input_sender: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    agent_done_receiver: std::sync::mpsc::Receiver<()>,
) {
    let mut editor = build_reedline_editor();
    let prompt = AgshPrompt {
        shared_permission: shared_permission.clone(),
    };

    loop {
        match editor.read_line(&prompt) {
            Ok(Signal::Success(buffer)) => {
                if buffer == CYCLE_PERMISSION_SENTINEL {
                    let new_permission = shared_permission.cycle();
                    tracing::debug!("permission cycled to {}", new_permission);
                    continue;
                }

                let trimmed = buffer.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                    let _ = input_sender.send(ShellEvent::Exit);
                    break;
                }

                if let Some(shell_command) = trimmed.strip_prefix('!') {
                    if shell_command.is_empty() {
                        continue;
                    }
                    #[cfg(windows)]
                    let status = std::process::Command::new("powershell")
                        .arg("-Command")
                        .arg(shell_command)
                        .status();

                    #[cfg(not(windows))]
                    let status = std::process::Command::new("sh")
                        .arg("-c")
                        .arg(shell_command)
                        .status();
                    match status {
                        Ok(exit_status) => {
                            if !exit_status.success() {
                                if let Some(code) = exit_status.code() {
                                    eprintln!("Command exited with status {}", code);
                                }
                            }
                        }
                        Err(error) => {
                            eprintln!("Failed to execute command: {}", error);
                        }
                    }
                    continue;
                }

                if input_sender
                    .send(ShellEvent::UserInput(trimmed.to_string()))
                    .is_err()
                {
                    break;
                }

                if agent_done_receiver.recv().is_err() {
                    break;
                }
            }
            Ok(Signal::CtrlC) => {
                continue;
            }
            Ok(Signal::CtrlD) => {
                let _ = input_sender.send(ShellEvent::Exit);
                break;
            }
            Err(error) => {
                tracing::error!("readline error: {}", error);
                let _ = input_sender.send(ShellEvent::Exit);
                break;
            }
        }
    }
}
