use std::borrow::Cow;
use std::path::{Path, PathBuf};

use crossterm::style::{Color, Stylize};
use reedline::{
    EditCommand, Emacs, KeyCode, KeyModifiers, Prompt, PromptEditMode, PromptHistorySearch,
    PromptHistorySearchStatus, Reedline, ReedlineEvent, Signal, default_emacs_keybindings,
};

use crate::permission::SharedPermission;

const CYCLE_PERMISSION_SENTINEL: &str = "__cycle_permission__";

struct AgshPrompt {
    shared_permission: SharedPermission,
    show_path: bool,
}

impl Prompt for AgshPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        if self.show_path {
            let cwd = std::env::current_dir()
                .map(|path| shorten_path_with_tilde(&path))
                .unwrap_or_else(|_| "?".to_string());
            Cow::Owned(format!("agsh:{} ", cwd))
        } else {
            Cow::Borrowed("agsh ")
        }
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
        Cow::Borrowed("::: ")
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
    Reedline::create()
        .with_edit_mode(Box::new(emacs_mode))
        .use_bracketed_paste(true)
}

pub enum SlashCommand {
    Exit,
    Help,
    Clear,
    Session,
    Permission(Option<String>),
    Compact,
    Cd(Option<String>),
}

pub enum ShellEvent {
    UserInput(String),
    Command(SlashCommand),
    Exit,
}

fn parse_slash_command(input: &str) -> Option<SlashCommand> {
    let input = input.strip_prefix('/')?;
    let mut parts = input.splitn(2, char::is_whitespace);
    let command = parts.next()?;
    let argument = parts.next().map(|s| s.trim().to_string());

    match command {
        "exit" | "quit" => Some(SlashCommand::Exit),
        "help" | "?" => Some(SlashCommand::Help),
        "clear" => Some(SlashCommand::Clear),
        "session" => Some(SlashCommand::Session),
        "permission" => Some(SlashCommand::Permission(argument)),
        "compact" => Some(SlashCommand::Compact),
        "cd" => Some(SlashCommand::Cd(argument)),
        _ => None,
    }
}

fn print_help() {
    eprintln!("Commands:");
    eprintln!("  /help                          Show this help message");
    eprintln!("  /exit                          Exit the shell");
    eprintln!("  /clear                         Clear the terminal screen");
    eprintln!("  /session                       Show the current session ID");
    eprintln!("  /permission [none|read|write]  Show or set the permission level");
    eprintln!("  /compact                       Summarize and compact the session");
    eprintln!("  /cd <path>                     Change working directory");
    eprintln!();
    eprintln!("Shortcuts:");
    eprintln!("  !<command>    Execute a shell command directly");
    eprintln!("  Shift+Tab     Cycle permission level");
    eprintln!("  Ctrl+D        Exit the shell");
}

pub fn run_repl(
    shared_permission: SharedPermission,
    show_path_in_prompt: bool,
    input_sender: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    agent_done_receiver: std::sync::mpsc::Receiver<()>,
) {
    let mut editor = build_reedline_editor();
    let prompt = AgshPrompt {
        shared_permission: shared_permission.clone(),
        show_path: show_path_in_prompt,
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

                if trimmed.starts_with('/') {
                    match parse_slash_command(trimmed) {
                        Some(SlashCommand::Exit) => {
                            if input_sender.send(ShellEvent::Exit).is_err() {
                                tracing::trace!("shell event receiver already dropped");
                            }
                            break;
                        }
                        Some(SlashCommand::Help) => {
                            print_help();
                            continue;
                        }
                        Some(SlashCommand::Clear) => {
                            if crossterm::execute!(
                                std::io::stdout(),
                                crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
                                crossterm::cursor::MoveTo(0, 0),
                            )
                            .is_err()
                            {
                                eprintln!("Failed to clear terminal");
                            }
                            continue;
                        }
                        Some(SlashCommand::Permission(argument)) => {
                            match argument {
                                None => {
                                    let current = shared_permission.get();
                                    eprintln!("Current permission level: {}", current);
                                }
                                Some(level) => {
                                    match level.parse::<crate::permission::Permission>() {
                                        Ok(permission) => {
                                            shared_permission.set(permission);
                                            eprintln!("Permission level set to: {}", permission);
                                        }
                                        Err(error) => {
                                            eprintln!("Error: {}", error);
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        Some(SlashCommand::Cd(argument)) => {
                            handle_cd(argument.as_deref().unwrap_or(""));
                            continue;
                        }
                        Some(command @ (SlashCommand::Session | SlashCommand::Compact)) => {
                            if input_sender.send(ShellEvent::Command(command)).is_err() {
                                break;
                            }
                            if agent_done_receiver.recv().is_err() {
                                break;
                            }
                            continue;
                        }
                        None => {
                            eprintln!(
                                "Unknown command: {}. Type /help for available commands.",
                                trimmed
                            );
                            continue;
                        }
                    }
                }

                if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                    if input_sender.send(ShellEvent::Exit).is_err() {
                        tracing::trace!("shell event receiver already dropped");
                    }
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
                            if !exit_status.success()
                                && let Some(code) = exit_status.code()
                            {
                                eprintln!("Command exited with status {}", code);
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
                if input_sender.send(ShellEvent::Exit).is_err() {
                    tracing::trace!("shell event receiver already dropped");
                }
                break;
            }
            Err(error) => {
                tracing::error!("readline error: {}", error);
                if input_sender.send(ShellEvent::Exit).is_err() {
                    tracing::trace!("shell event receiver already dropped");
                }
                break;
            }
        }
    }
}

fn shorten_path_with_tilde(path: &Path) -> String {
    if let Some(home) = dirs::home_dir() {
        if path == home {
            return "~".to_string();
        }
        if let Ok(relative) = path.strip_prefix(&home) {
            return format!("~/{}", relative.display());
        }
    }
    path.display().to_string()
}

fn handle_cd(target: &str) {
    let path = if target.is_empty() || target == "~" {
        match dirs::home_dir() {
            Some(home) => home,
            None => {
                eprintln!("cd: could not determine home directory");
                return;
            }
        }
    } else if let Some(rest) = target.strip_prefix("~/") {
        match dirs::home_dir() {
            Some(home) => home.join(rest),
            None => {
                eprintln!("cd: could not determine home directory");
                return;
            }
        }
    } else {
        PathBuf::from(target)
    };

    if let Err(error) = std::env::set_current_dir(&path) {
        eprintln!("cd: {}: {}", path.display(), error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_slash_command_exit() {
        assert!(matches!(
            parse_slash_command("/exit"),
            Some(SlashCommand::Exit)
        ));
        assert!(matches!(
            parse_slash_command("/quit"),
            Some(SlashCommand::Exit)
        ));
    }

    #[test]
    fn test_parse_slash_command_help() {
        assert!(matches!(
            parse_slash_command("/help"),
            Some(SlashCommand::Help)
        ));
        assert!(matches!(
            parse_slash_command("/?"),
            Some(SlashCommand::Help)
        ));
    }

    #[test]
    fn test_parse_slash_command_clear() {
        assert!(matches!(
            parse_slash_command("/clear"),
            Some(SlashCommand::Clear)
        ));
    }

    #[test]
    fn test_parse_slash_command_session() {
        assert!(matches!(
            parse_slash_command("/session"),
            Some(SlashCommand::Session)
        ));
    }

    #[test]
    fn test_parse_slash_command_permission() {
        assert!(matches!(
            parse_slash_command("/permission"),
            Some(SlashCommand::Permission(None))
        ));
        match parse_slash_command("/permission write") {
            Some(SlashCommand::Permission(Some(arg))) => assert_eq!(arg, "write"),
            _ => panic!("expected Permission with argument"),
        }
    }

    #[test]
    fn test_parse_slash_command_compact() {
        assert!(matches!(
            parse_slash_command("/compact"),
            Some(SlashCommand::Compact)
        ));
    }

    #[test]
    fn test_parse_slash_command_unknown() {
        assert!(parse_slash_command("/unknown").is_none());
    }

    #[test]
    fn test_parse_slash_command_not_slash() {
        assert!(parse_slash_command("hello").is_none());
    }

    #[test]
    fn test_parse_slash_command_empty() {
        assert!(parse_slash_command("/").is_none());
    }

    #[test]
    fn test_parse_slash_command_cd_no_arg() {
        assert!(matches!(
            parse_slash_command("/cd"),
            Some(SlashCommand::Cd(None))
        ));
    }

    #[test]
    fn test_parse_slash_command_cd_with_path() {
        match parse_slash_command("/cd /tmp") {
            Some(SlashCommand::Cd(Some(arg))) => assert_eq!(arg, "/tmp"),
            _ => panic!("expected Cd with argument"),
        }
    }

    #[test]
    fn test_shorten_path_with_tilde_home() {
        if let Some(home) = dirs::home_dir() {
            assert_eq!(shorten_path_with_tilde(&home), "~");
        }
    }

    #[test]
    fn test_shorten_path_with_tilde_subdir() {
        if let Some(home) = dirs::home_dir() {
            let subdir = home.join("projects").join("test");
            assert_eq!(shorten_path_with_tilde(&subdir), "~/projects/test");
        }
    }

    #[test]
    fn test_shorten_path_with_tilde_non_home() {
        let path = std::path::Path::new("/tmp/something");
        assert_eq!(shorten_path_with_tilde(path), "/tmp/something");
    }
}
