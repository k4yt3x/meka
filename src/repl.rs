//! Interactive REPL: reedline-driven prompt loop, slash-command parsing,
//! `!command` shell pass-through, and the channels that exchange events
//! between the REPL thread and the agent loop.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use crossterm::style::{Color, Stylize};
use reedline::{
    EditCommand, Emacs, ExternalPrinter, Highlighter, KeyCode, KeyModifiers, Prompt,
    PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus, Reedline, ReedlineEvent,
    Signal, StyledText, default_emacs_keybindings,
};

use crate::permission::{EnabledPermissions, SharedPermission};
use crate::relay::RELAY;

/// Reedline highlighter that paints the entire input buffer with a single
/// style. The final paint reedline emits on submit is what lands in
/// scrollback, so this is what visually separates user prompts from
/// assistant output.
struct UserInputHighlighter {
    style: nu_ansi_term::Style,
}

impl Highlighter for UserInputHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut text = StyledText::new();
        text.push((self.style, line.to_string()));
        text
    }
}

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
            Cow::Owned(format!("agsh {} ", cwd))
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

fn build_reedline_editor(
    input_style: nu_ansi_term::Style,
    printer: ExternalPrinter<String>,
) -> Reedline {
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

    keybindings.add_binding(
        KeyModifiers::SHIFT,
        KeyCode::Enter,
        ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
    );

    let emacs_mode = Emacs::new(keybindings);
    Reedline::create()
        .with_edit_mode(Box::new(emacs_mode))
        .with_highlighter(Box::new(UserInputHighlighter { style: input_style }))
        .use_bracketed_paste(true)
        .with_external_printer(printer)
}

pub enum SlashCommand {
    Exit,
    Help,
    Clear,
    Session,
    Permission(Option<String>),
    Compact,
    Export,
    Cd(Option<String>),
    /// `/mcp <server>:<prompt> [args...]` — render an MCP prompt and send
    /// its messages as the next user turn.
    McpPrompt {
        server: String,
        prompt: String,
        args: Vec<String>,
    },
    /// `/mcp list` — display configured MCP servers.
    McpList,
    /// `/mcp reconnect <server>` — smoke-test connect for one server.
    McpReconnect {
        server: String,
    },
    /// `/mcp login <server>` — run the OAuth flow from the REPL.
    McpLogin {
        server: String,
    },
    /// `/mcp logout <server>` — clear stored credentials + revoke.
    McpLogout {
        server: String,
    },
    /// `/skill` (no argument) — list installed skills.
    SkillList,
    /// `/skill <name> [extra...]` — invoke a user-invocable skill directly.
    /// Anything the user types after the skill name is captured verbatim
    /// in `extra` and prepended to the rendered skill body before the
    /// agent turn, so the model reads the user's directive first and
    /// the skill body as the method. Empty when the user just typed
    /// `/skill <name>`.
    SkillInvoke {
        name: String,
        extra: String,
    },
}

pub enum ReplEvent {
    UserInput(String),
    Command(SlashCommand),
    Exit,
}

/// Sent from the agent to the REPL when a tool call needs user approval in
/// Ask mode.
pub struct ToolApprovalRequest {
    pub tool_name: String,
    /// Pre-computed summary (first required argument) to show next to the
    /// tool name in the approval prompt. Resolved agent-side because the
    /// REPL thread has no access to the tool registry needed for MCP
    /// schema lookups.
    pub primary_param: Option<String>,
    pub response_sender: std::sync::mpsc::SyncSender<bool>,
}

/// Messages sent from the agent to the REPL thread.
pub enum AgentToReplEvent {
    Done,
    ApprovalRequest(ToolApprovalRequest),
    /// Server-driven elicitation — the REPL prompts the user and replies
    /// via the embedded responder channel.
    McpElicitation(crate::mcp::elicitation::ElicitationPrompt),
    /// Incremental progress update for a running MCP tool.
    McpProgress(crate::mcp::progress::ProgressUpdate),
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
        "export" => Some(SlashCommand::Export),
        "cd" => Some(SlashCommand::Cd(argument)),
        "mcp" => parse_mcp_slash(argument.as_deref().unwrap_or("")),
        "skill" => Some(parse_skill_slash(argument.as_deref().unwrap_or(""))),
        _ => None,
    }
}

/// Parse the argument to `/skill …`.
///
/// - Empty argument (bare `/skill`) → list installed skills. There is
///   no `list` keyword: that token would be treated as a skill name to
///   invoke.
/// - Otherwise: first whitespace-separated token is the skill name;
///   the remainder (if any) is free-form extra context that gets
///   prepended to the skill body before the agent turn. The remainder
///   is trimmed so trailing whitespace doesn't bloat the body.
fn parse_skill_slash(rest: &str) -> SlashCommand {
    let rest = rest.trim();
    if rest.is_empty() {
        return SlashCommand::SkillList;
    }
    let (name, extra) = match rest.split_once(char::is_whitespace) {
        Some((name, extra)) => (name.to_string(), extra.trim().to_string()),
        None => (rest.to_string(), String::new()),
    };
    SlashCommand::SkillInvoke { name, extra }
}

/// Parse the argument to `/mcp …`.
fn parse_mcp_slash(rest: &str) -> Option<SlashCommand> {
    let rest = rest.trim();
    if rest.is_empty() || rest == "list" {
        return Some(SlashCommand::McpList);
    }
    // `<subcommand> <server>` shapes. Reject bare `reconnect` / `login`
    // / `logout` with no server argument so users see the "Unknown
    // command" error instead of silently firing against no target.
    type McpServerCtor = fn(String) -> SlashCommand;
    fn mk_reconnect(s: String) -> SlashCommand {
        SlashCommand::McpReconnect { server: s }
    }
    fn mk_login(s: String) -> SlashCommand {
        SlashCommand::McpLogin { server: s }
    }
    fn mk_logout(s: String) -> SlashCommand {
        SlashCommand::McpLogout { server: s }
    }
    let subcommands: [(&str, McpServerCtor); 3] = [
        ("reconnect ", mk_reconnect),
        ("login ", mk_login),
        ("logout ", mk_logout),
    ];
    for (keyword, ctor) in subcommands {
        if let Some(server) = rest.strip_prefix(keyword) {
            let server = server.trim();
            if server.is_empty() {
                return None;
            }
            return Some(ctor(server.to_string()));
        }
    }
    // `<server>:<prompt> [args...]` — the first token is the prompt spec.
    let mut parts = rest.split_whitespace();
    let spec = parts.next()?;
    let (server, prompt) = spec.split_once(':')?;
    if server.is_empty() || prompt.is_empty() {
        return None;
    }
    let args = parts.map(str::to_string).collect();
    Some(SlashCommand::McpPrompt {
        server: server.to_string(),
        prompt: prompt.to_string(),
        args,
    })
}

fn format_enabled(enabled: EnabledPermissions) -> String {
    enabled
        .iter()
        .map(|mode| mode.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_help() {
    eprintln!("Commands:");
    eprintln!("  /help                          Show this help message");
    eprintln!("  /exit                          Exit the shell");
    eprintln!("  /clear                         Clear the terminal screen");
    eprintln!("  /session                       Show the current session ID");
    eprintln!("  /permission [none|read|ask|write]  Show or set the permission level");
    eprintln!("  /compact                       Summarize and compact the session");
    eprintln!("  /export                        Export the current session as Markdown");
    eprintln!("  /cd <path>                     Change working directory");
    eprintln!("  /skill                         List installed skills");
    eprintln!("  /skill <name> [extra...]       Invoke a skill, optionally with extra context");
    eprintln!("  /mcp                           List configured MCP servers");
    eprintln!("  /mcp reconnect <server>        Reconnect smoke-test for one server");
    eprintln!("  /mcp login <server>            Run the OAuth flow for a server");
    eprintln!("  /mcp logout <server>           Clear stored credentials for a server");
    eprintln!("  /mcp <server>:<prompt> [args]  Render an MCP prompt as the next turn");
    eprintln!();
    eprintln!("Shortcuts:");
    eprintln!("  !<command>    Execute a shell command directly");
    eprintln!("  Shift+Tab     Cycle permission level");
    eprintln!("  Ctrl+D        Exit the shell");
}

pub fn run_repl(
    shared_permission: SharedPermission,
    show_path_in_prompt: bool,
    input_style: nu_ansi_term::Style,
    initial_turn_pending: bool,
    input_sender: tokio::sync::mpsc::UnboundedSender<ReplEvent>,
    agent_event_receiver: std::sync::mpsc::Receiver<AgentToReplEvent>,
) {
    // Install reedline's `ExternalPrinter` on the process-global tracing
    // writer BEFORE the first `read_line()`. From this point on, log
    // lines (including async MCP-connect warnings that fire while the
    // REPL is starting) print *above* the live prompt instead of being
    // overwritten by reedline's redraw.
    let printer = ExternalPrinter::default();
    RELAY.install(printer.clone());

    let mut editor = build_reedline_editor(input_style, printer);
    let prompt = AgshPrompt {
        shared_permission: shared_permission.clone(),
        show_path: show_path_in_prompt,
    };

    // If the caller queued a synthetic first turn (e.g. `--skill` or a bare
    // positional `[PROMPT]` in interactive mode), drain agent events for
    // that turn before drawing the first reedline prompt. Otherwise the
    // prompt indicator and the agent's stdout output collide on screen.
    if initial_turn_pending && !wait_for_agent(&agent_event_receiver) {
        return;
    }

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
                            if input_sender.send(ReplEvent::Exit).is_err() {
                                tracing::trace!("REPL event receiver already dropped");
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
                                            match shared_permission.try_set(permission) {
                                                Ok(()) => {
                                                    eprintln!(
                                                        "Permission level set to: {}",
                                                        permission
                                                    );
                                                }
                                                Err(_) => {
                                                    eprintln!(
                                                        "Error: '{}' is disabled in this config (enabled: {})",
                                                        permission,
                                                        format_enabled(shared_permission.enabled()),
                                                    );
                                                }
                                            }
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
                        Some(
                            command @ (SlashCommand::Session
                            | SlashCommand::Compact
                            | SlashCommand::Export
                            | SlashCommand::McpPrompt { .. }
                            | SlashCommand::McpList
                            | SlashCommand::McpReconnect { .. }
                            | SlashCommand::McpLogin { .. }
                            | SlashCommand::McpLogout { .. }
                            | SlashCommand::SkillList
                            | SlashCommand::SkillInvoke { .. }),
                        ) => {
                            if input_sender.send(ReplEvent::Command(command)).is_err() {
                                break;
                            }
                            if !wait_for_agent(&agent_event_receiver) {
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
                    if input_sender.send(ReplEvent::Exit).is_err() {
                        tracing::trace!("REPL event receiver already dropped");
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
                    .send(ReplEvent::UserInput(trimmed.to_string()))
                    .is_err()
                {
                    break;
                }

                if !wait_for_agent(&agent_event_receiver) {
                    break;
                }
            }
            Ok(Signal::CtrlC) => {
                continue;
            }
            Ok(Signal::CtrlD) => {
                if input_sender.send(ReplEvent::Exit).is_err() {
                    tracing::trace!("REPL event receiver already dropped");
                }
                break;
            }
            // The pinned reedline fork (wtfbbqhax/reedline @ 3a457ff) has
            // a slimmer `Signal` enum than upstream — no `ExternalBreak`
            // variant — so this catch-all is currently unreachable. When
            // we switch back to upstream after #1005 lands in a release,
            // it'll fire on the unhandled `ExternalBreak`.
            #[allow(unreachable_patterns)]
            Ok(other) => {
                tracing::warn!("unexpected reedline signal: {:?}", other);
                if input_sender.send(ReplEvent::Exit).is_err() {
                    tracing::trace!("REPL event receiver already dropped");
                }
                break;
            }
            Err(error) => {
                tracing::error!("readline error: {}", error);
                if input_sender.send(ReplEvent::Exit).is_err() {
                    tracing::trace!("REPL event receiver already dropped");
                }
                break;
            }
        }
    }
}

/// Wait for the agent to signal it is done, while also handling tool approval
/// requests that arrive in Ask mode.
fn wait_for_agent(agent_event_receiver: &std::sync::mpsc::Receiver<AgentToReplEvent>) -> bool {
    loop {
        match agent_event_receiver.recv() {
            Ok(AgentToReplEvent::Done) => return true,
            Ok(AgentToReplEvent::ApprovalRequest(request)) => {
                handle_approval_request(&request);
            }
            Ok(AgentToReplEvent::McpElicitation(prompt)) => {
                handle_elicitation_prompt(prompt);
            }
            Ok(AgentToReplEvent::McpProgress(update)) => {
                render_progress_update(&update);
            }
            Err(_) => return false,
        }
    }
}

/// One-line status overwrite on stderr for a running MCP tool.
fn render_progress_update(update: &crate::mcp::progress::ProgressUpdate) {
    let line = format_progress_update(update);
    eprint!("{}", line);
    use std::io::Write;
    let _ = std::io::stderr().flush();
}

/// Format a progress line. Sanitises server-controlled strings so an MCP
/// server can't inject ANSI escapes to clear the screen or spoof prompts.
fn format_progress_update(update: &crate::mcp::progress::ProgressUpdate) -> String {
    let message = update
        .message
        .as_deref()
        .map(crate::mcp::sanitize::sanitize_text)
        .unwrap_or_default();
    let server = crate::mcp::sanitize::sanitize_text(&update.server_name);
    let tool = crate::mcp::sanitize::sanitize_text(&update.tool_name);
    let body = match update.total {
        Some(total) if total > 0.0 => format!(
            "\r[mcp:{}/{}] {:.0}/{:.0} {}",
            server, tool, update.progress, total, message
        ),
        _ => format!(
            "\r[mcp:{}/{}] {:.0} {}",
            server, tool, update.progress, message
        ),
    };
    // Pad with a few spaces so the next print clears trailing chars from
    // any longer previous line.
    format!("{}     ", body)
}

/// Route a structured/url elicitation request to the user. For forms, walks
/// the JSON Schema one property at a time, collecting input. For URLs, opens
/// the browser and waits for the user to confirm.
fn handle_elicitation_prompt(prompt: crate::mcp::elicitation::ElicitationPrompt) {
    use crate::mcp::elicitation::{ElicitationKind, ElicitationResponse};
    use crate::mcp::sanitize::sanitize_text;
    // Server-controlled strings get stripped of control/format codepoints
    // before they reach the terminal. Without this a malicious server could
    // ship ANSI escapes to clear the screen or RTL overrides to spoof the
    // field the user thinks they're filling in.
    eprintln!(
        "[mcp elicit: {}] {}",
        sanitize_text(&prompt.server_name),
        sanitize_text(&prompt.message)
    );

    let response = match &prompt.kind {
        ElicitationKind::Url { url } => {
            eprint!(
                "Open {} in your browser? [Y/n/s=skip]: ",
                sanitize_text(url)
            );
            use std::io::Write;
            let _ = std::io::stderr().flush();
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).is_err() {
                ElicitationResponse::Decline
            } else {
                match line.trim().to_ascii_lowercase().as_str() {
                    "" | "y" | "yes" => {
                        if let Err(error) = open::that(url) {
                            // URL was printed right above; launch
                            // failure on headless hosts is expected
                            // noise — diagnostic only.
                            tracing::debug!(
                                "failed to open browser for URL elicitation: {}",
                                error
                            );
                        }
                        ElicitationResponse::Accept { content: None }
                    }
                    "s" | "skip" => ElicitationResponse::Cancel,
                    _ => ElicitationResponse::Decline,
                }
            }
        }
        ElicitationKind::Form { schema } => {
            let mut filled = serde_json::Map::new();
            if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
                for (field_name, field_schema) in properties {
                    let description = field_schema
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let ty = field_schema
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("string");
                    let hint = if description.is_empty() {
                        ty
                    } else {
                        description
                    };
                    eprint!(
                        "  {} ({}): ",
                        sanitize_text(field_name),
                        sanitize_text(hint)
                    );
                    use std::io::Write;
                    let _ = std::io::stderr().flush();
                    let mut line = String::new();
                    if std::io::stdin().read_line(&mut line).is_err() {
                        break;
                    }
                    let value = line.trim().to_string();
                    if value.is_empty() {
                        continue;
                    }
                    let parsed = match ty {
                        "boolean" => match value.to_ascii_lowercase().as_str() {
                            "true" | "yes" | "y" => serde_json::Value::Bool(true),
                            "false" | "no" | "n" => serde_json::Value::Bool(false),
                            _ => serde_json::Value::String(value),
                        },
                        "integer" | "number" => value
                            .parse::<f64>()
                            .ok()
                            .and_then(serde_json::Number::from_f64)
                            .map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::String(value)),
                        _ => serde_json::Value::String(value),
                    };
                    filled.insert(field_name.clone(), parsed);
                }
            }
            ElicitationResponse::Accept {
                content: Some(serde_json::Value::Object(filled)),
            }
        }
    };
    let _ = prompt.responder.send(response);
}

fn handle_approval_request(request: &ToolApprovalRequest) {
    use crossterm::style::Stylize;

    let display_name = crate::render::tool_display_name_for_approval(&request.tool_name);
    let summary = request
        .primary_param
        .as_deref()
        .map(|s| s.replace('\n', " "))
        .unwrap_or_default();

    eprint!(
        "{} ",
        format!("[ask] {} {}", display_name, summary).with(crossterm::style::Color::Magenta)
    );
    eprint!("{}", "(Y/n) ".with(crossterm::style::Color::DarkGrey));

    if let Err(error) = std::io::Write::flush(&mut std::io::stderr()) {
        tracing::debug!("failed to flush stderr: {}", error);
    }

    let mut response = String::new();
    let allowed = match std::io::stdin().read_line(&mut response) {
        Ok(_) => {
            let trimmed = response.trim().to_lowercase();
            trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
        }
        Err(_) => false,
    };

    if let Err(error) = request.response_sender.send(allowed) {
        tracing::warn!(
            "failed to send approval response (agent disconnected): {}",
            error
        );
    }
}

fn shorten_path_with_tilde(path: &Path) -> String {
    if let Some(home) = dirs::home_dir() {
        if path == home {
            return "~".to_string();
        }
        if let Ok(relative) = path.strip_prefix(&home) {
            // Normalize to forward slashes so the tilde form reads the
            // same way on every platform (Windows' native `\` looks
            // jarring next to the `~/` prefix and breaks tests that
            // compare against a hard-coded literal).
            let relative_str = relative.display().to_string().replace('\\', "/");
            return format!("~/{}", relative_str);
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
    fn test_user_input_highlighter_default_preset_preserves_literal() {
        let highlighter = UserInputHighlighter {
            style: crate::config::default_input_style(),
        };
        let rendered = highlighter.highlight("hello world", 5).render_simple();
        assert!(
            rendered.contains("hello world"),
            "literal input must survive: {rendered:?}"
        );
        assert!(
            rendered.contains("\x1b[") && rendered.contains('m'),
            "at least one SGR escape must be emitted: {rendered:?}"
        );
    }

    #[test]
    fn test_user_input_highlighter_none_emits_no_escape() {
        let highlighter = UserInputHighlighter {
            style: nu_ansi_term::Style::default(),
        };
        let rendered = highlighter.highlight("hello", 0).render_simple();
        assert_eq!(rendered, "hello");
    }

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
    fn test_parse_slash_command_export() {
        assert!(matches!(
            parse_slash_command("/export"),
            Some(SlashCommand::Export)
        ));
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

    #[test]
    fn test_format_progress_update_strips_ansi_escapes() {
        let update = crate::mcp::progress::ProgressUpdate {
            server_name: "svr".to_string(),
            tool_name: "tool".to_string(),
            tool_use_id: None,
            message: Some("\x1b[2Jspoofed\x1b[H".to_string()),
            progress: 1.0,
            total: Some(4.0),
        };
        let line = format_progress_update(&update);
        assert!(
            !line.contains('\x1b'),
            "ANSI escape leaked into progress line: {:?}",
            line
        );
        assert!(line.contains("spoofed"));
        assert!(line.contains("[mcp:svr/tool]"));
    }

    #[test]
    fn test_parse_mcp_slash_empty_is_list() {
        assert!(matches!(
            parse_slash_command("/mcp"),
            Some(SlashCommand::McpList)
        ));
    }

    #[test]
    fn test_parse_mcp_slash_explicit_list() {
        assert!(matches!(
            parse_slash_command("/mcp list"),
            Some(SlashCommand::McpList)
        ));
    }

    #[test]
    fn test_parse_mcp_slash_reconnect_with_server() {
        match parse_slash_command("/mcp reconnect postgres") {
            Some(SlashCommand::McpReconnect { server }) => assert_eq!(server, "postgres"),
            other => panic!("expected McpReconnect, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_reconnect_without_server_is_none() {
        // Bare `reconnect` with no server name: neither the reconnect arm nor
        // the `<server>:<prompt>` arm matches, so the command is rejected
        // rather than silently firing against some default.
        assert!(parse_slash_command("/mcp reconnect").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_login_with_server() {
        match parse_slash_command("/mcp login notion") {
            Some(SlashCommand::McpLogin { server }) => assert_eq!(server, "notion"),
            other => panic!("expected McpLogin, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_logout_with_server() {
        match parse_slash_command("/mcp logout notion") {
            Some(SlashCommand::McpLogout { server }) => assert_eq!(server, "notion"),
            other => panic!("expected McpLogout, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_login_without_server_is_none() {
        assert!(parse_slash_command("/mcp login").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_logout_without_server_is_none() {
        assert!(parse_slash_command("/mcp logout").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_login_trims_whitespace() {
        match parse_slash_command("/mcp login   notion  ") {
            Some(SlashCommand::McpLogin { server }) => assert_eq!(server, "notion"),
            other => panic!("expected McpLogin, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_prompt_no_args() {
        match parse_slash_command("/mcp postgres:schema") {
            Some(SlashCommand::McpPrompt {
                server,
                prompt,
                args,
            }) => {
                assert_eq!(server, "postgres");
                assert_eq!(prompt, "schema");
                assert!(args.is_empty());
            }
            other => panic!("expected McpPrompt, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_prompt_with_args() {
        match parse_slash_command("/mcp pg:query table=users limit=10") {
            Some(SlashCommand::McpPrompt {
                server,
                prompt,
                args,
            }) => {
                assert_eq!(server, "pg");
                assert_eq!(prompt, "query");
                assert_eq!(args, vec!["table=users", "limit=10"]);
            }
            other => panic!("expected McpPrompt, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_mcp_slash_empty_server_rejected() {
        assert!(parse_slash_command("/mcp :prompt").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_empty_prompt_rejected() {
        assert!(parse_slash_command("/mcp server:").is_none());
    }

    #[test]
    fn test_parse_mcp_slash_multiple_colons_splits_on_first() {
        // `split_once` returns the first colon, so prompt names can contain
        // further colons.
        match parse_slash_command("/mcp srv:ns:prompt") {
            Some(SlashCommand::McpPrompt { server, prompt, .. }) => {
                assert_eq!(server, "srv");
                assert_eq!(prompt, "ns:prompt");
            }
            other => panic!("expected McpPrompt, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_skill_slash_empty_is_list() {
        assert!(matches!(
            parse_slash_command("/skill"),
            Some(SlashCommand::SkillList)
        ));
        // Trailing whitespace is treated as no argument.
        assert!(matches!(
            parse_slash_command("/skill   "),
            Some(SlashCommand::SkillList)
        ));
    }

    #[test]
    fn test_parse_skill_slash_invokes_named_skill() {
        match parse_slash_command("/skill demo") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "demo");
                assert!(extra.is_empty());
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_skill_slash_captures_free_form_extra() {
        // The whole remainder after the skill name is captured verbatim
        // (preserving inner whitespace) and trimmed at the edges. This is
        // free-form text the user wants prepended to the skill body — no
        // positional argument parsing.
        match parse_slash_command("/skill demo only fetch UK news") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "demo");
                assert_eq!(extra, "only fetch UK news");
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_skill_slash_trims_trailing_whitespace() {
        // Trailing whitespace after the skill name should produce an
        // empty extra, not a whitespace-padded one — equivalent to the
        // bare-name invocation.
        match parse_slash_command("/skill demo   ") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "demo");
                assert!(extra.is_empty());
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn test_parse_skill_slash_no_list_keyword() {
        // The token "list" is treated as a skill name, not a subcommand.
        // (Bare `/skill` is the listing form; `/skill list` would error
        // at dispatch with "unknown skill 'list'" if no such skill exists.)
        match parse_slash_command("/skill list") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "list");
                assert!(extra.is_empty());
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    /// Short debug label — SlashCommand doesn't implement Debug so we map
    /// the few variants we care about manually to keep assertion messages
    /// readable.
    fn option_label(cmd: &Option<SlashCommand>) -> &'static str {
        match cmd {
            None => "None",
            Some(SlashCommand::Exit) => "Exit",
            Some(SlashCommand::Help) => "Help",
            Some(SlashCommand::Clear) => "Clear",
            Some(SlashCommand::Session) => "Session",
            Some(SlashCommand::Permission(_)) => "Permission",
            Some(SlashCommand::Compact) => "Compact",
            Some(SlashCommand::Export) => "Export",
            Some(SlashCommand::Cd(_)) => "Cd",
            Some(SlashCommand::McpList) => "McpList",
            Some(SlashCommand::McpReconnect { .. }) => "McpReconnect",
            Some(SlashCommand::McpLogin { .. }) => "McpLogin",
            Some(SlashCommand::McpLogout { .. }) => "McpLogout",
            Some(SlashCommand::McpPrompt { .. }) => "McpPrompt",
            Some(SlashCommand::SkillList) => "SkillList",
            Some(SlashCommand::SkillInvoke { .. }) => "SkillInvoke",
        }
    }

    #[test]
    fn test_format_progress_update_strips_rtl_override_in_names() {
        // Defensive: even though server/tool names are normalised at
        // registration time, this confirms the renderer can't be tricked
        // by a handler that someday forgets to normalise.
        let update = crate::mcp::progress::ProgressUpdate {
            server_name: "sv\u{202E}r".to_string(),
            tool_name: "t\u{200B}ool".to_string(),
            tool_use_id: None,
            message: None,
            progress: 0.5,
            total: None,
        };
        let line = format_progress_update(&update);
        assert!(!line.contains('\u{202E}'));
        assert!(!line.contains('\u{200B}'));
    }
}
