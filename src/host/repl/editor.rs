//! Interactive REPL: reedline-driven prompt loop, slash-command parsing, `!command` shell
//! pass-through, and the channels that exchange events between the REPL thread and the agent loop.

use std::{
    borrow::Cow,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crossterm::style::Stylize;
use reedline::{
    ColumnarMenu, Completer, CompletionResult, EditCommand, Emacs, ExternalPrinter, Highlighter,
    History, KeyCode, KeyModifiers, MenuBuilder, Prompt, PromptEditMode, PromptHistorySearch,
    PromptHistorySearchStatus, Reedline, ReedlineEvent, ReedlineMenu, Signal, Span, StyledText,
    Suggestion, default_emacs_keybindings,
};

use super::{commands::*, prompts::*};
use crate::{permission::SharedPermission, relay::RELAY};

/// Foreground applied to the leading token of a recognized slash command.
pub(super) const KNOWN_COLOR: nu_ansi_term::Color = nu_ansi_term::Color::Green;
/// Foreground applied to the leading token when it starts with `/` but is not a known command.
pub(super) const UNKNOWN_COLOR: nu_ansi_term::Color = nu_ansi_term::Color::Red;

/// Reedline highlighter for the input buffer, painting two things on two different schedules.
///
/// The leading `/command` token is recolored on every keystroke to signal whether it is recognized,
/// because that answers "is this command real" while there is still time to fix the spelling.
///
/// The base style waits for submission. What it marks is the seam between a finished prompt and the
/// reply printed underneath, which is a question about scrollback rather than about the line in
/// hand, so a line still being edited keeps the terminal's own colors. Reedline repaints once more
/// on its way out of `read_line`, and that paint is the one that stays on screen.
pub(super) struct UserInputHighlighter {
    pub(super) style: nu_ansi_term::Style,
    /// Raised by [`SubmitWatcher`] the instant reedline commits to submitting, and lowered by the
    /// prompt loop once `read_line` has returned.
    pub(super) submitted: Arc<AtomicBool>,
}

impl Highlighter for UserInputHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let base = if self.submitted.load(Ordering::Relaxed) {
            self.style
        } else {
            nu_ansi_term::Style::new()
        };
        let mut text = StyledText::new();
        if let Some(after_slash) = line.strip_prefix('/') {
            let word_len = after_slash
                .find(char::is_whitespace)
                .unwrap_or(after_slash.len());
            let word = &after_slash[..word_len];
            let (token, remainder) = line.split_at(word_len + 1);
            let known = crate::host::COMMANDS
                .iter()
                .any(|command| command.name == word || command.aliases.contains(&word));
            let token_color = if known { KNOWN_COLOR } else { UNKNOWN_COLOR };
            text.push((base.fg(token_color), token.to_string()));
            if !remainder.is_empty() {
                text.push((base, remainder.to_string()));
            }
        } else {
            text.push((base, line.to_string()));
        }
        text
    }
}

/// Tells [`UserInputHighlighter`] that the paint about to happen is the one that stays on screen.
///
/// Reedline consults a validator once per submit attempt, from the `Enter` arm, after it has ruled
/// out an open completion menu and immediately before `submit_buffer` repaints. That makes it the
/// only place in reedline's API that reports the *decision* to submit rather than a keystroke that
/// might have caused one, and the difference is most of the cases: Enter with a menu open accepts
/// the completion, Enter during a Ctrl+R search recalls the match into the buffer, and Alt+Enter
/// and Shift+Enter open a second line. Watching the key raises the flag on every one of those and
/// then leaves it raised for the rest of a line that is still being edited.
///
/// Always `Complete`, which is the arm reedline takes when no validator is installed at all. This
/// one is here for the notification, not to hold a line back.
pub(super) struct SubmitWatcher {
    pub(super) submitted: Arc<AtomicBool>,
}

impl reedline::Validator for SubmitWatcher {
    fn validate(&self, _line: &str) -> reedline::ValidationResult {
        self.submitted.store(true, Ordering::Relaxed);
        reedline::ValidationResult::Complete
    }
}

/// The highlighter and the validator that releases it, over one shared cell.
///
/// Built as a pair because the pair is the whole mechanism: two independently constructed flags
/// type-check, wire up, and silently never paint anything.
pub(super) fn submit_aware_input_painter(
    style: nu_ansi_term::Style,
    submitted: Arc<AtomicBool>,
) -> (UserInputHighlighter, SubmitWatcher) {
    (
        UserInputHighlighter {
            style,
            submitted: Arc::clone(&submitted),
        },
        SubmitWatcher { submitted },
    )
}

/// Tab completer for slash commands. The data needed to complete arguments (MCP server names,
/// skill names) is snapshotted rather than gathered here, because reedline re-invokes `complete()`
/// on every keystroke while the menu is open, so a per-keystroke filesystem scan like the skill
/// walk (which reads every `SKILL.md`) must never live in the hot path.
pub(super) struct SlashCompleter {
    pub(super) mcp_server_names: Vec<String>,
    /// Refreshed once per prompt by the loop below rather than frozen at construction. With
    /// `[skills] agent_managed`, `skill_write` and `skill_delete` change the set mid-session; a
    /// frozen list went on offering a skill the agent had deleted, and `/skill <name>` then failed
    /// on a name Tab had just supplied.
    pub(super) skill_names: Arc<std::sync::RwLock<Vec<String>>>,
    pub(super) profile_names: Vec<String>,
    pub(super) cwd: crate::workspace::SharedCwd,
}

/// `/mcp` first-argument keywords, mirroring the grammar of `parse_mcp_slash`.
pub(super) const MCP_SUBCOMMANDS: [&str; 4] = ["list", "reconnect", "login", "logout"];

/// Permission levels in canonical order, sourced through the `Display` impl so the completions
/// cannot drift from what the parser accepts.
pub(super) const PERMISSION_LEVELS: [crate::permission::Permission; 4] = [
    crate::permission::Permission::None,
    crate::permission::Permission::Read,
    crate::permission::Permission::Workspace,
    crate::permission::Permission::Unrestricted,
];

impl Completer for SlashCompleter {
    /// Slash-command completion is computed synchronously from in-memory snapshots, so every result
    /// is authoritative the moment it is produced. reedline's `Stale` / `Pending` variants exist
    /// for completers that compute off-thread; this one never has a partial answer to hand
    /// back.
    fn complete(&mut self, line: &str, pos: usize) -> CompletionResult {
        CompletionResult::fresh(self.suggestions(line, pos))
    }
}

impl SlashCompleter {
    /// The completion logic proper, returning a plain `Vec` so callers (and tests) work with the
    /// suggestions directly instead of destructuring [`CompletionResult`].
    pub(super) fn suggestions(&self, line: &str, pos: usize) -> Vec<Suggestion> {
        let Some(after_slash) = line.strip_prefix('/') else {
            return Vec::new();
        };
        let before_cursor = line.get(..pos).unwrap_or(line);

        if !before_cursor.contains(char::is_whitespace) {
            // Cursor is still in the command word: complete command names. Aliases are
            // intentionally not prefix-matched, since offering both `/exit` and `/quit`
            // would just be noise.
            let typed = line.get(1..pos).unwrap_or("");
            return crate::host::COMMANDS
                .iter()
                .filter(|command| command.name.starts_with(typed))
                .map(|command| Suggestion {
                    value: format!("/{}", command.name),
                    description: Some(command.help.to_string()),
                    append_whitespace: !command.arg_hint.is_empty(),
                    span: Span::new(0, pos),
                    ..Suggestion::default()
                })
                .collect();
        }

        let command = after_slash.split_whitespace().next().unwrap_or("");
        let token_start = before_cursor
            .char_indices()
            .rev()
            .find(|(_, character)| character.is_whitespace())
            .map_or(0, |(index, character)| index + character.len_utf8());
        let prefix = line.get(token_start..pos).unwrap_or("");
        // The command word is token 0, so the first argument is token 1.
        let argument_index = before_cursor
            .get(..token_start)
            .unwrap_or("")
            .split_whitespace()
            .count();

        match command {
            "permission" if argument_index == 1 => terminal_suggestions(
                PERMISSION_LEVELS.iter().map(|level| level.to_string()),
                prefix,
                token_start,
                pos,
            ),
            "approvals" if argument_index == 1 => terminal_suggestions(
                ["on", "off"].iter().map(|state| state.to_string()),
                prefix,
                token_start,
                pos,
            ),
            "profile" if argument_index == 1 => {
                terminal_suggestions(self.profile_names.iter().cloned(), prefix, token_start, pos)
            }
            "skill" if argument_index == 1 => {
                let names = crate::sync::read(&self.skill_names).clone();
                terminal_suggestions(names, prefix, token_start, pos)
            }
            "mcp" if argument_index == 1 => terminal_suggestions(
                MCP_SUBCOMMANDS.iter().map(|keyword| keyword.to_string()),
                prefix,
                token_start,
                pos,
            ),
            "mcp" if argument_index == 2 => {
                let subcommand = before_cursor
                    .get(..token_start)
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("");
                if matches!(subcommand, "reconnect" | "login" | "logout") {
                    terminal_suggestions(
                        self.mcp_server_names.iter().cloned(),
                        prefix,
                        token_start,
                        pos,
                    )
                } else {
                    Vec::new()
                }
            }
            "cd" => complete_cd_path(&self.cwd, prefix, token_start, pos),
            _ => Vec::new(),
        }
    }
}

/// Build suggestions for a terminal (single-token) argument, prefix-filtered. A trailing space is
/// appended so the user can move on once a value is chosen.
pub(super) fn terminal_suggestions(
    candidates: impl IntoIterator<Item = String>,
    prefix: &str,
    token_start: usize,
    pos: usize,
) -> Vec<Suggestion> {
    candidates
        .into_iter()
        .filter(|candidate| candidate.starts_with(prefix))
        .map(|candidate| Suggestion {
            value: candidate,
            append_whitespace: true,
            span: Span::new(token_start, pos),
            ..Suggestion::default()
        })
        .collect()
}

/// Complete a `/cd` argument token to matching subdirectories. Only directories are offered (`/cd`
/// rejects files), and each value ends in `/` so Tab can keep drilling into nested directories.
pub(super) fn complete_cd_path(
    cwd: &crate::workspace::SharedCwd,
    token: &str,
    token_start: usize,
    pos: usize,
) -> Vec<Suggestion> {
    let (parent_portion, partial) = match token.rfind('/') {
        Some(index) => (&token[..=index], &token[index + 1..]),
        None => ("", token),
    };

    let scan_dir = if parent_portion.is_empty() {
        cwd.get()
    } else {
        // `expand_user_path` rather than `expand_cd_target`: this branch only runs for a non-empty
        // portion, so the bare-`/cd` default has nothing to say about it, and reaching for the
        // `/cd`-specific door would mean handing the completer a launch directory it never uses.
        let Some(expanded) = crate::paths::expand_user_path(parent_portion) else {
            return Vec::new();
        };
        crate::workspace::resolve_against_cwd(cwd, expanded)
    };

    let entries = match std::fs::read_dir(&scan_dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };

    let mut suggestions = Vec::new();
    for entry in entries.flatten() {
        if !entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        // Hide dotfiles unless the user has started typing a dot, mirroring shell completion.
        if name.starts_with('.') && !partial.starts_with('.') {
            continue;
        }
        if !name.starts_with(partial) {
            continue;
        }
        suggestions.push(Suggestion {
            value: format!("{parent_portion}{name}/"),
            append_whitespace: false,
            span: Span::new(token_start, pos),
            ..Suggestion::default()
        });
    }
    suggestions
}

pub(super) const COMPLETION_MENU: &str = "completion_menu";

pub(super) struct MekaPrompt {
    pub(super) shared_permission: SharedPermission,
    pub(super) show_path: bool,
    /// Per-session working directory shared with the agent and the `/cd` slash command. Reading
    /// the lock per prompt render is cheap (microseconds) and bounded; `/cd` is the only
    /// writer.
    pub(super) cwd: crate::workspace::SharedCwd,
    /// Live context-window gauge, present only when `display.show_context_in_prompt` is set.
    pub(super) context: Option<ContextIndicator>,
}

/// Shared handle to the live context-token counter plus the model window, for the optional prompt
/// gauge. The counter is the agent's `last_context_tokens` (updated after each turn / on compact).
///
/// Both are handles. A `u64` read from the process default profile before the agent exists leaves a
/// session resumed onto another profile, or moved by `/profile`, dividing by a window it is not
/// gauged against, so the prompt and `/status` disagree.
pub(super) struct ContextIndicator {
    pub(super) tokens: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pub(super) window: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl ContextIndicator {
    /// Format as `used/window pct%`, or `None` before the first turn (no measurement yet) or when
    /// the window is unknown.
    pub(super) fn render(&self) -> Option<String> {
        let tokens = self.tokens.load(std::sync::atomic::Ordering::Relaxed);
        let window = self.window.load(std::sync::atomic::Ordering::Relaxed);
        if tokens == 0 || window == 0 {
            return None;
        }
        let pct = ((tokens as f64 / window as f64) * 100.0).round() as u64;
        Some(format!(
            "{}/{} {}%",
            crate::text::format_token_count(tokens),
            crate::text::format_token_count(window),
            pct
        ))
    }
}

impl Prompt for MekaPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        let mut left = if self.show_path {
            let path = self.cwd.get();
            format!("meka {} ", shorten_path_with_tilde(&path))
        } else {
            "meka ".to_string()
        };
        if let Some(gauge) = self.context.as_ref().and_then(ContextIndicator::render) {
            left.push_str(&gauge);
            left.push(' ');
        }
        Cow::Owned(left)
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> Cow<'_, str> {
        let permission = self.shared_permission.get();
        let colored_indicator =
            format!("[{}]", permission.indicator()).with(permission.indicator_color());
        Cow::Owned(format!("{colored_indicator} > "))
    }

    fn get_prompt_color(&self) -> nu_ansi_term::Color {
        nu_ansi_term::Color::White
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

    fn get_indicator_color(&self) -> nu_ansi_term::Color {
        // `nu_ansi_term::Color` has no `Reset`; `Default` is its equivalent, leaving the
        // indicator's own crossterm styling (set in `render_prompt_indicator`) untouched.
        nu_ansi_term::Color::Default
    }
}

/// Emacs bindings plus one key that reedline has no vocabulary for: cycling meka's permission.
///
/// Wraps rather than replaces, so every other binding is stock reedline and stays that way when
/// reedline changes.
///
/// The point of intercepting here rather than binding Shift+Tab to `ExecuteHostCommand` is that
/// cycling is not a host command. That signal means "the editor is exiting, the host is about to
/// run something and may scroll the terminal", and reedline reasonably refuses to re-use a prompt
/// row after it -- `select_prompt_row` gives up whenever the suspended prompt sat flush against the
/// bottom of the screen (nushell/reedline#1130). Cycling runs nothing and paints nothing, so
/// `read_line` never needs to return: `parse_event` takes `&mut self`, which is the supported way
/// for a host to react to a key, and `Repaint` re-renders the prompt in place from the permission
/// cell it just moved. Every earlier attempt at this fought that mismatch from the outside, first
/// stacking a prompt line per press and then flashing when the line was cleared to stop the
/// stacking.
pub(super) struct CyclePermissionMode {
    pub(super) inner: Emacs,
    pub(super) shared_permission: SharedPermission,
    /// Best-effort: a closed channel means the agent loop is gone, which is the one case where
    /// nothing is left to act on the recorded level.
    pub(super) input_sender: tokio::sync::mpsc::UnboundedSender<ReplEvent>,
    pub(super) sandbox_state: crate::sandbox::SandboxState,
}

impl reedline::EditMode for CyclePermissionMode {
    fn parse_event(&mut self, event: reedline::ReedlineRawEvent) -> ReedlineEvent {
        let raw: crossterm::event::Event = event.into();
        if matches!(
            raw,
            crossterm::event::Event::Key(crossterm::event::KeyEvent {
                code: KeyCode::BackTab,
                ..
            })
        ) {
            let new_permission = self.shared_permission.cycle();
            tracing::debug!("permission cycled to {new_permission}");
            // Recorded on the session row, not just in this process's cell. A scheduled gate is
            // re-checked at fire time against the row, and a row that carries no level falls back
            // to the *polling process's* startup flag -- so a `meka serve` sharing the data
            // directory kept firing a gate the user had just withdrawn here.
            if self
                .input_sender
                .send(ReplEvent::PermissionChanged(new_permission))
                .is_err()
            {
                warn_unrecorded("the permission level");
            }
            // Re-emit the "backend unavailable" warn at the moment the user enters the read level,
            // so a misconfigured sandbox surfaces immediately instead of waiting for
            // the first `execute_command` failure. The "stronger sandbox available"
            // nudge (Warn 2) intentionally does not fire here: startup-only, to avoid
            // nagging.
            //
            // Reached while `read_line` is still running, so the relay routes this through the
            // `ExternalPrinter` and it lands cleanly above the live prompt rather than in the gap
            // between two of them.
            if new_permission == crate::permission::Permission::Read {
                crate::sandbox::warn_if_sandbox_issues(
                    &self.sandbox_state,
                    crate::sandbox::WarnContext::ReadModeEntry,
                );
            }
            return ReedlineEvent::Repaint;
        }
        // Rebuilt rather than cloned: `ReedlineRawEvent` is consumed by the conversion above, and
        // its `TryFrom` is the only constructor. It rejects a key *release*, which cannot appear
        // here because this event already passed that same filter on the way in.
        match reedline::ReedlineRawEvent::try_from(raw) {
            Ok(event) => self.inner.parse_event(event),
            Err(()) => ReedlineEvent::None,
        }
    }

    fn edit_mode(&self) -> reedline::PromptEditMode {
        self.inner.edit_mode()
    }

    // `handle_mode_specific_event` is deliberately not forwarded: it exists for vi's mode changes,
    // `Emacs` leaves it at the trait's `Inapplicable` default, and `EventStatus` is not exported
    // from reedline's root so it cannot be named here anyway.
}

/// Emacs defaults plus meka's own. Shift+Tab is deliberately absent: [`CyclePermissionMode`]
/// answers that key before the bindings are consulted, and a binding here would only be dead
/// weight that a reader has to reconcile with the interception.
pub(super) fn meka_keybindings() -> reedline::Keybindings {
    let mut keybindings = default_emacs_keybindings();

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

    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu(COMPLETION_MENU.to_string()),
            ReedlineEvent::MenuNext,
        ]),
    );

    keybindings
}

/// The editor, given the edit mode it should drive.
///
/// The mode arrives assembled rather than as the three values needed to build one, so this stays a
/// function about reedline wiring and knows nothing about permissions.
pub(super) fn build_reedline_editor(
    input_style: nu_ansi_term::Style,
    printer: ExternalPrinter<String>,
    history: Option<Box<dyn History>>,
    completer: SlashCompleter,
    wake: Arc<AtomicBool>,
    edit_mode: CyclePermissionMode,
    submitted: Arc<AtomicBool>,
) -> Reedline {
    let (highlighter, submit_watcher) = submit_aware_input_painter(input_style, submitted);
    let mut editor = Reedline::create()
        .with_edit_mode(Box::new(edit_mode))
        .with_highlighter(Box::new(highlighter))
        .with_validator(Box::new(submit_watcher))
        .with_completer(Box::new(completer))
        .with_menu(ReedlineMenu::EngineCompleter(Box::new(
            ColumnarMenu::default().with_name(COMPLETION_MENU),
        )))
        .use_bracketed_paste(true)
        // Lets the scheduler interrupt an idle prompt. reedline polls this inside `read_line` and
        // returns `Signal::ExternalBreak` with the current buffer, resetting the flag itself.
        .with_break_signal(wake)
        .with_external_printer(printer);
    if let Some(history) = history {
        editor = editor.with_history(history);
    }
    editor
}

pub(crate) enum ReplEvent {
    UserInput(String),
    Command(SlashCommand),
    /// Something out-of-band wants a turn: a scheduled job came due, or a background task has an
    /// outcome to report. Carries no payload; the agent side re-reads both, because between the
    /// watcher noticing and this arriving the job could have been canceled, and firing a prompt
    /// the user just canceled is worse than missing it.
    Wake,
    /// The user moved this session's permission level.
    ///
    /// Sent so the agent side can record it on the session row. The REPL thread is not async and
    /// holds no `Store`, so it cannot write it itself; and the level has to reach the
    /// database *without* waiting for a turn, because the whole point of the withdrawal is that
    /// Shift+Tab-ing down and walking away stops a gate.
    PermissionChanged(crate::permission::Permission),
    /// The user toggled whether calls above the level are submitted for approval.
    ///
    /// Sent for the same reason as `PermissionChanged`: the row is what a resume brings back.
    ApprovalsChanged(bool),
    /// The user asked to move this session onto another profile.
    ///
    /// Sent rather than done here for the same reason as `PermissionChanged`: the REPL thread is
    /// not async and holds neither the store nor the provider registry. The agent side
    /// resolves the name, rebuilds the provider and records it on the row.
    ProfileChange(String),
    /// The user moved this session's working directory.
    ///
    /// Sent for the same reason as `PermissionChanged`, and it matters for the same reason: the
    /// recorded directory is where the *next* resume opens, and it is where a scheduled tool-gate
    /// is re-checked. A `/cd` that never reached the row left both answering with the directory the
    /// session was created in, so a gate stopped matching the command the model had just watched
    /// succeed.
    ///
    /// Carries the canonical path [`handle_cd`] stored rather than a signal to re-read the cell,
    /// so what lands on the row is exactly what the user was shown.
    CwdChanged(PathBuf),
    Exit,
}

/// Sent from the agent to the REPL when a tool call needs user approval.
pub(crate) struct ToolApprovalRequest {
    pub(crate) tool_name: String,
    /// Every argument the call was made with, rendered in full by the prompt. See
    /// [`crate::frontend::PermissionRequest::input`] for why the primary param alone is not enough
    /// to authorize a call.
    pub(crate) input: serde_json::Value,
    pub(crate) response_sender: tokio::sync::oneshot::Sender<super::prompts::ApprovalDecision>,
}

/// Messages sent from the agent to the REPL thread.
pub(crate) enum AgentToReplEvent {
    Done,
    ApprovalRequest(ToolApprovalRequest),
    /// Server-driven elicitation: the REPL prompts the user, then sends the response back via the
    /// embedded oneshot. `ReplFrontend::handle_elicitation` is the producer; the await on the
    /// matching receiver carries the response into the agent's task.
    McpElicitation {
        prompt: crate::frontend::ElicitationPrompt,
        responder: tokio::sync::oneshot::Sender<crate::frontend::ElicitationResponse>,
    },
    /// Incremental progress update for a running MCP tool.
    McpProgress(crate::frontend::ProgressUpdate),
}

/// Borrow the shared console for one synchronous run of writes.
///
/// A poisoned lock is recovered from rather than propagated: the console holds spacing state, and
/// losing the terminal's layout is a worse outcome than continuing from a state one panicking
/// writer may have left mid-transition.
pub(super) fn with_console<T>(
    console: &Mutex<crate::console::Console>,
    act: impl FnOnce(&mut crate::console::Console) -> T,
) -> T {
    let mut guard = crate::sync::lock(console);
    act(&mut guard)
}

/// What reedline left on the row it was drawing the prompt on.
///
/// It writes a CRLF on its way out of `read_line`, but only when it is genuinely exiting: the guard
/// is `suspended_state.is_none()`, and the external-break path sets `suspended_state` precisely
/// because the host is expected to print and come back. So a scheduler wake returns with the drawn
/// prompt still on the row and the cursor at the end of it, and every other signal returns at
/// column zero.
pub(super) fn row_after(signal: &Result<Signal, std::io::Error>) -> crate::console::RowState {
    match signal {
        Ok(Signal::ExternalBreak(_)) => crate::console::RowState::PromptParked,
        _ => crate::console::RowState::Empty,
    }
}

/// Everything the editor thread is launched with. Built by [`crate::host::repl`] from the
/// session it has just assembled; the editor owns nothing here that the agent side does not also
/// hold a handle to.
pub(crate) struct ReplLaunch {
    pub(crate) shared_permission: SharedPermission,
    pub(crate) show_path_in_prompt: bool,
    pub(crate) context_indicator: Option<(
        std::sync::Arc<std::sync::atomic::AtomicU64>,
        std::sync::Arc<std::sync::atomic::AtomicU64>,
    )>,
    pub(crate) input_style: nu_ansi_term::Style,
    pub(crate) initial_turn_pending: bool,
    pub(crate) sandbox_state: crate::sandbox::SandboxState,
    pub(crate) input_sender: tokio::sync::mpsc::UnboundedSender<ReplEvent>,
    pub(crate) agent_event_receiver: std::sync::mpsc::Receiver<AgentToReplEvent>,
    pub(crate) cwd: crate::workspace::SharedCwd,
    /// Where meka was started, which is what a bare `/cd` returns to. Distinct from `cwd`'s
    /// initial value because a resumed session opens in the directory it recorded, so the two
    /// differ from the first prompt whenever `meka -c` is run from somewhere else.
    pub(crate) launch_cwd: PathBuf,
    pub(crate) mcp_server_names: Vec<String>,
    /// Every root `[skills] extra_paths` resolves to, so `/skill ` completes an external skill as
    /// well as a native one. Execution already honors them, so without this the completer is the
    /// only surface that pretends they are not installed.
    pub(crate) skill_roots: Vec<PathBuf>,
    pub(crate) history_db_path: Option<PathBuf>,
    /// `wake` is set by the scheduler watcher when one of this session's jobs is due. reedline
    /// polls it inside `read_line` and returns `Signal::ExternalBreak`, which is what lets a
    /// wakeup interrupt an idle prompt instead of waiting for the user to press Enter.
    pub(crate) wake: Arc<AtomicBool>,
    /// Which profile this session runs on, and every profile configured. The first is shared
    /// because the agent side changes it; the second is a snapshot because `config.toml` is
    /// read once.
    pub(crate) current_profile: Arc<std::sync::RwLock<String>>,
    /// Every configured profile as `(name, backend)`, in name order. The backend is what makes the
    /// listing worth reading once a user has more than a handful: the names are theirs and say
    /// nothing about which wire protocol each one speaks.
    pub(crate) configured_profiles: Vec<crate::config::ProfileSummary>,
    /// Everything printed between two prompts, whichever side printed it. Shared with the agent's
    /// frontend rather than duplicated, because the blanks that bracket an episode are decided by
    /// what the *episode* did and not by which thread happened to answer.
    pub(crate) console: Arc<Mutex<crate::console::Console>>,
}

pub(crate) fn run_repl(launch: ReplLaunch) {
    let ReplLaunch {
        shared_permission,
        show_path_in_prompt,
        context_indicator,
        input_style,
        initial_turn_pending,
        sandbox_state,
        input_sender,
        agent_event_receiver,
        cwd,
        launch_cwd,
        mcp_server_names,
        skill_roots,
        history_db_path,
        wake,
        current_profile,
        configured_profiles,
        console,
    } = launch;
    // Install reedline's `ExternalPrinter` on the process-global tracing writer BEFORE the first
    // `read_line()`. From this point on, log lines (including async MCP-connect warnings that fire
    // while the REPL is starting) print *above* the live prompt instead of being overwritten by
    // reedline's redraw.
    let printer = ExternalPrinter::default();
    RELAY.install(printer.clone());

    // Persistent, cross-session input history backed by the SQLite DB. On failure, degrade to
    // reedline's default in-memory history rather than taking down the REPL.
    const HISTORY_CAPACITY: usize = 5000;
    let history: Option<Box<dyn History>> = history_db_path.and_then(|path| {
        match crate::host::repl::history::PromptHistory::open(&path, HISTORY_CAPACITY) {
            Ok(history) => Some(Box::new(history) as Box<dyn History>),
            Err(error) => {
                tracing::warn!("failed to open input history database: {error}");
                None
            }
        }
    });

    // Checked once per prompt, not per keystroke, and re-read only when the files have actually
    // moved. Frozen at construction it was simply wrong under `[skills] agent_managed`, where
    // `skill_write` and `skill_delete` move the set mid-session; re-discovered unconditionally it
    // parsed every `SKILL.md` before drawing every prompt and reprinted the unloadable-skill
    // warnings with it. `SkillNameWatch` is the stat-and-compare `SkillCache` makes on the agent
    // side, for a caller that cannot await it.
    let skill_names = Arc::new(std::sync::RwLock::new(Vec::new()));
    let refresh_skill_names = {
        let skill_names = Arc::clone(&skill_names);
        let watch = std::cell::RefCell::new(crate::skills::SkillNameWatch::new(skill_roots));
        move || {
            let Some(discovered) = watch.borrow_mut().refresh() else {
                return;
            };
            *crate::sync::write(&skill_names) = discovered;
        }
    };
    refresh_skill_names();
    let completer = SlashCompleter {
        mcp_server_names,
        skill_names,
        profile_names: configured_profiles
            .iter()
            .map(|profile| profile.name.clone())
            .collect(),
        cwd: cwd.clone(),
    };

    // Raised the instant reedline commits to submitting and lowered as soon as `read_line` has
    // returned, so `input_style` paints the line that stays on screen and nothing else.
    let submitted = Arc::new(AtomicBool::new(false));

    let mut editor = build_reedline_editor(
        input_style,
        printer,
        history,
        completer,
        wake,
        CyclePermissionMode {
            inner: Emacs::new(meka_keybindings()),
            shared_permission: shared_permission.clone(),
            input_sender: input_sender.clone(),
            sandbox_state,
        },
        Arc::clone(&submitted),
    );
    let prompt = MekaPrompt {
        shared_permission: shared_permission.clone(),
        show_path: show_path_in_prompt,
        cwd: cwd.clone(),
        context: context_indicator.map(|(tokens, window)| ContextIndicator { tokens, window }),
    };

    // If the caller queued a synthetic first turn (e.g. `--skill` or a bare positional `[PROMPT]`
    // in interactive mode), drain agent events for that turn before drawing the first reedline
    // prompt. Otherwise the prompt indicator and the agent's stdout output collide on screen.
    if initial_turn_pending && !wait_for_agent(&agent_event_receiver, &console) {
        return;
    }

    loop {
        // reedline drains the relay's `ExternalPrinter` only inside `read_line()`. Flag that window
        // so log lines route through the printer (cleanly above the live prompt) while it's active
        // and go straight to stderr otherwise (e.g. during a turn), surfacing immediately instead
        // of buffering until the turn ends and the next prompt is drawn.
        // Between turns, so a skill the agent has just written or deleted is what Tab offers. Once
        // per prompt is the right cadence for the stat pass, and it happens while the user has not
        // started typing; the parse behind it only runs when the stats have moved.
        refresh_skill_names();
        // The one place an episode can end before a prompt, which is what makes the bracket
        // impossible to skip: no `continue`, `break` or dispatch arm below reaches the next prompt
        // without passing here. `main` closes the last one, which no prompt follows.
        with_console(&console, |console| {
            console.close_episode(crate::console::Neighbor::Prompt)
        });
        RELAY.set_at_prompt(true);
        let signal = editor.read_line(&prompt);
        RELAY.set_at_prompt(false);
        with_console(&console, |console| {
            console.open_episode(row_after(&signal), crate::console::Neighbor::Prompt)
        });
        // Every exit lowers it, not just a submitted line: a Ctrl+C or a scheduler wake leaves the
        // buffer to be edited further, and it must go back to being edited plainly.
        submitted.store(false, Ordering::Relaxed);
        match signal {
            // A scheduled job came due while the prompt sat idle. `read_line` has returned, so the
            // terminal is back in cooked mode and the turn that follows is indistinguishable from
            // one the user typed: it streams, Ctrl+C reaches it, and the absent prompt is what
            // reads as "busy". The buffer is whatever the user had half-typed, restored below.
            Ok(Signal::ExternalBreak(buffer)) => {
                // The prompt line is closed on the agent side, not here, because only that side
                // knows whether a job actually fires: a wake can be spurious when another process
                // claims the job first, and a blank line printed for a turn that never happens is
                // a stray gap above the redrawn prompt. See `ReplEvent::Wake` in `main`.
                if input_sender.send(ReplEvent::Wake).is_err() {
                    break;
                }
                if !wait_for_agent(&agent_event_receiver, &console) {
                    break;
                }
                // Nothing to restore: reedline hands back a *copy* of the line editor's contents
                // and leaves the editor itself untouched (its break handler only resets the undo
                // stack, unlike `submit_buffer`, which clears). Re-inserting would give the user
                // their half-typed line twice.
                let _still_in_the_editor = buffer;
                continue;
            }
            Ok(Signal::Success(buffer)) => {
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
                            with_console(&console, |console| console.chrome(print_help));
                            continue;
                        }
                        Some(SlashCommand::Clear) => {
                            // The screen the REPL draws on is stderr, and stdout carries only the
                            // answers: with stdout redirected the escape sequences landed in the
                            // file and the visible screen stayed as it was.
                            use std::io::IsTerminal;
                            if !std::io::stderr().is_terminal()
                                || crossterm::execute!(
                                    std::io::stderr(),
                                    crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
                                    crossterm::cursor::MoveTo(0, 0),
                                )
                                .is_err()
                            {
                                with_console(&console, |console| {
                                    console.line("Failed to clear terminal.")
                                });
                            }
                            continue;
                        }
                        Some(SlashCommand::Profile(argument)) => {
                            match argument {
                                None => {
                                    let current = crate::sync::read(&current_profile).clone();
                                    // One profile per line rather than a comma-joined run of
                                    // names. The list grows with every account and endpoint a
                                    // user adds, and a single line stops fitting long before it
                                    // stops being worth reading. The `name (backend)` shape is
                                    // the one `/status` already uses for the same pair.
                                    //
                                    // The profile this session runs on first, then every profile
                                    // there is. One per line rather than a comma-joined run of
                                    // names: the list grows with every account and endpoint a user
                                    // adds, and a single line stops fitting long before it stops
                                    // being worth reading. Both the `name (backend)` shape and the
                                    // heading are `/status`'s, so the two commands read alike.
                                    //
                                    // Deliberately not called "available". This is what
                                    // `config.toml` holds, and says nothing about whether each
                                    // profile has a credential to go with it; `meka profile list`
                                    // answers that, and promising it here would list a profile
                                    // that has never been logged into as ready to use.
                                    with_console(&console, |console| {
                                        console.line(&format!("Current profile: {current}"));
                                        if !configured_profiles.is_empty() {
                                            console.line("");
                                            console.heading("Configured profiles");
                                            for profile in &configured_profiles {
                                                console.line(&format!(
                                                    "- {} ({}, {})",
                                                    profile.name, profile.account, profile.backend
                                                ));
                                            }
                                        }
                                    });
                                }
                                Some(name) => {
                                    let name = name.trim().to_string();
                                    // Resolved and recorded on the agent side, which owns both the
                                    // registry and the session row; this thread only asks. Waited
                                    // on like every other forwarded command: the agent prints the
                                    // outcome, and a prompt painted before it arrives lands under
                                    // whatever the user types next.
                                    if input_sender.send(ReplEvent::ProfileChange(name)).is_err() {
                                        // Loud, and the end of the shell: the profile a session
                                        // runs on is only moved on the agent's side, so a debug
                                        // log here left the user looking at a prompt that had
                                        // silently declined to do the one thing they asked for.
                                        with_console(&console, |console| {
                                            console.error(
                                                &"the agent stopped; the profile was not changed",
                                            )
                                        });
                                        break;
                                    } else if !wait_for_agent(&agent_event_receiver, &console) {
                                        break;
                                    }
                                }
                            }
                            continue;
                        }
                        Some(SlashCommand::Permission(argument)) => {
                            match argument {
                                None => {
                                    let current = shared_permission.get();
                                    with_console(&console, |console| {
                                        console
                                            .line(&format!("Current permission level: {current}"))
                                    });
                                }
                                Some(level) => {
                                    match level.parse::<crate::permission::Permission>() {
                                        Ok(permission) => {
                                            match shared_permission.try_set(permission) {
                                                Ok(()) => {
                                                    with_console(&console, |console| {
                                                        console.line(&format!(
                                                            "Permission level set to: {permission}"
                                                        ))
                                                    });
                                                    // Persisted for the same reason as the
                                                    // Shift+Tab path above.
                                                    if input_sender
                                                        .send(ReplEvent::PermissionChanged(
                                                            permission,
                                                        ))
                                                        .is_err()
                                                    {
                                                        warn_unrecorded("the permission level");
                                                    }
                                                }
                                                Err(error) => {
                                                    with_console(&console, |console| {
                                                        console.error(&error)
                                                    });
                                                }
                                            }
                                        }
                                        Err(error) => {
                                            with_console(&console, |console| console.error(&error));
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        Some(SlashCommand::Approvals(argument)) => {
                            match argument.as_deref().map(str::trim) {
                                None | Some("") => {
                                    let current = if shared_permission.approvals() {
                                        "on"
                                    } else {
                                        "off"
                                    };
                                    with_console(&console, |console| {
                                        console.line(&format!("Approvals: {current}"))
                                    });
                                }
                                Some(word @ ("on" | "off")) => {
                                    let approvals = word == "on";
                                    shared_permission.set_approvals(approvals);
                                    with_console(&console, |console| {
                                        console.line(&format!("Approvals set to: {word}"))
                                    });
                                    // Persisted for the same reason as the level.
                                    if input_sender
                                        .send(ReplEvent::ApprovalsChanged(approvals))
                                        .is_err()
                                    {
                                        warn_unrecorded("the approvals switch");
                                    }
                                }
                                Some(other) => {
                                    with_console(&console, |console| {
                                        console.error(&format!(
                                            "'{other}' is not a setting; use `/approvals on` or \
                                             `/approvals off`"
                                        ))
                                    });
                                }
                            }
                            continue;
                        }
                        Some(SlashCommand::Cd(argument)) => {
                            match handle_cd(&cwd, &launch_cwd, argument.as_deref().unwrap_or("")) {
                                // Not waited on, unlike `/profile`: the move has already happened
                                // in this process and the prompt itself is the confirmation, so
                                // holding the line for a bookkeeping write would make the one
                                // command that prints nothing feel like the slowest.
                                Ok(moved) => {
                                    if input_sender.send(ReplEvent::CwdChanged(moved)).is_err() {
                                        warn_unrecorded("the working directory");
                                    }
                                }
                                Err(message) => {
                                    with_console(&console, |console| console.line(&message));
                                }
                            }
                            continue;
                        }
                        // Everything the arms above did not answer goes to the host. What makes
                        // that safe is not this arm: it is that `answered_by` and the host's own
                        // `match` are both exhaustive, so a new variant fails to compile until
                        // someone has said which side owns it. The assertion catches the remaining
                        // drift -- a variant `answered_by` calls ours that no arm above handles --
                        // in the builds where a test would see it.
                        Some(command) => {
                            debug_assert_eq!(
                                command.answered_by(),
                                Answerer::Host,
                                "the REPL thread answers this command, so an arm above should \
                                 have; forwarding it sends the host something it will not match"
                            );
                            if input_sender.send(ReplEvent::Command(command)).is_err() {
                                break;
                            }
                            if !wait_for_agent(&agent_event_receiver, &console) {
                                break;
                            }
                            continue;
                        }
                        None => {
                            // Not the `unknown_name` template: listing every slash command on one
                            // line is noise, and `/help` already is the list (k4yt3x's call).
                            with_console(&console, |console| {
                                console.line(&format!(
                                    "Unknown command: {trimmed}. Type /help for available commands."
                                ))
                            });
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
                    // Spaced like any other command: the child inherits stdio, so its output lands
                    // between two prompts exactly as a slash command's does. Unlike a slash
                    // command this is spaced unconditionally, because the terminal is the child's
                    // from here and meka never learns whether it wrote anything: a silent
                    // `!touch foo` gets brackets around nothing, and the alternative is capturing
                    // the child's output, which would break every interactive `!` command.
                    with_console(&console, |console| console.announce_foreign_output());
                    // Run in the session's working directory so `!` commands track `/cd`. `/cd`
                    // updates the `SharedCwd` (not the process cwd), so without this `!pwd` would
                    // report the original launch directory.
                    let working_dir = cwd.get();
                    #[cfg(windows)]
                    let status = std::process::Command::new("powershell")
                        .arg("-Command")
                        .arg(shell_command)
                        .current_dir(&working_dir)
                        .status();

                    #[cfg(not(windows))]
                    let status = std::process::Command::new("sh")
                        .arg("-c")
                        .arg(shell_command)
                        .current_dir(&working_dir)
                        .status();
                    match status {
                        Ok(exit_status) => {
                            if !exit_status.success()
                                && let Some(code) = exit_status.code()
                            {
                                with_console(&console, |console| {
                                    console.line(&format!("Command exited with status {code}"))
                                });
                            }
                        }
                        Err(error) => {
                            with_console(&console, |console| {
                                console.line(&format!("Failed to execute command: {error}"))
                            });
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

                if !wait_for_agent(&agent_event_receiver, &console) {
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
            // A host command with no handler. Nothing in meka binds `ExecuteHostCommand`, so
            // reaching here means someone added a binding and forgot the arm. Ignore it rather than
            // ending the session: dropping a keystroke is a smaller surprise than the REPL quitting
            // under the user.
            Ok(Signal::HostCommand(command)) => {
                tracing::warn!("unhandled reedline host command: {command}");
                continue;
            }
            // `Signal` is `#[non_exhaustive]`, so this arm is mandatory rather than defensive: it
            // exists for variants a future reedline adds, which by definition we cannot interpret.
            Ok(other) => {
                tracing::warn!("unexpected reedline signal: {other:?}");
                if input_sender.send(ReplEvent::Exit).is_err() {
                    tracing::trace!("REPL event receiver already dropped");
                }
                break;
            }
            Err(error) => {
                tracing::error!("readline error: {error}");
                if input_sender.send(ReplEvent::Exit).is_err() {
                    tracing::trace!("REPL event receiver already dropped");
                }
                break;
            }
        }
    }
}

/// Wait for the agent to signal it is done, while also handling tool approval requests that arrive
/// while approvals are on.
///
/// `false` means the agent side is gone, and every caller leaves the shell on it. It is said out
/// loud rather than returned quietly because the alternative, seen live, is a shell that accepts
/// `/profile` and `/session` and answers neither: everything those commands do happens on the
/// agent's side of this channel, so without a word here the user is left typing into something that
/// ignores them.
/// A change the editor thread applied to its cells but has nobody to hand to the row writer.
fn warn_unrecorded(what: &str) {
    tracing::warn!("failed to record {what} for the session: the agent loop has exited");
}
pub(super) fn wait_for_agent(
    agent_event_receiver: &std::sync::mpsc::Receiver<AgentToReplEvent>,
    console: &Mutex<crate::console::Console>,
) -> bool {
    loop {
        match agent_event_receiver.recv() {
            Ok(AgentToReplEvent::Done) => return true,
            Ok(AgentToReplEvent::ApprovalRequest(request)) => {
                handle_approval_request(request, console);
            }
            Ok(AgentToReplEvent::McpElicitation { prompt, responder }) => {
                handle_elicitation_prompt(prompt, responder, console);
            }
            Ok(AgentToReplEvent::McpProgress(update)) => {
                render_progress_update(&update, console);
            }
            Err(_) => {
                with_console(console, |console| {
                    console.error(&"the agent stopped; leaving the shell")
                });
                return false;
            }
        }
    }
}

/// One-line status overwrite on stderr for a running MCP tool.
///
/// Drawn through the console as a transient row, because it is: the line carries no newline and the
/// text is the server's, so the next thing meka prints has to replace it rather than continue it.
/// Before the console tracked that, whatever printed next spent its own blank line terminating this
/// row -- most visibly the blank before the prompt, at the end of a turn whose last act was an MCP
/// call.
pub(super) fn render_progress_update(
    update: &crate::frontend::ProgressUpdate,
    console: &Mutex<crate::console::Console>,
) {
    let line = format_progress_update(update);
    with_console(console, |console| {
        // Always drawn as far as this can tell. `write_stderr` accepts a failed write rather than
        // reporting it down the stream that just refused it, so there is nothing left to answer
        // `false` on: the row is claimed, and a stderr that will not take a progress line will not
        // take the redraw that replaces it either.
        console.transient(|| {
            crate::streams::write_stderr(&line);
            true
        })
    });
}

/// Format a progress line. Sanitizes server-controlled strings so an MCP server can't inject ANSI
/// escapes to clear the screen or spoof prompts.
///
/// Every field is flattened to a single line and width-bounded, not merely stripped of controls.
/// The line opens with meka's own `\r` to overwrite the previous progress, so anything that
/// survives a newline in a server's string would be painted at column zero on a *fresh* row, below
/// chrome the user has already read -- which is a forged approval prompt with no escape sequence
/// involved. `begin_own_line` cannot help there: it clears the current row, and the newline has
/// already committed the rows above it.
pub(super) fn format_progress_update(update: &crate::frontend::ProgressUpdate) -> String {
    // Flattened, not merely sanitized. `sanitize_text` deliberately keeps `\n`, and both of these
    // are server-controlled: `tool_name` is the raw name the server advertised (only the namespaced
    // form goes through `normalize_server_name`). A tool called "x\n[approval]
    // execute_command\n..." would otherwise open new rows inside meka's own chrome, which is
    // the forgery the message half of this line was already fixed for.
    let server = crate::text::sanitize_to_line(&update.server_name, usize::MAX);
    let tool = crate::text::sanitize_to_line(&update.tool_name, usize::MAX);
    let counter = match update.total {
        Some(total) if total > 0.0 => format!("{:.0}/{:.0}", update.progress, total),
        _ => format!("{:.0}", update.progress),
    };
    let prefix = format!("[mcp:{server}/{tool}] {counter} ");

    // Budget what is left of the row for the server's message, after meka's own chrome and the
    // trailing pad. Saturating: a narrow terminal or a long server/tool name simply leaves no room
    // for the message rather than underflowing.
    let pad = 5;
    let budget = crate::render::output_width()
        .saturating_sub(prefix.chars().count())
        .saturating_sub(pad);
    let message = update
        .message
        .as_deref()
        .map(|raw| crate::text::sanitize_to_line(raw, budget))
        .unwrap_or_default();

    // Pad with a few spaces so the next print clears trailing chars from any longer previous line.
    format!("\r{}{}{}", prefix, message, " ".repeat(pad))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_cd_updates_shared_cwd_without_mutating_process_cwd() {
        // Working directory mutation is per-session now; verify `/cd` writes to the `SharedCwd` and
        // leaves `std::env::current_dir()` untouched. Use a tempdir + canonicalize so the assertion
        // is robust to platform-specific symlinks (e.g. `/tmp` → `/private/tmp` on macOS).
        let temp = tempfile::tempdir().expect("tempdir");
        let target = crate::workspace::canonical_for_test(temp.path());
        let process_cwd_before = std::env::current_dir().expect("read process cwd before /cd");

        let cwd =
            crate::workspace::SharedCwd::new(std::path::PathBuf::from("/this/path/does/not/exist"));
        let landed = handle_cd(
            &cwd,
            &process_cwd_before,
            target.to_str().expect("utf-8 tempdir"),
        );

        assert_eq!(
            landed.as_deref(),
            Ok(target.as_path()),
            "a `cd` that worked reports where it landed and has nothing to print, which is what \
             the caller keys the `[display]` blank lines off",
        );
        let stored = cwd.get();
        assert_eq!(stored, target, "shared cwd must point at the new directory");
        let process_cwd_after = std::env::current_dir().expect("read process cwd after /cd");
        assert_eq!(
            process_cwd_after, process_cwd_before,
            "process cwd must NOT be mutated by /cd",
        );
    }

    #[test]
    fn handle_cd_reports_a_failure_rather_than_moving() {
        let temp = tempfile::tempdir().expect("tempdir");
        let file = temp.path().join("not-a-directory");
        std::fs::write(&file, b"x").expect("write file");
        let start = crate::workspace::canonical_for_test(temp.path());

        let cwd = crate::workspace::SharedCwd::new(start.clone());

        let missing = handle_cd(&cwd, &start, "/this/path/does/not/exist");
        assert!(
            missing.is_err_and(|message| message.starts_with("cd: ")),
            "a target that cannot be resolved must produce a message to print",
        );

        let not_a_directory = handle_cd(&cwd, &start, file.to_str().expect("utf-8 path"));
        assert!(
            not_a_directory.is_err_and(|message| message.contains("not a directory")),
            "an existing non-directory must be refused, not silently accepted",
        );

        assert_eq!(
            cwd.get(),
            start,
            "a failed `cd` must leave the session where it was",
        );
    }

    /// A bare `/cd` returns to the launch directory, where a shell's `cd` goes home.
    ///
    /// The two differ deliberately: a resumed session opens in the directory it recorded, so
    /// "take me back to my shell" is the move this actually serves, and at `workspace` the working
    /// directory is the writable boundary -- which makes `$HOME` the worst available default.
    #[test]
    fn a_bare_cd_returns_to_the_launch_directory_rather_than_home() {
        let temp = tempfile::tempdir().expect("tempdir");
        let launch = crate::workspace::canonical_for_test(temp.path());
        let elsewhere = temp.path().join("elsewhere");
        std::fs::create_dir(&elsewhere).expect("create elsewhere");
        let elsewhere = crate::workspace::canonical_for_test(&elsewhere);

        let cwd = crate::workspace::SharedCwd::new(elsewhere);

        assert_eq!(
            handle_cd(&cwd, &launch, "").as_deref(),
            Ok(launch.as_path()),
            "`/cd` with no argument goes back to where meka was started",
        );

        // And `~` still spells home, so nothing is taken away -- only the default moved.
        let Some(home) = dirs::home_dir() else {
            return;
        };
        let home = crate::workspace::canonical_for_test(&home);
        assert_eq!(
            handle_cd(&cwd, &launch, "~").as_deref(),
            Ok(home.as_path()),
            "`/cd ~` must still reach the home directory",
        );
    }

    /// A relative target still resolves against where the session is now, not the launch directory.
    /// Threading a second path in is exactly the sort of change that quietly re-bases this.
    #[test]
    fn a_relative_cd_still_resolves_against_the_session_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let launch = crate::workspace::canonical_for_test(temp.path());
        let nested = temp.path().join("outer");
        std::fs::create_dir_all(nested.join("inner")).expect("create nested dirs");
        let outer = crate::workspace::canonical_for_test(&nested);

        let cwd = crate::workspace::SharedCwd::new(outer.clone());

        assert_eq!(
            handle_cd(&cwd, &launch, "inner").as_deref(),
            Ok(outer.join("inner").as_path()),
            "a relative `/cd` lands inside the directory the session is in",
        );
    }

    /// A highlighter already past the moment of submission, which is the state the styling tests
    /// below are about.
    fn submitted_highlighter(style: nu_ansi_term::Style) -> UserInputHighlighter {
        UserInputHighlighter {
            style,
            submitted: Arc::new(AtomicBool::new(true)),
        }
    }

    #[test]
    fn user_input_highlighter_default_preset_preserves_literal() {
        let highlighter = submitted_highlighter(crate::config::default_input_style());
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
    fn user_input_highlighter_none_emits_no_escape() {
        let highlighter = submitted_highlighter(nu_ansi_term::Style::default());
        let rendered = highlighter.highlight("hello", 0).render_simple();
        assert_eq!(rendered, "hello");
    }

    #[test]
    fn user_input_highlighter_known_command_distinct_from_unknown() {
        let highlighter = submitted_highlighter(crate::config::default_input_style());
        let known = highlighter.highlight("/compact", 8).render_simple();
        let unknown = highlighter.highlight("/bogus", 6).render_simple();
        assert!(
            known.contains("/compact"),
            "known token survives: {known:?}"
        );
        assert!(
            unknown.contains("/bogus"),
            "unknown token survives: {unknown:?}"
        );
        assert_ne!(
            known, unknown,
            "known and unknown commands must render with different styles"
        );
    }

    #[test]
    fn user_input_highlighter_non_slash_single_style() {
        let highlighter = submitted_highlighter(crate::config::default_input_style());
        let line = "hello world";
        let mut expected = StyledText::new();
        expected.push((highlighter.style, line.to_string()));
        assert_eq!(
            highlighter.highlight(line, 0).render_simple(),
            expected.render_simple()
        );
    }

    #[test]
    fn only_the_base_style_waits_for_submit() {
        let submitted = Arc::new(AtomicBool::new(false));
        let highlighter = UserInputHighlighter {
            style: crate::config::default_input_style(),
            submitted: Arc::clone(&submitted),
        };

        // Three cases, because two would not pin this. Asserting only that a line being typed is
        // plain also passes against a highlighter that stopped styling altogether, and asserting
        // only that a submitted line is styled also passes against one that styles always.
        let typing = highlighter.highlight("hello world", 5).render_simple();
        assert_eq!(
            typing, "hello world",
            "a line still being typed keeps the terminal's own colors: {typing:?}"
        );

        let typing_command = highlighter.highlight("/help", 5).render_simple();
        assert!(
            typing_command.contains("\x1b["),
            "the slash token is recolored while there is still time to fix the spelling: \
             {typing_command:?}"
        );

        submitted.store(true, Ordering::Relaxed);
        let sent = highlighter.highlight("hello world", 5).render_simple();
        assert!(
            sent.contains("\x1b["),
            "a submitted line carries the input style, which is what separates it from the reply \
             printed under it: {sent:?}"
        );
    }

    #[test]
    fn the_validator_releases_the_highlighter_it_was_built_with() {
        use reedline::Validator;

        let (highlighter, watcher) = submit_aware_input_painter(
            crate::config::default_input_style(),
            Arc::new(AtomicBool::new(false)),
        );
        assert_eq!(
            highlighter.highlight("hello", 0).render_simple(),
            "hello",
            "nothing has been submitted yet"
        );

        assert!(
            matches!(
                watcher.validate("hello"),
                reedline::ValidationResult::Complete
            ),
            "the watcher must never hold a line back; it only reports the decision"
        );
        assert!(
            highlighter
                .highlight("hello", 0)
                .render_simple()
                .contains("\x1b["),
            "the pair shares one cell, so reedline's submit decision reaches the paint that follows"
        );
    }

    fn empty_completer() -> SlashCompleter {
        SlashCompleter {
            mcp_server_names: Vec::new(),
            skill_names: Arc::new(std::sync::RwLock::new(Vec::new())),
            profile_names: Vec::new(),
            cwd: crate::workspace::cwd_for_test(),
        }
    }

    fn completer_at(cwd: crate::workspace::SharedCwd) -> SlashCompleter {
        SlashCompleter {
            mcp_server_names: vec!["postgres".into(), "github".into()],
            skill_names: Arc::new(std::sync::RwLock::new(vec![
                "search".into(),
                "deep-research".into(),
            ])),
            profile_names: Vec::new(),
            cwd,
        }
    }

    #[test]
    fn slash_completer_prefix_matches_expected() {
        let completer = empty_completer();
        let suggestions = completer.suggestions("/comp", 5);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].value, "/compact");
    }

    #[test]
    fn slash_completer_bare_slash_returns_all() {
        let completer = empty_completer();
        let suggestions = completer.suggestions("/", 1);
        assert_eq!(suggestions.len(), crate::host::COMMANDS.len());
        assert!(suggestions.iter().all(|s| s.value.starts_with('/')));
    }

    #[test]
    fn slash_completer_non_slash_returns_empty() {
        let completer = empty_completer();
        assert!(completer.suggestions("hello", 5).is_empty());
        assert!(completer.suggestions("", 0).is_empty());
    }

    #[test]
    fn slash_completer_no_args_for_argless_commands() {
        let completer = empty_completer();
        // Commands without an argument completer return nothing once past the command word.
        assert!(completer.suggestions("/compact ", 9).is_empty());
        assert!(completer.suggestions("/status foo", 11).is_empty());
    }

    #[test]
    fn slash_completer_span_replaces_whole_token() {
        let completer = empty_completer();
        let suggestions = completer.suggestions("/comp", 5);
        assert_eq!(suggestions[0].span.start, 0);
        assert_eq!(suggestions[0].span.end, 5);
    }

    #[test]
    fn slash_completer_append_whitespace_tracks_arguments() {
        let completer = empty_completer();
        assert!(completer.suggestions("/permission", 11)[0].append_whitespace);
        assert!(completer.suggestions("/cd", 3)[0].append_whitespace);
        // `/compact` takes optional instructions, so completing it leaves the cursor ready to type
        // them.
        assert!(completer.suggestions("/compact", 8)[0].append_whitespace);
        assert!(!completer.suggestions("/help", 5)[0].append_whitespace);
    }

    #[test]
    fn slash_completer_descriptions_present() {
        let completer = empty_completer();
        assert!(
            completer
                .suggestions("/", 1)
                .iter()
                .all(|s| s.description.as_deref().is_some_and(|d| !d.is_empty()))
        );
    }

    #[test]
    fn slash_completer_does_not_offer_aliases() {
        let completer = empty_completer();
        // `/q` matches the `quit` alias of `exit`, but aliases are never completed.
        assert!(completer.suggestions("/q", 2).is_empty());
    }

    #[test]
    fn slash_completer_permission_arg_prefix() {
        let completer = empty_completer();
        let one = completer.suggestions("/permission wo", 14);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].value, "workspace");
        assert!(one[0].append_whitespace);
        let all: Vec<String> = completer
            .suggestions("/permission ", 12)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect();
        assert_eq!(all, ["none", "read", "workspace", "unrestricted"]);
    }

    #[test]
    fn slash_completer_permission_no_complete_second_arg() {
        let completer = empty_completer();
        assert!(
            completer
                .suggestions("/permission workspace extra", 27)
                .is_empty()
        );
    }

    #[test]
    fn slash_completer_skill_arg_prefix() {
        let completer = completer_at(crate::workspace::cwd_for_test());
        let suggestions = completer.suggestions("/skill sea", 10);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].value, "search");
    }

    /// The completer follows the skill set rather than the one it was built with.
    ///
    /// A `Vec<String>` frozen at construction cannot follow a list that changes. With `[skills]
    /// agent_managed`, `skill_write` and `skill_delete` move that set mid-session, so Tab went on
    /// offering a skill the agent had deleted and `/skill <name>` then failed on a name Tab had
    /// just supplied. The prompt loop refreshes the shared handle before every `read_line`; this is
    /// the half of that arrangement a unit test can reach.
    #[test]
    fn the_skill_completer_follows_a_set_that_changes_under_it() {
        let completer = completer_at(crate::workspace::cwd_for_test());
        assert_eq!(
            completer.suggestions("/skill sea", 10).len(),
            1,
            "the fixture starts with `search` installed"
        );

        *crate::sync::write(&completer.skill_names) = vec!["deploy".into()];

        assert!(
            completer.suggestions("/skill sea", 10).is_empty(),
            "a deleted skill must stop being offered"
        );
        let suggestions = completer.suggestions("/skill dep", 10);
        assert_eq!(suggestions.len(), 1, "and a new one must start");
        assert_eq!(suggestions[0].value, "deploy");
    }

    #[test]
    fn slash_completer_skill_no_complete_second_arg() {
        let completer = completer_at(crate::workspace::cwd_for_test());
        assert!(completer.suggestions("/skill search foo", 17).is_empty());
    }

    #[test]
    fn slash_completer_mcp_arg1_keywords() {
        let completer = completer_at(crate::workspace::cwd_for_test());
        let all: Vec<String> = completer
            .suggestions("/mcp ", 5)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect();
        assert_eq!(all, ["list", "reconnect", "login", "logout"]);
        let rec = completer.suggestions("/mcp rec", 8);
        assert_eq!(rec.len(), 1);
        assert_eq!(rec[0].value, "reconnect");
    }

    #[test]
    fn slash_completer_mcp_arg2_server_after_subcommand() {
        let completer = completer_at(crate::workspace::cwd_for_test());
        let servers: Vec<String> = completer
            .suggestions("/mcp reconnect ", 15)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect();
        assert_eq!(servers, ["postgres", "github"]);
        assert_eq!(
            completer.suggestions("/mcp login git", 14)[0].value,
            "github"
        );
        // `list` takes no server argument, so its second token completes nothing.
        assert!(completer.suggestions("/mcp list ", 10).is_empty());
    }

    #[test]
    fn slash_completer_cd_lists_directories() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = crate::workspace::canonical_for_test(temp.path());
        std::fs::create_dir(root.join("src")).expect("mkdir src");
        std::fs::create_dir(root.join("target")).expect("mkdir target");
        std::fs::create_dir(root.join(".git")).expect("mkdir .git");
        std::fs::write(root.join("README"), b"x").expect("write file");
        std::fs::create_dir_all(root.join("src/tools")).expect("mkdir src/tools");
        let cwd = crate::workspace::SharedCwd::new(root);
        let completer = completer_at(cwd);

        let bare: Vec<String> = completer
            .suggestions("/cd ", 4)
            .into_iter()
            .map(|suggestion| suggestion.value)
            .collect();
        // Directories returned with a trailing slash; the file and dotdir are excluded.
        assert!(bare.contains(&"src/".to_string()));
        assert!(bare.contains(&"target/".to_string()));
        assert!(!bare.iter().any(|value| value.contains("README")));
        assert!(!bare.contains(&".git/".to_string()));

        // A leading dot in the partial opts dotdirs back in.
        let dot = completer.suggestions("/cd .gi", 7);
        assert_eq!(dot.len(), 1);
        assert_eq!(dot[0].value, ".git/");

        // Relative drill-down keeps the parent portion intact.
        let nested = completer.suggestions("/cd src/too", 11);
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0].value, "src/tools/");
        assert!(!nested[0].append_whitespace);
        assert_eq!(nested[0].span.start, 4);
        assert_eq!(nested[0].span.end, 11);
    }

    #[test]
    fn slash_completer_command_word_still_completes() {
        let completer = completer_at(crate::workspace::cwd_for_test());
        let suggestions = completer.suggestions("/comp", 5);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].value, "/compact");
        assert_eq!(suggestions[0].span.start, 0);
        assert_eq!(suggestions[0].span.end, 5);
    }

    #[test]
    fn parse_slash_command_exit() {
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
    fn parse_slash_command_help() {
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
    fn parse_slash_command_clear() {
        assert!(matches!(
            parse_slash_command("/clear"),
            Some(SlashCommand::Clear)
        ));
    }

    #[test]
    fn parse_slash_command_session() {
        assert!(matches!(
            parse_slash_command("/session"),
            Some(SlashCommand::Session)
        ));
    }

    #[test]
    fn parse_slash_command_permission() {
        assert!(matches!(
            parse_slash_command("/permission"),
            Some(SlashCommand::Permission(None))
        ));
        match parse_slash_command("/permission workspace") {
            Some(SlashCommand::Permission(Some(arg))) => assert_eq!(arg, "workspace"),
            _ => panic!("expected Permission with argument"),
        }
    }

    #[test]
    fn parse_slash_command_compact() {
        assert!(matches!(
            parse_slash_command("/compact"),
            Some(SlashCommand::Compact(None))
        ));
    }

    /// Everything after the command is the instruction, verbatim: it is prose for a model, not a
    /// parsed argument, so splitting or validating it would only be able to get it wrong.
    #[test]
    fn parse_slash_command_compact_with_instructions() {
        assert!(matches!(
            parse_slash_command("/compact keep the auth decisions, drop the debugging"),
            Some(SlashCommand::Compact(Some(instructions)))
                if instructions == "keep the auth decisions, drop the debugging"
        ));
    }

    #[test]
    fn parse_slash_command_rewind() {
        assert!(matches!(
            parse_slash_command("/rewind"),
            Some(SlashCommand::Rewind(None))
        ));
        assert!(matches!(
            parse_slash_command("/rewind 3"),
            Some(SlashCommand::Rewind(Some(3)))
        ));
        // A non-numeric argument is refused, not read as the default: `/rewind all` rewound one
        // turn and reported a count the user had not asked for.
        assert!(matches!(
            parse_slash_command("/rewind all"),
            Some(SlashCommand::RewindInvalid(argument)) if argument == "all"
        ));
    }

    #[test]
    fn parse_slash_command_unknown() {
        assert!(parse_slash_command("/unknown").is_none());
    }

    #[test]
    fn parse_slash_command_not_slash() {
        assert!(parse_slash_command("hello").is_none());
    }

    #[test]
    fn parse_slash_command_empty() {
        assert!(parse_slash_command("/").is_none());
    }

    #[test]
    fn parse_slash_command_cd_no_arg() {
        assert!(matches!(
            parse_slash_command("/cd"),
            Some(SlashCommand::Cd(None))
        ));
    }

    #[test]
    fn parse_slash_command_cd_with_path() {
        match parse_slash_command("/cd /tmp") {
            Some(SlashCommand::Cd(Some(arg))) => assert_eq!(arg, "/tmp"),
            _ => panic!("expected Cd with argument"),
        }
    }

    #[test]
    fn parse_slash_command_export() {
        assert!(matches!(
            parse_slash_command("/export"),
            Some(SlashCommand::Export)
        ));
    }

    #[test]
    fn parse_slash_command_fork() {
        assert!(matches!(
            parse_slash_command("/fork"),
            Some(SlashCommand::Fork)
        ));
    }

    #[test]
    fn parse_slash_command_history_no_args() {
        assert!(matches!(
            parse_slash_command("/history"),
            Some(SlashCommand::History(None))
        ));
    }

    #[test]
    fn parse_slash_command_history_with_n() {
        assert!(matches!(
            parse_slash_command("/history 5"),
            Some(SlashCommand::History(Some(5)))
        ));
        // Whitespace is tolerated.
        assert!(matches!(
            parse_slash_command("/history   12"),
            Some(SlashCommand::History(Some(12)))
        ));
    }

    #[test]
    fn parse_slash_command_history_garbage_falls_back_to_all() {
        // Non-numeric argument (including `all`) collapses to None so the
        // dispatch dumps the whole conversation. Documented behavior:
        // graceful fallback, no error.
        assert!(matches!(
            parse_slash_command("/history all"),
            Some(SlashCommand::History(None))
        ));
        assert!(matches!(
            parse_slash_command("/history banana"),
            Some(SlashCommand::History(None))
        ));
    }

    #[test]
    fn shorten_path_with_tilde_home() {
        if let Some(home) = dirs::home_dir() {
            assert_eq!(shorten_path_with_tilde(&home), "~");
        }
    }

    #[test]
    fn shorten_path_with_tilde_subdir() {
        if let Some(home) = dirs::home_dir() {
            let subdir = home.join("projects").join("test");
            assert_eq!(shorten_path_with_tilde(&subdir), "~/projects/test");
        }
    }

    #[test]
    fn shorten_path_with_tilde_non_home() {
        let path = std::path::Path::new("/tmp/something");
        assert_eq!(shorten_path_with_tilde(path), "/tmp/something");
    }

    #[test]
    fn format_progress_update_strips_ansi_escapes() {
        let update = crate::frontend::ProgressUpdate {
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
            "ANSI escape leaked into progress line: {line:?}"
        );
        assert!(line.contains("spoofed"));
        assert!(line.contains("[mcp:svr/tool]"));
    }

    /// The progress line opens with meka's own `\r` to overwrite the previous one. A newline in the
    /// server's message would therefore commit that row and start painting at column zero on a
    /// fresh line, below chrome the user has already read -- a forged approval block needing no
    /// escape sequence at all. `begin_own_line` cannot undo it, because it only clears the
    /// current row.
    #[test]
    fn a_progress_message_cannot_open_a_second_row() {
        let update = crate::frontend::ProgressUpdate {
            server_name: "svr".to_string(),
            tool_name: "tool".to_string(),
            tool_use_id: None,
            message: Some(
                "working\n[approval] Shell\n  command: ls -la\nAllow? (Y/n) ".to_string(),
            ),
            progress: 1.0,
            total: None,
        };

        let line = format_progress_update(&update);

        // One leading `\r` (meka's own) and no other line break of any kind.
        assert_eq!(line.matches('\r').count(), 1, "{line:?}");
        assert!(line.starts_with('\r'), "{line:?}");
        assert!(!line.contains('\n'), "{line:?}");
    }

    /// A server that pads its message must not be able to scroll the transcript by writing a line
    /// longer than the terminal.
    #[test]
    fn a_progress_message_is_bounded_by_the_terminal_width() {
        let update = crate::frontend::ProgressUpdate {
            server_name: "svr".to_string(),
            tool_name: "tool".to_string(),
            tool_use_id: None,
            message: Some("x".repeat(10_000)),
            progress: 1.0,
            total: None,
        };

        let line = format_progress_update(&update);
        let visible = line.trim_start_matches('\r').chars().count();
        assert!(
            visible <= crate::render::output_width(),
            "progress line ran to {visible} columns: {line:?}"
        );
    }

    #[test]
    fn parse_mcp_slash_empty_is_list() {
        assert!(matches!(
            parse_slash_command("/mcp"),
            Some(SlashCommand::McpList)
        ));
    }

    #[test]
    fn parse_mcp_slash_explicit_list() {
        assert!(matches!(
            parse_slash_command("/mcp list"),
            Some(SlashCommand::McpList)
        ));
    }

    #[test]
    fn parse_mcp_slash_reconnect_with_server() {
        match parse_slash_command("/mcp reconnect postgres") {
            Some(SlashCommand::McpReconnect { server }) => assert_eq!(server, "postgres"),
            other => panic!("expected McpReconnect, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn parse_mcp_slash_reconnect_without_server_is_none() {
        // Bare `reconnect` with no server name: neither the reconnect arm nor the
        // `<server>:<prompt>` arm matches, so the command is rejected rather than silently firing
        // against some default.
        assert!(parse_slash_command("/mcp reconnect").is_none());
    }

    #[test]
    fn parse_mcp_slash_login_with_server() {
        match parse_slash_command("/mcp login notion") {
            Some(SlashCommand::McpLogin { server }) => assert_eq!(server, "notion"),
            other => panic!("expected McpLogin, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn parse_mcp_slash_logout_with_server() {
        match parse_slash_command("/mcp logout notion") {
            Some(SlashCommand::McpLogout { server }) => assert_eq!(server, "notion"),
            other => panic!("expected McpLogout, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn parse_mcp_slash_login_without_server_is_none() {
        assert!(parse_slash_command("/mcp login").is_none());
    }

    #[test]
    fn parse_mcp_slash_logout_without_server_is_none() {
        assert!(parse_slash_command("/mcp logout").is_none());
    }

    #[test]
    fn parse_mcp_slash_login_trims_whitespace() {
        match parse_slash_command("/mcp login   notion  ") {
            Some(SlashCommand::McpLogin { server }) => assert_eq!(server, "notion"),
            other => panic!("expected McpLogin, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn parse_mcp_slash_prompt_no_args() {
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
    fn parse_mcp_slash_prompt_with_args() {
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
    fn parse_mcp_slash_empty_server_rejected() {
        assert!(parse_slash_command("/mcp :prompt").is_none());
    }

    #[test]
    fn parse_mcp_slash_empty_prompt_rejected() {
        assert!(parse_slash_command("/mcp server:").is_none());
    }

    #[test]
    fn parse_mcp_slash_multiple_colons_splits_on_first() {
        // `split_once` returns the first colon, so prompt names can contain further colons.
        match parse_slash_command("/mcp srv:ns:prompt") {
            Some(SlashCommand::McpPrompt { server, prompt, .. }) => {
                assert_eq!(server, "srv");
                assert_eq!(prompt, "ns:prompt");
            }
            other => panic!("expected McpPrompt, got {:?}", option_label(&other)),
        }
    }

    /// Bare `/memory` lists. Mirrors `/skill`'s empty-argument behavior.
    #[test]
    fn parse_memory_slash_empty_is_list() {
        assert!(matches!(
            parse_slash_command("/memory"),
            Some(SlashCommand::MemoryList)
        ));
        assert!(matches!(
            parse_slash_command("/memory   "),
            Some(SlashCommand::MemoryList)
        ));
    }

    /// `/memory <name>` shows that memory. Falling through to the bare-list arm silently discards
    /// the name and lists everything instead.
    #[test]
    fn parse_memory_slash_shows_named_memory() {
        match parse_slash_command("/memory alice-timezone") {
            Some(SlashCommand::MemoryShow { name }) => assert_eq!(name, "alice-timezone"),
            other => panic!("expected MemoryShow, got {:?}", option_label(&other)),
        }
    }

    /// There is no `list` keyword, for the same reason `/skill` has none: it would shadow a
    /// legitimately-named entry.
    #[test]
    fn parse_memory_slash_no_list_keyword() {
        match parse_slash_command("/memory list") {
            Some(SlashCommand::MemoryShow { name }) => assert_eq!(name, "list"),
            other => panic!("expected MemoryShow, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn parse_skill_slash_empty_is_list() {
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
    fn parse_skill_slash_invokes_named_skill() {
        match parse_slash_command("/skill demo") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "demo");
                assert!(extra.is_empty());
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn parse_skill_slash_captures_free_form_extra() {
        // The whole remainder after the skill name is captured verbatim (preserving inner
        // whitespace) and trimmed at the edges. This is free-form text the user wants prepended to
        // the skill body: no positional argument parsing.
        match parse_slash_command("/skill demo only fetch UK news") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "demo");
                assert_eq!(extra, "only fetch UK news");
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn parse_skill_slash_trims_trailing_whitespace() {
        // Trailing whitespace after the skill name should produce an empty extra, not a
        // whitespace-padded one, equivalent to the bare-name invocation.
        match parse_slash_command("/skill demo   ") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "demo");
                assert!(extra.is_empty());
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    #[test]
    fn parse_skill_slash_no_list_keyword() {
        // The token "list" is treated as a skill name, not a subcommand. (Bare `/skill` is the
        // listing form; `/skill list` would error at dispatch with "unknown skill 'list'" if no
        // such skill exists.)
        match parse_slash_command("/skill list") {
            Some(SlashCommand::SkillInvoke { name, extra }) => {
                assert_eq!(name, "list");
                assert!(extra.is_empty());
            }
            other => panic!("expected SkillInvoke, got {:?}", option_label(&other)),
        }
    }

    /// Short debug label: SlashCommand doesn't implement Debug so we map the few variants we care
    /// about manually to keep assertion messages readable.
    fn option_label(cmd: &Option<SlashCommand>) -> &'static str {
        match cmd {
            None => "None",
            Some(SlashCommand::Exit) => "Exit",
            Some(SlashCommand::Help) => "Help",
            Some(SlashCommand::Clear) => "Clear",
            Some(SlashCommand::Session) => "Session",
            Some(SlashCommand::Permission(_)) => "Permission",
            Some(SlashCommand::Approvals(_)) => "Approvals",
            Some(SlashCommand::Profile(_)) => "Profile",
            Some(SlashCommand::Compact(_)) => "Compact",
            Some(SlashCommand::Export) => "Export",
            Some(SlashCommand::Fork) => "Fork",
            Some(SlashCommand::RewindInvalid(_)) => "RewindInvalid",
            Some(SlashCommand::Cd(_)) => "Cd",
            Some(SlashCommand::McpList) => "McpList",
            Some(SlashCommand::McpReconnect { .. }) => "McpReconnect",
            Some(SlashCommand::McpLogin { .. }) => "McpLogin",
            Some(SlashCommand::McpLogout { .. }) => "McpLogout",
            Some(SlashCommand::McpPrompt { .. }) => "McpPrompt",
            Some(SlashCommand::MemoryList) => "MemoryList",
            Some(SlashCommand::MemoryShow { .. }) => "MemoryShow",
            Some(SlashCommand::ScheduleList) => "ScheduleList",
            Some(SlashCommand::ScheduleShow { .. }) => "ScheduleShow",
            Some(SlashCommand::ScheduleCancel { .. }) => "ScheduleCancel",
            Some(SlashCommand::TaskList) => "TaskList",
            Some(SlashCommand::TaskShow { .. }) => "TaskShow",
            Some(SlashCommand::TaskCancel { .. }) => "TaskCancel",
            Some(SlashCommand::SkillList) => "SkillList",
            Some(SlashCommand::SkillInvoke { .. }) => "SkillInvoke",
            Some(SlashCommand::Status) => "Status",
            Some(SlashCommand::Usage) => "Usage",
            Some(SlashCommand::Rewind(_)) => "Rewind",
            Some(SlashCommand::History(_)) => "History",
        }
    }

    #[test]
    fn format_progress_update_strips_rtl_override_in_names() {
        // Defensive: even though server/tool names are normalized at registration time, this
        // confirms the renderer can't be tricked by a handler that someday forgets to normalize.
        let update = crate::frontend::ProgressUpdate {
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
