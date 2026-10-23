//! The composer and the keys: the text area the user types into, its history, slash completion,
//! and the one table that says what a key does.

use std::{path::PathBuf, sync::Arc};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
};
use ratatui_textarea::{CursorMove, TextArea, WrapMode};

use crate::store::history::{CommandLineMatch, HistoryFilter, HistoryOrder, HistoryStore};

/// The composer's background: a dark block that reads as "type here" on a dark theme and stays
/// quiet on a light one.
pub(crate) const COMPOSER_BACKGROUND: Color = Color::Indexed(236);

/// Entries loaded into memory when a walk through the history begins.
const HISTORY_WALK_ENTRIES: usize = 500;

/// Matches shown for a history search.
const HISTORY_SEARCH_MATCHES: i64 = 50;

/// Where a key lands, which decides what it means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Focus {
    /// A picker whose filter is being typed: Enter and the arrows keep their meaning, Esc stops
    /// the typing, and every other key is text.
    Filtering,
    /// The composer, with nothing open over it.
    Composer,
    /// The composer with the completion list open.
    Completion,
    /// The history search.
    Search,
    /// A panel in place of the composer: a command's answer, a picker, the approval prompt.
    Panel,
    /// A question from an MCP server, answered by typing: the elicitation prompt.
    Question,
}

/// What a key does. One table, so every binding is readable in one place and a key cannot mean
/// two things in two places.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    /// Enter: send the composer, accept a pick, answer a question.
    Submit,
    Newline,
    CyclePermission,
    /// Esc or Ctrl+C while the host is busy: the escalation ladder. Ctrl+C when idle clears the
    /// composer.
    Interrupt,
    /// Ctrl+D on an empty composer.
    Leave,
    /// Esc when nothing runs: close whatever is open, let go of the item cursor, or return to the
    /// bottom.
    Escape,
    HistoryPrevious,
    HistoryNext,
    /// Ctrl+R: open the history search, or step to its next match.
    HistorySearch,
    /// Tab: fill in the picked completion, or open the list again after Esc.
    Complete,
    /// Ctrl+V: attach the image on the clipboard to the next prompt.
    PasteImage,
    /// Up and Down while a list is open.
    PickPrevious,
    PickNext,
    ScrollUp,
    ScrollDown,
    ScrollTop,
    ScrollBottom,
    /// Alt+Up and Alt+Down: walk the collapsible items.
    CursorPrevious,
    CursorNext,
    /// Enter or Space with an item picked out.
    ToggleItem,
    ToggleThinking,
    ToggleTools,
    CycleSendClass,
    Redraw,
    /// Anything else goes to the text in focus.
    Edit,
}

/// What the composer is like when a key arrives, for the keys whose meaning depends on it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ComposerState {
    pub(crate) empty: bool,
    pub(crate) at_first_line: bool,
    pub(crate) at_last_line: bool,
    /// An item is picked out, so Enter and Space act on it.
    pub(crate) cursor: bool,
    /// The host is answering something, so Esc stops it rather than backing out of the view.
    pub(crate) busy: bool,
}

/// The key table.
pub(crate) fn action_for(key: &KeyEvent, focus: Focus, composer: ComposerState) -> Action {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    match focus {
        Focus::Question => match key.code {
            KeyCode::Enter => Action::Submit,
            KeyCode::Esc => Action::Escape,
            KeyCode::Char('c') if ctrl => Action::Interrupt,
            KeyCode::Char('l') if ctrl => Action::Redraw,
            KeyCode::PageUp => Action::ScrollUp,
            KeyCode::PageDown => Action::ScrollDown,
            _ => Action::Edit,
        },
        Focus::Filtering => match key.code {
            KeyCode::Enter => Action::Submit,
            KeyCode::Esc => Action::Escape,
            KeyCode::Char('c') if ctrl => Action::Interrupt,
            KeyCode::Char('l') if ctrl => Action::Redraw,
            KeyCode::Up => Action::PickPrevious,
            KeyCode::Down => Action::PickNext,
            KeyCode::PageUp => Action::ScrollUp,
            KeyCode::PageDown => Action::ScrollDown,
            KeyCode::Home if ctrl => Action::ScrollTop,
            KeyCode::End if ctrl => Action::ScrollBottom,
            _ => Action::Edit,
        },
        // Typing into a panel does nothing but pick: it has no field. `j` and `k` move as the
        // arrows do, for the hands that expect them.
        Focus::Panel => match key.code {
            KeyCode::Enter => Action::Submit,
            KeyCode::Esc | KeyCode::Char('q') => Action::Escape,
            KeyCode::Char('c') if ctrl => Action::Interrupt,
            KeyCode::Char('l') if ctrl => Action::Redraw,
            KeyCode::Up | KeyCode::Char('k') => Action::PickPrevious,
            KeyCode::Down | KeyCode::Char('j') => Action::PickNext,
            KeyCode::PageUp => Action::ScrollUp,
            KeyCode::PageDown => Action::ScrollDown,
            KeyCode::Home if ctrl => Action::ScrollTop,
            KeyCode::End if ctrl => Action::ScrollBottom,
            _ => Action::Edit,
        },
        Focus::Search => match key.code {
            KeyCode::Enter => Action::Submit,
            KeyCode::Esc => Action::Escape,
            KeyCode::Char('r') if ctrl => Action::HistorySearch,
            KeyCode::Char('c') if ctrl => Action::Escape,
            KeyCode::Up => Action::PickPrevious,
            KeyCode::Down => Action::PickNext,
            _ => Action::Edit,
        },
        Focus::Completion => match key.code {
            KeyCode::Tab => Action::Complete,
            KeyCode::BackTab => Action::PickPrevious,
            KeyCode::Enter => Action::Submit,
            KeyCode::Esc => Action::Escape,
            KeyCode::Up => Action::PickPrevious,
            KeyCode::Down => Action::PickNext,
            KeyCode::Char('c') if ctrl => Action::Escape,
            _ => Action::Edit,
        },
        Focus::Composer => match key.code {
            KeyCode::Enter if alt || shift => Action::Newline,
            KeyCode::Char('j') if ctrl => Action::Newline,
            KeyCode::Enter if composer.cursor => Action::ToggleItem,
            KeyCode::Char(' ') if composer.cursor => Action::ToggleItem,
            KeyCode::Enter => Action::Submit,
            KeyCode::BackTab => Action::CyclePermission,
            KeyCode::Char('c') if ctrl => Action::Interrupt,
            KeyCode::Esc if composer.busy => Action::Interrupt,
            KeyCode::Char('d') if ctrl && composer.empty => Action::Leave,
            KeyCode::Esc => Action::Escape,
            KeyCode::Up if alt => Action::CursorPrevious,
            KeyCode::Down if alt => Action::CursorNext,
            KeyCode::Up if composer.at_first_line => Action::HistoryPrevious,
            KeyCode::Down if composer.at_last_line => Action::HistoryNext,
            KeyCode::Char('r') if ctrl => Action::HistorySearch,
            KeyCode::Tab => Action::Complete,
            KeyCode::PageUp => Action::ScrollUp,
            KeyCode::PageDown => Action::ScrollDown,
            KeyCode::Home if ctrl => Action::ScrollTop,
            KeyCode::End if ctrl => Action::ScrollBottom,
            KeyCode::Char('t') if ctrl => Action::ToggleThinking,
            KeyCode::Char('o') if ctrl => Action::ToggleTools,
            KeyCode::Char('s') if alt => Action::CycleSendClass,
            KeyCode::Char('v') if ctrl => Action::PasteImage,
            KeyCode::Char('l') if ctrl => Action::Redraw,
            _ => Action::Edit,
        },
    }
}

/// One thing Tab can complete to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Suggestion {
    pub(crate) value: String,
    pub(crate) description: Option<String>,
    /// Whether a space follows, so the user can move on once a value is chosen.
    pub(crate) append_whitespace: bool,
    /// The byte range of the line the value replaces.
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// `/mcp` first-argument keywords, mirroring the grammar the parser accepts.
pub(crate) const MCP_SUBCOMMANDS: [&str; 4] = ["list", "reconnect", "login", "logout"];

/// Permission levels in canonical order, so the completions cannot drift from what the parser
/// accepts.
const PERMISSION_LEVELS: [crate::permission::Permission; 4] = [
    crate::permission::Permission::None,
    crate::permission::Permission::Read,
    crate::permission::Permission::Workspace,
    crate::permission::Permission::Unrestricted,
];

/// What Tab completes from. Snapshots rather than lookups, because completion runs on every
/// keystroke while the list is open, and a skill walk reads every `SKILL.md`.
pub(crate) struct Completer {
    pub(crate) mcp_server_names: Vec<String>,
    /// Refreshed before each prompt by the screen, since `[skills] agent_managed` lets the agent
    /// change the set mid-session.
    pub(crate) skill_names: Arc<std::sync::RwLock<Vec<String>>>,
    pub(crate) profile_names: Vec<String>,
    pub(crate) cwd: crate::workspace::SharedCwd,
}

impl Completer {
    pub(crate) fn suggestions(&self, line: &str, pos: usize) -> Vec<Suggestion> {
        let Some(after_slash) = line.strip_prefix('/') else {
            return Vec::new();
        };
        let before_cursor = line.get(..pos).unwrap_or(line);

        if !before_cursor.contains(char::is_whitespace) {
            // Still in the command word: complete command names. Aliases are not offered, since
            // `/exit` and `/quit` side by side would be noise.
            let typed = line.get(1..pos).unwrap_or("");
            return crate::host::COMMANDS
                .iter()
                .filter(|command| command.name.starts_with(typed))
                .map(|command| Suggestion {
                    value: format!("/{}", command.name),
                    description: Some(command.help.to_string()),
                    append_whitespace: !command.arg_hint.is_empty(),
                    start: 0,
                    end: pos,
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

/// Suggestions for a single-token argument, prefix-filtered, each followed by a space.
fn terminal_suggestions(
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
            description: None,
            append_whitespace: true,
            start: token_start,
            end: pos,
        })
        .collect()
}

/// Complete a `/cd` argument to matching subdirectories. Only directories are offered, and each
/// value ends in `/` so Tab can keep drilling into nested directories.
fn complete_cd_path(
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
        let Some(expanded) = crate::paths::expand_user_path(parent_portion) else {
            return Vec::new();
        };
        crate::workspace::resolve_against_cwd(cwd, expanded)
    };
    let Ok(entries) = std::fs::read_dir(&scan_dir) else {
        return Vec::new();
    };
    let mut suggestions = Vec::new();
    for entry in entries.flatten() {
        if !entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        // Hidden unless the user has started typing a dot, as a shell completes.
        if name.starts_with('.') && !partial.starts_with('.') {
            continue;
        }
        if !name.starts_with(partial) {
            continue;
        }
        suggestions.push(Suggestion {
            value: format!("{parent_portion}{name}/"),
            description: None,
            append_whitespace: false,
            start: token_start,
            end: pos,
        });
    }
    suggestions.sort_by(|left, right| left.value.cmp(&right.value));
    suggestions
}

/// A walk through the history with Up and Down: what was being typed when it began, and where it
/// is.
struct Walk {
    draft: String,
    /// Oldest first, as the store hands them back.
    entries: Vec<String>,
    /// Index into `entries`, or its length for the draft.
    index: usize,
}

/// The history search Ctrl+R opens.
pub(crate) struct Search {
    pub(crate) query: String,
    /// Newest first.
    pub(crate) matches: Vec<String>,
    pub(crate) picked: usize,
}

impl Search {
    pub(crate) fn picked_match(&self) -> Option<&str> {
        self.matches.get(self.picked).map(String::as_str)
    }
}

/// The completion list, open while the line has something to complete.
pub(crate) struct Completion {
    pub(crate) rows: Vec<Suggestion>,
    pub(crate) picked: usize,
}

/// An image attached to the composer, sent with the next prompt.
pub(crate) struct Attachment {
    pub(crate) source: crate::image::ImageSource,
    /// What the chip says: the file's name, or the pixel size of a clipboard image.
    pub(crate) label: String,
}

/// The text the user is typing, with its history and the lists that open over it.
pub(crate) struct Composer {
    area: TextArea<'static>,
    /// Images to send with the next prompt, shown as chips above the text.
    attachments: Vec<Attachment>,
    /// Whether what is typed is a shell command rather than a prompt: entered by typing `!`
    /// into an empty composer, which the mode holds in place of the character, and left by
    /// Backspace on the empty composer. The `!` is the text's again in [`Self::text`], so the
    /// command runs, is remembered and is recalled as `!cmd`.
    shell_mode: bool,
    history: Option<HistoryStore>,
    walk: Option<Walk>,
    pub(crate) search: Option<Search>,
    pub(crate) completion: Option<Completion>,
    /// The first line as it stood when Esc closed the list, which stays closed until the line
    /// changes: a list that came straight back would make Esc mean nothing.
    dismissed: Option<String>,
    /// What the empty composer says, drawn here rather than by the text area, which puts its
    /// cursor cell ahead of its placeholder and so starts the words one column late.
    placeholder: String,
}

impl Composer {
    pub(crate) fn new(history: Option<HistoryStore>) -> Self {
        let mut area = TextArea::default();
        area.set_wrap_mode(WrapMode::WordOrGlyph);
        area.set_cursor_line_style(Style::default());
        area.set_style(Style::default().bg(COMPOSER_BACKGROUND));
        // Tab completes rather than indents.
        area.set_tab_length(0);
        Self {
            area,
            attachments: Vec::new(),
            shell_mode: false,
            history,
            walk: None,
            search: None,
            completion: None,
            dismissed: None,
            placeholder: String::new(),
        }
    }

    pub(crate) fn attach(&mut self, attachment: Attachment) {
        self.attachments.push(attachment);
    }

    pub(crate) fn has_attachments(&self) -> bool {
        !self.attachments.is_empty()
    }

    /// The attachments, for the prompt about to be sent; the composer holds none afterward.
    pub(crate) fn take_attachments(&mut self) -> Vec<Attachment> {
        std::mem::take(&mut self.attachments)
    }

    /// The row the chips take above the text, when there are any.
    pub(crate) fn attachment_rows(&self) -> u16 {
        u16::from(self.has_attachments())
    }

    /// What the composer holds, the shell mode's `!` in front of it.
    pub(crate) fn text(&self) -> String {
        let text = self.area.lines().join("\n");
        if self.shell_mode {
            format!("!{text}")
        } else {
            text
        }
    }

    pub(crate) fn is_shell_mode(&self) -> bool {
        self.shell_mode
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.area.lines().iter().all(|line| line.is_empty())
    }

    pub(crate) fn state(&self) -> ComposerState {
        let ratatui_textarea::DataCursor(row, _) = self.area.cursor();
        ComposerState {
            empty: self.is_empty(),
            at_first_line: row == 0,
            at_last_line: row + 1 >= self.area.lines().len(),
            cursor: false,
            busy: false,
        }
    }

    pub(crate) fn focus(&self) -> Focus {
        if self.search.is_some() {
            Focus::Search
        } else if self.completion.is_some() {
            Focus::Completion
        } else {
            Focus::Composer
        }
    }

    pub(crate) fn set_text(&mut self, text: &str) {
        // A recalled `!cmd` comes back in the mode it was typed in.
        let (shell_mode, text) = match text.strip_prefix('!') {
            Some(command) => (true, command),
            None => (false, text),
        };
        self.shell_mode = shell_mode;
        self.area = {
            let mut fresh = TextArea::from(text.lines().map(str::to_string));
            fresh.set_wrap_mode(self.area.wrap_mode());
            fresh.set_cursor_line_style(self.area.cursor_line_style());
            fresh.set_style(self.area.style());
            fresh.set_tab_length(0);
            fresh.move_cursor(ratatui_textarea::CursorMove::Bottom);
            fresh.move_cursor(ratatui_textarea::CursorMove::End);
            fresh
        };
        self.completion = None;
    }

    /// What the empty composer says: nothing while the agent waits for a prompt, and while it
    /// works, what Enter will do with the next message.
    pub(crate) fn set_placeholder(&mut self, text: &str) {
        if self.placeholder != text {
            self.placeholder = text.to_string();
        }
    }

    pub(crate) fn clear(&mut self) {
        self.set_text("");
        self.attachments.clear();
        self.walk = None;
        self.dismissed = None;
    }

    /// Give a key to the text area, under Emacs bindings.
    ///
    /// Bound here rather than left to the text area's own table, which puts undo on Ctrl+U and
    /// kill-to-start on Ctrl+J: the composer is a prompt line, and a prompt line answers to the
    /// keys every shell does.
    pub(crate) fn edit(&mut self, key: KeyEvent) {
        let before = self.area.lines().to_vec();
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        // `!` into an empty composer is the shell mode, not a character; Backspace with nothing
        // typed leaves the mode first, then takes the last chip, as it would the last character.
        if self.is_empty() && !ctrl && !alt {
            if key.code == KeyCode::Char('!') && !self.shell_mode {
                self.shell_mode = true;
                self.walk = None;
                return;
            }
            if key.code == KeyCode::Backspace {
                if self.shell_mode {
                    self.shell_mode = false;
                    self.walk = None;
                    return;
                }
                if self.attachments.pop().is_some() {
                    self.walk = None;
                    return;
                }
            }
        }
        match (key.code, ctrl, alt) {
            (KeyCode::Char('a'), true, _) | (KeyCode::Home, ..) => {
                self.area.move_cursor(CursorMove::Head);
            }
            (KeyCode::Char('e'), true, _) | (KeyCode::End, ..) => {
                self.area.move_cursor(CursorMove::End);
            }
            (KeyCode::Char('f'), false, true) | (KeyCode::Right, true, _) => {
                self.area.move_cursor(CursorMove::WordForward);
            }
            (KeyCode::Char('b'), false, true) | (KeyCode::Left, true, _) => {
                self.area.move_cursor(CursorMove::WordBack);
            }
            (KeyCode::Char('f'), true, _) | (KeyCode::Right, ..) => {
                self.area.move_cursor(CursorMove::Forward);
            }
            (KeyCode::Char('b'), true, _) | (KeyCode::Left, ..) => {
                self.area.move_cursor(CursorMove::Back);
            }
            (KeyCode::Char('p'), true, _) | (KeyCode::Up, ..) => {
                self.area.move_cursor(CursorMove::Up);
            }
            (KeyCode::Char('n'), true, _) | (KeyCode::Down, ..) => {
                self.area.move_cursor(CursorMove::Down);
            }
            (KeyCode::Char('k'), true, _) => {
                self.area.delete_line_by_end();
            }
            (KeyCode::Char('u'), true, _) => {
                self.area.delete_line_by_head();
            }
            (KeyCode::Char('w'), true, _) | (KeyCode::Backspace, _, true) => {
                self.area.delete_word();
            }
            (KeyCode::Char('d'), false, true) => {
                self.area.delete_next_word();
            }
            (KeyCode::Char('d'), true, _) | (KeyCode::Delete, ..) => {
                self.area.delete_next_char();
            }
            (KeyCode::Char('h'), true, _) | (KeyCode::Backspace, ..) => {
                self.area.delete_char();
            }
            (KeyCode::Char('y'), true, _) => {
                self.area.paste();
            }
            (KeyCode::Char('_'), true, _) | (KeyCode::Char('z'), true, _) => {
                self.area.undo();
            }
            (KeyCode::Char(character), false, false) => self.area.insert_char(character),
            _ => {}
        }
        // A key that only moved the cursor keeps the walk: Up inside a recalled entry of several
        // lines is a step within it, and the Up that reaches its first line goes on to the entry
        // before, not back to the newest.
        if self.area.lines() != before.as_slice() {
            self.walk = None;
        }
    }

    pub(crate) fn newline(&mut self) {
        self.area.insert_newline();
    }

    pub(crate) fn insert(&mut self, text: &str) {
        self.walk = None;
        self.area.insert_str(text);
    }

    /// Rows the composer wants, between one and `most`, the chips' row included.
    pub(crate) fn height(&self, width: u16, most: u16) -> u16 {
        let width = width.max(1) as usize;
        let rows: usize = self
            .area
            .lines()
            .iter()
            .map(|line| wrapped_rows(line, width))
            .sum();
        (rows.max(1) as u16 + self.attachment_rows()).min(most.max(1))
    }

    pub(crate) fn draw(&self, frame: &mut Frame<'_>, area: Rect) {
        // The block spans the whole row, typed or not: the text area paints only the cells it
        // has text for.
        frame.render_widget(
            ratatui::widgets::Block::new().style(Style::default().bg(COMPOSER_BACKGROUND)),
            area,
        );
        let area = if self.has_attachments() && area.height > 0 {
            let dim = Style::default().fg(Color::DarkGray).bg(COMPOSER_BACKGROUND);
            let chips: Vec<ratatui::text::Span<'static>> = self
                .attachments
                .iter()
                .enumerate()
                .map(|(index, attachment)| {
                    ratatui::text::Span::styled(
                        format!("[image {}  {}]  ", index + 1, attachment.label),
                        dim,
                    )
                })
                .collect();
            frame.render_widget(
                ratatui::widgets::Paragraph::new(ratatui::text::Line::from(chips)),
                Rect { height: 1, ..area },
            );
            Rect {
                y: area.y + 1,
                height: area.height - 1,
                ..area
            }
        } else {
            area
        };
        if area.height == 0 {
            return;
        }
        if self.is_empty() && !self.placeholder.is_empty() {
            // The words start where typed text starts, with the cursor on their first character
            // as it would be on a typed one.
            let mut characters = self.placeholder.chars();
            let first = characters.next().map(String::from).unwrap_or_default();
            let rest: String = characters.collect();
            let dim = Style::default().fg(Color::DarkGray).bg(COMPOSER_BACKGROUND);
            frame.render_widget(
                ratatui::widgets::Paragraph::new(ratatui::text::Line::from(vec![
                    ratatui::text::Span::styled(first, dim.add_modifier(Modifier::REVERSED)),
                    ratatui::text::Span::styled(rest, dim),
                ])),
                area,
            );
            return;
        }
        frame.render_widget(&self.area, area);
    }

    /// Store what was sent, so Up recalls it next time.
    pub(crate) fn remember(&mut self, text: &str) {
        self.walk = None;
        let Some(history) = &self.history else {
            return;
        };
        // The newest entry is not repeated, so an answer sent twice is one row to walk past.
        if history.latest().ok().flatten().as_deref() == Some(text) {
            return;
        }
        if let Err(error) = history.append(text) {
            tracing::warn!("failed to record the prompt in the input history: {error}");
        }
    }

    fn begin_walk(&mut self) -> bool {
        if self.walk.is_some() {
            return true;
        }
        let Some(history) = &self.history else {
            return false;
        };
        let entries = match history.recent(HISTORY_WALK_ENTRIES) {
            Ok(entries) => entries,
            Err(error) => {
                tracing::warn!("failed to read the input history: {error}");
                return false;
            }
        };
        if entries.is_empty() {
            return false;
        }
        let index = entries.len();
        self.walk = Some(Walk {
            draft: self.text(),
            entries,
            index,
        });
        true
    }

    /// Up at the top of the composer: the entry before the one shown.
    pub(crate) fn history_previous(&mut self) {
        if !self.begin_walk() {
            return;
        }
        let Some(walk) = self.walk.as_mut() else {
            return;
        };
        if walk.index == 0 {
            return;
        }
        walk.index -= 1;
        let text = walk.entries.get(walk.index).cloned().unwrap_or_default();
        self.set_text(&text);
    }

    /// Down at the bottom of the composer: the entry after the one shown, or the draft.
    pub(crate) fn history_next(&mut self) {
        let Some(walk) = self.walk.as_mut() else {
            return;
        };
        if walk.index >= walk.entries.len() {
            return;
        }
        walk.index += 1;
        let text = if walk.index == walk.entries.len() {
            std::mem::take(&mut walk.draft)
        } else {
            walk.entries.get(walk.index).cloned().unwrap_or_default()
        };
        let finished = walk.index == walk.entries.len();
        self.set_text(&text);
        if finished {
            self.walk = None;
        }
    }

    /// Ctrl+R: open the search, or step to the next match of the query.
    pub(crate) fn history_search(&mut self) {
        match self.search.as_mut() {
            Some(search) => {
                if !search.matches.is_empty() {
                    search.picked = (search.picked + 1) % search.matches.len();
                }
            }
            None => {
                self.search = Some(Search {
                    query: String::new(),
                    matches: Vec::new(),
                    picked: 0,
                });
                self.refresh_search();
            }
        }
    }

    /// A key typed into the search's query.
    pub(crate) fn edit_search(&mut self, key: KeyEvent) {
        let Some(search) = self.search.as_mut() else {
            return;
        };
        match key.code {
            KeyCode::Backspace => {
                search.query.pop();
            }
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                search.query.push(character);
            }
            _ => return,
        }
        self.refresh_search();
    }

    fn refresh_search(&mut self) {
        let Some(search) = self.search.as_mut() else {
            return;
        };
        let Some(history) = &self.history else {
            search.matches.clear();
            return;
        };
        let filter = HistoryFilter {
            command_line: (!search.query.is_empty())
                .then(|| CommandLineMatch::Substring(search.query.clone())),
            ..HistoryFilter::default()
        };
        search.matches = match history.search(
            &filter,
            HistoryOrder::NewestFirst,
            Some(HISTORY_SEARCH_MATCHES),
        ) {
            Ok(rows) => rows.into_iter().map(|(_, line)| line).collect(),
            Err(error) => {
                tracing::warn!("failed to search the input history: {error}");
                Vec::new()
            }
        };
        search.picked = 0;
    }

    pub(crate) fn pick_previous(&mut self) {
        if let Some(search) = self.search.as_mut() {
            if search.picked > 0 {
                search.picked -= 1;
            }
        } else if let Some(completion) = self.completion.as_mut() {
            if completion.picked > 0 {
                completion.picked -= 1;
            } else {
                completion.picked = completion.rows.len().saturating_sub(1);
            }
        }
    }

    pub(crate) fn pick_next(&mut self) {
        if let Some(search) = self.search.as_mut() {
            if search.picked + 1 < search.matches.len() {
                search.picked += 1;
            }
        } else if let Some(completion) = self.completion.as_mut()
            && !completion.rows.is_empty()
        {
            completion.picked = (completion.picked + 1) % completion.rows.len();
        }
    }

    /// Enter in the search: the picked match becomes the composer's text.
    pub(crate) fn accept_search(&mut self) {
        if let Some(search) = self.search.take()
            && let Some(found) = search.picked_match()
        {
            let text = found.to_string();
            self.set_text(&text);
        }
    }

    pub(crate) fn close_search(&mut self) {
        self.search = None;
    }

    /// The first line and the byte offset of the cursor in it, when the cursor is there. The
    /// list completes the command line alone.
    fn line_at_cursor(&self) -> Option<(String, usize)> {
        let ratatui_textarea::DataCursor(row, column) = self.area.cursor();
        if row != 0 {
            return None;
        }
        let line = self.area.lines().first().cloned().unwrap_or_default();
        let pos = line
            .char_indices()
            .nth(column)
            .map_or(line.len(), |(index, _)| index);
        Some((line, pos))
    }

    /// Open, narrow or close the list for the line as it now stands: there as soon as `/` is
    /// typed, gone when nothing fits. The pick follows its value, so narrowing keeps what the
    /// user moved to.
    pub(crate) fn suggest(&mut self, completer: &Completer) {
        let Some((line, pos)) = self.line_at_cursor() else {
            self.completion = None;
            return;
        };
        if self.dismissed.as_deref() == Some(line.as_str()) {
            self.completion = None;
            return;
        }
        self.dismissed = None;
        let rows = completer.suggestions(&line, pos);
        if rows.is_empty() {
            self.completion = None;
            return;
        }
        let picked = self
            .completion
            .as_ref()
            .and_then(|open| open.rows.get(open.picked))
            .and_then(|was| rows.iter().position(|row| row.value == was.value))
            .unwrap_or(0);
        self.completion = Some(Completion { rows, picked });
    }

    /// Tab: fill in the picked row and offer what comes next, or open the list again after Esc
    /// closed it.
    pub(crate) fn complete(&mut self, completer: &Completer) {
        if self.completion.is_none() {
            self.dismissed = None;
            self.suggest(completer);
        }
        self.accept_completion();
        self.suggest(completer);
    }

    /// Enter on the list: the picked row replaces the token it completes.
    pub(crate) fn accept_completion(&mut self) {
        let Some(completion) = self.completion.take() else {
            return;
        };
        let Some(picked) = completion.rows.get(completion.picked) else {
            return;
        };
        let line = self.area.lines().first().cloned().unwrap_or_default();
        let start = picked.start.min(line.len());
        let end = picked.end.min(line.len()).max(start);
        let mut rebuilt = String::new();
        rebuilt.push_str(&line[..start]);
        rebuilt.push_str(&picked.value);
        if picked.append_whitespace && !line[end..].starts_with(' ') {
            rebuilt.push(' ');
        }
        rebuilt.push_str(&line[end..]);
        let rest: Vec<String> = self.area.lines().iter().skip(1).cloned().collect();
        let mut text = rebuilt;
        for line in rest {
            text.push('\n');
            text.push_str(&line);
        }
        self.set_text(&text);
    }

    /// Esc: close the list until the line changes.
    pub(crate) fn close_completion(&mut self) {
        self.completion = None;
        self.dismissed = self.line_at_cursor().map(|(line, _)| line);
    }
}

/// The directory as the status row shows it, with the home directory as `~`.
pub(crate) fn shorten_path_with_tilde(path: &std::path::Path) -> String {
    // Resolved once: the status row asks every frame, and the home does not move under a run.
    static HOMES: std::sync::OnceLock<Option<HomeSpellings>> = std::sync::OnceLock::new();
    let homes = HOMES.get_or_init(|| dirs::home_dir().map(HomeSpellings::of));
    shorten_under_home(path, homes.as_ref())
}

/// The home directory as given and, where it differs, in its canonical spelling.
struct HomeSpellings {
    given: PathBuf,
    canonical: Option<PathBuf>,
}

impl HomeSpellings {
    fn of(home: PathBuf) -> Self {
        let canonical = std::fs::canonicalize(&home)
            .ok()
            .filter(|canonical| *canonical != home);
        Self {
            given: home,
            canonical,
        }
    }
}

/// [`shorten_path_with_tilde`] against the home as given and in its canonical spelling, since a
/// session's directory is recorded canonical and a home reached through a link (macOS keeps
/// `/var` under `/private`) would otherwise never shorten.
fn shorten_under_home(path: &std::path::Path, homes: Option<&HomeSpellings>) -> String {
    let Some(homes) = homes else {
        return path.display().to_string();
    };
    for home in std::iter::once(&homes.given).chain(homes.canonical.as_ref()) {
        if path == home {
            return "~".to_string();
        }
        if let Ok(relative) = path.strip_prefix(home) {
            // Forward slashes on every platform, so the tilde form reads the same way everywhere.
            let relative = relative.display().to_string().replace('\\', "/");
            return format!("~/{relative}");
        }
    }
    path.display().to_string()
}

/// Resolve what a `/cd` argument names: the launch directory when it is empty, otherwise whatever
/// [`crate::paths::expand_user_path`] makes of it. `None` only when a tilde needs the home
/// directory and it cannot be determined.
///
/// A bare `/cd` returns to the directory meka was started in, where a shell's `cd` would go home.
/// A resumed session opens in the directory it recorded, so "take me back to my shell" is the
/// move a user wants here, and at `workspace` the working directory is the writable boundary,
/// which makes `$HOME` the widest possible default. `~` still spells home.
fn expand_cd_target(launch_cwd: &std::path::Path, target: &str) -> Option<PathBuf> {
    if target.is_empty() {
        return Some(launch_cwd.to_path_buf());
    }
    crate::paths::expand_user_path(target)
}

/// Move the session's working directory, returning where it landed or the failure to report.
pub(crate) fn handle_cd(
    cwd: &crate::workspace::SharedCwd,
    launch_cwd: &std::path::Path,
    target: &str,
) -> Result<PathBuf, String> {
    let Some(raw) = expand_cd_target(launch_cwd, target) else {
        return Err("cd: failed to determine the home directory".to_string());
    };
    // Relative inputs resolve against the session's own directory, so `/cd subdir` lands inside
    // the agent's current view; the acceptor then puts the result in the one spelling every door
    // records.
    let resolved = crate::workspace::resolve_against_cwd(cwd, &raw);
    let canonical =
        crate::workspace::accept_cwd(&resolved).map_err(|error| format!("cd: {error}"))?;
    cwd.set(canonical.clone());
    Ok(canonical)
}

/// Rows `line` takes at `width` under the composer's wrap: a row breaks at a word boundary
/// where the next word does not fit, and a word wider than a row breaks by grapheme. The text
/// area's own rule, restated so the composer asks for the rows it is about to draw.
fn wrapped_rows(line: &str, width: usize) -> usize {
    use unicode_segmentation::UnicodeSegmentation as _;
    let width = width.max(1);
    let mut rows = 0;
    let mut row_open = false;
    let mut row_width = 0;
    for word in line.split_word_bounds() {
        let word_width = crate::text::display_width(word);
        if row_open && row_width + word_width > width {
            rows += 1;
            row_open = false;
            row_width = 0;
        }
        if word_width <= width {
            row_open = true;
            row_width += word_width;
            continue;
        }
        let mut used = 0;
        for grapheme in word.graphemes(true) {
            let grapheme_width = crate::text::display_width(grapheme);
            if used > 0 && used + grapheme_width > width {
                rows += 1;
                used = 0;
            }
            used += grapheme_width;
        }
        rows += usize::from(used > 0);
    }
    (rows + usize::from(row_open)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn idle() -> ComposerState {
        ComposerState {
            empty: true,
            at_first_line: true,
            at_last_line: true,
            cursor: false,
            busy: false,
        }
    }

    /// Esc is the interrupt while the host is busy and the way back when it is not; Ctrl+C is the
    /// interrupt either way; a panel in front keeps Esc, since closing it is the panel's answer.
    #[test]
    fn esc_interrupts_while_busy_and_backs_out_when_idle() {
        let busy = ComposerState {
            busy: true,
            ..idle()
        };
        assert_eq!(
            action_for(
                &key(KeyCode::Esc, KeyModifiers::NONE),
                Focus::Composer,
                busy
            ),
            Action::Interrupt
        );
        assert_eq!(
            action_for(
                &key(KeyCode::Esc, KeyModifiers::NONE),
                Focus::Composer,
                idle()
            ),
            Action::Escape
        );
        assert_eq!(
            action_for(
                &key(KeyCode::Char('c'), KeyModifiers::CONTROL),
                Focus::Composer,
                busy
            ),
            Action::Interrupt
        );
        assert_eq!(
            action_for(&key(KeyCode::Esc, KeyModifiers::NONE), Focus::Panel, busy),
            Action::Escape
        );
    }

    /// The keys whose meaning the docs promise, in the composer.
    #[test]
    fn the_composer_keys_mean_what_the_docs_say() {
        let cases = [
            (KeyCode::Enter, KeyModifiers::NONE, Action::Submit),
            (KeyCode::Enter, KeyModifiers::ALT, Action::Newline),
            (KeyCode::Enter, KeyModifiers::SHIFT, Action::Newline),
            (
                KeyCode::BackTab,
                KeyModifiers::SHIFT,
                Action::CyclePermission,
            ),
            (KeyCode::Char('c'), KeyModifiers::CONTROL, Action::Interrupt),
            (KeyCode::Char('d'), KeyModifiers::CONTROL, Action::Leave),
            (
                KeyCode::Char('r'),
                KeyModifiers::CONTROL,
                Action::HistorySearch,
            ),
            (KeyCode::Tab, KeyModifiers::NONE, Action::Complete),
            (KeyCode::PageUp, KeyModifiers::NONE, Action::ScrollUp),
            (KeyCode::PageDown, KeyModifiers::NONE, Action::ScrollDown),
            (KeyCode::Home, KeyModifiers::CONTROL, Action::ScrollTop),
            (KeyCode::End, KeyModifiers::CONTROL, Action::ScrollBottom),
            (KeyCode::Up, KeyModifiers::ALT, Action::CursorPrevious),
            (KeyCode::Down, KeyModifiers::ALT, Action::CursorNext),
            (KeyCode::Up, KeyModifiers::NONE, Action::HistoryPrevious),
            (KeyCode::Down, KeyModifiers::NONE, Action::HistoryNext),
            (
                KeyCode::Char('t'),
                KeyModifiers::CONTROL,
                Action::ToggleThinking,
            ),
            (
                KeyCode::Char('o'),
                KeyModifiers::CONTROL,
                Action::ToggleTools,
            ),
            (
                KeyCode::Char('s'),
                KeyModifiers::ALT,
                Action::CycleSendClass,
            ),
            (KeyCode::Char('l'), KeyModifiers::CONTROL, Action::Redraw),
            (KeyCode::Char('x'), KeyModifiers::NONE, Action::Edit),
        ];
        for (code, modifiers, expected) in cases {
            assert_eq!(
                action_for(&key(code, modifiers), Focus::Composer, idle()),
                expected,
                "{code:?} with {modifiers:?}"
            );
        }
    }

    /// Ctrl+D leaves only on an empty composer; with text it is the editing key it always was.
    /// Up and Down recall history only at the composer's edges.
    #[test]
    fn the_keys_that_depend_on_the_composer_read_it() {
        let typing = ComposerState {
            empty: false,
            at_first_line: false,
            at_last_line: false,
            cursor: false,
            busy: false,
        };
        assert_eq!(
            action_for(
                &key(KeyCode::Char('d'), KeyModifiers::CONTROL),
                Focus::Composer,
                typing
            ),
            Action::Edit
        );
        assert_eq!(
            action_for(
                &key(KeyCode::Up, KeyModifiers::NONE),
                Focus::Composer,
                typing
            ),
            Action::Edit
        );
        let picked = ComposerState {
            cursor: true,
            ..idle()
        };
        assert_eq!(
            action_for(
                &key(KeyCode::Enter, KeyModifiers::NONE),
                Focus::Composer,
                picked
            ),
            Action::ToggleItem
        );
        assert_eq!(
            action_for(
                &key(KeyCode::Char(' '), KeyModifiers::NONE),
                Focus::Composer,
                picked
            ),
            Action::ToggleItem
        );
    }

    /// A question takes Enter, Esc and Ctrl+C and nothing else from the table.
    #[test]
    fn a_question_answers_to_three_keys() {
        assert_eq!(
            action_for(
                &key(KeyCode::Enter, KeyModifiers::NONE),
                Focus::Question,
                idle()
            ),
            Action::Submit
        );
        assert_eq!(
            action_for(
                &key(KeyCode::Esc, KeyModifiers::NONE),
                Focus::Question,
                idle()
            ),
            Action::Escape
        );
        assert_eq!(
            action_for(
                &key(KeyCode::Char('c'), KeyModifiers::CONTROL),
                Focus::Question,
                idle()
            ),
            Action::Interrupt
        );
        assert_eq!(
            action_for(
                &key(KeyCode::Tab, KeyModifiers::NONE),
                Focus::Question,
                idle()
            ),
            Action::Edit
        );
    }

    fn completer() -> Completer {
        Completer {
            mcp_server_names: vec!["exa".into(), "notion".into()],
            skill_names: Arc::new(std::sync::RwLock::new(vec!["review".into()])),
            profile_names: vec!["work".into(), "play".into()],
            cwd: crate::workspace::SharedCwd::new(std::env::temp_dir()),
        }
    }

    #[test]
    fn a_command_word_completes_from_the_table() {
        let rows = completer().suggestions("/he", 3);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].value, "/help");
        assert_eq!((rows[0].start, rows[0].end), (0, 3));
        assert!(!rows[0].append_whitespace, "help takes no argument");
    }

    #[test]
    fn an_argument_completes_from_what_the_command_takes() {
        let completer = completer();
        let levels: Vec<String> = completer
            .suggestions("/permission w", 13)
            .into_iter()
            .map(|row| row.value)
            .collect();
        assert_eq!(levels, vec!["workspace"]);
        let servers: Vec<String> = completer
            .suggestions("/mcp login ", 11)
            .into_iter()
            .map(|row| row.value)
            .collect();
        assert_eq!(servers, vec!["exa", "notion"]);
        let skills: Vec<String> = completer
            .suggestions("/skill r", 8)
            .into_iter()
            .map(|row| row.value)
            .collect();
        assert_eq!(skills, vec!["review"]);
        assert!(completer.suggestions("/mcp list ", 10).is_empty());
    }

    #[test]
    fn cd_completes_directories_only_and_hides_dotfiles() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(temp.path().join("docs")).expect("dir");
        std::fs::create_dir(temp.path().join(".git")).expect("dir");
        std::fs::write(temp.path().join("notes.txt"), b"x").expect("file");
        let completer = Completer {
            cwd: crate::workspace::SharedCwd::new(temp.path().to_path_buf()),
            ..completer()
        };
        let rows: Vec<String> = completer
            .suggestions("/cd ", 4)
            .into_iter()
            .map(|row| row.value)
            .collect();
        assert_eq!(rows, vec!["docs/"]);
        let hidden: Vec<String> = completer
            .suggestions("/cd .", 5)
            .into_iter()
            .map(|row| row.value)
            .collect();
        assert_eq!(hidden, vec![".git/"]);
    }

    /// The list opens as `/` is typed, narrows with every key, keeps the pick by its value, and
    /// once Esc has closed it stays closed until the line changes.
    #[test]
    fn the_list_follows_the_line_and_esc_closes_it_until_the_line_changes() {
        let completer = completer();
        let mut composer = Composer::new(None);
        composer.edit(key(KeyCode::Char('/'), KeyModifiers::NONE));
        composer.suggest(&completer);
        let open = composer
            .completion
            .as_ref()
            .expect("the list opens on the slash");
        assert_eq!(open.rows.len(), crate::host::COMMANDS.len());
        composer.pick_next();
        let moved_to = composer.completion.as_ref().unwrap().rows[1].value.clone();
        let narrowing = moved_to.chars().nth(1).unwrap();
        composer.edit(key(KeyCode::Char(narrowing), KeyModifiers::NONE));
        composer.suggest(&completer);
        let open = composer
            .completion
            .as_ref()
            .expect("still something to complete");
        assert_eq!(
            open.rows[open.picked].value, moved_to,
            "the pick follows its value through a narrowing"
        );
        composer.close_completion();
        composer.suggest(&completer);
        assert!(
            composer.completion.is_none(),
            "closed stays closed while the line stands"
        );
        composer.edit(key(KeyCode::Backspace, KeyModifiers::NONE));
        composer.suggest(&completer);
        assert!(
            composer.completion.is_some(),
            "a changed line opens the list again"
        );
        composer.set_text("/nothing-like-this");
        composer.suggest(&completer);
        assert!(composer.completion.is_none(), "nothing fits, no list");
    }

    fn attachment(label: &str) -> Attachment {
        Attachment {
            source: crate::image::ImageSource::Base64 {
                media_type: "image/png".into(),
                data: String::new(),
            },
            label: label.into(),
        }
    }

    /// `!` into an empty composer enters the shell mode, which holds the `!` for the text;
    /// Backspace on the empty composer leaves it; a recalled `!cmd` comes back in the mode.
    #[test]
    fn a_leading_bang_is_the_shell_mode() {
        let mut composer = Composer::new(None);
        composer.edit(key(KeyCode::Char('!'), KeyModifiers::NONE));
        assert!(composer.is_shell_mode());
        assert_eq!(composer.text(), "!");
        assert!(composer.is_empty(), "the mode holds no text yet");
        composer.edit(key(KeyCode::Char('l'), KeyModifiers::NONE));
        composer.edit(key(KeyCode::Char('s'), KeyModifiers::NONE));
        assert_eq!(composer.text(), "!ls");
        composer.edit(key(KeyCode::Backspace, KeyModifiers::NONE));
        composer.edit(key(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(composer.is_shell_mode(), "the text goes first");
        composer.edit(key(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(!composer.is_shell_mode(), "then the mode");
        assert_eq!(composer.text(), "");
        composer.set_text("!pwd");
        assert!(composer.is_shell_mode());
        assert_eq!(composer.text(), "!pwd");
        composer.clear();
        assert!(!composer.is_shell_mode());
        composer.set_text("hi!");
        composer.edit(key(KeyCode::Char('!'), KeyModifiers::NONE));
        assert_eq!(composer.text(), "hi!!", "a `!` after text is a character");
    }

    /// An attachment takes a row above the text, Backspace on an empty composer drops the last
    /// one and otherwise edits the text, and sending takes them all.
    #[test]
    fn attachments_ride_above_the_text_until_sent() {
        let mut composer = Composer::new(None);
        assert_eq!(composer.height(40, 10), 1);
        composer.attach(attachment("a.png"));
        assert_eq!(composer.height(40, 10), 2, "the chips' row");
        composer.edit(key(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(
            !composer.has_attachments(),
            "backspace with nothing typed drops the chip"
        );
        composer.attach(attachment("b.png"));
        composer.set_text("words");
        composer.edit(key(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(composer.text(), "word");
        assert!(
            composer.has_attachments(),
            "with text, backspace edits the text"
        );
        assert_eq!(composer.take_attachments().len(), 1);
        assert!(!composer.has_attachments());
    }

    /// Accepting a completion replaces the token under the cursor and leaves the rest alone.
    #[test]
    fn accepting_a_completion_replaces_the_token_it_completes() {
        let mut composer = Composer::new(None);
        composer.set_text("/perm");
        composer.complete(&completer());
        assert_eq!(composer.text(), "/permission ");
        composer.set_text("/permission w");
        composer.complete(&completer());
        assert_eq!(composer.text(), "/permission workspace ");
    }

    /// The history table is the store's, so a migrated store has to exist before the history can
    /// be opened on it.
    async fn history_in(temp: &tempfile::TempDir) -> HistoryStore {
        let path = temp.path().join("meka.db");
        crate::store::Store::open(Some(&path), &Default::default())
            .await
            .expect("a migrated store");
        HistoryStore::open(&path, 100).expect("history")
    }

    /// The editing keys are Emacs's: Ctrl+U kills to the start of the line, Ctrl+K to its end,
    /// Ctrl+W the word before the cursor, Ctrl+A and Ctrl+E move to the ends.
    #[test]
    fn the_editing_keys_are_emacs_s() {
        let mut composer = Composer::new(None);
        composer.set_text("one two three");
        composer.edit(key(KeyCode::Char('w'), KeyModifiers::CONTROL));
        assert_eq!(composer.text(), "one two ");
        composer.edit(key(KeyCode::Char('a'), KeyModifiers::CONTROL));
        composer.edit(key(KeyCode::Char('k'), KeyModifiers::CONTROL));
        assert_eq!(composer.text(), "");
        composer.set_text("kill to the start");
        composer.edit(key(KeyCode::Char('b'), KeyModifiers::ALT));
        composer.edit(key(KeyCode::Char('u'), KeyModifiers::CONTROL));
        assert_eq!(composer.text(), "start");
        composer.edit(key(KeyCode::Char('e'), KeyModifiers::CONTROL));
        composer.edit(key(KeyCode::Char('!'), KeyModifiers::SHIFT));
        assert_eq!(composer.text(), "start!");
    }

    /// Moving the cursor inside a recalled entry of several lines does not end the walk: the Up
    /// that reaches its first line goes on to the entry before it.
    #[tokio::test]
    async fn the_history_walk_survives_a_cursor_move_inside_a_recalled_entry() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = history_in(&temp).await;
        let mut composer = Composer::new(Some(store));
        composer.remember("first");
        composer.remember("a\nb");
        composer.history_previous();
        assert_eq!(composer.text(), "a\nb");
        assert!(!composer.state().at_first_line, "recalled at its end");
        composer.edit(key(KeyCode::Up, KeyModifiers::NONE));
        assert!(composer.state().at_first_line);
        composer.history_previous();
        assert_eq!(
            composer.text(),
            "first",
            "the walk went on to the entry before"
        );
        composer.edit(key(KeyCode::Char('!'), KeyModifiers::NONE));
        composer.history_previous();
        assert_eq!(
            composer.text(),
            "a\nb",
            "typing ends the walk, which starts over"
        );
    }

    /// Up walks back through what was sent and Down returns to the draft.
    #[tokio::test]
    async fn the_history_walk_returns_to_the_draft() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = history_in(&temp).await;
        let mut composer = Composer::new(Some(store));
        composer.remember("first");
        composer.remember("second");
        composer.set_text("draft");
        composer.history_previous();
        assert_eq!(composer.text(), "second");
        composer.history_previous();
        assert_eq!(composer.text(), "first");
        composer.history_previous();
        assert_eq!(composer.text(), "first", "the oldest entry holds");
        composer.history_next();
        composer.history_next();
        assert_eq!(composer.text(), "draft");
    }

    #[tokio::test]
    async fn the_search_finds_by_substring_newest_first() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = history_in(&temp).await;
        let mut composer = Composer::new(Some(store));
        composer.remember("list the files");
        composer.remember("count the lines");
        composer.remember("list the users");
        composer.history_search();
        for character in "list".chars() {
            composer.edit_search(key(KeyCode::Char(character), KeyModifiers::NONE));
        }
        let search = composer.search.as_ref().expect("open");
        assert_eq!(search.matches, vec!["list the users", "list the files"]);
        composer.history_search();
        composer.accept_search();
        assert_eq!(composer.text(), "list the files");
    }

    #[test]
    fn cd_moves_the_session_directory_and_not_the_process() {
        let temp = tempfile::tempdir().expect("tempdir");
        let target = crate::workspace::canonical_for_test(temp.path());
        let process_cwd = std::env::current_dir().expect("cwd");
        let cwd = crate::workspace::SharedCwd::new(PathBuf::from("/this/path/does/not/exist"));
        let landed = handle_cd(&cwd, &process_cwd, target.to_str().expect("utf-8"));
        assert_eq!(landed.as_deref(), Ok(target.as_path()));
        assert_eq!(cwd.get(), target);
        assert_eq!(std::env::current_dir().expect("cwd"), process_cwd);
        let file = temp.path().join("not-a-directory");
        std::fs::write(&file, b"x").expect("file");
        assert!(handle_cd(&cwd, &process_cwd, file.to_str().expect("utf-8")).is_err());
        assert_eq!(
            cwd.get(),
            target,
            "a refused move leaves the directory where it was"
        );
    }

    #[test]
    fn the_home_directory_shortens_to_a_tilde() {
        let Some(home) = dirs::home_dir() else {
            return;
        };
        assert_eq!(shorten_path_with_tilde(&home), "~");
        assert_eq!(
            shorten_path_with_tilde(&home.join("projects")),
            "~/projects"
        );
    }

    /// A home reached through a link shortens a directory spelled either way: as recorded, in
    /// the canonical spelling, or through the link itself.
    #[cfg(unix)]
    #[test]
    fn a_linked_home_still_shortens_to_a_tilde() {
        let temp = tempfile::tempdir().expect("tempdir");
        let real = temp.path().join("real");
        std::fs::create_dir_all(real.join("work")).expect("dirs");
        let real = std::fs::canonicalize(&real).expect("canonical");
        let link = temp.path().join("link");
        std::os::unix::fs::symlink(&real, &link).expect("link");
        let homes = HomeSpellings::of(link.clone());
        assert_eq!(
            shorten_under_home(&real.join("work"), Some(&homes)),
            "~/work"
        );
        assert_eq!(
            shorten_under_home(&link.join("work"), Some(&homes)),
            "~/work"
        );
        assert_eq!(shorten_under_home(&real, Some(&homes)), "~");
        assert_eq!(
            shorten_under_home(&temp.path().join("elsewhere"), Some(&homes)),
            temp.path().join("elsewhere").display().to_string()
        );
    }

    /// The composer's height is the rows its word wrap takes, not the columns divided by the
    /// width: a row that ends ahead of a word that does not fit is a row all the same.
    #[test]
    fn the_composer_asks_for_the_rows_its_word_wrap_takes() {
        assert_eq!(wrapped_rows("aaaaaa bbbbbb cccccc", 10), 3);
        assert_eq!(wrapped_rows(&"x".repeat(25), 10), 3);
        assert_eq!(wrapped_rows("short", 10), 1);
        assert_eq!(wrapped_rows("", 10), 1);
    }
}
