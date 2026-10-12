//! The bottom panel: what a slash command shows, or a question asks, in place of the composer and
//! the status row until it is closed. Nothing drawn here enters the transcript, which is what
//! keeps a `/status` from being a row of history every time it is asked.
//!
//! One shape serves two uses: a body to read (`/help`, `/status`, a listing, a refusal) and a
//! body with choices to pick from (`/permission`, `/profile`, the approval prompt). It wears the
//! composer's background, since it stands where the composer stood and is a layer over the
//! transcript, not part of it.

use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    text::{Line, Span, Text},
    widgets::{Clear, Paragraph},
};

use crate::render::tui::{self as draw, TEXT_EDGE, gray};

/// One thing a panel offers to pick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Choice {
    pub(crate) label: String,
    /// The letter that picks it without moving the cursor, shown after the label.
    pub(crate) key: Option<char>,
    /// What it is, shown after the label in place of a key.
    pub(crate) description: Option<String>,
}

/// The panel's state between frames.
pub(crate) struct Panel {
    /// A first row naming what the panel is about, where the user did not type it: the approval
    /// prompt's call.
    heading: Option<Line<'static>>,
    /// Whether the body's rows are set in from the edge when drawn: a text panel's are, where
    /// a picker's body comes as rows drawn the way the transcript draws them.
    body_at_edge: bool,
    /// What has been typed to filter the choices, for a picker that can be filtered; `None` for
    /// one that cannot, where a typed letter or digit is a pick.
    filter: Option<String>,
    /// Whether typed keys go to the filter. A picker that can filter opens browsing, where keys
    /// move and pick as in any panel; `/` starts typing and Esc stops it, keeping the text.
    typing: bool,
    /// Where each choice's description stands.
    facts: super::view::ListLayout,
    /// The choice rows the last frame had room for: the page PgUp and PgDn move the pick by.
    page_rows: usize,
    body: Vec<Line<'static>>,
    /// The body wrapped at the width it was last drawn at.
    wrapped: Option<(usize, Vec<Line<'static>>)>,
    choices: Vec<Choice>,
    picked: usize,
    /// The first body row on screen.
    scroll: usize,
}

impl Panel {
    /// A panel over text as a command printed it, escapes included. An indent the whole body
    /// shares is taken off, since the panel supplies the edge: a report printed for a terminal
    /// indents its rows under a heading the panel does without.
    pub(crate) fn with_text(text: &str) -> Self {
        let text = text.trim_end_matches('\n');
        let shared = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.len() - line.trim_start_matches(' ').len())
            .min()
            .unwrap_or(0);
        let dedented: Vec<&str> = text
            .lines()
            .map(|line| line.get(shared..).unwrap_or(""))
            .collect();
        Self {
            heading: None,
            body: draw::ansi_lines(&dedented.join("\n"), usize::MAX),
            body_at_edge: true,
            filter: None,
            typing: false,
            facts: super::view::ListLayout::Column,
            page_rows: 1,
            wrapped: None,
            choices: Vec::new(),
            picked: 0,
            scroll: 0,
        }
    }

    /// A panel over rows with something to pick under them, the cursor on `picked`.
    pub(crate) fn with_choices(
        body: Vec<Line<'static>>,
        choices: Vec<Choice>,
        picked: usize,
    ) -> Self {
        Self {
            heading: None,
            body,
            body_at_edge: false,
            filter: None,
            typing: false,
            facts: super::view::ListLayout::Column,
            page_rows: 1,
            wrapped: None,
            picked: picked.min(choices.len().saturating_sub(1)),
            choices,
            scroll: 0,
        }
    }

    /// The same panel opened at the end of its body, where the newest of a log is.
    pub(crate) fn scrolled_to_end(mut self) -> Self {
        self.scroll = usize::MAX;
        self
    }

    /// The same panel with a first row naming what it is about.
    pub(crate) fn with_heading(mut self, heading: impl Into<Line<'static>>) -> Self {
        self.heading = Some(heading.into());
        self
    }

    /// The same panel with a filter `/` starts typing into, for a list too long to pick from by
    /// eye.
    pub(crate) fn with_filter(mut self) -> Self {
        self.filter = Some(String::new());
        self
    }

    /// Whether `/` has a filter to start typing into.
    pub(crate) fn can_filter(&self) -> bool {
        self.filter.is_some()
    }

    /// Whether typed keys go to the filter now.
    pub(crate) fn is_typing(&self) -> bool {
        self.typing
    }

    /// `/`: typed keys go to the filter from here, on a panel that has one.
    pub(crate) fn start_filtering(&mut self) {
        if self.can_filter() {
            self.typing = true;
        }
    }

    /// Stop typing; what was typed stays and keeps filtering.
    pub(crate) fn stop_filtering(&mut self) {
        self.typing = false;
    }

    /// The same panel with each choice's description flush with the right edge, for facts read
    /// as a column of their own beside a title of any length.
    pub(crate) fn with_facts_at_the_edge(mut self) -> Self {
        self.facts = super::view::ListLayout::Edge;
        self
    }

    /// Whether there is something to pick, as against a body only read.
    pub(crate) fn has_choices(&self) -> bool {
        !self.choices.is_empty()
    }

    /// The choices the typed text lets through, as their places among all of them; every choice
    /// where nothing filters.
    fn shown(&self) -> Vec<usize> {
        match self.filter.as_deref().filter(|filter| !filter.is_empty()) {
            None => (0..self.choices.len()).collect(),
            Some(filter) => {
                let filter = filter.to_lowercase();
                self.choices
                    .iter()
                    .enumerate()
                    .filter(|(_, choice)| {
                        choice.label.to_lowercase().contains(&filter)
                            || choice.description.as_deref().is_some_and(|description| {
                                description.to_lowercase().contains(&filter)
                            })
                    })
                    .map(|(index, _)| index)
                    .collect()
            }
        }
    }

    /// Filter by one more typed character; the cursor goes back to the first match.
    pub(crate) fn filter_type(&mut self, character: char) {
        if let Some(filter) = &mut self.filter {
            filter.push(character);
            self.picked = 0;
        }
    }

    /// Take back the last typed character.
    pub(crate) fn filter_erase(&mut self) {
        if let Some(filter) = &mut self.filter {
            filter.pop();
            self.picked = 0;
        }
    }

    /// The choice under the cursor, by its place among all of them, if the panel has any.
    pub(crate) fn picked(&self) -> Option<usize> {
        self.shown().get(self.picked).copied()
    }

    /// Up: the pick one back, or the body one row up.
    pub(crate) fn pick_previous(&mut self) {
        if self.has_choices() {
            self.picked = self.picked.saturating_sub(1);
        } else {
            self.scroll = self.scroll.saturating_sub(1);
        }
    }

    /// Down: the pick one on, among what is shown, or the body one row down.
    pub(crate) fn pick_next(&mut self) {
        if self.has_choices() {
            self.picked = (self.picked + 1).min(self.shown().len().saturating_sub(1));
        } else {
            self.scroll = self.scroll.saturating_add(1);
        }
    }

    /// PgUp: the pick a page of choices up, or the body a page of `rows` up.
    pub(crate) fn page_previous(&mut self, rows: usize) {
        if self.has_choices() {
            self.picked = self.picked.saturating_sub(self.page_rows.max(1));
        } else {
            self.scroll = self.scroll.saturating_sub(rows);
        }
    }

    /// PgDn: the pick a page of choices down, or the body a page of `rows` down.
    pub(crate) fn page_next(&mut self, rows: usize) {
        if self.has_choices() {
            self.picked =
                (self.picked + self.page_rows.max(1)).min(self.shown().len().saturating_sub(1));
        } else {
            self.scroll = self.scroll.saturating_add(rows);
        }
    }

    /// Ctrl+Home: the first choice, or the top of the body.
    pub(crate) fn pick_first(&mut self) {
        self.picked = 0;
        self.scroll = 0;
    }

    /// Ctrl+End: the last choice, or the end of the body, which the next frame clamps to.
    pub(crate) fn pick_last(&mut self) {
        if self.has_choices() {
            self.picked = self.shown().len().saturating_sub(1);
        } else {
            self.scroll = usize::MAX;
        }
    }

    /// The choice a typed letter or digit picks, if one does: a digit counts the choices shown,
    /// and nothing picks while the keys are the filter's.
    pub(crate) fn choice_for_key(&self, key: char) -> Option<usize> {
        if self.typing {
            return None;
        }
        if let Some(digit) = key.to_digit(10)
            && digit >= 1
        {
            return self.shown().get(digit as usize - 1).copied();
        }
        self.choices
            .iter()
            .position(|choice| choice.key == Some(key.to_ascii_lowercase()))
    }

    /// The wheel: the pick by `delta` rows where there are choices, else the body.
    pub(crate) fn scroll_by(&mut self, delta: isize) {
        if self.has_choices() {
            self.picked = if delta < 0 {
                self.picked.saturating_sub(delta.unsigned_abs())
            } else {
                (self.picked + delta.unsigned_abs()).min(self.shown().len().saturating_sub(1))
            };
            return;
        }
        self.scroll = if delta < 0 {
            self.scroll.saturating_sub(delta.unsigned_abs())
        } else {
            self.scroll.saturating_add(delta as usize)
        };
    }

    fn footer(&self) -> String {
        if !self.has_choices() {
            return "esc to close".to_string();
        }
        match &self.filter {
            None => "enter to select · esc to close".to_string(),
            Some(filter) if self.typing => {
                format!("enter to select · esc to stop filtering · /{filter}")
            }
            Some(filter) if filter.is_empty() => {
                "enter to select · esc to close · / to filter".to_string()
            }
            Some(filter) => format!("enter to select · esc to close · /{filter}"),
        }
    }

    fn wrapped_body(&mut self, width: usize) -> &[Line<'static>] {
        let inner = width.saturating_sub(TEXT_EDGE).max(1);
        if self.wrapped.as_ref().is_none_or(|(at, _)| *at != inner) {
            let rows = self
                .body
                .iter()
                .flat_map(|line| draw::wrap_line(line, inner))
                .collect();
            self.wrapped = Some((inner, rows));
        }
        self.wrapped
            .as_ref()
            .map(|(_, rows)| rows.as_slice())
            .unwrap_or_default()
    }

    /// The rows that are not body: the heading, the choices, the footer.
    fn fixed_rows(&self) -> usize {
        usize::from(self.heading.is_some()) + self.shown().len() + 1
    }

    /// Rows the panel wants at `width`, between two and `most`.
    pub(crate) fn height(&mut self, width: usize, most: u16) -> u16 {
        let body = self.wrapped_body(width).len();
        let rows = self.fixed_rows() + body;
        (rows.max(2) as u16).min(most.max(2))
    }

    /// Draw the panel into `area`, on the composer's background: the heading when there is one,
    /// the body and the choices at the text edge, the footer on the status row's shade.
    pub(crate) fn draw(&mut self, frame: &mut Frame<'_>, area: Rect) {
        frame.render_widget(Clear, area);
        let width = area.width as usize;
        let mut lines = Vec::new();
        if let Some(heading) = &self.heading {
            lines.push(heading.clone());
        }
        // The choices come first and scroll to keep the pick in view; the body takes the rest.
        let frame_rows = usize::from(self.heading.is_some()) + 1;
        let offered = self.shown();
        let choice_rows = offered
            .len()
            .min((area.height as usize).saturating_sub(frame_rows));
        self.page_rows = choice_rows;
        let room = (area.height as usize).saturating_sub(frame_rows + choice_rows);
        let body_rows = self.wrapped_body(width).len();
        let most = body_rows.saturating_sub(room);
        self.scroll = self.scroll.min(most);
        let scroll = self.scroll;
        let at_edge = self.body_at_edge;
        let shown: Vec<Line<'static>> = self
            .wrapped_body(width)
            .iter()
            .skip(scroll)
            .take(room)
            .cloned()
            .map(|mut line| {
                if at_edge {
                    line.spans.insert(0, Span::raw(" ".repeat(TEXT_EDGE)));
                }
                line
            })
            .collect();
        let hidden_below = body_rows.saturating_sub(scroll + room);
        lines.extend(shown);
        let rows: Vec<super::view::ListRow> = offered
            .iter()
            .filter_map(|&index| self.choices.get(index))
            .map(|choice| super::view::ListRow {
                value: choice.label.clone(),
                description: choice
                    .description
                    .clone()
                    .or_else(|| choice.key.map(|key| key.to_string())),
            })
            .collect();
        let first_choice = self.picked.saturating_sub(choice_rows.saturating_sub(1));
        lines.extend(
            super::view::list_lines(&rows, self.picked, width, self.facts)
                .into_iter()
                .skip(first_choice)
                .take(choice_rows),
        );
        let mut footer = self.footer();
        if self.scroll > 0 || hidden_below > 0 || choice_rows < offered.len() {
            footer.push_str(" · ↑↓ scroll");
            if hidden_below > 0 {
                footer.push_str(&format!(" · {hidden_below} more below"));
            }
        }
        let footer = draw::cut_text(&footer, width.saturating_sub(TEXT_EDGE));
        let footer_row = Rect {
            y: area.y + area.height.saturating_sub(1),
            height: 1.min(area.height),
            ..area
        };
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .style(Style::default().bg(super::input::COMPOSER_BACKGROUND)),
            Rect {
                height: area.height.saturating_sub(1),
                ..area
            },
        );
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(" ".repeat(TEXT_EDGE)),
                Span::styled(footer, gray()),
            ]))
            .style(Style::default().bg(super::view::STATUS_BACKGROUND)),
            footer_row,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(terminal: &ratatui::Terminal<ratatui::backend::TestBackend>) -> Vec<String> {
        let buffer = terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    /// A picker can set its facts flush with the right edge, one column short of it, whatever
    /// the length of the title before them.
    #[test]
    fn a_picker_s_facts_can_stand_at_the_edge() {
        let choices = [("a", "x"), ("a longer title", "yy")]
            .iter()
            .map(|(label, facts)| Choice {
                label: (*label).to_string(),
                key: None,
                description: Some((*facts).to_string()),
            })
            .collect();
        let mut panel = Panel::with_choices(Vec::new(), choices, 0).with_facts_at_the_edge();
        let backend = ratatui::backend::TestBackend::new(40, 4);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| panel.draw(frame, frame.area()))
            .expect("a frame");
        let shown = rows(&terminal);
        assert_eq!(shown[0].len(), 39, "{shown:#?}");
        assert!(
            shown[0].starts_with("  a ") && shown[0].ends_with("x"),
            "{shown:#?}"
        );
        assert_eq!(shown[1].len(), 39, "{shown:#?}");
        assert!(
            shown[1].starts_with("  a longer title ") && shown[1].ends_with("yy"),
            "{shown:#?}"
        );
    }

    /// PgUp and PgDn move the pick by the rows the panel showed, Ctrl+Home and Ctrl+End to the
    /// ends, so a long list is crossed without a key per row.
    #[test]
    fn a_pick_moves_by_the_page_and_to_the_ends() {
        let choices = (1..=30)
            .map(|n| Choice {
                label: format!("choice {n}"),
                key: None,
                description: None,
            })
            .collect();
        let mut panel = Panel::with_choices(Vec::new(), choices, 0);
        let backend = ratatui::backend::TestBackend::new(40, 6);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| panel.draw(frame, frame.area()))
            .expect("a frame");
        panel.page_next(100);
        assert_eq!(panel.picked(), Some(5), "five rows were shown");
        panel.pick_last();
        assert_eq!(panel.picked(), Some(29));
        panel.page_previous(100);
        assert_eq!(panel.picked(), Some(24));
        panel.pick_first();
        assert_eq!(panel.picked(), Some(0));
    }

    /// A filtering picker opens browsing; `/` starts typing, the choices the text matches are
    /// what is shown, a pick names the choice by its place among all of them, and stopping
    /// keeps the text.
    #[test]
    fn a_filtering_picker_picks_among_what_it_shows() {
        let choices = ["alpha work", "beta home", "gamma work"]
            .iter()
            .map(|label| Choice {
                label: (*label).to_string(),
                key: None,
                description: None,
            })
            .collect();
        let mut panel = Panel::with_choices(Vec::new(), choices, 0).with_filter();
        assert!(!panel.is_typing(), "a filtering picker opens browsing");
        panel.start_filtering();
        assert!(panel.is_typing());
        for character in "work".chars() {
            panel.filter_type(character);
        }
        assert_eq!(
            panel.choice_for_key('2'),
            None,
            "a digit is typed, not a pick"
        );
        panel.stop_filtering();
        assert!(!panel.is_typing());
        assert_eq!(panel.picked(), Some(0));
        panel.pick_next();
        assert_eq!(
            panel.picked(),
            Some(2),
            "the second match is the third choice"
        );
        panel.pick_next();
        assert_eq!(
            panel.picked(),
            Some(2),
            "the cursor stops at the last match"
        );
        assert_eq!(
            panel.choice_for_key('2'),
            Some(2),
            "a digit counts the choices shown"
        );
        let backend = ratatui::backend::TestBackend::new(40, 5);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| panel.draw(frame, frame.area()))
            .expect("a frame");
        let shown = rows(&terminal);
        assert_eq!(shown[0], "  alpha work", "{shown:#?}");
        assert_eq!(shown[1], "  gamma work", "{shown:#?}");
        assert!(
            shown[4].ends_with(" · /work"),
            "the footer shows the text: {shown:#?}"
        );
        for _ in 0..4 {
            panel.filter_erase();
        }
        assert_eq!(panel.picked(), Some(0));
        panel.pick_next();
        assert_eq!(panel.picked(), Some(1), "every choice is back");
    }

    /// A picker with more choices than the panel has rows scrolls them to keep the pick in view,
    /// so Enter runs what is seen.
    #[test]
    fn a_picker_taller_than_its_panel_keeps_the_pick_in_view() {
        let choices = (1..=30)
            .map(|n| Choice {
                label: format!("choice {n}"),
                key: None,
                description: None,
            })
            .collect();
        let mut panel = Panel::with_choices(Vec::new(), choices, 24);
        let backend = ratatui::backend::TestBackend::new(50, 6);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| panel.draw(frame, frame.area()))
            .expect("a frame");
        let shown = rows(&terminal);
        assert_eq!(shown[0], "  choice 21", "{shown:#?}");
        assert_eq!(shown[4], "  choice 25", "{shown:#?}");
        assert_eq!(shown[5], "  enter to select · esc to close · ↑↓ scroll");
    }

    /// A panel is its body under no heading, since the user typed the command, and a footer
    /// naming the key that closes it, on the composer's background; a body longer than the room
    /// scrolls and says how much is below.
    #[test]
    fn a_panel_shows_its_body_and_footer_and_scrolls_a_long_body() {
        let text: String = (1..=10).map(|n| format!("row {n}\n")).collect();
        let mut panel = Panel::with_text(&text);
        let backend = ratatui::backend::TestBackend::new(50, 6);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| panel.draw(frame, frame.area()))
            .expect("a frame");
        let shown = rows(&terminal);
        assert_eq!(shown[0], "  row 1");
        assert_eq!(shown[4], "  row 5");
        assert_eq!(shown[5], "  esc to close · ↑↓ scroll · 5 more below");
        let buffer = terminal.backend().buffer();
        assert_eq!(
            buffer[(buffer.area.width - 1, 2)].style().bg,
            Some(super::super::input::COMPOSER_BACKGROUND),
            "the panel wears the composer's background out to its edge"
        );
        panel.scroll_by(100);
        terminal
            .draw(|frame| panel.draw(frame, frame.area()))
            .expect("a frame");
        let shown = rows(&terminal);
        assert_eq!(
            shown[4], "  row 10",
            "the scroll clamps at the end: {shown:?}"
        );
        assert_eq!(panel.height(50, 30), 11, "the footer and ten rows of body");
    }

    /// Choices take a cursor, a letter and a digit; a heading names what the choices are about,
    /// and the keys sit in a column after the labels.
    #[test]
    fn a_choice_is_picked_by_the_cursor_a_letter_or_a_digit() {
        let mut panel = Panel::with_choices(
            vec![Line::from("path: notes.txt")],
            vec![
                Choice {
                    label: "Allow".into(),
                    key: Some('y'),
                    description: None,
                },
                Choice {
                    label: "Deny".into(),
                    key: Some('n'),
                    description: None,
                },
            ],
            0,
        )
        .with_heading("file_write notes.txt");
        assert_eq!(panel.picked(), Some(0));
        panel.pick_next();
        assert_eq!(panel.picked(), Some(1));
        panel.pick_next();
        assert_eq!(panel.picked(), Some(1), "the cursor stops at the end");
        assert_eq!(panel.choice_for_key('Y'), Some(0));
        assert_eq!(panel.choice_for_key('2'), Some(1));
        assert_eq!(panel.choice_for_key('3'), None);
        assert_eq!(panel.choice_for_key('q'), None);
        let backend = ratatui::backend::TestBackend::new(40, 6);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| panel.draw(frame, frame.area()))
            .expect("a frame");
        let shown = rows(&terminal);
        assert_eq!(shown[0], "file_write notes.txt");
        assert_eq!(shown[2], "  Allow  y");
        assert_eq!(shown[3], "  Deny   n");
        assert_eq!(shown[5], "  enter to select · esc to close");
        let buffer = terminal.backend().buffer();
        assert!(
            buffer[(0, 3)]
                .style()
                .add_modifier
                .contains(ratatui::style::Modifier::REVERSED),
            "the picked choice is the reversed row"
        );
    }
}
