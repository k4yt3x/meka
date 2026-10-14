//! How the transcript is drawn: each item as lines at a width, cached by the item's revision; the
//! viewport that scrolls them and maps a click to an item; the status row; and the rows the
//! screen draws between and under the transcript.

use std::{collections::HashMap, time::Duration};

use ratatui::{
    Frame,
    buffer::{Buffer, CellDiffOption},
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Clear, Paragraph},
};

use super::{
    input::COMPOSER_BACKGROUND,
    model::{Entry, Item, ToolState, Transcript},
};
use crate::{
    frontend::FrontendEvent,
    render::tui::{self as draw, ACT_GLYPH, ERROR_GLYPH, TEXT_EDGE, WARNING_GLYPH, gray},
    store::inbox::InboxClass,
    text::{display_width, wrap_to_width},
};

/// Rows left between two items.
const ITEM_GAP: usize = 1;
/// The status row's background: a shade darker than the composer's, so the two read as a footer
/// under a field rather than one block.
pub(crate) const STATUS_BACKGROUND: Color = Color::Indexed(234);
/// What the gap row says while the view is scrolled up from the bottom.
pub(crate) const BACK_TO_BOTTOM: &str = "↓ back to bottom";
/// meka's mark, the `M` of its icon, in half-block pixels: three rows, one beside each of the
/// banner's lines.
const BANNER_MARK: [&str; 3] = ["█▄ ▄█", "█ ▀ █", "█   █"];
/// The columns between the banner and the left edge of the screen; one row stands above it.
const BANNER_GAP: usize = 2;

/// The answer's text as lines: what has settled through markdown, and while it streams, the
/// unterminated tail as plain text so a token shows the moment it arrives.
fn text_lines(text: &str, streaming: bool, width: usize, reasoning: bool) -> Vec<Line<'static>> {
    let (settled, open) = if streaming {
        match text.rfind('\n') {
            Some(cut) => text.split_at(cut + 1),
            None => ("", text),
        }
    } else {
        (text, "")
    };
    let mut lines = if reasoning {
        draw::thinking_lines(settled, width)
    } else {
        draw::markdown_lines(settled, width)
    };
    if !open.is_empty() {
        let style = if reasoning { gray() } else { Style::default() };
        lines.extend(draw::plain_lines(open, width, style));
    }
    lines
}

/// Whether a tool's result is drawn as an item of its own, so the call's row shows nothing of it.
fn result_is_its_own_item(name: &str) -> bool {
    name.starts_with("checklist_")
}

/// The rows under a tool call's row: what it is doing or what it produced. Nothing for any other
/// item.
fn tool_body(item: &Item, expanded: bool, width: usize) -> Vec<Line<'static>> {
    let Item::ToolCall {
        name,
        state,
        output,
        result,
        result_images,
        activity,
        progress,
        change,
        ..
    } = item
    else {
        return Vec::new();
    };
    let (state, result_images) = (*state, *result_images);
    let result = result.as_deref();
    let progress = progress.as_deref();
    let inner = width.saturating_sub(draw::connector_indent()).max(1);
    // Expanded, every line wraps whole; collapsed, a line is one row cut at the edge, since a
    // preview is a glance at how the result opens and its count is then a count of lines.
    let wrap = if expanded { inner } else { usize::MAX };
    let mut rows = Vec::new();
    if let Some(progress) = progress {
        rows.extend(draw::plain_lines(progress, wrap, gray()));
    }
    if !activity.is_empty() {
        rows.extend(draw::plain_lines(activity.trim_end(), wrap, gray()));
    }
    let failed = state == ToolState::Done { is_error: true };
    let shown = if !output.is_empty() {
        Some(output.as_str())
    } else {
        result.filter(|text| !text.trim().is_empty())
    };
    match shown {
        // What the call changed is the result worth reading; the words the tool wrote about it
        // are the model's.
        Some(_) if change.is_some() && !failed => {
            if let Some(change) = change {
                let (diff, _) = draw::diff_rows(change.old.as_deref(), &change.new, inner);
                rows.extend(diff);
            }
        }
        Some(text) if failed || !result_is_its_own_item(name) => {
            let text = text.trim_end_matches('\n');
            let mut lines = if failed {
                // The label shares the first row, so the text is wrapped with the label's columns
                // spent and its later rows hang under the text; wrapped at the full width, the
                // first row would overrun the edge by the label.
                let label = "failed: ";
                let hang = display_width(label);
                let mut lines = draw::ansi_lines(text, wrap.saturating_sub(hang).max(1));
                for (index, line) in lines.iter_mut().enumerate() {
                    if index == 0 {
                        for span in &mut line.spans {
                            span.style = span.style.fg(Color::Red);
                        }
                        line.spans.insert(
                            0,
                            Span::styled(label.to_string(), Style::default().fg(Color::Red)),
                        );
                    } else {
                        line.spans.insert(0, Span::raw(" ".repeat(hang)));
                    }
                }
                lines
            } else {
                draw::ansi_lines(text, wrap)
            };
            if !failed {
                for line in &mut lines {
                    for span in &mut line.spans {
                        if span.style.fg.is_none() {
                            span.style = span.style.fg(Color::DarkGray);
                        }
                    }
                }
            }
            rows.extend(lines);
        }
        _ => {}
    }
    if state == ToolState::Running && rows.is_empty() {
        rows.push(Line::from(Span::styled("running".to_string(), gray())));
    }
    if result_images > 0 {
        rows.push(Line::from(Span::styled(
            format!(
                "{result_images} image{}",
                if result_images == 1 { "" } else { "s" }
            ),
            gray(),
        )));
    }
    if expanded {
        rows
    } else {
        let most = if change.is_some() && !failed {
            draw::DIFF_PREVIEW_ROWS
        } else {
            draw::PREVIEW_ROWS
        };
        draw::preview_rows(rows, most)
            .into_iter()
            .map(|row| draw::cut_line(row, inner))
            .collect()
    }
}

/// One item as the lines it occupies at `width`.
pub(crate) fn entry_lines(entry: &Entry, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let inner = width.saturating_sub(TEXT_EDGE).max(1);
    match &entry.item {
        Item::Prompt {
            text,
            class,
            inbox_id: _,
            delivered,
        } => {
            // The block the composer is, kept once sent: the same gray across the whole row, so a
            // prompt is told from the answer under it at a glance, the words in white rather than
            // the terminal's foreground so they stand out of the gray.
            let block = Style::default().fg(draw::WHITE).bg(COMPOSER_BACKGROUND);
            let padded = |row: String, style: Style, first: bool| {
                let lead = if first {
                    format!("{:<TEXT_EDGE$}", draw::PROMPT_GLYPH)
                } else {
                    " ".repeat(TEXT_EDGE)
                };
                let fill = width.saturating_sub(TEXT_EDGE + display_width(&row));
                Line::from(vec![
                    // The glyph recedes behind the words.
                    Span::styled(lead, block.fg(draw::PROMPT_GLYPH_COLOR)),
                    Span::styled(row, style),
                    Span::styled(" ".repeat(fill), block),
                ])
            };
            let mut lines = Vec::new();
            for source in text.lines() {
                // The user's own words, sanitized as an answer is: the joiners that build an
                // emoji or a conjunct stay, where the one-row sanitizer for a server's strings
                // would drop them.
                let source = crate::text::expand_tabs(&crate::text::sanitize_stream_text(source));
                for row in wrap_to_width(&source, inner, usize::MAX) {
                    let first = lines.is_empty();
                    lines.push(padded(row, block, first));
                }
            }
            if let Some(class) = class {
                let word = match class {
                    InboxClass::Steer => "steer",
                    InboxClass::Followup => "followup",
                    InboxClass::Interrupt => "interrupt",
                };
                let note = if *delivered {
                    format!("({word})")
                } else {
                    format!("({word}, waiting)")
                };
                lines.push(padded(
                    note,
                    gray().bg(COMPOSER_BACKGROUND),
                    lines.is_empty(),
                ));
            }
            lines
        }
        Item::Text { text, streaming } => {
            draw::edge_rows(text_lines(text, *streaming, inner, false))
        }
        Item::Thinking {
            text,
            estimate,
            ended,
        } => {
            if !entry.expanded || text.trim().is_empty() {
                return vec![draw::thinking_row(text, *estimate, !*ended, width)];
            }
            let mut lines = vec![draw::thinking_label_row()];
            lines.extend(draw::edge_rows(text_lines(text, !*ended, inner, true)));
            lines
        }
        Item::ToolCall {
            name,
            input,
            summary,
            state,
            change,
            ..
        } => {
            let empty = serde_json::Value::Null;
            let input = input.as_ref().unwrap_or(&empty);
            let code = crate::tools::builtin_code_param(name);
            let stat = change
                .as_ref()
                .filter(|_| *state != ToolState::Done { is_error: true })
                .map(|change| draw::diff_rows(change.old.as_deref(), &change.new, width).1);
            // The arguments are shown in full under the row when the call is expanded; the row
            // then carries no summary, which would say the same thing twice.
            let summary = match state {
                ToolState::Composing => Some("…".to_string()),
                _ if entry.expanded => None,
                _ => draw::tool_summary(name, input, summary.as_deref()),
            };
            let mut lines = vec![draw::tool_row(
                name,
                summary.as_deref(),
                code.map(|(_, language)| language),
                stat,
                width,
            )];
            if *state == ToolState::Composing {
                return lines;
            }
            let text_width = width.saturating_sub(draw::connector_indent()).max(1);
            let (arguments, program_first) = if entry.expanded {
                // A change drawn as a diff says what the two texts were; their keys would say it
                // again.
                let omit: &[&str] = if stat.is_some() {
                    crate::tools::builtin_change_params(name)
                } else {
                    &[]
                };
                draw::call_argument_rows(name, input, omit, gray(), text_width)
            } else {
                (Vec::new(), false)
            };
            lines.extend(draw::tree_rows(
                arguments,
                program_first,
                tool_body(&entry.item, entry.expanded, width),
            ));
            lines
        }
        Item::Checklist { items } => draw::checklist_rows(items, width),
        Item::Notice { level, text } => match level {
            crate::frontend::NoticeLevel::Info => {
                draw::glyph_text_rows(ACT_GLYPH, gray(), text, gray(), width)
            }
            crate::frontend::NoticeLevel::Warn => {
                let yellow = Style::default().fg(Color::Yellow);
                draw::glyph_text_rows(WARNING_GLYPH, yellow, text, yellow, width)
            }
        },
        Item::Error(text) => {
            let red = Style::default().fg(Color::Red);
            draw::glyph_text_rows(ERROR_GLYPH, red, text, red, width)
        }
        Item::ShellOutput(text) => {
            draw::edge_rows(draw::ansi_lines(text.trim_end_matches('\n'), inner))
        }
        Item::SessionId { label, id } => {
            draw::glyph_text_rows(ACT_GLYPH, gray(), &format!("{label} {id}"), gray(), width)
        }
        Item::TurnEnd(text) => draw::glyph_text_rows(ACT_GLYPH, gray(), text, gray(), width),
        Item::Image => draw::glyph_text_rows(ACT_GLYPH, gray(), "image", gray(), width),
        // The mark in the terminal's cyan beside three lines, on no background: a box of the
        // composer's gray was painted short by tmux under Alacritty, which drops the background
        // of the spaces after a foreground reset, and a box with no padding is no box.
        Item::Banner {
            version,
            profile,
            model,
            directory,
        } => {
            let profile = match model {
                Some(model) => format!("{profile} ({model})"),
                None => profile.clone(),
            };
            let lead = BANNER_GAP + display_width(BANNER_MARK[0]) + 2;
            let room = width.saturating_sub(lead).max(1);
            // The name bold and white, so it is what the eye lands on, the version dimmed beside
            // it; the facts a step down from the name, in the palette's plain white, which most
            // themes set to a light gray, rather than the bright one.
            let title = vec![
                Span::styled(
                    "meka".to_string(),
                    Style::default()
                        .fg(draw::WHITE)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!(" v{version}"), gray()),
            ];
            let fact = |text: &str| {
                vec![Span::styled(
                    draw::cut_text(text, room),
                    Style::default().fg(Color::Gray),
                )]
            };
            let rows = [
                draw::cut_spans(title, room),
                fact(&profile),
                fact(directory),
            ];
            let mut lines = vec![Line::default()];
            lines.extend(BANNER_MARK.iter().zip(rows).map(|(mark, text)| {
                let mut spans = vec![
                    Span::raw(" ".repeat(BANNER_GAP)),
                    Span::styled((*mark).to_string(), Style::default().fg(Color::Cyan)),
                    Span::raw("  ".to_string()),
                ];
                spans.extend(text);
                Line::from(spans)
            }));
            lines
        }
    }
}

/// What one item was last drawn as.
struct Cached {
    revision: u64,
    width: usize,
    lines: Vec<Line<'static>>,
}

/// The transcript on screen: which lines are visible, and what sits where.
pub(crate) struct Viewport {
    /// What each resident item was last drawn as, by the item's id.
    cache: HashMap<u64, Cached>,
    /// The first visible line, counted over every item and the gaps between them.
    scroll: usize,
    /// Whether the view sticks to the bottom as items arrive.
    follow: bool,
    /// The item the keyboard has picked out, if any, by its place on screen.
    cursor: Option<usize>,
    /// Where each item's lines start and how many there are, as of the last draw.
    layout: Vec<(usize, usize)>,
    /// The item at the top of the view and how far into it the view starts, so a change above
    /// it (a page loaded, a re-wrap) leaves what is on screen where it is. Cleared by every
    /// scroll the user makes, and set again by the draw that follows.
    anchor: Option<(u64, isize)>,
    /// The row of the view the mouse is over, when it is over the view: a collapsible item
    /// drawn there lights up, which is the hint that a click would open it.
    hover_row: Option<usize>,
    /// The nearest prompt scrolled off the top, by id and first row, for the bar that brings
    /// it back.
    sticky: Option<(u64, usize)>,
    total: usize,
    height: usize,
}

impl Viewport {
    pub(crate) fn new() -> Self {
        Self {
            cache: HashMap::new(),
            scroll: 0,
            follow: true,
            cursor: None,
            layout: Vec::new(),
            anchor: None,
            sticky: None,
            hover_row: None,
            total: 0,
            height: 0,
        }
    }

    /// Say which row of the view the mouse is over, or that it is not over the view; whether
    /// that moves the light to another item.
    pub(crate) fn set_hover_row(&mut self, row: Option<usize>) -> bool {
        let before = self.hover_row.and_then(|row| self.item_at_row(row));
        self.hover_row = row;
        before != row.and_then(|row| self.item_at_row(row))
    }

    pub(crate) fn cursor(&self) -> Option<usize> {
        self.cursor
    }

    /// The prompt stuck at the top, as its first line and the row it starts on: the nearest one
    /// scrolled off above, so a click can bring it back.
    pub(crate) fn sticky_prompt(&self) -> Option<(Line<'static>, usize)> {
        let (id, start) = self.sticky?;
        let line = self.cache.get(&id)?.lines.first()?.clone();
        Some((line, start))
    }

    /// Put `row` at the top, and stop following the bottom unless that is where it lands.
    pub(crate) fn scroll_to(&mut self, row: usize) {
        let bottom = self.total.saturating_sub(self.height);
        self.scroll = row.min(bottom);
        self.follow = self.scroll >= bottom;
        self.anchor = None;
    }

    pub(crate) fn is_following(&self) -> bool {
        self.follow
    }

    /// Whether the view is within a screen of the top, where the page above would be read.
    pub(crate) fn is_near_top(&self) -> bool {
        self.scroll < self.height
    }

    /// Whether the view is within a screen of the bottom, where the page below would be read.
    pub(crate) fn is_near_bottom(&self) -> bool {
        self.scroll + self.height.saturating_mul(2) >= self.total
    }

    /// Scroll by `delta` rows; scrolling up stops following, reaching the bottom resumes it.
    pub(crate) fn scroll_by(&mut self, delta: isize) {
        let bottom = self.total.saturating_sub(self.height);
        let target = if delta < 0 {
            self.scroll.saturating_sub(delta.unsigned_abs())
        } else {
            self.scroll.saturating_add(delta as usize).min(bottom)
        };
        self.scroll = target;
        self.follow = target >= bottom;
        self.anchor = None;
    }

    pub(crate) fn scroll_to_top(&mut self) {
        self.scroll = 0;
        self.follow = self.total <= self.height;
        self.anchor = None;
    }

    pub(crate) fn scroll_to_bottom(&mut self) {
        self.follow = true;
        self.anchor = None;
    }

    /// Half a screen, the page size every scroll key uses.
    pub(crate) fn page(&self) -> isize {
        (self.height / 2).max(1) as isize
    }

    /// Point the cursor at an item, or clear it.
    pub(crate) fn set_cursor(&mut self, cursor: Option<usize>) {
        self.cursor = cursor;
        if let Some(index) = cursor
            && let Some(&(start, len)) = self.layout.get(index)
        {
            // Brought into view rather than centered: the item above and below stay where they
            // were, which is what keeps a walk through the list legible.
            if start < self.scroll {
                self.scroll = start;
                self.follow = false;
                self.anchor = None;
            } else if start + len > self.scroll + self.height {
                self.scroll = (start + len).saturating_sub(self.height);
                self.follow = self.scroll >= self.total.saturating_sub(self.height);
                self.anchor = None;
            }
        }
    }

    /// The item drawn on `row` of the last frame, if any, by its place on screen.
    pub(crate) fn item_at_row(&self, row: usize) -> Option<usize> {
        let line = self.scroll.checked_add(row)?;
        self.layout
            .iter()
            .position(|&(start, len)| line >= start && line < start + len)
    }

    /// Lay `entry` out at `width` unless the cache already has it, and say how many rows it is.
    fn lay_out(&mut self, entry: &Entry, width: usize) -> usize {
        let stale = self
            .cache
            .get(&entry.id)
            .is_none_or(|cached| cached.revision != entry.revision || cached.width != width);
        if stale {
            self.cache.insert(entry.id, Cached {
                revision: entry.revision,
                width,
                lines: entry_lines(entry, width),
            });
        }
        self.cache
            .get(&entry.id)
            .map_or(0, |cached| cached.lines.len())
    }

    /// Draw the transcript into `area`.
    pub(crate) fn draw(&mut self, frame: &mut Frame<'_>, area: Rect, transcript: &Transcript) {
        self.height = area.height as usize;
        let width = area.width as usize;

        // Lay every item out first, so the scroll position is clamped against the whole.
        self.layout.clear();
        let mut ids: Vec<u64> = Vec::with_capacity(transcript.drawn_len());
        let mut prompts: Vec<(u64, usize)> = Vec::new();
        let mut total = 0;
        for entry in transcript.drawn() {
            ids.push(entry.id);
            let len = self.lay_out(entry, width);
            if len == 0 {
                self.layout.push((total, 0));
                continue;
            }
            if total > 0 {
                total += ITEM_GAP;
            }
            if matches!(entry.item, Item::Prompt { .. }) {
                prompts.push((entry.id, total));
            }
            self.layout.push((total, len));
            total += len;
        }
        // Only what is on the list stays laid out: a page the window dropped takes its rows
        // with it.
        {
            let resident: std::collections::HashSet<u64> = ids.iter().copied().collect();
            self.cache.retain(|id, _| resident.contains(id));
        }
        self.total = total;
        if self.cursor.is_some_and(|index| index >= self.layout.len()) {
            self.cursor = None;
        }
        let bottom = total.saturating_sub(self.height);
        if self.follow {
            self.scroll = bottom;
        } else if let Some((id, offset)) = self.anchor
            && let Some(position) = ids.iter().position(|&held| held == id)
        {
            let start = self.layout[position].0 as isize;
            self.scroll = ((start + offset).max(0) as usize).min(bottom);
        } else {
            self.scroll = self.scroll.min(bottom);
        }
        let scroll = self.scroll;
        // The offset is signed: a top row that is the gap above an item anchors to that item
        // one row early, which a saturated zero would quietly undo on the next frame.
        self.anchor = self
            .layout
            .iter()
            .zip(&ids)
            .find(|((start, len), _)| *len > 0 && scroll < start + len)
            .map(|((start, _), id)| (*id, scroll as isize - *start as isize));
        // The bar covers the top row, so it is not drawn while a prompt's own first row is there:
        // nothing is hidden, and one more row of scrolling makes that prompt the bar's.
        self.sticky = if prompts.iter().any(|&(_, start)| start == scroll) {
            None
        } else {
            prompts
                .iter()
                .rev()
                .find(|&&(_, start)| start < scroll)
                .copied()
        };

        // The light goes on the collapsible item under the mouse, resolved against the layout of
        // this frame, so it follows the transcript as it scrolls under a still pointer.
        let lit = self
            .hover_row
            .and_then(|row| self.item_at_row(row))
            .filter(|&index| {
                transcript
                    .drawn()
                    .nth(index)
                    .is_some_and(Entry::is_collapsible)
            });

        let mut rows: Vec<Line<'static>> = Vec::with_capacity(self.height);
        let first = self.scroll;
        let last = first + self.height;
        for (index, &(start, len)) in self.layout.iter().enumerate() {
            if len == 0 || start + len <= first {
                continue;
            }
            if start >= last {
                break;
            }
            let picked = self.cursor == Some(index);
            let hovered = lit == Some(index);
            let lines = self
                .cache
                .get(&ids[index])
                .map(|cached| cached.lines.as_slice())
                .unwrap_or_default();
            for (offset, line) in lines.iter().enumerate() {
                let at = start + offset;
                if at < first || at >= last {
                    continue;
                }
                let mut line = line.clone();
                if hovered {
                    line = lit_up(line);
                }
                if picked && offset == 0 {
                    line = line.patch_style(Style::default().add_modifier(Modifier::REVERSED));
                }
                while rows.len() < at - first {
                    rows.push(Line::default());
                }
                rows.push(line);
            }
        }
        frame.render_widget(Paragraph::new(Text::from(rows)), area);
    }
}

/// The prompt stuck at the top of the transcript: its first line, cut at the edge, dimmed and on
/// the composer's shade so it reads as a reminder over the rows rather than the prompt itself,
/// which matters where it sits right above another prompt; it lights under the mouse, since a
/// click on it brings the prompt back.
pub(crate) fn draw_sticky_prompt(
    frame: &mut Frame<'_>,
    area: Rect,
    line: Line<'static>,
    hovered: bool,
) {
    if area.height == 0 {
        return;
    }
    let mut spans = draw::cut_spans(line.spans, area.width as usize);
    let tone = if hovered {
        draw::WHITE
    } else {
        Color::DarkGray
    };
    for span in &mut spans {
        span.style = span.style.fg(tone);
    }
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(Line::from(spans))
            .style(Style::default().bg(super::input::COMPOSER_BACKGROUND)),
        area,
    );
}

/// A line with its dimmed text lit to white, the way a collapsible item answers the mouse over
/// it; what is colored already keeps its color.
fn lit_up(mut line: Line<'static>) -> Line<'static> {
    for span in &mut line.spans {
        if span.style.fg == Some(Color::DarkGray) {
            span.style = span.style.fg(draw::WHITE);
        }
    }
    line
}

/// Where a frame drew a symbol wider than one column out of more than one code point: a flag, a
/// keycap, a joined emoji. Each entry is the cell and the columns the symbol covers.
///
/// ratatui rewrites the cell after a wide symbol only when the symbol carries a variation
/// selector or the old cell had a visible background. A terminal that keeps a flag's second code
/// point in a cell of its own then shows that letter after the flag has scrolled away, because the
/// cell under it compares equal between the two frames. These positions let the next frame force
/// those cells out again.
pub(crate) fn wide_symbols(buffer: &Buffer) -> Vec<(u16, u16, u16)> {
    use unicode_width::UnicodeWidthStr;
    let area = buffer.area();
    let mut found = Vec::new();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let symbol = buffer[(x, y)].symbol();
            if symbol.chars().nth(1).is_some() {
                let width = symbol.width();
                if width > 1 {
                    found.push((x, y, width as u16));
                }
            }
        }
    }
    found
}

/// Make the cells that followed the last frame's wide symbols rewrite themselves, whether or not
/// the frame's own contents changed there. See [`wide_symbols`].
pub(crate) fn refresh_after_wide_symbols(buffer: &mut Buffer, previous: &[(u16, u16, u16)]) {
    let area = *buffer.area();
    for &(x, y, width) in previous {
        for dx in 1..width {
            let column = x.saturating_add(dx);
            if column < area.right() && y < area.bottom() {
                buffer[(column, y)].set_diff_option(CellDiffOption::AlwaysUpdate);
            }
        }
    }
}

/// What runs beside the turn: the session's background tasks and the sub-agents of this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct RunningWork {
    pub(crate) tasks: usize,
    pub(crate) subagents: usize,
}

/// The facts the status row can show, and which of them it shows, in order.
pub(crate) struct StatusFacts<'a> {
    /// `[display].status_row`.
    pub(crate) shown: &'a [crate::config::StatusFact],
    pub(crate) permission: crate::permission::Permission,
    pub(crate) path: Option<String>,
    /// The profile the session runs on and its model, as `name (model)`.
    pub(crate) profile: Option<String>,
    pub(crate) context: Option<String>,
    /// Background tasks and sub-agents, shown while any run.
    pub(crate) work: RunningWork,
    pub(crate) hint: Option<&'a str>,
    /// Whether the composer is in its shell mode, which the row says in red in place of every
    /// fact, so what Enter will do is never a surprise.
    pub(crate) shell_mode: bool,
}

/// A crossterm color as ratatui's.
pub(crate) fn color(color: crossterm::style::Color) -> Color {
    use ratatui::backend::FromCrossterm;
    Color::from_crossterm(color)
}

/// What the status row says, alone and in red, while the composer is in its shell mode.
pub(crate) const SHELL_MODE_LABEL: &str = "Run command in shell";

/// The status row: the permission level, always, as a cell of its color in the gutter column and
/// its name at the text edge in the same color, then the facts the config lists, and the hint
/// against the right edge when it fits there.
pub(crate) fn status_line(facts: &StatusFacts<'_>, width: usize) -> Line<'static> {
    use crate::config::StatusFact;
    // Every fact follows a dot: the level, or the shell mode's sentence, is always ahead of it.
    fn lead(spans: &mut Vec<Span<'static>>) {
        spans.push(Span::styled(" · ".to_string(), gray()));
    }
    let counted =
        |count: usize, noun: &str| format!("{count} {noun}{}", if count == 1 { "" } else { "s" });
    let mut spans: Vec<Span<'static>> = Vec::new();
    // The shell mode takes the whole row: one red sentence where the level and the facts were,
    // since those are about the session and the sentence is about what the next Enter does. It
    // starts at the text edge, under the command it is about.
    if facts.shell_mode {
        spans.push(Span::styled(
            format!("{:TEXT_EDGE$}{SHELL_MODE_LABEL}", ""),
            Style::default().fg(Color::Red),
        ));
    } else {
        let level = color(facts.permission.indicator_color());
        spans.push(Span::styled(" ".to_string(), Style::default().bg(level)));
        spans.push(Span::raw(" ".repeat(TEXT_EDGE - 1)));
        spans.push(Span::styled(
            facts.permission.to_string(),
            Style::default().fg(level),
        ));
    }
    for fact in facts.shown.iter().filter(|_| !facts.shell_mode) {
        let text = match fact {
            StatusFact::Path => facts.path.clone(),
            StatusFact::Profile => facts.profile.clone(),
            StatusFact::Context => facts.context.clone(),
            StatusFact::Tasks => {
                (facts.work.tasks > 0).then(|| counted(facts.work.tasks, "background task"))
            }
            StatusFact::Subagents => {
                (facts.work.subagents > 0).then(|| counted(facts.work.subagents, "sub-agent"))
            }
        };
        if let Some(text) = text {
            lead(&mut spans);
            spans.push(Span::styled(text, gray()));
        }
    }
    // The right edge: the hint.
    let mut trailing: Vec<Span<'static>> = Vec::new();
    if let Some(hint) = facts.hint {
        trailing.push(Span::styled(
            hint.to_string(),
            Style::default().fg(Color::Yellow),
        ));
    }
    if !trailing.is_empty() {
        let used: usize = spans.iter().map(|span| display_width(&span.content)).sum();
        let trailing_width: usize = trailing
            .iter()
            .map(|span| display_width(&span.content))
            .sum();
        if used + 2 + trailing_width < width {
            spans.push(Span::raw(" ".repeat(width - used - trailing_width - 1)));
        } else {
            lead(&mut spans);
        }
        spans.extend(trailing);
    }
    Line::from(spans)
}

/// What the agent is doing, as the stream says it: the word on the working row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Activity {
    /// Busy with nothing said yet about what: the request is on its way, or a result is.
    Working,
    Thinking,
    Answering,
    /// A tool call is being written or runs; which tool, and which of the two, the call's own
    /// row on the transcript says.
    UsingTool,
    /// The turn waits on the user: an approval or a question is open.
    Waiting,
    /// A `!command` runs beside the screen.
    Shell,
}

impl Activity {
    /// What the agent is doing once `event` has passed: the stream names the thinking, the
    /// answer and the calls as they start, and a call's end or a thinking block's leaves the
    /// agent at work with nothing said yet about what comes next.
    pub(crate) fn note(&mut self, event: &FrontendEvent) {
        match event {
            FrontendEvent::TurnStarted { .. }
            | FrontendEvent::TurnFinished
            | FrontendEvent::ThinkingEnded
            | FrontendEvent::ToolCallCompleted { .. } => *self = Self::Working,
            FrontendEvent::ThinkingDelta(_)
            | FrontendEvent::ThinkingProgress { .. }
            | FrontendEvent::ThinkingBlock { .. } => *self = Self::Thinking,
            FrontendEvent::AssistantTextDelta(_) => *self = Self::Answering,
            FrontendEvent::ToolCallComposing { .. } | FrontendEvent::ToolCallStarted { .. } => {
                *self = Self::UsingTool;
            }
            _ => {}
        }
    }

    /// Whether nothing moves until the user answers, which stills the spinner.
    pub(crate) fn is_waiting(&self) -> bool {
        matches!(self, Self::Waiting)
    }
}

/// The spinner's frames, one cell wide in every font, in the order they turn.
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// How long each frame of the spinner stands.
const SPINNER_STEP: Duration = Duration::from_millis(100);

/// The spinner's frame at `elapsed` into the work.
pub(crate) fn spinner_frame(elapsed: Duration) -> usize {
    ((elapsed.as_millis() / SPINNER_STEP.as_millis()) % SPINNER.len() as u128) as usize
}

/// The working row: the spinner in column 0 (the agent's glyph, still, while the turn waits on
/// the user), what the agent is doing at the text edge, then the seconds and the key that stops
/// it, dimmed. What does not fit at `width` goes from the end.
pub(crate) fn working_line(
    activity: &Activity,
    elapsed: Duration,
    frame: usize,
    width: usize,
) -> Line<'static> {
    let glyph = if activity.is_waiting() {
        draw::ACT_GLYPH
    } else {
        SPINNER[frame % SPINNER.len()]
    };
    let mut spans = vec![Span::styled(
        format!("{glyph:<TEXT_EDGE$}"),
        Style::default().add_modifier(Modifier::BOLD),
    )];
    let word = match activity {
        Activity::Working => "Working",
        Activity::Thinking => "Thinking",
        Activity::Answering => "Answering",
        Activity::UsingTool => "Using a tool",
        Activity::Waiting => "Waiting for your answer",
        Activity::Shell => "Running command",
    };
    spans.push(Span::raw(word.to_string()));
    let mut tail = vec![Span::styled(format!(" · {}s", elapsed.as_secs()), gray())];
    if !activity.is_waiting() {
        tail.push(Span::styled(" · Esc to stop".to_string(), gray()));
    }
    let used = |spans: &[Span<'static>]| -> usize {
        spans.iter().map(|span| display_width(&span.content)).sum()
    };
    while !tail.is_empty() && used(&spans) + used(&tail) > width {
        tail.pop();
    }
    spans.extend(tail);
    Line::from(spans)
}

/// The working row, its own under the blank one: [`working_line`] from the left edge.
pub(crate) fn draw_working_row(
    frame: &mut Frame<'_>,
    area: Rect,
    activity: &Activity,
    elapsed: Duration,
    spinner: usize,
) -> u16 {
    let line = working_line(activity, elapsed, spinner, area.width as usize);
    let width = line.width() as u16;
    frame.render_widget(Paragraph::new(line), area);
    width
}

/// The pill while the view is scrolled up: the words on the composer's gray, as a pill the
/// mouse can hit, centered on the row it is given, or against the right edge where the center
/// would cover the row's first `taken` columns (the working row's words). Returns the cells it
/// covers.
pub(crate) fn draw_back_to_bottom(frame: &mut Frame<'_>, area: Rect, taken: u16) -> Rect {
    let label = format!(" {BACK_TO_BOTTOM} ");
    let label_width = display_width(&label).min(area.width as usize) as u16;
    let centered = area.width.saturating_sub(label_width) / 2;
    let x = if taken == 0 || centered >= taken.saturating_add(2) {
        centered
    } else {
        area.width.saturating_sub(label_width)
    };
    let pill = Rect {
        x: area.x + x,
        y: area.y,
        width: label_width,
        height: 1.min(area.height),
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            label,
            Style::default().fg(draw::WHITE).bg(COMPOSER_BACKGROUND),
        ))),
        pill,
    );
    pill
}

/// A question drawn where the composer was: its body, then the prompt line with the answer so
/// far.
pub(crate) fn draw_question(
    frame: &mut Frame<'_>,
    area: Rect,
    body: &[Line<'static>],
    prompt: &str,
    answer: &str,
) {
    frame.render_widget(Clear, area);
    let mut lines = body.to_vec();
    lines.push(Line::from(vec![
        Span::raw(" ".repeat(TEXT_EDGE)),
        Span::styled(prompt.to_string(), Style::default().fg(Color::Yellow)),
        Span::raw(answer.to_string()),
    ]));
    // The last rows are the ones that matter: the question and the answer under it.
    let skip = lines.len().saturating_sub(area.height as usize);
    let shown: Vec<Line<'static>> = lines.into_iter().skip(skip).collect();
    let cursor_row = shown.len().saturating_sub(1);
    let cursor_column = TEXT_EDGE + display_width(prompt) + display_width(answer);
    // The composer's background: the question stands where the composer stood.
    frame.render_widget(
        Paragraph::new(Text::from(shown))
            .style(Style::default().bg(super::input::COMPOSER_BACKGROUND)),
        area,
    );
    frame.set_cursor_position((
        area.x
            .saturating_add(cursor_column.min(area.width.saturating_sub(1) as usize) as u16),
        area.y.saturating_add(cursor_row as u16),
    ));
}

/// Rows the completion list shows at most; the pick scrolls the rest into view.
pub(crate) const COMPLETION_ROWS: usize = 10;

/// One row of a list to pick from, the completion list or a panel's choices: what is inserted or
/// chosen, and beside it what it is, or the key that picks it. A command's argument grammar is
/// `/help`'s to show; here it would set the column for every row.
pub(crate) struct ListRow {
    pub(crate) value: String,
    pub(crate) description: Option<String>,
}

/// Where a row's description stands: in a column after the longest value, as the key beside a
/// choice reads, or flush with the right edge, as facts read beside a title.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ListLayout {
    /// Two columns after the longest value.
    Column,
    /// One column short of the right edge.
    Edge,
}

/// The rows of a list `width` wide: values at the text edge, descriptions dim in a column after
/// the longest value, the picked row one reversed bar from edge to edge, nothing on it dimmed.
pub(crate) fn list_lines(
    rows: &[ListRow],
    picked: usize,
    width: usize,
    layout: ListLayout,
) -> Vec<Line<'static>> {
    let column = rows
        .iter()
        .map(|row| display_width(&row.value))
        .max()
        .unwrap_or(0);
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let dim = if index == picked {
                Style::default()
            } else {
                gray()
            };
            let mut spans = vec![Span::raw(format!("{:TEXT_EDGE$}{}", "", row.value))];
            if let Some(description) = &row.description {
                let pad = match layout {
                    ListLayout::Column => column - display_width(&row.value) + 2,
                    // One column short of the edge, and never closer than two to the value.
                    ListLayout::Edge => width
                        .saturating_sub(
                            TEXT_EDGE + display_width(&row.value) + display_width(description) + 1,
                        )
                        .max(2),
                };
                spans.push(Span::styled(format!("{:pad$}{description}", ""), dim));
            }
            if index == picked {
                let used: usize = spans.iter().map(|span| display_width(&span.content)).sum();
                spans.push(Span::raw(" ".repeat(width.saturating_sub(used))));
                Line::from(spans).patch_style(Style::default().add_modifier(Modifier::REVERSED))
            } else {
                Line::from(spans)
            }
        })
        .collect()
}

/// The completion list, drawn flush above the composer, at most `COMPLETION_ROWS` of it with the
/// picked row kept in view.
pub(crate) fn draw_completions(
    frame: &mut Frame<'_>,
    above: Rect,
    rows: &[ListRow],
    picked: usize,
) {
    if rows.is_empty() || above.height == 0 {
        return;
    }
    let shown = rows.len().min(above.height as usize).min(COMPLETION_ROWS);
    let first = picked.saturating_sub(shown.saturating_sub(1));
    let area = Rect {
        x: above.x,
        y: above.y + above.height - shown as u16,
        width: above.width,
        height: shown as u16,
    };
    frame.render_widget(Clear, area);
    let lines: Vec<Line<'static>> =
        list_lines(rows, picked, area.width as usize, ListLayout::Column)
            .into_iter()
            .skip(first)
            .take(shown)
            .collect();
    frame.render_widget(
        Paragraph::new(Text::from(lines)).style(Style::default().bg(COMPOSER_BACKGROUND)),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::tui::model::Settings;

    fn plain(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    fn settings() -> Settings {
        Settings {
            show_session_id_on_create: false,
            turn_end_summary: crate::config::TurnEndFact::DEFAULT_SUMMARY.to_vec(),
        }
    }

    fn entry(item: Item, expanded: bool) -> Entry {
        Entry {
            id: 1,
            item,
            expanded,
            revision: 1,
        }
    }

    fn tool_call(state: ToolState, output: &str, result: Option<&str>) -> Item {
        Item::ToolCall {
            id: "1".into(),
            name: "shell_execute".into(),
            input: Some(serde_json::json!({"command": "ls -la"})),
            summary: Some("ls -la".into()),
            state,
            output: output.into(),
            result: result.map(str::to_string),
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: None,
        }
    }

    /// While an answer streams, the row after its last newline is plain text, so a token shows
    /// at once; once it ends, the same row is markdown.
    #[test]
    fn the_open_row_is_plain_until_the_answer_ends() {
        let streaming = entry(
            Item::Text {
                text: "Done.\n\n- item".into(),
                streaming: true,
            },
            false,
        );
        let rows = plain(&entry_lines(&streaming, 40));
        assert_eq!(
            rows.last().map(String::as_str),
            Some("  - item"),
            "{rows:?}"
        );
        let ended = entry(
            Item::Text {
                text: "Done.\n\n- item".into(),
                streaming: false,
            },
            false,
        );
        let rows = plain(&entry_lines(&ended, 40));
        assert!(
            rows.last()
                .is_some_and(|row| row.contains('•') && row.ends_with("item")),
            "the same row is a bullet once the answer ends: {rows:?}"
        );
    }

    /// Every kind of item puts its glyph in column 0 and its text at the edge, so the whole
    /// transcript shares one left margin.
    #[test]
    fn every_item_keeps_its_glyph_in_the_gutter_and_its_text_at_the_edge() {
        let cases: Vec<(Item, &str)> = vec![
            (
                Item::Text {
                    text: "Answer.".into(),
                    streaming: false,
                },
                "  Answer.",
            ),
            (
                Item::Thinking {
                    text: "Hmm.".into(),
                    estimate: None,
                    ended: true,
                },
                "• Thinking · Hmm.",
            ),
            (
                tool_call(ToolState::Done { is_error: false }, "", None),
                "• shell_execute ls -la",
            ),
            (
                Item::Notice {
                    level: crate::frontend::NoticeLevel::Info,
                    text: "slow today".into(),
                },
                "• slow today",
            ),
            (
                Item::Notice {
                    level: crate::frontend::NoticeLevel::Warn,
                    text: "careful".into(),
                },
                "! careful",
            ),
            (
                Item::note("checklist: items still open, continuing"),
                "• checklist: items still open, continuing",
            ),
            (
                Item::note("stopping 2 background tasks"),
                "• stopping 2 background tasks",
            ),
            (Item::interrupted(), "! interrupted"),
            (Item::Error("provider error".into()), "■ provider error"),
            (Item::note("Rewound 1 turn."), "• Rewound 1 turn."),
            (Item::note("Approvals on"), "• Approvals on"),
            (Item::ShellOutput("total 0".into()), "  total 0"),
            (
                Item::SessionId {
                    label: "Resuming session".into(),
                    id: "abc".into(),
                },
                "• Resuming session abc",
            ),
            (Item::Image, "• image"),
        ];
        for (item, expected) in cases {
            let rows = plain(&entry_lines(&entry(item.clone(), false), 60));
            assert_eq!(rows[0], expected, "{item:?}");
        }
        let error = entry_lines(&entry(Item::Error("x".into()), false), 60);
        assert_eq!(error[0].spans[0].style.fg, Some(Color::Red));
        let warning = entry_lines(
            &entry(
                Item::Notice {
                    level: crate::frontend::NoticeLevel::Warn,
                    text: "x".into(),
                },
                false,
            ),
            60,
        );
        assert_eq!(warning[0].spans[0].style.fg, Some(Color::Yellow));
    }

    /// No row an item lays out is wider than the width it was laid out at, whatever the content,
    /// because the transcript clips what is wider: a token longer than the row, an address, a code
    /// row, a table cell, a command, a diff line, a checklist item, a path. And no row holds a tab,
    /// because the screen hands cells to the terminal and drops a tab with the indentation it
    /// carried. Every kind, expanded and collapsed, at every width.
    #[test]
    fn no_item_lays_out_a_row_wider_than_its_width_or_holding_a_tab() {
        let token = "x".repeat(300);
        let address = format!("https://example.invalid/{}", "p/".repeat(150));
        let markdown = format!(
            "# Heading {token}\n\nProse {address} end.\n\nInline `{token}` code.\n\n> Quote \
             {token}\n\n- Bullet {token}\n1. Numbered {address}\n\n| Column {token} | Other |\n\
             |---|---|\n| cell {address} | {token} |\n\n```bash\n    --flag={token}\n\tindented\n```\n\n\
             Tail\there."
        );
        let call =
            |name: &str, input: serde_json::Value, state: ToolState, change| Item::ToolCall {
                id: "1".into(),
                name: name.into(),
                input: Some(input),
                summary: Some(token.clone()),
                state,
                output: format!("\x1b[31m{token}\x1b[0m {address}\n\tindented\n"),
                result: Some(format!("{address}\tend")),
                result_images: 0,
                activity: format!("{token}\tnow"),
                progress: Some(format!("{address}\tstep")),
                change,
            };
        let command = serde_json::json!({ "command": format!("\trun --flag={token} {address}") });
        let edit = serde_json::json!({ "path": format!("{token}.rs"), "old_string": "a", "new_string": "b" });
        let change = super::super::model::FileChange {
            path: format!("{token}.rs"),
            old: Some(format!("old {token}\n")),
            new: format!("new {address}\n\tsecond {token}\n"),
        };
        let items = vec![
            Item::Prompt {
                text: format!("{token} {address}\n\tindented {token}"),
                class: Some(InboxClass::Steer),
                inbox_id: None,
                delivered: false,
            },
            Item::Text {
                text: markdown.clone(),
                streaming: false,
            },
            Item::Text {
                text: markdown.clone(),
                streaming: true,
            },
            Item::Thinking {
                text: markdown.clone(),
                estimate: Some(12),
                ended: true,
            },
            Item::Thinking {
                text: markdown.clone(),
                estimate: None,
                ended: false,
            },
            call(
                "shell_execute",
                command.clone(),
                ToolState::Done { is_error: false },
                None,
            ),
            call(
                "shell_execute",
                command.clone(),
                ToolState::Done { is_error: true },
                None,
            ),
            call("shell_execute", command.clone(), ToolState::Running, None),
            call("shell_execute", command, ToolState::Composing, None),
            call(
                "file_edit",
                edit,
                ToolState::Done { is_error: false },
                Some(change),
            ),
            Item::Checklist {
                items: vec![
                    crate::checklist::ChecklistItem {
                        id: 1,
                        text: format!("{token}\tnow"),
                        status: crate::checklist::ChecklistStatus::InProgress,
                        reason: None,
                        task: None,
                    },
                    crate::checklist::ChecklistItem {
                        id: 2,
                        text: address.clone(),
                        status: crate::checklist::ChecklistStatus::Deferred,
                        reason: Some(token.clone()),
                        task: Some(address.clone()),
                    },
                ],
            },
            Item::Notice {
                level: crate::frontend::NoticeLevel::Info,
                text: address.clone(),
            },
            Item::Notice {
                level: crate::frontend::NoticeLevel::Warn,
                text: format!("{token}\t!"),
            },
            Item::Error(format!("{token}\t{address}")),
            Item::ShellOutput(format!("\x1b[32m{token}\x1b[0m\n\t{address}")),
            Item::SessionId {
                label: token.clone(),
                id: address.clone(),
            },
            Item::TurnEnd(format!(
                "Finished at 19:28 after 19m 19s with {token} tokens in"
            )),
            Item::Image,
            Item::Banner {
                version: token.clone(),
                profile: token.clone(),
                model: Some(address.clone()),
                directory: address.clone(),
            },
        ];
        let mut too_wide = Vec::new();
        for width in [24, 40, 80, 200] {
            for (index, item) in items.iter().enumerate() {
                for expanded in [false, true] {
                    let lines = entry_lines(&entry(item.clone(), expanded), width);
                    for line in &lines {
                        let text: String = line
                            .spans
                            .iter()
                            .map(|span| span.content.as_ref())
                            .collect();
                        let head: String = text.chars().take(60).collect();
                        let case = format!("item {index} at {width} columns, expanded {expanded}");
                        if display_width(&text) > width {
                            too_wide
                                .push(format!("{case}: {} wide: {head:?}", display_width(&text)));
                        }
                        if text.contains('\t') {
                            too_wide.push(format!("{case}: holds a tab: {head:?}"));
                        }
                    }
                }
            }
        }
        assert!(too_wide.is_empty(), "{too_wide:#?}");
    }

    /// A collapsed thinking block is one row; expanded it is the label and the text under it.
    #[test]
    fn a_thinking_block_collapses_to_one_row() {
        let item = Item::Thinking {
            text: "First line.\n\nSecond paragraph.".into(),
            estimate: None,
            ended: true,
        };
        let collapsed = plain(&entry_lines(&entry(item.clone(), false), 60));
        assert_eq!(collapsed, vec![
            "• Thinking · First line. Second paragraph."
        ]);
        let expanded = plain(&entry_lines(&entry(item, true), 60));
        assert_eq!(expanded[0], "• Thinking");
        assert_eq!(expanded[1], "  First line.", "{expanded:?}");
        assert_eq!(
            expanded.last().map(String::as_str),
            Some("  Second paragraph.")
        );
    }

    /// A collapsed tool call is its row and a preview of its result under the connector;
    /// expanded it shows every argument and the whole output.
    #[test]
    fn a_tool_call_expands_to_its_arguments_and_output() {
        let output: String = (1..=8).map(|n| format!("line {n}\n")).collect();
        let item = tool_call(ToolState::Done { is_error: false }, &output, None);
        let collapsed = plain(&entry_lines(&entry(item.clone(), false), 60));
        assert_eq!(collapsed, vec![
            "• shell_execute ls -la",
            "  └ line 1",
            "    line 2",
            "    line 3",
            "    … +5 lines",
        ]);
        let expanded = plain(&entry_lines(&entry(item, true), 60));
        assert_eq!(
            expanded[0], "• shell_execute",
            "the row carries no summary once the arguments are under it"
        );
        assert_eq!(
            expanded[1], "  $ ls -la",
            "the command hangs under the row behind its lead: {expanded:?}"
        );
        assert!(
            expanded.iter().any(|row| row == "  └ line 1"),
            "the output sits on the elbow: {expanded:?}"
        );
        assert!(
            expanded.iter().any(|row| row == "    line 4"),
            "every row of the output is shown: {expanded:?}"
        );
        let running = plain(&entry_lines(
            &entry(tool_call(ToolState::Running, "", None), false),
            60,
        ));
        assert_eq!(running, vec!["• shell_execute ls -la", "  └ running"]);
    }

    /// Expanded, a call shows every argument whole: a command longer than the row wraps under
    /// its key, with no marker in place of any of it, because what a cut hides is what the user
    /// is reading the call to see.
    #[test]
    fn an_expanded_call_shows_a_long_command_whole() {
        let command = format!(
            "seq 1 5000 | tr '\\n' ' ' | rev | {} | fold -w 40 | head -1",
            "grep -v nothing | ".repeat(12)
        );
        let item = Item::ToolCall {
            id: "1".into(),
            name: "shell_execute".into(),
            input: Some(serde_json::json!({ "command": command })),
            summary: Some(command.clone()),
            state: ToolState::Done { is_error: false },
            output: "2 TOTAL: 719\n".into(),
            result: None,
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: None,
        };
        let rows = plain(&entry_lines(&entry(item, true), 60));
        assert_eq!(rows[0], "• shell_execute", "{rows:?}");
        assert!(rows[1].starts_with("  $ seq 1 5000"), "{rows:?}");
        assert!(
            rows[2].starts_with("  │ "),
            "the continuation rows hang on the bar down to the result: {rows:?}"
        );
        let block: String = rows[1..]
            .iter()
            .take_while(|row| !row.contains('└'))
            .map(|row| row.trim_start().trim_start_matches(['$', '│', ' ']))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            block.contains("head -1") && block.contains("seq 1 5000"),
            "both ends of the command are in the block: {rows:?}"
        );
        assert_eq!(
            block.matches("grep -v nothing").count(),
            12,
            "nothing in the middle is cut: {rows:?}"
        );
        assert!(
            rows[1..]
                .iter()
                .all(|row| !row.contains("...") && !row.contains('…')),
            "no marker stands in for any of it: {rows:?}"
        );
    }

    /// A preview row that does not fit stops at the edge with `…` rather than wrapping, and
    /// the count under it is a count of lines; expanded, the same row wraps whole.
    #[test]
    fn a_preview_row_is_cut_at_the_edge_and_an_expanded_one_wraps() {
        let output = format!(
            "ID        Status     Tool                      What\n{}\n\nA finished task's full \
             result is delivered to you on its own.\n",
            "dca8209a  completed  mcp__exa__web_search_exa  top news headlines for today"
        );
        let item = tool_call(ToolState::Done { is_error: false }, &output, None);
        let collapsed = plain(&entry_lines(&entry(item.clone(), false), 40));
        assert_eq!(collapsed, vec![
            "• shell_execute ls -la",
            "  └ ID        Status     Tool          …",
            "    dca8209a  completed  mcp__exa__web_…",
            "    … +2 lines",
        ]);
        assert!(
            collapsed.iter().all(|row| display_width(row) <= 40),
            "{collapsed:?}"
        );
        let expanded = plain(&entry_lines(&entry(item, true), 40));
        assert!(
            expanded.iter().any(|row| row.ends_with("What"))
                && expanded.iter().any(|row| row.contains("for today")),
            "expanded, the long rows wrap whole: {expanded:?}"
        );
        assert!(
            expanded.iter().all(|row| !row.contains('…')),
            "and nothing is cut: {expanded:?}"
        );
    }

    /// A call that changed a file draws the change as a diff in place of the words the tool
    /// wrote, with the counts on its row; collapsed, the first three rows are the line before,
    /// the line taken out and the line put in.
    #[test]
    fn a_file_edit_draws_its_diff_under_the_row() {
        let item = Item::ToolCall {
            id: "1".into(),
            name: "file_edit".into(),
            input: Some(serde_json::json!({
                "path": "hello.py",
                "old_string": "print(\"Hello, World!\")",
                "new_string": "print(\"Hello, world!\")"
            })),
            summary: Some("hello.py".into()),
            state: ToolState::Done { is_error: false },
            output: String::new(),
            result: Some("Successfully edited 'hello.py': 1 occurrence(s)".into()),
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: Some(super::super::model::FileChange {
                path: "hello.py".into(),
                old: Some("import sys\nprint(\"Hello, World!\")\nsys.exit(0)\n".into()),
                new: "import sys\nprint(\"Hello, world!\")\nsys.exit(0)\n".into(),
            }),
        };
        let collapsed = plain(&entry_lines(&entry(item.clone(), false), 60));
        assert_eq!(
            collapsed,
            vec![
                "• file_edit hello.py  +1 -1",
                "  └ 1   import sys",
                "    2 - print(\"Hello, World!\")",
                "    2 + print(\"Hello, world!\")",
                "    3   sys.exit(0)",
            ],
            "a small change shows whole even collapsed"
        );
        let rewrite = Item::ToolCall {
            id: "2".into(),
            name: "file_write".into(),
            input: Some(serde_json::json!({"path": "big.py"})),
            summary: Some("big.py".into()),
            state: ToolState::Done { is_error: false },
            output: String::new(),
            result: Some("Successfully wrote".into()),
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: Some(super::super::model::FileChange {
                path: "big.py".into(),
                old: None,
                new: (1..=60).map(|n| format!("line {n}\n")).collect(),
            }),
        };
        let collapsed = plain(&entry_lines(&entry(rewrite, false), 60));
        assert_eq!(
            collapsed.len(),
            1 + draw::DIFF_PREVIEW_ROWS + 1,
            "a long change shows its first rows and counts the rest: {collapsed:?}"
        );
        assert_eq!(
            collapsed.last().map(String::as_str),
            Some("    … +40 lines")
        );
        let expanded = plain(&entry_lines(&entry(item, true), 60));
        assert!(
            expanded.iter().any(|row| row == "    3   sys.exit(0)"),
            "expanded, the line after shows too: {expanded:?}"
        );
        assert!(
            !expanded.iter().any(|row| row.contains("Successfully")),
            "the tool's words are not drawn beside the diff: {expanded:?}"
        );
        assert!(
            expanded.iter().any(|row| row == "  │ path: hello.py")
                && !expanded.iter().any(|row| row.contains("old_string")),
            "the arguments the diff already shows are left out of the block: {expanded:?}"
        );
    }

    /// A failed call says so under its row, in red, whether or not it is expanded.
    #[test]
    fn a_failed_call_says_so_under_its_row() {
        let item = Item::ToolCall {
            id: "1".into(),
            name: "file_read".into(),
            input: Some(serde_json::json!({"path": "/nope"})),
            summary: None,
            state: ToolState::Done { is_error: true },
            output: String::new(),
            result: Some("no such file".into()),
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: None,
        };
        let lines = entry_lines(&entry(item, false), 60);
        assert_eq!(plain(&lines), vec![
            "• file_read /nope",
            "  └ failed: no such file"
        ]);
        let failed = &lines[1].spans[1];
        assert_eq!(failed.content.as_ref(), "failed: ");
        assert_eq!(failed.style.fg, Some(Color::Red));
        let mut long = item_with_a_failure();
        if let Item::ToolCall { result, .. } = &mut long {
            *result = Some(format!("{} {}", "a".repeat(20), "b".repeat(20)));
        }
        let expanded = plain(&entry_lines(&entry(long, true), 40));
        assert_eq!(
            expanded,
            vec![
                "• file_read",
                "  │ path: /nope",
                &format!("  └ failed: {}", "a".repeat(20)),
                &format!("            {}", "b".repeat(20)),
            ],
            "the label shares the first row and the rest hangs under the text"
        );
    }

    fn item_with_a_failure() -> Item {
        Item::ToolCall {
            id: "1".into(),
            name: "file_read".into(),
            input: Some(serde_json::json!({"path": "/nope"})),
            summary: None,
            state: ToolState::Done { is_error: true },
            output: String::new(),
            result: Some("no such file".into()),
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: None,
        }
    }

    /// The banner is meka's mark in cyan beside three lines on no background, two columns in
    /// from the edge under a blank row: the name bold and white with the version dimmed, the
    /// profile with its model and the directory a step down in the palette's plain white; a
    /// profile whose model the config leaves to the backend shows its name alone.
    #[test]
    fn the_banner_is_the_mark_beside_three_lines() {
        let item = Item::Banner {
            version: "0.72.0".into(),
            profile: "synthetic".into(),
            model: Some("hf:zai-org/GLM-5.3".into()),
            directory: "~/projects/something".into(),
        };
        let lines = entry_lines(&entry(item, false), 80);
        assert_eq!(plain(&lines), vec![
            "",
            "  █▄ ▄█  meka v0.72.0",
            "  █ ▀ █  synthetic (hf:zai-org/GLM-5.3)",
            "  █   █  ~/projects/something",
        ]);
        assert!(
            lines
                .iter()
                .flat_map(|line| line.spans.iter())
                .all(|span| span.style.bg.is_none()),
            "nothing has a background: {lines:?}"
        );
        assert!(
            lines[1..]
                .iter()
                .all(|line| line.spans[1].style.fg == Some(Color::Cyan)),
            "the mark is cyan: {lines:?}"
        );
        let name = &lines[1].spans[3];
        assert_eq!(name.content.as_ref(), "meka");
        assert_eq!(name.style.fg, Some(draw::WHITE));
        assert!(
            name.style.add_modifier.contains(Modifier::BOLD),
            "the name is bold: {name:?}"
        );
        let version = &lines[1].spans[4];
        assert_eq!(version.content.as_ref(), " v0.72.0");
        assert_eq!(
            version.style.fg,
            Some(Color::DarkGray),
            "the version is dimmed"
        );
        assert_eq!(
            lines[2].spans[3].style.fg,
            Some(Color::Gray),
            "the facts are the palette's plain white, a step down from the name"
        );
        let bare = entry_lines(
            &entry(
                Item::Banner {
                    version: "0.72.0".into(),
                    profile: "work".into(),
                    model: None,
                    directory: "~".into(),
                },
                false,
            ),
            60,
        );
        assert_eq!(plain(&bare)[2], "  █ ▀ █  work");
    }

    /// A prompt keeps the zero-width joiners of what the user typed, so a family emoji stays
    /// one glyph rather than the four it is built from.
    #[test]
    fn a_prompt_keeps_its_joiners() {
        let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        let sent = entry(
            Item::Prompt {
                text: format!("look {family}"),
                class: None,
                inbox_id: None,
                delivered: true,
            },
            false,
        );
        let lines = entry_lines(&sent, 60);
        assert_eq!(lines[0].spans[1].content.as_ref(), format!("look {family}"));
    }

    /// A prompt that went to the inbox says what it is and whether the loop has read it.
    #[test]
    fn a_steer_says_it_is_waiting_until_delivered() {
        let waiting = entry(
            Item::Prompt {
                text: "also check the logs".into(),
                class: Some(InboxClass::Steer),
                inbox_id: None,
                delivered: false,
            },
            false,
        );
        let lines = entry_lines(&waiting, 60);
        let rows = plain(&lines);
        assert_eq!(rows, vec!["> also check the logs", "  (steer, waiting)"]);
        assert!(
            lines.iter().all(|line| line.width() == 60
                && line
                    .spans
                    .iter()
                    .all(|span| span.style.bg == Some(COMPOSER_BACKGROUND))),
            "every row is the composer's block across the whole width: {lines:?}"
        );
        let words = &lines[0].spans[1];
        assert_eq!(words.content.as_ref(), "also check the logs");
        assert!(
            words.style.fg == Some(draw::WHITE)
                && !words.style.add_modifier.contains(Modifier::BOLD),
            "the words are white, not bold: {words:?}"
        );
    }

    /// The list shows at most `COMPLETION_ROWS`, keeps the pick on screen, and lines the
    /// descriptions up in one column.
    #[test]
    fn the_completion_list_is_capped_and_its_descriptions_line_up() {
        let rows: Vec<ListRow> = (0..30)
            .map(|index| ListRow {
                value: format!("/c{index}"),
                description: Some(format!("command {index}")),
            })
            .collect();
        let backend = ratatui::backend::TestBackend::new(40, 20);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let shown = |terminal: &ratatui::Terminal<ratatui::backend::TestBackend>| -> Vec<String> {
            let buffer = terminal.backend().buffer();
            (0..buffer.area.height)
                .map(|y| {
                    (0..buffer.area.width)
                        .map(|x| buffer[(x, y)].symbol().to_string())
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                })
                .filter(|row| !row.is_empty())
                .collect()
        };
        terminal
            .draw(|frame| draw_completions(frame, frame.area(), &rows, 0))
            .expect("a frame");
        let drawn = shown(&terminal);
        assert_eq!(drawn.len(), COMPLETION_ROWS, "{drawn:#?}");
        assert_eq!(drawn[0], "  /c0   command 0");
        assert_eq!(drawn[9], "  /c9   command 9");
        terminal
            .draw(|frame| draw_completions(frame, frame.area(), &rows, 29))
            .expect("a frame");
        let drawn = shown(&terminal);
        assert_eq!(
            drawn[9], "  /c29  command 29",
            "the pick scrolls into view: {drawn:#?}"
        );
        let buffer = terminal.backend().buffer();
        let last_row = buffer.area.height - 1;
        let picked_cell = |x: u16| buffer[(x, last_row)].style();
        assert!(
            picked_cell(0).add_modifier.contains(Modifier::REVERSED)
                && picked_cell(buffer.area.width - 1)
                    .add_modifier
                    .contains(Modifier::REVERSED),
            "the pick is one bar from edge to edge"
        );
        assert_eq!(
            picked_cell(10).fg,
            picked_cell(0).fg,
            "nothing on the picked row is dimmed"
        );
    }

    /// The bar is dim, so it reads as a reminder rather than the prompt, and lights under the
    /// mouse.
    #[test]
    fn the_sticky_bar_is_dim_until_the_mouse_is_on_it() {
        let line = Line::from(vec![
            Span::styled("> ", Style::default().fg(draw::PROMPT_GLYPH_COLOR)),
            Span::styled("words", Style::default().fg(draw::WHITE)),
        ]);
        let backend = ratatui::backend::TestBackend::new(20, 1);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        for (hovered, expected) in [(false, Color::DarkGray), (true, draw::WHITE)] {
            terminal
                .draw(|frame| draw_sticky_prompt(frame, frame.area(), line.clone(), hovered))
                .expect("a frame");
            let buffer = terminal.backend().buffer();
            assert_eq!(
                buffer[(0, 0)].style().fg,
                Some(expected),
                "the glyph, hovered={hovered}"
            );
            assert_eq!(
                buffer[(2, 0)].style().fg,
                Some(expected),
                "the words, hovered={hovered}"
            );
            assert_eq!(
                buffer[(2, 0)].style().bg,
                Some(super::super::input::COMPOSER_BACKGROUND)
            );
        }
    }

    /// The pill sits centered in the row it is given, on the composer's gray; on a working row
    /// whose words reach the center it moves to the right edge, and stays centered where they
    /// do not.
    #[test]
    fn the_back_to_bottom_pill_is_centered_in_the_gap() {
        let backend = ratatui::backend::TestBackend::new(60, 1);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let mut pill = Rect::default();
        terminal
            .draw(|frame| pill = draw_back_to_bottom(frame, frame.area(), 0))
            .expect("a frame");
        let label_width = display_width(BACK_TO_BOTTOM) as u16 + 2;
        assert_eq!(pill.width, label_width);
        assert_eq!(pill.x, (60 - label_width) / 2);
        let cell = &terminal.backend().buffer()[(pill.x + 1, 0)];
        assert_eq!(cell.symbol(), "↓");
        assert_eq!(cell.bg, COMPOSER_BACKGROUND);
        terminal
            .draw(|frame| pill = draw_back_to_bottom(frame, frame.area(), 40))
            .expect("a frame");
        assert_eq!(pill.x, 60 - label_width, "clear of the working row's words");
        terminal
            .draw(|frame| pill = draw_back_to_bottom(frame, frame.area(), 10))
            .expect("a frame");
        assert_eq!(
            pill.x,
            (60 - label_width) / 2,
            "centered where the words end early"
        );
    }

    /// The working row names what the agent is doing, with the seconds and the stop key after
    /// it; the stop key goes first when the row is too narrow, and the row stands still, without
    /// it, while the turn waits on the user.
    #[test]
    fn the_working_row_says_what_the_agent_is_doing() {
        let text = |line: Line<'static>| -> String {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect()
        };
        let seconds = Duration::from_secs(12);
        assert_eq!(
            text(working_line(&Activity::Working, seconds, 0, 80)),
            "⠋ Working · 12s · Esc to stop"
        );
        assert_eq!(
            text(working_line(&Activity::UsingTool, seconds, 3, 80)),
            "⠸ Using a tool · 12s · Esc to stop"
        );
        assert_eq!(
            text(working_line(&Activity::Thinking, seconds, 0, 20)),
            "⠋ Thinking · 12s",
            "the stop key is the first to go"
        );
        assert_eq!(
            text(working_line(&Activity::Waiting, seconds, 7, 80)),
            "• Waiting for your answer · 12s"
        );
        assert_eq!(spinner_frame(Duration::from_millis(1050)), 0);
        assert_eq!(spinner_frame(Duration::from_millis(350)), 3);
    }

    /// The stream moves the word: a delta names the thinking or the answer, a call the tool
    /// use, and a call's end or a thinking block's leaves plain work; what says nothing about
    /// the agent's doing leaves the word alone.
    #[test]
    fn the_activity_follows_the_stream() {
        let mut activity = Activity::Working;
        activity.note(&FrontendEvent::ThinkingDelta("hm".into()));
        assert_eq!(activity, Activity::Thinking);
        activity.note(&FrontendEvent::ThinkingEnded);
        assert_eq!(activity, Activity::Working);
        activity.note(&FrontendEvent::ToolCallComposing {
            id: "1".into(),
            name: "file_read".into(),
        });
        assert_eq!(activity, Activity::UsingTool);
        activity.note(&FrontendEvent::ToolCallStarted {
            id: "1".into(),
            name: "file_read".into(),
            input: serde_json::json!({}),
            display_summary: None,
        });
        assert_eq!(activity, Activity::UsingTool);
        activity.note(&FrontendEvent::ChecklistUpdated { items: Vec::new() });
        assert_eq!(activity, Activity::UsingTool);
        activity.note(&FrontendEvent::ToolCallCompleted {
            id: "1".into(),
            name: "file_read".into(),
            is_error: false,
            content: Vec::new(),
            metadata: None,
        });
        assert_eq!(activity, Activity::Working);
        activity.note(&FrontendEvent::AssistantTextDelta("Sure".into()));
        assert_eq!(activity, Activity::Answering);
    }

    /// A flag drawn one frame and gone the next leaves its second cell untouched by ratatui's
    /// own diff; the refresh makes that cell rewrite itself.
    #[test]
    fn the_cell_after_a_departed_flag_is_rewritten() {
        let area = Rect::new(0, 0, 10, 1);
        let mut before = Buffer::empty(area);
        before.set_string(0, 0, "\u{1F1FA}\u{1F1F8} x", Style::default());
        let wide = wide_symbols(&before);
        assert_eq!(wide, vec![(0, 0, 2)]);
        let mut after = Buffer::empty(area);
        after.set_string(0, 0, "a  x", Style::default());
        let untouched: Vec<(u16, u16)> = before
            .diff(&after)
            .into_iter()
            .map(|(x, y, _)| (x, y))
            .collect();
        assert!(
            !untouched.contains(&(1, 0)),
            "ratatui alone leaves the flag's second cell alone: {untouched:?}"
        );
        refresh_after_wide_symbols(&mut after, &wide);
        let refreshed: Vec<(u16, u16)> = before
            .diff(&after)
            .into_iter()
            .map(|(x, y, _)| (x, y))
            .collect();
        assert!(refreshed.contains(&(1, 0)), "{refreshed:?}");
    }

    /// The status row opens with the permission level, a cell of its color and its name in it,
    /// says the elapsed time only while the host works, and puts a hint against the right edge.
    #[test]
    fn the_status_row_shows_the_work_only_while_it_runs_and_the_hint_on_the_right() {
        let text = |line: Line<'static>| -> String {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect()
        };
        let idle = status_line(
            &StatusFacts {
                shown: &crate::config::StatusFact::ALL,
                permission: crate::permission::Permission::Workspace,
                path: Some("~/projects".into()),
                profile: Some("work (claude-opus-4-8)".into()),
                context: Some("34%".into()),
                work: RunningWork::default(),
                hint: None,
                shell_mode: false,
            },
            60,
        );
        assert_eq!(
            text(idle.clone()),
            "  workspace · ~/projects · work (claude-opus-4-8) · 34%"
        );
        let level = color(crate::permission::Permission::Workspace.indicator_color());
        let cell = &idle.spans[0];
        assert_eq!(cell.content.as_ref(), " ");
        assert_eq!(
            cell.style.bg,
            Some(level),
            "the first cell is the level's color"
        );
        assert_eq!(
            idle.spans[2].style.fg,
            Some(level),
            "and the name wears it as text"
        );
        assert_eq!(idle.spans[2].style.bg, None);
        let busy = status_line(
            &StatusFacts {
                shown: &crate::config::StatusFact::DEFAULT_ROW,
                permission: crate::permission::Permission::Workspace,
                path: None,
                profile: None,
                context: None,
                work: RunningWork {
                    tasks: 2,
                    subagents: 1,
                },
                hint: Some("hold Shift to select"),
                shell_mode: false,
            },
            100,
        );
        let row = text(busy);
        assert!(
            row.starts_with("  workspace · 2 background tasks · 1 sub-agent"),
            "the work running beside the turn is counted: {row:?}"
        );
        assert!(row.ends_with("hold Shift to select"), "{row:?}");
        assert_eq!(
            display_width(&row),
            100 - 1,
            "the hint ends one column from the right edge: {row:?}"
        );
        let one = status_line(
            &StatusFacts {
                shown: &crate::config::StatusFact::DEFAULT_ROW,
                permission: crate::permission::Permission::Read,
                path: None,
                profile: None,
                context: None,
                work: RunningWork {
                    tasks: 1,
                    subagents: 0,
                },
                hint: None,
                shell_mode: false,
            },
            60,
        );
        assert_eq!(text(one), "  read · 1 background task");
    }

    /// The row opens with the level whatever the list says, then shows the facts the config
    /// lists in the config's order and nothing else; with no list, the level and the hint.
    #[test]
    fn the_status_row_follows_the_configured_list() {
        use crate::config::StatusFact;
        let text = |line: Line<'static>| -> String {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect()
        };
        let facts = |shown: &'static [StatusFact], hint: Option<&'static str>| StatusFacts {
            shown,
            permission: crate::permission::Permission::Read,
            path: Some("~/x".into()),
            profile: Some("work (m)".into()),
            context: Some("34%".into()),
            work: RunningWork {
                tasks: 1,
                subagents: 0,
            },
            hint,
            shell_mode: false,
        };
        assert_eq!(
            text(status_line(
                &facts(&[StatusFact::Profile, StatusFact::Path], None),
                60
            )),
            "  read · work (m) · ~/x"
        );
        assert_eq!(
            text(status_line(&facts(&[StatusFact::Context], None), 60)),
            "  read · 34%"
        );
        let bare = text(status_line(&facts(&[], Some("copied 2 rows")), 30));
        assert!(
            bare.starts_with("  read ") && bare.ends_with("copied 2 rows"),
            "{bare:?}"
        );
        let mut shell = facts(&[StatusFact::Path], Some("copied 2 rows"));
        shell.shell_mode = true;
        let line = status_line(&shell, 60);
        assert_eq!(
            text(line.clone()),
            format!(
                "  {SHELL_MODE_LABEL}{}copied 2 rows",
                " ".repeat(60 - 1 - 2 - SHELL_MODE_LABEL.len() - 13)
            ),
            "the mode takes the facts' place; the hint keeps the edge"
        );
        let label = line.spans.first().expect("the label");
        assert_eq!(
            label.style.fg,
            Some(Color::Red),
            "the mode is red, as the composer's glyph is"
        );
    }

    /// A collapsible item lights up under the mouse: its dimmed text turns white while the
    /// pointer is on one of its rows, and goes back to gray when the pointer leaves. An item
    /// that cannot expand does not answer.
    #[test]
    fn a_collapsible_item_lights_up_under_the_mouse() {
        let mut transcript = Transcript::new(settings());
        transcript.push(Item::note("outcome"));
        transcript.push(Item::Thinking {
            text: "Hmm.".into(),
            estimate: None,
            ended: true,
        });
        let mut viewport = Viewport::new();
        let backend = ratatui::backend::TestBackend::new(40, 6);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let glyph_fg = |terminal: &ratatui::Terminal<ratatui::backend::TestBackend>, row: u16| {
            terminal.backend().buffer()[(0, row)].fg
        };
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
            .expect("a frame");
        assert_eq!(
            glyph_fg(&terminal, 2),
            Color::DarkGray,
            "the thinking row starts dim"
        );
        assert!(
            viewport.set_hover_row(Some(2)),
            "the light moved onto the item"
        );
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
            .expect("a frame");
        assert_eq!(
            glyph_fg(&terminal, 2),
            draw::WHITE,
            "and lights up under the mouse"
        );
        assert_eq!(
            glyph_fg(&terminal, 0),
            Color::DarkGray,
            "the row above is untouched"
        );
        assert!(viewport.set_hover_row(Some(0)), "the light moved off it");
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
            .expect("a frame");
        assert_eq!(
            glyph_fg(&terminal, 2),
            Color::DarkGray,
            "it goes back to gray"
        );
        assert_eq!(
            glyph_fg(&terminal, 0),
            Color::DarkGray,
            "an outcome row cannot expand, so it does not answer"
        );
    }

    /// A page loaded above the view leaves what is on screen where it is: the view is anchored
    /// to the item at its top, not to a line number.
    #[test]
    fn a_page_loaded_above_leaves_the_view_where_it_was() {
        use crate::{
            conversation::{ContentBlock, Event, Message, Role},
            store::{LogPage, LogRow},
        };
        let page = |first: i64, words: &[&str], more: bool| LogPage {
            rows: words
                .iter()
                .enumerate()
                .map(|(offset, text)| LogRow {
                    turn_id: None,
                    id: first + offset as i64,
                    event: Event::Append(Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::Text {
                            text: (*text).to_string(),
                        }],
                    }),
                })
                .collect(),
            more,
        };
        let mut transcript = Transcript::new(settings());
        transcript.set_log_end(Some(20));
        transcript.load_page_above(page(
            11,
            &[
                "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20",
            ],
            true,
        ));
        let mut viewport = Viewport::new();
        let backend = ratatui::backend::TestBackend::new(20, 6);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let top_row = |terminal: &ratatui::Terminal<ratatui::backend::TestBackend>| {
            let buffer = terminal.backend().buffer();
            (0..buffer.area.width)
                .map(|x| buffer[(x, 0)].symbol().to_string())
                .collect::<String>()
                .trim_end()
                .to_string()
        };
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
            .expect("a frame");
        viewport.scroll_by(-7);
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
            .expect("a frame");
        let held = top_row(&terminal);
        assert!(
            held.starts_with("  r1"),
            "scrolled up into the page: {held:?}"
        );
        transcript.load_page_above(page(1, &["r1", "r2", "r3", "r4", "r5"], false));
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
            .expect("a frame");
        assert_eq!(
            top_row(&terminal),
            held,
            "the rows loaded above did not move what was on screen"
        );
    }

    /// The viewport maps a row of the last frame back to the item drawn there, gaps excluded.
    #[test]
    fn a_row_maps_back_to_the_item_drawn_on_it() {
        let mut transcript = Transcript::new(settings());
        transcript.push(Item::note("one"));
        transcript.push(Item::note("two"));
        let mut viewport = Viewport::new();
        let backend = ratatui::backend::TestBackend::new(40, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
            .expect("a frame");
        assert_eq!(viewport.item_at_row(0), Some(0));
        assert_eq!(viewport.item_at_row(1), None, "the gap belongs to nobody");
        assert_eq!(viewport.item_at_row(2), Some(1));
        assert_eq!(viewport.item_at_row(3), None);
    }

    /// Following the bottom holds as items arrive; scrolling up releases it and reaching the
    /// bottom again takes it back.
    #[test]
    fn following_releases_on_scroll_up_and_resumes_at_the_bottom() {
        let mut transcript = Transcript::new(settings());
        for index in 0..20 {
            transcript.push(Item::note(format!("row {index}")));
        }
        let mut viewport = Viewport::new();
        let backend = ratatui::backend::TestBackend::new(40, 6);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let mut frame = |viewport: &mut Viewport| {
            terminal
                .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
                .expect("a frame");
        };
        frame(&mut viewport);
        assert!(viewport.is_following());
        viewport.scroll_by(-3);
        assert!(!viewport.is_following());
        frame(&mut viewport);
        viewport.scroll_by(3);
        assert!(viewport.is_following(), "the bottom resumes following");
    }

    /// The prompt nearest above the top row sticks as a bar; a jump puts it just under the bar,
    /// which then holds the prompt before it, and the first prompt at the very top sticks nothing.
    #[test]
    fn the_last_prompt_above_the_view_sticks_and_a_jump_walks_back() {
        let mut transcript = Transcript::new(settings());
        let prompt = |text: &str| Item::Prompt {
            text: text.into(),
            class: None,
            inbox_id: None,
            delivered: true,
        };
        transcript.push(prompt("first"));
        transcript.push(Item::Text {
            text: (1..=8).map(|n| format!("a{n}\n\n")).collect(),
            streaming: false,
        });
        transcript.push(prompt("second"));
        transcript.push(Item::Text {
            text: (1..=8).map(|n| format!("b{n}\n\n")).collect(),
            streaming: false,
        });
        let mut viewport = Viewport::new();
        let backend = ratatui::backend::TestBackend::new(40, 6);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let mut draw = |viewport: &mut Viewport| {
            terminal
                .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
                .expect("a frame");
        };
        draw(&mut viewport);
        let (line, start) = viewport
            .sticky_prompt()
            .expect("the second prompt is above");
        assert_eq!(plain(&[line])[0], "> second");
        viewport.scroll_to(start.saturating_sub(1));
        draw(&mut viewport);
        let (line, start) = viewport
            .sticky_prompt()
            .expect("with the second at the top, the first sticks");
        assert_eq!(plain(&[line])[0], "> first");
        assert!(!viewport.is_following());
        viewport.scroll_to(start.saturating_sub(1));
        draw(&mut viewport);
        assert!(
            viewport.sticky_prompt().is_none(),
            "the first prompt is on screen, nothing is above it"
        );
    }
}
