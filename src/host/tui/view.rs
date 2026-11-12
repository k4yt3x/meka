//! How the transcript is drawn: each item as lines at a width, cached by the item's revision; the
//! viewport that scrolls them and maps a click to an item; the status row; and the rows the
//! screen draws between and under the transcript.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Duration,
};

use ratatui::{
    Frame,
    buffer::{Buffer, Cell, CellDiffOption},
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Clear, Paragraph},
};

use super::{
    input::COMPOSER_BACKGROUND,
    model::{Entry, Item, RanUnder, ToolState, Transcript},
};
use crate::{
    config::PromptPadding,
    frontend::FrontendEvent,
    render::tui::{self as draw, ACT_GLYPH, ERROR_GLYPH, TEXT_EDGE, WARNING_GLYPH, gray},
    store::inbox::InboxClass,
    text::{display_width, wrap_to_width},
};

/// Rows left between two items.
const ITEM_GAP: usize = 1;
/// The icon on a compaction's boundary row: two arrows pressing in, as Zed's icon of four does.
const COMPACTION_ICON: &str = "→←";
/// The words on a compaction's boundary row.
const COMPACTION_LABEL: &str = "Context compacted";
/// What the gap row says while the view is scrolled up from the bottom.
pub(crate) const BACK_TO_BOTTOM: &str = "↓ back to bottom";
/// Meka's mark, the slanted `M` of its icon in quadrant-block pixels with a shadow one pixel to
/// its right: three rows of runs, one beside each of the banner's lines.
const BANNER_MARK: [&[(&str, MarkTone)]; 3] = [
    &[
        (" ██", MarkTone::Face),
        ("▖", MarkTone::FaceOnShadow),
        (" ▗██", MarkTone::Face),
        ("▌", MarkTone::Shadow),
    ],
    &[
        ("▐█", MarkTone::Face),
        ("▝▙", MarkTone::FaceOnShadow),
        ("▟", MarkTone::Face),
        ("▘", MarkTone::FaceOnShadow),
        ("█", MarkTone::Face),
        ("▌", MarkTone::FaceOnShadow),
    ],
    &[
        ("█", MarkTone::Face),
        ("▌", MarkTone::FaceOnShadow),
        (" ▀", MarkTone::Face),
        ("▘", MarkTone::Shadow),
        ("▐█", MarkTone::Face),
        ("▌", MarkTone::Shadow),
    ],
];
/// The mark's shadow, a deep blue that reads as depth beside the terminal's cyan.
const BANNER_SHADOW: Color = Color::Indexed(24);

/// How a run of the banner's mark is drawn: the letter's face, its shadow, or the face over the
/// shadow where one cell holds both, since a cell has one foreground.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkTone {
    Face,
    Shadow,
    FaceOnShadow,
}

impl MarkTone {
    fn style(self) -> Style {
        match self {
            Self::Face => Style::default().fg(Color::Cyan),
            Self::Shadow => Style::default().fg(BANNER_SHADOW),
            Self::FaceOnShadow => Style::default().fg(Color::Cyan).bg(BANNER_SHADOW),
        }
    }
}
/// The columns between the banner and the left edge of the screen; one row stands above it.
const BANNER_GAP: usize = 2;

/// The answer's text as rows, laid out as markdown, while it streams too.
fn text_rows(text: &str, streaming: bool, width: usize, reasoning: bool) -> Vec<draw::Row> {
    match (streaming, reasoning) {
        (true, _) => draw::streaming_rows(text, width, reasoning),
        (false, true) => draw::thinking_rows(text, width),
        (false, false) => draw::markdown_rows(text, width),
    }
}

/// The rows of an item some of whose rows do not copy as text, with how each copies: an answer,
/// an opened thinking block and a compaction's summary, drawn as markdown, a prompt, whose
/// padding copies as nothing, and a picture, whose half blocks are no text. `None` for any other
/// item, whose rows all copy as text. `prompt_padding` is `[display].prompt_padding`.
fn copied_rows_of(
    entry: &Entry,
    width: usize,
    prompt_padding: PromptPadding,
) -> Option<Vec<draw::Row>> {
    let inner = width.max(1).saturating_sub(TEXT_EDGE).max(1);
    match &entry.item {
        Item::Prompt {
            text,
            class,
            inbox_id: _,
            delivered,
            ran_under,
        } => Some(prompt_rows(
            text,
            *class,
            *delivered,
            *ran_under,
            width.max(1),
            prompt_padding,
        )),
        Item::Text { text, streaming } => Some(draw::edge_copy_rows(text_rows(
            text, *streaming, inner, false,
        ))),
        Item::Thinking { text, ended, .. } if entry.expanded && !text.trim().is_empty() => {
            let mut rows = vec![(draw::thinking_label_row(), draw::RowCopy::Text)];
            rows.extend(draw::edge_copy_rows(text_rows(text, !*ended, inner, true)));
            Some(rows)
        }
        Item::Image(picture) => {
            let rows = draw::picture_rows(picture, inner)?;
            Some(
                draw::edge_rows(rows)
                    .into_iter()
                    .map(|row| (row, draw::RowCopy::Padding))
                    .collect(),
            )
        }
        Item::CompactionSummary { text } => {
            let label = Line::from(vec![
                Span::styled(format!("{ACT_GLYPH:<TEXT_EDGE$}"), gray()),
                Span::styled("Summary".to_string(), gray()),
            ]);
            let mut rows = vec![(label, draw::RowCopy::Text)];
            rows.extend(draw::edge_copy_rows(draw::markdown_rows(text, inner)));
            Some(rows)
        }
        _ => None,
    }
}

/// The color of the bar down a sent prompt's left edge: its turn's permission level, as the
/// composer's bar is the level's now; red for a `!command`, as the composer's is in the shell
/// mode; and gray while nothing says the level, for a turn the log has no record of or one that
/// has not opened yet.
fn ran_under_color(ran_under: RanUnder) -> Color {
    match ran_under {
        RanUnder::Level(level) => color(level.indicator_color()),
        RanUnder::Shell => Color::Red,
        RanUnder::Unrecorded | RanUnder::Pending => draw::DIM_ON_SHADE,
    }
}

/// A sent prompt's block: the composer's gray, the words in white rather than the terminal's
/// foreground so they stand out of it.
const PROMPT_BLOCK: Style = Style::new().fg(draw::WHITE).bg(COMPOSER_BACKGROUND);
/// The glyph of a half padding row above the words: the cell's lower half, next to them.
const HALF_PADDING_ABOVE: &str = "▄";
/// The glyph of a half padding row below the words: the cell's upper half.
const HALF_PADDING_BELOW: &str = "▀";
/// A half padding row's cells: the glyph's half in the composer's gray, the other half the
/// terminal's own background.
const HALF_PADDING: Style = Style::new().fg(COMPOSER_BACKGROUND).bg(Color::Reset);

/// Which side of the words a padding row is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PaddingSide {
    Above,
    Below,
}

/// A padding row of `PromptPadding::Half` at `width`: the half of each cell next to the words in
/// the composer's gray, the other half left to the terminal, so the block seems to start half a
/// row from the words.
pub(crate) fn half_padding_row(side: PaddingSide, width: usize) -> Line<'static> {
    let glyph = match side {
        PaddingSide::Above => HALF_PADDING_ABOVE,
        PaddingSide::Below => HALF_PADDING_BELOW,
    };
    Line::from(Span::styled(glyph.repeat(width), HALF_PADDING))
}

/// Light `cell` as part of a selection, its colors swapped. A cell of a half padding row is lit as
/// a cell of a whole one: swapped as drawn, its gray would move to the half away from the words and
/// leave a dark line across the selection.
pub(crate) fn light_selected(cell: &mut Cell) {
    if matches!(cell.symbol(), HALF_PADDING_ABOVE | HALF_PADDING_BELOW)
        && Some(cell.fg) == HALF_PADDING.fg
        && Some(cell.bg) == HALF_PADDING.bg
    {
        cell.set_symbol(" ");
        cell.set_style(PROMPT_BLOCK);
    }
    cell.modifier |= Modifier::REVERSED;
}

/// A prompt as the block the composer is, kept once sent: the composer's bar down its left edge
/// in the color of what it ran under, and the words, with padding above and below them as
/// `padding` draws it, which copies as nothing. A half row carries no bar, since a cell that is
/// half the terminal's and half gray has no third color for it.
fn prompt_rows(
    text: &str,
    class: Option<InboxClass>,
    delivered: bool,
    ran_under: RanUnder,
    width: usize,
    padding: PromptPadding,
) -> Vec<draw::Row> {
    let inner = width.saturating_sub(TEXT_EDGE + draw::RIGHT_MARGIN).max(1);
    // The same gray across the whole row, so a prompt is told from the answer under it at a
    // glance.
    let block = PROMPT_BLOCK;
    let bar = Span::styled(
        format!("{:<TEXT_EDGE$}", draw::COMPOSER_BAR),
        block.fg(ran_under_color(ran_under)),
    );
    let block_row = |row: String, style: Style| {
        let fill = width.saturating_sub(TEXT_EDGE + display_width(&row));
        Line::from(vec![
            bar.clone(),
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
            lines.push((block_row(row, block), draw::RowCopy::Text));
        }
    }
    if let Some(class) = class {
        let word = match class {
            InboxClass::Steer => "steer",
            InboxClass::Followup => "followup",
            InboxClass::Interrupt => "interrupt",
        };
        let note = if delivered {
            format!("({word})")
        } else {
            format!("({word}, waiting)")
        };
        // The screen's word on the prompt, not part of it, so it copies as nothing.
        lines.push((
            block_row(
                note,
                Style::default()
                    .fg(draw::DIM_ON_SHADE)
                    .bg(COMPOSER_BACKGROUND),
            ),
            draw::RowCopy::Padding,
        ));
    }
    let (above, below) = match padding {
        PromptPadding::None => return lines,
        PromptPadding::Half => (
            half_padding_row(PaddingSide::Above, width),
            half_padding_row(PaddingSide::Below, width),
        ),
        PromptPadding::Full => {
            let pad = block_row(String::new(), block);
            (pad.clone(), pad)
        }
    };
    std::iter::once((above, draw::RowCopy::Padding))
        .chain(lines)
        .chain(std::iter::once((below, draw::RowCopy::Padding)))
        .collect()
}

/// One item as the lines it occupies at `width` and how each copies, its paths shown against
/// `working_directory`. How a row copies is said only where it differs from text.
/// `prompt_padding` is `[display].prompt_padding`.
pub(crate) fn entry_rows(
    entry: &Entry,
    width: usize,
    working_directory: &Path,
    prompt_padding: PromptPadding,
) -> (Vec<Line<'static>>, Vec<draw::RowCopy>) {
    let (lines, copies): (Vec<Line<'static>>, Vec<draw::RowCopy>) =
        match copied_rows_of(entry, width, prompt_padding) {
            Some(rows) => rows.into_iter().unzip(),
            None => (
                entry_lines(entry, width, working_directory, prompt_padding),
                Vec::new(),
            ),
        };
    // A row an item cannot lay out within a width too narrow for its fixed words (a note, a
    // count, a lead) is cut at the edge, as every other row is, rather than left to the terminal
    // to clip without a mark.
    let lines = lines
        .into_iter()
        .map(|line| {
            let text: String = line
                .spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect();
            if display_width(&text) <= width {
                return line;
            }
            Line {
                spans: draw::cut_spans(line.spans, width),
                ..line
            }
        })
        .collect();
    (lines, copies)
}

/// Whether a tool's result is drawn as an item of its own, so the call's row shows nothing of it.
fn result_is_its_own_item(name: &str) -> bool {
    name.starts_with("checklist_")
}

/// What a call's row says after its name. A program's first line, highlighted, with a leading `cd`
/// into the working directory dimmed, since it does nothing there; a path as the status row shows
/// a directory, relative to the working directory under it; otherwise the call's summary.
fn call_summary(
    name: &str,
    input: &serde_json::Value,
    summary: Option<&str>,
    working_directory: &Path,
    width: usize,
) -> Vec<Span<'static>> {
    if let Some((key, language)) = crate::tools::builtin_code_param(name)
        && let Some(command) = input.get(key).and_then(serde_json::Value::as_str)
    {
        let first = draw::command_lines(command, language, 1, width)
            .into_iter()
            .next()
            .map(|line| line.spans)
            .unwrap_or_default();
        return draw::dim_front(first, idle_cd_columns(command, working_directory));
    }
    let Some(text) = draw::tool_summary(name, input, summary) else {
        return Vec::new();
    };
    let text = if crate::tools::builtin_path_param(name) {
        shown_path(&text, working_directory)
    } else {
        text
    };
    vec![Span::styled(text, Style::default().fg(Color::Gray))]
}

/// The columns of a command's opening `cd` into `working_directory` and the `&&` or `;` after it,
/// which change nothing for a command that already runs there; 0 when it opens otherwise.
fn idle_cd_columns(command: &str, working_directory: &Path) -> usize {
    let Some(rest) = command.strip_prefix("cd ") else {
        return 0;
    };
    let target = rest.trim_start();
    let Some(end) = target.find(|character: char| character.is_whitespace() || character == ';')
    else {
        return 0;
    };
    let (path, after) = target.split_at(end);
    let path = path.trim_matches(|quote| quote == '\'' || quote == '"');
    if working_directory.as_os_str().is_empty()
        || Path::new(path.trim_end_matches('/')) != working_directory
    {
        return 0;
    }
    let after_path = after.trim_start();
    let Some(after_separator) = after_path
        .strip_prefix("&&")
        .or_else(|| after_path.strip_prefix(';'))
    else {
        return 0;
    };
    let idle = command.len() - after_separator.trim_start().len();
    display_width(&command[..idle])
}

/// A path a call names, as the status row shows a directory: relative to the working directory
/// under it, `~` for the home directory, and as written otherwise.
fn shown_path(value: &str, working_directory: &Path) -> String {
    let path = Path::new(value);
    if !path.is_absolute() {
        return value.to_string();
    }
    match path.strip_prefix(working_directory) {
        Ok(rest) if working_directory.as_os_str().is_empty() => rest.display().to_string(),
        Ok(rest) if rest.as_os_str().is_empty() => ".".to_string(),
        Ok(rest) => rest.display().to_string(),
        Err(_) => super::input::shorten_path_with_tilde(path),
    }
}

/// A collapsed call's program after its first line, which its row holds: the next
/// [`draw::COMMAND_PREVIEW_ROWS`] lines cut at the edge and a count of the rest, or every line
/// when only one more would be left to count.
fn command_continuation(command: &str, language: &str, width: usize) -> Vec<Line<'static>> {
    let rest = command.lines().count().saturating_sub(1);
    let shown = if rest > draw::COMMAND_PREVIEW_ROWS + 1 {
        draw::COMMAND_PREVIEW_ROWS
    } else {
        rest
    };
    // Highlighted from the first line, which the row holds, so the lines under it are read in
    // the state it leaves.
    let mut rows: Vec<Line<'static>> = draw::command_lines(command, language, 1 + shown, width)
        .into_iter()
        .skip(1)
        .map(|line| draw::cut_line(line, width))
        .collect();
    if shown < rest {
        rows.push(Line::from(Span::styled(
            format!("… +{} lines", rest - shown),
            gray(),
        )));
    }
    rows
}

/// `text` without the blank lines it opens with.
fn without_leading_blank_lines(text: &str) -> &str {
    let mut rest = text;
    while let Some((line, after)) = rest.split_once('\n') {
        if !line.trim().is_empty() {
            break;
        }
        rest = after;
    }
    rest
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
        exit_code,
        ..
    } = item
    else {
        return Vec::new();
    };
    let (state, result_images) = (*state, *result_images);
    let inner = width.saturating_sub(draw::connector_indent()).max(1);
    // Expanded, every line wraps whole; collapsed, a line is one row cut at the edge, since a
    // preview is a glance at how the result opens and its count is then a count of lines.
    let wrap = if expanded { inner } else { usize::MAX };
    let mut rows = Vec::new();
    // Whether the rows are a change's preview already, which counts what it leaves out itself.
    let mut previewed = false;
    if let Some(progress) = progress.as_deref() {
        rows.extend(draw::plain_lines(progress, wrap, gray()));
    }
    if !activity.is_empty() {
        rows.extend(draw::plain_lines(activity.trim_end(), wrap, gray()));
    }
    let failed = state == ToolState::Done { is_error: true };
    let streamed = Some(output.as_str()).filter(|text| !text.is_empty());
    // A finished call shows what the log keeps of it, which is what a replay shows; what it wrote
    // as it ran stands in until then.
    let shown = match state {
        ToolState::Done { .. } => result
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .or(streamed),
        _ => streamed,
    };
    match (shown, change) {
        // What the call changed is the result worth reading; the words the tool wrote about it
        // are the model's.
        (Some(_), Some(change)) if !failed => {
            if expanded {
                rows.extend(draw::change_rows(change, inner));
                let left_out = change.lines_left_out();
                if left_out > 0 {
                    rows.push(Line::from(Span::styled(
                        format!("… +{left_out} line{}", if left_out == 1 { "" } else { "s" }),
                        gray(),
                    )));
                }
            } else {
                rows.extend(draw::change_preview(change, inner, draw::DIFF_PREVIEW_ROWS));
                previewed = true;
            }
        }
        (Some(text), _) if failed || !result_is_its_own_item(name) => {
            let mut text = text.trim_end_matches('\n');
            // The row states the code, so the line the output ends with to say it is not shown.
            if let Some(code) = exit_code
                && let Some(output) = text.strip_suffix(&crate::tools::shell::exit_line(*code))
            {
                text = output.trim_end_matches('\n');
            }
            let mut lines = draw::ansi_lines(without_leading_blank_lines(text), wrap);
            // What a tool or Meka said of a failure is the failure, in red; what a program wrote
            // is its output, however it exited.
            let color = if failed && exit_code.is_none() {
                Color::Red
            } else {
                Color::DarkGray
            };
            for line in &mut lines {
                for span in &mut line.spans {
                    if span.style.fg.is_none() {
                        span.style = span.style.fg(color);
                    }
                }
            }
            rows.extend(lines);
        }
        _ => {}
    }
    if state == ToolState::Running && rows.is_empty() {
        rows.push(Line::from(Span::styled("Running".to_string(), gray())));
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
        return rows;
    }
    let rows = if previewed {
        rows
    } else {
        draw::preview_rows(rows, draw::PREVIEW_ROWS)
    };
    rows.into_iter()
        .map(|row| draw::cut_line(row, inner))
        .collect()
}

/// One item as the lines it occupies at `width`, its paths shown against `working_directory`.
/// `prompt_padding` is `[display].prompt_padding`.
pub(crate) fn entry_lines(
    entry: &Entry,
    width: usize,
    working_directory: &Path,
    prompt_padding: PromptPadding,
) -> Vec<Line<'static>> {
    let width = width.max(1);
    let inner = width.saturating_sub(TEXT_EDGE).max(1);
    match &entry.item {
        Item::Prompt { .. } => copied_rows_of(entry, width, prompt_padding)
            .unwrap_or_default()
            .into_iter()
            .map(|(line, _)| line)
            .collect(),
        Item::Text { .. } | Item::CompactionSummary { .. } => {
            copied_rows_of(entry, width, prompt_padding)
                .unwrap_or_default()
                .into_iter()
                .map(|(line, _)| line)
                .collect()
        }
        Item::Thinking {
            text,
            estimate,
            ended,
        } => match copied_rows_of(entry, width, prompt_padding) {
            Some(rows) => rows.into_iter().map(|(line, _)| line).collect(),
            None => vec![draw::thinking_row(text, *estimate, !*ended, width)],
        },
        Item::ToolCall {
            name,
            input,
            summary,
            state,
            change,
            exit_code,
            ..
        } => {
            let empty = serde_json::Value::Null;
            let input = input.as_ref().unwrap_or(&empty);
            let failed = *state == ToolState::Done { is_error: true };
            let tag = if failed {
                draw::failure_tag(*exit_code)
            } else {
                change.as_ref().map(draw::change_tag).unwrap_or_default()
            };
            // The arguments are shown in full under the row when the call is expanded; the row
            // then carries no summary, which would say the same thing twice.
            let head = match state {
                ToolState::Composing => vec![Span::styled("…".to_string(), gray())],
                _ if entry.expanded => Vec::new(),
                _ => call_summary(name, input, summary.as_deref(), working_directory, width),
            };
            // A command's next lines follow its first, and stand without it as a command that
            // starts at its second line, so a row too narrow for the first shows neither.
            let first_shown = draw::summary_room(name, &tag, width) > 0;
            let mut lines = vec![draw::tool_row(name, head, tag, width)];
            if *state == ToolState::Composing {
                return lines;
            }
            let own_width = width.saturating_sub(draw::connector_indent()).max(1);
            let code = crate::tools::builtin_code_param(name);
            let command =
                code.and_then(|(key, language)| Some((key, input.get(key)?.as_str()?, language)));
            let mut own = Vec::new();
            let mut program_first = false;
            if entry.expanded {
                // A change drawn as a diff says what the two texts were; their keys would say it
                // again.
                let mut omit: Vec<&str> = if change.is_some() && !failed {
                    crate::tools::builtin_change_params(name).to_vec()
                } else {
                    Vec::new()
                };
                if let Some((key, command, language)) = command {
                    own.extend(draw::command_rows(command, language, own_width));
                    omit.push(key);
                    program_first = true;
                }
                own.extend(draw::argument_rows(input, &omit, gray(), own_width));
            } else if let Some((_, command, language)) = command
                && first_shown
            {
                own.extend(command_continuation(command, language, own_width));
            }
            lines.extend(draw::tree_rows(
                own,
                program_first,
                tool_body(&entry.item, entry.expanded, width),
            ));
            lines
        }
        Item::Compaction {
            headline,
            departures,
            ..
        } => {
            let glyph = if entry.expanded {
                draw::OPENED_GLYPH
            } else {
                draw::FOLDED_GLYPH
            };
            let mut lines = vec![draw::boundary_row(
                glyph,
                COMPACTION_ICON,
                COMPACTION_LABEL,
                width,
            )];
            if entry.expanded {
                for sentence in std::iter::once(headline).chain(departures) {
                    lines.push(Line::default());
                    lines.extend(draw::glyph_text_rows(
                        ACT_GLYPH,
                        gray(),
                        &draw::screen_voice(sentence),
                        gray(),
                        width,
                    ));
                }
            }
            lines
        }
        Item::Checklist { items } => draw::checklist_rows(items, width),
        Item::Notice { level, text } => {
            let text = &draw::screen_voice(text);
            match level {
                crate::frontend::NoticeLevel::Info => {
                    draw::glyph_text_rows(ACT_GLYPH, gray(), text, gray(), width)
                }
                crate::frontend::NoticeLevel::Warn => {
                    let yellow = Style::default().fg(Color::Yellow);
                    draw::glyph_text_rows(WARNING_GLYPH, yellow, text, yellow, width)
                }
            }
        }
        Item::Log { target, message } => draw::glyph_text_rows(
            ACT_GLYPH,
            gray(),
            &format!("{target}: {}", message.trim_end()),
            gray(),
            width,
        ),
        Item::Error(text) => {
            let red = Style::default().fg(Color::Red);
            draw::glyph_text_rows(ERROR_GLYPH, red, &draw::screen_voice(text), red, width)
        }
        Item::ShellOutput(text) => {
            draw::edge_rows(draw::ansi_lines(text.trim_end_matches('\n'), inner))
        }
        Item::SessionId { label, id } => {
            draw::glyph_text_rows(ACT_GLYPH, gray(), &format!("{label} {id}"), gray(), width)
        }
        Item::TurnEnd(text) => {
            draw::glyph_text_rows(ACT_GLYPH, gray(), &draw::screen_voice(text), gray(), width)
        }
        Item::Image(picture) => match draw::picture_rows(picture, inner) {
            Some(rows) => draw::edge_rows(rows),
            None => draw::glyph_text_rows(ACT_GLYPH, gray(), &picture.label(), gray(), width),
        },
        // The mark beside three lines, in no box: a box of the composer's gray was painted short
        // by tmux under Alacritty, which drops the background of the spaces after a foreground
        // reset, and a box with no padding is no box. For the same reason the shadow's
        // background lies under a glyph only, never a space.
        Item::Banner {
            version,
            revision,
            debug,
            mcp_servers,
            skills,
            memories,
            instructions,
        } => {
            let mark_width = BANNER_MARK
                .iter()
                .map(|row| row.iter().map(|(run, _)| display_width(run)).sum::<usize>())
                .max()
                .unwrap_or(0);
            let lead = BANNER_GAP + mark_width + 2;
            let room = width.saturating_sub(lead).max(1);
            // The name bold and white, so it is what the eye lands on, the version dimmed beside
            // it; the facts a step down from the name, in the palette's plain white, which most
            // themes set to a light gray, rather than the bright one.
            // The build beside the version, dimmed with it: a release built from its tag says
            // nothing more, since the version is the revision.
            let build: Vec<&str> = revision
                .iter()
                .map(String::as_str)
                .chain(debug.then_some("debug"))
                .collect();
            let build = if build.is_empty() {
                String::new()
            } else {
                format!(" ({})", build.join(", "))
            };
            let title = vec![
                Span::styled(
                    "Meka".to_string(),
                    Style::default()
                        .fg(draw::WHITE)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!(" v{version}{build}"), gray()),
            ];
            let fact = |text: &str| {
                vec![Span::styled(
                    draw::cut_text(text, room),
                    Style::default().fg(Color::Gray),
                )]
            };
            // What the agent can do on one line and what it knows on the next.
            let rows = [
                draw::cut_spans(title, room),
                fact(&format!(
                    "{} · {}",
                    counted(*mcp_servers, "MCP server", "MCP servers"),
                    counted(*skills, "skill", "skills")
                )),
                fact(&format!(
                    "{} · {}",
                    counted(*memories, "memory", "memories"),
                    counted(*instructions, "instructions", "instructions")
                )),
            ];
            let mut lines = vec![Line::default()];
            lines.extend(BANNER_MARK.iter().zip(rows).map(|(mark, text)| {
                let mut spans = vec![Span::raw(" ".repeat(BANNER_GAP))];
                let mut drawn = 0;
                for (run, tone) in mark.iter() {
                    drawn += display_width(run);
                    spans.push(Span::styled((*run).to_string(), tone.style()));
                }
                spans.push(Span::raw(" ".repeat(mark_width.saturating_sub(drawn) + 2)));
                spans.extend(text);
                Line::from(spans)
            }));
            lines
        }
    }
}

/// What an item is to folding: a step of the agent's work (counted as `Some` kind, or uncounted),
/// a change it made, its words, the edge of a turn, or anything else shown. Steps fold; the
/// rest stand on their own, and only words or a turn's edge finish the steps before them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    Step(Option<StepKind>),
    Change,
    Words,
    TurnEdge,
    Other,
}

/// What a step did, as a folded run counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepKind {
    Thought,
    Command,
    Read,
    Search,
    Fetch,
    Tool,
}

impl StepKind {
    /// What `count` steps of this kind did, as the run's row says it.
    fn phrase(self, count: usize) -> String {
        let times = || match count {
            1 => "once".to_string(),
            2 => "twice".to_string(),
            _ => format!("{count} times"),
        };
        let counted = |one: &str, many: &str| {
            if count == 1 {
                one.to_string()
            } else {
                format!("{count} {many}")
            }
        };
        match self {
            Self::Thought => format!("thought {}", times()),
            Self::Command => format!("ran {}", counted("a command", "commands")),
            Self::Read => format!("read {}", counted("a file", "files")),
            Self::Search => format!("searched {}", times()),
            Self::Fetch => format!("fetched {}", counted("a page", "pages")),
            Self::Tool => format!("used {}", counted("a tool", "tools")),
        }
    }
}

/// What `item` is to folding. A call that changed a file, a memory or a skill shows the change
/// rather than folding, and a sub-agent's call shows what it does.
fn part_of(item: &Item) -> Part {
    match item {
        Item::Thinking { .. } => Part::Step(Some(StepKind::Thought)),
        Item::Checklist { .. } => Part::Step(None),
        Item::ToolCall {
            name,
            change,
            state,
            ..
        } => {
            if change.is_some() && *state != (ToolState::Done { is_error: true }) {
                Part::Change
            } else if matches!(name.as_str(), "agent_spawn" | "agent_followup") {
                Part::Other
            } else {
                Part::Step(Some(match name.as_str() {
                    "shell_execute" => StepKind::Command,
                    "file_read" => StepKind::Read,
                    "web_fetch" => StepKind::Fetch,
                    name if name == "file_find" || name.ends_with("_search") => StepKind::Search,
                    _ => StepKind::Tool,
                }))
            }
        }
        // Words of spaces alone say nothing yet.
        Item::Text { text, .. } if text.trim().is_empty() => Part::Other,
        Item::Text { .. } => Part::Words,
        // A prompt waiting to be delivered, and a `!command` beside the session, are not said to
        // the agent, so the run before them goes on.
        Item::Prompt {
            delivered: false, ..
        }
        | Item::Prompt {
            ran_under: RanUnder::Shell,
            ..
        } => Part::Other,
        Item::Prompt { .. } | Item::TurnEnd(_) => Part::TurnEdge,
        _ => Part::Other,
    }
}

/// A run of steps among the items drawn, `from` to `to` inclusive, and whether it is finished:
/// the agent has said something after it, or its turn has ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Run {
    from: usize,
    to: usize,
    finished: bool,
}

/// The runs of steps among `parts`, in order. `idle` says no turn is running, which finishes
/// every run.
fn runs_of(parts: &[Part], idle: bool) -> Vec<Run> {
    let last_close = parts
        .iter()
        .rposition(|part| matches!(part, Part::Words | Part::TurnEdge));
    let mut runs = Vec::new();
    let mut from = None;
    for (index, part) in parts.iter().enumerate() {
        match (part, from) {
            (Part::Step(_), None) => from = Some(index),
            (Part::Step(_), Some(_)) => {}
            (_, Some(start)) => {
                runs.push(Run {
                    from: start,
                    to: index - 1,
                    finished: idle || last_close.is_some_and(|close| close > index - 1),
                });
                from = None;
            }
            (_, None) => {}
        }
    }
    if let Some(start) = from {
        let to = parts.len() - 1;
        runs.push(Run {
            from: start,
            to,
            finished: idle || last_close.is_some_and(|close| close > to),
        });
    }
    runs
}

/// The row a run of steps is drawn as when it can fold: what it did, in the order it first did
/// each, as "Ran 5 commands, read 2 files, thought 3 times", and how many of its calls failed, in
/// red, behind the glyph that says whether it is folded or open.
fn run_row(entries: &[&Entry], open: bool, width: usize) -> Line<'static> {
    let mut counts: Vec<(StepKind, usize)> = Vec::new();
    let mut failed = 0;
    for entry in entries {
        if let Part::Step(Some(kind)) = part_of(&entry.item) {
            match counts.iter_mut().find(|(held, _)| *held == kind) {
                Some((_, count)) => *count += 1,
                None => counts.push((kind, 1)),
            }
        }
        if matches!(entry.item, Item::ToolCall {
            state: ToolState::Done { is_error: true },
            ..
        }) {
            failed += 1;
        }
    }
    let said = counts
        .iter()
        .map(|(kind, count)| kind.phrase(*count))
        .collect::<Vec<_>>()
        .join(", ");
    let mut said_characters = said.chars();
    let said = match said_characters.next() {
        Some(first) => first.to_uppercase().chain(said_characters).collect(),
        None => "Updated the checklist".to_string(),
    };
    let failure = (failed > 0).then(|| format!("{failed} failed"));
    let failure_width = failure.as_deref().map_or(0, |text| display_width(text) + 2);
    let budget = width.saturating_sub(TEXT_EDGE + failure_width).max(1);
    let glyph = if open {
        draw::OPENED_GLYPH
    } else {
        draw::FOLDED_GLYPH
    };
    let mut spans = vec![
        Span::styled(format!("{glyph:<TEXT_EDGE$}"), gray()),
        Span::styled(draw::cut_text(&said, budget), gray()),
    ];
    if let Some(failure) = failure {
        spans.push(Span::raw("  ".to_string()));
        spans.push(Span::styled(failure, Style::default().fg(Color::Red)));
    }
    // Cut whole where the width leaves the count no room either.
    draw::cut_line(Line::from(spans), width)
}

/// What the viewport keeps of a run of steps, under the id of each of the run's items.
#[derive(Debug, Clone, Copy, Default)]
struct RunState {
    /// Opened (`true`) or folded (`false`) by hand, which holds whatever else changes.
    choice: Option<bool>,
    /// Folded on its own once finished.
    settled: bool,
    /// Drawn while unfinished: such a run waits to fold until the view is back at the bottom, so
    /// what the reader is looking at does not move under them.
    seen_unfinished: bool,
    /// Whether the last frame drew it open.
    open: bool,
}

/// How an item is drawn in this frame: as itself, hidden in a folded run, or heading a run with
/// the run's row, open or folded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    Itself,
    Hidden,
    Heading { run: u64, open: bool },
}

/// The width `entry` is laid out at on a transcript `width` wide: a column short of the right
/// edge, which no words reach, so an open run's shade runs past them as a prompt's gray does; a
/// prompt itself takes the whole width, its gray running to the edge with the column kept inside.
fn item_width(entry: &Entry, width: usize) -> usize {
    if matches!(entry.item, Item::Prompt { .. }) {
        width
    } else {
        width.saturating_sub(draw::RIGHT_MARGIN)
    }
}

/// What one item was last drawn as.
struct Cached {
    revision: u64,
    width: usize,
    lines: Vec<Line<'static>>,
    /// How each line copies, where that differs from text; empty when no line does.
    copies: Vec<draw::RowCopy>,
}

/// The transcript on screen: which lines are visible, and what sits where.
pub(crate) struct Viewport {
    /// What each resident item was last drawn as, by the item's id.
    cache: HashMap<u64, Cached>,
    /// The first visible line, counted over every item and the gaps between them.
    scroll: usize,
    /// Whether the view sticks to the bottom as items arrive.
    follow: bool,
    /// Where the keyboard's cursor stands, if anywhere: held by the item's id rather than its
    /// place, which a page loaded above moves, and checked against the stops at every frame,
    /// since a fold or an item gone can take its stop away.
    cursor: Option<HeldCursor>,
    /// Where each item's lines start and how many there are, as of the last draw.
    layout: Vec<(usize, usize)>,
    /// The item at the top of the view and how far into it the view starts, so a change above
    /// it (a page loaded, a re-wrap) leaves what is on screen where it is. Cleared by every
    /// scroll the user makes, and set again by the draw that follows.
    anchor: Option<(u64, isize)>,
    /// Whether the anchor holds a row the user opened or closed in place, so the draw that
    /// places it follows the bottom again when that is where it leaves the view.
    holding: bool,
    /// The row of the view the mouse is over, when it is over the view: a collapsible item
    /// drawn there lights up, which is the hint that a click would open it.
    hover_row: Option<usize>,
    /// The nearest prompt scrolled off the top, by id and first row, for the bar that brings
    /// it back.
    sticky: Option<(u64, usize)>,
    total: usize,
    height: usize,
    /// The directory the session works in, which the paths calls name are shown against.
    working_directory: PathBuf,
    /// What is kept of each run of steps, under the id of each of its items.
    runs: HashMap<u64, RunState>,
    /// How each item was placed in the last frame, by its place on screen.
    placements: Vec<Placement>,
    /// The rows of each folded run, by run, as of the last frame.
    run_rows: HashMap<u64, Line<'static>>,
    /// The first and last item of each open run or compaction, by place on screen, as of the last
    /// frame.
    open_runs: Vec<(usize, usize)>,
    /// The id of each item drawn in the last frame, by place on screen.
    ids: Vec<u64>,
    /// How each row of the last frame copies, from the top of the view.
    copies: Vec<draw::RowCopy>,
}

impl Viewport {
    pub(crate) fn new() -> Self {
        Self {
            working_directory: PathBuf::new(),
            runs: HashMap::new(),
            placements: Vec::new(),
            run_rows: HashMap::new(),
            open_runs: Vec::new(),
            ids: Vec::new(),
            copies: Vec::new(),
            cache: HashMap::new(),
            scroll: 0,
            follow: true,
            cursor: None,
            layout: Vec::new(),
            anchor: None,
            holding: false,
            sticky: None,
            hover_row: None,
            total: 0,
            height: 0,
        }
    }

    /// Show the paths calls name against `directory` from the next frame, laying every item out
    /// again when it moved.
    pub(crate) fn set_working_directory(&mut self, directory: PathBuf) {
        if directory != self.working_directory {
            self.working_directory = directory;
            self.cache.clear();
        }
    }

    /// Say which row of the view the mouse is over, or that it is not over the view; whether
    /// that moves the light to another item.
    pub(crate) fn set_hover_row(&mut self, row: Option<usize>) -> bool {
        let at = |viewport: &Self, row: Option<usize>| {
            let row = row?;
            let index = viewport.item_at_row(row)?;
            Some((index, viewport.run_at_row(row).is_some()))
        };
        let before = at(self, self.hover_row);
        self.hover_row = row;
        before != at(self, row)
    }

    /// Where the keyboard's cursor stands, as of the last frame: on the item, under its run's row
    /// when it heads a run, or on that row.
    pub(crate) fn cursor(&self) -> Option<CursorStop> {
        let held = self.cursor?;
        let index = self.ids.iter().position(|&id| id == held.id)?;
        let heads = matches!(self.placements.get(index), Some(Placement::Heading { .. }));
        Some(CursorStop {
            index,
            under_run: heads && !held.run_row,
        })
    }

    /// The cursor held at `stop` of the last frame, by what it names.
    fn held_at(&self, stop: CursorStop) -> Option<HeldCursor> {
        let heads = matches!(
            self.placements.get(stop.index),
            Some(Placement::Heading { .. })
        );
        Some(HeldCursor {
            id: *self.ids.get(stop.index)?,
            run_row: heads && !stop.under_run,
        })
    }

    /// The prompt stuck at the top, as its first line of words and the row it starts on: the
    /// nearest one scrolled off above, so a click can bring it back.
    pub(crate) fn sticky_prompt(&self) -> Option<(Line<'static>, usize)> {
        let (id, start) = self.sticky?;
        let cached = self.cache.get(&id)?;
        let line = cached
            .lines
            .iter()
            .enumerate()
            .find(|(index, _)| cached.copies.get(*index) != Some(&draw::RowCopy::Padding))?
            .1
            .clone();
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
    pub(crate) fn set_cursor(&mut self, cursor: Option<CursorStop>) {
        self.cursor = cursor.and_then(|stop| self.held_at(stop));
        if let Some(stop) = cursor
            && let Some(&(start, _)) = self.layout.get(stop.index)
        {
            // The row it lights brought into view rather than the item centered: the items
            // above and below stay where they were, which is what keeps a walk through the list
            // legible. A row below the top one, which the stuck prompt's bar may cover.
            let row = start + if stop.under_run { 2 } else { 0 };
            let bottom = self.total.saturating_sub(self.height);
            let margin = usize::from(self.height > 1);
            if row < self.scroll + margin {
                self.scroll = row.saturating_sub(margin);
            } else if row >= self.scroll + self.height {
                self.scroll = (row + 1).saturating_sub(self.height).min(bottom);
            } else {
                return;
            }
            self.follow = self.scroll >= bottom;
            self.anchor = None;
        }
    }

    /// The id of the item at `index` of the last frame, which names it whatever moves it before
    /// the next.
    pub(crate) fn id_at(&self, index: usize) -> Option<u64> {
        self.ids.get(index).copied()
    }

    /// How `row` of the last frame copies: as text, unless an item said otherwise, and as nothing
    /// under the stuck prompt's bar, which covers the row it is drawn on.
    pub(crate) fn row_copy(&self, row: usize) -> draw::RowCopy {
        if row == 0 && self.sticky_prompt().is_some() {
            return draw::RowCopy::Padding;
        }
        self.copies.get(row).copied().unwrap_or(draw::RowCopy::Text)
    }

    /// The item drawn on `row` of the last frame, if any, by its place on screen.
    pub(crate) fn item_at_row(&self, row: usize) -> Option<usize> {
        let line = self.scroll.checked_add(row)?;
        self.layout
            .iter()
            .position(|&(start, len)| line >= start && line < start + len)
    }

    /// Lay `entry` out on a transcript `width` wide unless the cache already has it, and say how
    /// many rows it is. `prompt_padding` is the transcript's, which never changes under the cache.
    fn lay_out(&mut self, entry: &Entry, width: usize, prompt_padding: PromptPadding) -> usize {
        let width = item_width(entry, width);
        let stale = self
            .cache
            .get(&entry.id)
            .is_none_or(|cached| cached.revision != entry.revision || cached.width != width);
        if stale {
            let (lines, copies) = entry_rows(entry, width, &self.working_directory, prompt_padding);
            self.cache.insert(entry.id, Cached {
                revision: entry.revision,
                width,
                lines,
                copies,
            });
        }
        self.cache
            .get(&entry.id)
            .map_or(0, |cached| cached.lines.len())
    }

    /// Lay every item out at `width`, so a frame drawn at it does no more than place the rows. An
    /// item a folded run hides is laid out when the run opens, since a session's history is mostly
    /// steps and most of them are never opened.
    pub(crate) fn lay_out_all(&mut self, transcript: &Transcript, width: usize) {
        self.place(transcript, width);
        for (index, entry) in transcript.drawn().enumerate() {
            if self.placements.get(index) != Some(&Placement::Hidden) {
                self.lay_out(entry, width, transcript.prompt_padding());
            }
        }
    }

    /// How each drawn item is placed this frame, and the row of each run that has one. A run of
    /// steps folds once it is finished and holds two or more, at once when it was first drawn
    /// finished, otherwise once the view follows the bottom again; a run the user opened or
    /// folded stays as they left it, and showing every call expanded opens them all. A
    /// compaction's contents show while it is open and are hidden while it is not, and never fold
    /// into a run, which would hide them behind a second row.
    fn place(&mut self, transcript: &Transcript, width: usize) {
        let entries: Vec<&Entry> = transcript.drawn().collect();
        let parts: Vec<Part> = entries
            .iter()
            .map(|entry| match entry.compaction {
                Some(_) => Part::Other,
                None => part_of(&entry.item),
            })
            .collect();
        let mut placements = vec![Placement::Itself; entries.len()];
        let mut run_rows = HashMap::new();
        let mut open_runs = Vec::new();
        let mut kept = HashMap::new();
        let folds = transcript.folds_steps();
        let idle = !transcript.turn_running();
        for run in runs_of(&parts, idle) {
            let Some(first) = entries.get(run.from) else {
                continue;
            };
            let members = entries.get(run.from..=run.to).unwrap_or_default();
            // Found by any of its steps, being kept under every one below: a page read above the
            // run, or a first step that turns out to be a change, moves which step is first.
            let mut state = members
                .iter()
                .find_map(|entry| self.runs.get(&entry.id))
                .copied()
                .unwrap_or_default();
            // A call still at work keeps its run open, whatever has been said after it, so what
            // it writes stays in view.
            let working = !idle
                && members.iter().any(|entry| {
                    matches!(entry.item, Item::ToolCall {
                        state: ToolState::Running | ToolState::Composing,
                        ..
                    })
                });
            let finished = run.finished && !working;
            if !finished {
                state.seen_unfinished = true;
            }
            let foldable = finished && run.to > run.from;
            if foldable && !state.settled && (!state.seen_unfinished || self.follow) {
                state.settled = true;
            }
            // A run that finished while the view was up the transcript waits to fold until it
            // settles, drawn as it was, so the rows the reader is on do not move.
            let foldable = foldable && state.settled;
            if foldable {
                state.open = state.choice.unwrap_or(!folds);
                run_rows.insert(
                    first.id,
                    run_row(
                        members,
                        state.open,
                        width.saturating_sub(draw::RIGHT_MARGIN),
                    ),
                );
                if state.open {
                    open_runs.push((run.from, run.to));
                }
                if let Some(placement) = placements.get_mut(run.from) {
                    *placement = Placement::Heading {
                        run: first.id,
                        open: state.open,
                    };
                }
                if !state.open {
                    for placement in placements.iter_mut().take(run.to + 1).skip(run.from + 1) {
                        *placement = Placement::Hidden;
                    }
                }
            }
            // Under every step, so the state is found again by whichever step stays in the run.
            for member in members {
                kept.insert(member.id, state);
            }
        }
        let mut compactions: HashMap<u64, (usize, bool)> = HashMap::new();
        for (index, entry) in entries.iter().enumerate() {
            if matches!(entry.item, Item::Compaction { .. }) {
                let open = entry.expanded && entry.is_collapsible();
                compactions.insert(entry.id, (open_runs.len(), open));
                if open {
                    open_runs.push((index, index));
                }
            } else if let Some(owner) = entry.compaction
                && let Some(&(run, open)) = compactions.get(&owner)
            {
                if !open {
                    if let Some(placement) = placements.get_mut(index) {
                        *placement = Placement::Hidden;
                    }
                } else if let Some((_, to)) = open_runs.get_mut(run) {
                    *to = index;
                }
            }
        }
        self.runs = kept;
        self.placements = placements;
        self.run_rows = run_rows;
        self.open_runs = open_runs;
    }

    /// Draw the transcript into `area`.
    pub(crate) fn draw(&mut self, frame: &mut Frame<'_>, area: Rect, transcript: &Transcript) {
        self.height = area.height as usize;
        let width = area.width as usize;
        self.place(transcript, width);

        // Lay every item out first, so the scroll position is clamped against the whole.
        self.layout.clear();
        let mut ids: Vec<u64> = Vec::with_capacity(transcript.drawn_len());
        let mut prompts: Vec<(u64, usize)> = Vec::new();
        let mut total = 0;
        for (index, entry) in transcript.drawn().enumerate() {
            ids.push(entry.id);
            let hidden = self.placements.get(index) == Some(&Placement::Hidden);
            let own = if hidden {
                0
            } else {
                self.lay_out(entry, width, transcript.prompt_padding())
            };
            let len = match self.placements.get(index) {
                Some(Placement::Hidden) => 0,
                Some(Placement::Heading { open: false, .. }) => 1,
                // The run's row, a blank under it, then the first step.
                Some(Placement::Heading { open: true, .. }) => 2 + own,
                _ => own,
            };
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
        self.ids.clone_from(&ids);
        self.total = total;
        self.settle_cursor(transcript);
        let bottom = total.saturating_sub(self.height);
        let holding = std::mem::take(&mut self.holding);
        if self.follow {
            self.scroll = bottom;
        } else if let Some((id, offset)) = self.anchor
            && let Some(position) = ids.iter().position(|&held| held == id)
            && let Some(&(start, len)) = self.layout.get(position)
        {
            // An item a run has folded over since is held by the row of what hides it, the last
            // item drawn before it.
            let (start, offset) = if len > 0 {
                (start, offset)
            } else {
                let shown = self
                    .layout
                    .get(..position)
                    .unwrap_or_default()
                    .iter()
                    .rev()
                    .find(|(_, len)| *len > 0)
                    .map_or(0, |(start, _)| *start);
                (shown, 0)
            };
            let start = start as isize;
            self.scroll = ((start + offset).max(0) as usize).min(bottom);
            // A row closed at the bottom, or one opened in a transcript that still fits, leaves
            // the view at the bottom, where it goes on following what comes.
            if holding {
                self.follow = self.scroll >= bottom;
            }
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
        // nothing is hidden, and one more row of scrolling makes that prompt the bar's. Nor where
        // the top row is the only one, which the bar would leave no transcript in.
        self.sticky = if self.height <= 1 || prompts.iter().any(|&(_, start)| start == scroll) {
            None
        } else {
            prompts
                .iter()
                .rev()
                .find(|&&(_, start)| start < scroll)
                .copied()
        };

        // The light goes on what a click under the mouse would open, resolved against the layout
        // of this frame, so it follows the transcript as it scrolls under a still pointer: a run's
        // row, or the collapsible item drawn there.
        let lit = self
            .hover_row
            .and_then(|row| self.target_at_row(row, transcript));

        let cursor = self.cursor();
        let mut rows: Vec<Line<'static>> = Vec::with_capacity(self.height);
        let mut copies: Vec<draw::RowCopy> = Vec::with_capacity(self.height);
        let first = self.scroll;
        let last = first + self.height;
        for (index, &(start, len)) in self.layout.iter().enumerate() {
            if len == 0 || start + len <= first {
                continue;
            }
            if start >= last {
                break;
            }
            // The row the cursor lights: a run's own, or under an open run's row and the gap after
            // it, the first step's.
            let picked = cursor
                .filter(|stop| stop.index == index)
                .map(|stop| if stop.under_run { 2 } else { 0 });
            let cached = ids.get(index).and_then(|id| self.cache.get(id));
            let own = cached
                .map(|cached| cached.lines.as_slice())
                .unwrap_or_default();
            let own_copies = cached
                .map(|cached| cached.copies.as_slice())
                .unwrap_or_default();
            let copy_of =
                |line: usize| own_copies.get(line).copied().unwrap_or(draw::RowCopy::Text);
            let heading = match self.placements.get(index) {
                Some(Placement::Heading { run, open }) => {
                    Some((self.run_rows.get(run).cloned().unwrap_or_default(), *open))
                }
                _ => None,
            };
            // The run's row and the gap under it when the item heads a run, then the item's own
            // rows unless the run is folded; only the rows in view are copied out of the cache,
            // since an item can be thousands of rows long and this runs every frame.
            let own_shown = !matches!(heading, Some((_, false)));
            let head: Vec<(Line<'static>, Target)> = match heading {
                Some((row, false)) => vec![(row, Target::Run(index))],
                Some((row, true)) => vec![
                    (row, Target::Run(index)),
                    (Line::default(), Target::Item(index)),
                ],
                None => Vec::new(),
            };
            let (from, to) = (first.saturating_sub(start), last.saturating_sub(start));
            let own_from = from.saturating_sub(head.len());
            let own_to = if own_shown {
                to.saturating_sub(head.len())
            } else {
                0
            };
            let head_rows = head.len();
            let visible =
                head.into_iter()
                    .enumerate()
                    .filter(|(offset, _)| (from..to).contains(offset))
                    .map(|(offset, (line, target))| (offset, line, target, draw::RowCopy::Text))
                    .chain(own.iter().enumerate().take(own_to).skip(own_from).map(
                        |(line, row)| {
                            (
                                head_rows + line,
                                row.clone(),
                                Target::Item(index),
                                copy_of(line),
                            )
                        },
                    ));
            for (offset, line, target, copy) in visible {
                let at = start + offset;
                let mut line = line;
                if lit == Some(target) {
                    line = lit_up(line);
                }
                if picked == Some(offset) {
                    line = line.patch_style(Style::default().add_modifier(Modifier::REVERSED));
                }
                while rows.len() < at - first {
                    rows.push(Line::default());
                    copies.push(draw::RowCopy::Text);
                }
                rows.push(line);
                copies.push(copy);
            }
        }
        self.copies = copies;
        // An open run is one block on its shade, from its row through its last step and the gaps
        // between them.
        let shaded: Vec<(usize, usize)> = self
            .open_runs
            .iter()
            .filter_map(|&(from, to)| {
                let &(start, _) = self.layout.get(from)?;
                let &(last, len) = self.layout.get(to)?;
                Some((start, last + len))
            })
            .collect();
        if !shaded.is_empty() {
            while rows.len() < self.height {
                rows.push(Line::default());
            }
            for (offset, row) in rows.iter_mut().enumerate() {
                let at = first + offset;
                if shaded.iter().any(|&(start, end)| at >= start && at < end) {
                    *row = draw::shaded_row(std::mem::take(row), draw::RUN_BACKGROUND, width);
                }
            }
        }
        frame.render_widget(Paragraph::new(Text::from(rows)), area);
    }

    /// What a click on `row` of the last frame would open or close: the run whose row is drawn
    /// there, or the collapsible item, by its place on screen.
    fn target_at_row(&self, row: usize, transcript: &Transcript) -> Option<Target> {
        let index = self.item_at_row(row)?;
        let (start, _) = *self.layout.get(index)?;
        match self.placements.get(index) {
            Some(Placement::Heading { open: false, .. }) => Some(Target::Run(index)),
            Some(Placement::Heading { open: true, .. }) if self.scroll + row == start => {
                Some(Target::Run(index))
            }
            _ => transcript
                .drawn()
                .nth(index)
                .is_some_and(Entry::is_collapsible)
                .then_some(Target::Item(index)),
        }
    }

    /// The run whose row is drawn on `row` of the last frame, if any.
    pub(crate) fn run_at_row(&self, row: usize) -> Option<u64> {
        let index = self.item_at_row(row)?;
        let (start, _) = *self.layout.get(index)?;
        match self.placements.get(index) {
            Some(Placement::Heading { run, open: false }) => Some(*run),
            Some(Placement::Heading { run, open: true }) if self.scroll + row == start => {
                Some(*run)
            }
            _ => None,
        }
    }

    /// Keep the cursor on a stop of this frame: the item's own row when the run it stood under
    /// has folded, else the nearest stop before it, else after it, else none; an item gone takes
    /// the cursor with it.
    fn settle_cursor(&mut self, transcript: &Transcript) {
        let Some(stop) = self.cursor() else {
            self.cursor = None;
            return;
        };
        let stops = self.cursor_stops(transcript);
        if stops.contains(&stop) {
            return;
        }
        let nearest = stops
            .iter()
            .rev()
            .find(|held| held.index <= stop.index)
            .or_else(|| stops.iter().find(|held| held.index > stop.index))
            .copied();
        self.cursor = nearest.and_then(|stop| self.held_at(stop));
    }

    /// The run whose row the keyboard's cursor stands on, if it stands on one.
    pub(crate) fn run_under_cursor(&self) -> Option<u64> {
        let stop = self.cursor().filter(|stop| !stop.under_run)?;
        match self.placements.get(stop.index) {
            Some(Placement::Heading { run, .. }) => Some(*run),
            _ => None,
        }
    }

    /// Open a folded run, or fold an open one, and keep it that way, its row where it is.
    pub(crate) fn toggle_run(&mut self, run: u64) {
        let Some(state) = self.runs.get_mut(&run) else {
            return;
        };
        state.choice = Some(!state.open);
        if let Some(index) = self.placements.iter().position(
            |placement| matches!(placement, Placement::Heading { run: held, .. } if *held == run),
        ) {
            self.keep_in_place(index);
        }
    }

    /// Hold the item at `index` on the screen row it is on through the next frame, so what opens
    /// under it grows below it rather than pushing it out of view, and what closes draws the rows
    /// under it up to it.
    pub(crate) fn keep_in_place(&mut self, index: usize) {
        if let (Some(&id), Some(&(start, _))) = (self.ids.get(index), self.layout.get(index)) {
            self.anchor = Some((id, self.scroll as isize - start as isize));
            self.follow = false;
            self.holding = true;
        }
    }

    /// Forget which runs the user opened or folded, so every run follows the screen's setting
    /// again: what showing every call expanded, or hiding them again, asks of them.
    pub(crate) fn forget_run_choices(&mut self) {
        for state in self.runs.values_mut() {
            state.choice = None;
        }
    }

    /// The stop the cursor moves to before `from`, or the last one when `from` is none: a run's
    /// row, or a collapsible item outside a folded run.
    pub(crate) fn cursor_before(
        &self,
        transcript: &Transcript,
        from: Option<CursorStop>,
    ) -> Option<CursorStop> {
        let stops = self.cursor_stops(transcript);
        let end = from
            .and_then(|from| stops.iter().position(|stop| *stop == from))
            .unwrap_or(stops.len());
        stops.get(..end)?.last().copied()
    }

    /// The stop the cursor moves to after `from`.
    pub(crate) fn cursor_after(
        &self,
        transcript: &Transcript,
        from: CursorStop,
    ) -> Option<CursorStop> {
        let stops = self.cursor_stops(transcript);
        let at = stops.iter().position(|stop| *stop == from)?;
        stops.get(at + 1).copied()
    }

    /// Every stop of the cursor, in order, as of the last frame: a run's row, then, when the run
    /// is open, its first step where that step is collapsible, which shares its place on screen.
    fn cursor_stops(&self, transcript: &Transcript) -> Vec<CursorStop> {
        let mut stops = Vec::new();
        for (index, entry) in transcript.drawn().enumerate() {
            let on = |under_run| CursorStop { index, under_run };
            match self.placements.get(index) {
                Some(Placement::Hidden) => {}
                Some(Placement::Heading { open, .. }) => {
                    stops.push(on(false));
                    if *open && entry.is_collapsible() {
                        stops.push(on(true));
                    }
                }
                _ if entry.is_collapsible() => stops.push(on(false)),
                _ => {}
            }
        }
        stops
    }
}

/// The cursor as the viewport holds it: the item by its id, and whether it stands on the row of
/// the run the item heads rather than on the item. What it names holds as runs form and fold
/// around the item; where the cursor is drawn follows from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HeldCursor {
    id: u64,
    run_row: bool,
}

/// A place the keyboard's cursor stands: an item, by its place on screen, or, under the row of
/// the open run it heads, the item itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CursorStop {
    /// The item's place on screen.
    pub(crate) index: usize,
    /// Whether the cursor stands on the item under the row of the open run it heads, rather
    /// than on that row.
    pub(crate) under_run: bool,
}

/// What a click opens or closes: the run an item heads, or the item itself, by its place on
/// screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Run(usize),
    Item(usize),
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
        draw::DIM_ON_SHADE
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

/// The status row's context gauge: tokens in context, the model's window and the share used, as
/// `128.4k / 1.0M (13%)`, the short form of `/status`'s line. Shown from the first frame, at zero
/// occupancy, since the window is known at launch and a fact in the default row that appears only
/// after the first turn reads as missing; `None` only while the window is unknown, as `/status`
/// omits its line then.
pub(crate) fn context_gauge_text(tokens: u64, window: u64) -> Option<String> {
    if window == 0 {
        return None;
    }
    let percent = ((tokens as f64 / window as f64) * 100.0).round() as u64;
    Some(format!(
        "{} / {} ({}%)",
        crate::text::format_token_count(tokens),
        crate::text::format_token_count(window),
        percent
    ))
}

/// The facts the status row can show, and which of them it shows, in order.
pub(crate) struct StatusFacts<'a> {
    /// `[display].status_row`.
    pub(crate) shown: &'a [crate::config::StatusFact],
    pub(crate) permission: crate::permission::Permission,
    pub(crate) path: Option<String>,
    /// The project the session is on, while it is on one.
    pub(crate) project: Option<String>,
    /// The profile the session runs on, by name.
    pub(crate) profile: Option<String>,
    /// The model the profile runs, when the config names one.
    pub(crate) model: Option<String>,
    pub(crate) context: Option<String>,
    /// Background tasks and sub-agents, shown while any run.
    pub(crate) work: RunningWork,
    pub(crate) hint: Option<&'a str>,
    /// Whether the composer is in its shell mode, which the row says in red in place of the level
    /// and every fact on the left, so what Enter will do is never a surprise; the readings stay.
    pub(crate) shell_mode: bool,
}

/// A crossterm color as ratatui's.
pub(crate) fn color(color: crossterm::style::Color) -> Color {
    use ratatui::backend::FromCrossterm;
    Color::from_crossterm(color)
}

/// What the status row says, alone and in red, while the composer is in its shell mode.
pub(crate) const SHELL_MODE_LABEL: &str = "Run command in shell";

/// `count` of a noun as the banner and the status row say it: `0 skills`, `1 skill`, `2 skills`.
fn counted(count: usize, singular: &str, plural: &str) -> String {
    let noun = if count == 1 { singular } else { plural };
    format!("{count} {noun}")
}

/// The color of the bar down the composer's left edge: the permission level's, or red in the
/// shell mode, so what Enter will do is never a surprise.
pub(crate) fn composer_bar_color(facts: &StatusFacts<'_>) -> Color {
    if facts.shell_mode {
        Color::Red
    } else {
        color(facts.permission.indicator_color())
    }
}

/// The status row under the composer, on the terminal's background in its own colors. Against the
/// left, after a dot each, what the next prompt goes under: the permission level in its color, the
/// profile, and the model dimmed, then where the session works, the path and the project, dimmed;
/// in the shell mode, what Enter does instead, in red. Against the right edge, the session's live
/// readings, the context gauge and the work running, or a hint while one stands. Each fact as the
/// config lists it.
///
/// Where the row is narrow, the level stays whole and the readings give way to it; the facts after
/// the level give way from the last, the one that no longer fits cut to what is left (a path from
/// its start, whose tail tells it apart) when enough is left to read, and dropped with its dot
/// otherwise.
pub(crate) fn status_line(facts: &StatusFacts<'_>, width: usize) -> Line<'static> {
    use crate::config::StatusFact;
    /// The dot each fact after the level stands behind.
    const DOT: &str = " · ";
    /// The fewest columns a cut fact keeps, its ellipsis included, below which it is dropped.
    const LEAST_CUT_COLUMNS: usize = 4;
    let level = if facts.shell_mode {
        Span::styled(
            SHELL_MODE_LABEL.to_string(),
            Style::default().fg(Color::Red),
        )
    } else {
        Span::styled(
            facts.permission.to_string(),
            Style::default().fg(composer_bar_color(facts)),
        )
    };
    // Each fact after the level, in the order the config lists them, and whether a cut keeps its
    // start or its end.
    let mut after: Vec<(Span<'static>, bool)> = Vec::new();
    let mut readings: Vec<String> = Vec::new();
    for fact in facts.shown {
        match fact {
            _ if facts.shell_mode
                && matches!(
                    fact,
                    StatusFact::Profile
                        | StatusFact::Model
                        | StatusFact::Path
                        | StatusFact::Project
                ) => {}
            StatusFact::Profile => after.extend(
                facts
                    .profile
                    .clone()
                    .map(|profile| (Span::raw(profile), false)),
            ),
            StatusFact::Model => after.extend(
                facts
                    .model
                    .clone()
                    .map(|model| (Span::styled(model, gray()), false)),
            ),
            StatusFact::Path => after.extend(
                facts
                    .path
                    .clone()
                    .map(|path| (Span::styled(path, gray()), true)),
            ),
            StatusFact::Project => after.extend(
                facts
                    .project
                    .clone()
                    .map(|project| (Span::styled(project, gray()), false)),
            ),
            StatusFact::Context => readings.extend(facts.context.clone()),
            StatusFact::Tasks => readings.extend(
                (facts.work.tasks > 0)
                    .then(|| counted(facts.work.tasks, "background task", "background tasks")),
            ),
            StatusFact::Subagents => readings.extend(
                (facts.work.subagents > 0)
                    .then(|| counted(facts.work.subagents, "sub-agent", "sub-agents")),
            ),
        }
    }
    // A column clear at the right edge, and two between the sides. The right side has what the
    // level leaves: a hint cut to it, unless it would show no more than its ellipsis, or the
    // readings that fit whole, in order, up to the first that does not, since a reading cut
    // short reads as another number.
    let level_width = TEXT_EDGE + display_width(&level.content);
    let room = width.saturating_sub(level_width + 2 + draw::RIGHT_MARGIN);
    let right = match facts.hint {
        Some(hint) if display_width(hint) <= room || room >= LEAST_CUT_COLUMNS => Span::styled(
            draw::cut_text(hint, room),
            Style::default().fg(Color::Yellow),
        ),
        Some(_) => Span::raw(String::new()),
        None => {
            let mut shown = String::new();
            for reading in &readings {
                let dot = if shown.is_empty() { "" } else { DOT };
                let left = room.saturating_sub(display_width(&shown) + display_width(dot));
                if display_width(reading) > left {
                    break;
                }
                shown.push_str(dot);
                shown.push_str(reading);
            }
            Span::styled(shown, gray())
        }
    };
    let right_width = display_width(&right.content);
    // The gap between the sides only when there is a right side to keep apart.
    let reserved = if right_width == 0 {
        draw::RIGHT_MARGIN
    } else {
        right_width + 2 + draw::RIGHT_MARGIN
    };
    let mut budget = width.saturating_sub(reserved + level_width);
    let mut left = vec![Span::raw(" ".repeat(TEXT_EDGE)), level];
    for (fact, from_start) in after {
        let wanted = display_width(DOT) + display_width(&fact.content);
        if wanted <= budget {
            budget -= wanted;
            left.push(Span::styled(DOT.to_string(), gray()));
            left.push(fact);
            continue;
        }
        let room = budget.saturating_sub(display_width(DOT));
        if room >= LEAST_CUT_COLUMNS {
            let cut = if from_start {
                draw::cut_text_start(&fact.content, room)
            } else {
                draw::cut_text(&fact.content, room)
            };
            left.push(Span::styled(DOT.to_string(), gray()));
            left.push(Span::styled(cut, fact.style));
        }
        break;
    }
    let left = draw::cut_line(Line::from(left), width.saturating_sub(reserved));
    let left_width: usize = left
        .spans
        .iter()
        .map(|span| display_width(&span.content))
        .sum();
    let fill = width.saturating_sub(left_width + right_width + 1);
    let mut spans = left.spans;
    spans.push(Span::raw(" ".repeat(fill)));
    spans.push(right);
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
    /// A compaction runs and nothing more specific has been said: the summarizer, or the
    /// checkpoint turn between the things it says.
    Compacting,
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
        SPINNER
            .get(frame % SPINNER.len())
            .copied()
            .unwrap_or_default()
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
        Activity::Compacting => "Compacting the conversation",
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
/// would cover the row's first `taken` columns (the working row's words); `hovered`, the gray and
/// the words swap, which is the hint that a click goes there. Returns the cells it covers.
pub(crate) fn draw_back_to_bottom(
    frame: &mut Frame<'_>,
    area: Rect,
    taken: u16,
    hovered: bool,
) -> Rect {
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
    let style = if hovered {
        Style::default().fg(COMPOSER_BACKGROUND).bg(draw::WHITE)
    } else {
        Style::default().fg(draw::WHITE).bg(COMPOSER_BACKGROUND)
    };
    frame.render_widget(Paragraph::new(Line::from(Span::styled(label, style))), pill);
    pill
}

/// Whether the cell at `column`, `row` is on `pill`, the back-to-bottom pill of the last frame:
/// what a press there and the pointer over it both ask.
pub(crate) fn is_on_pill(pill: Option<Rect>, column: u16, row: u16) -> bool {
    pill.is_some_and(|pill| {
        row == pill.y && column >= pill.x && column < pill.x.saturating_add(pill.width)
    })
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
    let shown: Vec<Line<'static>> = lines.into_iter().skip(skip).map(draw::on_shade).collect();
    let cursor_row = shown.len().saturating_sub(1);
    let cursor_column = TEXT_EDGE + display_width(prompt) + display_width(answer);
    // The composer's background, words a column short of its edge: the question stands where the
    // composer stood.
    let block = Style::default().bg(super::input::COMPOSER_BACKGROUND);
    frame.render_widget(ratatui::widgets::Block::new().style(block), area);
    frame.render_widget(Paragraph::new(Text::from(shown)).style(block), Rect {
        width: area.width.saturating_sub(draw::RIGHT_MARGIN as u16),
        ..area
    });
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
/// the longest value, a row too long cut a column short of the right edge, the picked row one
/// reversed bar from edge to edge, nothing on it dimmed.
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
                            TEXT_EDGE
                                + display_width(&row.value)
                                + display_width(description)
                                + draw::RIGHT_MARGIN,
                        )
                        .max(2),
                };
                spans.push(Span::styled(format!("{:pad$}{description}", ""), dim));
            }
            let mut line =
                draw::cut_line(Line::from(spans), width.saturating_sub(draw::RIGHT_MARGIN));
            if index == picked {
                let used = line.width();
                line.spans
                    .push(Span::raw(" ".repeat(width.saturating_sub(used))));
                line.patch_style(Style::default().add_modifier(Modifier::REVERSED))
            } else {
                line
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
            .map(draw::on_shade)
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
            fold_steps: true,
            expand_streaming_thinking: false,
            prompt_padding: PromptPadding::Full,
        }
    }

    /// An item's lines at `width`, its paths shown against `/w`.
    fn lines_at(entry: &Entry, width: usize) -> Vec<Line<'static>> {
        entry_lines(entry, width, Path::new("/w"), PromptPadding::Full)
    }

    fn entry(item: Item, expanded: bool) -> Entry {
        Entry {
            id: 1,
            item,
            expanded,
            revision: 1,
            compaction: None,
            left_to_replay: false,
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
            exit_code: None,
        }
    }

    /// While an answer streams, the row after its last newline is laid out as markdown on its own:
    /// an item still arriving already hangs under its marker. What it says yet never reaches back
    /// into the rows above, which a `2` behind an item or a `1` behind a line would as a
    /// paragraph's next word, and `---` under a paragraph as its heading underline; a line inside
    /// a code block still open is a code row, not markdown.
    #[test]
    fn the_open_row_is_laid_out_as_it_will_be_once_it_ends() {
        let rows = |text: &str| {
            plain(&lines_at(
                &entry(
                    Item::Text {
                        text: text.into(),
                        streaming: true,
                    },
                    false,
                ),
                30,
            ))
        };
        assert_eq!(
            rows("Done.\n\n1. an item still arriving, past the width"),
            vec![
                "  Done.",
                "",
                "  1. an item still arriving,",
                "     past the width",
            ]
        );
        let underlined = lines_at(
            &entry(
                Item::Text {
                    text: "Done.\n---".into(),
                    streaming: true,
                },
                false,
            ),
            30,
        );
        assert!(
            underlined
                .iter()
                .flat_map(|line| &line.spans)
                .filter(|span| span.content.contains("Done."))
                .all(|span| !span.style.add_modifier.contains(Modifier::BOLD)),
            "a paragraph is not made a heading by the line arriving under it: {underlined:?}"
        );
        assert_eq!(rows("1. a\n2"), vec!["  1. a", "  2"]);
        assert_eq!(rows("Steps:\n1"), vec!["  Steps:", "  1"]);
        assert_eq!(rows("```\nlet x = 1;\n# not a heading"), vec![
            "",
            "   let x = 1;",
            "   # not a heading",
            ""
        ]);
    }

    /// Every kind of item puts its glyph in column 0 and its text at the edge, so the whole
    /// transcript shares one left margin. A line in Meka's voice starts with a capital and ends in
    /// no period, whatever the sentence it was drawn from; a log row is drawn as written, less
    /// its trailing whitespace.
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
                "• Slow today",
            ),
            (
                Item::Notice {
                    level: crate::frontend::NoticeLevel::Warn,
                    text: "careful".into(),
                },
                "! Careful",
            ),
            (
                Item::note("Checklist: items still open, continuing"),
                "• Checklist: items still open, continuing",
            ),
            (
                Item::note("Stopping 2 background tasks"),
                "• Stopping 2 background tasks",
            ),
            (Item::interrupted(), "! Interrupted"),
            (Item::Error("provider error".into()), "■ Provider error"),
            (Item::note("file_write allowed"), "• file_write allowed"),
            (
                Item::Log {
                    target: "meka::agent::turn".into(),
                    message: "turn started".into(),
                },
                "• meka::agent::turn: turn started",
            ),
            (
                Item::Log {
                    target: "meka".into(),
                    message: "resolved the config.".into(),
                },
                "• meka: resolved the config.",
            ),
            (Item::note("Rewound 1 turn."), "• Rewound 1 turn"),
            (Item::note("Still going..."), "• Still going..."),
            (Item::note("Approvals on"), "• Approvals on"),
            (Item::ShellOutput("total 0".into()), "  total 0"),
            (
                Item::SessionId {
                    label: "Resuming session".into(),
                    id: "abc".into(),
                },
                "• Resuming session abc",
            ),
            (Item::Image(draw::Picture::unread()), "• [image]"),
        ];
        for (item, expected) in cases {
            let rows = plain(&lines_at(&entry(item.clone(), false), 60));
            assert_eq!(rows[0], expected, "{item:?}");
        }
        let error = lines_at(&entry(Item::Error("x".into()), false), 60);
        assert_eq!(error[0].spans[0].style.fg, Some(Color::Red));
        let warning = lines_at(
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
        let log = Item::Log {
            target: "meka".into(),
            message: "kept\n  ".into(),
        };
        let rows = plain(&lines_at(&entry(log, false), 60));
        assert_eq!(rows, ["• meka: kept"], "{rows:?}");
    }

    /// No row an item lays out is wider than the width it was laid out at, whatever the content,
    /// because the transcript clips what is wider: a token longer than the row, an address, a code
    /// row, a table cell, a command, a diff line, a checklist item, a path. And no row holds a tab,
    /// because the screen hands cells to the terminal and drops a tab with the indentation it
    /// carried. Every kind, expanded and collapsed, at widths from a column to a wide pane, as the
    /// viewport lays them out: what does not fit even a narrow one is cut at the edge.
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
                exit_code: None,
            };
        let command = serde_json::json!({ "command": format!("\trun --flag={token} {address}") });
        let edit = serde_json::json!({ "path": format!("{token}.rs"), "old_string": "a", "new_string": "b" });
        let change = crate::conversation::Change::between(
            Some(&format!("old {token}\n")),
            Some(&format!("new {address}\n\tsecond {token}\n")),
        );
        let items = vec![
            Item::Prompt {
                text: format!("{token} {address}\n\tindented {token}"),
                class: Some(InboxClass::Steer),
                inbox_id: None,
                delivered: false,
                ran_under: RanUnder::Level(crate::permission::Permission::Read),
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
            Item::Log {
                target: address.clone(),
                message: format!("{token}\t{address}"),
            },
            Item::ShellOutput(format!("\x1b[32m{token}\x1b[0m\n\t{address}")),
            Item::SessionId {
                label: token.clone(),
                id: address.clone(),
            },
            Item::TurnEnd(format!(
                "Finished at 19:28 after 19m 19s with {token} tokens in"
            )),
            Item::Image(draw::Picture::unread()),
            Item::Banner {
                version: token.clone(),
                revision: Some(address.clone()),
                debug: true,
                mcp_servers: 2,
                skills: 12,
                memories: 48,
                instructions: 3,
            },
        ];
        let mut too_wide = Vec::new();
        for width in [1, 2, 5, 10, 16, 24, 40, 80, 200] {
            for (index, item) in items.iter().enumerate() {
                for expanded in [false, true] {
                    let lines = entry_rows(
                        &entry(item.clone(), expanded),
                        width,
                        Path::new("/w"),
                        PromptPadding::Full,
                    )
                    .0;
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
        let collapsed = plain(&lines_at(&entry(item.clone(), false), 60));
        assert_eq!(collapsed, vec![
            "• Thinking · First line. Second paragraph."
        ]);
        let expanded = plain(&lines_at(&entry(item, true), 60));
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
        let collapsed = plain(&lines_at(&entry(item.clone(), false), 60));
        assert_eq!(collapsed, vec![
            "• shell_execute ls -la",
            "  └ line 1",
            "    line 2",
            "    line 3",
            "    … +5 lines",
        ]);
        let expanded = plain(&lines_at(&entry(item, true), 60));
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
        let running = plain(&lines_at(
            &entry(tool_call(ToolState::Running, "", None), false),
            60,
        ));
        assert_eq!(running, vec!["• shell_execute ls -la", "  └ Running"]);
    }

    /// A finished call shows its result, which is what the log keeps and a resume shows, rather
    /// than what it wrote while it ran.
    #[test]
    fn a_finished_call_shows_its_result_rather_than_its_stream() {
        let item = tool_call(
            ToolState::Done { is_error: false },
            "streamed\n",
            Some("kept\n"),
        );
        assert_eq!(plain(&lines_at(&entry(item, false), 60)), vec![
            "• shell_execute ls -la",
            "  └ kept",
        ]);
        let running = tool_call(ToolState::Running, "streamed\n", None);
        assert_eq!(plain(&lines_at(&entry(running, false), 60)), vec![
            "• shell_execute ls -la",
            "  └ streamed",
        ]);
    }

    /// A collapsed call shows its program's first line on its row and the next two on the bar
    /// under it, then a count of the rest; a single line past the two is shown rather than
    /// counted, since the count would take its row.
    #[test]
    fn a_command_shows_its_first_lines_and_counts_the_rest() {
        let call = |command: &str| {
            let item = Item::ToolCall {
                id: "1".into(),
                name: "shell_execute".into(),
                input: Some(serde_json::json!({"command": command})),
                summary: None,
                state: ToolState::Done { is_error: false },
                output: String::new(),
                result: Some("(no output)".into()),
                result_images: 0,
                activity: String::new(),
                progress: None,
                change: None,
                exit_code: None,
            };
            plain(&lines_at(&entry(item, false), 60))
        };
        assert_eq!(call("one\ntwo\nthree\nfour"), vec![
            "• shell_execute one",
            "  │ two",
            "  │ three",
            "  │ four",
            "  └ (no output)",
        ]);
        assert_eq!(call("one\ntwo\nthree\nfour\nfive"), vec![
            "• shell_execute one",
            "  │ two",
            "  │ three",
            "  │ … +2 lines",
            "  └ (no output)",
        ]);
        assert_eq!(call("ls\nfor i in 1 2; do echo $i; done"), vec![
            "• shell_execute ls",
            "  │ for i in 1 2; do echo $i; done",
            "  └ (no output)",
        ]);
    }

    /// A path a call names is shown as the status row shows a directory: relative to the
    /// working directory under it, the directory itself as `.`, `~` for the home directory, and
    /// a relative path as written.
    #[test]
    fn a_path_is_shown_against_the_working_directory() {
        let working_directory = if cfg!(windows) {
            Path::new(r"C:\w")
        } else {
            Path::new("/w")
        };
        let read = |path: &str| {
            let item = Item::ToolCall {
                id: "1".into(),
                name: "file_read".into(),
                input: Some(serde_json::json!({"path": path})),
                summary: Some(path.into()),
                state: ToolState::Done { is_error: false },
                output: String::new(),
                result: None,
                result_images: 0,
                activity: String::new(),
                progress: None,
                change: None,
                exit_code: None,
            };
            plain(&entry_lines(
                &entry(item, false),
                80,
                working_directory,
                PromptPadding::Full,
            ))
            .remove(0)
        };
        let under = working_directory.join("src/main.rs");
        assert_eq!(
            read(&under.display().to_string()),
            "• file_read src/main.rs"
        );
        assert_eq!(
            read(&working_directory.display().to_string()),
            "• file_read ."
        );
        assert_eq!(
            read("/elsewhere/notes.md"),
            "• file_read /elsewhere/notes.md"
        );
        assert_eq!(read("docs/a.md"), "• file_read docs/a.md");
        if let Some(home) = dirs::home_dir() {
            let inside = home.join("notes.md");
            assert_eq!(
                read(&inside.display().to_string()),
                "• file_read ~/notes.md"
            );
        }
    }

    /// A command that opens by changing into the directory it already runs in has that opening
    /// dimmed, as it ran, so the part that does something stands out; a `cd` anywhere else is
    /// part of the command.
    #[test]
    fn a_cd_into_the_working_directory_is_dimmed() {
        let dimmed = |command: &str| -> String {
            call_summary(
                "shell_execute",
                &serde_json::json!({ "command": command }),
                None,
                Path::new("/w"),
                80,
            )
            .iter()
            .filter(|span| span.style.fg == Some(Color::DarkGray))
            .map(|span| span.content.as_ref())
            .collect()
        };
        assert_eq!(dimmed("cd /w && ls"), "cd /w && ");
        assert_eq!(dimmed("cd '/w/' ; ls"), "cd '/w/' ; ");
        assert_eq!(dimmed("cd /other && ls"), "");
        assert_eq!(dimmed("cd /w"), "", "a lone cd is the whole command");
        assert_eq!(dimmed("cd /work && ls"), "", "only the directory itself");
        assert_eq!(idle_cd_columns("cd /w && ls", Path::new("")), 0);
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
            exit_code: None,
        };
        let rows = plain(&lines_at(&entry(item, true), 60));
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
        let collapsed = plain(&lines_at(&entry(item.clone(), false), 40));
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
        let expanded = plain(&lines_at(&entry(item, true), 40));
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
            change: Some(crate::conversation::Change::between(
                Some("import sys\nprint(\"Hello, World!\")\nsys.exit(0)\n"),
                Some("import sys\nprint(\"Hello, world!\")\nsys.exit(0)\n"),
            )),
            exit_code: None,
        };
        let collapsed = plain(&lines_at(&entry(item.clone(), false), 60));
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
            change: Some(crate::conversation::Change::between(
                None,
                Some(&(1..=60).map(|n| format!("line {n}\n")).collect::<String>()),
            )),
            exit_code: None,
        };
        let collapsed = plain(&lines_at(&entry(rewrite, false), 60));
        assert_eq!(
            collapsed.len(),
            1 + draw::DIFF_PREVIEW_ROWS + 1,
            "a long change shows its first rows and counts the rest: {collapsed:?}"
        );
        assert_eq!(
            collapsed.last().map(String::as_str),
            Some("    … +40 lines")
        );
        let expanded = plain(&lines_at(&entry(item, true), 60));
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

    /// A delete draws every line it removed, in red with its old number, and a memory's change
    /// leaves the fields the diff shows out of the argument block, the way a file's does.
    #[test]
    fn a_delete_draws_every_line_removed() {
        let item = Item::ToolCall {
            id: "1".into(),
            name: "memory_delete".into(),
            input: Some(serde_json::json!({"name": "alice-timezone"})),
            summary: Some("alice-timezone".into()),
            state: ToolState::Done { is_error: false },
            output: String::new(),
            result: Some("Deleted memory 'alice-timezone'.".into()),
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: Some(crate::conversation::Change::between(
                Some("---\ndescription: Alice is in Lisbon\n---\n\nSince 2024.\n"),
                None,
            )),
            exit_code: None,
        };
        let collapsed = plain(&lines_at(&entry(item, false), 60));
        assert_eq!(collapsed, vec![
            "• memory_delete alice-timezone  +0 -5",
            "  └ 1 - ---",
            "    2 - description: Alice is in Lisbon",
            "    3 - ---",
            "    4 -",
            "    5 - Since 2024.",
        ]);
        let write = Item::ToolCall {
            id: "2".into(),
            name: "memory_write".into(),
            input: Some(serde_json::json!({"name": "alice-timezone", "body": "Since 2025."})),
            summary: Some("alice-timezone".into()),
            state: ToolState::Done { is_error: false },
            output: String::new(),
            result: Some("Saved memory".into()),
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: Some(crate::conversation::Change::between(
                Some("Since 2024.\n"),
                Some("Since 2025.\n"),
            )),
            exit_code: None,
        };
        let expanded = plain(&lines_at(&entry(write, true), 60));
        assert!(
            expanded.iter().any(|row| row == "  │ name: alice-timezone")
                && !expanded.iter().any(|row| row.contains("body")),
            "the fields the diff already shows are left out of the block: {expanded:?}"
        );
    }

    /// A failed call says so on its row, in red: the code a program exited with, or that it
    /// failed. What a tool said of the failure is drawn in red under it; what a program wrote is
    /// its output, without the line that states the code the row already states.
    #[test]
    fn a_failed_call_says_so_on_its_row() {
        let lines = lines_at(&entry(item_with_a_failure(), false), 60);
        assert_eq!(plain(&lines), vec![
            "• file_read /nope  failed",
            "  └ no such file"
        ]);
        let tag = lines[0].spans.last().expect("the tag");
        assert_eq!(
            (tag.content.as_ref(), tag.style.fg),
            ("failed", Some(Color::Red))
        );
        assert!(
            lines[1]
                .spans
                .iter()
                .any(|span| span.content.contains("no such file")
                    && span.style.fg == Some(Color::Red)),
            "{lines:?}"
        );
        let mut long = item_with_a_failure();
        if let Item::ToolCall { result, .. } = &mut long {
            *result = Some(format!("{} {}", "a".repeat(20), "b".repeat(20)));
        }
        let expanded = plain(&lines_at(&entry(long, true), 30));
        assert_eq!(expanded, vec![
            "• file_read  failed",
            "  │ path: /nope",
            &format!("  └ {}", "a".repeat(20)),
            &format!("    {}", "b".repeat(20)),
        ]);

        let exited = Item::ToolCall {
            id: "1".into(),
            name: "shell_execute".into(),
            input: Some(serde_json::json!({"command": "make check"})),
            summary: None,
            state: ToolState::Done { is_error: true },
            output: String::new(),
            result: Some("make: *** No rule to make target 'check'.\n\nExit code: 2".into()),
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: None,
            exit_code: Some(2),
        };
        let lines = lines_at(&entry(exited, false), 60);
        assert_eq!(plain(&lines), vec![
            "• shell_execute make check  exit 2",
            "  └ make: *** No rule to make target 'check'.",
        ]);
        assert!(
            lines[1]
                .spans
                .iter()
                .all(|span| span.style.fg != Some(Color::Red)),
            "a program's output is not the failure: {lines:?}"
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
            exit_code: None,
        }
    }

    fn call_named(name: &str, state: ToolState) -> Item {
        Item::ToolCall {
            id: name.into(),
            name: name.into(),
            input: Some(serde_json::json!({})),
            summary: None,
            state,
            output: String::new(),
            result: Some("done".into()),
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: None,
            exit_code: None,
        }
    }

    fn thought() -> Item {
        Item::Thinking {
            text: "Hmm.".into(),
            estimate: None,
            ended: true,
        }
    }

    fn said(words: &str) -> Item {
        Item::Text {
            text: words.into(),
            streaming: false,
        }
    }

    /// The rows of a frame of `transcript` at 60 columns, trailing blanks dropped.
    fn frame_rows(viewport: &mut Viewport, transcript: &Transcript, height: u16) -> Vec<String> {
        let backend = ratatui::backend::TestBackend::new(60, height);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), transcript))
            .expect("a frame");
        let buffer = terminal.backend().buffer().clone();
        (0..height)
            .map(|row| {
                (0..60)
                    .map(|column| buffer[(column, row)].symbol().to_string())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    /// Each row of a frame as text, with whether its first cell is drawn reversed, as the
    /// keyboard's cursor is.
    fn frame_rows_styled(
        viewport: &mut Viewport,
        transcript: &Transcript,
        height: u16,
    ) -> Vec<(String, bool)> {
        let backend = ratatui::backend::TestBackend::new(60, height);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), transcript))
            .expect("a frame");
        let buffer = terminal.backend().buffer().clone();
        (0..height)
            .map(|row| {
                let text = (0..60)
                    .map(|column| buffer[(column, row)].symbol().to_string())
                    .collect::<String>()
                    .trim_end()
                    .to_string();
                let reversed = buffer[(0, row)]
                    .style()
                    .add_modifier
                    .contains(Modifier::REVERSED);
                (text, reversed)
            })
            .collect()
    }

    /// No item's words reach the last column, a prompt's included, while the prompt's gray does:
    /// an answer wraps a column short of the edge, as the prompt above it does.
    #[test]
    fn the_transcript_keeps_the_last_column_clear_of_words() {
        let mut transcript = Transcript::new(settings());
        transcript.push(Item::Prompt {
            text: "x".repeat(58),
            class: None,
            inbox_id: None,
            delivered: true,
            ran_under: RanUnder::Level(crate::permission::Permission::Read),
        });
        transcript.push(said(&"y".repeat(58)));
        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let mut viewport = Viewport::new();
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
            .expect("a frame");
        let buffer = terminal.backend().buffer();
        let row = |y: u16| {
            (0..60)
                .map(|x| buffer[(x, y)].symbol().to_string())
                .collect::<String>()
                .trim_end()
                .to_string()
        };
        let rows: Vec<String> = (0..10).map(row).filter(|row| !row.is_empty()).collect();
        assert_eq!(rows, vec![
            "▎".to_string(),
            format!("▎ {}", "x".repeat(57)),
            "▎ x".to_string(),
            "▎".to_string(),
            format!("  {}", "y".repeat(57)),
            "  y".to_string(),
        ]);
        assert_eq!(
            buffer[(59, 1)].bg,
            COMPOSER_BACKGROUND,
            "the prompt's gray runs to the edge"
        );
    }

    /// A finished run of two or more steps folds into one row that says what it did, in the
    /// order it first did each, with its failures in red; a single step, a change and the words
    /// around them stand as they are.
    #[test]
    fn a_finished_run_folds_into_one_row() {
        let mut transcript = Transcript::new(settings());
        transcript.push(thought());
        transcript.push(call_named("shell_execute", ToolState::Done {
            is_error: false,
        }));
        transcript.push(call_named("shell_execute", ToolState::Done {
            is_error: true,
        }));
        transcript.push(call_named("file_read", ToolState::Done { is_error: false }));
        transcript.push(thought());
        transcript.push(said("Now the change."));
        let mut edit = call_named("file_edit", ToolState::Done { is_error: false });
        if let Item::ToolCall { change, .. } = &mut edit {
            *change = Some(crate::conversation::Change::between(
                Some("a\n"),
                Some("b\n"),
            ));
        }
        transcript.push(edit);
        transcript.push(thought());
        transcript.push(said("Done."));
        let rows = frame_rows(&mut Viewport::new(), &transcript, 20);
        let shown: Vec<&str> = rows
            .iter()
            .map(String::as_str)
            .filter(|row| !row.is_empty())
            .collect();
        assert_eq!(shown, vec![
            "▸ Thought twice, ran 2 commands, read a file  1 failed",
            "  Now the change.",
            "• file_edit  +1 -1",
            "  └ 1 - a",
            "    1 + b",
            "• Thinking · Hmm.",
            "  Done.",
        ]);
    }

    /// A run the agent is still working through shows its steps; it folds once the agent says
    /// something after it, and a run that finishes while the view is scrolled up waits until the
    /// view is back at the bottom, so what the reader is looking at does not move.
    #[test]
    fn a_run_folds_once_finished_and_waits_for_the_bottom() {
        let mut transcript = Transcript::new(settings());
        transcript.apply(FrontendEvent::TurnStarted {
            turn_id: uuid::Uuid::new_v4(),
        });
        transcript.push(thought());
        transcript.push(call_named("shell_execute", ToolState::Done {
            is_error: false,
        }));
        let mut viewport = Viewport::new();
        let rows = frame_rows(&mut viewport, &transcript, 3);
        assert!(
            rows.iter().any(|row| row.starts_with("• shell_execute")),
            "an unfinished run shows its steps: {rows:?}"
        );
        viewport.scroll_to_top();
        transcript.push(said("Found it."));
        let rows = frame_rows(&mut viewport, &transcript, 3);
        assert!(
            rows.iter().any(|row| row.starts_with("• Thinking"))
                && !rows.iter().any(|row| row.starts_with(['▾', '▸'])),
            "scrolled up, the finished run waits as it was drawn, no row put above it: {rows:?}"
        );
        viewport.scroll_to_bottom();
        let rows = frame_rows(&mut viewport, &transcript, 3);
        assert!(
            rows.iter()
                .any(|row| row == "▸ Thought once, ran a command")
                && !rows.iter().any(|row| row.starts_with("• shell_execute")),
            "back at the bottom, it folds: {rows:?}"
        );
    }

    /// A run of a running turn stays open while its call is at work, even once a steer the turn
    /// has read follows it, and while what follows is a steer still waiting to be delivered, which
    /// the agent has not been told; so what the call writes, and what the steer comes after, stay
    /// in view.
    #[test]
    fn a_run_whose_call_is_at_work_stays_open() {
        let steer = |delivered| Item::Prompt {
            text: "steer now".into(),
            class: Some(InboxClass::Steer),
            inbox_id: None,
            delivered,
            ran_under: RanUnder::Pending,
        };
        for (state, delivered) in [
            (ToolState::Running, true),
            (ToolState::Done { is_error: false }, false),
        ] {
            let mut transcript = Transcript::new(settings());
            transcript.apply(FrontendEvent::TurnStarted {
                turn_id: uuid::Uuid::new_v4(),
            });
            transcript.push(thought());
            transcript.push(call_named("shell_execute", state));
            transcript.push(steer(delivered));
            let rows = frame_rows(&mut Viewport::new(), &transcript, 20);
            assert!(
                rows.iter().any(|row| row.starts_with("• shell_execute"))
                    && !rows.iter().any(|row| row.starts_with('▸')),
                "{state:?}, delivered={delivered}: {rows:?}"
            );
        }
    }

    /// Words of spaces alone and a `!command` beside the session are not said to the agent, so
    /// the run before them goes on while the turn runs.
    #[test]
    fn what_is_not_said_to_the_agent_does_not_end_a_run() {
        for after in [said("  \n"), Item::Prompt {
            text: "!ls".into(),
            class: None,
            inbox_id: None,
            delivered: true,
            ran_under: RanUnder::Shell,
        }] {
            let mut transcript = Transcript::new(settings());
            transcript.apply(FrontendEvent::TurnStarted {
                turn_id: uuid::Uuid::new_v4(),
            });
            transcript.push(thought());
            transcript.push(call_named("shell_execute", ToolState::Done {
                is_error: false,
            }));
            transcript.push(after.clone());
            let rows = frame_rows(&mut Viewport::new(), &transcript, 20);
            assert!(
                !rows.iter().any(|row| row.starts_with('▸')),
                "{after:?}: {rows:?}"
            );
        }
    }

    /// A run the user opened stays open when the step it was first drawn under leaves it, as a
    /// call that turns out to have changed a file does: its state is found by any step still in
    /// it.
    #[test]
    fn a_run_stays_open_when_its_first_step_leaves_it() {
        let mut transcript = Transcript::new(settings());
        transcript.push(call_named("file_edit", ToolState::Running));
        transcript.push(thought());
        transcript.push(call_named("shell_execute", ToolState::Done {
            is_error: false,
        }));
        transcript.push(said("Done."));
        let mut viewport = Viewport::new();
        frame_rows(&mut viewport, &transcript, 30);
        let run = viewport.run_at_row(0).expect("the run's row");
        viewport.toggle_run(run);
        frame_rows(&mut viewport, &transcript, 30);
        transcript.apply(FrontendEvent::ToolCallCompleted {
            id: "file_edit".into(),
            name: "file_edit".into(),
            is_error: false,
            content: Vec::new(),
            metadata: None,
            display: Some(crate::conversation::ResultDisplay {
                change: Some(crate::conversation::Change::between(
                    Some("a\n"),
                    Some("b\n"),
                )),
                ..crate::conversation::ResultDisplay::default()
            }),
        });
        let rows = frame_rows(&mut viewport, &transcript, 30);
        assert!(
            rows.iter()
                .any(|row| row.starts_with("▾ Thought once, ran a command"))
                && rows.iter().any(|row| row.starts_with("• shell_execute")),
            "{rows:?}"
        );
    }

    /// The cursor stands on a stop of the frame it is drawn in: under a run that folds, it moves
    /// to the run's own row and lights it.
    #[test]
    fn the_cursor_moves_to_a_run_s_row_when_the_run_folds() {
        let mut transcript = Transcript::new(settings());
        transcript.push(thought());
        transcript.push(call_named("shell_execute", ToolState::Done {
            is_error: false,
        }));
        transcript.push(said("Done."));
        let mut viewport = Viewport::new();
        frame_rows(&mut viewport, &transcript, 12);
        let run_row = viewport
            .cursor_before(&transcript, None)
            .expect("the run's row");
        viewport.set_cursor(Some(run_row));
        let run = viewport.run_under_cursor().expect("a run");
        viewport.toggle_run(run);
        frame_rows(&mut viewport, &transcript, 12);
        let first_step = viewport
            .cursor_after(&transcript, run_row)
            .expect("the first step");
        assert!(first_step.under_run);
        viewport.set_cursor(Some(first_step));
        viewport.toggle_run(run);
        let lit = frame_rows_styled(&mut viewport, &transcript, 12);
        assert_eq!(viewport.cursor(), Some(run_row), "{lit:?}");
        let reversed: Vec<&str> = lit
            .iter()
            .filter(|(_, reversed)| *reversed)
            .map(|(text, _)| text.as_str())
            .collect();
        assert_eq!(reversed, vec!["▸ Thought once, ran a command"]);
    }

    /// A cursor on a step stays on the step when a run forms around it and heads it with its row,
    /// rather than passing to that row, which a toggle would fold.
    #[test]
    fn the_cursor_stays_on_a_step_a_run_forms_around() {
        let mut transcript = Transcript::new(Settings {
            fold_steps: false,
            expand_streaming_thinking: false,
            ..settings()
        });
        transcript.apply(FrontendEvent::TurnStarted {
            turn_id: uuid::Uuid::new_v4(),
        });
        transcript.push(thought());
        transcript.push(call_named("shell_execute", ToolState::Done {
            is_error: false,
        }));
        let mut viewport = Viewport::new();
        frame_rows(&mut viewport, &transcript, 12);
        let step = CursorStop {
            index: 0,
            under_run: false,
        };
        viewport.set_cursor(Some(step));
        assert_eq!(viewport.cursor(), Some(step), "a step on its own");
        transcript.push(said("Done."));
        frame_rows(&mut viewport, &transcript, 12);
        assert_eq!(
            viewport.cursor(),
            Some(CursorStop {
                index: 0,
                under_run: true,
            }),
            "the step, under the row of the run it now heads"
        );
        assert_eq!(viewport.run_under_cursor(), None);
    }

    /// The bar a prompt scrolled off the top is stuck in covers the top row, so that row copies as
    /// nothing; where the top row is the only one, no bar is stuck over it.
    #[test]
    fn the_stuck_prompt_s_bar_copies_as_nothing_and_needs_a_second_row() {
        let mut transcript = Transcript::new(settings());
        transcript.push(Item::Prompt {
            text: "a question".into(),
            class: None,
            inbox_id: None,
            delivered: true,
            ran_under: RanUnder::Level(crate::permission::Permission::Read),
        });
        for words in 0..30 {
            transcript.push(said(&format!("words {words}")));
        }
        let mut viewport = Viewport::new();
        frame_rows(&mut viewport, &transcript, 8);
        assert!(
            viewport.sticky_prompt().is_some(),
            "the prompt is off the top"
        );
        assert_eq!(viewport.row_copy(0), draw::RowCopy::Padding);
        assert_eq!(viewport.row_copy(1), draw::RowCopy::Text);
        frame_rows(&mut viewport, &transcript, 1);
        assert!(viewport.sticky_prompt().is_none());
        assert_eq!(viewport.row_copy(0), draw::RowCopy::Text);
    }

    /// Opening an item where the whole transcript fits, and closing one at the bottom, leave the
    /// view following the bottom, so the pill does not show and what comes next scrolls into
    /// view.
    #[test]
    fn toggling_an_item_at_the_bottom_keeps_following_it() {
        let mut transcript = Transcript::new(settings());
        transcript.push(thought());
        transcript.push(said("Done."));
        let mut viewport = Viewport::new();
        frame_rows(&mut viewport, &transcript, 20);
        assert!(viewport.is_following());
        for _ in 0..2 {
            assert!(transcript.toggle(0));
            viewport.keep_in_place(0);
            frame_rows(&mut viewport, &transcript, 20);
            assert!(viewport.is_following(), "the transcript still fits");
        }
        // Taller than the view: the last item opened grows past the bottom, and closed again it
        // leaves the view at the bottom, following.
        for words in 0..30 {
            transcript.push(said(&format!("words {words}")));
        }
        transcript.push(Item::Thinking {
            text: (0..20).map(|line| format!("line {line}\n")).collect(),
            estimate: None,
            ended: true,
        });
        frame_rows(&mut viewport, &transcript, 20);
        let last = transcript.drawn().count() - 1;
        assert!(transcript.toggle(last));
        viewport.keep_in_place(last);
        frame_rows(&mut viewport, &transcript, 20);
        assert!(!viewport.is_following(), "what opened runs past the bottom");
        assert!(transcript.toggle(last));
        viewport.keep_in_place(last);
        frame_rows(&mut viewport, &transcript, 20);
        assert!(
            viewport.is_following(),
            "closed at the bottom, it follows again"
        );
    }

    /// A command's first line shows on its row and the next under it; a row too narrow for the
    /// first shows neither, rather than a command that seems to start at its second line.
    #[test]
    fn a_command_too_narrow_for_its_first_line_shows_none_of_it() {
        let call = entry(
            Item::ToolCall {
                id: "c1".into(),
                name: "shell_execute".into(),
                input: Some(serde_json::json!({"command": "make\nmake check"})),
                summary: None,
                state: ToolState::Done { is_error: false },
                output: String::new(),
                result: Some("done".into()),
                result_images: 0,
                activity: String::new(),
                progress: None,
                change: None,
                exit_code: None,
            },
            false,
        );
        let shown = |width| -> Vec<String> {
            lines_at(&call, width)
                .iter()
                .map(|line| {
                    line.spans
                        .iter()
                        .map(|span| span.content.as_ref())
                        .collect::<String>()
                })
                .collect()
        };
        assert!(
            shown(60).iter().any(|row| row.contains("make check")),
            "{:?}",
            shown(60)
        );
        assert!(
            !shown(20).iter().any(|row| row.contains("make")),
            "{:?}",
            shown(20)
        );
    }

    /// An open run is one block on its shade, from its row through its last step and the gaps
    /// between them; nothing on it is drawn in the terminal's dark gray, which the shade would
    /// swallow, and nothing past the run is shaded.
    #[test]
    fn an_open_run_is_one_shaded_block() {
        let mut transcript = Transcript::new(settings());
        transcript.push(thought());
        transcript.push(call_named("shell_execute", ToolState::Done {
            is_error: false,
        }));
        transcript.push(said("Done."));
        let mut viewport = Viewport::new();
        let backend = ratatui::backend::TestBackend::new(60, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let mut draw = |viewport: &mut Viewport| {
            terminal
                .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
                .expect("a frame");
            terminal.backend().buffer().clone()
        };
        let folded = draw(&mut viewport);
        assert!(
            (0..60).all(|column| folded[(column, 0)].bg == Color::Reset),
            "a folded run's row stands on no shade"
        );
        assert_eq!(folded[(0, 0)].symbol(), draw::FOLDED_GLYPH);
        let run = viewport.run_at_row(0).expect("the run's row");
        viewport.toggle_run(run);
        let open = draw(&mut viewport);
        let text = |row: u16| -> String {
            (0..60)
                .map(|column| open[(column, row)].symbol().to_string())
                .collect::<String>()
                .trim_end()
                .to_string()
        };
        assert_eq!(open[(0, 0)].symbol(), draw::OPENED_GLYPH);
        let last_step = (0..10)
            .rev()
            .find(|&row| text(row) == "  └ done")
            .expect("the step's result");
        for row in 0..=last_step {
            for column in 0..60 {
                let cell = &open[(column, row)];
                assert!(
                    cell.bg == draw::RUN_BACKGROUND,
                    "row {row} column {column} of the open run is shaded: {:?}",
                    text(row)
                );
                assert_ne!(
                    cell.fg,
                    Color::DarkGray,
                    "row {row} is drawn in a gray the shade swallows: {:?}",
                    text(row)
                );
            }
        }
        assert_eq!(open[(0, 0)].bg, draw::RUN_BACKGROUND);
        let words = (0..10)
            .find(|&row| text(row) == "  Done.")
            .expect("the words");
        assert!(
            (0..60).all(|column| open[(column, words)].bg == Color::Reset),
            "the words after the run stand on no shade"
        );
    }

    /// What a folded run hides is not laid out until the run opens, since most of a long
    /// session's history is steps nobody opens and each costs its highlighting.
    #[test]
    fn a_folded_run_lays_out_only_its_row() {
        let mut transcript = Transcript::new(settings());
        transcript.push(thought());
        transcript.push(call_named("shell_execute", ToolState::Done {
            is_error: false,
        }));
        transcript.push(said("Done."));
        let ids: Vec<u64> = transcript.drawn().map(|entry| entry.id).collect();
        let mut viewport = Viewport::new();
        viewport.lay_out_all(&transcript, 60);
        frame_rows(&mut viewport, &transcript, 12);
        assert!(
            viewport.cache.contains_key(&ids[0]),
            "the run's head is laid out"
        );
        assert!(
            !viewport.cache.contains_key(&ids[1]),
            "the step the run hides is not"
        );
        viewport.set_cursor(viewport.cursor_before(&transcript, None));
        let run = viewport.run_under_cursor().expect("a run");
        viewport.toggle_run(run);
        frame_rows(&mut viewport, &transcript, 12);
        assert!(
            viewport.cache.contains_key(&ids[1]),
            "it is laid out once the run opens"
        );
    }

    /// A compaction is a rule across the text labeled `→← Context compacted`, a boundary rather
    /// than something said. It opens to what set it going and what set it apart, its checkpoint's
    /// steps and its summary, one block on the shade an open run takes, and each step opens on its
    /// own: a memory longer than a collapsed diff shows whole once its call is opened. One that
    /// recorded nothing still opens, to what set it going.
    #[test]
    fn a_compaction_opens_to_its_checkpoint_and_summary() {
        use crate::conversation::{
            Change, CompactionRecord, CompactionTrigger, ContentBlock, Message, ResultDisplay,
            Role, ToolResultContent,
        };
        let memory: String = (1..=30).map(|line| format!("fact {line}\n")).collect();
        let record = CompactionRecord {
            trigger: CompactionTrigger::Requested,
            kept_recent: false,
            person_keeps_recent: None,
            checkpoint: [
                Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentBlock::Thinking {
                            parts: vec![crate::conversation::ReasoningPart::reasoning("Hmm.")],
                            opaque: None,
                            provider_fields: None,
                        },
                        ContentBlock::ToolUse {
                            id: "call-0".into(),
                            name: "memory_search".into(),
                            input: serde_json::json!({"queries": ["release"]}),
                            unparsed: None,
                            provider_fields: None,
                        },
                        ContentBlock::ToolUse {
                            id: "call-1".into(),
                            name: "memory_write".into(),
                            input: serde_json::json!({"name": "notes", "content": memory}),
                            unparsed: None,
                            provider_fields: None,
                        },
                    ],
                    provider_fields: None,
                },
                Message {
                    role: Role::User,
                    content: vec![
                        ContentBlock::ToolResult {
                            tool_use_id: "call-0".into(),
                            content: vec![ToolResultContent::Text {
                                text: "No memories.".into(),
                            }],
                            is_error: false,
                            display: None,
                        },
                        ContentBlock::ToolResult {
                            tool_use_id: "call-1".into(),
                            content: vec![ToolResultContent::Text {
                                text: "Saved memory 'notes'.".into(),
                            }],
                            is_error: false,
                            display: Some(ResultDisplay {
                                change: Some(Change::between(None, Some(&memory))),
                                ..ResultDisplay::default()
                            }),
                        },
                    ],
                    provider_fields: None,
                },
            ]
            .into_iter()
            .map(crate::conversation::RecordedMessage::unrecorded)
            .collect(),
            written_summary: "**Done:** the release.".into(),
            ..CompactionRecord::for_test()
        };
        let mut transcript = Transcript::new(settings());
        transcript.apply(FrontendEvent::Compacted {
            id: "c1".into(),
            record,
            generation: 1,
        });
        let mut viewport = Viewport::new();
        let backend = ratatui::backend::TestBackend::new(60, 80);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let mut draw = |viewport: &mut Viewport, transcript: &Transcript| {
            terminal
                .draw(|frame| viewport.draw(frame, frame.area(), transcript))
                .expect("a frame");
            let buffer = terminal.backend().buffer().clone();
            (0..80u16)
                .map(|row| {
                    let text = (0..60u16)
                        .map(|column| buffer[(column, row)].symbol().to_string())
                        .collect::<String>()
                        .trim_end()
                        .to_string();
                    let shaded = (0..60u16).all(|column| buffer[(column, row)].bg != Color::Reset);
                    (text, shaded)
                })
                .collect::<Vec<(String, bool)>>()
        };
        let text = |rows: &[(String, bool)]| -> Vec<String> {
            rows.iter()
                .map(|(text, _)| text.clone())
                .filter(|text| !text.is_empty())
                .collect()
        };
        // The words are centered on the 59 columns a row of the transcript takes, a column short of
        // the edge, starting at column 21, the icon to their left.
        let rule = |glyph: &str| {
            format!(
                "{glyph} {} →← Context compacted {}",
                "─".repeat(15),
                "─".repeat(20)
            )
        };
        let folded = draw(&mut viewport, &transcript);
        assert_eq!(text(&folded), vec![rule("▸")]);
        let at = |index| CursorStop {
            index,
            under_run: false,
        };
        assert_eq!(
            viewport.cursor_after(&transcript, at(0)),
            None,
            "its contents are hidden"
        );
        assert!(transcript.toggle(0));
        let open = draw(&mut viewport, &transcript);
        let shown = text(&open);
        let position = |wanted: &dyn Fn(&str) -> bool| shown.iter().position(|row| wanted(row));
        let call = position(&|row| row.starts_with("• memory_write notes")).expect("the call");
        assert_eq!(
            shown[..5],
            [
                rule("▾").as_str(),
                "• The agent compacted the conversation",
                "• No recent turns were kept",
                "• Thinking · Hmm.",
                "• memory_search release",
            ],
            "it opens to what set it going, then its steps as themselves rather than folded"
        );
        assert_eq!(call, 6, "{shown:?}");
        assert!(
            position(&|row| row.ends_with("+ fact 1")).is_some(),
            "{shown:?}"
        );
        assert!(
            position(&|row| row.ends_with("+ fact 30")).is_none(),
            "a collapsed call shows the opening of its diff: {shown:?}"
        );
        let summary = position(&|row| row == "• Summary").expect("the summary");
        assert!(summary > call, "{shown:?}");
        assert_eq!(
            shown.last().map(String::as_str),
            Some("  Done: the release.")
        );
        let last = open
            .iter()
            .rposition(|(text, _)| text == "  Done: the release.")
            .expect("the summary's text");
        assert!(
            open.iter().take(last + 1).all(|(_, shaded)| *shaded),
            "the open compaction is one shaded block: {open:?}"
        );
        assert!(
            open.iter().skip(last + 1).all(|(_, shaded)| !*shaded),
            "nothing past it is shaded"
        );
        assert_eq!(
            viewport.cursor_after(&transcript, at(0)),
            Some(at(1)),
            "the cursor stops at its steps"
        );
        let row = open
            .iter()
            .position(|(text, _)| text.starts_with("• memory_write"))
            .expect("the call's row");
        let index = viewport.item_at_row(row).expect("the call is an item");
        assert!(transcript.toggle(index));
        let opened_call = text(&draw(&mut viewport, &transcript));
        assert!(
            opened_call.iter().any(|row| row.ends_with("+ fact 30")),
            "an opened call shows its whole diff: {opened_call:?}"
        );
        assert!(transcript.toggle(0));
        assert_eq!(text(&draw(&mut viewport, &transcript)), vec![rule("▸")]);

        let mut bare = Transcript::new(settings());
        bare.apply(FrontendEvent::Compacted {
            id: "c2".into(),
            record: CompactionRecord {
                trigger: CompactionTrigger::Manual,
                ..CompactionRecord::for_test()
            },
            generation: 1,
        });
        assert!(
            bare.toggle(0),
            "a compaction that recorded nothing still opens to what set it going"
        );
        assert_eq!(text(&draw(&mut Viewport::new(), &bare)), vec![
            rule("▾"),
            "• Compacted the conversation on user request".to_string(),
        ]);
    }

    /// An item opened while the view follows the bottom keeps its row where it was, and what
    /// opens grows below it, rather than the view staying at the bottom and the row the user
    /// clicked leaving the screen.
    #[test]
    fn an_opened_item_keeps_its_row_where_it_was() {
        let mut transcript = Transcript::new(settings());
        for _ in 0..3 {
            transcript.push(said("Words."));
        }
        transcript.push(Item::Thinking {
            text: (1..=12).map(|n| format!("line {n}\n\n")).collect(),
            estimate: None,
            ended: true,
        });
        transcript.push(said("More words."));
        let mut viewport = Viewport::new();
        let rows = frame_rows(&mut viewport, &transcript, 6);
        let row = rows
            .iter()
            .position(|row| row.starts_with("• Thinking"))
            .expect("the thinking row");
        let index = viewport.item_at_row(row).expect("an item");
        assert!(transcript.toggle(index));
        viewport.keep_in_place(index);
        let rows = frame_rows(&mut viewport, &transcript, 6);
        assert_eq!(
            rows[row], "• Thinking",
            "the row stays where it was: {rows:?}"
        );
        assert!(
            rows.get(row + 1).is_some_and(|below| below == "  line 1"),
            "and the block opens under it: {rows:?}"
        );
    }

    /// A folded run opens under a click on its row, or the cursor's Enter on it, and folds again
    /// the same way; showing every call expanded opens every run, whatever was chosen for it.
    #[test]
    fn a_folded_run_opens_and_folds_again() {
        let mut transcript = Transcript::new(settings());
        transcript.push(thought());
        transcript.push(call_named("shell_execute", ToolState::Done {
            is_error: false,
        }));
        transcript.push(said("Done."));
        let mut viewport = Viewport::new();
        let rows = frame_rows(&mut viewport, &transcript, 12);
        let row = rows
            .iter()
            .position(|row| row == "▸ Thought once, ran a command")
            .expect("the run's row");
        let run = viewport.run_at_row(row).expect("a run is on its row");
        let run_row = CursorStop {
            index: 0,
            under_run: false,
        };
        assert_eq!(viewport.cursor_before(&transcript, None), Some(run_row));
        viewport.set_cursor(Some(run_row));
        assert_eq!(viewport.run_under_cursor(), Some(run));
        viewport.toggle_run(run);
        let rows = frame_rows(&mut viewport, &transcript, 12);
        let shown: Vec<&str> = rows
            .iter()
            .map(String::as_str)
            .filter(|row| !row.is_empty())
            .collect();
        assert_eq!(shown, vec![
            "▾ Thought once, ran a command",
            "• Thinking · Hmm.",
            "• shell_execute",
            "  └ done",
            "  Done.",
        ]);
        let first_step = CursorStop {
            index: 0,
            under_run: true,
        };
        assert_eq!(
            viewport.cursor_after(&transcript, run_row),
            Some(first_step),
            "an open run's first step is a stop of its own, under the run's row"
        );
        viewport.set_cursor(Some(first_step));
        assert_eq!(
            viewport.run_under_cursor(),
            None,
            "Enter there opens the step"
        );
        assert_eq!(
            viewport.cursor_after(&transcript, first_step),
            Some(CursorStop {
                index: 1,
                under_run: false
            }),
            "and the steps after it are stops of their own"
        );
        let lit = frame_rows_styled(&mut viewport, &transcript, 12);
        let reversed: Vec<&str> = lit
            .iter()
            .filter(|(_, reversed)| *reversed)
            .map(|(text, _)| text.as_str())
            .collect();
        assert_eq!(
            reversed,
            vec!["• Thinking · Hmm."],
            "the cursor is drawn on the step, inside the run's shade"
        );
        viewport.toggle_run(run);
        let rows = frame_rows(&mut viewport, &transcript, 12);
        assert!(
            !rows.iter().any(|row| row.starts_with("• shell_execute")),
            "folded again: {rows:?}"
        );
        transcript.set_tools_expanded(true);
        viewport.forget_run_choices();
        let rows = frame_rows(&mut viewport, &transcript, 12);
        assert!(
            rows.iter().any(|row| row.starts_with("• shell_execute")),
            "every call expanded opens the run: {rows:?}"
        );
    }

    /// The banner is Meka's mark beside three lines, two columns in from the edge under a blank
    /// row: the letter in cyan with its shadow in a deep blue, which lies under a glyph where the
    /// two share a cell and never under a space; then the name bold and white with the version
    /// and the build dimmed beside it, then what the agent can do and what it knows a step down
    /// in the palette's plain white. A release built from its tag shows the version alone.
    #[test]
    fn the_banner_is_the_mark_beside_three_lines() {
        let item = Item::Banner {
            version: "0.72.0".into(),
            revision: Some("a1b2c3d-dirty".into()),
            debug: true,
            mcp_servers: 2,
            skills: 1,
            memories: 48,
            instructions: 3,
        };
        let lines = lines_at(&entry(item, false), 80);
        assert_eq!(plain(&lines), vec![
            "",
            "   ██▖ ▗██▌  Meka v0.72.0 (a1b2c3d-dirty, debug)",
            "  ▐█▝▙▟▘█▌   2 MCP servers · 1 skill",
            "  █▌ ▀▘▐█▌   48 memories · 3 instructions",
        ]);
        assert!(
            lines
                .iter()
                .flat_map(|line| line.spans.iter())
                .all(|span| span.style.bg.is_none() || !span.content.contains(' ')),
            "no space has a background: {lines:?}"
        );
        let style_at = |line: &Line<'static>, column: usize| {
            let mut start = 0;
            line.spans.iter().find_map(|span| {
                let width = display_width(&span.content);
                let found = (column < start + width).then_some(span.style);
                start += width;
                found
            })
        };
        let face = Style::default().fg(Color::Cyan);
        assert_eq!(style_at(&lines[1], 3), Some(face), "the letter is cyan");
        assert_eq!(
            style_at(&lines[1], 10),
            Some(Style::default().fg(BANNER_SHADOW)),
            "a cell of shadow alone is drawn in the shadow's color"
        );
        assert_eq!(
            style_at(&lines[1], 5),
            Some(face.bg(BANNER_SHADOW)),
            "a cell of both is the letter over the shadow"
        );
        let spans = &lines[1].spans;
        let at = spans
            .iter()
            .position(|span| span.content == "Meka")
            .expect("the name");
        let name = &spans[at];
        assert_eq!(name.style.fg, Some(draw::WHITE));
        assert!(
            name.style.add_modifier.contains(Modifier::BOLD),
            "the name is bold: {name:?}"
        );
        let version = &spans[at + 1];
        assert_eq!(version.content.as_ref(), " v0.72.0 (a1b2c3d-dirty, debug)");
        assert_eq!(
            version.style.fg,
            Some(Color::DarkGray),
            "the version and the build are dimmed"
        );
        assert_eq!(
            lines[2].spans.last().map(|span| span.style.fg),
            Some(Some(Color::Gray)),
            "the facts are the palette's plain white, a step down from the name"
        );
        let release = lines_at(
            &entry(
                Item::Banner {
                    version: "0.72.0".into(),
                    revision: None,
                    debug: false,
                    mcp_servers: 0,
                    skills: 0,
                    memories: 1,
                    instructions: 0,
                },
                false,
            ),
            60,
        );
        assert_eq!(plain(&release)[1..], [
            "   ██▖ ▗██▌  Meka v0.72.0",
            "  ▐█▝▙▟▘█▌   0 MCP servers · 0 skills",
            "  █▌ ▀▘▐█▌   1 memory · 0 instructions",
        ]);
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
                ran_under: RanUnder::Level(crate::permission::Permission::Read),
            },
            false,
        );
        let lines = lines_at(&sent, 60);
        assert_eq!(lines[1].spans[1].content.as_ref(), format!("look {family}"));
    }

    /// A prompt that went to the inbox says what it is and whether the loop has read it, on the
    /// composer's block with a row of it above and below the words and the composer's bar down
    /// every row, in the color of what it ran under.
    #[test]
    fn a_steer_says_it_is_waiting_until_delivered() {
        let waiting = entry(
            Item::Prompt {
                text: "also check the logs".into(),
                class: Some(InboxClass::Steer),
                inbox_id: None,
                delivered: false,
                ran_under: RanUnder::Level(crate::permission::Permission::Read),
            },
            false,
        );
        let lines = lines_at(&waiting, 60);
        let rows = plain(&lines);
        assert_eq!(rows, vec![
            "▎",
            "▎ also check the logs",
            "▎ (steer, waiting)",
            "▎"
        ]);
        let read = color(crate::permission::Permission::Read.indicator_color());
        assert!(
            lines
                .iter()
                .all(|line| line.spans[0].content.starts_with(draw::COMPOSER_BAR)
                    && line.spans[0].style.fg == Some(read)),
            "the bar is the level's color on every row: {lines:?}"
        );
        let bar_of = |ran_under: RanUnder| {
            let sent = entry(
                Item::Prompt {
                    text: "ls".into(),
                    class: None,
                    inbox_id: None,
                    delivered: true,
                    ran_under,
                },
                false,
            );
            lines_at(&sent, 60)[1].spans[0].style.fg
        };
        assert_eq!(bar_of(RanUnder::Shell), Some(Color::Red));
        assert_eq!(bar_of(RanUnder::Unrecorded), Some(draw::DIM_ON_SHADE));
        assert_eq!(bar_of(RanUnder::Pending), Some(draw::DIM_ON_SHADE));
        assert!(
            lines.iter().all(|line| line.width() == 60
                && line
                    .spans
                    .iter()
                    .all(|span| span.style.bg == Some(COMPOSER_BACKGROUND))),
            "every row is the composer's block across the whole width: {lines:?}"
        );
        let (_, copies) = entry_rows(&waiting, 60, Path::new("/w"), PromptPadding::Full);
        assert_eq!(
            copies,
            vec![
                draw::RowCopy::Padding,
                draw::RowCopy::Text,
                draw::RowCopy::Padding,
                draw::RowCopy::Padding
            ],
            "the padding and the screen's note copy as nothing"
        );
        let words = &lines[1].spans[1];
        assert_eq!(words.content.as_ref(), "also check the logs");
        assert!(
            words.style.fg == Some(draw::WHITE)
                && !words.style.add_modifier.contains(Modifier::BOLD),
            "the words are white, not bold: {words:?}"
        );
    }

    /// Without padding a prompt is its words and the screen's note on them, still the composer's
    /// block across the width, and the note still copies as nothing.
    #[test]
    fn a_prompt_without_padding_is_its_words_and_its_note() {
        let waiting = entry(
            Item::Prompt {
                text: "also check the logs".into(),
                class: Some(InboxClass::Steer),
                inbox_id: None,
                delivered: false,
                ran_under: RanUnder::Level(crate::permission::Permission::Read),
            },
            false,
        );
        let (lines, copies) = entry_rows(&waiting, 60, Path::new("/w"), PromptPadding::None);
        assert_eq!(plain(&lines), vec![
            "▎ also check the logs",
            "▎ (steer, waiting)"
        ]);
        assert_eq!(copies, vec![draw::RowCopy::Text, draw::RowCopy::Padding]);
        assert!(
            lines.iter().all(|line| line.width() == 60
                && line
                    .spans
                    .iter()
                    .all(|span| span.style.bg == Some(COMPOSER_BACKGROUND))),
            "every row is the composer's block across the whole width: {lines:?}"
        );
    }

    /// A half row is the composer's gray in the half of each cell next to the words and the
    /// terminal's own background in the other, across the whole width, with no bar; it copies as
    /// nothing, and the words between keep their bar and their gray.
    #[test]
    fn a_half_padded_prompt_is_gray_only_in_the_half_next_to_its_words() {
        let sent = entry(
            Item::Prompt {
                text: "also check the logs".into(),
                class: None,
                inbox_id: None,
                delivered: true,
                ran_under: RanUnder::Level(crate::permission::Permission::Read),
            },
            false,
        );
        let (lines, copies) = entry_rows(&sent, 60, Path::new("/w"), PromptPadding::Half);
        assert_eq!(plain(&lines), vec![
            "▄".repeat(60),
            "▎ also check the logs".to_string(),
            "▀".repeat(60)
        ]);
        assert_eq!(copies, vec![
            draw::RowCopy::Padding,
            draw::RowCopy::Text,
            draw::RowCopy::Padding
        ]);
        for padding in [&lines[0], &lines[2]] {
            assert!(
                padding
                    .spans
                    .iter()
                    .all(|span| span.style.fg == Some(COMPOSER_BACKGROUND)
                        && span.style.bg == Some(Color::Reset)),
                "the gray is the glyph, the rest the terminal's: {padding:?}"
            );
        }
        assert!(
            lines[1].width() == 60
                && lines[1]
                    .spans
                    .iter()
                    .all(|span| span.style.bg == Some(COMPOSER_BACKGROUND)),
            "the words' row is the block across the whole width: {lines:?}"
        );
    }

    /// A selection lights a half padding row as it lights a whole one, so the light stays one band
    /// across a prompt rather than moving the gray to the half away from the words.
    #[test]
    fn a_selected_half_padding_row_lights_as_a_whole_one() {
        let sent = entry(
            Item::Prompt {
                text: "ls".into(),
                class: None,
                inbox_id: None,
                delivered: true,
                ran_under: RanUnder::Level(crate::permission::Permission::Read),
            },
            false,
        );
        let lit = |padding| {
            let (lines, _) = entry_rows(&sent, 8, Path::new("/w"), padding);
            let mut buffer = Buffer::empty(Rect::new(0, 0, 8, 3));
            for (y, line) in (0..).zip(&lines) {
                buffer.set_line(0, y, line, 8);
            }
            for cell in &mut buffer.content {
                light_selected(cell);
            }
            buffer
        };
        let (half, full) = (lit(PromptPadding::Half), lit(PromptPadding::Full));
        // Past the bar, which a half row does not carry.
        for (x, y) in (TEXT_EDGE as u16..8).flat_map(|x| [(x, 0), (x, 2)]) {
            assert_eq!(half[(x, y)], full[(x, y)], "the cell at {x}, {y}");
        }
        assert_eq!(
            half[(3, 1)],
            full[(3, 1)],
            "the words' row lights as before"
        );
    }

    /// A prompt's words stop a column short of the right edge, as the space after the bar keeps
    /// one at the left, while its gray still runs to the edge.
    #[test]
    fn a_prompt_s_words_keep_a_column_clear_at_the_right_edge() {
        let sent = entry(
            Item::Prompt {
                text: "x".repeat(58),
                class: None,
                inbox_id: None,
                delivered: true,
                ran_under: RanUnder::Level(crate::permission::Permission::Read),
            },
            false,
        );
        let lines = lines_at(&sent, 60);
        assert_eq!(plain(&lines), vec![
            "▎".to_string(),
            format!("▎ {}", "x".repeat(57)),
            "▎ x".to_string(),
            "▎".to_string()
        ]);
        assert!(
            lines.iter().all(|line| line.width() == 60),
            "every row is the block across the whole width: {lines:?}"
        );
    }

    /// A list row too long for the width is cut a column short of the right edge, the picked
    /// row's bar still reaching it; a question's words stop there too.
    #[test]
    fn a_list_row_and_a_question_stop_a_column_short_of_the_edge() {
        let rows = vec![ListRow {
            value: "/x".to_string(),
            description: Some("y".repeat(40)),
        }];
        let lines = list_lines(&rows, 0, 30, ListLayout::Column);
        assert_eq!(plain(&lines), vec![format!("  /x  {}…", "y".repeat(22))]);
        assert_eq!(lines[0].width(), 30, "the bar runs to the edge: {lines:?}");

        let backend = ratatui::backend::TestBackend::new(30, 3);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        terminal
            .draw(|frame| {
                draw_question(
                    frame,
                    frame.area(),
                    &[Line::from(format!("  {}", "b".repeat(40)))],
                    "name: ",
                    &"a".repeat(40),
                );
            })
            .expect("a frame");
        let buffer = terminal.backend().buffer();
        for y in 0..2 {
            assert_eq!(buffer[(28, y)].symbol(), if y == 0 { "b" } else { "a" });
            assert_eq!(
                buffer[(29, y)].symbol(),
                " ",
                "row {y} keeps the last column clear"
            );
            assert_eq!(buffer[(29, y)].style().bg, Some(COMPOSER_BACKGROUND));
        }
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
            Span::styled("▎ ", Style::default().fg(draw::DIM_ON_SHADE)),
            Span::styled("words", Style::default().fg(draw::WHITE)),
        ]);
        let backend = ratatui::backend::TestBackend::new(20, 1);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        for (hovered, expected) in [(false, draw::DIM_ON_SHADE), (true, draw::WHITE)] {
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
            .draw(|frame| pill = draw_back_to_bottom(frame, frame.area(), 0, false))
            .expect("a frame");
        let label_width = display_width(BACK_TO_BOTTOM) as u16 + 2;
        assert_eq!(pill.width, label_width);
        assert_eq!(pill.x, (60 - label_width) / 2);
        let cell = &terminal.backend().buffer()[(pill.x + 1, 0)];
        assert_eq!(cell.symbol(), "↓");
        assert_eq!(cell.bg, COMPOSER_BACKGROUND);
        terminal
            .draw(|frame| pill = draw_back_to_bottom(frame, frame.area(), 40, false))
            .expect("a frame");
        assert_eq!(pill.x, 60 - label_width, "clear of the working row's words");
        terminal
            .draw(|frame| pill = draw_back_to_bottom(frame, frame.area(), 10, false))
            .expect("a frame");
        assert_eq!(
            pill.x,
            (60 - label_width) / 2,
            "centered where the words end early"
        );
    }

    /// Under the pointer, the pill swaps its gray and its words, and the pointer is on it exactly
    /// over the cells it covers.
    #[test]
    fn the_pill_lights_under_the_pointer() {
        let backend = ratatui::backend::TestBackend::new(60, 1);
        let mut terminal = ratatui::Terminal::new(backend).expect("a test terminal");
        let mut pill = Rect::default();
        terminal
            .draw(|frame| pill = draw_back_to_bottom(frame, frame.area(), 0, true))
            .expect("a frame");
        let cell = &terminal.backend().buffer()[(pill.x + 1, 0)];
        assert_eq!((cell.fg, cell.bg), (COMPOSER_BACKGROUND, draw::WHITE));
        assert!(is_on_pill(Some(pill), pill.x, 0));
        assert!(is_on_pill(Some(pill), pill.x + pill.width - 1, 0));
        assert!(!is_on_pill(Some(pill), pill.x + pill.width, 0));
        assert!(!is_on_pill(Some(pill), pill.x.saturating_sub(1), 0));
        assert!(!is_on_pill(Some(pill), pill.x, 1));
        assert!(!is_on_pill(None, pill.x, 0));
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
            unparsed: None,
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
            display: None,
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

    /// The status row has what the next prompt goes under and where the session works against
    /// the left, each after a dot, the level in its color, the profile in white and the model
    /// dimmed, then the path and the project, and its live readings against the right edge, one
    /// column clear of it; a hint takes the right side while it stands, and where the row is narrow
    /// the left side gives way from its end, the level last.
    #[test]
    fn the_status_row_has_the_next_send_and_the_place_left_and_the_readings_right() {
        let text = |line: &Line<'static>| -> String {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect()
        };
        let facts = |work: RunningWork, hint: Option<&'static str>| StatusFacts {
            shown: &[
                crate::config::StatusFact::Profile,
                crate::config::StatusFact::Model,
                crate::config::StatusFact::Path,
                crate::config::StatusFact::Project,
                crate::config::StatusFact::Context,
                crate::config::StatusFact::Tasks,
                crate::config::StatusFact::Subagents,
            ],
            permission: crate::permission::Permission::Workspace,
            path: Some("~/projects".into()),
            project: Some("webdev".into()),
            profile: Some("work".into()),
            model: Some("claude-opus-4-8".into()),
            context: Some("34%".into()),
            work,
            hint,
            shell_mode: false,
        };
        let idle = status_line(&facts(RunningWork::default(), None), 80);
        assert_eq!(
            text(&idle),
            format!(
                "  workspace · work · claude-opus-4-8 · ~/projects · webdev{}34%",
                " ".repeat(18)
            )
        );
        let level = color(crate::permission::Permission::Workspace.indicator_color());
        let style_of = |word: &str| {
            idle.spans
                .iter()
                .find(|span| span.content == word)
                .map(|span| span.style.fg)
        };
        assert_eq!(style_of("workspace"), Some(Some(level)));
        assert_eq!(
            style_of("work"),
            Some(None),
            "the profile is in the terminal's own color"
        );
        assert_eq!(
            style_of("claude-opus-4-8"),
            Some(Some(Color::DarkGray)),
            "what is dimmed is in the terminal's dark gray, with no shade to swallow it"
        );
        let running = RunningWork {
            tasks: 2,
            subagents: 1,
        };
        let busy = text(&status_line(&facts(running, None), 120));
        assert!(
            busy.ends_with("34% · 2 background tasks · 1 sub-agent") && display_width(&busy) == 119,
            "the work running beside the turn is counted after the gauge: {busy:?}"
        );
        let hinted = text(&status_line(&facts(running, Some("Copied 2 rows")), 120));
        assert!(
            hinted.starts_with("  workspace · work ")
                && hinted.ends_with(" Copied 2 rows")
                && display_width(&hinted) == 119,
            "a hint takes the right side: {hinted:?}"
        );
        let at = |width| text(&status_line(&facts(RunningWork::default(), None), width));
        assert_eq!(
            at(50),
            format!(
                "  workspace · work · claude-opus-4-8 · …ects{}34%",
                " ".repeat(2)
            ),
            "the fact that no longer fits is cut, a path from its start, and the rest give way"
        );
        assert_eq!(
            at(30),
            format!("  workspace · work{}34%", " ".repeat(8)),
            "a fact with too little room to read is dropped with its dot"
        );
        assert_eq!(
            at(16),
            format!("  workspace{}", " ".repeat(4)),
            "the level stays whole and the readings give way to it, none cut past reading"
        );
        let running = |width| {
            text(&status_line(
                &facts(
                    RunningWork {
                        tasks: 2,
                        subagents: 1,
                    },
                    None,
                ),
                width,
            ))
        };
        for width in 12..60 {
            let row = running(width);
            assert!(
                !row.trim_end().ends_with('·')
                    && !row.contains("·…")
                    && !row.contains("· …")
                    && display_width(&row) < width.max(12),
                "no dot left behind and no bare ellipsis at {width}: {row:?}"
            );
            assert!(row.starts_with("  workspace"), "{row:?}");
        }
        let gauged = |width| {
            text(&status_line(
                &StatusFacts {
                    context: Some("0 / 200.0k (0%)".into()),
                    ..facts(RunningWork::default(), None)
                },
                width,
            ))
        };
        for width in 12..40 {
            let row = gauged(width);
            assert!(
                row.contains("0 / 200.0k (0%)") || !row.contains("0 /"),
                "a reading is whole or not there, never cut to another number: {row:?}"
            );
        }
        let placed = |path: &str, width| {
            text(&status_line(
                &StatusFacts {
                    shown: &[
                        crate::config::StatusFact::Profile,
                        crate::config::StatusFact::Path,
                    ],
                    path: Some(path.into()),
                    ..facts(RunningWork::default(), None)
                },
                width,
            ))
        };
        assert_eq!(
            placed("~/projects", 32),
            "  workspace · work · ~/projects",
            "with nothing on the right, the facts take the gap the readings would keep"
        );
        let hearts = "\u{2764}\u{fe0f}".repeat(10);
        for width in 12..60 {
            let row = placed(&format!("~/{hearts}"), width);
            assert!(
                display_width(&row) < width,
                "a fact cut by the width it takes beside what is before it: {width} {row:?}"
            );
            assert!(
                !row.contains('\u{2764}') || row.trim_end().ends_with('\u{fe0f}'),
                "a path cut from its start keeps its end: {width} {row:?}"
            );
            assert!(
                !row.contains("…\u{fe0f}"),
                "a cut never parts an emoji from its selector: {width} {row:?}"
            );
            let row = text(&status_line(
                &StatusFacts {
                    profile: Some(hearts.clone()),
                    ..facts(RunningWork::default(), None)
                },
                width,
            ));
            assert!(display_width(&row) < width, "{width} {row:?}");
        }
    }

    /// The gauge shows from the first frame at zero occupancy, and only an unknown window hides it.
    #[test]
    fn the_context_gauge_shows_from_the_first_frame() {
        assert_eq!(
            context_gauge_text(0, 1_000_000).as_deref(),
            Some("0 / 1.0M (0%)")
        );
        assert_eq!(
            context_gauge_text(128_400, 1_000_000).as_deref(),
            Some("128.4k / 1.0M (13%)")
        );
        assert_eq!(context_gauge_text(0, 0), None);
        assert_eq!(context_gauge_text(5, 0), None);
    }

    /// The status row shows the facts the config lists, the level always; in the shell mode, what
    /// Enter does instead, in red, in place of the level and the place, as the bar down the
    /// composer is red.
    #[test]
    fn the_status_row_follows_the_list_and_says_the_shell_mode() {
        use crate::config::StatusFact;
        let text = |line: &Line<'static>| -> String {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
                .trim_end()
                .to_string()
        };
        let facts = |shown: &'static [StatusFact], shell_mode: bool| StatusFacts {
            shown,
            permission: crate::permission::Permission::Workspace,
            path: Some("~/x".into()),
            project: None,
            profile: Some("work".into()),
            model: Some("claude-opus-4-8".into()),
            context: None,
            work: RunningWork::default(),
            hint: None,
            shell_mode,
        };
        assert_eq!(
            text(&status_line(&facts(&StatusFact::DEFAULT_ROW, false), 60)),
            "  workspace · work · ~/x",
            "the default leaves the model out, and a session on no project shows none"
        );
        assert_eq!(
            text(&status_line(
                &facts(
                    &[StatusFact::Path, StatusFact::Model, StatusFact::Profile],
                    false
                ),
                60
            )),
            "  workspace · ~/x · claude-opus-4-8 · work",
            "the facts after the level follow the list's order"
        );
        assert_eq!(text(&status_line(&facts(&[], false), 60)), "  workspace");
        let level = color(crate::permission::Permission::Workspace.indicator_color());
        assert_eq!(composer_bar_color(&facts(&[], false)), level);
        let shell = facts(&StatusFact::ALL, true);
        let line = status_line(&shell, 60);
        assert_eq!(text(&line), format!("  {SHELL_MODE_LABEL}"));
        assert!(
            line.spans
                .iter()
                .filter(|span| !span.content.trim().is_empty())
                .all(|span| span.style.fg == Some(Color::Red)),
            "{line:?}"
        );
        assert_eq!(composer_bar_color(&shell), Color::Red);
    }

    /// Items laid out ahead at a width are not laid out again by a frame drawn at it, so a frame
    /// after a resize does no more than place the rows its layout already made.
    #[test]
    fn a_frame_at_the_width_laid_out_ahead_lays_nothing_out() {
        let mut transcript = Transcript::new(settings());
        transcript.push(Item::note("outcome"));
        transcript.push(Item::Text {
            text: "An answer long enough to wrap at the width it is drawn at.".into(),
            streaming: false,
        });
        let mut viewport = Viewport::new();
        let rows_made = |viewport: &Viewport| -> Vec<*const ratatui::text::Line<'static>> {
            let mut made: Vec<_> = viewport
                .cache
                .values()
                .map(|cached| cached.lines.as_ptr())
                .collect();
            made.sort();
            made
        };
        viewport.lay_out_all(&transcript, 30);
        let ahead = rows_made(&viewport);
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(30, 8)).expect("terminal");
        terminal
            .draw(|frame| viewport.draw(frame, frame.area(), &transcript))
            .expect("a frame");
        assert_eq!(rows_made(&viewport), ahead);
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
                    created_at: String::new(),
                    said: None,
                    id: first + offset as i64,
                    event: Event::Append(Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::Text {
                            text: (*text).to_string(),
                            provider_fields: None,
                        }],
                        provider_fields: None,
                    }),
                    reply: None,
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
            ran_under: RanUnder::Level(crate::permission::Permission::Read),
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
        assert_eq!(plain(&[line])[0], "▎ second");
        viewport.scroll_to(start.saturating_sub(1));
        draw(&mut viewport);
        let (line, start) = viewport
            .sticky_prompt()
            .expect("with the second at the top, the first sticks");
        assert_eq!(plain(&[line])[0], "▎ first");
        assert!(!viewport.is_following());
        viewport.scroll_to(start.saturating_sub(1));
        draw(&mut viewport);
        assert!(
            viewport.sticky_prompt().is_none(),
            "the first prompt is on screen, nothing is above it"
        );
    }
}
