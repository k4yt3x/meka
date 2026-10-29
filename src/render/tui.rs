//! The transcript's text as the terminal UI draws it: markdown, code fences, tool output and
//! plain text, laid out at a width as ratatui lines, and the rows every item is dressed in.
//!
//! Markdown is laid out here from the blocks [`super::markdown`] reads: prose wraps at spaces,
//! under the marker of every list item and the bar of every quote it sits in, and a table's cells
//! wrap in columns fitted to the width; a code block is highlighted by `syntect`. The rows are
//! lines the UI owns rather than bytes on a stream, which is what lets it draw them again at
//! another width.
//!
//! Every item on the screen puts a one-glyph gutter in column 0 and its text at [`TEXT_EDGE`]:
//! `>` for a prompt, `•` for an act of the agent or the host, `!` for a warning, `■` for an
//! error, nothing for the answer. The palette is the terminal's foreground, gray, red and yellow.
//! No glyph has an emoji presentation: a terminal that draws U+26A0 as a picture draws it the
//! size of two cells in a color of its own, and the gutter is meant to be a mark, not a picture.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::*;

/// The column every item's text starts in, behind its one-glyph gutter. The composer and the
/// prompt block use the same edge, so the whole screen shares one left margin.
pub(crate) const TEXT_EDGE: usize = 2;
/// The gutter glyph of what the user typed: on the composer and on the prompt it becomes. The
/// angle quotation mark `›` would be smaller, but fonts set it low on the line.
pub(crate) const PROMPT_GLYPH: &str = ">";
/// The glyph's color: a mid gray from the fixed palette, lighter than the chrome's gray, which
/// sinks into the composer's block, and dimmer than the words beside it.
pub(crate) const PROMPT_GLYPH_COLOR: Color = Color::Indexed(245);
/// The gutter glyph of the composer in its shell mode, in red: what is typed runs as a command.
pub(crate) const SHELL_GLYPH: &str = "!";
/// The gutter glyph of an act: a tool call, a thinking block, a notice, a stage direction.
pub(crate) const ACT_GLYPH: &str = "•";
/// The gutter glyph of a warning.
pub(crate) const WARNING_GLYPH: &str = "!";
/// The gutter glyph of an error.
pub(crate) const ERROR_GLYPH: &str = "■";
/// The lead of the first row under a tool call: its result, its progress, its sub-agent's record.
const CONNECTOR: &str = "└ ";
/// Rows a collapsed tool call shows of its result: the opening of it, which is where a listing's
/// header, a file's first lines and an error's first word sit.
pub(crate) const PREVIEW_ROWS: usize = 3;
/// Rows a collapsed call shows of the change it made to a file. A change is the result rather
/// than a glance at one, so it gets the room a small edit takes whole; a rewrite of a file is
/// still counted past this.
pub(crate) const DIFF_PREVIEW_ROWS: usize = 20;

/// The gray every dimmed row is drawn in.
pub(crate) fn gray() -> Style {
    Style::default().fg(Color::DarkGray)
}

/// The one accent on the screen: what inline code is drawn in, and the cursor of a picker. The
/// terminal's own cyan, so it follows the theme the terminal has rather than one meka ships.
pub(crate) const ACCENT: Color = Color::Cyan;
/// Full white, for what has to stand out of the gray blocks: the words of a prompt, the picked
/// choice, the back-to-bottom pill. True color rather than the palette's white, because most
/// terminal themes paint that slot the same shade as the default foreground, and a prompt in it
/// stands out of nothing.
pub(crate) const WHITE: Color = Color::Rgb(255, 255, 255);

/// Markdown laid out at `width` in the answer's look.
pub(crate) fn markdown_lines(markdown: &str, width: usize) -> Vec<Line<'static>> {
    lay_out(markdown, width, &ANSWER)
}

/// Markdown laid out at `width` in the reasoning's look.
pub(crate) fn thinking_lines(markdown: &str, width: usize) -> Vec<Line<'static>> {
    lay_out(markdown, width, &REASONING)
}

/// An answer still streaming as lines: what has settled laid out as markdown, and the line still
/// arriving under it laid out on its own, so a token shows the moment it arrives and nothing the
/// line says yet reaches back into the rows above: `2` behind an item is not yet the next item's
/// number, nor a word of the item, and `---` under a paragraph does not make it a heading. A line
/// inside a code block the settled text has not closed is laid out with it, as the code row it
/// is; a line that continues a block above it, as an item's second line does, settles into place
/// when it ends. `reasoning` picks the reasoning's look.
pub(crate) fn streaming_lines(text: &str, width: usize, reasoning: bool) -> Vec<Line<'static>> {
    let look = if reasoning { &REASONING } else { &ANSWER };
    let (settled, open) = match text.rfind('\n') {
        Some(cut) => text.split_at(cut + 1),
        None => ("", text),
    };
    if open.is_empty() || ends_inside_a_fence(settled) {
        return lay_out(text, width, look);
    }
    let mut lines = lay_out(settled, width, look);
    // A blank line ends the block above, so the line arriving starts one of its own.
    if settled.ends_with("\n\n") && !lines.is_empty() {
        lines.push(Line::default());
    }
    lines.extend(lay_out(open, width, look));
    lines
}

/// The narrowest text a list item or a quote indents its blocks to. Narrower, an item's marker
/// takes a row of its own, a quote drops its bar, and the blocks keep the width: an indent that
/// leaves a word or two per row no longer reads as nesting.
const MIN_NESTED_WIDTH: usize = 10;
/// The columns between two cells of a table, and under them in its rule.
const TABLE_GAP: usize = 2;
const BULLET: char = '•';
const QUOTE_BAR: char = '▐';
const RULE: char = '―';
const TABLE_RULE: char = '─';

/// The styles a document is drawn in.
struct Look {
    /// Prose and table cells.
    text: Style,
    /// A heading at each level, from `#` down.
    headings: [Style; 6],
    /// What each emphasis lays over the style under it, in this order, so code in struck text
    /// keeps the code's color.
    italic: Style,
    strikeout: Style,
    bold: Style,
    code: Style,
    /// A code block's rows when they are not highlighted for their language.
    code_block: Style,
    /// Bullets, rules and a table's rule.
    structure: Style,
    quote_bar: Style,
    /// Whether a code block is highlighted for its language rather than drawn in `code_block`.
    highlight: bool,
}

/// The answer's look: emphasis by weight rather than hue, so a bold word is bold and not the red
/// of an error; code in the one accent; the structure in the gray the rest of the chrome is.
/// Every color is the terminal's own palette, which is what keeps the answer legible on a light
/// theme as on a dark one.
const ANSWER: Look = Look {
    text: Style::new(),
    headings: [Style::new().add_modifier(Modifier::BOLD); 6],
    italic: Style::new().add_modifier(Modifier::ITALIC),
    strikeout: Style::new()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::CROSSED_OUT),
    bold: Style::new().add_modifier(Modifier::BOLD),
    code: Style::new().fg(ACCENT),
    code_block: Style::new(),
    structure: Style::new().fg(Color::DarkGray),
    quote_bar: Style::new()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::BOLD),
    highlight: true,
};

/// The gray of the reasoning's look, and of its underlined headings.
const REASONING_HEADING: Style = Style::new()
    .fg(Color::DarkGray)
    .add_modifier(Modifier::UNDERLINED);

/// The reasoning's look: one gray throughout, code included, so it reads as a footnote beside the
/// answer and is told apart from it without being read. Emphasis survives as an attribute alone,
/// and a heading is underlined, the first bold as well.
const REASONING: Look = Look {
    text: Style::new().fg(Color::DarkGray),
    headings: [
        REASONING_HEADING.add_modifier(Modifier::BOLD),
        REASONING_HEADING,
        REASONING_HEADING,
        REASONING_HEADING,
        REASONING_HEADING,
        REASONING_HEADING,
    ],
    italic: Style::new().add_modifier(Modifier::ITALIC),
    strikeout: Style::new().add_modifier(Modifier::CROSSED_OUT),
    bold: Style::new().add_modifier(Modifier::BOLD),
    code: Style::new(),
    code_block: Style::new().fg(Color::DarkGray),
    structure: Style::new().fg(Color::DarkGray),
    quote_bar: Style::new()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::BOLD),
    highlight: false,
};

fn lay_out(markdown: &str, width: usize, look: &Look) -> Vec<Line<'static>> {
    let width = width.max(MIN_OUTPUT_WIDTH);
    let markdown = crate::text::expand_tabs(markdown);
    // The blank a table owes the line after it: GFM reads a non-blank line after a table as one
    // more row.
    let markdown = normalize_spacing(&markdown);
    let mut lines = look.blocks(&markdown::parse(&markdown), width, false);
    // A document ends where its last row does; the blank that separates it from the next block is
    // the transcript's to draw.
    while lines.last().is_some_and(is_blank) {
        lines.pop();
    }
    lines
}

impl Look {
    /// `blocks` at `width`, one blank row between two of them unless `tight`, as the blocks of an
    /// item in a tight list are.
    fn blocks(&self, blocks: &[markdown::Block], width: usize, tight: bool) -> Vec<Line<'static>> {
        let mut rows = Vec::new();
        for block in blocks {
            let block_rows = self.block(block, width);
            if block_rows.is_empty() {
                continue;
            }
            if !tight && !rows.is_empty() {
                rows.push(Line::default());
            }
            rows.extend(block_rows);
        }
        rows
    }

    fn block(&self, block: &markdown::Block, width: usize) -> Vec<Line<'static>> {
        match block {
            markdown::Block::Paragraph(runs) => self.prose(runs, self.text, width),
            markdown::Block::Heading { level, runs } => {
                let heading = self
                    .headings
                    .get(usize::from(*level).saturating_sub(1))
                    .copied()
                    .unwrap_or(self.text);
                self.prose(runs, heading, width)
            }
            markdown::Block::Code { language, rows } => self.code(rows, language.as_deref(), width),
            markdown::Block::List {
                start,
                loose,
                items,
            } => self.list(*start, *loose, items, width),
            markdown::Block::Quote(blocks) => self.quote(blocks, width),
            markdown::Block::Table(table) => self.table(table, width),
            markdown::Block::Rule => vec![Line::from(Span::styled(
                RULE.to_string().repeat(width),
                self.structure,
            ))],
        }
    }

    /// The style of `run` over `base`.
    fn run_style(&self, base: Style, run: &markdown::OwnedCompound) -> Style {
        let mut style = base;
        for (applies, emphasis) in [
            (run.italic, self.italic),
            (run.strikeout, self.strikeout),
            (run.bold, self.bold),
            (run.code, self.code),
        ] {
            if applies {
                style = style.patch(emphasis);
            }
        }
        style
    }

    /// The characters of `runs`, each in its run's style over `base`.
    fn characters(&self, runs: &[markdown::OwnedCompound], base: Style) -> Vec<(char, Style)> {
        runs.iter()
            .flat_map(|run| {
                let style = self.run_style(base, run);
                run.text.chars().map(move |character| (character, style))
            })
            .collect()
    }

    /// Runs wrapped at `width`, at a space and nowhere else a word goes on: a change of style
    /// inside a word, or the `(` before a code span, is no place to break.
    fn prose(
        &self,
        runs: &[markdown::OwnedCompound],
        base: Style,
        width: usize,
    ) -> Vec<Line<'static>> {
        self.characters(runs, base)
            .split(|(character, _)| *character == '\n')
            .flat_map(|row| fold_characters(row, width, Breaks::Words))
            .collect()
    }

    fn code(&self, rows: &[String], language: Option<&str>, width: usize) -> Vec<Line<'static>> {
        let lines = if self.highlight {
            code_lines(rows, language)
        } else {
            rows.iter()
                .map(|row| Line::from(Span::styled(row.clone(), self.code_block)))
                .collect()
        };
        // Folded at the edge rather than clipped there, as the terminal folds a one-shot's output:
        // at a column, never at a space, because a code row's spaces are content and its
        // indentation is where a break at a space would land.
        lines
            .iter()
            .flat_map(|row| wrap_line_at_columns(row, width))
            .collect()
    }

    /// A list's items, each behind its marker with its blocks hanging at the marker's width: a
    /// bullet, or a number right-aligned to the widest in the list, as a browser draws one.
    fn list(
        &self,
        start: Option<u64>,
        loose: bool,
        items: &[Vec<markdown::Block>],
        width: usize,
    ) -> Vec<Line<'static>> {
        let markers: Vec<Span<'static>> = match start {
            Some(first) => {
                let number = |index: usize| first.saturating_add(index as u64);
                let widest = format!("{}.", number(items.len().saturating_sub(1))).len();
                (0..items.len())
                    .map(|index| {
                        Span::styled(
                            format!("{:>widest$} ", format!("{}.", number(index))),
                            self.text,
                        )
                    })
                    .collect()
            }
            None => vec![Span::styled(format!("{BULLET} "), self.structure); items.len()],
        };
        let mut rows = Vec::new();
        for (item, marker) in items.iter().zip(markers) {
            if loose && !rows.is_empty() {
                rows.push(Line::default());
            }
            let indent = display_width(&marker.content);
            if width < indent + MIN_NESTED_WIDTH {
                rows.push(Line::from(marker));
                rows.extend(self.blocks(item, width, !loose));
                continue;
            }
            let item_rows = self.blocks(item, width - indent, !loose);
            if item_rows.is_empty() {
                rows.push(Line::from(marker));
                continue;
            }
            let pad = Span::raw(" ".repeat(indent));
            for (index, mut row) in item_rows.into_iter().enumerate() {
                if index == 0 {
                    row.spans.insert(0, marker.clone());
                } else if !is_blank(&row) {
                    row.spans.insert(0, pad.clone());
                }
                rows.push(row);
            }
        }
        rows
    }

    /// A table, its columns fitted to the width: as wide as their widest cells when those fit,
    /// else each as wide as its longest word where those fit, the rest shared by how much more
    /// each column holds, so an identifier stays whole while the prose beside it wraps. A cell
    /// wraps in its column as prose does, and a header cell is centered unless its column says
    /// otherwise, as a browser draws one. Columns too many for the width are drawn a row at a
    /// time.
    fn table(&self, table: &markdown::Table, width: usize) -> Vec<Line<'static>> {
        let columns = table.alignments.len();
        let rows: Vec<(bool, &Vec<Vec<markdown::OwnedCompound>>)> = table
            .head
            .iter()
            .map(|row| (true, row))
            .chain(table.rows.iter().map(|row| (false, row)))
            .collect();
        let gaps = TABLE_GAP * columns.saturating_sub(1);
        if columns == 0 || width < gaps + columns {
            let mut lines = Vec::new();
            for (_, row) in &rows {
                let mut runs = Vec::new();
                for (index, cell) in row.iter().enumerate() {
                    if index > 0 {
                        runs.push(markdown::OwnedCompound {
                            text: " · ".to_string(),
                            ..markdown::OwnedCompound::default()
                        });
                    }
                    runs.extend(cell.iter().cloned());
                }
                lines.extend(self.prose(&runs, self.text, width));
            }
            return lines;
        }
        let cells: Vec<Vec<Vec<(char, Style)>>> = rows
            .iter()
            .map(|(_, row)| {
                (0..columns)
                    .map(|column| {
                        row.get(column)
                            .map(|cell| self.characters(cell, self.text))
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .collect();
        let widest = |measure: fn(&[(char, Style)]) -> usize| -> Vec<usize> {
            (0..columns)
                .map(|column| {
                    cells
                        .iter()
                        .filter_map(|row| row.get(column))
                        .map(|cell| measure(cell))
                        .max()
                        .unwrap_or(0)
                        .max(1)
                })
                .collect()
        };
        let widths = fit_columns(
            &widest(characters_width),
            &widest(longest_unbreakable),
            width - gaps,
        );
        let gap = Span::raw(" ".repeat(TABLE_GAP));
        let mut lines = Vec::new();
        for ((head, _), row) in rows.iter().zip(&cells) {
            let wrapped: Vec<Vec<Line<'static>>> = row
                .iter()
                .zip(&widths)
                .map(|(cell, &column_width)| {
                    if cell.is_empty() {
                        Vec::new()
                    } else {
                        fold_characters(cell, column_width, Breaks::Words)
                    }
                })
                .collect();
            let height = wrapped.iter().map(Vec::len).max().unwrap_or(0).max(1);
            for index in 0..height {
                let mut spans = Vec::new();
                for (column, (cell, &column_width)) in wrapped.iter().zip(&widths).enumerate() {
                    if column > 0 {
                        spans.push(gap.clone());
                    }
                    let alignment = match table.alignments.get(column) {
                        Some(pulldown_cmark::Alignment::None) | None if *head => {
                            pulldown_cmark::Alignment::Center
                        }
                        Some(alignment) => *alignment,
                        None => pulldown_cmark::Alignment::None,
                    };
                    let row = cell.get(index).cloned().unwrap_or_default();
                    spans.extend(aligned(row, column_width, alignment).spans);
                }
                lines.push(Line::from(spans));
            }
            if *head {
                let rule: Vec<String> = widths
                    .iter()
                    .map(|&column_width| TABLE_RULE.to_string().repeat(column_width))
                    .collect();
                lines.push(Line::from(Span::styled(
                    rule.join(&" ".repeat(TABLE_GAP)),
                    self.structure,
                )));
            }
        }
        lines
    }

    /// A quote's blocks behind its bar on every row, the blank ones between its blocks included.
    fn quote(&self, blocks: &[markdown::Block], width: usize) -> Vec<Line<'static>> {
        let bar = Span::styled(QUOTE_BAR.to_string(), self.quote_bar);
        let indent = display_width(&bar.content) + 1;
        if width < indent + MIN_NESTED_WIDTH {
            return self.blocks(blocks, width, false);
        }
        self.blocks(blocks, width - indent, false)
            .into_iter()
            .map(|mut row| {
                if is_blank(&row) {
                    return Line::from(bar.clone());
                }
                row.spans.insert(0, Span::raw(" "));
                row.spans.insert(0, bar.clone());
                row
            })
            .collect()
    }
}

/// Each column's width, `available` shared among them: the widest cells when they fit; else each
/// column's longest word first and what is left by how much more each column holds; else, when
/// even the longest words do not fit, those cut down widest first.
fn fit_columns(natural: &[usize], shortest: &[usize], available: usize) -> Vec<usize> {
    if natural.iter().sum::<usize>() <= available {
        return natural.to_vec();
    }
    let floor: usize = shortest.iter().sum();
    if floor > available {
        return capped(shortest, available);
    }
    let wants: Vec<usize> = natural
        .iter()
        .zip(shortest)
        .map(|(natural, shortest)| natural.saturating_sub(*shortest))
        .collect();
    shortest
        .iter()
        .zip(apportion_extra(available - floor, &wants))
        .map(|(shortest, extra)| shortest + extra)
        .collect()
}

/// `widths` cut to fit `available` widest first: each to the largest width at which they all fit,
/// a narrower one keeping its own, so a column whose longest word is too wide gives up the excess
/// and the others keep their words whole. The columns the cut leaves over go back to the columns
/// cut, one each.
fn capped(widths: &[usize], available: usize) -> Vec<usize> {
    let total = |cap: usize| widths.iter().map(|width| (*width).min(cap)).sum::<usize>();
    let mut cap = widths.iter().copied().max().unwrap_or(0);
    while cap > 1 && total(cap) > available {
        cap -= 1;
    }
    let mut left = available.saturating_sub(total(cap));
    widths
        .iter()
        .map(|&width| {
            let mut fitted = width.min(cap);
            if fitted < width && left > 0 {
                fitted += 1;
                left -= 1;
            }
            fitted
        })
        .collect()
}

/// `total` shared in proportion to `weights`, nothing to a weight of zero, the columns rounding
/// leaves over going to the largest remainders.
fn apportion_extra(total: usize, weights: &[usize]) -> Vec<usize> {
    let sum: usize = weights.iter().sum();
    if sum == 0 {
        return vec![0; weights.len()];
    }
    let mut shares: Vec<usize> = weights.iter().map(|weight| total * weight / sum).collect();
    let mut left = total - shares.iter().sum::<usize>();
    let mut by_remainder: Vec<usize> = (0..weights.len()).collect();
    by_remainder.sort_by_key(|&index| {
        std::cmp::Reverse(total * weights.get(index).copied().unwrap_or(0) % sum)
    });
    for index in by_remainder {
        if left == 0 {
            break;
        }
        if let (Some(share), Some(weight)) = (shares.get_mut(index), weights.get(index))
            && *weight > 0
        {
            *share += 1;
            left -= 1;
        }
    }
    shares
}

fn characters_width(characters: &[(char, Style)]) -> usize {
    character_columns(characters).into_iter().sum()
}

/// The columns each character takes: a grapheme cluster's on its first character and none on the
/// rest, measured as [`display_width`] measures, so an emoji its presentation selector widens is
/// as wide as a terminal draws it.
fn character_columns(characters: &[(char, Style)]) -> Vec<usize> {
    use unicode_segmentation::UnicodeSegmentation;
    let text: String = characters.iter().map(|(character, _)| character).collect();
    let mut columns = vec![0; characters.len()];
    let mut index = 0;
    for cluster in text.graphemes(true) {
        if let Some(slot) = columns.get_mut(index) {
            *slot = display_width(cluster);
        }
        index += cluster.chars().count();
    }
    columns
}

/// The widest run of `characters` no row may break inside: a word between spaces, or less where
/// it holds ideographs to break between.
fn longest_unbreakable(characters: &[(char, Style)]) -> usize {
    let breaks = ideographic_breaks(characters);
    let columns = character_columns(characters);
    let mut longest = 0;
    let mut current = 0;
    for (index, (character, _)) in characters.iter().enumerate() {
        if *character == ' ' {
            current = 0;
            continue;
        }
        if breaks.get(index) == Some(&true) {
            current = 0;
        }
        current += columns.get(index).copied().unwrap_or(0);
        longest = longest.max(current);
    }
    longest
}

/// `row` padded to `width` on the side `alignment` leaves free.
fn aligned(
    row: Line<'static>,
    width: usize,
    alignment: pulldown_cmark::Alignment,
) -> Line<'static> {
    let free = width.saturating_sub(
        row.spans
            .iter()
            .map(|span| display_width(&span.content))
            .sum(),
    );
    let left = match alignment {
        pulldown_cmark::Alignment::Center => free / 2,
        pulldown_cmark::Alignment::Right => free,
        pulldown_cmark::Alignment::Left | pulldown_cmark::Alignment::None => 0,
    };
    let mut spans = vec![Span::raw(" ".repeat(left))];
    spans.extend(row.spans);
    spans.push(Span::raw(" ".repeat(free - left)));
    Line::from(spans)
}

pub(crate) fn is_blank(line: &Line<'static>) -> bool {
    line.spans.iter().all(|span| span.content.trim().is_empty())
}

/// A fenced block's body, highlighted for `language` in the screen's code theme, one row per source
/// line, left for the caller to fold at its width.
pub(crate) fn code_lines(body: &[String], language: Option<&str>) -> Vec<Line<'static>> {
    let highlighter = highlighter();
    let syntax = syntax_for_language(language);
    let mut highlight = HighlightLines::new(syntax, &highlighter.theme);
    body.iter()
        .map(|row| {
            let source = format!("{row}\n");
            match highlight.highlight_line(&source, &highlighter.syntax_set) {
                Ok(ranges) => Line::from(
                    ranges
                        .into_iter()
                        .map(|(style, text)| {
                            Span::styled(
                                text.trim_end_matches('\n').to_string(),
                                syntect_style(style),
                            )
                        })
                        .collect::<Vec<_>>(),
                ),
                Err(error) => {
                    // Plain rather than nothing, so a grammar failure never loses content.
                    tracing::debug!("failed to highlight a line: {error}");
                    Line::from(row.clone())
                }
            }
        })
        .collect()
}

/// A syntect style as a ratatui one, decoding the palette the code theme encodes: foreground
/// only, for the reason no look has a background, which is that a hardcoded dark background
/// fights any terminal not already using one.
fn syntect_style(style: syntect::highlighting::Style) -> Style {
    match palette_slot(style.foreground) {
        Some(slot) => Style::default().fg(Color::Indexed(slot)),
        None => Style::default(),
    }
}

/// Text that may carry the escapes a shell wrote, one line per row, keeping its colors, wrapped
/// to `width`.
pub(crate) fn ansi_lines(text: &str, width: usize) -> Vec<Line<'static>> {
    use ansi_to_tui::IntoText;
    let lines = match text.into_text() {
        Ok(parsed) => parsed.lines,
        Err(_) => text
            .lines()
            .map(|row| Line::from(row.to_string()))
            .collect(),
    };
    lines
        .iter()
        .flat_map(|line| wrap_line(line, width))
        .collect()
}

/// Plain text wrapped to `width`, every row in `style`. Any width above zero is honored: this
/// lays out text inside an area the caller has already sized, with no chrome of its own to fit.
pub(crate) fn plain_lines(text: &str, width: usize, style: Style) -> Vec<Line<'static>> {
    let width = width.max(1);
    let text = crate::text::expand_tabs(text);
    let mut lines = Vec::new();
    for source in text.split('\n') {
        if source.is_empty() {
            lines.push(Line::default());
            continue;
        }
        for row in wrap_to_width(source, width, usize::MAX) {
            lines.push(Line::from(Span::styled(row, style)));
        }
    }
    lines
}

/// One styled line cut into rows no wider than `width`, at a space where one falls in the row and
/// inside a word otherwise, every character keeping the style it came with. What the terminal
/// would otherwise clip at the right edge is what this keeps.
pub(crate) fn wrap_line(line: &Line<'static>, width: usize) -> Vec<Line<'static>> {
    fold_line(line, width, Breaks::Spaces)
}

/// [`wrap_line`] breaking at a column only, every space kept: for text whose spaces are content,
/// such as a code row, where a break at a space would land on its indentation.
pub(crate) fn wrap_line_at_columns(line: &Line<'static>, width: usize) -> Vec<Line<'static>> {
    fold_line(line, width, Breaks::Columns)
}

/// Where a row may break.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Breaks {
    /// At a column only, every space kept.
    Columns,
    /// At a space, which is drawn on neither row.
    Spaces,
    /// At a space, with every space around the break dropped: prose, where a space at either end
    /// of a row is nothing to read.
    Words,
}

fn fold_line(line: &Line<'static>, width: usize, breaks: Breaks) -> Vec<Line<'static>> {
    let characters: Vec<(char, Style)> = line
        .spans
        .iter()
        .flat_map(|span| {
            crate::text::expand_tabs(&span.content)
                .chars()
                .map(|character| (character, span.style))
                .collect::<Vec<_>>()
        })
        .collect();
    fold_characters(&characters, width, breaks)
}

/// The rows of `characters` within `width`, broken where `breaks` allows, and inside a word where
/// no break falls in a row.
fn fold_characters(
    characters: &[(char, Style)],
    width: usize,
    breaks: Breaks,
) -> Vec<Line<'static>> {
    let width = width.max(1);
    if characters.is_empty() {
        return vec![Line::default()];
    }
    let is_space = |index: usize| {
        characters
            .get(index)
            .is_some_and(|(character, _)| *character == ' ')
    };
    let ideographic = if breaks == Breaks::Words {
        ideographic_breaks(characters)
    } else {
        Vec::new()
    };
    let starts_a_row = |index: usize| ideographic.get(index) == Some(&true);
    let widths = character_columns(characters);
    let mut rows = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let mut end = start;
        let mut columns = 0;
        // Where the row would end and the next one start, at the last break that fits.
        let mut last_break = None;
        while let Some((character, _)) = characters.get(end) {
            let advance = widths.get(end).copied().unwrap_or(0);
            if columns + advance > width && end > start {
                break;
            }
            if *character == ' ' {
                last_break = Some((end, end + 1));
            } else if end > start && starts_a_row(end) {
                last_break = Some((end, end));
            }
            columns += advance;
            end += 1;
        }
        let (mut row_end, mut next) = if end == characters.len() || breaks == Breaks::Columns {
            (end, end)
        } else if is_space(end) {
            // The row is full up to a space, which is the break.
            (end, end + 1)
        } else if starts_a_row(end) {
            (end, end)
        } else if let Some((at, after)) = last_break.filter(|&(at, _)| at > start) {
            (at, after)
        } else {
            (end, end)
        };
        if breaks == Breaks::Words {
            while row_end > start && is_space(row_end - 1) {
                row_end -= 1;
            }
            while is_space(next) {
                next += 1;
            }
        }
        // A row of spaces alone, trimmed to nothing, goes with the break.
        if row_end > start || breaks != Breaks::Words {
            rows.push(join_characters(
                characters.get(start..row_end).unwrap_or_default(),
            ));
        }
        start = next;
    }
    if rows.is_empty() {
        rows.push(Line::default());
    }
    rows
}

/// Where a row may start with no space before it, by index: beside an ideograph, wherever the
/// Unicode line-breaking algorithm (UAX #14) allows a break, which keeps a closing mark such as
/// `，` off the start of a row. Only there: the same algorithm breaks after a hyphen or a slash,
/// which would split an identifier or a path a reader copies, and a terminal breaks those at
/// spaces. Hangul is left to its spaces, which Korean is written with.
fn ideographic_breaks(characters: &[(char, Style)]) -> Vec<bool> {
    use unicode_linebreak::{BreakClass, BreakOpportunity, break_property, linebreaks};
    let mut starts = vec![false; characters.len()];
    if characters.iter().all(|(character, _)| character.is_ascii()) {
        return starts;
    }
    let is_ideographic = |index: usize| {
        characters.get(index).is_some_and(|(character, _)| {
            matches!(
                break_property(u32::from(*character)),
                BreakClass::Ideographic | BreakClass::ConditionalJapaneseStarter
            )
        })
    };
    let text: String = characters.iter().map(|(character, _)| character).collect();
    let offsets: Vec<usize> = text.char_indices().map(|(offset, _)| offset).collect();
    for (offset, opportunity) in linebreaks(&text) {
        if opportunity != BreakOpportunity::Allowed {
            continue;
        }
        if let Ok(index) = offsets.binary_search(&offset)
            && index > 0
            && (is_ideographic(index - 1) || is_ideographic(index))
            && let Some(start) = starts.get_mut(index)
        {
            *start = true;
        }
    }
    starts
}

/// Characters back into spans, one span per run of one style.
fn join_characters(characters: &[(char, Style)]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    for &(character, style) in characters {
        match spans.last_mut() {
            Some(last) if last.style == style => last.content.to_mut().push(character),
            _ => spans.push(Span::styled(character.to_string(), style)),
        }
    }
    Line::from(spans)
}

/// Rows at the text edge behind a glyph: the first row carries `glyph` in the gutter, the rest
/// two spaces, so the text reads as one block under one mark.
pub(crate) fn gutter_rows(
    glyph: &str,
    glyph_style: Style,
    rows: Vec<Line<'static>>,
) -> Vec<Line<'static>> {
    let pad = " ".repeat(TEXT_EDGE);
    rows.into_iter()
        .enumerate()
        .map(|(index, mut line)| {
            let lead = if index == 0 {
                Span::styled(format!("{glyph:<TEXT_EDGE$}"), glyph_style)
            } else {
                Span::raw(pad.clone())
            };
            line.spans.insert(0, lead);
            line
        })
        .collect()
}

/// Rows at the text edge with nothing in the gutter.
pub(crate) fn edge_rows(rows: Vec<Line<'static>>) -> Vec<Line<'static>> {
    let pad = " ".repeat(TEXT_EDGE);
    rows.into_iter()
        .map(|mut line| {
            if !line.spans.is_empty() {
                line.spans.insert(0, Span::raw(pad.clone()));
            }
            line
        })
        .collect()
}

/// Plain text wrapped under a glyph: a notice, a stage direction, an error.
pub(crate) fn glyph_text_rows(
    glyph: &str,
    glyph_style: Style,
    text: &str,
    style: Style,
    width: usize,
) -> Vec<Line<'static>> {
    let rows = plain_lines(
        text.trim_end_matches('\n'),
        width.saturating_sub(TEXT_EDGE),
        style,
    );
    gutter_rows(glyph, glyph_style, rows)
}

/// Rows under a tool call: the first behind the connector, the rest flush under its text.
pub(crate) fn connector_rows(rows: Vec<Line<'static>>) -> Vec<Line<'static>> {
    let lead_width = TEXT_EDGE + display_width(CONNECTOR);
    let pad = " ".repeat(lead_width);
    rows.into_iter()
        .enumerate()
        .map(|(index, mut line)| {
            let lead = if index == 0 {
                Span::styled(format!("{}{CONNECTOR}", " ".repeat(TEXT_EDGE)), gray())
            } else {
                Span::raw(pad.clone())
            };
            line.spans.insert(0, lead);
            line
        })
        .collect()
}

/// The columns the connector takes from a row's width.
pub(crate) fn connector_indent() -> usize {
    TEXT_EDGE + display_width(CONNECTOR)
}

/// The opening of a result: its first block, the rows up to the first blank one, at most
/// `most` of them, and a count of everything not shown. A tool writes what it has to say to
/// the model after its payload, set off by a blank line (a listing's footnote, a read's
/// `showing lines` marker), so stopping at the blank shows the payload and counts the rest
/// without knowing either by name. A preview never holds a blank row.
pub(crate) fn preview_rows(rows: Vec<Line<'static>>, most: usize) -> Vec<Line<'static>> {
    let block = rows
        .iter()
        .position(is_blank)
        .unwrap_or(rows.len())
        .min(most);
    if block == rows.len() {
        return rows;
    }
    let omitted = rows.len() - block;
    let mut out: Vec<Line<'static>> = rows.iter().take(block).cloned().collect();
    out.push(Line::from(Span::styled(
        format!("… +{omitted} line{}", if omitted == 1 { "" } else { "s" }),
        gray(),
    )));
    out
}

/// The one row a collapsed thinking block is: the label, then the first words of the block once
/// it has ended, or its last line while it streams, which reads as what the model is doing now;
/// the token estimate when the provider only counts; the bare label while the block is empty.
pub(crate) fn thinking_row(
    thinking: &str,
    estimate: Option<u64>,
    streaming: bool,
    width: usize,
) -> Line<'static> {
    let label = "Thinking";
    let mut spans = vec![
        Span::styled(format!("{ACT_GLYPH:<TEXT_EDGE$}"), gray()),
        Span::styled(label.to_string(), gray()),
    ];
    let budget = width
        .saturating_sub(TEXT_EDGE + display_width(label) + " · ".len())
        .max(1);
    if thinking.trim().is_empty() {
        if let Some(tokens) = estimate {
            spans.push(Span::styled(
                format!(" · {} tokens", format_token_count(tokens)),
                gray(),
            ));
        }
        return Line::from(spans);
    }
    spans.push(Span::styled(" · ".to_string(), gray()));
    if streaming {
        let latest = thinking
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("");
        spans.push(Span::styled(
            cut_text(&sanitize_to_line(latest.trim(), usize::MAX), budget),
            gray(),
        ));
        return Line::from(spans);
    }
    // The runs whole, then cut here at the edge, so the one marker on the screen is the
    // screen's own.
    let mut preview = Vec::new();
    for run in thinking_preview_runs(thinking) {
        let style = REASONING.run_style(REASONING.text, &run);
        preview.push(Span::styled(run.text, style));
    }
    spans.extend(cut_spans(preview, budget));
    Line::from(spans)
}

/// The row an expanded thinking block opens with.
pub(crate) fn thinking_label_row() -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{ACT_GLYPH:<TEXT_EDGE$}"), gray()),
        Span::styled("Thinking".to_string(), gray()),
    ])
}

/// What a tool call's row says after its name: the primary parameter, trimmed and on one line,
/// or nothing when the call has none worth showing.
pub(crate) fn tool_summary(
    name: &str,
    input: &serde_json::Value,
    display_summary: Option<&str>,
) -> Option<String> {
    display_summary
        .map(str::to_string)
        .or_else(|| crate::tools::resolve_primary_param(name, input, None))
        .map(|value| sanitize_to_line(value.trim(), usize::MAX))
        .filter(|value| !value.is_empty())
}

/// One line of `text` highlighted as `language`, in the screen's code theme.
fn highlight_spans(text: &str, language: &str) -> Vec<Span<'static>> {
    code_lines(&[text.to_string()], Some(language))
        .into_iter()
        .next()
        .map(|line| line.spans)
        .unwrap_or_default()
}

/// The characters of `spans` in order, each with the style it was drawn in.
fn characters_of(spans: &[Span<'static>]) -> Vec<(char, Style)> {
    spans
        .iter()
        .flat_map(|span| {
            span.content
                .chars()
                .map(move |character| (character, span.style))
        })
        .collect()
}

/// As many characters from the front of `characters` as fit in `columns`.
fn front_columns(characters: &[(char, Style)], columns: usize) -> &[(char, Style)] {
    let mut used = 0;
    let mut end = 0;
    for (index, (character, _)) in characters.iter().enumerate() {
        let width = crate::text::char_width(*character);
        if used + width > columns {
            break;
        }
        used += width;
        end = index + 1;
    }
    characters.get(..end).unwrap_or_default()
}

/// Spans cut to `budget` columns at their end, `…` in the last column when something went, every
/// character keeping its style; whole when they fit. Always the end and never the middle: a
/// mark in the middle of a line is easy to miss, and what is shown then passes for the whole.
pub(crate) fn cut_spans(spans: Vec<Span<'static>>, budget: usize) -> Vec<Span<'static>> {
    let characters = characters_of(&spans);
    let total: usize = characters
        .iter()
        .map(|(character, _)| crate::text::char_width(*character))
        .sum();
    if total <= budget {
        return spans;
    }
    let mut cut = join_characters(front_columns(&characters, budget.saturating_sub(1))).spans;
    cut.push(Span::styled("…".to_string(), gray()));
    cut
}

/// A line cut the way [`cut_spans`] cuts.
pub(crate) fn cut_line(line: Line<'static>, budget: usize) -> Line<'static> {
    Line::from(cut_spans(line.spans, budget))
}

/// Plain text cut the way [`cut_spans`] cuts.
pub(crate) fn cut_text(text: &str, budget: usize) -> String {
    cut_spans(vec![Span::raw(text.to_string())], budget)
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

/// Plain text cut the way [`cut_text`] cuts but from its start, `…` in the first column: for a
/// path, whose tail is what tells it from the next.
pub(crate) fn cut_text_start(text: &str, budget: usize) -> String {
    if crate::text::display_width(text) <= budget {
        return text.to_string();
    }
    format!("…{}", back_columns(text, budget.saturating_sub(1)))
}

/// The longest suffix of `text` within `columns`.
fn back_columns(text: &str, columns: usize) -> &str {
    let mut used = 0;
    let mut start = text.len();
    for (index, character) in text.char_indices().rev() {
        let width = crate::text::char_width(character);
        if used + width > columns {
            break;
        }
        used += width;
        start = index;
    }
    &text[start..]
}

/// How many lines a call added to a file and took from it, shown on its row as `+3 -1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiffStat {
    pub(crate) added: usize,
    pub(crate) removed: usize,
}

impl DiffStat {
    fn spans(self) -> Vec<Span<'static>> {
        vec![
            Span::styled(
                format!("  +{}", self.added),
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                format!(" -{}", self.removed),
                Style::default().fg(Color::Red),
            ),
        ]
    }
}

/// A tool call's row: the name in the accent, as a function's name is in a code block, so a
/// call is told from the prose around it; its summary after it, dimmed, or highlighted as code
/// when `language` says the summary is a program, cut at the edge when it does not fit; and the
/// count of what it changed, when it changed a file.
pub(crate) fn tool_row(
    name: &str,
    summary: Option<&str>,
    language: Option<&str>,
    stat: Option<DiffStat>,
    width: usize,
) -> Line<'static> {
    let name = cut_text(
        &sanitize_to_line(name, usize::MAX),
        width.saturating_sub(TEXT_EDGE).max(1),
    );
    let mut spans = vec![
        Span::styled(format!("{ACT_GLYPH:<TEXT_EDGE$}"), gray()),
        Span::styled(name.clone(), Style::default().fg(ACCENT)),
    ];
    let stat_spans = stat.map(DiffStat::spans).unwrap_or_default();
    let stat_width: usize = stat_spans
        .iter()
        .map(|span| display_width(&span.content))
        .sum();
    if let Some(summary) = summary {
        let budget = width.saturating_sub(TEXT_EDGE + display_width(&name) + 1 + stat_width);
        if budget >= TOOL_ARGUMENT_FLOOR {
            let styled = match language {
                Some(language) => highlight_spans(summary, language),
                None => vec![Span::styled(summary.to_string(), gray())],
            };
            spans.push(Span::raw(" ".to_string()));
            spans.extend(cut_spans(styled, budget));
        }
    }
    spans.extend(stat_spans);
    Line::from(spans)
}

/// Lines of unchanged text shown on either side of a change, so a preview of three rows holds
/// the line before, the line taken out and the line put in.
const DIFF_CONTEXT_LINES: usize = 1;

/// A file's change as a diff: every changed line with the line before and after it, numbered
/// in the old file for what was there and in the new for what is, `-` red and `+` green and the
/// rest dimmed, hunks apart from each other set off by `…`. A long line wraps under its text
/// rather than being cut, because this is the expanded form's content too.
pub(crate) fn diff_rows(
    old: Option<&str>,
    new: &str,
    width: usize,
) -> (Vec<Line<'static>>, DiffStat) {
    use similar::{ChangeTag, TextDiff};
    let old = old.unwrap_or("");
    let diff = TextDiff::from_lines(old, new);
    let mut stat = DiffStat {
        added: 0,
        removed: 0,
    };
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => stat.added += 1,
            ChangeTag::Delete => stat.removed += 1,
            ChangeTag::Equal => {}
        }
    }
    let lines = old.lines().count().max(new.lines().count()).max(1);
    let number_width = lines.to_string().len();
    let lead_width = number_width + 3;
    let inner = width.saturating_sub(lead_width).max(1);
    let mut rows = Vec::new();
    for (group_index, group) in diff.grouped_ops(DIFF_CONTEXT_LINES).iter().enumerate() {
        if group_index > 0 {
            rows.push(Line::from(Span::styled(
                format!("{:>number_width$} …", ""),
                gray(),
            )));
        }
        for op in group {
            for change in diff.iter_changes(op) {
                let (sign, number, style) = match change.tag() {
                    ChangeTag::Equal => (' ', change.old_index(), gray()),
                    ChangeTag::Delete => ('-', change.old_index(), Style::default().fg(Color::Red)),
                    ChangeTag::Insert => {
                        ('+', change.new_index(), Style::default().fg(Color::Green))
                    }
                };
                let number = number.map_or(String::new(), |index| (index + 1).to_string());
                let text =
                    sanitize_to_line(change.value().trim_end_matches(['\n', '\r']), usize::MAX);
                let lead = Span::styled(format!("{number:>number_width$} {sign} "), gray());
                for (index, row) in wrap_line(&Line::from(Span::styled(text, style)), inner)
                    .into_iter()
                    .enumerate()
                {
                    let mut row = row;
                    row.spans.insert(
                        0,
                        if index == 0 {
                            lead.clone()
                        } else {
                            Span::raw(" ".repeat(lead_width))
                        },
                    );
                    rows.push(row);
                }
            }
        }
    }
    (rows, stat)
}

/// A program an expanded call ran, highlighted as `language` and wrapped whole at `width`, for
/// the tree under the call's row to hang on its bar.
pub(crate) fn command_rows(command: &str, language: &str, width: usize) -> Vec<Line<'static>> {
    let source: Vec<String> = command.lines().map(str::to_string).collect();
    code_lines(&source, Some(language))
        .iter()
        .flat_map(|line| wrap_line(line, width.max(1)))
        .collect()
}

/// The rows under a call's row as one tree: the arguments on a `│` bar, a program's first row
/// behind `$`, and what the call produced on the `└` elbow with its continuation rows flush
/// under it. With nothing produced the last argument takes the elbow, so the tree ends rather
/// than hangs. Every row's text sits at the same column, [`connector_indent`].
pub(crate) fn tree_rows(
    arguments: Vec<Line<'static>>,
    program_first: bool,
    body: Vec<Line<'static>>,
) -> Vec<Line<'static>> {
    let edge = " ".repeat(TEXT_EDGE);
    let lead = |glyph: &str| Span::styled(format!("{edge}{glyph} "), gray());
    let last_argument = arguments.len().saturating_sub(1);
    let mut rows = Vec::with_capacity(arguments.len() + body.len());
    for (index, mut row) in arguments.into_iter().enumerate() {
        let glyph = if index == 0 && program_first {
            "$"
        } else if index == last_argument && body.is_empty() {
            "└"
        } else {
            "│"
        };
        row.spans.insert(0, lead(glyph));
        rows.push(row);
    }
    rows.extend(connector_rows(body));
    rows
}

/// The argument block an expanded tool call hangs under its row, dimmed: every argument in
/// full, wrapped to `width`, because expanded means everything and a cut is what hides a
/// command's tail. The block's own first level of indent is taken off, since the tree supplies
/// the column; deeper levels keep theirs.
pub(crate) fn tool_argument_rows(
    input: &serde_json::Value,
    width: usize,
    style: Style,
) -> Vec<Line<'static>> {
    render_whole_params(input, width.saturating_add(TOOL_PARAM_INDENT.len()))
        .into_iter()
        .map(|row| {
            let row = row
                .strip_prefix(TOOL_PARAM_INDENT)
                .map_or(row.clone(), str::to_string);
            Line::from(Span::styled(row, style))
        })
        .collect()
}

/// A call's arguments as the rows under its row: the code parameter first as highlighted
/// source, then the rest as `key: value` rows in `style`, less the keys in `omit`. The second
/// value says whether a code row leads, which the tree marks with `$`.
pub(crate) fn call_argument_rows(
    name: &str,
    input: &serde_json::Value,
    omit: &[&str],
    style: Style,
    width: usize,
) -> (Vec<Line<'static>>, bool) {
    let mut rows = Vec::new();
    let mut program_first = false;
    let mut rest = input.clone();
    if let Some((key, language)) = crate::tools::builtin_code_param(name)
        && let Some(command) = input.get(key).and_then(serde_json::Value::as_str)
    {
        rows.extend(command_rows(command, language, width));
        program_first = true;
        if let serde_json::Value::Object(fields) = &mut rest {
            fields.remove(key);
        }
    }
    if let serde_json::Value::Object(fields) = &mut rest {
        for key in omit {
            fields.remove(*key);
        }
    }
    let empty_object = matches!(&rest, serde_json::Value::Object(fields) if fields.is_empty());
    if !empty_object && !rest.is_null() {
        rows.extend(tool_argument_rows(&rest, width, style));
    }
    (rows, program_first)
}

/// The checklist as rows: the label, then one row per item under the connector, the item in
/// progress in yellow ahead of the rest in gray. Empty for an empty list.
pub(crate) fn checklist_rows(
    items: &[crate::checklist::ChecklistItem],
    width: usize,
) -> Vec<Line<'static>> {
    use crate::checklist::ChecklistStatus;
    if items.is_empty() {
        return Vec::new();
    }
    let mut lines = vec![Line::from(vec![
        Span::styled(format!("{ACT_GLYPH:<TEXT_EDGE$}"), gray()),
        Span::styled("Checklist".to_string(), Style::default().fg(ACCENT)),
    ])];
    // An item wraps under its marker rather than being cut: the list is what the agent has
    // committed to, and a commitment cut short reads as a different one.
    let budget = width.saturating_sub(connector_indent() + 2).max(1);
    let mut rows = Vec::new();
    for item in items {
        let (marker, color) = match item.status {
            ChecklistStatus::InProgress => ("›", Color::Yellow),
            ChecklistStatus::Pending => ("□", Color::DarkGray),
            ChecklistStatus::Deferred => ("»", Color::DarkGray),
        };
        let mut text = sanitize_to_line(&item.text, usize::MAX);
        if item.status == ChecklistStatus::Deferred
            && let Some(reason) = item.reason.as_deref().filter(|reason| !reason.is_empty())
        {
            text.push_str(&format!(" ({})", sanitize_to_line(reason, usize::MAX)));
        }
        let style = Style::default().fg(color);
        for (index, row) in wrap_to_width(&text, budget, usize::MAX)
            .into_iter()
            .enumerate()
        {
            let lead = if index == 0 { marker } else { " " };
            rows.push(Line::from(Span::styled(format!("{lead} {row}"), style)));
        }
    }
    lines.extend(connector_rows(rows));
    lines
}

/// The approval prompt's body: the call's arguments as the transcript draws them under an
/// expanded call, every one of them and wrapped rather than cut, because what the block hides is
/// what the user authorizes unseen. In the foreground rather than dimmed, since the panel's
/// background is itself gray.
pub(crate) fn approval_rows(
    name: &str,
    input: &serde_json::Value,
    width: usize,
) -> Vec<Line<'static>> {
    let width = width
        .saturating_sub(connector_indent())
        .max(MIN_OUTPUT_WIDTH);
    let (arguments, program_first) = call_argument_rows(name, input, &[], Style::default(), width);
    tree_rows(arguments, program_first, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The approval's body is the call's arguments as a tree under its row, in the foreground:
    /// the connectors are what is dimmed, the words the user is authorizing are not.
    #[test]
    fn the_approval_body_is_the_call_s_arguments_as_a_tree() {
        let input = serde_json::json!({"path": "a.txt", "content": "a"});
        let rows = approval_rows("file_write", &input, 40);
        assert_eq!(plain(&rows), vec!["  │ path: a.txt", "  └ content: a"]);
        for row in &rows {
            assert_eq!(
                row.spans[0].style,
                gray(),
                "the connector is dimmed: {row:?}"
            );
            assert_eq!(
                row.spans[1].style,
                Style::default(),
                "the argument is in the foreground: {row:?}"
            );
        }
    }

    /// A loose list, whose items the source separates with blank lines, keeps its bullets and
    /// the blank between items; a tight one keeps its items flush.
    #[test]
    fn a_loose_list_keeps_its_bullets_and_its_blanks() {
        let rows = plain(&markdown_lines(
            "## Today\n\n- **One**: first thing.\n\n- **Two**: second thing.\n\n## Next\n\n- Three\n",
            60,
        ));
        assert_eq!(
            rows,
            vec![
                "Today",
                "",
                "• One: first thing.",
                "",
                "• Two: second thing.",
                "",
                "Next",
                "",
                "• Three",
            ],
            "{rows:?}"
        );
        let tight = plain(&markdown_lines("- **A**: one\n- **B**: two\n", 60));
        assert_eq!(tight, vec!["• A: one", "• B: two"]);
    }

    /// A bullet that does not fit keeps its hanging indent on the rows it wraps onto, which is the
    /// layout a width-blind renderer loses.
    #[test]
    fn a_wrapped_bullet_keeps_its_hanging_indent() {
        let rows = plain(&markdown_lines(
            "- a bullet whose text runs past the width it was given to wrap in\n",
            30,
        ));
        assert!(rows.len() >= 2, "{rows:?}");
        assert!(rows[0].trim_start().starts_with('•'), "{rows:?}");
        assert!(
            rows[1].starts_with("  ") && !rows[1].trim_start().starts_with('•'),
            "the continuation row is indented under the text, not the bullet: {rows:?}"
        );
        assert!(rows.iter().all(|row| display_width(row) <= 30), "{rows:?}");
    }

    /// A code block is drawn as written whatever opens it: a `#` line in a tilde fence or in an
    /// indented block is code, not a heading owed a blank row.
    #[test]
    fn a_code_block_s_rows_are_drawn_as_written() {
        assert_eq!(
            plain(&markdown_lines("~~~sh\n# comment\necho hi\n~~~\n", 40)),
            vec!["# comment", "echo hi"]
        );
        assert_eq!(
            plain(&markdown_lines("    # indented\n    echo\n", 40)),
            vec!["# indented", "echo"]
        );
    }

    /// Prose never draws a row of nothing: spaces wider than the width, as an indented line of
    /// raw HTML can open with, go with the break.
    #[test]
    fn prose_draws_no_empty_row() {
        let rows = plain(&markdown_lines(
            "<div>\n                              <span>inner</span>\n</div>\n",
            MIN_OUTPUT_WIDTH,
        ));
        assert!(rows.iter().all(|row| !row.is_empty()), "{rows:?}");
    }

    /// An item hangs under its own text whatever its marker, a number right-aligned to the widest
    /// in its list as a bullet does, and every block nested in it hangs there too: a list, a
    /// paragraph after that list, a fenced block.
    #[test]
    fn a_wrapped_item_hangs_under_its_text_and_so_do_its_blocks() {
        let rows = plain(&markdown_lines(
            "9. nine\n10. a tenth item whose text runs past the width it is laid out at\n",
            30,
        ));
        assert_eq!(rows, vec![
            " 9. nine",
            "10. a tenth item whose text",
            "    runs past the width it is",
            "    laid out at",
        ]);
        let rows = plain(&markdown_lines(
            "1. one\n   - a nested bullet that runs past the width\n\n   a later paragraph of the \
             item\n\n   ```sh\n   echo hi\n   ```\n",
            30,
        ));
        assert_eq!(rows, vec![
            "1. one",
            "",
            "   • a nested bullet that runs",
            "     past the width",
            "",
            "   a later paragraph of the",
            "   item",
            "",
            "   echo hi",
        ]);
    }

    /// A row breaks at a space and nowhere else a word goes on: not between `(` and the code span
    /// it opens, however the style changes there.
    #[test]
    fn a_change_of_style_inside_a_word_is_no_place_to_break() {
        let rows = plain(&markdown_lines(
            "falls back by design (`xdp-portal-config.c` in it).\n",
            30,
        ));
        assert_eq!(rows, vec![
            "falls back by design",
            "(xdp-portal-config.c in it)."
        ]);
    }

    /// A row that fills the width up to a space breaks at that space, not at the one before it.
    #[test]
    fn a_row_full_up_to_a_space_breaks_at_that_space() {
        assert_eq!(plain(&wrap_line(&Line::from("aaa bbb ccc"), 7)), vec![
            "aaa bbb", "ccc"
        ]);
        assert_eq!(
            plain(&markdown_lines("aaaaaaaaa bbbbbbbbbb ccc\n", 20)),
            vec!["aaaaaaaaa bbbbbbbbbb", "ccc"]
        );
    }

    /// A quote's bar is on every row of what it holds: a list in it, and the blank between its
    /// blocks.
    #[test]
    fn a_quote_bars_every_row_it_holds() {
        let rows = plain(&markdown_lines(
            "> A quote.\n>\n> - an item in it that runs past the width\n",
            30,
        ));
        assert_eq!(rows, vec![
            "▐ A quote.",
            "▐",
            "▐ • an item in it that runs",
            "▐   past the width",
        ]);
    }

    /// A link's destination follows its text unless the text is the destination: an autolink, an
    /// email and a link written with its own URL as its text each show it once, and an image
    /// inside a link shows its own and the link's.
    #[test]
    fn a_link_shows_its_destination_once() {
        let rows = plain(&markdown_lines(
            "<https://a.example> [https://b.example](https://b.example) <me@c.example> \
             [docs](https://d.example)\n",
            200,
        ));
        assert_eq!(rows, vec![
            "https://a.example https://b.example me@c.example docs (https://d.example)"
        ]);
        assert_eq!(
            plain(&markdown_lines(
                "[![build](https://e.example/badge.svg)](https://e.example)\n",
                200
            )),
            vec!["build (https://e.example/badge.svg) (https://e.example)"],
            "a linked image shows where both go"
        );
        assert_eq!(
            plain(&markdown_lines("see [](https://f.example) here\n", 200)),
            vec!["see https://f.example here"],
            "a link with no text shows its destination as its text"
        );
    }

    /// Text written without spaces breaks between its characters where the Unicode line-breaking
    /// algorithm allows, so a row is filled rather than left with the word before it; a closing
    /// mark never starts a row, so the character before it goes down with it.
    #[test]
    fn ideographic_text_breaks_between_its_characters() {
        let rows = plain(&markdown_lines(
            "Wide: 中文字符的换行测试，这是一段很长的中文文本。\n",
            24,
        ));
        assert_eq!(rows, vec![
            "Wide: 中文字符的换行测",
            "试，这是一段很长的中文文",
            "本。"
        ]);
    }

    /// An emoji a presentation selector widens is measured as a terminal draws it, two columns,
    /// so no row of them is wider than its width, in prose or in a table.
    #[test]
    fn an_emoji_is_measured_as_it_is_drawn() {
        for source in [
            format!("{}\n", "\u{26a0}\u{fe0f} ".repeat(10)),
            "| \u{2714}\u{fe0f} done | b |\n|---|---|\n| x | y |\n".to_string(),
        ] {
            for lines in [markdown_lines(&source, 20), thinking_lines(&source, 20)] {
                for row in plain(&lines) {
                    assert!(display_width(&row) <= 20, "{row:?}");
                }
            }
        }
        let rows = plain(&markdown_lines(
            "| \u{2714}\u{fe0f} done | b |\n|---|---|\n| x | y |\n",
            40,
        ));
        assert_eq!(
            display_width(&rows[0]),
            display_width(&rows[1]),
            "the header and its rule are as wide: {rows:?}"
        );
    }

    /// No row is wider than the width it was laid out at, however deep its nesting: past the
    /// depth an indent leaves room for, an item's marker takes a row of its own and a quote drops
    /// its bar, and the blocks keep the width.
    #[test]
    fn no_row_is_wider_than_its_width_however_deep_the_nesting() {
        let mut source = String::new();
        for depth in 0..12 {
            source.push_str(&format!(
                "{}- level {depth} with a few words to wrap\n",
                "  ".repeat(depth)
            ));
        }
        source.push_str("\n> > > > > > > > > > a deeply quoted line of words\n");
        for width in MIN_OUTPUT_WIDTH..=40 {
            for lines in [
                markdown_lines(&source, width),
                thinking_lines(&source, width),
            ] {
                for row in plain(&lines) {
                    assert!(display_width(&row) <= width, "{width}: {row:?}");
                }
            }
        }
    }

    /// A fenced block is highlighted by its language rather than painted flat, and the fence lines
    /// themselves are not shown as text.
    /// A fenced row wider than the width folds at the width with every character kept: a code
    /// line's spaces are content, so the fold is at a column and never at a space, and the
    /// indentation stays on the first row.
    #[test]
    fn a_fenced_row_wider_than_the_width_folds_at_a_column() {
        let source = format!("```bash\n    --flag={}\n```\n", "x".repeat(100));
        for lines in [markdown_lines(&source, 60), thinking_lines(&source, 60)] {
            let rows = plain(&lines);
            assert_eq!(rows.len(), 2, "{rows:?}");
            assert_eq!(rows[0].chars().count(), 60, "{rows:?}");
            assert!(rows[0].starts_with("    --flag=xxx"), "{rows:?}");
            assert_eq!(rows.concat(), format!("    --flag={}", "x".repeat(100)));
        }
    }

    /// A tab is four columns on every surface the screen draws, since the screen hands cells to
    /// the terminal and a tab among them is dropped with the indentation it carried: a fenced
    /// row, prose, a command, shell output, a diff line and plain text.
    #[test]
    fn a_tab_is_four_columns_on_every_surface() {
        assert_eq!(plain(&markdown_lines("```go\n\tx := 1\n```\n", 60)), vec![
            "    x := 1"
        ]);
        assert_eq!(plain(&thinking_lines("```\n\ty\n```\n", 60)), vec!["    y"]);
        let prose = plain(&markdown_lines("a\tb\n", 60));
        assert!(
            prose.iter().all(|row| !row.contains('\t')) && prose[0].contains("a    b"),
            "{prose:?}"
        );
        assert_eq!(plain(&ansi_lines("\x1b[31m\tout\x1b[0m", 60)), vec![
            "    out"
        ]);
        assert_eq!(plain(&plain_lines("\tp", 60, Style::default())), vec![
            "    p"
        ]);
        assert_eq!(plain(&command_rows("\tmake", "bash", 60)), vec!["    make"]);
        let (diff, _) = diff_rows(None, "\tnew\n", 60);
        let diff = plain(&diff);
        assert!(
            diff.iter().any(|row| row.ends_with("+     new")) && !diff.concat().contains('\t'),
            "{diff:?}"
        );
    }

    #[test]
    fn a_fenced_block_is_highlighted_and_its_fences_are_not_text() {
        let lines = markdown_lines("Before.\n\n```rust\nlet x = 1;\n```\n\nAfter.\n", 60);
        let rows = plain(&lines);
        assert!(
            rows.iter().any(|row| row.contains("let x = 1;")),
            "{rows:?}"
        );
        assert!(rows.iter().all(|row| !row.contains("```")), "{rows:?}");
        let code = lines
            .iter()
            .find(|line| line.spans.iter().any(|span| span.content.contains("let")))
            .expect("the code row");
        assert!(
            code.spans.iter().any(|span| span.style.fg.is_some()),
            "the row is colored by the grammar: {code:?}"
        );
        assert!(rows.contains(&"Before.".to_string()) && rows.contains(&"After.".to_string()));
        let code = rows
            .iter()
            .position(|row| row.contains("let x"))
            .expect("the code row");
        assert_eq!(&rows[code - 2..=code + 2], &[
            "Before.",
            "",
            "let x = 1;",
            "",
            "After."
        ]);
    }

    /// Emphasis is weight, not hue: a bold word is bold in the plain foreground, an italic one
    /// italic, and only inline code takes the accent. Red is left to errors.
    #[test]
    fn emphasis_is_weight_and_code_is_the_accent() {
        let lines = markdown_lines("Some **bold** and *italic* and `code` here.\n", 60);
        let span = |needle: &str| {
            lines
                .iter()
                .flat_map(|line| line.spans.iter())
                .find(|span| span.content.contains(needle))
                .unwrap_or_else(|| panic!("a span holding {needle:?}: {lines:?}"))
                .clone()
        };
        let bold = span("bold");
        assert!(bold.style.add_modifier.contains(Modifier::BOLD), "{bold:?}");
        assert_eq!(bold.style.fg, None, "{bold:?}");
        let italic = span("italic");
        assert!(
            italic.style.add_modifier.contains(Modifier::ITALIC),
            "{italic:?}"
        );
        assert_eq!(italic.style.fg, None, "{italic:?}");
        assert_eq!(span("code").style.fg, Some(ACCENT));
        assert!(
            lines
                .iter()
                .flat_map(|line| line.spans.iter())
                .all(|span| span.style.fg != Some(Color::Red)),
            "nothing in an answer is red: {lines:?}"
        );
    }

    /// Fenced code is colored from the terminal's sixteen, never from a shipped theme's RGB: a
    /// comment is the gray of the chrome, a keyword and a string each take one slot, and the
    /// rest is the plain foreground.
    #[test]
    fn fenced_code_takes_the_terminal_s_palette() {
        let lines = code_lines(
            &["// note".to_string(), "let x = \"hi\";".to_string()],
            Some("rust"),
        );
        let fg = |needle: &str| {
            lines
                .iter()
                .flat_map(|line| line.spans.iter())
                .find(|span| span.content.contains(needle))
                .unwrap_or_else(|| panic!("a span holding {needle:?}: {lines:?}"))
                .style
                .fg
        };
        assert_eq!(fg("note"), Some(Color::Indexed(8)));
        assert_eq!(fg("let"), Some(Color::Indexed(4)));
        assert_eq!(fg("hi"), Some(Color::Indexed(2)));
        assert_eq!(fg("x"), None, "an identifier is the plain foreground");
        assert!(
            lines
                .iter()
                .flat_map(|line| line.spans.iter())
                .all(|span| !matches!(span.style.fg, Some(Color::Rgb(..)))),
            "no RGB reaches the screen: {lines:?}"
        );
    }

    /// No glyph the screen draws has an emoji presentation, so none is drawn as a picture by a
    /// terminal that has them: the mark in the gutter stays one cell in the row's own color.
    #[test]
    fn no_gutter_glyph_is_an_emoji() {
        let is_emoji_range = |character: char| {
            matches!(
                character as u32,
                0x2600..=0x27BF | 0x2B00..=0x2BFF | 0x1F000..=0x1FAFF | 0xFE0F
            )
        };
        for glyph in [
            PROMPT_GLYPH,
            SHELL_GLYPH,
            ACT_GLYPH,
            WARNING_GLYPH,
            ERROR_GLYPH,
            CONNECTOR,
            crate::host::tui::view::BACK_TO_BOTTOM,
        ] {
            assert!(
                !glyph.chars().any(is_emoji_range),
                "{glyph:?} would draw as a picture"
            );
        }
        for glyph in [BULLET, QUOTE_BAR, RULE, TABLE_RULE] {
            assert!(!is_emoji_range(glyph), "{glyph:?} would draw as a picture");
        }
        use crate::checklist::{ChecklistItem, ChecklistStatus};
        let items: Vec<ChecklistItem> = [
            ChecklistStatus::InProgress,
            ChecklistStatus::Pending,
            ChecklistStatus::Deferred,
        ]
        .into_iter()
        .map(|status| ChecklistItem {
            id: 1,
            text: "item".into(),
            status,
            reason: None,
            task: None,
        })
        .collect();
        for row in plain(&checklist_rows(&items, 40)) {
            assert!(!row.chars().any(is_emoji_range), "{row:?}");
        }
    }

    /// A heading is bold in the plain foreground: structure by weight, not by color.
    #[test]
    fn a_heading_is_bold_and_uncolored() {
        let lines = markdown_lines("## Project state\n\nText.\n", 60);
        let heading = lines
            .iter()
            .find(|line| {
                line.spans
                    .iter()
                    .any(|span| span.content.contains("Project"))
            })
            .expect("the heading");
        let span = heading
            .spans
            .iter()
            .find(|span| span.content.contains("Project"))
            .expect("its span");
        assert!(
            span.style.add_modifier.contains(Modifier::BOLD),
            "{heading:?}"
        );
        assert_eq!(span.style.fg, None, "{heading:?}");
    }

    /// A table is laid out with two columns between its cells, a rule under each header cell and
    /// nothing at its edges, a header cell centered over its column.
    #[test]
    fn a_table_has_a_gap_between_its_cells_and_a_rule_under_its_header() {
        let rows = plain(&markdown_lines(
            "| Area | State |\n|---|---|\n| Build | green |\n",
            40,
        ));
        assert_eq!(
            rows,
            vec!["Area   State", "─────  ─────", "Build  green"],
            "{rows:?}"
        );
    }

    /// A table too wide for the width gives each column its longest word first, so an identifier
    /// stays whole and the prose beside it wraps, at spaces, in what is left; a word too wide for
    /// any column gives up its own columns, not the words beside it.
    #[test]
    fn a_table_too_wide_keeps_its_words_whole() {
        let rows = plain(&markdown_lines(
            "| Key | What it does |\n|---|---|\n| `session.context_window` | The window a session \
             is measured against |\n",
            40,
        ));
        assert_eq!(rows, vec![
            "         Key              What it does",
            "──────────────────────  ────────────────",
            "session.context_window  The window a",
            "                        session is",
            "                        measured against",
        ]);
        let rows = plain(&markdown_lines(
            "| Link | Note |\n|---|---|\n| https://example.com/a/very/long/path/that/goes/on/and/on | \
             describing everything here |\n",
            40,
        ));
        for word in ["describing", "everything"] {
            assert!(
                rows.iter().any(|row| row.contains(word)),
                "a word that fits its column stays whole beside one too wide for it: {rows:?}"
            );
        }
        for width in MIN_OUTPUT_WIDTH..=40 {
            let rows = plain(&markdown_lines(
                "| a | b | c |\n|---|---|---|\n| one two | three four five | six |\n",
                width,
            ));
            assert!(
                rows.iter().all(|row| display_width(row) <= width),
                "{width}: {rows:?}"
            );
        }
    }

    /// Reasoning is one gray throughout, code included, so it reads as a footnote beside the
    /// answer.
    #[test]
    fn reasoning_is_gray_throughout() {
        let lines = thinking_lines("Some **thought**.\n\n```rust\nlet x = 1;\n```\n", 60);
        for line in &lines {
            for span in &line.spans {
                if span.content.trim().is_empty() {
                    continue;
                }
                assert_eq!(span.style.fg, Some(Color::DarkGray), "{line:?}");
            }
        }
    }

    /// Plain text wraps at the width and keeps its row breaks.
    #[test]
    fn plain_text_wraps_and_keeps_its_breaks() {
        let rows = plain(&plain_lines(
            "one two three four five six\n\nseven",
            12,
            Style::default(),
        ));
        assert!(rows.len() >= 4, "{rows:?}");
        assert!(rows.iter().all(|row| display_width(row) <= 12), "{rows:?}");
        assert_eq!(rows.last().map(String::as_str), Some("seven"));
    }

    /// Escapes in tool output become styles rather than text, and a row longer than the width
    /// wraps with its style rather than being clipped.
    #[test]
    fn ansi_output_keeps_its_color_and_wraps_at_the_width() {
        let lines = ansi_lines("\x1b[31mred\x1b[0m plain", 40);
        let rows = plain(&lines);
        assert_eq!(rows, vec!["red plain"]);
        assert_eq!(lines[0].spans[0].style.fg, Some(Color::Red));

        let long = ansi_lines(
            "\x1b[32mfailed to resolve path '/home/someone/projects/example/target/missing.txt'\x1b[0m",
            30,
        );
        let rows = plain(&long);
        assert!(rows.len() >= 2, "the row wraps: {rows:?}");
        assert!(rows.iter().all(|row| display_width(row) <= 30), "{rows:?}");
        assert!(
            rows.concat().contains("missing.txt"),
            "nothing is lost at the edge: {rows:?}"
        );
        assert!(
            long.iter().all(|line| line
                .spans
                .iter()
                .all(|span| span.style.fg == Some(Color::Green))),
            "every row keeps the style: {long:?}"
        );
    }

    /// A preview is the opening of a result: its first block up to three rows, then the count
    /// of the rest, trailer included. A result with no blank line and few rows shows whole.
    #[test]
    fn a_preview_is_the_first_block_and_a_count_of_the_rest() {
        let rows = |texts: &[&str]| -> Vec<Line<'static>> {
            texts
                .iter()
                .map(|text| Line::from(text.to_string()))
                .collect()
        };
        let listing = plain(&preview_rows(
            rows(&[
                "ID        Status",
                "dca8209a  completed",
                "",
                "A finished task's full result is delivered to you on its own.",
            ]),
            PREVIEW_ROWS,
        ));
        assert_eq!(
            listing,
            vec!["ID        Status", "dca8209a  completed", "… +2 lines"],
            "the trailer after the blank is counted, not shown"
        );
        let long: Vec<Line<'static>> = (1..=9).map(|n| Line::from(format!("line {n}"))).collect();
        assert_eq!(plain(&preview_rows(long, PREVIEW_ROWS)), vec![
            "line 1",
            "line 2",
            "line 3",
            "… +6 lines"
        ]);
        assert_eq!(
            plain(&preview_rows(rows(&["one", "two", "three"]), PREVIEW_ROWS)),
            vec!["one", "two", "three"],
            "three rows show whole"
        );
        assert!(
            !plain(&preview_rows(rows(&["a", "", "b"]), PREVIEW_ROWS))
                .iter()
                .any(String::is_empty),
            "a preview never holds a blank row"
        );
    }

    /// The thinking row shows the block's first words once it has ended, and its last line while
    /// it still streams.
    #[test]
    fn the_thinking_row_shows_the_opening_after_and_the_latest_line_during() {
        let text = "First thought here.\n\nSecond thought, still going";
        let ended = plain(&[thinking_row(text, None, false, 80)]);
        assert_eq!(ended, vec![
            "• Thinking · First thought here. Second thought, still going"
        ]);
        let streaming = plain(&[thinking_row(text, None, true, 80)]);
        assert_eq!(streaming, vec!["• Thinking · Second thought, still going"]);
        let counted = plain(&[thinking_row("", Some(820), true, 80)]);
        assert_eq!(counted, vec!["• Thinking · 820 tokens"]);
    }

    /// A tool row is the name in the accent, then the summary dimmed; the brackets of the
    /// write-once indicator are not on the screen.
    #[test]
    fn a_tool_row_is_the_name_and_its_summary() {
        let row = tool_row("file_read", Some("notes.txt"), None, None, 80);
        assert_eq!(plain(std::slice::from_ref(&row)), vec![
            "• file_read notes.txt"
        ]);
        assert_eq!(
            row.spans[1].style.fg,
            Some(ACCENT),
            "the name takes the accent"
        );
        assert_eq!(
            row.spans[3].style.fg,
            Some(Color::DarkGray),
            "the summary is dimmed"
        );
        assert_eq!(
            plain(&[tool_row("schedule_list", None, None, None, 80)]),
            vec!["• schedule_list"]
        );
    }

    /// A shell command on a tool row is highlighted as code and, when it does not fit, elided
    /// from the middle with both ends still styled; expanded, it is a block of its own behind
    /// `$ `, wrapped whole.
    #[test]
    fn a_shell_command_is_highlighted_on_the_row_and_whole_under_it() {
        let command = "seq 1 5000 | grep -o 'survived' | sort | uniq -c | head -1";
        let row = tool_row("shell_execute", Some(command), Some("bash"), None, 40);
        let text = plain(std::slice::from_ref(&row)).remove(0);
        assert!(text.starts_with("• shell_execute seq 1 5000"), "{text:?}");
        assert!(
            text.ends_with('…'),
            "cut at the edge, never in the middle: {text:?}"
        );
        assert_eq!(text.matches('…').count(), 1, "{text:?}");
        assert!(
            command.starts_with(
                text.trim_start_matches("• shell_execute ")
                    .trim_end_matches('…')
            ),
            "what shows is the opening of the command as written: {text:?}"
        );
        assert!(display_width(&text) <= 40, "{text:?}");
        assert!(
            row.spans
                .iter()
                .any(|span| span.content == "|" && span.style.fg == Some(Color::Indexed(4))),
            "the command is highlighted as code: {row:?}"
        );
        let whole = tool_row("shell_execute", Some("ls -la"), Some("bash"), None, 80);
        assert_eq!(plain(std::slice::from_ref(&whole)), vec![
            "• shell_execute ls -la"
        ]);

        let rows = command_rows(command, "bash", 30);
        let shown = plain(&rows);
        assert!(shown[0].starts_with("seq 1 5000"), "{shown:?}");
        assert!(shown.len() > 1, "{shown:?}");
        assert!(
            shown.iter().all(|row| display_width(row) <= 30),
            "{shown:?}"
        );
        let joined = shown.join(" ");
        assert!(
            joined.contains("uniq -c | head -1"),
            "nothing is cut: {shown:?}"
        );
    }

    /// A row cut at the edge keeps its styles up to the cut and ends in the marker; plain text
    /// is cut the same way, and a thinking preview too long for its row ends in the one marker.
    #[test]
    fn a_cut_row_keeps_its_styles_and_ends_in_the_marker() {
        let line = Line::from(vec![
            Span::styled("abc".to_string(), Style::default().fg(Color::Red)),
            Span::raw("defghij".to_string()),
        ]);
        let cut = cut_line(line.clone(), 6);
        assert_eq!(plain(std::slice::from_ref(&cut)), vec!["abcde…"]);
        assert_eq!(cut.spans[0].style.fg, Some(Color::Red));
        assert_eq!(plain(std::slice::from_ref(&cut_line(line, 10))), vec![
            "abcdefghij"
        ]);
        assert_eq!(cut_text("abcdefghij", 6), "abcde…");
        assert_eq!(cut_text("abc", 6), "abc");
        let long =
            "The user wants **everything** explained at length, so I will go on for a while.";
        let row = plain(&[thinking_row(long, None, false, 40)]).remove(0);
        assert!(row.ends_with('…') && !row.contains("..."), "{row:?}");
        assert!(display_width(&row) <= 40, "{row:?}");
    }

    /// The rows under a call hang as one tree: the program's first row behind `$`, the rest of
    /// the arguments on the bar, the result on the elbow; with no result the last argument takes
    /// the elbow.
    #[test]
    fn the_rows_under_a_call_hang_as_one_tree() {
        let row = |text: &str| Line::from(text.to_string());
        let shown = plain(&tree_rows(
            vec![row("seq 1 | head"), row("-n 1"), row("timeout: 30")],
            true,
            vec![row("1"), row("done")],
        ));
        assert_eq!(shown, vec![
            "  $ seq 1 | head",
            "  │ -n 1",
            "  │ timeout: 30",
            "  └ 1",
            "    done",
        ]);
        let ended = plain(&tree_rows(
            vec![row("path: notes.txt"), row("mode: read")],
            false,
            Vec::new(),
        ));
        assert_eq!(ended, vec!["  │ path: notes.txt", "  └ mode: read"]);
        assert_eq!(
            plain(&tree_rows(Vec::new(), false, vec![row("alpha")])),
            vec!["  └ alpha"]
        );
    }

    /// A change draws as a diff: the line before, the line taken out in red with its old number,
    /// the line put in in green with its new number, the line after; two changes apart from each
    /// other are two hunks set off by `…`; a file the call created is additions alone.
    #[test]
    fn a_change_draws_as_a_numbered_diff() {
        let old = "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n";
        let new = "one\nTWO\nthree\nfour\nfive\nsix\nseven\neight\nnine\nTEN\nELEVEN\n";
        let (rows, stat) = diff_rows(Some(old), new, 40);
        assert_eq!(stat, DiffStat {
            added: 3,
            removed: 2
        });
        assert_eq!(plain(&rows), vec![
            " 1   one",
            " 2 - two",
            " 2 + TWO",
            " 3   three",
            "   …",
            " 9   nine",
            "10 - ten",
            "10 + TEN",
            "11 + ELEVEN",
        ]);
        let removed = rows[1].spans.last().expect("the text");
        assert_eq!(removed.style.fg, Some(Color::Red));
        let added = rows[2].spans.last().expect("the text");
        assert_eq!(added.style.fg, Some(Color::Green));
        let (created, stat) = diff_rows(None, "a\nb\n", 40);
        assert_eq!(stat, DiffStat {
            added: 2,
            removed: 0
        });
        assert_eq!(plain(&created), vec!["1 + a", "2 + b"]);
        let row = tool_row("file_edit", Some("hello.py"), None, Some(stat), 80);
        assert_eq!(plain(std::slice::from_ref(&row)), vec![
            "• file_edit hello.py  +2 -0"
        ]);
    }

    /// The checklist draws its items under the connector without their ids, the one in progress
    /// marked out from the rest.
    #[test]
    fn the_checklist_rows_mark_the_item_in_progress() {
        use crate::checklist::{ChecklistItem, ChecklistStatus};
        let items = vec![
            ChecklistItem {
                id: 1,
                text: "Review the conflict section".into(),
                status: ChecklistStatus::InProgress,
                reason: None,
                task: None,
            },
            ChecklistItem {
                id: 2,
                text: "Ask Sam".into(),
                status: ChecklistStatus::Deferred,
                reason: Some("waiting on Sam".into()),
                task: None,
            },
        ];
        let rows = plain(&checklist_rows(&items, 80));
        assert_eq!(rows, vec![
            "• Checklist",
            "  └ › Review the conflict section",
            "    » Ask Sam (waiting on Sam)",
        ]);
        assert!(checklist_rows(&[], 80).is_empty());
        let narrow = plain(&checklist_rows(&items[..1], 24));
        assert_eq!(
            narrow,
            vec!["• Checklist", "  └ › Review the", "      conflict section"],
            "a long item wraps under its marker rather than being cut: {narrow:?}"
        );
    }

    /// A cut from the start keeps the tail and marks the cut in the first column, by display
    /// width, so a wide character that would straddle the budget goes with the start.
    #[test]
    fn a_cut_from_the_start_keeps_the_tail() {
        assert_eq!(cut_text_start("~/.config/config", 20), "~/.config/config");
        assert_eq!(cut_text_start("~/.config/config", 12), "…nfig/config");
        assert_eq!(cut_text_start("ab漢字", 3), "…字");
        assert_eq!(cut_text_start("abc", 1), "…");
    }

    /// A call's row shows the one argument that identifies it: a supplied summary first, else
    /// the primary parameter the built-in declares, joined when it is a list; an MCP tool has no
    /// declared parameter and stays bare, and a blank summary is no summary.
    #[test]
    fn a_call_s_summary_is_the_argument_that_identifies_it() {
        let cases: [(&str, serde_json::Value, Option<&str>, Option<&str>); 6] = [
            (
                "memory_search",
                serde_json::json!({"queries": ["window size", "context window"]}),
                None,
                Some("window size, context window"),
            ),
            (
                "file_read",
                serde_json::json!({"path": "/etc/hosts"}),
                None,
                Some("/etc/hosts"),
            ),
            (
                "schedule_cancel",
                serde_json::json!({"id": "7f3a1c22"}),
                None,
                Some("7f3a1c22"),
            ),
            (
                "mcp__ida__decompile",
                serde_json::json!({"address": "0x1400"}),
                None,
                None,
            ),
            (
                "file_read",
                serde_json::json!({"path": "/etc/hosts"}),
                Some("/resolved/by/the/agent"),
                Some("/resolved/by/the/agent"),
            ),
            ("checklist_read", serde_json::json!({}), Some("   "), None),
        ];
        for (name, input, supplied, expected) in cases {
            assert_eq!(
                tool_summary(name, &input, supplied).as_deref(),
                expected,
                "{name} with {supplied:?}"
            );
        }
    }

    /// A tool name is model-supplied too, by way of an MCP server: an escape in it never reaches
    /// the row, and the row stays one row.
    #[test]
    fn a_forged_tool_name_cannot_leave_its_row() {
        let forged = "file_read\u{1b}[2J\u{1b}[1;1H";
        let rows = plain(&[tool_row(forged, None, None, None, 80)]);
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].contains('\u{1b}'), "{rows:?}");
        assert!(rows[0].contains("file_read"), "{rows:?}");
    }
}
