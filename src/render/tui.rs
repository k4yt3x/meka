//! The transcript's text as the terminal UI draws it: markdown, code fences, tool output and
//! plain text, laid out at a width as ratatui lines, and the rows every item is dressed in.
//!
//! Markdown is laid out here from the blocks [`super::markdown`] reads: prose wraps at spaces,
//! under the marker of every list item and the bar of every quote it sits in, and a table's cells
//! wrap in columns fitted to the width; a code block is highlighted by `syntect`. The rows are
//! lines the UI owns rather than bytes on a stream, which is what lets it draw them again at
//! another width.
//!
//! Every item on the screen puts a one-glyph gutter in column 0 and its text at [`TEXT_EDGE`], and
//! stops it a column short of the right edge ([`RIGHT_MARGIN`]). The gutter holds a bar in its
//! level's color for a prompt, `•` for an act of the agent or the host, `▸` for a folded run of
//! steps or a closed compaction and `▾` for an open one, `!` for a warning, `■` for an error,
//! nothing for the answer. The palette is the terminal's own colors, so the screen follows its
//! theme, but for the shades behind blocks, a diff's tints, the white of a prompt's words and the
//! gray dimmed text takes on a shade. No glyph has an emoji presentation: a terminal that draws
//! U+26A0 as a picture draws it the size of two cells in a color of its own, and the gutter is
//! meant to be a mark, not a picture.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::*;

/// The column every item's text starts in, behind its one-glyph gutter. The composer and the
/// prompt block use the same edge, so the whole screen shares one left margin.
pub(crate) const TEXT_EDGE: usize = 2;
/// The column at the screen's right edge no words reach, as the gutter keeps the left: the
/// transcript, the composer, a panel, a question, the completion list and the status row all stop
/// a column short, while a shade behind them (a prompt's, an open run's, a panel's) runs to the
/// edge.
pub(crate) const RIGHT_MARGIN: usize = 1;
/// The bar down the composer's left edge, and a sent prompt's, in the permission level's color,
/// or red in the shell mode.
pub(crate) const COMPOSER_BAR: &str = "▎";
/// The gutter glyph of an act: a tool call, a thinking block, a notice, a stage direction.
pub(crate) const ACT_GLYPH: &str = "•";
/// The gutter glyph of a warning.
pub(crate) const WARNING_GLYPH: &str = "!";
/// The gutter glyph of an error.
pub(crate) const ERROR_GLYPH: &str = "■";
/// The lead of the first row under a tool call: its result, its progress, its sub-agent's record.
const CONNECTOR: &str = "└ ";
/// The shade an open run of steps sits on, from its row through its last step, so what the run
/// holds is told from what stands around it.
pub(crate) const RUN_BACKGROUND: Color = Color::Indexed(234);
/// The shade a code block sits on, so it is told from the prose around it: a step darker than an
/// open run's and a sent prompt's, so a block inside a run reads as an inset and a block is not
/// taken for a prompt.
const CODE_BACKGROUND: Color = Color::Indexed(233);
/// What is dimmed on a shade in place of the terminal's dark gray, which themes set close to their
/// own background and a shade then swallows: a mid gray from the fixed palette, lighter than the
/// chrome's gray and dimmer than the words beside it. Also the bar of a prompt nothing says the
/// level of yet.
pub(crate) const DIM_ON_SHADE: Color = Color::Indexed(245);
/// The gutter glyph of a folded run of steps or a closed compaction.
pub(crate) const FOLDED_GLYPH: &str = "▸";
/// The gutter glyph of a run of steps opened under its row, or an open compaction.
pub(crate) const OPENED_GLYPH: &str = "▾";
/// Lines of a command a collapsed call shows after its first, which its row holds; a single line
/// past these is shown rather than counted, since the count would take its row.
pub(crate) const COMMAND_PREVIEW_ROWS: usize = 2;
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

/// How a row of the transcript copies when a selection covers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowCopy {
    /// Its text from the text edge on: what stands in the gutter is the screen's mark, not text.
    Text,
    /// A code row: the `columns` from `from`, its code without the margins around it, glued to
    /// the row above when it `continues` a line folded at the width.
    Code {
        from: usize,
        columns: usize,
        continues: bool,
    },
    /// A row that is the screen's own and no text: a code block's or a prompt's padding, a
    /// prompt's note, a picture. It copies as nothing.
    Padding,
}

impl RowCopy {
    /// The same row behind `columns` more of prefix: a list's hang, a quote's bar, the text edge.
    pub(crate) fn behind(self, columns: usize) -> Self {
        match self {
            Self::Code {
                from,
                columns: shown,
                continues,
            } => Self::Code {
                from: from + columns,
                columns: shown,
                continues,
            },
            other => other,
        }
    }
}

/// A row as it is drawn and as it copies.
pub(crate) type Row = (Line<'static>, RowCopy);

/// The lines of `rows`, without how they copy.
#[cfg(test)]
fn lines_of(rows: Vec<Row>) -> Vec<Line<'static>> {
    rows.into_iter().map(|(line, _)| line).collect()
}

/// [`markdown_rows`] without how the rows copy.
#[cfg(test)]
pub(crate) fn markdown_lines(markdown: &str, width: usize) -> Vec<Line<'static>> {
    lines_of(markdown_rows(markdown, width))
}

/// Markdown laid out at `width` in the answer's look, each row with how it copies.
pub(crate) fn markdown_rows(markdown: &str, width: usize) -> Vec<Row> {
    lay_out(markdown, width, &ANSWER)
}

/// [`thinking_rows`] without how the rows copy.
#[cfg(test)]
pub(crate) fn thinking_lines(markdown: &str, width: usize) -> Vec<Line<'static>> {
    lines_of(thinking_rows(markdown, width))
}

/// Markdown laid out at `width` in the reasoning's look, each row with how it copies.
pub(crate) fn thinking_rows(markdown: &str, width: usize) -> Vec<Row> {
    lay_out(markdown, width, &REASONING)
}

/// An answer still streaming as rows: what has settled laid out as markdown, and the line still
/// arriving under it laid out on its own, so a token shows the moment it arrives and nothing the
/// line says yet reaches back into the rows above: `2` behind an item is not yet the next item's
/// number, nor a word of the item, and `---` under a paragraph does not make it a heading. A line
/// inside a code block the settled text has not closed is laid out with it, as the code row it
/// is; a line that continues a block above it, as an item's second line does, settles into place
/// when it ends. `reasoning` picks the reasoning's look.
pub(crate) fn streaming_rows(text: &str, width: usize, reasoning: bool) -> Vec<Row> {
    let look = if reasoning { &REASONING } else { &ANSWER };
    let (settled, open) = match text.rfind('\n') {
        Some(cut) => text.split_at(cut + 1),
        None => ("", text),
    };
    if open.is_empty() || ends_inside_a_fence(settled) {
        return lay_out(text, width, look);
    }
    let mut rows = lay_out(settled, width, look);
    // A blank line ends the block above, so the line arriving starts one of its own.
    if settled.ends_with("\n\n") && !rows.is_empty() {
        rows.push((Line::default(), RowCopy::Text));
    }
    rows.extend(lay_out(open, width, look));
    rows
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

fn lay_out(markdown: &str, width: usize, look: &Look) -> Vec<Row> {
    let width = width.max(MIN_OUTPUT_WIDTH);
    let markdown = crate::text::expand_tabs(markdown);
    // The blank a table owes the line after it: GFM reads a non-blank line after a table as one
    // more row.
    let markdown = normalize_spacing(&markdown);
    let mut rows = look.blocks(&markdown::parse(&markdown), width, false);
    // A document ends where its last row does; the blank that separates it from the next block is
    // the transcript's to draw.
    while rows.last().is_some_and(|(line, _)| is_blank(line)) {
        rows.pop();
    }
    rows
}

/// `lines` as rows that copy as text.
fn text_rows(lines: Vec<Line<'static>>) -> Vec<Row> {
    lines
        .into_iter()
        .map(|line| (line, RowCopy::Text))
        .collect()
}

impl Look {
    /// `blocks` at `width`, one blank row between two of them unless `tight`, as the blocks of an
    /// item in a tight list are.
    fn blocks(&self, blocks: &[markdown::Block], width: usize, tight: bool) -> Vec<Row> {
        let mut rows = Vec::new();
        for block in blocks {
            let block_rows = self.block(block, width);
            if block_rows.is_empty() {
                continue;
            }
            if !tight && !rows.is_empty() {
                rows.push((Line::default(), RowCopy::Text));
            }
            rows.extend(block_rows);
        }
        rows
    }

    fn block(&self, block: &markdown::Block, width: usize) -> Vec<Row> {
        match block {
            markdown::Block::Paragraph(runs) => text_rows(self.prose(runs, self.text, width)),
            markdown::Block::Heading { level, runs } => {
                let heading = self
                    .headings
                    .get(usize::from(*level).saturating_sub(1))
                    .copied()
                    .unwrap_or(self.text);
                text_rows(self.prose(runs, heading, width))
            }
            markdown::Block::Code { language, rows } => self.code(rows, language.as_deref(), width),
            markdown::Block::List {
                start,
                loose,
                items,
            } => self.list(*start, *loose, items, width),
            markdown::Block::Quote(blocks) => self.quote(blocks, width),
            markdown::Block::Table(table) => text_rows(self.table(table, width)),
            markdown::Block::Rule => vec![(
                Line::from(Span::styled(RULE.to_string().repeat(width), self.structure)),
                RowCopy::Text,
            )],
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

    fn code(&self, rows: &[String], language: Option<&str>, width: usize) -> Vec<Row> {
        let lines = if self.highlight {
            code_lines(rows, language)
        } else {
            rows.iter()
                .map(|row| Line::from(Span::styled(row.clone(), self.code_block)))
                .collect()
        };
        // Folded at the edge rather than clipped there, as the terminal folds a one-shot's output:
        // at a column, never at a space, because a code row's spaces are content and its
        // indentation is where a break at a space would land. The shade keeps a column clear on
        // each side of the code.
        let folded = width.saturating_sub(2).max(1);
        let pad = (
            shaded_row(Line::default(), CODE_BACKGROUND, width),
            RowCopy::Padding,
        );
        std::iter::once(pad.clone())
            .chain(lines.iter().flat_map(|line| {
                wrap_line_at_columns(line, folded)
                    .into_iter()
                    .enumerate()
                    .map(|(index, row)| {
                        let copy = RowCopy::Code {
                            from: 1,
                            columns: row
                                .spans
                                .iter()
                                .map(|span| display_width(&span.content))
                                .sum(),
                            continues: index > 0,
                        };
                        let mut spans = vec![Span::raw(" ")];
                        spans.extend(row.spans);
                        (shaded_row(Line::from(spans), CODE_BACKGROUND, width), copy)
                    })
            }))
            .chain(std::iter::once(pad))
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
    ) -> Vec<Row> {
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
                rows.push((Line::default(), RowCopy::Text));
            }
            let indent = display_width(&marker.content);
            if width < indent + MIN_NESTED_WIDTH {
                rows.push((Line::from(marker), RowCopy::Text));
                rows.extend(self.blocks(item, width, !loose));
                continue;
            }
            let item_rows = self.blocks(item, width - indent, !loose);
            if item_rows.is_empty() {
                rows.push((Line::from(marker), RowCopy::Text));
                continue;
            }
            let pad = Span::raw(" ".repeat(indent));
            for (index, (mut row, copy)) in item_rows.into_iter().enumerate() {
                if index == 0 {
                    row.spans.insert(0, marker.clone());
                } else if !is_blank(&row) {
                    row.spans.insert(0, pad.clone());
                } else {
                    rows.push((row, copy));
                    continue;
                }
                rows.push((row, copy.behind(indent)));
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
    fn quote(&self, blocks: &[markdown::Block], width: usize) -> Vec<Row> {
        let bar = Span::styled(QUOTE_BAR.to_string(), self.quote_bar);
        let indent = display_width(&bar.content) + 1;
        if width < indent + MIN_NESTED_WIDTH {
            return self.blocks(blocks, width, false);
        }
        self.blocks(blocks, width - indent, false)
            .into_iter()
            .map(|(mut row, copy)| {
                if is_blank(&row) {
                    return (Line::from(bar.clone()), copy);
                }
                row.spans.insert(0, Span::raw(" "));
                row.spans.insert(0, bar.clone());
                (row, copy.behind(indent))
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

/// Whether `line` draws nothing: no character and no shade, since a shaded row of spaces is part of
/// the block it shades.
pub(crate) fn is_blank(line: &Line<'static>) -> bool {
    line.spans
        .iter()
        .all(|span| span.content.trim().is_empty() && !has_background(span.style))
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
/// to `width`. A reset is kept as no style at all: it asks for the terminal's own colors, which
/// is what a span without a style has, and kept as a reset it would undo what the screen lays
/// over a row, the shade of an open run and the reversal of the keyboard's cursor.
pub(crate) fn ansi_lines(text: &str, width: usize) -> Vec<Line<'static>> {
    use ansi_to_tui::IntoText;
    let lines = match text.into_text() {
        Ok(parsed) => parsed.lines,
        Err(_) => text
            .lines()
            .map(|row| Line::from(row.to_string()))
            .collect(),
    };
    let own = |color: Option<Color>| color.filter(|color| *color != Color::Reset);
    lines
        .into_iter()
        .map(|line| {
            Line::from(
                line.spans
                    .into_iter()
                    .map(|span| {
                        let style = Style {
                            fg: own(span.style.fg),
                            bg: own(span.style.bg),
                            underline_color: own(span.style.underline_color),
                            sub_modifier: Modifier::empty(),
                            ..span.style
                        };
                        Span::styled(span.content, style)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .flat_map(|line| wrap_line(&line, width))
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

/// A row of a listing wrapped to `width` as [`wrap_line`] wraps it, the rows after the first set
/// in under the column the first break falls in, so a description wraps under its own column. A
/// column starts after a gap of two spaces or more, the gap meka's listings put between columns
/// ([`crate::text::format_columns_row`]); a row with no gap ahead of the break hangs at its own
/// indentation, and one whose hang would leave fewer than [`MIN_NESTED_WIDTH`] columns hangs at its
/// indentation, then not at all.
pub(crate) fn wrap_listing_line(line: &Line<'static>, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let characters = line_characters(line);
    let rows = fold_characters(&characters, width, Breaks::Spaces);
    let [first, _, ..] = rows.as_slice() else {
        return rows;
    };
    let is_space = |index: usize| {
        characters
            .get(index)
            .is_some_and(|(character, _)| *character == ' ')
    };
    // Where the second row starts: past the first row's characters and the spaces at the break.
    let mut next: usize = first
        .spans
        .iter()
        .map(|span| span.content.chars().count())
        .sum();
    while is_space(next) {
        next += 1;
    }
    if next >= characters.len() {
        return rows;
    }
    let indentation = (0..characters.len())
        .take_while(|&index| is_space(index))
        .count();
    let column = (indentation + 2..=next)
        .rev()
        .find(|&index| !is_space(index) && is_space(index - 1) && is_space(index - 2));
    let columns = character_columns(&characters);
    let column_of = |index: usize| columns.iter().take(index).sum::<usize>();
    let fits = |hang: usize| hang + MIN_NESTED_WIDTH <= width;
    let hang = match column.map(column_of) {
        Some(hang) if fits(hang) => hang,
        _ if fits(indentation) => indentation,
        _ => 0,
    };
    if hang == 0 {
        return rows;
    }
    let mut hung = vec![first.clone()];
    hung.extend(
        fold_characters(
            characters.get(next..).unwrap_or_default(),
            width - hang,
            Breaks::Spaces,
        )
        .into_iter()
        .map(|mut row| {
            row.spans.insert(0, Span::raw(" ".repeat(hang)));
            row
        }),
    );
    hung
}

fn fold_line(line: &Line<'static>, width: usize, breaks: Breaks) -> Vec<Line<'static>> {
    fold_characters(&line_characters(line), width, breaks)
}

/// The characters of `line`, each with its span's style, its tabs expanded to spaces.
fn line_characters(line: &Line<'static>) -> Vec<(char, Style)> {
    line.spans
        .iter()
        .flat_map(|span| {
            crate::text::expand_tabs(&span.content)
                .chars()
                .map(|character| (character, span.style))
                .collect::<Vec<_>>()
        })
        .collect()
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

/// [`edge_rows`] for rows that say how they copy.
pub(crate) fn edge_copy_rows(rows: Vec<Row>) -> Vec<Row> {
    let (lines, copies): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    edge_rows(lines)
        .into_iter()
        .zip(copies.into_iter().map(|copy| copy.behind(TEXT_EDGE)))
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

/// Rows an image is drawn in at most, two pixels to a row.
const PICTURE_ROWS: u32 = 20;
/// The widest an image's kept copy is, in pixels: a column each, wider than any screen's text.
const PICTURE_COLUMNS: u32 = 512;

/// An image as the screen draws it: the pixel size its header states, and a copy small enough to
/// scale to any width when it is laid out, or none when its pixels would not decode.
#[derive(Clone, PartialEq)]
pub(crate) struct Picture {
    size: Option<(u32, u32)>,
    pixels: Option<std::sync::Arc<image::RgbaImage>>,
}

impl std::fmt::Debug for Picture {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Picture")
            .field("size", &self.size)
            .field("decoded", &self.pixels.is_some())
            .finish()
    }
}

impl Picture {
    /// `bytes` decoded, and scaled down to the most the screen ever draws of them.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        let (size, decoded) = crate::image::decode_for_display(bytes);
        let pixels = decoded.map(|decoded| std::sync::Arc::new(kept_copy(decoded)));
        Self { size, pixels }
    }

    /// The image `source` holds, when its bytes are here rather than in the store.
    pub(crate) fn from_source(source: &crate::image::ImageSource) -> Self {
        use base64::Engine as _;
        let bytes = source.base64_data().and_then(|data| {
            base64::engine::general_purpose::STANDARD
                .decode(data.as_bytes())
                .ok()
        });
        match bytes {
            Some(bytes) => Self::from_bytes(&bytes),
            None => Self::unread(),
        }
    }

    /// An image whose bytes were not read.
    pub(crate) fn unread() -> Self {
        Self {
            size: None,
            pixels: None,
        }
    }

    /// What stands in for the image when its pixels would not decode, the size its header states
    /// included when it states one.
    pub(crate) fn label(&self) -> String {
        match self.size {
            Some((width, height)) => format!("[image {width}x{height}]"),
            None => "[image]".to_string(),
        }
    }
}

/// `decoded` as the copy a [`Picture`] keeps: shrunk to its bounds in its own pixel type, turned
/// upright and widened to RGBA once small, so no full-size copy is held beside the decode.
fn kept_copy(decoded: crate::image::Decoded) -> image::RgbaImage {
    decoded
        .upright_within(
            PICTURE_COLUMNS,
            2 * PICTURE_ROWS,
            |pixels, width, height| {
                let size = (pixels.width(), pixels.height());
                let fit = fitted(size, width, height);
                if fit == size {
                    pixels
                } else {
                    shrunk(pixels, fit)
                }
            },
        )
        .into_rgba8()
}

/// `pixels` shrunk to `size` by the fast sampling of a thumbnail, in their own pixel type, each
/// color weighed by its opacity while pixels are averaged; see [`weigh_by_opacity`].
fn shrunk(mut pixels: image::DynamicImage, size: (u32, u32)) -> image::DynamicImage {
    weigh_by_opacity(&mut pixels, true);
    let mut small = pixels.thumbnail_exact(size.0, size.1);
    weigh_by_opacity(&mut small, false);
    small
}

/// Each color of `pixels` multiplied by its opacity, or with `weigh` false divided by it again,
/// for a pixel type that has one: a transparent pixel's color is no color, and averaged in as it
/// is stored it darkens the edge it borders. Done in the pixels' own type, which a widened copy
/// would hold beside them.
fn weigh_by_opacity(pixels: &mut image::DynamicImage, weigh: bool) {
    use image::DynamicImage;
    match pixels {
        DynamicImage::ImageLumaA8(buffer) => weigh_samples::<u8>(buffer, 2, weigh),
        DynamicImage::ImageRgba8(buffer) => weigh_samples::<u8>(buffer, 4, weigh),
        DynamicImage::ImageLumaA16(buffer) => weigh_samples::<u16>(buffer, 2, weigh),
        DynamicImage::ImageRgba16(buffer) => weigh_samples::<u16>(buffer, 4, weigh),
        DynamicImage::ImageRgba32F(buffer) => weigh_samples::<f32>(buffer, 4, weigh),
        _ => {}
    }
}

/// [`weigh_by_opacity`] over the samples of pixels of `channels` channels each, opacity last.
fn weigh_samples<S: Sample>(samples: &mut [S], channels: usize, weigh: bool) {
    for pixel in samples.chunks_exact_mut(channels) {
        let Some((opacity, colors)) = pixel.split_last_mut() else {
            continue;
        };
        let opacity = opacity.fraction();
        if !weigh && opacity <= 0.0 {
            continue;
        }
        for color in colors {
            let value = color.fraction();
            *color = S::from_fraction(if weigh {
                value * opacity
            } else {
                value / opacity
            });
        }
    }
}

/// A sample of a pixel as a fraction of its full scale, and back.
trait Sample: Copy {
    fn fraction(self) -> f32;
    fn from_fraction(fraction: f32) -> Self;
}

impl Sample for u8 {
    fn fraction(self) -> f32 {
        f32::from(self) / f32::from(u8::MAX)
    }

    fn from_fraction(fraction: f32) -> Self {
        (fraction * f32::from(u8::MAX))
            .round()
            .clamp(0.0, f32::from(u8::MAX)) as u8
    }
}

impl Sample for u16 {
    fn fraction(self) -> f32 {
        f32::from(self) / f32::from(u16::MAX)
    }

    fn from_fraction(fraction: f32) -> Self {
        (fraction * f32::from(u16::MAX))
            .round()
            .clamp(0.0, f32::from(u16::MAX)) as u16
    }
}

impl Sample for f32 {
    fn fraction(self) -> f32 {
        self
    }

    fn from_fraction(fraction: f32) -> Self {
        fraction
    }
}

/// `picture` in half-block cells, two pixels to a row, scaled to fit `width` columns and
/// [`PICTURE_ROWS`] rows without growing past its kept copy; a pixel more than half transparent
/// leaves the terminal's own background. `None` when its pixels would not decode, or none of them
/// is opaque enough to draw, which would be rows of nothing.
pub(crate) fn picture_rows(picture: &Picture, width: usize) -> Option<Vec<Line<'static>>> {
    let pixels = picture.pixels.as_deref()?;
    let columns = u32::try_from(width).unwrap_or(u32::MAX);
    let (scaled_width, scaled_height) = fitted(pixels.dimensions(), columns, 2 * PICTURE_ROWS);
    let shown = if (scaled_width, scaled_height) == pixels.dimensions() {
        pixels.clone()
    } else {
        scaled(
            pixels.clone(),
            (scaled_width, scaled_height),
            image::imageops::FilterType::Triangle,
        )
    };
    let color = |x: u32, y: u32| {
        shown
            .get_pixel_checked(x, y)
            .filter(|pixel| pixel.0[3] >= 128)
            .map(|pixel| Color::Rgb(pixel.0[0], pixel.0[1], pixel.0[2]))
    };
    let rows = (0..scaled_height)
        .step_by(2)
        .map(|y| {
            Line::from(
                (0..scaled_width)
                    .map(|x| match (color(x, y), color(x, y + 1)) {
                        (Some(top), Some(bottom)) => {
                            Span::styled("▀", Style::default().fg(top).bg(bottom))
                        }
                        (Some(top), None) => Span::styled("▀", Style::default().fg(top)),
                        (None, Some(bottom)) => Span::styled("▄", Style::default().fg(bottom)),
                        (None, None) => Span::raw(" "),
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<Line<'static>>>();
    let drawn = rows
        .iter()
        .flat_map(|row| &row.spans)
        .any(|span| span.style != Style::default());
    drawn.then_some(rows)
}

/// `size` scaled down to fit `width` by `height` with its proportions kept, and never up.
fn fitted(size: (u32, u32), width: u32, height: u32) -> (u32, u32) {
    let (own_width, own_height) = (size.0.max(1), size.1.max(1));
    let scale = (f64::from(width.max(1)) / f64::from(own_width))
        .min(f64::from(height.max(1)) / f64::from(own_height))
        .min(1.0);
    let side = |own: u32| ((f64::from(own) * scale).round() as u32).max(1);
    (side(own_width), side(own_height))
}

/// `pixels` scaled to `size` smoothly through `filter`, each color weighed by its opacity while
/// pixels are averaged; see [`weigh_by_opacity`].
fn scaled(
    mut weighed: image::RgbaImage,
    size: (u32, u32),
    filter: image::imageops::FilterType,
) -> image::RgbaImage {
    weigh_samples::<u8>(&mut weighed, 4, true);
    let mut out = image::imageops::resize(&weighed, size.0, size.1, filter);
    weigh_samples::<u8>(&mut out, 4, false);
    out
}

/// A rule across the text behind `glyph` in the gutter, with `icon` and `words` on it: a boundary
/// in the conversation rather than something said in it. The words are centered on the whole
/// row, which is how the eye takes it, and the glyph and the icon stand outside that measure.
/// Where the rule has no room on either side, the row is the icon and the words, cut at the edge.
pub(crate) fn boundary_row(glyph: &str, icon: &str, words: &str, width: usize) -> Line<'static> {
    let gutter = Span::styled(format!("{glyph:<TEXT_EDGE$}"), gray());
    let words_width = display_width(words);
    let words_start = width.saturating_sub(words_width) / 2;
    // A space before the icon and one between it and the words.
    let left = words_start
        .checked_sub(display_width(icon) + 2 + TEXT_EDGE)
        .filter(|&left| left > 0);
    let right = width
        .checked_sub(words_start + words_width + 1)
        .filter(|&right| right > 0);
    let (Some(left), Some(right)) = (left, right) else {
        return Line::from(vec![
            gutter,
            Span::styled(
                cut_text(&format!("{icon} {words}"), width.saturating_sub(TEXT_EDGE)),
                gray(),
            ),
        ]);
    };
    Line::from(vec![
        gutter,
        Span::styled(TABLE_RULE.to_string().repeat(left), gray()),
        Span::styled(format!(" {icon} {words} "), gray()),
        Span::styled(TABLE_RULE.to_string().repeat(right), gray()),
    ])
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
    let lead_width = connector_indent();
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

/// `line` on `shade` across `width` columns: every cell without a background of its own takes
/// the shade, the rest of the row is painted with it, since a terminal drops the background of
/// trailing spaces after a reset, and it is dimmed as [`on_shade`] dims. The line's own style, a
/// cursor's reversal say, stays the line's.
pub(crate) fn shaded_row(line: Line<'static>, shade: Color, width: usize) -> Line<'static> {
    let line = on_shade(line);
    let mut used = 0;
    let mut spans: Vec<Span<'static>> = line
        .spans
        .into_iter()
        .map(|span| {
            // Measured as the frame draws it, so the shade reaches the edge whatever the row holds.
            used += span.width();
            let mut style = span.style;
            if !has_background(style) {
                style = style.bg(shade);
            }
            Span::styled(span.content, style)
        })
        .collect();
    spans.push(Span::styled(
        " ".repeat(width.saturating_sub(used)),
        Style::default().bg(shade),
    ));
    Line {
        spans,
        style: line.style,
        alignment: line.alignment,
    }
}

/// `line` as it is drawn on a shade, whichever shade: what is dimmed in the terminal's dark gray
/// in [`DIM_ON_SHADE`], which the shade does not swallow. Every row a shaded surface draws from
/// rows laid out elsewhere passes here: the transcript's runs and code blocks, a panel, a
/// question, the completion list, the status row. The composer, which draws its own few dimmed
/// words, draws them in [`DIM_ON_SHADE`] itself.
pub(crate) fn on_shade(line: Line<'static>) -> Line<'static> {
    // The palette's dark gray by name, or by its slot, as a program's `38;5;8` names it.
    let dim = |style: Style| {
        if matches!(style.fg, Some(Color::DarkGray | Color::Indexed(8))) {
            style.fg(DIM_ON_SHADE)
        } else {
            style
        }
    };
    Line {
        spans: line
            .spans
            .into_iter()
            .map(|span| Span::styled(span.content, dim(span.style)))
            .collect(),
        style: dim(line.style),
        alignment: line.alignment,
    }
}

/// Whether `style` paints a background of its own: a reset, which a program's colored output
/// ends every run with, is the terminal's background, as no background is.
fn has_background(style: Style) -> bool {
    !matches!(style.bg, None | Some(Color::Reset))
}

/// The opening of a result: its first block, the rows up to the first blank one, at most
/// `most` of them, and a count of everything not shown. A tool writes what it has to say to the
/// model after its payload, set off by a blank line (a listing's footnote, a read's `showing lines`
/// marker), so stopping at the blank shows the payload and counts the rest without knowing either
/// by name. A single row past `most` is shown rather than counted, since the count would take its
/// row. Blank rows ahead of the first block are dropped, and a preview never holds a blank row.
pub(crate) fn preview_rows(rows: Vec<Line<'static>>, most: usize) -> Vec<Line<'static>> {
    let rows: Vec<Line<'static>> = rows.into_iter().skip_while(is_blank).collect();
    let mut block = rows
        .iter()
        .position(is_blank)
        .unwrap_or(rows.len())
        .min(most);
    if rows.len() == block + 1 && rows.get(block).is_some_and(|row| !is_blank(row)) {
        block += 1;
    }
    let omitted = rows.len() - block;
    if omitted == 0 {
        return rows;
    }
    let mut out: Vec<Line<'static>> = rows.into_iter().take(block).collect();
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
        .or_else(|| crate::tools::resolve_primary_param(name, input))
        .map(|value| sanitize_to_line(value.trim(), usize::MAX))
        .filter(|value| !value.is_empty())
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

/// As many characters from the front of `characters` as fit in `columns`; see [`front_text`].
fn front_columns(characters: &[(char, Style)], columns: usize) -> &[(char, Style)] {
    let text: String = characters.iter().map(|(character, _)| *character).collect();
    let kept = front_text(&text, columns).chars().count();
    characters.get(..kept).unwrap_or_default()
}

/// The longest prefix of `text` within `columns`, whole graphemes, so a cut never parts a
/// character from the marks that shape it, measured as [`crate::text::display_width`] measures
/// it, which is the measure a fit is judged by: a character's own width can be less than what it
/// takes beside the one before it, as an emoji's presentation selector widens it.
fn front_text(text: &str, columns: usize) -> &str {
    use unicode_segmentation::UnicodeSegmentation as _;
    let ends: Vec<usize> = text
        .grapheme_indices(true)
        .map(|(start, grapheme)| start + grapheme.len())
        .collect();
    let prefix = |count: usize| {
        count
            .checked_sub(1)
            .and_then(|last| ends.get(last))
            .map_or("", |&end| text.get(..end).unwrap_or_default())
    };
    prefix(longest_fitting(ends.len(), |count| {
        crate::text::display_width(prefix(count)) <= columns
    }))
}

/// The most of `count` items, taken in order from one end, that `fits`: a search over the
/// counts, since a measure of the whole is all there is and it is not taken one item at a time.
/// Zero items always fit.
fn longest_fitting(count: usize, fits: impl Fn(usize) -> bool) -> usize {
    if fits(count) {
        return count;
    }
    let (mut fitting, mut over) = (0, count);
    while over - fitting > 1 {
        let middle = fitting + (over - fitting) / 2;
        if fits(middle) {
            fitting = middle;
        } else {
            over = middle;
        }
    }
    fitting
}

/// Spans cut to `budget` columns at their end, `…` in the last column when something went, every
/// character keeping its style; whole when they fit. Always the end and never the middle: a
/// mark in the middle of a line is easy to miss, and what is shown then passes for the whole.
pub(crate) fn cut_spans(spans: Vec<Span<'static>>, budget: usize) -> Vec<Span<'static>> {
    let whole: String = spans.iter().map(|span| span.content.as_ref()).collect();
    if crate::text::display_width(&whole) <= budget {
        return spans;
    }
    let characters = characters_of(&spans);
    let mut cut = join_characters(front_columns(&characters, budget.saturating_sub(1))).spans;
    // On the background of what it stands in for, so a tinted row stays tinted to its end, in
    // the gray that reads on one.
    let background = cut.last().map_or_else(Style::default, |span| span.style);
    let ellipsis = Style {
        bg: background.bg,
        ..if has_background(background) {
            Style::default().fg(DIM_ON_SHADE)
        } else {
            gray()
        }
    };
    cut.push(Span::styled("…".to_string(), ellipsis));
    cut
}

/// A line cut the way [`cut_spans`] cuts.
pub(crate) fn cut_line(line: Line<'static>, budget: usize) -> Line<'static> {
    Line::from(cut_spans(line.spans, budget))
}

/// A line meka says in its own voice as the screen draws it: without a closing period, which no
/// line on the screen ends in, so a sentence another surface shares reads here as the screen's own
/// do. An ellipsis is not a period.
pub(crate) fn without_closing_period(text: &str) -> &str {
    let text = text.trim_end();
    match text.strip_suffix('.') {
        Some(rest) if !rest.ends_with('.') => rest,
        _ => text,
    }
}

/// A line meka says in its own voice as the screen draws it: led by a capital and without a
/// closing period ([`without_closing_period`]), so a sentence another surface shares, an error's
/// above all, whose words follow a prefix there, reads here as the screen's own do. A line that
/// opens with a number, a mark or a name keeps it: a word of anything but lowercase letters, a
/// tool's name among them, or meka's own, which is written lowercase.
pub(crate) fn screen_voice(text: &str) -> std::borrow::Cow<'_, str> {
    let text = without_closing_period(text);
    let first_word = text
        .split(|character: char| character.is_whitespace() || character == ':')
        .next()
        .unwrap_or_default();
    if first_word.is_empty() || first_word == "meka" || !first_word.chars().all(char::is_lowercase)
    {
        return text.into();
    }
    let mut characters = text.chars();
    match characters.next() {
        Some(first) => first
            .to_uppercase()
            .chain(characters)
            .collect::<String>()
            .into(),
        None => text.into(),
    }
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

/// The longest suffix of `text` within `columns`, whole graphemes, measured as
/// [`front_columns`] measures.
fn back_columns(text: &str, columns: usize) -> &str {
    use unicode_segmentation::UnicodeSegmentation as _;
    let starts: Vec<usize> = text
        .grapheme_indices(true)
        .map(|(start, _)| start)
        .collect();
    let suffix = |count: usize| {
        starts
            .len()
            .checked_sub(count)
            .and_then(|first| starts.get(first))
            .map_or("", |&start| text.get(start..).unwrap_or_default())
    };
    suffix(longest_fitting(starts.len(), |count| {
        crate::text::display_width(suffix(count)) <= columns
    }))
}

/// A tool's name as its row shows it, in the accent, as a function's name is in a code block, so a
/// call is told from the prose around it; an MCP tool's `mcp__server__` prefix dimmed, so the
/// name it is known by stands out while the whole name stays on the row.
pub(crate) fn tool_name_spans(name: &str, budget: usize) -> Vec<Span<'static>> {
    let name = cut_text(&sanitize_to_line(name, usize::MAX), budget.max(1));
    let prefix = crate::tools::server_of_tool(&name)
        .map_or(0, |server| "mcp__".len() + server.len() + "__".len());
    let (prefix, own) = name.split_at(prefix.min(name.len()));
    let mut spans = Vec::new();
    if !prefix.is_empty() {
        spans.push(Span::styled(prefix.to_string(), gray()));
    }
    spans.push(Span::styled(own.to_string(), Style::default().fg(ACCENT)));
    spans
}

/// How many lines a change put in and took out, shown on its call's row as `+3 -1`.
pub(crate) fn change_tag(change: &crate::conversation::Change) -> Vec<Span<'static>> {
    vec![
        Span::styled(
            format!("+{}", change.added),
            Style::default().fg(Color::Green),
        ),
        Span::styled(
            format!(" -{}", change.removed),
            Style::default().fg(Color::Red),
        ),
    ]
}

/// What a failed call's row says of it, in red: the code a command exited with, or that it failed.
pub(crate) fn failure_tag(exit_code: Option<i32>) -> Vec<Span<'static>> {
    let text = match exit_code {
        Some(code) => format!("exit {code}"),
        None => "failed".to_string(),
    };
    vec![Span::styled(text, Style::default().fg(Color::Red))]
}

/// A tool call's row: its name, then its summary (styled by the caller, cut at the edge when it
/// does not fit), then `tag`, what the call changed or how it failed, which is never cut.
pub(crate) fn tool_row(
    name: &str,
    summary: Vec<Span<'static>>,
    tag: Vec<Span<'static>>,
    width: usize,
) -> Line<'static> {
    let mut spans = vec![Span::styled(format!("{ACT_GLYPH:<TEXT_EDGE$}"), gray())];
    let budget = summary_room(name, &tag, width);
    spans.extend(tool_name_spans(
        name,
        width.saturating_sub(TEXT_EDGE + tag_room(&tag)),
    ));
    if !summary.is_empty() && budget > 0 {
        spans.push(Span::raw(" ".to_string()));
        spans.extend(cut_spans(summary, budget));
    }
    if !tag.is_empty() {
        spans.push(Span::raw("  ".to_string()));
        spans.extend(tag);
    }
    Line::from(spans)
}

/// The columns a [`tool_row`] `width` wide leaves its summary after the name and `tag`, or none
/// where that is below [`TOOL_ARGUMENT_FLOOR`] and the row shows no summary.
pub(crate) fn summary_room(name: &str, tag: &[Span<'static>], width: usize) -> usize {
    let name_width: usize = tool_name_spans(name, width.saturating_sub(TEXT_EDGE + tag_room(tag)))
        .iter()
        .map(|span| display_width(&span.content))
        .sum();
    let budget = width.saturating_sub(TEXT_EDGE + name_width + 1 + tag_room(tag));
    if budget >= TOOL_ARGUMENT_FLOOR {
        budget
    } else {
        0
    }
}

/// The columns a tool row's tag takes with the gap before it.
fn tag_room(tag: &[Span<'static>]) -> usize {
    if tag.is_empty() {
        0
    } else {
        tag.iter()
            .map(|span| display_width(&span.content))
            .sum::<usize>()
            + 2
    }
}

/// `spans` with their first `columns` dimmed: a part of a command that does nothing, drawn as it
/// ran but out of the way of the part that does.
pub(crate) fn dim_front(spans: Vec<Span<'static>>, columns: usize) -> Vec<Span<'static>> {
    let characters = characters_of(&spans);
    let front = front_columns(&characters, columns).len();
    let mut dimmed: Vec<(char, Style)> = characters;
    for (_, style) in dimmed.iter_mut().take(front) {
        *style = gray();
    }
    join_characters(&dimmed).spans
}

/// The tint a removed line of a diff sits on, and its number and sign are drawn in.
const REMOVED_BACKGROUND: Color = Color::Indexed(52);
const REMOVED_MARK: Color = Color::Indexed(210);
/// The tint an added line of a diff sits on, and its number and sign are drawn in.
const ADDED_BACKGROUND: Color = Color::Indexed(22);
const ADDED_MARK: Color = Color::Indexed(120);

/// A change as its diff: every changed line with the line before and after it, numbered in the
/// old text for what was there and in the new for what is, a `-` line on a red tint and a `+` line
/// on a green one across the width, its number and sign in the tint's light shade and its text in
/// white, which reads on both whatever the terminal's theme, and the rest dimmed, hunks apart from
/// each other set off by `…`. A long line wraps under its text, as the expanded form shows it.
pub(crate) fn change_rows(
    change: &crate::conversation::Change,
    width: usize,
) -> Vec<Line<'static>> {
    diff_rows(change, width, true)
        .into_iter()
        .map(|(row, _)| row)
        .collect()
}

/// A change as a collapsed call shows it: its first `most` rows, each line one row for the caller
/// to cut at the edge, and a count of the lines past them, the lines the record left out among
/// them and the marks between hunks not. A single line past them is shown rather than counted,
/// as a result's preview shows it, and a mark between hunks never ends the rows shown.
pub(crate) fn change_preview(
    change: &crate::conversation::Change,
    width: usize,
    most: usize,
) -> Vec<Line<'static>> {
    let rows = diff_rows(change, width, false);
    let left_out = change.lines_left_out();
    if left_out == 0 && rows.len() <= most + 1 {
        return rows.into_iter().map(|(row, _)| row).collect();
    }
    let past = rows.iter().skip(most).filter(|(_, line)| *line).count() + left_out;
    let mut shown: Vec<(Line<'static>, bool)> = rows.into_iter().take(most).collect();
    if shown.last().is_some_and(|(_, line)| !*line) {
        shown.pop();
    }
    let mut shown: Vec<Line<'static>> = shown.into_iter().map(|(row, _)| row).collect();
    if past > 0 {
        shown.push(Line::from(Span::styled(
            format!("… +{past} line{}", if past == 1 { "" } else { "s" }),
            gray(),
        )));
    }
    shown
}

/// The rows of [`change_rows`], wrapped or a line to a row, each with whether it is a line of
/// the change rather than the mark between two hunks.
fn diff_rows(
    change: &crate::conversation::Change,
    width: usize,
    wraps: bool,
) -> Vec<(Line<'static>, bool)> {
    use crate::conversation::ChangeKind;
    let number_width = change
        .hunks
        .iter()
        .flatten()
        .map(|line| line.number)
        .max()
        .unwrap_or(1)
        .to_string()
        .len();
    let lead_width = number_width + 3;
    let inner = width.saturating_sub(lead_width).max(1);
    let mut rows = Vec::new();
    for (index, hunk) in change.hunks.iter().enumerate() {
        if index > 0 {
            rows.push((
                Line::from(Span::styled(format!("{:>number_width$} …", ""), gray())),
                false,
            ));
        }
        for line in hunk {
            let (sign, mark, style, tint) = match line.kind {
                ChangeKind::Kept => (' ', gray(), gray(), None),
                ChangeKind::Removed => (
                    '-',
                    Style::default().fg(REMOVED_MARK),
                    Style::default().fg(WHITE),
                    Some(REMOVED_BACKGROUND),
                ),
                ChangeKind::Added => (
                    '+',
                    Style::default().fg(ADDED_MARK),
                    Style::default().fg(WHITE),
                    Some(ADDED_BACKGROUND),
                ),
            };
            let text = sanitize_to_line(&line.text, usize::MAX);
            let lead = Span::styled(format!("{:>number_width$} {sign} ", line.number), mark);
            let line = Line::from(Span::styled(text, style));
            let wrapped = if wraps {
                wrap_line(&line, inner)
            } else {
                vec![line]
            };
            for (row_index, row) in wrapped.into_iter().enumerate() {
                let mut row = row;
                row.spans.insert(
                    0,
                    if row_index == 0 {
                        lead.clone()
                    } else {
                        Span::raw(" ".repeat(lead_width))
                    },
                );
                rows.push((
                    match tint {
                        Some(tint) => shaded_row(row, tint, width),
                        None => row,
                    },
                    true,
                ));
            }
        }
    }
    rows
}

/// The first `most` lines of a program, each cut to a column past `columns` and highlighted as
/// `language`, one row per line, for the caller to cut at its edge: what a collapsed call shows
/// of it. Only what is shown is highlighted, since the highlighter's cost grows faster than a
/// line's length, a running call's row is laid out again with every line of output, and a session
/// holds hundreds of calls. The lines are highlighted together, so each reads in the state the one
/// before leaves, but afresh after a line the cut shortened, whose state its whole line would not
/// leave: a string the cut opened would color every line under it.
pub(crate) fn command_lines(
    command: &str,
    language: &str,
    most: usize,
    columns: usize,
) -> Vec<Line<'static>> {
    let mut rows = Vec::new();
    let mut together: Vec<String> = Vec::new();
    for line in command.lines().take(most) {
        let line = sanitize_to_line(line, usize::MAX);
        let kept = front_text(&line, columns.saturating_add(1));
        let cut = kept.len() < line.len();
        together.push(kept.to_string());
        if cut {
            rows.extend(code_lines(&std::mem::take(&mut together), Some(language)));
        }
    }
    rows.extend(code_lines(&together, Some(language)));
    rows
}

/// A program an expanded call ran, highlighted as `language` and wrapped whole at `width`.
pub(crate) fn command_rows(command: &str, language: &str, width: usize) -> Vec<Line<'static>> {
    command_lines(command, language, usize::MAX, usize::MAX)
        .iter()
        .flat_map(|line| wrap_line(line, width.max(1)))
        .collect()
}

/// A call's arguments as `key: value` rows in `style`, less the keys in `omit`: every argument in
/// full, wrapped to `width`, because expanded means everything and a cut is what hides a command's
/// tail. The block's own first level of indent is taken off, since the call's rows supply the
/// column; deeper levels keep theirs.
pub(crate) fn argument_rows(
    input: &serde_json::Value,
    omit: &[&str],
    style: Style,
    width: usize,
) -> Vec<Line<'static>> {
    let mut rest = input.clone();
    if let serde_json::Value::Object(fields) = &mut rest {
        for key in omit {
            fields.remove(*key);
        }
    }
    let empty_object = matches!(&rest, serde_json::Value::Object(fields) if fields.is_empty());
    if empty_object || rest.is_null() {
        return Vec::new();
    }
    render_whole_params(&rest, width.saturating_add(TOOL_PARAM_INDENT.len()))
        .into_iter()
        .map(|row| {
            let row = row
                .strip_prefix(TOOL_PARAM_INDENT)
                .map_or(row.clone(), str::to_string);
            Line::from(Span::styled(row, style))
        })
        .collect()
}

/// The checklist as rows: the label, then one row per item under it, the item in progress in
/// yellow ahead of the rest in gray. Empty for an empty list.
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
    let mut arguments = Vec::new();
    let mut omit = Vec::new();
    let mut program_first = false;
    if let Some((key, language)) = crate::tools::builtin_code_param(name)
        && let Some(command) = input.get(key).and_then(serde_json::Value::as_str)
    {
        arguments.extend(command_rows(command, language, width));
        omit.push(key);
        program_first = true;
    }
    arguments.extend(argument_rows(input, &omit, Style::default(), width));
    tree_rows(arguments, program_first, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A code row says how it copies: its code's first column, past the shade's margin and the
    /// hang of the list it sits in, its width, and whether it continues the row above; the
    /// block's padding copies as nothing and the prose around it as text.
    #[test]
    fn a_code_block_s_rows_say_how_they_copy() {
        let source = format!("- item\n\n  ```\n  {}\n  ```\n", "x".repeat(70));
        let copies: Vec<RowCopy> = markdown_rows(&source, 40)
            .into_iter()
            .map(|(_, copy)| copy)
            .collect();
        assert_eq!(copies, vec![
            RowCopy::Text,
            RowCopy::Text,
            RowCopy::Padding,
            RowCopy::Code {
                from: 3,
                columns: 36,
                continues: false
            },
            RowCopy::Code {
                from: 3,
                columns: 34,
                continues: true
            },
            RowCopy::Padding,
        ]);
    }

    fn png_of(pixels: &image::RgbaImage) -> Vec<u8> {
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgba8(pixels.clone())
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode");
        bytes
    }

    /// An image draws in half-block cells, a top and a bottom pixel to each: two opaque pixels are
    /// `▀` over the bottom's color, a transparent half leaves the terminal's own background, and an
    /// image is scaled to the width and the row cap with its proportions kept, and never grown.
    #[test]
    fn a_picture_is_drawn_in_half_blocks_scaled_to_fit() {
        let red = image::Rgba([255, 0, 0, 255]);
        let blue = image::Rgba([0, 0, 255, 255]);
        let pixels = image::RgbaImage::from_fn(2, 2, |x, y| match (x, y) {
            (0, 0) => red,
            (1, 0) => image::Rgba([0, 0, 0, 0]),
            _ => blue,
        });
        let rows = picture_rows(&Picture::from_bytes(&png_of(&pixels)), 60).expect("decoded");
        let [row] = rows.as_slice() else {
            panic!("two pixels tall is one row: {rows:?}");
        };
        let cells: Vec<(&str, Option<Color>, Option<Color>)> = row
            .spans
            .iter()
            .map(|span| (span.content.as_ref(), span.style.fg, span.style.bg))
            .collect();
        assert_eq!(cells, vec![
            (
                "▀",
                Some(Color::Rgb(255, 0, 0)),
                Some(Color::Rgb(0, 0, 255))
            ),
            ("▄", Some(Color::Rgb(0, 0, 255)), None),
        ]);
        let wide = Picture::from_bytes(&png_of(&image::RgbaImage::from_pixel(400, 100, red)));
        let rows = picture_rows(&wide, 60).expect("decoded");
        assert_eq!(rows.len(), 8, "60 by 15 pixels");
        assert!(rows.iter().all(|row| row.spans.len() == 60), "{rows:?}");
        let tall = Picture::from_bytes(&png_of(&image::RgbaImage::from_pixel(10, 400, red)));
        assert_eq!(
            picture_rows(&tall, 60).expect("decoded").len(),
            PICTURE_ROWS as usize
        );
        let mut gray = Vec::new();
        image::DynamicImage::ImageLuma8(image::GrayImage::from_pixel(400, 100, image::Luma([200])))
            .write_to(
                &mut std::io::Cursor::new(&mut gray),
                image::ImageFormat::Png,
            )
            .expect("encode");
        let rows = picture_rows(&Picture::from_bytes(&gray), 60).expect("decoded");
        assert_eq!(
            rows.len(),
            8,
            "an image with no transparency scales the same"
        );
        assert_eq!(
            rows[0].spans[0].style.fg,
            Some(Color::Rgb(200, 200, 200)),
            "{rows:?}"
        );
    }

    /// An image that records it was taken turned is drawn the way up it was taken: a red pixel
    /// left of a blue one, in an image whose orientation turns it a quarter clockwise, is red over
    /// blue.
    #[test]
    fn a_picture_is_drawn_the_way_up_its_image_records() {
        let red = image::Rgba([255, 0, 0, 255]);
        let blue = image::Rgba([0, 0, 255, 255]);
        let png = png_of(&image::RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 { red } else { blue }
        }));
        // A big-endian TIFF header and one entry: orientation (0x0112), a short, 6.
        let exif: [u8; 26] = [
            0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08, 0x00, 0x01, 0x01, 0x12, 0x00, 0x03,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let crc = |bytes: &[u8]| {
            let mut crc: u32 = 0xFFFF_FFFF;
            for byte in bytes {
                crc ^= u32::from(*byte);
                for _ in 0..8 {
                    crc = if crc & 1 == 1 {
                        (crc >> 1) ^ 0xEDB8_8320
                    } else {
                        crc >> 1
                    };
                }
            }
            !crc
        };
        let mut chunk = b"eXIf".to_vec();
        chunk.extend_from_slice(&exif);
        let mut turned = png[..33].to_vec();
        turned.extend_from_slice(&(exif.len() as u32).to_be_bytes());
        turned.extend_from_slice(&chunk);
        turned.extend_from_slice(&crc(&chunk).to_be_bytes());
        turned.extend_from_slice(&png[33..]);
        let rows = picture_rows(&Picture::from_bytes(&turned), 60).expect("decoded");
        let cells: Vec<(&str, Option<Color>, Option<Color>)> = rows
            .iter()
            .flat_map(|row| &row.spans)
            .map(|span| (span.content.as_ref(), span.style.fg, span.style.bg))
            .collect();
        assert_eq!(cells, vec![(
            "▀",
            Some(Color::Rgb(255, 0, 0)),
            Some(Color::Rgb(0, 0, 255))
        )]);
    }

    /// An image with nothing opaque enough to draw stands as its label, as one that will not
    /// decode does, rather than as rows of nothing.
    #[test]
    fn a_picture_with_nothing_to_draw_is_its_label() {
        let faint = Picture::from_bytes(&png_of(&image::RgbaImage::from_pixel(
            4,
            4,
            image::Rgba([10, 20, 30, 64]),
        )));
        assert!(picture_rows(&faint, 60).is_none());
        assert_eq!(faint.label(), "[image 4x4]");
    }

    /// An image whose pixels will not decode stands as its label, with the size its header states
    /// when it states one.
    #[test]
    fn a_picture_that_will_not_decode_is_its_label() {
        let bytes = png_of(&image::RgbaImage::from_pixel(
            1280,
            720,
            image::Rgba([1, 2, 3, 255]),
        ));
        let cut = Picture::from_bytes(&bytes[..bytes.len() / 2]);
        assert!(picture_rows(&cut, 60).is_none());
        assert_eq!(cut.label(), "[image 1280x720]");
        assert_eq!(Picture::from_bytes(b"not an image").label(), "[image]");
    }

    /// Scaling weighs each color by its opacity: an opaque color averaged with transparent pixels
    /// keeps its hue, whatever color a transparent pixel stores, black or another, when a picture
    /// is drawn and when its kept copy is shrunk, in its own pixel type, gray with opacity too.
    #[test]
    fn scaling_keeps_an_edge_against_transparency_its_color() {
        let orange = image::Rgba([250, 180, 135, 255]);
        let near = |got: &[u8], want: &[u8]| {
            got.iter()
                .zip(want)
                .all(|(got, want)| got.abs_diff(*want) <= 2)
        };
        for clear in [image::Rgba([0, 0, 0, 0]), image::Rgba([0, 0, 255, 0])] {
            let pixels = image::RgbaImage::from_fn(3, 3, |x, _| if x < 2 { orange } else { clear });
            let drawn = scaled(
                pixels.clone(),
                (1, 1),
                image::imageops::FilterType::Triangle,
            );
            let kept = shrunk(image::DynamicImage::ImageRgba8(pixels), (1, 1)).into_rgba8();
            for (how, out) in [("drawn", drawn), ("kept", kept)] {
                let pixel = out.get_pixel(0, 0);
                assert!(pixel.0[3] > 0, "{clear:?} {how}: {pixel:?}");
                assert!(
                    near(&pixel.0[..3], &orange.0[..3]),
                    "{clear:?} {how}: {pixel:?}"
                );
            }
        }
        let gray = image::LumaA([200u8, 255]);
        let pixels =
            image::GrayAlphaImage::from_fn(
                3,
                3,
                |x, _| {
                    if x < 2 { gray } else { image::LumaA([0, 0]) }
                },
            );
        let kept = shrunk(image::DynamicImage::ImageLumaA8(pixels), (1, 1));
        let image::DynamicImage::ImageLumaA8(kept) = kept else {
            panic!("shrunk in its own type: {kept:?}");
        };
        let pixel = kept.get_pixel(0, 0);
        assert!(
            pixel.0[1] > 0 && near(&pixel.0[..1], &gray.0[..1]),
            "{pixel:?}"
        );
    }

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

    /// The approval's body is the call's arguments as a tree under its row, the program first
    /// behind `$`, in the foreground: the connectors are what is dimmed, the words the user is
    /// authorizing are not.
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
        let command = serde_json::json!({"command": "make\nmake install", "timeout_ms": 5});
        assert_eq!(plain(&approval_rows("shell_execute", &command, 40)), vec![
            "  $ make",
            "  │ make install",
            "  └ timeout_ms: 5"
        ]);
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

    /// A code block is drawn as written whatever opens it, behind its shade's margin: a `#` line
    /// in a tilde fence or in an indented block is code, not a heading owed a blank row.
    #[test]
    fn a_code_block_s_rows_are_drawn_as_written() {
        assert_eq!(
            plain(&markdown_lines("~~~sh\n# comment\necho hi\n~~~\n", 40)),
            vec!["", " # comment", " echo hi", ""]
        );
        assert_eq!(
            plain(&markdown_lines("    # indented\n    echo\n", 40)),
            vec!["", " # indented", " echo", ""]
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
            "",
            "    echo hi",
            "",
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
    /// A fenced row wider than the width folds inside its shade's margins with every character
    /// kept: a code line's spaces are content, so the fold is at a column and never at a space,
    /// and the indentation stays on the first row.
    #[test]
    fn a_fenced_row_wider_than_the_width_folds_at_a_column() {
        let source = format!("```bash\n    --flag={}\n```\n", "x".repeat(100));
        for lines in [markdown_lines(&source, 60), thinking_lines(&source, 60)] {
            let rows = plain(&lines);
            let [above, first, second, below] = rows.as_slice() else {
                panic!("two rows between the shade's padding: {rows:?}");
            };
            assert!(above.is_empty() && below.is_empty(), "{rows:?}");
            assert_eq!(first.chars().count(), 59, "{rows:?}");
            assert!(first.starts_with("     --flag=xxx"), "{rows:?}");
            assert_eq!(
                [first, second]
                    .map(|row| row.strip_prefix(' ').unwrap_or(row))
                    .concat(),
                format!("    --flag={}", "x".repeat(100))
            );
        }
    }

    /// A tab is four columns on every surface the screen draws, since the screen hands cells to
    /// the terminal and a tab among them is dropped with the indentation it carried: a fenced
    /// row, prose, a command, shell output, a diff line and plain text.
    #[test]
    fn a_tab_is_four_columns_on_every_surface() {
        assert_eq!(plain(&markdown_lines("```go\n\tx := 1\n```\n", 60)), vec![
            "",
            "     x := 1",
            ""
        ]);
        assert_eq!(plain(&thinking_lines("```\n\ty\n```\n", 60)), vec![
            "", "     y", ""
        ]);
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
        let diff = plain(&change_rows(
            &crate::conversation::Change::between(None, Some("\tnew\n")),
            60,
        ));
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
        assert_eq!(&rows[code - 3..=code + 3], &[
            "Before.",
            "",
            "",
            " let x = 1;",
            "",
            "",
            "After."
        ]);
        for (index, line) in lines.iter().enumerate() {
            let shaded = line
                .spans
                .iter()
                .all(|span| span.style.bg == Some(CODE_BACKGROUND));
            if (code - 1..=code + 1).contains(&index) {
                assert!(
                    shaded
                        && line
                            .spans
                            .iter()
                            .map(|span| display_width(&span.content))
                            .sum::<usize>()
                            == 60,
                    "the block and a row above and below it are on the shade across the width: \
                     {line:?}"
                );
            } else {
                assert!(
                    line.spans.iter().all(|span| span.style.bg.is_none()),
                    "the prose around it is not: {line:?}"
                );
            }
        }
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
            COMPOSER_BAR,
            ACT_GLYPH,
            WARNING_GLYPH,
            ERROR_GLYPH,
            FOLDED_GLYPH,
            OPENED_GLYPH,
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
    /// answer: the terminal's dark gray, and on a code block's shade the gray it does not swallow.
    #[test]
    fn reasoning_is_gray_throughout() {
        let lines = thinking_lines("Some **thought**.\n\n```rust\nlet x = 1;\n```\n", 60);
        for line in &lines {
            for span in &line.spans {
                if span.content.trim().is_empty() {
                    continue;
                }
                let gray = match span.style.bg {
                    Some(CODE_BACKGROUND) => DIM_ON_SHADE,
                    _ => Color::DarkGray,
                };
                assert_eq!(span.style.fg, Some(gray), "{line:?}");
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

    /// A listing's row wraps under the column the break falls in, a row with no column under its
    /// own indentation, and a hang that would leave too little room under the indentation.
    #[test]
    fn a_listing_row_wraps_under_its_own_column() {
        let wrapped = |text: &str, width: usize| {
            plain(&wrap_listing_line(
                &Line::from(Span::styled(
                    text.to_string(),
                    Style::default().fg(Color::Red),
                )),
                width,
            ))
        };
        let row =
            "  /approvals [on|off]  Pick or set whether calls above the level are asked about";
        assert_eq!(wrapped(row, 50), vec![
            "  /approvals [on|off]  Pick or set whether calls",
            "                       above the level are asked",
            "                       about",
        ]);
        assert_eq!(
            wrapped("  a row of prose with no columns in it at all", 20),
            vec!["  a row of prose", "  with no columns in", "  it at all"]
        );
        assert_eq!(wrapped(row, 30), vec![
            "  /approvals [on|off]  Pick or",
            "  set whether calls above the",
            "  level are asked about",
        ]);
        let styled = wrap_listing_line(
            &Line::from(Span::styled(row, Style::default().fg(Color::Red))),
            50,
        );
        assert!(
            styled
                .iter()
                .flat_map(|line| &line.spans)
                .all(|span| span.content.trim().is_empty() || span.style.fg == Some(Color::Red)),
            "the words keep their style: {styled:?}"
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
        assert_eq!(
            plain(&preview_rows(
                rows(&["one", "two", "three", "four"]),
                PREVIEW_ROWS
            )),
            vec!["one", "two", "three", "four"],
            "a single row past the opening shows rather than taking a row to count"
        );
        assert!(
            !plain(&preview_rows(rows(&["a", "", "b"]), PREVIEW_ROWS))
                .iter()
                .any(String::is_empty),
            "a preview never holds a blank row"
        );
        assert_eq!(
            plain(&preview_rows(
                rows(&["", " ", "after", "", "rest"]),
                PREVIEW_ROWS
            )),
            vec!["after", "… +2 lines"],
            "blank rows ahead of the result are not its first block"
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

    /// A tool row is the name in the accent, then the summary as the caller styled it; an MCP
    /// tool's server prefix is dimmed and its own name keeps the accent.
    #[test]
    fn a_tool_row_is_the_name_and_its_summary() {
        let summary = vec![Span::styled(
            "notes.txt".to_string(),
            Style::default().fg(Color::Gray),
        )];
        let row = tool_row("file_read", summary, Vec::new(), 80);
        assert_eq!(plain(std::slice::from_ref(&row)), vec![
            "• file_read notes.txt"
        ]);
        assert_eq!(
            row.spans[1].style.fg,
            Some(ACCENT),
            "the name takes the accent"
        );
        assert_eq!(row.spans[3].style.fg, Some(Color::Gray));
        assert_eq!(
            plain(&[tool_row("schedule_list", Vec::new(), Vec::new(), 80)]),
            vec!["• schedule_list"]
        );
        let mcp = tool_row("mcp__exa__web_search_exa", Vec::new(), Vec::new(), 80);
        assert_eq!(plain(std::slice::from_ref(&mcp)), vec![
            "• mcp__exa__web_search_exa"
        ]);
        assert_eq!(
            (mcp.spans[1].content.as_ref(), mcp.spans[1].style.fg),
            ("mcp__exa__", Some(Color::DarkGray)),
            "the server prefix is dimmed: {mcp:?}"
        );
        assert_eq!(
            (mcp.spans[2].content.as_ref(), mcp.spans[2].style.fg),
            ("web_search_exa", Some(ACCENT))
        );
    }

    /// A command on a tool row is its first line highlighted as code and, when it does not fit,
    /// cut at the edge with the tag still whole after it; expanded, it is a block of its own,
    /// wrapped whole.
    #[test]
    fn a_shell_command_is_highlighted_on_the_row_and_whole_under_it() {
        let command = "seq 1 5000 | grep -o 'survived' | sort | uniq -c | head -1";
        let first = command_lines(command, "bash", 1, 48).remove(0).spans;
        let row = tool_row("shell_execute", first, failure_tag(Some(2)), 48);
        let text = plain(std::slice::from_ref(&row)).remove(0);
        assert!(text.starts_with("• shell_execute seq 1 5000"), "{text:?}");
        assert!(
            text.ends_with("…  exit 2"),
            "cut at the edge, the tag after it: {text:?}"
        );
        assert_eq!(text.matches('…').count(), 1, "{text:?}");
        assert!(display_width(&text) <= 48, "{text:?}");
        assert!(
            row.spans
                .iter()
                .any(|span| span.content == "|" && span.style.fg == Some(Color::Indexed(4))),
            "the command is highlighted as code: {row:?}"
        );
        assert_eq!(
            row.spans.last().map(|span| span.style.fg),
            Some(Some(Color::Red)),
            "the exit is red: {row:?}"
        );

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
        let shown = plain(&command_lines(
            "first line\nsecond\nthird\nfourth",
            "bash",
            2,
            3,
        ));
        assert_eq!(
            shown,
            vec!["firs", "seco"],
            "only the lines asked for are highlighted, each cut to a column past the width"
        );
        let quoted = "echo \"a string that runs past the edge\" ; ls\nls -la";
        let style_of_second = |columns| -> Vec<Style> {
            command_lines(quoted, "bash", 2, columns)
                .remove(1)
                .spans
                .iter()
                .map(|span| span.style)
                .collect()
        };
        assert_eq!(
            style_of_second(10),
            style_of_second(usize::MAX),
            "a line under one cut inside a string reads as it does under the whole line"
        );
        let dimmed = dim_front(
            command_lines("cd /w && ls", "bash", 1, 40).remove(0).spans,
            9,
        );
        let gray_part: String = dimmed
            .iter()
            .filter(|span| span.style.fg == Some(Color::DarkGray))
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(
            gray_part, "cd /w && ",
            "what the front holds is dimmed and nothing after it: {dimmed:?}"
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

    /// A row on a shade is shaded to the edge, and what is dimmed in the terminal's dark gray on
    /// it is drawn in a gray the shade does not swallow.
    #[test]
    fn a_shaded_row_reaches_the_edge_in_a_gray_its_shade_does_not_swallow() {
        let row = shaded_row(
            Line::from(vec![
                Span::styled("• ".to_string(), gray()),
                Span::styled("ls".to_string(), Style::default().fg(ACCENT)),
            ]),
            RUN_BACKGROUND,
            20,
        );
        let text: String = row.spans.iter().map(|span| span.content.as_ref()).collect();
        assert_eq!(display_width(&text), 20, "{row:?}");
        assert!(
            row.spans
                .iter()
                .all(|span| span.style.bg == Some(RUN_BACKGROUND)),
            "{row:?}"
        );
        assert_eq!(row.spans[0].style.fg, Some(DIM_ON_SHADE));
        assert_eq!(
            row.spans[1].style.fg,
            Some(ACCENT),
            "a color of its own stays"
        );
        // A program's colored output ends each run with a reset, which is no background.
        let reset = Style::reset();
        let after_reset = shaded_row(
            Line::from(Span::styled("plain".to_string(), reset)),
            RUN_BACKGROUND,
            20,
        );
        assert_eq!(after_reset.spans[0].style.bg, Some(RUN_BACKGROUND));
        assert!(
            is_blank(&Line::from(Span::styled("  ".to_string(), reset))),
            "spaces after a reset are a blank line"
        );
        let picked = shaded_row(
            Line::from("step").patch_style(Style::default().add_modifier(Modifier::REVERSED)),
            RUN_BACKGROUND,
            20,
        );
        assert!(
            picked.style.add_modifier.contains(Modifier::REVERSED),
            "the line's own style, a cursor's, stays: {picked:?}"
        );
        let palette_gray = shaded_row(
            Line::from(Span::styled("dim", Style::default().fg(Color::Indexed(8)))),
            RUN_BACKGROUND,
            20,
        );
        assert_eq!(
            palette_gray.spans[0].style.fg,
            Some(DIM_ON_SHADE),
            "a program's `38;5;8` is the same gray"
        );
        let emoji = shaded_row(Line::from("ok 👍🏽 👨‍👩‍👧"), RUN_BACKGROUND, 20);
        assert_eq!(
            emoji.spans.iter().map(Span::width).sum::<usize>(),
            20,
            "the shade reaches the edge as the frame measures the row: {emoji:?}"
        );
        let tinted = cut_line(
            shaded_row(Line::from("x".repeat(30)), ADDED_BACKGROUND, 30),
            10,
        );
        assert_eq!(
            tinted.spans.last().and_then(|span| span.style.bg),
            Some(ADDED_BACKGROUND),
            "the ellipsis of a cut row keeps its tint: {tinted:?}"
        );
    }

    /// A program's reset is kept as no style at all, so it undoes neither a shade nor the
    /// cursor's reversal laid over its row.
    #[test]
    fn a_program_s_reset_is_no_style() {
        let rows = ansi_lines("\x1b[31mred\x1b[0m plain", 40);
        let plain = rows[0]
            .spans
            .iter()
            .find(|span| span.content.contains("plain"))
            .expect("the plain part");
        assert_eq!(plain.style, Style::default(), "{rows:?}");
    }

    /// A collapsed change shows its first rows and counts the lines past them, the record's
    /// left-out lines among them and the marks between hunks not; the rows shown never end on a
    /// mark, and a single line past them is shown rather than counted.
    #[test]
    fn a_collapsed_change_counts_lines_not_marks() {
        let old: String = (1..=20).map(|n| format!("line {n}\n")).collect();
        let new = old
            .replace("line 2\n", "LINE 2\n")
            .replace("line 19\n", "LINE 19\n");
        let change = crate::conversation::Change::between(Some(&old), Some(&new));
        assert_eq!(change.hunks.len(), 2);
        let rows = plain(&change_preview(&change, 40, 4));
        assert_eq!(rows.len(), 5, "{rows:?}");
        assert!(
            !rows[3].ends_with('…'),
            "no mark ends the rows shown: {rows:?}"
        );
        assert_eq!(rows[4], "… +4 lines", "the mark is not a line: {rows:?}");
        let whole = plain(&change_preview(&change, 40, 8));
        assert_eq!(
            whole.len(),
            9,
            "a single line past the rows is shown: {whole:?}"
        );
        let at_the_mark = plain(&change_preview(&change, 40, 5));
        assert_eq!(
            at_the_mark.len(),
            5,
            "the rows shown stop short of the mark that would end them: {at_the_mark:?}"
        );
        assert_eq!(at_the_mark[4], "… +4 lines");
    }

    /// A change draws as a diff: the line before, the line taken out on a red tint with its old
    /// number, the line put in on a green one with its new number, each tint across the width with
    /// the number and sign in its light shade and the text in white, the line after untinted; two
    /// changes apart from each other are two hunks set off by `…`; a file the call created is
    /// additions alone.
    #[test]
    fn a_change_draws_as_a_numbered_diff() {
        use crate::conversation::Change;
        let old = "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n";
        let new = "one\nTWO\nthree\nfour\nfive\nsix\nseven\neight\nnine\nTEN\nELEVEN\n";
        let change = Change::between(Some(old), Some(new));
        assert_eq!((change.added, change.removed), (3, 2));
        let rows = change_rows(&change, 40);
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
        let tinted = |row: &Line<'static>, tint: Color, mark: Color| {
            let [lead, text, rest] = row.spans.as_slice() else {
                panic!("a lead, the text and the tint's run to the width: {row:?}");
            };
            assert_eq!(
                row.spans
                    .iter()
                    .map(|span| display_width(&span.content))
                    .sum::<usize>(),
                40,
                "{row:?}"
            );
            assert!(
                row.spans.iter().all(|span| span.style.bg == Some(tint)),
                "{row:?}"
            );
            assert_eq!(lead.style.fg, Some(mark), "{row:?}");
            assert_eq!(text.style.fg, Some(WHITE), "{row:?}");
            assert!(rest.content.trim().is_empty(), "{row:?}");
        };
        tinted(&rows[1], REMOVED_BACKGROUND, REMOVED_MARK);
        tinted(&rows[2], ADDED_BACKGROUND, ADDED_MARK);
        assert!(
            rows[0].spans.iter().all(|span| span.style.bg.is_none()),
            "a kept line is not tinted: {:?}",
            rows[0]
        );
        let created = Change::between(None, Some("a\nb\n"));
        assert_eq!((created.added, created.removed), (2, 0));
        assert_eq!(plain(&change_rows(&created, 40)), vec!["1 + a", "2 + b"]);
        let row = tool_row(
            "file_edit",
            vec![Span::raw("hello.py".to_string())],
            change_tag(&created),
            80,
        );
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
    /// the primary parameter the built-in declares, joined when it is a list, or an MCP tool's
    /// first argument that reads as text; a blank summary is no summary.
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
                Some("0x1400"),
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
        let rows = plain(&[tool_row(forged, Vec::new(), Vec::new(), 80)]);
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].contains('\u{1b}'), "{rows:?}");
        assert!(rows[0].contains("file_read"), "{rows:?}");
    }
}
