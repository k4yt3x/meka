//! CommonMark read into the blocks the screen lays out.
//!
//! A real CommonMark parser produces the document structure, and this module reads it into a small
//! tree: paragraphs and headings of styled runs, code rows, lists of items, quotes, tables and
//! rules. The screen ([`super::tui`]) lays the tree out at a width, which is where nesting becomes
//! indentation: an item's blocks are laid out narrower by its marker and a quote's by its bar, so
//! a row that wraps stays under the text it continues. Markers never survive into a run's text,
//! because a run carries the styles they stood for.

use pulldown_cmark::{Alignment, CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

/// A run of text plus the styles covering it.
///
/// Readable outside the module because it is also what [`inline_runs`] hands back, for the callers
/// that lay text out themselves and want only the styling this module resolves.
#[derive(Clone, Default)]
pub(super) struct OwnedCompound {
    pub(super) text: String,
    pub(super) bold: bool,
    pub(super) italic: bool,
    pub(super) code: bool,
    pub(super) strikeout: bool,
}

/// One block of a document.
pub(super) enum Block {
    /// Text that wraps at the width: a paragraph, the text of an item in a tight list, or a block
    /// of raw HTML. A `\n` in a run is a hard break.
    Paragraph(Vec<OwnedCompound>),
    Heading {
        level: u8,
        runs: Vec<OwnedCompound>,
    },
    /// Rows drawn as written: a fenced or indented code block.
    Code {
        language: Option<String>,
        rows: Vec<String>,
    },
    /// `start` is an ordered list's first number, `None` for bullets. A loose list (CommonMark's
    /// term for one whose items or their blocks the source separates with blank lines) keeps a
    /// blank between its items and between the blocks of one item.
    List {
        start: Option<u64>,
        loose: bool,
        items: Vec<Vec<Block>>,
    },
    Quote(Vec<Block>),
    Table(Table),
    Rule,
}

/// A table's cells, each a list of runs, and how each column aligns them.
#[derive(Default)]
pub(super) struct Table {
    pub(super) alignments: Vec<Alignment>,
    pub(super) head: Option<Vec<Vec<OwnedCompound>>>,
    pub(super) rows: Vec<Vec<Vec<OwnedCompound>>>,
}

/// `markdown` as blocks.
pub(super) fn parse(markdown: &str) -> Vec<Block> {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    let mut builder = Builder::default();
    for event in Parser::new_ext(markdown, options) {
        builder.handle(event);
    }
    builder.finish()
}

/// The styled runs of `markdown`, with its block structure discarded.
///
/// For a caller that has already flattened its text to one line and does its own layout: what it
/// wants from this module is only the answer to "which part of this was emphasis", which is the
/// half a hand-rolled scanner gets wrong. CommonMark emphasis is context-sensitive enough that
/// `snake_case` and `2 * 3 * 4` both hinge on rules worth having a real parser for.
///
/// Blocks are concatenated in order with nothing between them, since a caller in this position has
/// no rows to separate. An ordered item keeps its number, which the text may refer to; a bullet is
/// dropped.
pub(super) fn inline_runs(markdown: &str) -> Vec<OwnedCompound> {
    fn collect(blocks: Vec<Block>, runs: &mut Vec<OwnedCompound>) {
        for block in blocks {
            match block {
                Block::Paragraph(text) | Block::Heading { runs: text, .. } => runs.extend(text),
                Block::Code { rows, .. } => runs.push(OwnedCompound {
                    text: rows.join(" "),
                    ..OwnedCompound::default()
                }),
                Block::List { start, items, .. } => {
                    for (index, item) in items.into_iter().enumerate() {
                        if let Some(first) = start {
                            runs.push(OwnedCompound {
                                text: format!("{}. ", first.saturating_add(index as u64)),
                                ..OwnedCompound::default()
                            });
                        }
                        collect(item, runs);
                    }
                }
                Block::Quote(blocks) => collect(blocks, runs),
                Block::Table(table) => {
                    runs.extend(table.head.into_iter().chain(table.rows).flatten().flatten())
                }
                Block::Rule => {}
            }
        }
    }
    let mut runs = Vec::new();
    collect(parse(markdown), &mut runs);
    runs
}

/// Nesting counters rather than booleans: `**a *b* c**` closes the inner emphasis without ending
/// the outer one, so a boolean would drop the rest of the bold run.
#[derive(Clone, Copy, Default)]
struct InlineStyle {
    bold: usize,
    italic: usize,
    code: usize,
    strikeout: usize,
}

/// A container whose events are still arriving.
enum Open {
    Quote(Vec<Block>),
    List {
        start: Option<u64>,
        loose: bool,
        items: Vec<Vec<Block>>,
    },
    Item(Vec<Block>),
}

#[derive(Default)]
struct Builder {
    /// The document's own blocks, outside every container.
    blocks: Vec<Block>,
    open: Vec<Open>,
    /// Inline runs of the text block being read.
    runs: Vec<OwnedCompound>,
    /// The level of the heading being read, while one is.
    heading: Option<u8>,
    /// The language and body of the code block being read, while one is.
    code: Option<(Option<String>, String)>,
    /// The raw HTML block being read, while one is.
    html: Option<String>,
    table: Option<TableState>,
    inline: InlineStyle,
    /// The links being read, innermost last: an image inside a link is one.
    links: Vec<OpenLink>,
}

/// A link being read: where it goes, and the text it shows so far.
struct OpenLink {
    target: String,
    shown: String,
}

#[derive(Default)]
struct TableState {
    table: Table,
    row: Vec<Vec<OwnedCompound>>,
    cell: Vec<OwnedCompound>,
}

impl Builder {
    fn handle(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => match &mut self.code {
                Some((_, body)) => body.push_str(&text),
                None => self.push_text(&text),
            },
            Event::Code(text) => {
                self.inline.code += 1;
                self.push_text(&text);
                self.inline.code -= 1;
            }
            // A soft break is a newline inside one paragraph. Joining with a space is what lets the
            // paragraph wrap to the terminal's width instead of the one the model wrote at.
            Event::SoftBreak => self.push_text(" "),
            Event::HardBreak => self.push_text("\n"),
            Event::Rule => {
                self.flush_text();
                self.push_block(Block::Rule);
            }
            Event::TaskListMarker(checked) => {
                self.push_text(if checked { "[x] " } else { "[ ] " });
            }
            // Raw text is the honest rendering of what has no terminal form, and dropping it would
            // lose content the model wrote.
            Event::Html(text) => match &mut self.html {
                Some(body) => body.push_str(&text),
                None => self.push_text(&text),
            },
            Event::InlineHtml(text) | Event::InlineMath(text) | Event::DisplayMath(text) => {
                self.push_text(&text);
            }
            Event::FootnoteReference(text) => self.push_text(&format!("[{text}]")),
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            // No flush: an item's task marker arrives ahead of the paragraph it belongs to. A
            // paragraph directly in an item is what makes its list loose.
            Tag::Paragraph => {
                if matches!(self.open.last(), Some(Open::Item(_)))
                    && let Some(Open::List { loose, .. }) = self.open.iter_mut().rev().nth(1)
                {
                    *loose = true;
                }
            }
            Tag::Heading { level, .. } => {
                self.flush_text();
                self.heading = Some(header_depth(level));
            }
            Tag::BlockQuote(_) => {
                self.flush_text();
                self.open.push(Open::Quote(Vec::new()));
            }
            Tag::CodeBlock(kind) => {
                self.flush_text();
                let language = match kind {
                    CodeBlockKind::Fenced(info) => {
                        super::parse_fence_language(&info).map(str::to_string)
                    }
                    CodeBlockKind::Indented => None,
                };
                self.code = Some((language, String::new()));
            }
            Tag::HtmlBlock => {
                self.flush_text();
                self.html = Some(String::new());
            }
            // A nested list opens while its parent item's own text is still pending.
            Tag::List(start) => {
                self.flush_text();
                self.open.push(Open::List {
                    start,
                    loose: false,
                    items: Vec::new(),
                });
            }
            Tag::Item => self.open.push(Open::Item(Vec::new())),
            Tag::Table(alignments) => {
                self.flush_text();
                self.table = Some(TableState {
                    table: Table {
                        alignments,
                        ..Table::default()
                    },
                    ..TableState::default()
                });
            }
            Tag::Emphasis => self.inline.italic += 1,
            Tag::Strong => self.inline.bold += 1,
            Tag::Strikethrough => self.inline.strikeout += 1,
            // Terminals can't follow a link, so the destination is rendered alongside the text.
            Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. } => {
                self.links.push(OpenLink {
                    target: dest_url.to_string(),
                    shown: String::new(),
                });
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => self.flush_text(),
            TagEnd::Heading(_) => {
                self.flush_text();
                self.heading = None;
            }
            TagEnd::CodeBlock => {
                if let Some((language, body)) = self.code.take() {
                    let rows = body.lines().map(str::to_string).collect();
                    self.push_block(Block::Code { language, rows });
                }
            }
            TagEnd::HtmlBlock => {
                if let Some(body) = self.html.take() {
                    let text = body.trim_end_matches('\n');
                    if !text.is_empty() {
                        self.push_block(Block::Paragraph(vec![OwnedCompound {
                            text: text.to_string(),
                            ..OwnedCompound::default()
                        }]));
                    }
                }
            }
            TagEnd::BlockQuote(_) | TagEnd::List(_) | TagEnd::Item => {
                self.flush_text();
                if let Some(open) = self.open.pop() {
                    self.close(open);
                }
            }
            TagEnd::TableHead => {
                // pulldown-cmark delimits the header row with `TableHead` itself and emits no
                // `TableRow` around it, so it is closed here or its cells merge into the first
                // body row.
                if let Some(state) = &mut self.table {
                    state.table.head = Some(std::mem::take(&mut state.row));
                }
            }
            TagEnd::TableRow => {
                if let Some(state) = &mut self.table {
                    let row = std::mem::take(&mut state.row);
                    state.table.rows.push(row);
                }
            }
            TagEnd::TableCell => {
                if let Some(state) = &mut self.table {
                    let cell = std::mem::take(&mut state.cell);
                    state.row.push(cell);
                }
            }
            TagEnd::Table => {
                if let Some(state) = self.table.take() {
                    self.push_block(Block::Table(state.table));
                }
            }
            TagEnd::Emphasis => self.inline.italic = self.inline.italic.saturating_sub(1),
            TagEnd::Strong => self.inline.bold = self.inline.bold.saturating_sub(1),
            TagEnd::Strikethrough => {
                self.inline.strikeout = self.inline.strikeout.saturating_sub(1);
            }
            TagEnd::Link | TagEnd::Image => {
                if let Some(link) = self.links.pop() {
                    let shown = link.shown.trim();
                    // An autolink's text is its destination, an email's less the scheme.
                    let already_shown =
                        shown == link.target || link.target.strip_prefix("mailto:") == Some(shown);
                    if shown.is_empty() {
                        self.push_text(&link.target);
                    } else if !already_shown && !link.target.is_empty() {
                        self.push_text(&format!(" ({})", link.target));
                    }
                }
            }
            _ => {}
        }
    }

    /// A container whose end has arrived, as the block it is, placed in the one around it.
    fn close(&mut self, open: Open) {
        match open {
            Open::Quote(blocks) => self.push_block(Block::Quote(blocks)),
            Open::List {
                start,
                loose,
                items,
            } => self.push_block(Block::List {
                start,
                loose,
                items,
            }),
            Open::Item(blocks) => match self.open.last_mut() {
                Some(Open::List { items, .. }) => items.push(blocks),
                // An item outside a list is no event pulldown-cmark emits; its blocks are kept.
                _ => {
                    for block in blocks {
                        self.push_block(block);
                    }
                }
            },
        }
    }

    fn push_block(&mut self, block: Block) {
        match self.open.last_mut() {
            Some(Open::Quote(blocks) | Open::Item(blocks)) => blocks.push(block),
            Some(Open::List { items, .. }) => items.push(vec![block]),
            None => self.blocks.push(block),
        }
    }

    /// The runs inline text goes to: the cell being read, or the text block's.
    fn active_runs(&mut self) -> &mut Vec<OwnedCompound> {
        match &mut self.table {
            Some(state) => &mut state.cell,
            None => &mut self.runs,
        }
    }

    fn push_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        for link in &mut self.links {
            link.shown.push_str(text);
        }
        let inline = self.inline;
        push_run(self.active_runs(), text, &inline);
    }

    /// The pending runs as the paragraph or heading they are.
    fn flush_text(&mut self) {
        if self.runs.is_empty() {
            return;
        }
        let runs = std::mem::take(&mut self.runs);
        let block = match self.heading {
            Some(level) => Block::Heading { level, runs },
            None => Block::Paragraph(runs),
        };
        self.push_block(block);
    }

    fn finish(mut self) -> Vec<Block> {
        self.flush_text();
        self.end(TagEnd::CodeBlock);
        while let Some(open) = self.open.pop() {
            self.close(open);
        }
        self.blocks
    }
}

/// Append `text` to `runs`, merging into the previous run when the styles match so a paragraph
/// doesn't become one run per event.
fn push_run(runs: &mut Vec<OwnedCompound>, text: &str, inline: &InlineStyle) {
    let bold = inline.bold > 0;
    let italic = inline.italic > 0;
    let code = inline.code > 0;
    let strikeout = inline.strikeout > 0;
    if let Some(last) = runs.last_mut()
        && last.bold == bold
        && last.italic == italic
        && last.code == code
        && last.strikeout == strikeout
    {
        last.text.push_str(text);
        return;
    }
    runs.push(OwnedCompound {
        text: text.to_string(),
        bold,
        italic,
        code,
        strikeout,
    });
}

fn header_depth(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}
