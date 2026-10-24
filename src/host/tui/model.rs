//! What the terminal UI knows: the transcript as a list of items, built from the agent's events
//! and the host's own actions, with the view state laid over it.
//!
//! An item is a fact and never changes meaning once added; the one exception is the item still
//! being streamed, which grows until its turn moves on. Whether a thinking block or a tool call is
//! expanded is view state, kept beside the item rather than in it, so the two global toggles and
//! a click share one place to read and write.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::{
    checklist::ChecklistItem,
    config::TurnEndFact,
    conversation::{ContentBlock, Role, ToolResultContent},
    frontend::{FrontendEvent, NoticeLevel},
    store::{
        LogRow,
        inbox::InboxClass,
        turns::{TurnRecord, TurnStatus},
    },
    text::sanitize_stream_text,
};

/// Bytes of a tool's output or result kept for display. Shell output can run to the 64 MiB bound
/// the tool itself imposes, and a transcript holding every byte of it would be as slow to draw as
/// it is pointless to read; the tail is what a person looks at.
const TOOL_TEXT_KEEP_BYTES: usize = 64 * 1024;

/// What the UI starts from, taken from the config once.
#[derive(Debug, Clone)]
pub(crate) struct Settings {
    pub(crate) show_session_id_on_create: bool,
    /// `[display].turn_end_summary`: the facts the row that closes a turn states, in order; empty
    /// for no row.
    pub(crate) turn_end_summary: Vec<TurnEndFact>,
}

/// What a call changed, as the tool reported it: what the entry held before (`None` when the
/// call created it) and what it holds now (`None` when the call deleted it). A file, a skill
/// file or a memory alike; the screen draws the two as a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TextChange {
    pub(crate) old: Option<String>,
    pub(crate) new: Option<String>,
}

/// Where a tool call is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolState {
    /// The model is still writing the call's arguments.
    Composing,
    Running,
    Done {
        is_error: bool,
    },
}

/// One thing on the transcript.
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "a transcript holds thousands of items at most, and boxing the tool call would cost an allocation per call to save a few hundred bytes on the rest"
)]
pub(crate) enum Item {
    /// What the user sent. `class` is set when it went to the inbox during a turn, and
    /// `delivered` says whether the loop has read it yet.
    Prompt {
        text: String,
        class: Option<InboxClass>,
        inbox_id: Option<Uuid>,
        delivered: bool,
    },
    /// The answer. While `streaming`, the tail after its last newline is still arriving.
    Text {
        text: String,
        streaming: bool,
    },
    /// Reasoning: the text when the provider shows it, the token estimate when it only counts.
    Thinking {
        text: String,
        estimate: Option<u64>,
        ended: bool,
    },
    ToolCall {
        id: String,
        name: String,
        input: Option<serde_json::Value>,
        summary: Option<String>,
        state: ToolState,
        /// What the tool has written so far, tail-capped.
        output: String,
        /// The result's text, tail-capped, once the call is done.
        result: Option<String>,
        /// Images the result carried, which the terminal cannot show.
        result_images: usize,
        /// A sub-agent's rolling record of its own calls, under the `agent_spawn` call it runs in.
        activity: String,
        /// An MCP server's progress line for the running call.
        progress: Option<String>,
        /// The file the call changed, when it reported one; drawn in place of the result's text.
        change: Option<TextChange>,
    },
    Checklist {
        items: Vec<ChecklistItem>,
    },
    /// A line said beside the conversation: a warning in its color, or at `Info` a note in the
    /// dim register of an act of the agent, which is also what the host says of its own doings
    /// (a command's outcome, a stage direction, a compaction read back from the log).
    Notice {
        level: NoticeLevel,
        text: String,
    },
    Error(String),
    /// What `!command` printed, escapes included.
    ShellOutput(String),
    SessionId {
        label: String,
        id: String,
    },
    /// The row that closes a turn: how it ended and the facts `[display].turn_end_summary` lists.
    TurnEnd(String),
    /// An input image, which the terminal cannot draw.
    Image,
    /// What a session with nothing in its log opens on: meka's version, the profile the session
    /// runs on with its model, and the directory it opened in. Gone once the conversation starts;
    /// see [`Self::keeps_banner`].
    Banner {
        version: String,
        profile: String,
        model: Option<String>,
        /// The project the session opened on, when it opened on one.
        project: Option<String>,
        directory: String,
    },
}

impl Item {
    /// Whether the banner stays with this item beside it: a line said beside the conversation
    /// does not start one, where a prompt, an answer, a call or the session's own rows do.
    fn keeps_banner(&self) -> bool {
        matches!(
            self,
            Item::Banner { .. } | Item::Notice { .. } | Item::Error(_)
        )
    }

    /// A note in the dim register: what the host says of its own doings.
    pub(crate) fn note(text: impl Into<String>) -> Self {
        Item::Notice {
            level: NoticeLevel::Info,
            text: text.into(),
        }
    }

    /// The row a stopped turn leaves. An interrupt is the user's own act, not a fault, but it
    /// ends the turn short, and a dim note is missed where the warning color is not.
    pub(crate) fn interrupted() -> Self {
        Item::Notice {
            level: crate::frontend::NoticeLevel::Warn,
            text: "Interrupted".to_string(),
        }
    }
}

/// An item with the state the view keeps about it.
#[derive(Debug, Clone)]
pub(crate) struct Entry {
    /// Stable for the life of the screen, however the list moves around it: what the viewport
    /// keys its cache and its anchor by.
    pub(crate) id: u64,
    pub(crate) item: Item,
    /// Whether a collapsible item shows its whole content. Meaningless for the rest.
    pub(crate) expanded: bool,
    /// Bumped on every change to the item or its expansion, so a cache of what it was drawn as
    /// knows when to redraw.
    pub(crate) revision: u64,
}

impl Entry {
    /// Whether the item has a collapsed and an expanded form.
    pub(crate) fn is_collapsible(&self) -> bool {
        matches!(self.item, Item::Thinking { .. } | Item::ToolCall { .. })
    }
}

/// Rows of the log held at once. A page beyond this from either end drops a page from the
/// other, so the view costs the same on a session of ten turns and one of ten thousand.
pub(crate) const HISTORY_WINDOW_ROWS: usize = 2000;
/// Rows one read of the log brings in.
pub(crate) const HISTORY_PAGE_ROWS: usize = 200;

/// One page of the log as the window holds it: the rows it covers, and how many entries they
/// became, so a page can be dropped whole.
#[derive(Debug, Clone, Copy)]
struct Page {
    first_row: i64,
    last_row: i64,
    entries: usize,
    /// The turns of the page's first and last rows, for the turn-end row a join between two pages
    /// owes when the turn changes there.
    first_turn: Option<Uuid>,
    last_turn: Option<Uuid>,
    /// Whether the page's entries open with the row closing the turn above it, or end with the
    /// row closing its own last turn: the join between two pages owes one such row, and whichever
    /// page loads second draws it unless the first already has.
    starts_with_turn_end: bool,
    ends_with_turn_end: bool,
}

/// A tool result a page held without its call, kept for the older page the call is on.
struct OrphanResult {
    text: String,
    images: usize,
    is_error: bool,
}

/// The transcript and its view state.
///
/// Two lists. The **window** is a contiguous run of the session's log, loaded in pages as the view
/// scrolls up and bounded by [`HISTORY_WINDOW_ROWS`]. The **tail** is this run: everything since
/// the row the screen opened at, always resident, the list every event lands in. The view draws
/// the window and then the tail, the tail only while the window reaches the opening row; a window
/// cut from its bottom to stay within the bound leaves a gap that scrolling down fills again.
pub(crate) struct Transcript {
    window: Vec<Entry>,
    pages: std::collections::VecDeque<Page>,
    /// Whether the log goes on above the window's first page.
    more_above: bool,
    /// Whether the window reaches the opening row, so the tail is drawn after it.
    attached: bool,
    /// The log's last row when the screen opened or reloaded; the tail begins after it, and a
    /// page is never read past it. `None` for a session with no rows yet.
    log_end: Option<i64>,
    /// Results whose calls lie on a page not yet loaded, by tool call id.
    orphan_results: std::collections::HashMap<String, OrphanResult>,
    tail: Vec<Entry>,
    settings: Settings,
    /// The session's task rows, for the direction a replayed report says; see
    /// [`Self::remember_background_tasks`].
    background_tasks: Vec<crate::store::background::BackgroundTask>,
    /// The recorded turns of the session, by id, for the row that closes each turn on a replay.
    turn_records: std::collections::HashMap<Uuid, TurnRecord>,
    /// When the running turn started, for the duration its closing row states.
    turn_started: Option<DateTime<Utc>>,
    /// The closing row of a turn that failed or was interrupted, held until the row that says so
    /// has been drawn, so the two read in that order.
    pending_turn_end: Option<Item>,
    /// What the thinking toggle last set, and what a new thinking item starts as.
    thinking_shown: bool,
    /// What the tool toggle last set, and what a new tool call starts as.
    tools_expanded: bool,
    /// The tail item the answer is streaming into, if one is open.
    open_text: Option<usize>,
    /// The tail item reasoning is streaming into, if one is open.
    open_thinking: Option<usize>,
    /// Bumped on every change, so the screen redraws only when something moved.
    revision: u64,
    next_id: u64,
}

impl Transcript {
    pub(crate) fn new(settings: Settings) -> Self {
        Self {
            window: Vec::new(),
            pages: std::collections::VecDeque::new(),
            more_above: false,
            attached: true,
            log_end: None,
            orphan_results: std::collections::HashMap::new(),
            tail: Vec::new(),
            settings,
            background_tasks: Vec::new(),
            turn_records: std::collections::HashMap::new(),
            turn_started: None,
            pending_turn_end: None,
            thinking_shown: false,
            tools_expanded: false,
            open_text: None,
            open_thinking: None,
            revision: 0,
            next_id: 0,
        }
    }

    /// The items on screen, oldest first: the window, then the tail while the window reaches it.
    pub(crate) fn drawn(&self) -> impl Iterator<Item = &Entry> {
        let tail = if self.attached { &self.tail[..] } else { &[] };
        self.window.iter().chain(tail.iter())
    }

    pub(crate) fn drawn_len(&self) -> usize {
        self.window.len() + if self.attached { self.tail.len() } else { 0 }
    }

    /// The tail alone: this run's items, in order.
    #[cfg(test)]
    pub(crate) fn tail(&self) -> &[Entry] {
        &self.tail
    }

    pub(crate) fn thinking_shown(&self) -> bool {
        self.thinking_shown
    }

    pub(crate) fn tools_expanded(&self) -> bool {
        self.tools_expanded
    }

    pub(crate) fn has_more_above(&self) -> bool {
        self.more_above
    }

    pub(crate) fn is_attached(&self) -> bool {
        self.attached
    }

    pub(crate) fn log_end(&self) -> Option<i64> {
        self.log_end
    }

    /// The first row the window holds, below which the next page up is read.
    pub(crate) fn window_floor(&self) -> Option<i64> {
        self.pages.front().map(|page| page.first_row)
    }

    /// The last row the window holds, above which the next page down is read.
    pub(crate) fn window_ceiling(&self) -> Option<i64> {
        self.pages.back().map(|page| page.last_row)
    }

    fn drawn_mut(&mut self, index: usize) -> Option<&mut Entry> {
        if index < self.window.len() {
            self.window.get_mut(index)
        } else if self.attached {
            self.tail.get_mut(index - self.window.len())
        } else {
            None
        }
    }

    fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn entry(&mut self, item: Item) -> Entry {
        let expanded = match &item {
            Item::Thinking { .. } => self.thinking_shown,
            Item::ToolCall { .. } => self.tools_expanded,
            _ => false,
        };
        self.revision += 1;
        Entry {
            id: self.next_id(),
            item,
            expanded,
            revision: self.revision,
        }
    }

    fn touch(&mut self, index: usize) {
        self.revision += 1;
        if let Some(entry) = self.tail.get_mut(index) {
            entry.revision = self.revision;
        }
    }

    /// Add an item to the tail and return its id, which is what outlives a reload: the tail's
    /// positions move when a reload keeps only part of it.
    pub(crate) fn push(&mut self, item: Item) -> u64 {
        let index = self.push_entry(item);
        let id = self.tail.get(index).map_or(0, |entry| entry.id);
        self.flush_turn_end();
        id
    }

    /// Draw the closing row of a turn that failed or was interrupted, if one is still held: after
    /// the row that says what went wrong, or when the host says the turn is over.
    pub(crate) fn flush_turn_end(&mut self) {
        if let Some(item) = self.pending_turn_end.take() {
            self.close_streams();
            self.push_entry(item);
        }
    }

    /// The session's recorded turns, which close each turn's rows on a replay.
    pub(crate) fn set_turn_records(&mut self, records: Vec<TurnRecord>) {
        self.turn_records = records
            .into_iter()
            .map(|record| (record.id, record))
            .collect();
    }

    /// The row that closes the recorded turn `turn`, or `None` while the turn is still open, when
    /// its record is unknown, or when the summary is switched off.
    fn turn_end_entry(&mut self, turn: Uuid) -> Option<Entry> {
        let record = self.turn_records.get(&turn)?;
        let status = record.status?;
        let ended_at = DateTime::parse_from_rfc3339(record.ended_at.as_deref()?)
            .ok()?
            .with_timezone(&Utc);
        let elapsed = DateTime::parse_from_rfc3339(&record.started_at)
            .ok()
            .and_then(|started| (ended_at - started.with_timezone(&Utc)).to_std().ok());
        let text = crate::render::turn_end_summary(
            &self.settings.turn_end_summary,
            status,
            ended_at,
            elapsed,
            &record.usage,
            Utc::now(),
        )?;
        Some(self.entry(Item::TurnEnd(text)))
    }

    /// Add an item to the tail and return where in the tail it landed, for the streams that
    /// append to it by position.
    fn push_entry(&mut self, item: Item) -> usize {
        let entry = self.entry(item);
        self.tail.push(entry);
        self.settle_banner();
        self.tail.len() - 1
    }

    /// The banner is the opening of a session with nothing in it: the first item that is not a
    /// line said beside the conversation takes it down, whichever door it came in by.
    fn settle_banner(&mut self) {
        if !matches!(
            self.tail.first().map(|entry| &entry.item),
            Some(Item::Banner { .. })
        ) {
            return;
        }
        let accompanied = self
            .window
            .iter()
            .chain(self.tail.iter().skip(1))
            .any(|entry| !entry.item.keeps_banner());
        if accompanied {
            self.tail.remove(0);
            self.revision += 1;
        }
    }

    /// Start over at `log_end`: the window empty and attached, and the tail down to what the
    /// log does not hold. What a reload after a rewind or a fork does before the last page is
    /// read again: the page brings this run's turns back in their stored form, so the tail keeps
    /// only what the host said on its own account, the fork's banner among it.
    pub(crate) fn reset(&mut self, log_end: Option<i64>) {
        self.window.clear();
        self.pages.clear();
        self.more_above = log_end.is_some();
        self.attached = true;
        self.log_end = log_end;
        self.orphan_results.clear();
        self.turn_records.clear();
        self.turn_started = None;
        self.pending_turn_end = None;
        self.tail.retain(|entry| {
            matches!(
                entry.item,
                Item::Banner { .. }
                    | Item::SessionId { .. }
                    | Item::Error(_)
                    | Item::Notice { .. }
                    | Item::ShellOutput(_)
            )
        });
        self.open_text = None;
        self.open_thinking = None;
        self.revision += 1;
    }

    /// Let the window go, keeping the tail attached: what returning to the bottom does when the
    /// window had been cut from it, before the last page is read again.
    pub(crate) fn drop_window(&mut self) {
        self.window.clear();
        self.pages.clear();
        self.orphan_results.clear();
        self.more_above = self.log_end.is_some();
        self.attached = true;
        self.revision += 1;
    }

    /// Where the screen opened: the log's end, which the first page is read down from.
    pub(crate) fn set_log_end(&mut self, log_end: Option<i64>) {
        self.log_end = log_end;
        self.more_above = log_end.is_some();
    }

    /// A page read upward from the window's floor (or from the log's end into an empty window),
    /// put above what the window holds. The newest pages go when the window would exceed its
    /// bound, which detaches the tail until scrolling down brings them back.
    pub(crate) fn load_page_above(&mut self, page: crate::store::LogPage) {
        self.more_above = page.more;
        let Some((first_row, last_row)) = page_span(&page) else {
            return;
        };
        // The first page after a reset is the newest, so its last turn ends with the log.
        let closes_log = self.pages.is_empty() && self.window.is_empty();
        let (mut entries, mut ends_with_turn_end) =
            self.build_entries(&page.rows, false, closes_log);
        self.attach_orphans(&mut entries);
        let (first_turn, last_turn) = edge_turns(&page);
        // The join with the page below: a turn that ends where the next page begins closes here,
        // unless that page already opens with the row.
        if let Some(below) = self.pages.front()
            && !below.starts_with_turn_end
            && let (Some(ended), Some(next)) = (last_turn, below.first_turn)
            && ended != next
            && let Some(entry) = self.turn_end_entry(ended)
        {
            entries.push(entry);
            ends_with_turn_end = true;
        }
        self.pages.push_front(Page {
            first_row,
            last_row,
            entries: entries.len(),
            first_turn,
            last_turn,
            starts_with_turn_end: false,
            ends_with_turn_end,
        });
        entries.append(&mut self.window);
        self.window = entries;
        while self.rows_held() > HISTORY_WINDOW_ROWS && self.pages.len() > 1 {
            if let Some(dropped) = self.pages.pop_back() {
                let keep = self.window.len().saturating_sub(dropped.entries);
                self.window.truncate(keep);
                self.attached = false;
            }
        }
        self.settle_banner();
        self.revision += 1;
    }

    /// A page read downward from the window's ceiling, put below what the window holds; the
    /// window reattaches to the tail when the page reaches the opening row. The oldest pages go
    /// when the window would exceed its bound.
    pub(crate) fn load_page_below(&mut self, page: crate::store::LogPage) {
        let Some((first_row, last_row)) = page_span(&page) else {
            self.attached = true;
            return;
        };
        let (mut entries, ends_with_turn_end) = self.build_entries(&page.rows, true, !page.more);
        let (first_turn, last_turn) = edge_turns(&page);
        // The join with the page above: the turn its last row belongs to ends here when this
        // page opens another, unless that page already closes it.
        let mut starts_with_turn_end = false;
        if let Some(above) = self.pages.back()
            && !above.ends_with_turn_end
            && let (Some(ended), Some(next)) = (above.last_turn, first_turn)
            && ended != next
            && let Some(entry) = self.turn_end_entry(ended)
        {
            entries.insert(0, entry);
            starts_with_turn_end = true;
        }
        self.pages.push_back(Page {
            first_row,
            last_row,
            entries: entries.len(),
            first_turn,
            last_turn,
            starts_with_turn_end,
            ends_with_turn_end,
        });
        self.window.extend(entries);
        if !page.more {
            self.attached = true;
        }
        while self.rows_held() > HISTORY_WINDOW_ROWS && self.pages.len() > 1 {
            if let Some(dropped) = self.pages.pop_front() {
                self.window.drain(..dropped.entries.min(self.window.len()));
                self.more_above = true;
            }
        }
        self.settle_banner();
        self.revision += 1;
    }

    fn rows_held(&self) -> usize {
        self.pages
            .iter()
            .map(|page| usize::try_from(page.last_row - page.first_row + 1).unwrap_or(0))
            .sum()
    }

    /// Give the calls on a freshly built older page the results a newer page held for them.
    fn attach_orphans(&mut self, entries: &mut [Entry]) {
        if self.orphan_results.is_empty() {
            return;
        }
        for entry in entries.iter_mut() {
            if let Item::ToolCall {
                id,
                result,
                result_images,
                state,
                ..
            } = &mut entry.item
                && let Some(orphan) = self.orphan_results.remove(id)
            {
                *result = Some(orphan.text);
                *result_images = orphan.images;
                *state = ToolState::Done {
                    is_error: orphan.is_error,
                };
            }
        }
    }

    /// Flip one collapsible item, by its place on screen.
    pub(crate) fn toggle(&mut self, index: usize) {
        self.revision += 1;
        let revision = self.revision;
        if let Some(entry) = self.drawn_mut(index)
            && entry.is_collapsible()
        {
            entry.expanded = !entry.expanded;
            entry.revision = revision;
        }
    }

    /// Show or hide every thinking block, and start new ones that way.
    pub(crate) fn set_thinking_shown(&mut self, shown: bool) {
        self.thinking_shown = shown;
        self.revision += 1;
        let revision = self.revision;
        for entry in self.window.iter_mut().chain(self.tail.iter_mut()) {
            if matches!(entry.item, Item::Thinking { .. }) {
                entry.expanded = shown;
                entry.revision = revision;
            }
        }
    }

    /// Expand or collapse every tool call, and start new ones that way.
    pub(crate) fn set_tools_expanded(&mut self, expanded: bool) {
        self.tools_expanded = expanded;
        self.revision += 1;
        let revision = self.revision;
        for entry in self.window.iter_mut().chain(self.tail.iter_mut()) {
            if matches!(entry.item, Item::ToolCall { .. }) {
                entry.expanded = expanded;
                entry.revision = revision;
            }
        }
    }

    /// Whether an inbox item is already drawn, by the id the screen gave it when it enqueued it.
    fn draws_inbox_item(&self, id: Uuid) -> bool {
        self.tail.iter().any(|entry| {
            matches!(&entry.item, Item::Prompt { inbox_id: Some(drawn), .. } if *drawn == id)
        })
    }

    /// Mark the prompt items the loop has read.
    pub(crate) fn mark_delivered(&mut self, ids: &[Uuid]) {
        for index in 0..self.tail.len() {
            if let Some(Entry {
                item:
                    Item::Prompt {
                        inbox_id: Some(id),
                        delivered,
                        ..
                    },
                ..
            }) = self.tail.get_mut(index)
                && ids.contains(id)
                && !*delivered
            {
                *delivered = true;
                self.touch(index);
            }
        }
    }

    /// Mark one prompt item delivered by its id, for a followup the UI itself handed to the
    /// loop; `false` when no prompt has that id any more, which a reload between the two does.
    pub(crate) fn mark_sent(&mut self, id: u64) -> bool {
        let Some(index) = self.tail.iter().position(|entry| entry.id == id) else {
            return false;
        };
        let Some(Entry {
            item: Item::Prompt { delivered, .. },
            ..
        }) = self.tail.get_mut(index)
        else {
            return false;
        };
        *delivered = true;
        self.touch(index);
        true
    }

    /// Where on screen the collapsible item before `from` is, or the last one when `from` is
    /// none.
    pub(crate) fn collapsible_before(&self, from: Option<usize>) -> Option<usize> {
        let end = from.unwrap_or(self.drawn_len());
        self.drawn()
            .enumerate()
            .take(end)
            .filter(|(_, entry)| entry.is_collapsible())
            .map(|(index, _)| index)
            .last()
    }

    /// Where on screen the collapsible item after `from` is.
    pub(crate) fn collapsible_after(&self, from: usize) -> Option<usize> {
        self.drawn()
            .enumerate()
            .skip(from + 1)
            .find(|(_, entry)| entry.is_collapsible())
            .map(|(index, _)| index)
    }

    fn close_streams(&mut self) {
        if let Some(index) = self.open_thinking.take()
            && let Some(Entry {
                item: Item::Thinking { ended, .. },
                ..
            }) = self.tail.get_mut(index)
        {
            *ended = true;
            self.touch(index);
        }
        if let Some(index) = self.open_text.take()
            && let Some(Entry {
                item: Item::Text { streaming, .. },
                ..
            }) = self.tail.get_mut(index)
        {
            *streaming = false;
            self.touch(index);
        }
    }

    fn tool_index(&self, id: &str) -> Option<usize> {
        self.tail
            .iter()
            .rposition(|entry| matches!(&entry.item, Item::ToolCall { id: call, .. } if call == id))
    }

    /// Append streamed answer text to the open answer, opening one when there is none.
    fn append_text(&mut self, delta: &str) {
        let delta = sanitize_stream_text(delta);
        if let Some(index) = self.open_thinking.take()
            && let Some(Entry {
                item: Item::Thinking { ended, .. },
                ..
            }) = self.tail.get_mut(index)
        {
            *ended = true;
            self.touch(index);
        }
        let index = match self.open_text {
            Some(index) => index,
            None => {
                // A leading newline is the provider clearing its throat, not a paragraph.
                if delta.trim_start_matches('\n').is_empty() {
                    return;
                }
                let index = self.push_entry(Item::Text {
                    text: String::new(),
                    streaming: true,
                });
                self.open_text = Some(index);
                index
            }
        };
        if let Some(Entry {
            item: Item::Text { text, .. },
            ..
        }) = self.tail.get_mut(index)
        {
            if text.is_empty() {
                text.push_str(delta.trim_start_matches('\n'));
            } else {
                text.push_str(&delta);
            }
        }
        self.touch(index);
    }

    /// Append streamed reasoning to the open thinking block, opening one when there is none.
    fn append_thinking(&mut self, delta: Option<&str>, estimate: Option<u64>) {
        if let Some(index) = self.open_text.take()
            && let Some(Entry {
                item: Item::Text { streaming, .. },
                ..
            }) = self.tail.get_mut(index)
        {
            *streaming = false;
            self.touch(index);
        }
        let index = match self.open_thinking {
            Some(index) => index,
            None => {
                let index = self.push_entry(Item::Thinking {
                    text: String::new(),
                    estimate: None,
                    ended: false,
                });
                self.open_thinking = Some(index);
                index
            }
        };
        if let Some(Entry {
            item:
                Item::Thinking {
                    text,
                    estimate: shown,
                    ..
                },
            ..
        }) = self.tail.get_mut(index)
        {
            if let Some(delta) = delta {
                text.push_str(&sanitize_stream_text(delta));
            }
            // The server's figure is not monotonic, and a counter that runs backwards reads as a
            // bug rather than as progress; real spend only accumulates, so the peak is kept.
            if let Some(estimate) = estimate {
                *shown = Some(shown.map_or(estimate, |peak| peak.max(estimate)));
            }
        }
        self.touch(index);
    }

    /// Take one event from the agent.
    pub(crate) fn apply(&mut self, event: FrontendEvent) {
        match event {
            FrontendEvent::SessionStarted { id } => {
                if self.settings.show_session_id_on_create {
                    self.push(Item::SessionId {
                        label: "Creating new session".to_string(),
                        id: id.to_string(),
                    });
                }
            }
            FrontendEvent::TurnStarted { .. } => {
                self.flush_turn_end();
                self.turn_started = Some(Utc::now());
                self.close_streams();
            }
            FrontendEvent::TurnFinished => self.close_streams(),
            // The turn's closing row. A turn that succeeded closes at once; one that failed or was
            // interrupted waits for the host's row saying so, which follows this event.
            FrontendEvent::TurnEnded { ending, .. } => {
                let ended_at = Utc::now();
                let elapsed = self
                    .turn_started
                    .take()
                    .and_then(|started| (ended_at - started).to_std().ok());
                if let Some(text) = crate::render::turn_end_summary(
                    &self.settings.turn_end_summary,
                    ending.status,
                    ended_at,
                    elapsed,
                    &ending.usage,
                    ended_at,
                ) {
                    if matches!(ending.status, TurnStatus::Succeeded) {
                        self.close_streams();
                        self.push_entry(Item::TurnEnd(text));
                    } else {
                        self.pending_turn_end = Some(Item::TurnEnd(text));
                    }
                }
            }
            // A typed prompt is never withdrawn here, and a scheduled fire's withdrawal is the
            // schedule's own bookkeeping.
            FrontendEvent::PromptWithdrawn => {}
            FrontendEvent::InboxDelivered { item_ids } => self.mark_delivered(&item_ids),
            // The words are the model's to read; the stage direction that follows is what the
            // person sees of a nudge.
            FrontendEvent::Nudged { .. } => {}
            // What reported is said as a stage direction; the words are the model's to read.
            FrontendEvent::BackgroundReported { tasks, .. } => {
                self.remember_background_tasks(tasks.clone());
                self.close_streams();
                self.push(Item::Notice {
                    level: NoticeLevel::Info,
                    text: crate::background::report_direction(&tasks),
                });
            }
            // The message a turn opens with, whichever door opened it. The person's own words are
            // drawn as they are typed and the context block is meka's; a fire and a report are
            // said as the stage directions a replay of the log says, so the two read alike.
            FrontendEvent::MessageAppended {
                message,
                outcomes,
                late,
            } => {
                self.remember_background_tasks(outcomes.clone());
                for block in &message.content {
                    let item = match block {
                        ContentBlock::Scheduled { job_id, prompt, .. } => {
                            Item::note(crate::render::fire_direction(job_id, prompt, late))
                        }
                        ContentBlock::BackgroundReport { .. } => {
                            Item::note(crate::background::report_direction(&outcomes))
                        }
                        // Words another client posted to the inbox, read by the model with this
                        // turn: drawn as a replay draws them, unless the screen drew them when it
                        // enqueued them itself; the delivery mark then reaches either.
                        ContentBlock::Inbox {
                            item_id,
                            class,
                            body,
                            ..
                        } => {
                            if item_id.is_some_and(|id| self.draws_inbox_item(id)) {
                                continue;
                            }
                            Item::Prompt {
                                text: body.trim().to_string(),
                                class: *class,
                                inbox_id: *item_id,
                                delivered: false,
                            }
                        }
                        ContentBlock::Text { .. }
                        | ContentBlock::TurnContext { .. }
                        | ContentBlock::Nudge { .. }
                        | ContentBlock::Image { .. }
                        | ContentBlock::Thinking { .. }
                        | ContentBlock::RedactedThinking { .. }
                        | ContentBlock::ToolUse { .. }
                        | ContentBlock::ToolResult { .. } => continue,
                    };
                    self.close_streams();
                    self.push(item);
                }
            }
            FrontendEvent::AssistantTextDelta(text) => self.append_text(&text),
            FrontendEvent::ThinkingProgress { estimated_tokens } => {
                self.append_thinking(None, estimated_tokens);
            }
            FrontendEvent::ThinkingDelta(text) => self.append_thinking(Some(&text), None),
            FrontendEvent::ThinkingEnded => {
                if let Some(index) = self.open_thinking.take()
                    && let Some(Entry {
                        item: Item::Thinking { ended, .. },
                        ..
                    }) = self.tail.get_mut(index)
                {
                    *ended = true;
                    self.touch(index);
                }
            }
            FrontendEvent::ThinkingBlock { content } => {
                match self.open_thinking.take() {
                    // The text is already here; this is only the end of the block.
                    Some(index) => {
                        if let Some(Entry {
                            item: Item::Thinking { text, ended, .. },
                            ..
                        }) = self.tail.get_mut(index)
                        {
                            if text.is_empty() {
                                *text = sanitize_stream_text(&content);
                            }
                            *ended = true;
                        }
                        self.touch(index);
                    }
                    None => {
                        self.close_streams();
                        self.push(Item::Thinking {
                            text: sanitize_stream_text(&content),
                            estimate: None,
                            ended: true,
                        });
                    }
                }
            }
            FrontendEvent::ToolCallComposing { id, name } => {
                self.close_streams();
                self.push(Item::ToolCall {
                    id,
                    name,
                    input: None,
                    summary: None,
                    state: ToolState::Composing,
                    output: String::new(),
                    result: None,
                    result_images: 0,
                    activity: String::new(),
                    progress: None,
                    change: None,
                });
            }
            FrontendEvent::ToolCallStarted {
                id,
                name,
                input,
                display_summary,
            } => {
                self.close_streams();
                let index = match self.tool_index(&id) {
                    Some(index) => index,
                    None => self.push_entry(Item::ToolCall {
                        id,
                        name: name.clone(),
                        input: None,
                        summary: None,
                        state: ToolState::Composing,
                        output: String::new(),
                        result: None,
                        result_images: 0,
                        activity: String::new(),
                        progress: None,
                        change: None,
                    }),
                };
                if let Some(Entry {
                    item:
                        Item::ToolCall {
                            name: shown,
                            input: held,
                            summary,
                            state,
                            ..
                        },
                    ..
                }) = self.tail.get_mut(index)
                {
                    *shown = name;
                    *held = Some(input);
                    *summary = display_summary;
                    *state = ToolState::Running;
                }
                self.touch(index);
            }
            FrontendEvent::ToolCallOutputDelta { id, chunk } => {
                if let Some(index) = self.tool_index(&id) {
                    if let Some(Entry {
                        item: Item::ToolCall { output, .. },
                        ..
                    }) = self.tail.get_mut(index)
                    {
                        output.push_str(&chunk);
                        keep_tail(output, TOOL_TEXT_KEEP_BYTES);
                    }
                    self.touch(index);
                }
            }
            FrontendEvent::ToolCallCompleted {
                id,
                is_error,
                content,
                metadata,
                ..
            } => {
                if let Some(index) = self.tool_index(&id) {
                    if let Some(Entry {
                        item:
                            Item::ToolCall {
                                state,
                                result,
                                result_images,
                                progress,
                                change,
                                ..
                            },
                        ..
                    }) = self.tail.get_mut(index)
                    {
                        let (text, images) = result_text(&content);
                        *result = Some(text);
                        *result_images = images;
                        *state = ToolState::Done { is_error };
                        *progress = None;
                        if let Some(crate::frontend::ToolOutputMetadata::Diff {
                            old_text,
                            new_text,
                            ..
                        }) = metadata
                        {
                            *change = Some(TextChange {
                                old: old_text,
                                new: new_text,
                            });
                        }
                    }
                    self.touch(index);
                }
            }
            FrontendEvent::ChecklistUpdated { items } => {
                self.close_streams();
                // The list as it stands replaces the list as it stood a moment ago, when nothing
                // came between them; a list after an answer is a new fact.
                if let Some(index) = self.tail.len().checked_sub(1)
                    && let Some(Entry {
                        item: Item::Checklist { items: held },
                        ..
                    }) = self.tail.get_mut(index)
                {
                    *held = items;
                    self.touch(index);
                } else if !items.is_empty() {
                    self.push(Item::Checklist { items });
                }
            }
            FrontendEvent::SubAgentActivity {
                tool_call_id,
                summary,
            } => {
                if let Some(index) = self.tool_index(&tool_call_id) {
                    if let Some(Entry {
                        item: Item::ToolCall { activity, .. },
                        ..
                    }) = self.tail.get_mut(index)
                    {
                        *activity = summary;
                    }
                    self.touch(index);
                }
            }
            // The turn's usage reaches the screen with its ending.
            FrontendEvent::TokenUsage(_) => {}
            // A stage direction and an advisory are one register on the screen; the level is
            // what tells a warning from a note.
            FrontendEvent::Notice(notice) => {
                self.close_streams();
                self.push(Item::Notice {
                    level: notice.level,
                    text: notice.text,
                });
            }
            FrontendEvent::McpProgress(update) => {
                let line = crate::text::sanitize_to_line(
                    &format!(
                        "{}/{} {}{}",
                        update.server_name,
                        update.tool_name,
                        match update.total {
                            Some(total) if total > 0.0 =>
                                format!("{:.0}/{:.0}", update.progress, total),
                            _ => format!("{:.0}", update.progress),
                        },
                        update
                            .message
                            .as_deref()
                            .map(|message| format!(" {message}"))
                            .unwrap_or_default()
                    ),
                    usize::MAX,
                );
                if let Some(index) = self.tail.iter().rposition(|entry| {
                    matches!(&entry.item, Item::ToolCall {
                        state: ToolState::Running,
                        ..
                    })
                }) {
                    if let Some(Entry {
                        item: Item::ToolCall { progress, .. },
                        ..
                    }) = self.tail.get_mut(index)
                    {
                        *progress = Some(line);
                    }
                    self.touch(index);
                }
            }
            // The transcript is a record of what was said, not a view of the model's window, so
            // a compaction changes nothing on it; `/compact` reports its own outcome.
            FrontendEvent::Compacted { .. } => {}
        }
    }

    /// Keep the session's task rows, so a replayed report can say which tasks reported and how
    /// they ended: the block names them by id, and the rows hold the facts. Replaces by id, so a
    /// row read again with a later page or finished live supersedes the one held.
    pub(crate) fn remember_background_tasks(
        &mut self,
        tasks: Vec<crate::store::background::BackgroundTask>,
    ) {
        for task in tasks {
            match self
                .background_tasks
                .iter_mut()
                .find(|held| held.id == task.id)
            {
                Some(held) => *held = task,
                None => self.background_tasks.push(task),
            }
        }
    }

    /// The stage direction for a replayed report: the tasks the ids name, said as the live screen
    /// said them. A converted row names a task by its short id, so a prefix matches; a task the
    /// store no longer holds leaves the direction bare.
    fn replayed_report_direction(&self, task_ids: &[String]) -> String {
        let tasks: Vec<crate::store::background::BackgroundTask> = task_ids
            .iter()
            .filter_map(|id| {
                self.background_tasks
                    .iter()
                    .find(|task| task.id.starts_with(id.as_str()))
                    .cloned()
            })
            .collect();
        if tasks.is_empty() {
            "Background task report".to_string()
        } else {
            crate::background::report_direction(&tasks)
        }
    }

    /// The items a run of log events shows, the way a resume shows them: what each side said, the
    /// calls the model made with their results, the stage directions the live run showed, and a
    /// row for each compaction and rewind. A result whose call is not in the run is given to the
    /// call in the window when `window_holds_older` (a page read below it), and otherwise kept
    /// for the older page the call is on: a call precedes its result, so a page read above the
    /// window cannot owe one to it, and a repeated call id must not take a newer call's result.
    fn build_entries(
        &mut self,
        rows: &[LogRow],
        window_holds_older: bool,
        closes_log: bool,
    ) -> (Vec<Entry>, bool) {
        use crate::conversation::Event;
        let mut out: Vec<Entry> = Vec::new();
        // The turn of the last row that named one: a turn ends where the next row names another,
        // and the last turn of the log ends with the log. A row naming no turn, written before
        // turns were recorded, changes nothing.
        let mut turn: Option<Uuid> = None;
        let find_call = |out: &[Entry], id: &str| {
            out.iter().rposition(
                |entry| matches!(&entry.item, Item::ToolCall { id: call, .. } if call == id),
            )
        };
        for row in rows {
            if let (Some(ended), Some(next)) = (turn, row.turn_id)
                && ended != next
                && let Some(entry) = self.turn_end_entry(ended)
            {
                out.push(entry);
            }
            if row.turn_id.is_some() {
                turn = row.turn_id;
            }
            let event = &row.event;
            let message = match event {
                Event::Append(message) => message,
                Event::CompactBoundary { .. } => {
                    let entry = self.entry(Item::note("Session compacted."));
                    out.push(entry);
                    continue;
                }
                Event::Repair {
                    replaced_count,
                    messages,
                } if messages.is_empty() => {
                    let entry = self.entry(Item::note(format!(
                        "Rewound {replaced_count} message{}.",
                        if *replaced_count == 1 { "" } else { "s" }
                    )));
                    out.push(entry);
                    continue;
                }
                // A repair rewrites rows the log keeps; the log is what is shown.
                Event::Repair { .. } | Event::Redact { .. } => continue,
            };
            if let Some(note) = crate::render::nudge_direction(message) {
                let entry = self.entry(Item::note(note));
                out.push(entry);
            }
            for block in &message.content {
                match block {
                    ContentBlock::Text { text } => {
                        if text.trim().is_empty() {
                            continue;
                        }
                        let item = match message.role {
                            Role::Assistant => Item::Text {
                                text: sanitize_stream_text(text),
                                streaming: false,
                            },
                            Role::User => Item::Prompt {
                                text: text.trim().to_string(),
                                class: None,
                                inbox_id: None,
                                delivered: true,
                            },
                        };
                        let entry = self.entry(item);
                        out.push(entry);
                    }
                    // The person's words, labeled as the live screen labeled them.
                    ContentBlock::Inbox {
                        item_id,
                        class,
                        body,
                        ..
                    } => {
                        let entry = self.entry(Item::Prompt {
                            text: body.trim().to_string(),
                            class: *class,
                            inbox_id: *item_id,
                            delivered: true,
                        });
                        out.push(entry);
                    }
                    // meka's words: a replay says the stage direction the live screen said.
                    ContentBlock::Scheduled { job_id, prompt, .. } => {
                        let entry = self.entry(Item::note(crate::render::fire_direction(
                            job_id, prompt, None,
                        )));
                        out.push(entry);
                    }
                    ContentBlock::BackgroundReport { task_ids, .. } => {
                        let entry =
                            self.entry(Item::note(self.replayed_report_direction(task_ids)));
                        out.push(entry);
                    }
                    // meka's own words for the turn; a replay shows what was typed.
                    ContentBlock::TurnContext { .. } | ContentBlock::Nudge { .. } => {}
                    ContentBlock::Image { .. } => {
                        let entry = self.entry(Item::Image);
                        out.push(entry);
                    }
                    ContentBlock::Thinking { thinking, .. } => {
                        if !thinking.trim().is_empty() {
                            let entry = self.entry(Item::Thinking {
                                text: sanitize_stream_text(thinking),
                                estimate: None,
                                ended: true,
                            });
                            out.push(entry);
                        }
                    }
                    ContentBlock::RedactedThinking { .. } => {
                        let entry = self.entry(Item::Thinking {
                            text: crate::conversation::REDACTED_THINKING.to_string(),
                            estimate: None,
                            ended: true,
                        });
                        out.push(entry);
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        let entry = self.entry(Item::ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: Some(input.clone()),
                            summary: None,
                            // Replayed as done without an error until its result says otherwise;
                            // a call whose result the log never recorded is one the turn lost.
                            state: ToolState::Done { is_error: false },
                            output: String::new(),
                            result: None,
                            result_images: 0,
                            activity: String::new(),
                            progress: None,
                            change: change_from_input(name, input),
                        });
                        out.push(entry);
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let (text, images) = result_text(content);
                        let index = find_call(&out, tool_use_id).or_else(|| {
                            if !window_holds_older {
                                return None;
                            }
                            self.window
                                .iter()
                                .rposition(|entry| matches!(&entry.item, Item::ToolCall { id, .. } if id == tool_use_id))
                                .map(|index| index + out.len())
                        });
                        match index {
                            Some(index) if index < out.len() => {
                                if let Some(Entry {
                                    item:
                                        Item::ToolCall {
                                            state,
                                            result,
                                            result_images,
                                            change,
                                            ..
                                        },
                                    ..
                                }) = out.get_mut(index)
                                {
                                    *result = Some(text);
                                    *result_images = images;
                                    *state = ToolState::Done {
                                        is_error: *is_error,
                                    };
                                    if *is_error {
                                        *change = None;
                                    }
                                }
                            }
                            Some(index) => {
                                let index = index - out.len();
                                self.revision += 1;
                                let revision = self.revision;
                                if let Some(Entry {
                                    item:
                                        Item::ToolCall {
                                            state,
                                            result,
                                            result_images,
                                            change,
                                            ..
                                        },
                                    revision: held,
                                    ..
                                }) = self.window.get_mut(index)
                                {
                                    *result = Some(text);
                                    *result_images = images;
                                    *state = ToolState::Done {
                                        is_error: *is_error,
                                    };
                                    if *is_error {
                                        *change = None;
                                    }
                                    *held = revision;
                                }
                            }
                            None => {
                                self.orphan_results
                                    .insert(tool_use_id.clone(), OrphanResult {
                                        text,
                                        images,
                                        is_error: *is_error,
                                    });
                            }
                        }
                    }
                }
            }
        }
        let mut ends_with_turn_end = false;
        if closes_log
            && let Some(ended) = turn
            && let Some(entry) = self.turn_end_entry(ended)
        {
            out.push(entry);
            ends_with_turn_end = true;
        }
        (out, ends_with_turn_end)
    }
}

/// The rows a page spans, or `None` for an empty one.
fn page_span(page: &crate::store::LogPage) -> Option<(i64, i64)> {
    Some((page.rows.first()?.id, page.rows.last()?.id))
}

/// The turns of a page's first and last rows that name one.
fn edge_turns(page: &crate::store::LogPage) -> (Option<Uuid>, Option<Uuid>) {
    (
        page.rows.iter().find_map(|row| row.turn_id),
        page.rows.iter().rev().find_map(|row| row.turn_id),
    )
}

/// The readable part of a tool result, tail-capped, and how many images it carried.
fn result_text(content: &[ToolResultContent]) -> (String, usize) {
    let mut text = String::new();
    let mut images = 0;
    for part in content {
        match part {
            ToolResultContent::Text { text: part }
            | ToolResultContent::ToolReference { tool_name: part } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(part);
            }
            ToolResultContent::Image { .. } => images += 1,
        }
    }
    keep_tail(&mut text, TOOL_TEXT_KEEP_BYTES);
    (text, images)
}

/// Cut `text` to its last `keep` bytes, on a character boundary.
pub(super) fn keep_tail(text: &mut String, keep: usize) {
    if text.len() <= keep {
        return;
    }
    let mut cut = text.len() - keep;
    while !text.is_char_boundary(cut) {
        cut += 1;
    }
    text.drain(..cut);
}

/// The change a replayed call stands for, from its input alone, since the log keeps no result
/// metadata: a tool declared to replace one parameter's text with another's is wholly described
/// by the two. A write's old content was never recorded, nor a deleted entry's, so those stand
/// for no change here.
fn change_from_input(name: &str, input: &serde_json::Value) -> Option<TextChange> {
    let &[old, new] = crate::tools::builtin_change_params(name) else {
        return None;
    };
    let text = |key: &str| input.get(key)?.as_str().map(str::to_string);
    Some(TextChange {
        old: Some(text(old)?),
        new: Some(text(new)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        conversation::{Event, Message},
        frontend::Notice,
        store::{LogPage, LogRow},
    };

    fn transcript() -> Transcript {
        Transcript::new(Settings {
            show_session_id_on_create: false,
            turn_end_summary: TurnEndFact::DEFAULT_SUMMARY.to_vec(),
        })
    }

    fn items(transcript: &Transcript) -> Vec<&Item> {
        transcript.drawn().map(|entry| &entry.item).collect()
    }

    fn page(events: Vec<Event>, first_row: i64, more: bool) -> LogPage {
        LogPage {
            rows: events
                .into_iter()
                .enumerate()
                .map(|(offset, event)| LogRow {
                    id: first_row + offset as i64,
                    turn_id: None,
                    event,
                })
                .collect(),
            more,
        }
    }

    fn said(words: &str) -> Event {
        Event::Append(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: words.to_string(),
            }],
        })
    }

    /// Deltas of one answer land in one item, and reasoning that arrives mid-answer never joins
    /// it: the answer resumes in an item of its own.
    #[test]
    fn an_answer_and_its_reasoning_never_share_an_item() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::AssistantTextDelta("part ".into()));
        transcript.apply(FrontendEvent::AssistantTextDelta("one".into()));
        transcript.apply(FrontendEvent::ThinkingDelta("second thoughts".into()));
        transcript.apply(FrontendEvent::AssistantTextDelta("part two".into()));
        let shown = items(&transcript);
        assert_eq!(shown.len(), 3, "{shown:?}");
        assert_eq!(shown[0], &Item::Text {
            text: "part one".into(),
            streaming: false,
        });
        assert!(
            matches!(shown[1], Item::Thinking { text, ended: true, .. } if text == "second thoughts")
        );
        assert_eq!(shown[2], &Item::Text {
            text: "part two".into(),
            streaming: true,
        });
    }

    /// A tool call is one item from the moment the model starts writing it to its result.
    #[test]
    fn a_tool_call_is_one_item_from_composing_to_its_result() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::ToolCallComposing {
            id: "call-1".into(),
            name: "shell_execute".into(),
        });
        transcript.apply(FrontendEvent::ToolCallStarted {
            id: "call-1".into(),
            name: "shell_execute".into(),
            input: serde_json::json!({"command": "ls"}),
            display_summary: Some("ls".into()),
        });
        transcript.apply(FrontendEvent::ToolCallOutputDelta {
            id: "call-1".into(),
            chunk: "a\nb\n".into(),
        });
        transcript.apply(FrontendEvent::ToolCallCompleted {
            id: "call-1".into(),
            name: "shell_execute".into(),
            is_error: false,
            content: vec![ToolResultContent::Text {
                text: "a\nb\n".into(),
            }],
            metadata: None,
        });
        let shown = items(&transcript);
        assert_eq!(shown.len(), 1, "{shown:?}");
        assert!(matches!(
            shown[0],
            Item::ToolCall { state: ToolState::Done { is_error: false }, output, result: Some(result), summary: Some(summary), .. }
                if output == "a\nb\n" && result == "a\nb\n" && summary == "ls"
        ));
    }

    /// A tool's result carries what the entry held before and after; the transcript keeps the
    /// two as the call's change, a deleted entry's with no after.
    #[test]
    fn a_call_keeps_the_change_it_reported() {
        let started = |name: &str| FrontendEvent::ToolCallStarted {
            id: "call-1".into(),
            name: name.into(),
            input: serde_json::json!({"path": "/tmp/hello.py"}),
            display_summary: None,
        };
        let completed = |name: &str, new_text: Option<&str>| FrontendEvent::ToolCallCompleted {
            id: "call-1".into(),
            name: name.into(),
            is_error: false,
            content: vec![ToolResultContent::Text {
                text: "done".into(),
            }],
            metadata: Some(crate::frontend::ToolOutputMetadata::Diff {
                path: Some(std::path::PathBuf::from("/tmp/hello.py")),
                old_text: Some("a\nb\n".into()),
                new_text: new_text.map(str::to_string),
            }),
        };
        let mut edited = transcript();
        edited.apply(started("file_edit"));
        edited.apply(completed("file_edit", Some("a\nB\n")));
        assert!(matches!(
            items(&edited)[0],
            Item::ToolCall { change: Some(TextChange { old: Some(old), new: Some(new) }), .. }
                if old == "a\nb\n" && new == "a\nB\n"
        ));
        let mut deleted = transcript();
        deleted.apply(started("skill_delete"));
        deleted.apply(completed("skill_delete", None));
        assert!(matches!(
            items(&deleted)[0],
            Item::ToolCall { change: Some(TextChange { old: Some(old), new: None }), .. }
                if old == "a\nb\n"
        ));
    }

    /// A prompt posted to the inbox stays pending until the loop names its id.
    #[test]
    fn a_steer_is_pending_until_the_loop_reads_it() {
        let mut transcript = transcript();
        let id = Uuid::new_v4();
        transcript.push(Item::Prompt {
            text: "also check the logs".into(),
            class: Some(InboxClass::Steer),
            inbox_id: Some(id),
            delivered: false,
        });
        transcript.apply(FrontendEvent::InboxDelivered {
            item_ids: vec![Uuid::new_v4()],
        });
        assert!(matches!(items(&transcript)[0], Item::Prompt {
            delivered: false,
            ..
        }));
        transcript.apply(FrontendEvent::InboxDelivered { item_ids: vec![id] });
        assert!(matches!(items(&transcript)[0], Item::Prompt {
            delivered: true,
            ..
        }));
    }

    /// Two checklist updates with nothing between them are one item, the latest; one after an
    /// answer is a new fact.
    #[test]
    fn a_checklist_update_replaces_the_one_it_follows() {
        let mut transcript = transcript();
        let item = |text: &str| ChecklistItem {
            id: 1,
            text: text.into(),
            status: crate::checklist::ChecklistStatus::Pending,
            reason: None,
            task: None,
        };
        transcript.apply(FrontendEvent::ChecklistUpdated {
            items: vec![item("first")],
        });
        transcript.apply(FrontendEvent::ChecklistUpdated {
            items: vec![item("second")],
        });
        assert_eq!(transcript.drawn_len(), 1);
        assert!(
            matches!(items(&transcript)[0], Item::Checklist { items } if items[0].text == "second")
        );
        transcript.apply(FrontendEvent::AssistantTextDelta("done".into()));
        transcript.apply(FrontendEvent::ChecklistUpdated {
            items: vec![item("third")],
        });
        assert_eq!(transcript.drawn_len(), 3);
    }

    /// A stage direction and an advisory are told apart by the notice's kind, not its words.
    #[test]
    fn a_notice_lands_in_its_register() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::Notice(Notice::warn("the provider retried")));
        transcript.apply(FrontendEvent::Notice(Notice::checklist_nudged(2, 1, 3)));
        let shown = items(&transcript);
        assert!(
            matches!(shown[0], Item::Notice {
                level: NoticeLevel::Warn,
                ..
            }),
            "{shown:?}"
        );
        assert!(
            matches!(shown[1], Item::Notice {
                level: NoticeLevel::Info,
                ..
            }),
            "{shown:?}"
        );
    }

    /// The token estimate never runs backwards, and a block that opens anew starts its own count.
    #[test]
    fn a_thinking_estimate_holds_its_peak() {
        let mut transcript = transcript();
        for estimate in [100, 150, 100] {
            transcript.apply(FrontendEvent::ThinkingProgress {
                estimated_tokens: Some(estimate),
            });
        }
        assert!(matches!(items(&transcript)[0], Item::Thinking {
            estimate: Some(150),
            ..
        }));
        transcript.apply(FrontendEvent::ThinkingEnded);
        transcript.apply(FrontendEvent::ThinkingProgress {
            estimated_tokens: Some(20),
        });
        let shown = items(&transcript);
        assert_eq!(shown.len(), 2, "{shown:?}");
        assert!(matches!(shown[1], Item::Thinking {
            estimate: Some(20),
            ..
        }));
    }

    /// A replayed `file_edit` stands for the change its input describes, since the log keeps no
    /// diff metadata; a failed one stands for none, and a write, whose old content was never
    /// recorded, for none either.
    #[test]
    fn a_replayed_edit_derives_its_change_from_the_call() {
        let replay = |name: &str, input: serde_json::Value, is_error: bool| {
            let mut transcript = transcript();
            transcript.load_page_above(page(
                vec![
                    Event::Append(Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolUse {
                            id: "call-1".into(),
                            name: name.into(),
                            input,
                        }],
                    }),
                    Event::Append(Message {
                        role: Role::User,
                        content: vec![ContentBlock::ToolResult {
                            tool_use_id: "call-1".into(),
                            content: vec![ToolResultContent::Text {
                                text: "done".into(),
                            }],
                            is_error,
                        }],
                    }),
                ],
                1,
                false,
            ));
            match items(&transcript).into_iter().next() {
                Some(Item::ToolCall { change, .. }) => change.clone(),
                other => panic!("a tool call: {other:?}"),
            }
        };
        let edit = serde_json::json!({
            "path": "/tmp/hello.py",
            "old_string": "a\n",
            "new_string": "b\n"
        });
        assert_eq!(
            replay("file_edit", edit.clone(), false),
            Some(TextChange {
                old: Some("a\n".into()),
                new: Some("b\n".into()),
            })
        );
        assert_eq!(
            replay("file_edit", edit, true),
            None,
            "a failed edit changed nothing"
        );
        assert_eq!(
            replay(
                "file_write",
                serde_json::json!({"path": "/tmp/x", "content": "all"}),
                false
            ),
            None,
            "a write's old content is not on record"
        );
        assert_eq!(
            replay(
                "memory_write",
                serde_json::json!({"name": "note", "body": "all"}),
                false
            ),
            None,
            "a memory's old text is not on record either"
        );
    }

    /// A stored conversation replays as the items the live run would have shown, with each call's
    /// result attached to it.
    #[test]
    fn a_stored_conversation_replays_as_items() {
        let mut transcript = transcript();
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::TurnContext {
                        text: "[context]".into(),
                    },
                    ContentBlock::Text {
                        text: "list the files".into(),
                    },
                ],
            },
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "ls will do".into(),
                        opaque: None,
                    },
                    ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "shell_execute".into(),
                        input: serde_json::json!({"command": "ls"}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call-1".into(),
                    content: vec![ToolResultContent::Text { text: "a b".into() }],
                    is_error: false,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "Two files.".into(),
                }],
            },
        ];
        transcript.load_page_above(page(
            messages.into_iter().map(Event::Append).collect(),
            1,
            false,
        ));
        let shown = items(&transcript);
        assert_eq!(shown.len(), 4, "{shown:?}");
        assert!(matches!(shown[0], Item::Prompt { text, .. } if text == "list the files"));
        assert!(matches!(shown[1], Item::Thinking { ended: true, .. }));
        assert!(
            matches!(shown[2], Item::ToolCall { result: Some(result), state: ToolState::Done { is_error: false }, .. } if result == "a b")
        );
        assert!(matches!(shown[3], Item::Text { text, streaming: false } if text == "Two files."));
    }

    /// The global toggles move every item of their kind and decide what a new one starts as.
    #[test]
    fn a_global_toggle_moves_every_item_of_its_kind() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::ThinkingBlock {
            content: "one".into(),
        });
        assert!(!transcript.tail()[0].expanded);
        transcript.set_thinking_shown(true);
        assert!(transcript.tail()[0].expanded);
        transcript.apply(FrontendEvent::ThinkingBlock {
            content: "two".into(),
        });
        assert!(
            transcript.tail()[1].expanded,
            "a new block starts as the toggle says"
        );
        transcript.toggle(1);
        assert!(!transcript.tail()[1].expanded);
        transcript.toggle(0);
        assert!(!transcript.tail()[0].expanded);
    }

    /// A page read above the window goes in front of it, keeps every id the window had, and a
    /// result on the newer page reaches its call on the older one.
    #[test]
    fn a_page_above_goes_in_front_and_keeps_the_ids_below_it() {
        let mut transcript = transcript();
        transcript.set_log_end(Some(6));
        let result = Event::Append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".into(),
                content: vec![ToolResultContent::Text { text: "a b".into() }],
                is_error: false,
            }],
        });
        transcript.load_page_above(page(vec![result, said("Two files.")], 5, true));
        let before: Vec<u64> = transcript.drawn().map(|entry| entry.id).collect();
        assert_eq!(before.len(), 1, "the result waits for its call");
        assert!(transcript.has_more_above());
        assert_eq!(transcript.window_floor(), Some(5));
        let call = Event::Append(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "shell_execute".into(),
                input: serde_json::json!({"command": "ls"}),
            }],
        });
        transcript.load_page_above(page(vec![said("Earlier."), call], 3, false));
        let shown = items(&transcript);
        assert_eq!(shown.len(), 3, "{shown:?}");
        assert!(matches!(shown[0], Item::Text { text, .. } if text == "Earlier."));
        assert!(
            matches!(shown[1], Item::ToolCall { result: Some(result), .. } if result == "a b"),
            "the orphaned result found its call: {shown:?}"
        );
        let after: Vec<u64> = transcript.drawn().map(|entry| entry.id).collect();
        assert_eq!(
            &after[2..],
            &before[..],
            "the rows already shown keep their ids"
        );
        assert!(!transcript.has_more_above());
        assert_eq!(transcript.window_floor(), Some(3));
    }

    /// The window holds a bounded number of rows: a page above past the bound drops the newest
    /// page and detaches the tail; a page below brings it back; returning to the bottom lets the
    /// window go.
    #[test]
    fn the_window_is_bounded_and_the_tail_is_drawn_only_at_its_end() {
        let mut transcript = transcript();
        transcript.set_log_end(Some(10_000));
        transcript.push(Item::Text {
            text: "live".into(),
            streaming: false,
        });
        let rows = HISTORY_WINDOW_ROWS as i64;
        // Four pages of a quarter of the bound each fill the window exactly.
        let quarter = rows / 4;
        let mut first = 10_000 - quarter + 1;
        for _ in 0..4 {
            let events: Vec<Event> = (0..quarter).map(|n| said(&format!("row {n}"))).collect();
            transcript.load_page_above(page(events, first, true));
            first -= quarter;
        }
        assert!(transcript.is_attached());
        assert_eq!(
            transcript.drawn_len(),
            rows as usize + 1,
            "the window and the tail"
        );
        let events: Vec<Event> = (0..quarter).map(|n| said(&format!("older {n}"))).collect();
        transcript.load_page_above(page(events, first, true));
        assert!(
            !transcript.is_attached(),
            "the newest page went to keep the bound, taking the tail off screen"
        );
        assert_eq!(transcript.drawn_len(), rows as usize);
        assert!(
            !items(&transcript)
                .iter()
                .any(|item| matches!(item, Item::Text { text, .. } if text == "live")),
            "the tail is not drawn while the window is cut from it"
        );
        let ceiling = transcript.window_ceiling().expect("a window");
        let events: Vec<Event> = (0..quarter).map(|n| said(&format!("newer {n}"))).collect();
        transcript.load_page_below(page(events, ceiling + 1, false));
        assert!(transcript.is_attached(), "the page reached the opening row");
        assert!(
            items(&transcript)
                .iter()
                .any(|item| matches!(item, Item::Text { text, .. } if text == "live")),
            "and the tail is drawn again"
        );
        assert!(
            transcript.has_more_above(),
            "the oldest page went to keep the bound"
        );
        transcript.drop_window();
        assert_eq!(
            transcript.drawn_len(),
            1,
            "only the tail, until the last page is read again"
        );
        assert!(transcript.is_attached() && transcript.has_more_above());
    }

    /// A compaction and a rewind show as rows of their own; a repair shows nothing.
    #[test]
    fn a_boundary_and_a_rewind_are_rows_and_a_repair_is_not() {
        let mut transcript = transcript();
        transcript.set_log_end(Some(3));
        transcript.load_page_above(page(
            vec![
                Event::CompactBoundary {
                    summary: Message::user("summary"),
                    replaced_count: 4,
                    loaded_tools_snapshot: Default::default(),
                    checklist_snapshot: Default::default(),
                },
                Event::Repair {
                    replaced_count: 2,
                    messages: Vec::new(),
                },
                Event::Repair {
                    replaced_count: 1,
                    messages: vec![Message::user("fixed")],
                },
            ],
            1,
            false,
        ));
        let shown = items(&transcript);
        assert_eq!(shown.len(), 2, "{shown:?}");
        assert!(matches!(shown[0], Item::Notice { text, .. } if text == "Session compacted."));
        assert!(matches!(shown[1], Item::Notice { text, .. } if text == "Rewound 2 messages."));
    }

    /// The banner stands only while nothing but notices stand with it: the first prompt takes it
    /// down, and so does a page of the log, whichever door it came in by.
    #[test]
    fn the_banner_goes_when_the_conversation_starts() {
        let banner = || Item::Banner {
            version: "0.1.0".into(),
            profile: "work".into(),
            model: None,
            project: None,
            directory: "~".into(),
        };
        let mut transcript = transcript();
        transcript.push(banner());
        transcript.push(Item::note("sandbox: bubblewrap not found"));
        assert!(matches!(items(&transcript)[0], Item::Banner { .. }));
        transcript.push(Item::Prompt {
            text: "hi".into(),
            class: None,
            inbox_id: None,
            delivered: true,
        });
        let shown = items(&transcript);
        assert_eq!(shown.len(), 2, "{shown:#?}");
        assert!(matches!(shown[0], Item::Notice { .. }));
        assert!(matches!(shown[1], Item::Prompt { .. }));

        let mut resumed = self::tests::transcript();
        resumed.set_log_end(Some(1));
        resumed.push(banner());
        resumed.load_page_above(page(vec![said("older")], 1, false));
        assert!(
            !items(&resumed)
                .iter()
                .any(|item| matches!(item, Item::Banner { .. })),
            "a page of the log is a conversation: {:#?}",
            items(&resumed)
        );
    }

    fn with_summary(facts: Vec<TurnEndFact>) -> Transcript {
        Transcript::new(Settings {
            show_session_id_on_create: false,
            turn_end_summary: facts,
        })
    }

    fn asked(words: &str) -> Event {
        Event::Append(Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: words.to_string(),
            }],
        })
    }

    /// A page whose rows name their turns, with the ids `ids` gives them.
    fn in_turns(rows: Vec<(Uuid, Event)>, ids: &[i64], more: bool) -> LogPage {
        LogPage {
            rows: rows
                .into_iter()
                .zip(ids)
                .map(|((turn, event), id)| LogRow {
                    id: *id,
                    turn_id: Some(turn),
                    event,
                })
                .collect(),
            more,
        }
    }

    fn closed_turn(id: Uuid, status: TurnStatus) -> TurnRecord {
        TurnRecord {
            id,
            source: "screen".to_string(),
            started_at: "2026-10-12T12:00:00+00:00".to_string(),
            ended_at: Some("2026-10-12T12:19:19+00:00".to_string()),
            status: Some(status),
            stop_reason: None,
            error: None,
            reason: None,
            profile: None,
            usage: crate::store::turns::TurnUsage::default(),
        }
    }

    fn kinds(transcript: &Transcript) -> Vec<&'static str> {
        transcript
            .drawn()
            .map(|entry| match &entry.item {
                Item::Prompt { .. } => "prompt",
                Item::Text { .. } => "text",
                Item::TurnEnd(_) => "end",
                Item::Notice { .. } => "note",
                _ => "other",
            })
            .collect()
    }

    fn turn_end_texts(transcript: &Transcript) -> Vec<String> {
        transcript
            .drawn()
            .filter_map(|entry| match &entry.item {
                Item::TurnEnd(text) => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    /// A turn that succeeded closes at once with the facts the summary lists, in their order.
    #[test]
    fn a_turn_closes_with_the_row_its_summary_lists() {
        let mut transcript = with_summary(TurnEndFact::ALL.to_vec());
        let turn = Uuid::new_v4();
        transcript.apply(FrontendEvent::TurnStarted { turn_id: turn });
        transcript.apply(FrontendEvent::AssistantTextDelta("Done.".into()));
        transcript.apply(FrontendEvent::TurnFinished);
        transcript.apply(FrontendEvent::TurnEnded {
            turn_id: turn,
            ending: crate::store::turns::TurnEnding::succeeded(
                "end_turn",
                crate::store::turns::TurnUsage {
                    input_tokens: 1000,
                    output_tokens: 200,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 500,
                },
            ),
        });
        assert_eq!(kinds(&transcript), vec!["text", "end"]);
        let text = &turn_end_texts(&transcript)[0];
        assert!(text.starts_with("Finished at "), "{text}");
        assert!(text.contains(" after "), "{text}");
        assert!(text.ends_with("tokens in, 200 out, 33% cached"), "{text}");
    }

    /// A turn that was interrupted or failed closes after the row that says so, which the host
    /// draws once the agent has returned, or when the host says the turn is over.
    #[test]
    fn an_interrupted_turn_s_row_follows_the_one_that_says_so() {
        let mut transcript = with_summary(TurnEndFact::DEFAULT_SUMMARY.to_vec());
        let turn = Uuid::new_v4();
        transcript.apply(FrontendEvent::TurnStarted { turn_id: turn });
        transcript.apply(FrontendEvent::AssistantTextDelta("Half an".into()));
        transcript.apply(FrontendEvent::TurnEnded {
            turn_id: turn,
            ending: crate::store::turns::TurnEnding::canceled(
                crate::store::turns::TurnUsage::default(),
                crate::store::turns::CancelReason::Client,
            ),
        });
        assert_eq!(
            kinds(&transcript),
            vec!["text"],
            "held until the host has spoken"
        );
        transcript.push(Item::interrupted());
        assert_eq!(kinds(&transcript), vec!["text", "note", "end"]);
        assert!(
            turn_end_texts(&transcript)[0].starts_with("Interrupted at "),
            "{:?}",
            turn_end_texts(&transcript)
        );

        let mut quiet = with_summary(TurnEndFact::DEFAULT_SUMMARY.to_vec());
        quiet.apply(FrontendEvent::TurnStarted { turn_id: turn });
        quiet.apply(FrontendEvent::TurnEnded {
            turn_id: turn,
            ending: crate::store::turns::TurnEnding::failed(
                &crate::error::MekaError::Interrupted,
                crate::store::turns::TurnUsage::default(),
            ),
        });
        quiet.flush_turn_end();
        assert_eq!(kinds(&quiet), vec!["end"]);
        assert!(turn_end_texts(&quiet)[0].starts_with("Failed at "));
        quiet.flush_turn_end();
        assert_eq!(kinds(&quiet), vec!["end"], "flushing twice draws once");
    }

    #[test]
    fn no_summary_listed_is_no_row() {
        let mut transcript = with_summary(Vec::new());
        let turn = Uuid::new_v4();
        transcript.apply(FrontendEvent::TurnStarted { turn_id: turn });
        transcript.apply(FrontendEvent::AssistantTextDelta("Done.".into()));
        transcript.apply(FrontendEvent::TurnEnded {
            turn_id: turn,
            ending: crate::store::turns::TurnEnding::succeeded(
                "end_turn",
                crate::store::turns::TurnUsage::default(),
            ),
        });
        assert_eq!(kinds(&transcript), vec!["text"]);
    }

    /// A resume shows what the live screen showed: an inbox item as the prompt it is, labeled by
    /// its class and tied to its row; a scheduled fire and a background report as the stage
    /// directions the live screen said, the report's naming the tasks from the rows the page came
    /// with, or saying only that one reported when the store no longer holds them.
    #[test]
    fn a_replay_shows_what_meka_relayed_as_the_live_screen_did() {
        use crate::{
            conversation::InboxClass,
            store::background::{BackgroundTask, TaskStatus},
        };
        let finished_at = Utc::now();
        let mut transcript = transcript();
        transcript.remember_background_tasks(vec![BackgroundTask {
            id: "7f3a1c22-0000-4000-8000-000000000000".to_string(),
            session_id: Uuid::nil(),
            tool: "shell_execute".to_string(),
            label: "cargo test".to_string(),
            status: TaskStatus::Completed,
            outcome: Some("ok".to_string()),
            scratchpad_entry: None,
            started_at: finished_at - chrono::Duration::seconds(30),
            finished_at: Some(finished_at),
            announced_at: None,
            delivered_at: Some(finished_at),
            subagent_id: None,
        }]);
        let item_id = Uuid::new_v4();
        let user = |content: Vec<ContentBlock>| {
            Event::Append(Message {
                role: Role::User,
                content,
            })
        };
        let context = || ContentBlock::TurnContext {
            text: "<context/>".to_string(),
        };
        transcript.load_page_above(page(
            vec![
                user(vec![ContentBlock::Inbox {
                    item_id: Some(item_id),
                    class: Some(InboxClass::Steer),
                    source: Some("telegram".to_string()),
                    body: "also check the logs".to_string(),
                    text: "[Message from telegram, arrived now]\nalso check the logs".to_string(),
                }]),
                said("Noted."),
                user(vec![context(), ContentBlock::Scheduled {
                    job_id: "1a2b3c4d-0000-4000-8000-000000000000".to_string(),
                    prompt: "remind me to stretch".to_string(),
                    text: "[Scheduled job 1a2b3c4d fired now]\n\nremind me to stretch".to_string(),
                }]),
                said("Stretching."),
                user(vec![context(), ContentBlock::BackgroundReport {
                    task_ids: vec!["7f3a1c22-0000-4000-8000-000000000000".to_string()],
                    text: "[Background task reporting at now]\nok".to_string(),
                }]),
                said("Good."),
                user(vec![context(), ContentBlock::BackgroundReport {
                    task_ids: vec!["deadbeef".to_string()],
                    text: "[Background task reporting at now]\ngone".to_string(),
                }]),
            ],
            1,
            false,
        ));
        let shown = items(&transcript);
        assert!(
            matches!(
                shown[0],
                Item::Prompt {
                    text,
                    class: Some(InboxClass::Steer),
                    inbox_id: Some(id),
                    delivered: true,
                } if text == "also check the logs" && *id == item_id
            ),
            "an inbox item is the prompt it was, labeled: {:?}",
            shown[0]
        );
        let notes: Vec<&str> = shown
            .iter()
            .filter_map(|item| match item {
                Item::Notice { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(notes, vec![
            "Scheduled job 1a2b3c4d fired: remind me to stretch",
            "Background task 7f3a1c22 finished after 30s",
            "Background task report",
        ]);
        assert!(
            !shown.iter().any(|item| matches!(item, Item::Prompt { text, .. } if text.contains("[Scheduled") || text.contains("[Background"))),
            "nothing meka wrote is shown as a prompt: {shown:?}"
        );
    }

    /// On a replay each recorded turn closes where its rows end: inside a page where the next
    /// turn's rows begin, and at the log's end for the last one. A turn still open or unknown
    /// closes nothing.
    #[test]
    fn a_replayed_turn_closes_where_its_rows_end() {
        let (first, second, open, unknown) = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        let mut transcript = with_summary(TurnEndFact::DEFAULT_SUMMARY.to_vec());
        let mut never_closed = closed_turn(open, TurnStatus::Succeeded);
        never_closed.ended_at = None;
        never_closed.status = None;
        transcript.set_turn_records(vec![
            closed_turn(first, TurnStatus::Succeeded),
            closed_turn(second, TurnStatus::Canceled),
            never_closed,
        ]);
        transcript.load_page_above(in_turns(
            vec![
                (first, asked("one?")),
                (first, said("One.")),
                (second, asked("two?")),
                (second, said("Tw")),
                (open, asked("three?")),
                (open, said("Three.")),
                (unknown, asked("four?")),
                (unknown, said("Four.")),
            ],
            &[1, 2, 3, 4, 5, 6, 7, 8],
            false,
        ));
        assert_eq!(kinds(&transcript), vec![
            "prompt", "text", "end", "prompt", "text", "end", "prompt", "text", "prompt", "text"
        ]);
        let texts = turn_end_texts(&transcript);
        assert!(
            texts[0].starts_with("Finished on 2026-10-1") && texts[0].ends_with("after 19m 19s"),
            "{texts:?}"
        );
        assert!(
            texts[1].starts_with("Interrupted on 2026-10-1"),
            "{texts:?}"
        );
    }

    /// A turn whose rows cross a page boundary closes once, where its rows end, whichever page
    /// loads first; a turn that ends exactly at a boundary closes at the join.
    #[test]
    fn a_turn_across_two_pages_closes_once() {
        let (first, second) = (Uuid::new_v4(), Uuid::new_v4());
        let records = || {
            vec![
                closed_turn(first, TurnStatus::Succeeded),
                closed_turn(second, TurnStatus::Succeeded),
            ]
        };
        // The second turn's rows straddle the pages.
        let mut straddling = with_summary(TurnEndFact::DEFAULT_SUMMARY.to_vec());
        straddling.set_turn_records(records());
        straddling.load_page_above(in_turns(vec![(second, said("rest."))], &[5], true));
        straddling.load_page_above(in_turns(
            vec![
                (first, asked("one?")),
                (first, said("One.")),
                (second, asked("two?")),
                (second, said("The first part and the ")),
            ],
            &[1, 2, 3, 4],
            false,
        ));
        assert_eq!(kinds(&straddling), vec![
            "prompt", "text", "end", "prompt", "text", "text", "end"
        ]);
        // The first turn ends exactly where the newer page begins.
        let mut at_the_join = with_summary(TurnEndFact::DEFAULT_SUMMARY.to_vec());
        at_the_join.set_turn_records(records());
        at_the_join.load_page_above(in_turns(
            vec![(second, asked("two?")), (second, said("Two."))],
            &[3, 4],
            true,
        ));
        at_the_join.load_page_above(in_turns(
            vec![(first, asked("one?")), (first, said("One."))],
            &[1, 2],
            false,
        ));
        assert_eq!(kinds(&at_the_join), vec![
            "prompt", "text", "end", "prompt", "text", "end"
        ]);
    }

    /// A page dropped from the window's end and read back below it finds the join's closing row
    /// already drawn by the page above, and draws it no second time. Three pages, because two
    /// that do not fit the window together can never meet in it.
    #[test]
    fn a_page_read_back_below_the_window_does_not_close_a_turn_twice() {
        let (zero, first, second) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let mut transcript = with_summary(TurnEndFact::DEFAULT_SUMMARY.to_vec());
        transcript.set_turn_records(vec![
            closed_turn(zero, TurnStatus::Succeeded),
            closed_turn(first, TurnStatus::Succeeded),
            closed_turn(second, TurnStatus::Canceled),
        ]);
        let newest = || {
            in_turns(
                vec![(second, asked("two?")), (second, said("Two."))],
                &[1500, 1501],
                true,
            )
        };
        transcript.load_page_above(newest());
        transcript.load_page_above(in_turns(
            vec![(first, asked("one?")), (first, said("One."))],
            &[1000, 1001],
            true,
        ));
        // Wide enough in row ids to push the newest page out of the window.
        transcript.load_page_above(in_turns(
            vec![(zero, asked("zero?")), (zero, said("Zero."))],
            &[1, 1998],
            false,
        ));
        assert_eq!(
            kinds(&transcript),
            vec!["prompt", "text", "end", "prompt", "text", "end"],
            "the newest page is out and the middle one still closes its turn at the join"
        );
        let mut back = newest();
        back.more = false;
        transcript.load_page_below(back);
        assert_eq!(kinds(&transcript), vec![
            "prompt", "text", "end", "prompt", "text", "end"
        ]);
        let texts = turn_end_texts(&transcript);
        assert!(
            texts[0].starts_with("Finished") && texts[1].starts_with("Interrupted"),
            "the oldest page went, the middle one kept its row and the newest closed itself: {texts:?}"
        );
    }
    /// The message a turn opens with is said as the stage directions a replay of the log says:
    /// what reported, then the fire with its lateness, which only the live announcement knows.
    /// The person's words are drawn as typed and the context block is meka's, so neither says
    /// anything here.
    #[test]
    fn a_turn_s_opening_says_its_report_and_its_fire_as_directions() {
        use crate::{
            conversation::{ContentBlock, Role},
            store::background::{BackgroundTask, TaskStatus},
        };
        let finished_at = Utc::now();
        let task = BackgroundTask {
            id: "7f3a1c22-0000-4000-8000-000000000000".to_string(),
            session_id: Uuid::nil(),
            tool: "shell_execute".to_string(),
            label: "cargo test".to_string(),
            status: TaskStatus::Completed,
            outcome: Some("ok".to_string()),
            scratchpad_entry: None,
            started_at: finished_at - chrono::Duration::seconds(30),
            finished_at: Some(finished_at),
            announced_at: None,
            delivered_at: Some(finished_at),
            subagent_id: None,
        };
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::MessageAppended {
            message: Message {
                role: Role::User,
                content: vec![
                    ContentBlock::TurnContext {
                        text: "[Context]".to_string(),
                    },
                    ContentBlock::BackgroundReport {
                        task_ids: vec![task.id.clone()],
                        text: "[Background task reporting]\nok".to_string(),
                    },
                    ContentBlock::Scheduled {
                        job_id: "1a2b3c4d-0000-4000-8000-000000000000".to_string(),
                        prompt: "stretch".to_string(),
                        text: "[Scheduled job 1a2b3c4d fired]\n\nstretch".to_string(),
                    },
                    ContentBlock::Text {
                        text: "typed".to_string(),
                    },
                ],
            },
            outcomes: vec![task],
            late: Some(crate::frontend::FireLateness {
                late_by: chrono::Duration::hours(2),
                coalesced: 0,
            }),
        });
        let drawn = items(&transcript);
        let notes: Vec<&str> = drawn
            .iter()
            .filter_map(|item| match item {
                Item::Notice { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(notes, [
            "Background task 7f3a1c22 finished after 30s",
            "Scheduled job 1a2b3c4d fired 2h late, replacing 1 missed occurrence: stretch",
        ]);
        assert!(
            !drawn
                .iter()
                .any(|item| matches!(item, Item::Prompt { text, .. } if text == "typed")),
            "the words are drawn as typed, not again on their announcement: {drawn:?}"
        );
    }
    /// An inbox item another client posted is drawn when the model reads it with a turn's
    /// opening, as a replay draws it, and the delivery mark reaches it as it reaches the
    /// screen's own; one the screen enqueued itself is not drawn a second time.
    #[test]
    fn an_inbox_item_from_elsewhere_is_drawn_when_the_opening_carries_it() {
        use crate::conversation::{ContentBlock, InboxClass, Role};
        let mut transcript = transcript();
        let own = Uuid::new_v4();
        transcript.push(Item::Prompt {
            text: "mine".to_string(),
            class: Some(InboxClass::Steer),
            inbox_id: Some(own),
            delivered: false,
        });
        let foreign = Uuid::new_v4();
        let inbox = |id: Uuid, body: &str| ContentBlock::Inbox {
            item_id: Some(id),
            class: Some(InboxClass::Steer),
            source: Some("telegram".to_string()),
            body: body.to_string(),
            text: format!("[Message from telegram, arrived now]\n{body}"),
        };
        transcript.apply(FrontendEvent::MessageAppended {
            message: Message {
                role: Role::User,
                content: vec![
                    ContentBlock::TurnContext {
                        text: "[Context]".to_string(),
                    },
                    inbox(own, "mine"),
                    inbox(foreign, "from afar"),
                ],
            },
            outcomes: Vec::new(),
            late: None,
        });
        let prompts = |transcript: &Transcript| -> Vec<(String, bool)> {
            items(transcript)
                .iter()
                .filter_map(|item| match item {
                    Item::Prompt {
                        text, delivered, ..
                    } => Some((text.clone(), *delivered)),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(
            prompts(&transcript),
            [
                ("mine".to_string(), false),
                ("from afar".to_string(), false)
            ],
            "the screen's own once, the other client's drawn as its words"
        );
        transcript.apply(FrontendEvent::InboxDelivered {
            item_ids: vec![own, foreign],
        });
        assert_eq!(prompts(&transcript), [
            ("mine".to_string(), true),
            ("from afar".to_string(), true)
        ]);
    }
}
