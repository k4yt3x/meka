//! What the terminal UI knows: the transcript as a list of items, built from the agent's events
//! and the host's own actions, with the view state laid over it.
//!
//! An item is a fact and never changes meaning once added; the one exception is the item still
//! being streamed, which grows until its turn moves on. Whether a thinking block or a tool call is
//! expanded is view state, kept beside the item rather than in it, so the two global toggles and
//! a click share one place to read and write.

use uuid::Uuid;

use crate::{
    checklist::ChecklistItem,
    conversation::{ContentBlock, Role, ToolResultContent},
    frontend::{FrontendEvent, NoticeLevel},
    stats::TokenUsage,
    store::inbox::InboxClass,
    text::sanitize_stream_text,
};

/// Bytes of a tool's output or result kept for display. Shell output can run to the 64 MiB bound
/// the tool itself imposes, and a transcript holding every byte of it would be as slow to draw as
/// it is pointless to read; the tail is what a person looks at.
const TOOL_TEXT_KEEP_BYTES: usize = 64 * 1024;

/// What the UI starts from, taken from the config once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Settings {
    pub(crate) show_session_id_on_create: bool,
    pub(crate) show_token_usage: bool,
}

/// What a call changed on disk, as the tool reported it: the file, what it held before (`None`
/// for a file the call created), and what it holds now. The screen draws the two as a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileChange {
    pub(crate) path: String,
    pub(crate) old: Option<String>,
    pub(crate) new: String,
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
        change: Option<FileChange>,
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
    Usage(TokenUsage),
    /// An input image, which the terminal cannot draw.
    Image,
    /// What a session with nothing in its log opens on: meka's version, the profile the session
    /// runs on with its model, and the directory it opened in. Gone once the conversation starts;
    /// see [`Self::keeps_banner`].
    Banner {
        version: String,
        profile: String,
        model: Option<String>,
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
            text: "interrupted".to_string(),
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
        self.tail[index].id
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
        let events: Vec<&crate::conversation::Event> =
            page.rows.iter().map(|row| &row.event).collect();
        let mut entries = self.build_entries(&events, false);
        self.attach_orphans(&mut entries);
        self.pages.push_front(Page {
            first_row,
            last_row,
            entries: entries.len(),
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
        let events: Vec<&crate::conversation::Event> =
            page.rows.iter().map(|row| &row.event).collect();
        let entries = self.build_entries(&events, true);
        self.pages.push_back(Page {
            first_row,
            last_row,
            entries: entries.len(),
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

    /// Mark the prompt items the loop has read.
    pub(crate) fn mark_delivered(&mut self, ids: &[Uuid]) {
        for index in 0..self.tail.len() {
            if let Item::Prompt {
                inbox_id: Some(id),
                delivered,
                ..
            } = &mut self.tail[index].item
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
            FrontendEvent::TurnStarted { .. } | FrontendEvent::TurnFinished => {
                self.close_streams();
            }
            // The host reports how a turn ended from the result it holds; the event is for a
            // frontend with no result in hand.
            FrontendEvent::TurnEnded { .. } => {}
            // A typed prompt is never withdrawn here, and a scheduled fire's withdrawal is the
            // schedule's own bookkeeping.
            FrontendEvent::PromptWithdrawn => {}
            FrontendEvent::InboxDelivered { item_ids } => self.mark_delivered(&item_ids),
            // The words are the model's to read; the stage direction that follows is what the
            // person sees of a nudge.
            FrontendEvent::Nudged { .. } => {}
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
                        // A call that failed changed nothing, whatever it reports.
                        if !is_error
                            && let Some(crate::frontend::ToolOutputMetadata::Diff {
                                path,
                                old_text,
                                new_text,
                            }) = metadata
                        {
                            *change = Some(FileChange {
                                path: path.display().to_string(),
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
            FrontendEvent::TokenUsage(usage) => {
                if self.settings.show_token_usage && !usage.is_empty() {
                    self.close_streams();
                    self.push(Item::Usage(usage));
                }
            }
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

    /// The items a run of log events shows, the way a resume shows them: what each side said, the
    /// calls the model made with their results, the stage directions the live run showed, and a
    /// row for each compaction and rewind. A result whose call is not in the run is given to the
    /// call in the window when `window_holds_older` (a page read below it), and otherwise kept
    /// for the older page the call is on: a call precedes its result, so a page read above the
    /// window cannot owe one to it, and a repeated call id must not take a newer call's result.
    fn build_entries(
        &mut self,
        events: &[&crate::conversation::Event],
        window_holds_older: bool,
    ) -> Vec<Entry> {
        use crate::conversation::Event;
        let mut out: Vec<Entry> = Vec::new();
        let find_call = |out: &[Entry], id: &str| {
            out.iter().rposition(
                |entry| matches!(&entry.item, Item::ToolCall { id: call, .. } if call == id),
            )
        };
        for event in events {
            let message = match event {
                Event::Append(message) => message,
                Event::CompactBoundary { .. } => {
                    let entry = self.entry(Item::note("context compacted"));
                    out.push(entry);
                    continue;
                }
                Event::Repair {
                    replaced_count,
                    messages,
                } if messages.is_empty() => {
                    let entry = self.entry(Item::note(format!(
                        "rewound {replaced_count} message{}",
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
                                if let Item::ToolCall {
                                    state,
                                    result,
                                    result_images,
                                    change,
                                    ..
                                } = &mut out[index].item
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
        out
    }
}

/// The rows a page spans, or `None` for an empty one.
fn page_span(page: &crate::store::LogPage) -> Option<(i64, i64)> {
    Some((page.rows.first()?.id, page.rows.last()?.id))
}

/// The readable part of a tool result, tail-capped, and how many images it carried.
fn result_text(content: &[ToolResultContent]) -> (String, usize) {
    let mut text = String::new();
    let mut images = 0;
    for part in content {
        match part {
            ToolResultContent::Text { text: part } => {
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
/// by the two. A write's old content was never recorded, so a write stands for no change here.
fn change_from_input(name: &str, input: &serde_json::Value) -> Option<FileChange> {
    let &[old, new] = crate::tools::builtin_change_params(name) else {
        return None;
    };
    let text = |key: &str| input.get(key)?.as_str().map(str::to_string);
    Some(FileChange {
        path: text("path")?,
        old: Some(text(old)?),
        new: text(new)?,
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
            show_token_usage: false,
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

    /// A file tool's result carries what the file held before and after; the transcript keeps
    /// the two as the call's change, and keeps nothing when the call failed.
    #[test]
    fn a_file_edit_keeps_the_change_it_reported() {
        let started = || FrontendEvent::ToolCallStarted {
            id: "call-1".into(),
            name: "file_edit".into(),
            input: serde_json::json!({"path": "/tmp/hello.py"}),
            display_summary: None,
        };
        let completed = |is_error: bool| FrontendEvent::ToolCallCompleted {
            id: "call-1".into(),
            name: "file_edit".into(),
            is_error,
            content: vec![ToolResultContent::Text {
                text: "Successfully edited".into(),
            }],
            metadata: Some(crate::frontend::ToolOutputMetadata::Diff {
                path: std::path::PathBuf::from("/tmp/hello.py"),
                old_text: Some("a\nb\n".into()),
                new_text: "a\nB\n".into(),
            }),
        };
        let mut edited = transcript();
        edited.apply(started());
        edited.apply(completed(false));
        assert!(matches!(
            items(&edited)[0],
            Item::ToolCall { change: Some(FileChange { path, old: Some(old), new, .. }), .. }
                if path == "/tmp/hello.py" && old == "a\nb\n" && new == "a\nB\n"
        ));
        let mut failed = transcript();
        failed.apply(started());
        failed.apply(completed(true));
        assert!(matches!(items(&failed)[0], Item::ToolCall {
            change: None,
            ..
        }));
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
            Some(FileChange {
                path: "/tmp/hello.py".into(),
                old: Some("a\n".into()),
                new: "b\n".into(),
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
        assert!(matches!(shown[0], Item::Notice { text, .. } if text == "context compacted"));
        assert!(matches!(shown[1], Item::Notice { text, .. } if text == "rewound 2 messages"));
    }

    /// The banner stands only while nothing but notices stand with it: the first prompt takes it
    /// down, and so does a page of the log, whichever door it came in by.
    #[test]
    fn the_banner_goes_when_the_conversation_starts() {
        let banner = || Item::Banner {
            version: "0.1.0".into(),
            profile: "work".into(),
            model: None,
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
}
