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
    conversation::{Change, ContentBlock, ResultDisplay, Role, ToolResultContent},
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
    /// `[display].fold_steps`: whether a finished run of steps folds into one row.
    pub(crate) fold_steps: bool,
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
    /// `delivered` says whether the loop has read it yet; `ran_under` is what its bar shows.
    Prompt {
        text: String,
        class: Option<InboxClass>,
        inbox_id: Option<Uuid>,
        delivered: bool,
        ran_under: RanUnder,
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
        /// What the call changed in a file, a memory or a skill, when it reported a change;
        /// drawn in place of the result's text.
        change: Option<Change>,
        /// The code the program the call ran exited with, when it was not 0.
        exit_code: Option<i32>,
    },
    Checklist {
        items: Vec<ChecklistItem>,
    },
    /// A compaction: a boundary row that opens to what set it going and what differed from the
    /// ordinary case. The items after it whose [`Entry::compaction`] names it are its own (the
    /// checkpoint turn's steps, then the summary), shown only while it is open.
    Compaction {
        headline: String,
        departures: Vec<String>,
    },
    /// The summary a compaction replaced the conversation with, the last of its contents.
    CompactionSummary {
        text: String,
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
    /// An input image, drawn as its picture.
    Image(crate::render::tui::Picture),
    /// What a session with nothing in its log opens on: the build, and what the agent carries
    /// for the whole session. The session's own facts are the status row's. Gone once the
    /// conversation starts; see [`Self::keeps_banner`].
    Banner {
        version: String,
        /// The commit the binary was built from, `-dirty` when the tree had uncommitted changes;
        /// `None` for a build outside a checkout.
        revision: Option<String>,
        /// Whether the binary is an unoptimized build.
        debug: bool,
        /// The MCP servers the configuration enables.
        mcp_servers: usize,
        skills: usize,
        /// The memories the session's opening view holds.
        memories: usize,
        /// The pieces of standing instructions in effect: each file read, or one for a flag or
        /// a variable; zero for none.
        instructions: usize,
    },
}

impl Item {
    /// Whether the banner stays with this item beside it: a line said beside the conversation
    /// does not start one, where a prompt, an answer, a call or the session's own rows do.
    fn keeps_banner(&self) -> bool {
        matches!(
            self,
            Item::Banner { .. } | Item::Notice { .. } | Item::Error(_) | Item::Compaction { .. }
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

/// What a sent prompt ran under, which the bar down its left edge shows in color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RanUnder {
    /// A turn at this permission level.
    Level(crate::permission::Permission),
    /// The shell, as a `!command` beside the session.
    Shell,
    /// A turn the log has no record of, as in a log written before turns were recorded.
    Unrecorded,
    /// A turn that has not opened yet: a prompt on its way, or one waiting to be delivered. The
    /// turn that takes it gives it the level it opened under.
    Pending,
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
    /// The id of the compaction this item is part of the contents of, if it is.
    pub(crate) compaction: Option<u64>,
    /// Whether a reset lets the item go for what the replay of the log says in its place: a stage
    /// direction for something the log records, a nudge, a fire, a report or an invocation, which
    /// the replay draws in its own right, and anything said while a turn ran, whose place among
    /// the turn's rows the replay cannot give back.
    pub(crate) left_to_replay: bool,
}

impl Entry {
    /// Whether the item has a collapsed and an expanded form.
    pub(crate) fn is_collapsible(&self) -> bool {
        matches!(
            self.item,
            Item::Thinking { .. } | Item::ToolCall { .. } | Item::Compaction { .. }
        )
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
    display: Option<ResultDisplay>,
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
    /// The permission level the running turn opened under, from its opening until its end:
    /// what a prompt that joins it runs under.
    turn_level: Option<crate::permission::Permission>,
    /// Whether a checklist nudge was the last thing said, so the stage direction that follows
    /// stands for a nudge the log records.
    nudged: bool,
    /// Where the running turn's opening stands on the tail, its first prompt, from the opening
    /// until its message is announced: what the opening's message says ahead of the words typed,
    /// a report, goes there, as a replay of the log draws it.
    opening_at: Option<usize>,
    /// The checklist the pages below the window drew last, while the window is cut from the
    /// tail: the list drawn last before the tail, which the window no longer holds.
    below_checklist: Option<Vec<ChecklistItem>>,
    /// Whether the transcript waits for the first page of a session it switched to, which a
    /// checklist the agent announces meanwhile waits for, as one announced at launch does.
    awaiting_first_page: bool,
    /// The inbox items the screen drew that the running turn put in the conversation, by id:
    /// delivered when it ends, whether or not the provider read them.
    appended_inbox: Vec<Uuid>,
    /// The checklist the agent announced while the first page of the log was still being read,
    /// and where on the tail it was announced, put there once the page lands, so a list the page
    /// already draws is not drawn twice and a turn under way is not cut by it.
    held_checklist: Option<(Vec<ChecklistItem>, usize)>,
    /// The stretches of the log the session's conversation no longer holds, which a replay leaves
    /// out, so the screen shows the conversation that goes on rather than the log; `None` until
    /// they are read after a reset.
    rewound: Option<Vec<crate::store::Rewound>>,
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
    /// The pictures of the images the next page of the log holds, by the hash they rest under,
    /// since a page holds an image as a reference into the store.
    pictures: std::collections::HashMap<String, crate::render::tui::Picture>,
    /// The pictures the composer drew of the images an invocation took, in the order it sent
    /// them, which the turn's opening shows rather than decoding the images again on the screen's
    /// thread.
    offered_pictures: Vec<crate::render::tui::Picture>,
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
            turn_level: None,
            nudged: false,
            opening_at: None,
            below_checklist: None,
            awaiting_first_page: false,
            appended_inbox: Vec::new(),
            held_checklist: None,
            rewound: None,
            turn_started: None,
            pending_turn_end: None,
            thinking_shown: false,
            tools_expanded: false,
            open_text: None,
            open_thinking: None,
            revision: 0,
            next_id: 0,
            pictures: std::collections::HashMap::new(),
            offered_pictures: Vec::new(),
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

    /// Whether finished runs of steps are folded: the configured default, which showing every
    /// call expanded suspends.
    pub(crate) fn folds_steps(&self) -> bool {
        self.settings.fold_steps && !self.tools_expanded
    }

    /// Whether a turn is running, whose steps may still grow.
    pub(crate) fn turn_running(&self) -> bool {
        self.turn_started.is_some()
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
            compaction: None,
            left_to_replay: false,
        }
    }

    /// A compaction's row and its contents: the checkpoint's steps, as the transcript draws a
    /// turn's, then `summary` unless it is blank.
    fn compaction_entries(&mut self, record: &crate::conversation::CompactionRecord) -> Vec<Entry> {
        let mut contents = compaction_steps(record);
        let summary = record.written_summary.trim();
        if !summary.is_empty() {
            contents.push(Item::CompactionSummary {
                text: summary.to_string(),
            });
        }
        let head = self.entry(Item::Compaction {
            headline: record.headline().to_string(),
            departures: record
                .departures()
                .into_iter()
                .map(str::to_string)
                .collect(),
        });
        let id = head.id;
        let mut entries = vec![head];
        for item in contents {
            let mut entry = self.entry(item);
            entry.compaction = Some(id);
            entries.push(entry);
        }
        entries
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

    /// Add a stage direction for something the log records; see [`Entry::left_to_replay`].
    fn push_left_to_replay(&mut self, item: Item) {
        let index = self.push_entry(item);
        if let Some(entry) = self.tail.get_mut(index) {
            entry.left_to_replay = true;
        }
        self.flush_turn_end();
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

    /// The pictures the composer drew of the images the turn about to open took with an
    /// invocation, in the order they were sent.
    pub(crate) fn offer_pictures(&mut self, pictures: Vec<crate::render::tui::Picture>) {
        self.offered_pictures = pictures;
    }

    /// The pictures of the images the page about to be loaded holds, by the hash they rest under.
    pub(crate) fn set_pictures(
        &mut self,
        pictures: std::collections::HashMap<String, crate::render::tui::Picture>,
    ) {
        self.pictures = pictures;
    }

    /// The picture of `source`: decoded from its bytes when they are here, else the one read for
    /// the page it is on.
    fn picture_of(&self, source: &crate::image::ImageSource) -> crate::render::tui::Picture {
        match source {
            crate::image::ImageSource::Blob { hash, .. } => self
                .pictures
                .get(hash)
                .cloned()
                .unwrap_or_else(crate::render::tui::Picture::unread),
            crate::image::ImageSource::Base64 { .. } => {
                crate::render::tui::Picture::from_source(source)
            }
        }
    }

    /// Whether the log has rows the transcript has not read a page of yet.
    fn reading_first_page(&self) -> bool {
        self.awaiting_first_page
            || (self.log_end.is_some() && self.pages.is_empty() && self.window.is_empty())
    }

    /// Put the checklist held for the first page where it was announced, now that the page is
    /// read, unless it is the list the page drew last.
    fn put_held_checklist(&mut self) {
        self.awaiting_first_page = false;
        let Some((items, at)) = self.held_checklist.take() else {
            return;
        };
        if items.is_empty() || self.last_checklist() == Some(&items) {
            return;
        }
        let entry = self.entry(Item::Checklist { items });
        self.insert_entries(at, vec![entry]);
    }

    /// Draw the checklist as it stands, unless it is the list drawn last.
    fn show_checklist(&mut self, items: Vec<ChecklistItem>) {
        if self.last_checklist() == Some(&items) {
            return;
        }
        self.close_streams();
        // The list as it stands replaces the list as it stood a moment ago, when nothing came
        // between them; a list after an answer is a new fact.
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

    /// The checklist the transcript drew last, live or from the log, the pages cut from below the
    /// window included.
    fn last_checklist(&self) -> Option<&Vec<ChecklistItem>> {
        fn last_list(entries: &[Entry]) -> Option<&Vec<ChecklistItem>> {
            entries.iter().rev().find_map(|entry| match &entry.item {
                Item::Checklist { items } => Some(items),
                _ => None,
            })
        }
        last_list(&self.tail)
            .or(self.below_checklist.as_ref())
            .or_else(|| last_list(&self.window))
    }

    /// What a prompt that arrives now runs under: the running turn's level once it has opened.
    fn joining_turn(&self) -> RanUnder {
        self.turn_level.map_or(RanUnder::Pending, RanUnder::Level)
    }

    /// What a replayed prompt of `turn` ran under: the level its turn recorded.
    fn ran_under(&self, turn: Option<Uuid>) -> RanUnder {
        turn.and_then(|turn| self.turn_records.get(&turn))
            .map_or(RanUnder::Unrecorded, |record| {
                RanUnder::Level(record.permission)
            })
    }

    /// What the session's conversation no longer holds of its log, which a replay leaves out.
    pub(crate) fn set_rewound(&mut self, rewound: Vec<crate::store::Rewound>) {
        self.rewound = Some(rewound);
    }

    /// Whether the session's rewound stretches have been read since the window was reset.
    pub(crate) fn knows_rewound(&self) -> bool {
        self.rewound.is_some()
    }

    /// `page` without the rows the conversation no longer holds, read before anything else is, so
    /// a rewound turn closes nowhere either, at a page's edge included. The page's span is the
    /// page as read, which is what the next read starts from.
    fn without_rewound(&self, mut page: crate::store::LogPage) -> crate::store::LogPage {
        let rewound = self.rewound.as_deref().unwrap_or_default();
        page.rows
            .retain(|row| !crate::store::is_rewound(rewound, row.id));
        page
    }

    /// The row that closes the recorded turn `turn`, or `None` while the turn is still open, when
    /// its record is unknown, or when the summary is switched off.
    fn turn_end_entry(&mut self, turn: Uuid) -> Option<Entry> {
        let record = self
            .turn_records
            .get(&turn)
            // A compaction's turn is said by its divider, as the live screen says it.
            .filter(|record| record.source != crate::store::turns::TurnSource::Compaction)?;
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
            record.stop_reason.as_deref(),
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
        let mut entry = self.entry(item);
        // The host's row saying how a turn failed is the turn's too, drawn before its closing row.
        entry.left_to_replay = self.turn_started.is_some() || self.pending_turn_end.is_some();
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
    /// log does not hold. What a reload after a rewind or a fork does once the last page is read
    /// again, before it is put up: the page brings this run's turns back in their stored form, so
    /// the tail keeps only what the host said on its own account between turns, the fork's
    /// banner among it, and the shell's commands; see [`Entry::left_to_replay`].
    pub(crate) fn reset(&mut self, log_end: Option<i64>) {
        self.window.clear();
        self.pages.clear();
        self.more_above = log_end.is_some();
        self.attached = true;
        self.below_checklist = None;
        self.log_end = log_end;
        self.orphan_results.clear();
        self.turn_records.clear();
        self.rewound = None;
        self.turn_started = None;
        self.pending_turn_end = None;
        self.tail.retain(|entry| match entry.item {
            Item::Banner { .. }
            | Item::SessionId { .. }
            | Item::ShellOutput(_)
            | Item::Prompt {
                ran_under: RanUnder::Shell,
                ..
            } => true,
            Item::Error(_) | Item::Notice { .. } => !entry.left_to_replay,
            _ => false,
        });
        self.open_text = None;
        self.open_thinking = None;
        self.nudged = false;
        self.opening_at = None;
        self.appended_inbox.clear();
        self.offered_pictures.clear();
        self.revision += 1;
    }

    /// Let go of everything the transcript holds of the session it showed but the banner: what
    /// switching to another session does before that session's log is read.
    pub(crate) fn forget_session(&mut self) {
        self.tail
            .retain(|entry| matches!(entry.item, Item::Banner { .. }));
        self.turn_level = None;
        self.held_checklist = None;
        self.background_tasks.clear();
        self.reset(None);
        self.awaiting_first_page = true;
    }

    /// Give the prompts sent for a turn that never opened, one refused before it began, the gray
    /// of a prompt nothing ran: what the host's answer to them closing says.
    pub(crate) fn settle_unopened_prompts(&mut self) {
        for index in 0..self.tail.len() {
            if let Some(Entry {
                item:
                    Item::Prompt {
                        delivered: true,
                        ran_under,
                        ..
                    },
                ..
            }) = self.tail.get_mut(index)
                && *ran_under == RanUnder::Pending
            {
                *ran_under = RanUnder::Unrecorded;
                self.touch(index);
            }
        }
    }

    /// Let the window go, keeping the tail attached: what returning to the bottom does when the
    /// window had been cut from it, before the last page is read again.
    pub(crate) fn drop_window(&mut self) {
        self.window.clear();
        self.pages.clear();
        self.orphan_results.clear();
        self.more_above = self.log_end.is_some();
        self.attached = true;
        self.below_checklist = None;
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
            self.put_held_checklist();
            return;
        };
        let page = self.without_rewound(page);
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
            && let Some(ended) = last_turn
            && below.first_turn != Some(ended)
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
                // The newest page goes first, so the first list found is the last drawn.
                if self.below_checklist.is_none() {
                    self.below_checklist = self
                        .window
                        .get(keep..)
                        .unwrap_or_default()
                        .iter()
                        .rev()
                        .find_map(|entry| match &entry.item {
                            Item::Checklist { items } => Some(items.clone()),
                            _ => None,
                        });
                }
                self.window.truncate(keep);
                self.attached = false;
            }
        }
        self.settle_banner();
        self.revision += 1;
        self.put_held_checklist();
    }

    /// A page read downward from the window's ceiling, put below what the window holds; the
    /// window reattaches to the tail when the page reaches the opening row. The oldest pages go
    /// when the window would exceed its bound.
    pub(crate) fn load_page_below(&mut self, page: crate::store::LogPage) {
        let Some((first_row, last_row)) = page_span(&page) else {
            self.attached = true;
            self.below_checklist = None;
            return;
        };
        let page = self.without_rewound(page);
        let (mut entries, ends_with_turn_end) = self.build_entries(&page.rows, true, !page.more);
        let (first_turn, last_turn) = edge_turns(&page);
        // The join with the page above: the turn its last row belongs to ends here when this
        // page opens another, unless that page already closes it.
        let mut starts_with_turn_end = false;
        if let Some(above) = self.pages.back()
            && !above.ends_with_turn_end
            && let Some(ended) = above.last_turn
            && first_turn != Some(ended)
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
            self.below_checklist = None;
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
        // A result follows its call, so an orphan from the page below belongs to the last call
        // with its id that has none, which matters on a backend that reuses ids round to round. A
        // compaction's steps carry their own results.
        for entry in entries
            .iter_mut()
            .rev()
            .filter(|entry| entry.compaction.is_none())
        {
            let orphan = match &entry.item {
                Item::ToolCall {
                    id, result: None, ..
                } => self.orphan_results.remove(id),
                _ => None,
            };
            if let Some(orphan) = orphan {
                settle_call(
                    &mut entry.item,
                    orphan.text,
                    orphan.images,
                    orphan.is_error,
                    orphan.display.as_ref(),
                );
            }
        }
    }

    /// Flip one collapsible item, by its id, which holds wherever the item has moved since the
    /// frame it was picked in; whether there was one to flip.
    pub(crate) fn toggle_item(&mut self, id: u64) -> bool {
        self.revision += 1;
        let revision = self.revision;
        let attached = self.attached;
        let tail = if attached {
            &mut self.tail[..]
        } else {
            &mut []
        };
        if let Some(entry) = self
            .window
            .iter_mut()
            .chain(tail.iter_mut())
            .find(|entry| entry.id == id)
            && entry.is_collapsible()
        {
            entry.expanded = !entry.expanded;
            entry.revision = revision;
            return true;
        }
        false
    }

    /// Flip one collapsible item, by its place on screen; see [`Self::toggle_item`].
    #[cfg(test)]
    pub(crate) fn toggle(&mut self, index: usize) -> bool {
        let id = self.drawn().nth(index).map(|entry| entry.id);
        id.is_some_and(|id| self.toggle_item(id))
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

    /// The prompt an inbox block stands for, to draw where the block lands. One the screen drew
    /// when it enqueued it is taken from where it was typed, which a replay of the log knows
    /// nothing of; it is the running turn's now, and reads delivered once the turn ends if the
    /// provider has not read it by then, as the replay draws it.
    fn inbox_prompt(&mut self, block: &ContentBlock) -> Option<Item> {
        let ContentBlock::Inbox {
            item_id,
            class,
            body,
            ..
        } = block
        else {
            return None;
        };
        if let Some(id) = *item_id
            && let Some(index) = self.tail.iter().position(|entry| {
                matches!(&entry.item, Item::Prompt { inbox_id: Some(drawn), .. } if *drawn == id)
            })
        {
            self.appended_inbox.push(id);
            let drawn = self.tail.get(index).map(|entry| entry.item.clone());
            self.remove_entry(index);
            return drawn;
        }
        Some(Item::Prompt {
            text: body.trim().to_string(),
            class: *class,
            inbox_id: *item_id,
            delivered: false,
            ran_under: self.joining_turn(),
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
                self.join_running_turn(index);
            }
        }
    }

    /// Give the prompt at `index`, once delivered, the level of the running turn it joined.
    fn join_running_turn(&mut self, index: usize) {
        let Some(level) = self.turn_level else {
            return;
        };
        if let Some(Entry {
            item:
                Item::Prompt {
                    delivered: true,
                    ran_under,
                    ..
                },
            ..
        }) = self.tail.get_mut(index)
            && *ran_under == RanUnder::Pending
        {
            *ran_under = RanUnder::Level(level);
            self.touch(index);
        }
    }

    /// Send the followup the screen held for the end of a turn, `id` while it waited: drawn again
    /// at the end of the tail, where the turn it opens begins, as the prompt it now is, its waiting
    /// copy let go wherever it was typed.
    pub(crate) fn send_followup(&mut self, id: u64, text: String) {
        if let Some(index) = self.tail.iter().position(|entry| entry.id == id) {
            self.remove_entry(index);
        }
        self.push(Item::Prompt {
            text,
            class: None,
            inbox_id: None,
            delivered: true,
            ran_under: RanUnder::Pending,
        });
    }

    fn close_streams(&mut self) {
        if let Some(index) = self.open_thinking.take() {
            self.end_thinking(index);
        }
        if let Some(index) = self.open_text.take() {
            self.end_text(index);
        }
    }

    /// End the thinking block at `index`. One that ended with no words is let go, as a replay of
    /// the log draws none: its counter stood while it streamed.
    fn end_thinking(&mut self, index: usize) {
        let Some(Entry {
            item: Item::Thinking { text, ended, .. },
            ..
        }) = self.tail.get_mut(index)
        else {
            return;
        };
        if text.trim().is_empty() {
            self.remove_entry(index);
        } else {
            *ended = true;
            self.touch(index);
        }
    }

    /// End the answer text at `index`. Text of spaces alone says nothing, and a replay of the
    /// log draws none.
    fn end_text(&mut self, index: usize) {
        let Some(Entry {
            item: Item::Text { streaming, text },
            ..
        }) = self.tail.get_mut(index)
        else {
            return;
        };
        if text.trim().is_empty() {
            self.remove_entry(index);
        } else {
            *streaming = false;
            self.touch(index);
        }
    }

    /// Put `entries` into the tail at `at`, keeping the open streams and the opening on the
    /// entries they were on.
    fn insert_entries(&mut self, at: usize, entries: Vec<Entry>) {
        let at = at.min(self.tail.len());
        let count = entries.len();
        self.tail.splice(at..at, entries);
        for open in [
            &mut self.open_text,
            &mut self.open_thinking,
            &mut self.opening_at,
        ] {
            if let Some(index) = open.as_mut()
                && *index >= at
            {
                *index += count;
            }
        }
        self.revision += 1;
        self.settle_banner();
    }

    /// Where the prompts sent for a turn that has not opened yet start on the tail: the first of
    /// the delivered prompts still waiting for their turn, with the pictures sent with them, that
    /// end it.
    fn unopened_prompts_at(&self) -> Option<usize> {
        let mut at = None;
        for (index, entry) in self.tail.iter().enumerate().rev() {
            match &entry.item {
                // The pictures are what a prompt of images alone is drawn as.
                Item::Prompt {
                    delivered: true,
                    ran_under: RanUnder::Pending,
                    ..
                }
                | Item::Image(_) => at = Some(index),
                _ => break,
            }
        }
        at
    }

    /// Take the tail's entry at `index` out, keeping the open streams on the entries they were on.
    fn remove_entry(&mut self, index: usize) {
        if index >= self.tail.len() {
            return;
        }
        self.tail.remove(index);
        let shift = |open: &mut Option<usize>| {
            *open = match *open {
                Some(at) if at == index => None,
                Some(at) if at > index => Some(at - 1),
                other => other,
            };
        };
        shift(&mut self.open_text);
        shift(&mut self.open_thinking);
        if let Some(at) = self.opening_at.as_mut()
            && *at > index
        {
            *at -= 1;
        }
        self.revision += 1;
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
        if let Some(index) = self.open_text.take() {
            self.end_text(index);
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
            // Every prompt sent and waiting for this turn is the turn's.
            FrontendEvent::TurnOpened { permission } => {
                self.turn_level = Some(permission);
                self.opening_at = self.unopened_prompts_at();
                for index in 0..self.tail.len() {
                    if let Some(Entry {
                        item:
                            Item::Prompt {
                                delivered: true,
                                ran_under,
                                ..
                            },
                        ..
                    }) = self.tail.get_mut(index)
                        && *ran_under == RanUnder::Pending
                    {
                        *ran_under = RanUnder::Level(permission);
                        self.touch(index);
                    }
                }
            }
            FrontendEvent::TurnFinished => self.close_streams(),
            // The turn's closing row. A turn that succeeded closes at once; one that failed or was
            // interrupted waits for the host's row saying so, which follows this event.
            FrontendEvent::TurnEnded { ending, .. } => {
                let appended = std::mem::take(&mut self.appended_inbox);
                self.mark_delivered(&appended);
                self.turn_level = None;
                self.nudged = false;
                self.opening_at = None;
                // A call with no result when the turn ended never ran, and the log holds none of
                // it: one still being written, or written whole and cut before it was run, an
                // attempt that failed and was tried again or one the stop cut. What ran was
                // answered, if only to say the stop interrupted it.
                while let Some(index) = self.tail.iter().rposition(|entry| {
                    matches!(entry.item, Item::ToolCall {
                        state: ToolState::Composing | ToolState::Running,
                        ..
                    })
                }) {
                    self.remove_entry(index);
                }
                let ended_at = Utc::now();
                let elapsed = self
                    .turn_started
                    .take()
                    .and_then(|started| (ended_at - started).to_std().ok());
                if let Some(text) = crate::render::turn_end_summary(
                    &self.settings.turn_end_summary,
                    ending.status,
                    ending.stop_reason.as_deref(),
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
            // Words another client posted, landing at a round boundary: drawn where they land,
            // as a replay draws them.
            FrontendEvent::InboxAppended { blocks } => {
                for block in &blocks {
                    if let Some(item) = self.inbox_prompt(block) {
                        self.close_streams();
                        self.push(item);
                    }
                }
            }
            FrontendEvent::InboxDelivered { item_ids } => self.mark_delivered(&item_ids),
            // The words are the model's to read; the stage direction that follows a checklist
            // nudge is what the person sees of it, counted live, where a replay of the log says it
            // in the words the log can carry.
            FrontendEvent::Nudged { kind, .. } => {
                self.nudged = kind == crate::conversation::NudgeKind::Checklist;
            }
            // What reported is said as a stage direction; the words are the model's to read.
            FrontendEvent::BackgroundReported { tasks, .. } => {
                self.remember_background_tasks(tasks.clone());
                self.close_streams();
                self.push_left_to_replay(Item::Notice {
                    level: NoticeLevel::Info,
                    text: crate::background::report_direction(&tasks),
                });
            }
            // The message a turn opens with, whichever door opened it. The person's own words are
            // drawn as they are typed and the context block is meka's; a fire and a report are
            // said as the stage directions a replay of the log says, so the two read alike.
            // A fire says how late it ran, news as it happens, which a replay of the log does
            // not know.
            FrontendEvent::MessageAppended {
                message,
                outcomes,
                late,
            } => {
                self.remember_background_tasks(outcomes.clone());
                let mut offered = std::mem::take(&mut self.offered_pictures).into_iter();
                let invoked = message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Invocation { .. }));
                // What the message says ahead of what was sent, a report, stands ahead of the
                // prompt it was drawn as, its words or the pictures of a prompt of images alone,
                // as a replay of the log draws it.
                let sent_at = message.content.iter().position(|block| {
                    matches!(
                        block,
                        ContentBlock::Text { .. } | ContentBlock::Image { .. }
                    )
                });
                let mut ahead = self.opening_at.take();
                for (position, block) in message.content.iter().enumerate() {
                    let item = match block {
                        ContentBlock::Scheduled { job_id, prompt, .. } => {
                            Item::note(crate::render::fire_direction(job_id, prompt, late))
                        }
                        ContentBlock::BackgroundReport { .. } => {
                            Item::note(crate::background::report_direction(&outcomes))
                        }
                        // A command line is not echoed when it is typed, so what it invoked is
                        // said here, as a replay says it, with the words typed with it under it.
                        ContentBlock::Invocation {
                            kind,
                            name,
                            directive,
                            ..
                        } => {
                            self.close_streams();
                            self.push_left_to_replay(Item::note(kind.direction(name)));
                            if crate::text::is_blank(directive) {
                                continue;
                            }
                            Item::Prompt {
                                text: directive.trim().to_string(),
                                class: None,
                                inbox_id: None,
                                delivered: true,
                                ran_under: self.joining_turn(),
                            }
                        }
                        // Words another client posted to the inbox, read by the model with this
                        // turn: drawn as a replay draws them, unless the screen drew them when it
                        // enqueued them itself; the delivery mark then reaches either.
                        ContentBlock::Inbox { .. } => match self.inbox_prompt(block) {
                            Some(item) => item,
                            None => continue,
                        },
                        // The images a command line took, said with what it invoked, as a replay
                        // says them.
                        ContentBlock::Image { source } if invoked => {
                            Item::Image(offered.next().unwrap_or_else(|| self.picture_of(source)))
                        }
                        // Drawn when they were typed; an editor's resources never reach the
                        // screen live.
                        ContentBlock::Text { .. }
                        | ContentBlock::Resource { .. }
                        | ContentBlock::TurnContext { .. }
                        | ContentBlock::Nudge { .. }
                        | ContentBlock::Image { .. }
                        | ContentBlock::Thinking { .. }
                        | ContentBlock::RedactedThinking { .. }
                        | ContentBlock::ToolUse { .. }
                        | ContentBlock::ToolResult { .. } => continue,
                    };
                    self.close_streams();
                    // A fire's and a report's directions are the replay's own words for blocks
                    // the log records.
                    let replayed = matches!(item, Item::Notice { .. });
                    match ahead {
                        Some(at) if sent_at.is_some_and(|sent| position < sent) => {
                            let mut entry = self.entry(item);
                            entry.left_to_replay = replayed;
                            self.insert_entries(at, vec![entry]);
                            ahead = Some(at + 1);
                        }
                        _ if replayed => self.push_left_to_replay(item),
                        _ => {
                            self.push(item);
                        }
                    }
                }
            }
            FrontendEvent::AssistantTextDelta(text) => self.append_text(&text),
            FrontendEvent::ThinkingProgress { estimated_tokens } => {
                self.append_thinking(None, estimated_tokens);
            }
            FrontendEvent::ThinkingDelta(text) => self.append_thinking(Some(&text), None),
            FrontendEvent::ThinkingEnded => {
                if let Some(index) = self.open_thinking.take() {
                    self.end_thinking(index);
                }
            }
            FrontendEvent::ThinkingBlock { content } => {
                match self.open_thinking.take() {
                    // The text is already here; this is only the end of the block.
                    Some(index) => {
                        if let Some(Entry {
                            item: Item::Thinking { text, .. },
                            ..
                        }) = self.tail.get_mut(index)
                            && text.is_empty()
                        {
                            *text = sanitize_stream_text(&content);
                        }
                        self.end_thinking(index);
                    }
                    // A block with no words is none, as on a replay of the log.
                    None if content.trim().is_empty() => {}
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
                    exit_code: None,
                });
            }
            FrontendEvent::ToolCallStarted {
                id,
                name,
                input,
                display_summary,
            } => {
                self.close_streams();
                // The call being written under this id, never an earlier round's call that a
                // provider gave the same id, which a call that streamed no composing first meets.
                let composing = self.tail.iter().rposition(|entry| {
                    matches!(&entry.item, Item::ToolCall {
                        id: call,
                        state: ToolState::Composing,
                        ..
                    } if *call == id)
                });
                let index = match composing {
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
                        exit_code: None,
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
                display,
                ..
            } => {
                if let Some(index) = self.tool_index(&id) {
                    if let Some(entry) = self.tail.get_mut(index) {
                        let (text, images) = result_text(&content);
                        settle_call(&mut entry.item, text, images, is_error, display.as_ref());
                    }
                    self.touch(index);
                }
            }
            // A list announced while the first page of the log is still being read waits for it,
            // and the list a resumed session hydrates is then the list its replay drew last, which
            // says nothing new.
            FrontendEvent::ChecklistUpdated { items } if self.reading_first_page() => {
                self.held_checklist = Some((items, self.tail.len()));
            }
            FrontendEvent::ChecklistUpdated { items } => self.show_checklist(items),
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
            // The gauge reads the cells the screen shares with the agent, on every frame.
            FrontendEvent::ContextMeasured { .. } => {}
            // A stage direction and an advisory are one register on the screen; the level is
            // what tells a warning from a note.
            FrontendEvent::Notice(notice) => {
                self.close_streams();
                let item = Item::Notice {
                    level: notice.level,
                    text: notice.text,
                };
                // A nudge's direction stands for a nudge the log records, which a replay draws
                // in its own words; another notice between the two leaves the nudge waiting for
                // it.
                if notice.kind == crate::frontend::NoticeKind::StageDirection
                    && std::mem::take(&mut self.nudged)
                {
                    self.push_left_to_replay(item);
                } else {
                    self.push(item);
                }
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
            // a compaction changes nothing already on it; the row says what happened, in the words
            // a replay of the log's boundary uses, so a resumed session reads as this one did.
            FrontendEvent::Compacted { record, .. } => {
                self.close_streams();
                let entries = self.compaction_entries(&record);
                // A compaction ahead of the turn's opening stands ahead of the prompt the turn
                // opens with, as the log writes the two.
                let ahead = (self.turn_started.is_some() && self.turn_level.is_none())
                    .then(|| self.unopened_prompts_at())
                    .flatten();
                match ahead {
                    Some(at) => self.insert_entries(at, entries),
                    None => self.tail.extend(entries),
                }
                self.settle_banner();
                self.flush_turn_end();
            }
            FrontendEvent::CompactionStarted { .. } | FrontendEvent::CompactionFailed { .. } => {}
            // The compaction's row shows its checkpoint once it lands, from the record.
            FrontendEvent::CheckpointCallCompleted { .. } => {}
            // A call its round ended without running goes now, as one the turn's end finds does:
            // the log holds none of it.
            //
            // By its state too, since a stream tried again can bring a call under the same id while
            // the attempt before left one composing.
            FrontendEvent::ToolCallDropped { id, started } => {
                let state = if started {
                    ToolState::Running
                } else {
                    ToolState::Composing
                };
                if let Some(index) = self.tail.iter().rposition(|entry| {
                    matches!(&entry.item, Item::ToolCall { id: call, state: drawn, .. }
                        if *call == id && *drawn == state)
                }) {
                    self.remove_entry(index);
                }
            }
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

    /// The stage direction for a replayed report, from the tasks the screen has read.
    fn replayed_report_direction(&self, task_ids: &[String]) -> String {
        crate::background::replayed_report_direction(&self.background_tasks, task_ids)
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
        use crate::store::Said;
        let mut out: Vec<Entry> = Vec::new();
        let mut shown_checklist: Option<Vec<ChecklistItem>> = None;
        // Without the stretches, which a failed read leaves the screen, every row says what it
        // holds and a rewind says nothing.
        let rewound = self.rewound.clone().unwrap_or_default();
        // The turn of the last row: a turn ends where the next row names another or none, and the
        // last turn of the log ends with the log.
        let mut turn: Option<Uuid> = None;
        let find_call = |out: &[Entry], id: &str| {
            out.iter().rposition(
                |entry| matches!(&entry.item, Item::ToolCall { id: call, .. } if call == id),
            )
        };
        for row in rows {
            if let Some(ended) = turn
                && row.turn_id != Some(ended)
                && let Some(entry) = self.turn_end_entry(ended)
            {
                out.push(entry);
            }
            // A row naming no turn was written between turns, as a rewind is, so the turn before it
            // has ended; in a log written before turns were recorded, no row names one.
            turn = row.turn_id;
            // A copy standing alone marks where the turn that wrote it ends, and is drawn as the
            // turn that said it.
            let ran_under = self.ran_under(row.said.as_ref().map_or(turn, |said| said.turn_id));
            let message = match row.said(&rewound) {
                Said::Message(message) => message,
                Said::Compaction { record } => {
                    let entries = self.compaction_entries(record);
                    out.extend(entries);
                    continue;
                }
                Said::Rewind { turns, .. } => {
                    let entry = self.entry(Item::note(rewound_note(turns)));
                    out.push(entry);
                    continue;
                }
                // The live run said nothing of a prompt it took back to join a message to it.
                Said::Join { .. } => continue,
                Said::Nothing => continue,
            };
            if let Some(note) = crate::render::nudge_direction(message) {
                let entry = self.entry(Item::note(note));
                out.push(entry);
            }
            let mut checklist = None;
            let with_mentions;
            let content = if message.role == Role::User
                && message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Resource { .. }))
            {
                with_mentions = crate::conversation::mentions_inline(&message.content);
                with_mentions.as_slice()
            } else {
                message.content.as_slice()
            };
            for block in content {
                if !matches!(block, ContentBlock::ToolResult { .. }) {
                    self.draw_batch_checklist(&mut out, &mut checklist, &mut shown_checklist);
                }
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
                                ran_under,
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
                            ran_under,
                        });
                        out.push(entry);
                    }
                    // What was invoked, said as the live screen said it, then the words typed
                    // with it as the person's.
                    ContentBlock::Invocation {
                        kind,
                        name,
                        directive,
                        ..
                    } => {
                        let entry = self.entry(Item::note(kind.direction(name)));
                        out.push(entry);
                        if !crate::text::is_blank(directive) {
                            let entry = self.entry(Item::Prompt {
                                text: directive.trim().to_string(),
                                class: None,
                                inbox_id: None,
                                delivered: true,
                                ran_under,
                            });
                            out.push(entry);
                        }
                    }
                    // Folded into the words around it by `conversation::mentions_inline`.
                    ContentBlock::Resource { .. } => {}
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
                    ContentBlock::Image { source } => {
                        let entry = self.entry(Item::Image(self.picture_of(source)));
                        out.push(entry);
                    }
                    // A block with no words is none: the live run let its counter go as it ended.
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
                            change: None,
                            exit_code: None,
                        });
                        out.push(entry);
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                        display,
                    } => {
                        if let Some(items) = display
                            .as_ref()
                            .and_then(|display| display.checklist.clone())
                        {
                            checklist = Some(items);
                        }
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
                                if let Some(entry) = out.get_mut(index) {
                                    settle_call(
                                        &mut entry.item,
                                        text,
                                        images,
                                        *is_error,
                                        display.as_ref(),
                                    );
                                }
                            }
                            Some(index) => {
                                let index = index - out.len();
                                self.revision += 1;
                                let revision = self.revision;
                                if let Some(entry) = self.window.get_mut(index) {
                                    settle_call(
                                        &mut entry.item,
                                        text,
                                        images,
                                        *is_error,
                                        display.as_ref(),
                                    );
                                    entry.revision = revision;
                                }
                            }
                            None => {
                                self.orphan_results
                                    .insert(tool_use_id.clone(), OrphanResult {
                                        text,
                                        images,
                                        is_error: *is_error,
                                        display: display.clone(),
                                    });
                            }
                        }
                    }
                }
            }
            self.draw_batch_checklist(&mut out, &mut checklist, &mut shown_checklist);
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

    /// Draw the list a batch of checklist calls left, `checklist`, where the live screen drew it:
    /// after the batch's results and ahead of what arrived behind them, standing in for the list
    /// drawn just before it, and only when it differs from `shown`, the last one drawn, as the
    /// dispatcher announces only a change.
    fn draw_batch_checklist(
        &mut self,
        out: &mut Vec<Entry>,
        checklist: &mut Option<Vec<ChecklistItem>>,
        shown: &mut Option<Vec<ChecklistItem>>,
    ) {
        let Some(items) = checklist.take() else {
            return;
        };
        if shown.as_ref() == Some(&items) {
            return;
        }
        *shown = Some(items.clone());
        if let Some(Entry {
            item: Item::Checklist { items: held },
            ..
        }) = out.last_mut()
        {
            *held = items;
        } else if !items.is_empty() {
            let entry = self.entry(Item::Checklist { items });
            out.push(entry);
        }
    }
}

/// The rows a page spans, or `None` for an empty one.
fn page_span(page: &crate::store::LogPage) -> Option<(i64, i64)> {
    Some((page.rows.first()?.id, page.rows.last()?.id))
}

/// The turns of a page's first and last rows, `None` for a row naming none, which was written
/// between turns.
fn edge_turns(page: &crate::store::LogPage) -> (Option<Uuid>, Option<Uuid>) {
    (
        page.rows.first().and_then(|row| row.turn_id),
        page.rows.last().and_then(|row| row.turn_id),
    )
}

/// The readable part of a tool result, tail-capped, and how many images it carried. What meka
/// wrote into a result for the model alone is left out or said for a reader: the line naming a
/// file read as an image, which the count says; the tools a load made callable, which its row
/// names; the directions around a large output kept in the scratchpad. The notice of a call sent
/// to the background is the call's to say; see [`settle_call`].
fn result_text(content: &[ToolResultContent]) -> (String, usize) {
    let images = content
        .iter()
        .filter(|part| matches!(part, ToolResultContent::Image { .. }))
        .count();
    let mut text = String::new();
    for part in content {
        let ToolResultContent::Text { text: part } = part else {
            continue;
        };
        if images > 0 && crate::tools::file::is_image_marker(part.trim()) {
            continue;
        }
        if !text.is_empty() {
            text.push('\n');
        }
        // A large output is spilled to the scratchpad after the live screen has drawn it whole,
        // so the opening the log keeps of it is what a replay can show of the same output.
        if let Some(preview) = crate::tools::scratchpad::large_output_preview(part) {
            text.push_str(preview);
        } else {
            text.push_str(part);
        }
    }
    keep_tail(&mut text, TOOL_TEXT_KEEP_BYTES);
    (text, images)
}

/// Close a call with its result, live or replayed, so the two read alike: the text, the images,
/// how it ended and what the log records of it for the screen. A result that records nothing
/// shows the change the call's arguments describe, unless it failed, which changed nothing.
fn settle_call(
    item: &mut Item,
    text: String,
    images: usize,
    is_error: bool,
    display: Option<&ResultDisplay>,
) {
    let Item::ToolCall {
        name,
        state,
        result,
        result_images,
        change,
        exit_code,
        input,
        progress,
        ..
    } = item
    else {
        return;
    };
    *result = Some(shown_result(text, input.as_ref()));
    *result_images = images;
    *state = ToolState::Done { is_error };
    *progress = None;
    *change = match display {
        Some(display) => display.change.clone(),
        None => input
            .as_ref()
            .and_then(|input| change_from_input(name, input)),
    };
    *exit_code = display.and_then(|display| display.exit_code);
    if is_error {
        *change = None;
    }
}

/// What a call's result shows of `text`: for a call that asked to go to the background, the
/// notice the dispatcher answered it with as the task it became, and otherwise the text itself.
fn shown_result(text: String, input: Option<&serde_json::Value>) -> String {
    let backgrounded = input
        .and_then(|input| input.get(crate::tools::BACKGROUND_PARAMETER))
        .and_then(serde_json::Value::as_bool)
        == Some(true);
    match backgrounded
        .then(|| crate::background::started_task(&text))
        .flatten()
    {
        Some(task) => format!("Background task {task}"),
        None => text,
    }
}

/// What a compaction's row shows it did: the checkpoint's rounds, or, on a boundary that recorded
/// none, the memories its record names, each as the call that wrote it, since a memory is the one
/// durable thing a checkpoint leaves and those names were observed from its calls.
fn compaction_steps(record: &crate::conversation::CompactionRecord) -> Vec<Item> {
    if !record.checkpoint.is_empty() {
        return checkpoint_items(&record.checkpoint);
    }
    record
        .memories_written
        .iter()
        .map(|name| Item::ToolCall {
            id: String::new(),
            name: "memory_write".to_string(),
            input: Some(serde_json::json!({ "name": name })),
            summary: None,
            state: ToolState::Done { is_error: false },
            output: String::new(),
            result: None,
            result_images: 0,
            activity: String::new(),
            progress: None,
            change: None,
            exit_code: None,
        })
        .collect()
}

/// The items a compaction's checkpoint rounds show, as a replay shows a turn's: its reasoning, its
/// words, and its calls with their results. The `context_replace` that handed in the summary is
/// left out, since the summary is shown under them whole.
fn checkpoint_items(rounds: &[crate::conversation::Message]) -> Vec<Item> {
    let mut items: Vec<Item> = Vec::new();
    for message in rounds {
        for block in &message.content {
            match block {
                ContentBlock::Thinking { thinking, .. } if !thinking.trim().is_empty() => {
                    items.push(Item::Thinking {
                        text: sanitize_stream_text(thinking),
                        estimate: None,
                        ended: true,
                    });
                }
                ContentBlock::Text { text } if message.role == Role::Assistant => {
                    if !text.trim().is_empty() {
                        items.push(Item::Text {
                            text: sanitize_stream_text(text),
                            streaming: false,
                        });
                    }
                }
                ContentBlock::ToolUse { id, name, input } if name != "context_replace" => {
                    items.push(Item::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        input: Some(input.clone()),
                        summary: None,
                        state: ToolState::Done { is_error: false },
                        output: String::new(),
                        result: None,
                        result_images: 0,
                        activity: String::new(),
                        progress: None,
                        change: change_from_input(name, input),
                        exit_code: None,
                    });
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    display,
                } => {
                    let (text, images) = result_text(content);
                    if let Some(item) = items
                        .iter_mut()
                        .rev()
                        .find(|item| matches!(item, Item::ToolCall { id, .. } if id == tool_use_id))
                    {
                        settle_call(item, text, images, *is_error, display.as_ref());
                    }
                }
                _ => {}
            }
        }
    }
    items
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

/// The change a call stands for when its result records none, from its arguments alone: a tool
/// declared to replace one parameter's text with another's is described by the two, numbered
/// within them, since the file around them is not recorded.
fn change_from_input(name: &str, input: &serde_json::Value) -> Option<Change> {
    let &[old, new] = crate::tools::builtin_change_params(name) else {
        return None;
    };
    let text = |key: &str| input.get(key)?.as_str();
    Some(Change::between(Some(text(old)?), Some(text(new)?)))
}

/// The row a rewind leaves where the turns it took were, live and on a replay alike.
pub(crate) fn rewound_note(turns: usize) -> String {
    format!("Rewound {turns} turn{}", if turns == 1 { "" } else { "s" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        conversation::{Event, Message},
        frontend::Notice,
        store::{LogPage, LogRow},
    };

    /// A replayed image draws the picture its page's loader read from the store, by the hash it
    /// rests under; one whose bytes were not read stands as its label.
    #[test]
    fn a_replayed_image_draws_the_picture_read_for_its_page() {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            4,
            4,
            image::Rgba([9, 9, 9, 255]),
        ))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode");
        let picture = crate::render::tui::Picture::from_bytes(&png);
        let message = Message {
            role: Role::User,
            content: vec![ContentBlock::Image {
                source: crate::image::ImageSource::Blob {
                    hash: "h1".into(),
                    media_type: "image/png".into(),
                    size: 10,
                },
            }],
        };
        let mut read = transcript();
        read.set_pictures(std::collections::HashMap::from([(
            "h1".to_string(),
            picture.clone(),
        )]));
        read.load_page_above(page(vec![Event::Append(message.clone())], 1, false));
        assert!(
            matches!(items(&read)[..], [Item::Image(shown)] if *shown == picture),
            "{:?}",
            items(&read)
        );
        let mut unread = transcript();
        unread.load_page_above(page(vec![Event::Append(message)], 1, false));
        assert!(
            matches!(items(&unread)[..], [Item::Image(shown)]
                if *shown == crate::render::tui::Picture::unread()),
            "{:?}",
            items(&unread)
        );
    }

    /// A replayed prompt runs under the level its turn recorded, and one whose turn the log has
    /// no record of is drawn as unrecorded. Live, a prompt runs under the level its turn opened
    /// under, as the turn's record says on a replay: one sent ahead of the opening takes it then,
    /// and one that joins a running turn, delivered or arriving, takes the running turn's.
    #[test]
    fn a_prompt_runs_under_the_level_its_turn_recorded() {
        use crate::permission::Permission;
        let recorded = Uuid::new_v4();
        let mut replayed = transcript();
        let mut record = closed_turn(recorded, TurnStatus::Succeeded);
        record.permission = Permission::Workspace;
        replayed.set_turn_records(vec![record]);
        replayed.load_page_above(LogPage {
            rows: vec![
                LogRow {
                    id: 1,
                    turn_id: Some(recorded),
                    created_at: String::new(),
                    said: None,
                    event: Event::Append(Message::user("in a recorded turn")),
                },
                LogRow {
                    id: 2,
                    turn_id: None,
                    created_at: String::new(),
                    said: None,
                    event: Event::Append(Message::user("before turns were recorded")),
                },
            ],
            more: false,
        });
        let ran = |transcript: &Transcript| -> Vec<RanUnder> {
            items(transcript)
                .into_iter()
                .filter_map(|item| match item {
                    Item::Prompt { ran_under, .. } => Some(*ran_under),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(ran(&replayed), vec![
            RanUnder::Level(Permission::Workspace),
            RanUnder::Unrecorded
        ]);
        let mut live = transcript();
        let pending = |text: &str, inbox_id: Option<Uuid>| Item::Prompt {
            text: text.to_string(),
            class: inbox_id.map(|_| InboxClass::Steer),
            inbox_id,
            delivered: inbox_id.is_none(),
            ran_under: RanUnder::Pending,
        };
        live.push(pending("typed", None));
        let steer = Uuid::new_v4();
        live.push(pending("a steer", Some(steer)));
        live.apply(FrontendEvent::TurnOpened {
            permission: Permission::Workspace,
        });
        assert_eq!(ran(&live), vec![
            RanUnder::Level(Permission::Workspace),
            RanUnder::Pending
        ]);
        live.mark_delivered(&[steer]);
        live.apply(FrontendEvent::MessageAppended {
            message: Message {
                role: Role::User,
                content: vec![ContentBlock::Inbox {
                    item_id: None,
                    class: Some(InboxClass::Steer),
                    source: Some("bridge".to_string()),
                    body: "from another client".to_string(),
                    text: "from another client".to_string(),
                }],
            },
            outcomes: Vec::new(),
            late: None,
        });
        assert_eq!(
            ran(&live),
            vec![RanUnder::Level(Permission::Workspace); 3],
            "a steer delivered and an item arriving join the running turn"
        );
        live.apply(FrontendEvent::TurnEnded {
            turn_id: Uuid::new_v4(),
            ending: crate::store::turns::TurnEnding::succeeded(
                "end_turn",
                crate::store::turns::TurnUsage::default(),
            ),
        });
        let followup = live.push(pending("a followup", None));
        live.send_followup(followup, "a followup".to_string());
        assert_eq!(ran(&live)[3], RanUnder::Pending, "no turn is running");
        live.apply(FrontendEvent::TurnOpened {
            permission: Permission::Read,
        });
        assert_eq!(ran(&live)[3], RanUnder::Level(Permission::Read));
    }

    fn transcript() -> Transcript {
        Transcript::new(Settings {
            show_session_id_on_create: false,
            turn_end_summary: TurnEndFact::DEFAULT_SUMMARY.to_vec(),
            fold_steps: true,
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
                    created_at: String::new(),
                    said: None,
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
            display: None,
        });
        let shown = items(&transcript);
        assert_eq!(shown.len(), 1, "{shown:?}");
        assert!(matches!(
            shown[0],
            Item::ToolCall { state: ToolState::Done { is_error: false }, output, result: Some(result), summary: Some(summary), .. }
                if output == "a\nb\n" && result == "a\nb\n" && summary == "ls"
        ));
    }

    /// A tool's result records the change it made, which its completion carries; the transcript
    /// keeps it as the call's change, a deleted entry's with no after.
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
            metadata: None,
            display: Some(ResultDisplay {
                change: Some(Change::between(Some("a\nb\n"), new_text)),
                ..ResultDisplay::default()
            }),
        };
        let mut edited = transcript();
        edited.apply(started("file_edit"));
        edited.apply(completed("file_edit", Some("a\nB\n")));
        assert!(matches!(
            items(&edited)[0],
            Item::ToolCall { change: Some(change), .. }
                if *change == Change::between(Some("a\nb\n"), Some("a\nB\n"))
        ));
        let mut deleted = transcript();
        deleted.apply(started("skill_delete"));
        deleted.apply(completed("skill_delete", None));
        assert!(matches!(
            items(&deleted)[0],
            Item::ToolCall { change: Some(change), .. }
                if *change == Change::between(Some("a\nb\n"), None)
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
            ran_under: RanUnder::Level(crate::permission::Permission::Read),
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

    /// A compaction landing live is a row in the words a replay of its boundary uses, with the
    /// same steps and summary under it, so a resumed session reads as this one did; the call that
    /// handed the summary in is the summary, so it is not a step. Its opening and a failure draw
    /// nothing, since the working row says a compaction runs.
    #[test]
    fn a_compaction_is_a_row_in_the_words_its_boundary_replays() {
        let rounds = vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "saving the notes first".into(),
                        opaque: None,
                    },
                    ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "memory_write".into(),
                        input: serde_json::json!({"name": "notes"}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call-1".into(),
                    content: vec![ToolResultContent::Text {
                        text: "Saved memory 'notes'.".into(),
                    }],
                    is_error: false,
                    display: None,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call-2".into(),
                    name: "context_replace".into(),
                    input: serde_json::json!({"summary": "a summary"}),
                }],
            },
        ];
        let record = crate::conversation::CompactionRecord {
            trigger: crate::conversation::CompactionTrigger::Manual,
            source: crate::conversation::CompactionSource::Checkpoint,
            memories_written: vec!["notes".to_string()],
            kept_recent: true,
            // A summary that carries an earlier summary's section heading over, as a second
            // compaction's often does: the record holds it as written.
            written_summary: "a summary\n\n[Checklist]\n\n1. carried over\n".to_string(),
            checkpoint: rounds,
        };
        let mut live = transcript();
        live.apply(FrontendEvent::CompactionStarted {
            id: "c0".to_string(),
            trigger: record.trigger,
        });
        live.apply(FrontendEvent::CompactionFailed {
            id: "c0".to_string(),
            trigger: record.trigger,
            error: "the summarizer failed".to_string(),
            interrupted: false,
        });
        live.apply(FrontendEvent::CompactionStarted {
            id: "c1".to_string(),
            trigger: record.trigger,
        });
        assert!(items(&live).is_empty(), "{:?}", items(&live));
        live.apply(FrontendEvent::Compacted {
            id: "c1".to_string(),
            record: record.clone(),
            generation: 1,
        });
        let mut resumed = transcript();
        resumed.load_page_above(page(
            vec![Event::CompactBoundary {
                summary: Message::user(format!(
                    "{}\n\na summary\n\n[Checklist]\n\n1. carried over\n\n[Checklist]\n\n\
                     [Continue the work.]",
                    crate::conversation::COMPACTION_SUMMARY_PREFIX,
                )),
                replaced_count: 3,
                record: record.clone(),
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: Default::default(),
            }],
            1,
            false,
        ));
        let shown = items(&live);
        assert_eq!(
            shown,
            items(&resumed),
            "the replay shows what the live run did"
        );
        let [
            Item::Compaction {
                headline,
                departures,
            },
            Item::Thinking { .. },
            Item::ToolCall {
                name,
                result: Some(_),
                ..
            },
            Item::CompactionSummary { text },
        ] = shown.as_slice()
        else {
            panic!("a compaction row and its three items: {shown:?}");
        };
        assert_eq!(
            headline, "Compacted the conversation on user request.",
            "the row says what set it going, and the memory it wrote is one of its steps"
        );
        assert!(departures.is_empty(), "{departures:?}");
        assert_eq!(name, "memory_write");
        assert_eq!(text, "a summary\n\n[Checklist]\n\n1. carried over");
        for transcript in [&live, &resumed] {
            let entries: Vec<&Entry> = transcript.drawn().collect();
            let head = entries.first().map(|entry| entry.id);
            assert!(
                entries
                    .iter()
                    .skip(1)
                    .all(|entry| entry.compaction.is_some() && entry.compaction == head),
                "every item after the row is the compaction's own: {entries:?}"
            );
        }
        assert!(
            record
                .note()
                .starts_with("Compacted the conversation on user request. Wrote 1 memory: notes."),
        );
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

    /// The token estimate never runs backwards, a block that opens anew starts its own count, and
    /// one that ended with no words is let go, as a replay of the log draws none.
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
        assert_eq!(shown.len(), 1, "{shown:?}");
        assert!(matches!(shown[0], Item::Thinking {
            estimate: Some(20),
            ..
        }));
    }

    /// A replayed `file_edit` the log recorded no change for stands for the change its input
    /// describes; a failed one stands for none, and a write, whose old content is not in its
    /// input, for none either.
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
                            display: None,
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
            Some(Change::between(Some("a\n"), Some("b\n")))
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

    /// A call resumed from the log draws as the live run drew it: the change with the file's own
    /// numbers, the code a command exited with, the argument an MCP call is shown by and the
    /// checklist a batch left, because the log records with each result what the screen draws.
    #[test]
    fn a_resumed_call_reads_as_the_live_one() {
        use crate::frontend::ToolOutputMetadata;
        let calls: Vec<(&str, serde_json::Value, bool, Option<ToolOutputMetadata>)> = vec![
            (
                "file_edit",
                serde_json::json!({"path": "a.txt", "old_string": "b\n", "new_string": "B\n"}),
                false,
                Some(ToolOutputMetadata::Diff {
                    path: None,
                    old_text: Some("a\nb\nc\n".into()),
                    new_text: Some("a\nB\nc\n".into()),
                }),
            ),
            (
                "file_write",
                serde_json::json!({"path": "new.txt", "content": "one\n"}),
                false,
                Some(ToolOutputMetadata::Diff {
                    path: None,
                    old_text: Some("old\n".into()),
                    new_text: Some("one\n".into()),
                }),
            ),
            (
                "shell_execute",
                serde_json::json!({"command": "make\nmake check"}),
                true,
                Some(ToolOutputMetadata::CommandExit {
                    exit_code: Some(2),
                    signal: None,
                }),
            ),
            (
                "mcp__exa__web_search_exa",
                serde_json::json!({"query": "ratatui", "numResults": 3}),
                false,
                None,
            ),
            (
                "checklist_add",
                serde_json::json!({"items": ["first thing"]}),
                false,
                None,
            ),
        ];
        let listed = vec![crate::checklist::ChecklistItem {
            id: 1,
            text: "first thing".into(),
            status: crate::checklist::ChecklistStatus::Pending,
            reason: None,
            task: None,
        }];
        let text = |name: &str| match name {
            "shell_execute" => "built\n\nExit code: 2".to_string(),
            _ => "done".to_string(),
        };
        // What the dispatcher records with each result and announces with its end.
        let display_of = |name: &str, metadata: &Option<ToolOutputMetadata>| {
            let mut display = metadata
                .as_ref()
                .map(ToolOutputMetadata::display)
                .unwrap_or_default();
            if crate::tools::checklist::changes_the_list(name) {
                display.checklist = Some(listed.clone());
            }
            display.recorded()
        };

        let mut live = transcript();
        for (index, (name, input, is_error, metadata)) in calls.iter().enumerate() {
            let id = format!("call-{index}");
            live.apply(FrontendEvent::ToolCallStarted {
                id: id.clone(),
                name: (*name).into(),
                input: input.clone(),
                display_summary: crate::tools::resolve_primary_param(name, input),
            });
            live.apply(FrontendEvent::ToolCallOutputDelta {
                id: id.clone(),
                chunk: "streamed while it ran\n".into(),
            });
            live.apply(FrontendEvent::ToolCallCompleted {
                id,
                name: (*name).into(),
                is_error: *is_error,
                content: vec![ToolResultContent::Text { text: text(name) }],
                metadata: metadata.clone(),
                display: display_of(name, metadata),
            });
        }
        live.apply(FrontendEvent::ChecklistUpdated {
            items: listed.clone(),
        });

        let uses = calls
            .iter()
            .enumerate()
            .map(|(index, (name, input, ..))| ContentBlock::ToolUse {
                id: format!("call-{index}"),
                name: (*name).into(),
                input: input.clone(),
            })
            .collect();
        let results = calls
            .iter()
            .enumerate()
            .map(
                |(index, (name, _, is_error, metadata))| ContentBlock::ToolResult {
                    tool_use_id: format!("call-{index}"),
                    content: vec![ToolResultContent::Text { text: text(name) }],
                    is_error: *is_error,
                    display: display_of(name, metadata),
                },
            )
            .collect();
        let mut resumed = transcript();
        resumed.load_page_above(page(
            vec![
                Event::Append(Message {
                    role: Role::Assistant,
                    content: uses,
                }),
                Event::Append(Message {
                    role: Role::User,
                    content: results,
                }),
            ],
            1,
            false,
        ));

        let drawn = |transcript: &Transcript| -> Vec<String> {
            transcript
                .drawn()
                .flat_map(|entry| {
                    super::super::view::entry_lines(entry, 80, std::path::Path::new("/w"))
                })
                .map(|line| {
                    line.spans
                        .iter()
                        .map(|span| span.content.as_ref())
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                })
                .collect()
        };
        let (live, resumed) = (drawn(&live), drawn(&resumed));
        assert_eq!(live, resumed, "the resumed calls draw as the live ones did");
        for expected in [
            "• file_edit a.txt  +1 -1",
            "    2 - b",
            "• file_write new.txt  +1 -1",
            "• shell_execute make  exit 2",
            "• mcp__exa__web_search_exa ratatui",
            "  └ □ first thing",
        ] {
            assert!(
                live.iter().any(|row| row == expected),
                "{expected:?} in {live:#?}"
            );
        }
        assert!(
            !live
                .iter()
                .any(|row| row.contains("Exit code") || row.contains("streamed")),
            "the code is on the row and the result stands for what streamed: {live:#?}"
        );
    }

    /// A batch of checklist calls replays the list it left after the batch, only when the list
    /// changed, and an emptied list draws nothing.
    #[test]
    fn a_replayed_checklist_shows_the_list_each_batch_left() {
        use crate::checklist::{ChecklistItem, ChecklistStatus};
        let item = |status| ChecklistItem {
            id: 1,
            text: "first thing".into(),
            status,
            reason: None,
            task: None,
        };
        let batch = |id: &str, list: Option<Vec<ChecklistItem>>| {
            vec![
                Event::Append(Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: id.into(),
                        name: "checklist_edit".into(),
                        input: serde_json::json!({"id": 1}),
                    }],
                }),
                Event::Append(Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: id.into(),
                        content: vec![ToolResultContent::Text { text: "ok".into() }],
                        is_error: false,
                        display: list.map(|items| ResultDisplay {
                            checklist: Some(items),
                            ..Default::default()
                        }),
                    }],
                }),
            ]
        };
        let mut events = batch("a", Some(vec![item(ChecklistStatus::Pending)]));
        events.extend(batch("b", Some(vec![item(ChecklistStatus::Pending)])));
        events.extend(batch("c", Some(vec![item(ChecklistStatus::InProgress)])));
        events.extend(batch("d", Some(Vec::new())));
        let mut transcript = transcript();
        transcript.load_page_above(page(events, 1, false));
        let lists: Vec<&Vec<ChecklistItem>> = items(&transcript)
            .into_iter()
            .filter_map(|item| match item {
                Item::Checklist { items } => Some(items),
                _ => None,
            })
            .collect();
        assert_eq!(
            lists,
            vec![&vec![item(ChecklistStatus::Pending)], &vec![item(
                ChecklistStatus::InProgress
            )],],
            "the unchanged list is not drawn again, and the emptied one draws nothing"
        );
    }

    /// What meka wrote into a result for the model alone is left out of the screen or said for
    /// a reader: the line naming an image, the tools a load made callable, the directions around
    /// a large output and around a call sent to the background.
    #[test]
    fn a_result_is_shown_without_the_directions_written_for_the_model() {
        let image = ToolResultContent::Image {
            source: crate::image::ImageSource::Base64 {
                media_type: "image/png".into(),
                data: String::new(),
            },
        };
        let text = |text: &str| ToolResultContent::Text { text: text.into() };
        assert_eq!(
            result_text(&[text("[Image: logo.png]"), image.clone()]),
            (String::new(), 1)
        );
        assert_eq!(
            result_text(&[text("[Image: logo.png]")]),
            ("[Image: logo.png]".to_string(), 0),
            "without an image, the line is what the tool said"
        );
        assert_eq!(
            result_text(&[ToolResultContent::ToolReference {
                tool_name: "mcp__exa__web_search_exa".into(),
            }]),
            (String::new(), 0)
        );
        let started = format!(
            "{} Use `task_list` for status.",
            crate::background::started_notice("7f3a1c22", "cargo build")
        );
        let detached = serde_json::json!({"command": "cargo build", "background": true});
        assert_eq!(
            shown_result(started.clone(), Some(&detached)),
            "Background task 7f3a1c22"
        );
        assert_eq!(
            shown_result(
                started.clone(),
                Some(&serde_json::json!({"command": "echo"}))
            ),
            started,
            "a call that ran in the foreground shows what it printed, whatever it reads like"
        );
        let spilled = "<large-output name=\"shell_execute_x_1\" size=\"47000\">\nOutput too \
                       large (45.9 KiB). Read it with `scratchpad_read`.\n\nPreview (first 5 \
                       bytes):\nhello\n...\n</large-output>";
        assert_eq!(result_text(&[text(spilled)]).0, "hello");
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
                    display: None,
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

    /// An invocation is said as what was invoked with the words typed with it under it, live and
    /// on a replay alike, never as the body it expanded to; the words an editor sent around a
    /// resource replay as one prompt, the resource as its mention.
    #[test]
    fn an_invocation_reads_alike_live_and_replayed() {
        let opening = Message {
            role: Role::User,
            content: vec![
                ContentBlock::TurnContext {
                    text: "[context]".into(),
                },
                ContentBlock::Invocation {
                    kind: crate::conversation::InvocationKind::Skill,
                    name: "review".into(),
                    directive: "the parser".into(),
                    text: "the parser\n\nBase directory for this skill: /skills/review".into(),
                },
                ContentBlock::Image {
                    source: crate::image::ImageSource::Blob {
                        hash: "0".repeat(64),
                        media_type: "image/png".into(),
                        size: 3,
                    },
                },
            ],
        };
        let mut live = transcript();
        live.apply(FrontendEvent::MessageAppended {
            message: opening.clone(),
            outcomes: Vec::new(),
            late: None,
        });
        let mut replayed = transcript();
        replayed.load_page_above(page(vec![Event::Append(opening)], 1, false));
        for shown in [items(&live), items(&replayed)] {
            assert_eq!(shown.len(), 3, "{shown:?}");
            assert!(
                matches!(shown[0], Item::Notice { text, .. } if text == "Running skill 'review'."),
                "{shown:?}"
            );
            assert!(
                matches!(shown[1], Item::Prompt { text, .. } if text == "the parser"),
                "{shown:?}"
            );
            assert!(matches!(shown[2], Item::Image(_)), "{shown:?}");
        }

        let mentioned = Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "compare ".into(),
                },
                ContentBlock::Resource {
                    uri: "file:///src/a.rs".into(),
                    name: "a.rs".into(),
                    text: "<resource uri=\"file:///src/a.rs\">fn a() {}</resource>".into(),
                },
                ContentBlock::Text {
                    text: " with the old one".into(),
                },
            ],
        };
        let mut replayed = transcript();
        replayed.load_page_above(page(vec![Event::Append(mentioned)], 1, false));
        let shown = items(&replayed);
        assert_eq!(shown.len(), 1, "{shown:?}");
        assert!(
            matches!(shown[0], Item::Prompt { text, .. } if text == "compare @a.rs with the old one"),
            "{shown:?}"
        );
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
                display: None,
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

    /// A compaction and a rewind show as rows of their own, the rewind counting the turns its
    /// stretch took; a repair shows nothing, and so does a rewind on a screen that could not read
    /// the session's stretches, since its row alone cannot say what it took.
    #[test]
    fn a_boundary_and_a_rewind_are_rows_and_a_repair_is_not() {
        let rows = || {
            vec![
                Event::CompactBoundary {
                    summary: Message::user("summary"),
                    replaced_count: 4,
                    record: crate::conversation::CompactionRecord::for_test(),
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
            ]
        };
        let mut transcript = transcript();
        transcript.set_log_end(Some(3));
        transcript.set_rewound(vec![crate::store::Rewound {
            rows: Vec::new(),
            at: 2,
            joined: false,
            turns: 1,
        }]);
        transcript.load_page_above(page(rows(), 1, false));
        let shown = items(&transcript);
        assert_eq!(shown.len(), 2, "{shown:?}");
        assert!(
            matches!(shown[0], Item::Compaction { headline, departures }
                if headline == "Autocompacted the conversation after context usage crossed the ceiling."
                    && departures.is_empty()),
            "{shown:?}"
        );
        let mut older = Transcript::new(Settings {
            show_session_id_on_create: false,
            turn_end_summary: TurnEndFact::DEFAULT_SUMMARY.to_vec(),
            fold_steps: true,
        });
        older.load_page_above(page(
            vec![Event::CompactBoundary {
                summary: Message::user("summary"),
                replaced_count: 4,
                record: crate::conversation::CompactionRecord {
                    memories_written: vec!["release-notes".to_string()],
                    ..crate::conversation::CompactionRecord::for_test()
                },
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: Default::default(),
            }],
            1,
            false,
        ));
        assert!(
            matches!(items(&older)[..], [
                Item::Compaction { .. },
                Item::ToolCall { name, input: Some(input), .. },
            ] if name == "memory_write" && input["name"] == "release-notes"),
            "a boundary that recorded no rounds shows the memories its record names: {:?}",
            items(&older)
        );
        assert!(matches!(shown[1], Item::Notice { text, .. } if text == "Rewound 1 turn"));
        let mut unread = self::transcript();
        unread.set_log_end(Some(3));
        unread.load_page_above(page(rows(), 1, false));
        assert_eq!(items(&unread).len(), 1, "{:?}", items(&unread));
    }

    /// The banner stands only while nothing but notices stand with it: the first prompt takes it
    /// down, and so does a page of the log, whichever door it came in by.
    #[test]
    fn the_banner_goes_when_the_conversation_starts() {
        let banner = || Item::Banner {
            version: "0.1.0".into(),
            revision: None,
            debug: false,
            mcp_servers: 0,
            skills: 0,
            memories: 0,
            instructions: 0,
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
            ran_under: RanUnder::Level(crate::permission::Permission::Read),
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
            fold_steps: true,
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
                    created_at: String::new(),
                    said: None,
                    event,
                })
                .collect(),
            more,
        }
    }

    fn closed_turn(id: Uuid, status: TurnStatus) -> TurnRecord {
        TurnRecord {
            id,
            source: crate::store::turns::TurnSource::Client,
            started_at: "2026-10-12T12:00:00+00:00".to_string(),
            ended_at: Some("2026-10-12T12:19:19+00:00".to_string()),
            status: Some(status),
            stop_reason: None,
            error: None,
            reason: None,
            profile: None,
            permission: crate::permission::Permission::Read,
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
                    ran_under: RanUnder::Unrecorded,
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

    /// A replay leaves out the rows a rewind took, with the rows closing their turns, and says at
    /// the rewind how many turns went from there. The turn before it closes ahead of that note,
    /// since a rewind is written between turns, and once, wherever a page's edge falls. A turn
    /// taking its prompt back to join a message to it is no rewind: the prompt shows once, in the
    /// message it joined, with no note, in a log that names its turns or one written before turns
    /// were recorded. A prompt withdrawn because the provider refused it stays, as it did live,
    /// with no note either.
    #[allow(
        clippy::single_range_in_vec_init,
        reason = "a stretch's rows are a list of runs, and these hold one run"
    )]
    #[test]
    fn a_replay_leaves_out_what_a_rewind_took() {
        use crate::store::Rewound;
        let turns = [
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        ];
        let taken = || Event::Repair {
            replaced_count: 1,
            messages: Vec::new(),
        };
        let log = vec![
            (Some(turns[0]), asked("one?")),
            (Some(turns[0]), said("One.")),
            (Some(turns[1]), asked("two?")),
            (Some(turns[1]), taken()),
            (Some(turns[1]), asked("two? and this")),
            (Some(turns[1]), said("Two.")),
            (Some(turns[2]), asked("three?")),
            (Some(turns[2]), said("Three.")),
            (None, taken()),
            (None, taken()),
            (Some(turns[3]), asked("four?")),
            (Some(turns[3]), said("Four.")),
        ];
        // Turn two joined a message to its prompt at row 4; turn three was rewound at row 9, then
        // turn two at row 10, which took from the prompt the joined message stands for.
        let joined = Rewound {
            rows: vec![3..=3],
            at: 4,
            joined: true,
            turns: 0,
        };
        let rewound = vec![
            joined.clone(),
            Rewound {
                rows: vec![7..=8],
                at: 9,
                joined: false,
                turns: 1,
            },
            Rewound {
                rows: vec![3..=9],
                at: 10,
                joined: false,
                turns: 2,
            },
        ];
        let page = |rows: &[(Option<Uuid>, Event)], first: i64, more: bool| LogPage {
            rows: rows
                .iter()
                .cloned()
                .enumerate()
                .map(|(offset, (turn_id, event))| LogRow {
                    id: first + offset as i64,
                    turn_id,
                    created_at: String::new(),
                    said: None,
                    event,
                })
                .collect(),
            more,
        };
        let replayed = |pages: Vec<LogPage>, rewound: Vec<Rewound>| {
            let mut transcript = with_summary(TurnEndFact::DEFAULT_SUMMARY.to_vec());
            transcript.set_turn_records(
                turns
                    .iter()
                    .map(|turn| closed_turn(*turn, TurnStatus::Succeeded))
                    .collect(),
            );
            transcript.set_rewound(rewound);
            for page in pages {
                transcript.load_page_above(page);
            }
            transcript
        };
        let whole = replayed(vec![page(&log, 1, false)], rewound.clone());
        assert_eq!(kinds(&whole), vec![
            "prompt", "text", "end", "note", "prompt", "text", "end"
        ]);
        assert!(
            matches!(items(&whole)[3], Item::Notice { text, .. } if text == "Rewound 2 turns"),
            "{:?}",
            items(&whole)
        );
        for edge in 1..log.len() {
            let split = replayed(
                vec![
                    page(&log[edge..], edge as i64 + 1, true),
                    page(&log[..edge], 1, false),
                ],
                rewound.clone(),
            );
            assert_eq!(kinds(&split), kinds(&whole), "split after row {edge}");
        }
        // A log written before turns were recorded names none, and a join in it is no rewind
        // either: its repair row says nothing.
        let unturned: Vec<(Option<Uuid>, Event)> = log[..6]
            .iter()
            .map(|(_, event)| (None, event.clone()))
            .collect();
        let joined_unturned = replayed(vec![page(&unturned, 1, false)], vec![joined.clone()]);
        assert_eq!(
            kinds(&joined_unturned),
            vec!["prompt", "text", "prompt", "text"],
            "{:?}",
            items(&joined_unturned)
        );
        let joined_only = replayed(vec![page(&log[..6], 1, false)], vec![joined]);
        assert_eq!(kinds(&joined_only), vec![
            "prompt", "text", "end", "prompt", "text", "end"
        ]);
        assert!(
            matches!(items(&joined_only)[3], Item::Prompt { text, .. } if text == "two? and this"),
            "{:?}",
            items(&joined_only)
        );
        let withdrawn = replayed(vec![page(&log[..4], 1, false)], Vec::new());
        assert_eq!(kinds(&withdrawn), vec![
            "prompt", "text", "end", "prompt", "end"
        ]);
    }

    /// A prompt a compaction kept, standing alone once turn retention took the row it was said
    /// in, is drawn in the level of the turn that said it, as it was live, not that of the turn
    /// whose compaction wrote the copy.
    #[test]
    fn a_kept_prompt_standing_alone_shows_the_level_it_was_said_under() {
        use crate::permission::Permission;
        let (said_in, compacting) = (Uuid::new_v4(), Uuid::new_v4());
        let mut transcript = with_summary(TurnEndFact::DEFAULT_SUMMARY.to_vec());
        let mut said_record = closed_turn(said_in, TurnStatus::Succeeded);
        said_record.permission = Permission::Workspace;
        transcript.set_turn_records(vec![
            said_record,
            closed_turn(compacting, TurnStatus::Succeeded),
        ]);
        let said = crate::store::SaidStamp {
            turn_id: Some(said_in),
            created_at: "2026-10-12T12:00:00+00:00".to_string(),
        };
        transcript.load_page_above(LogPage {
            rows: vec![LogRow {
                id: 7,
                turn_id: Some(compacting),
                created_at: String::new(),
                said: Some(said),
                event: asked("kept?"),
            }],
            more: false,
        });
        assert!(
            matches!(items(&transcript)[0], Item::Prompt {
                ran_under: RanUnder::Level(Permission::Workspace),
                ..
            }),
            "{:?}",
            items(&transcript)
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
            ran_under: RanUnder::Level(crate::permission::Permission::Read),
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

    /// The list a resumed session hydrates is the list its replay drew last, which says nothing
    /// new; a list that differs is drawn.
    #[test]
    fn a_resumed_session_shows_its_checklist_once() {
        let listed = vec![ChecklistItem {
            id: 1,
            text: "first thing".into(),
            status: crate::checklist::ChecklistStatus::Pending,
            reason: None,
            task: None,
        }];
        let mut transcript = transcript();
        // As a resume runs: the list the session hydrates is announced before the first page of
        // its log is read.
        transcript.set_log_end(Some(2));
        transcript.apply(FrontendEvent::ChecklistUpdated {
            items: listed.clone(),
        });
        transcript.load_page_above(page(
            vec![
                Event::Append(Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "checklist_add".into(),
                        input: serde_json::json!({"items": ["first thing"]}),
                    }],
                }),
                Event::Append(Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "c1".into(),
                        content: vec![ToolResultContent::Text { text: "ok".into() }],
                        is_error: false,
                        display: Some(ResultDisplay {
                            checklist: Some(listed.clone()),
                            ..ResultDisplay::default()
                        }),
                    }],
                }),
            ],
            1,
            false,
        ));
        let lists = |transcript: &Transcript| {
            items(transcript)
                .into_iter()
                .filter(|item| matches!(item, Item::Checklist { .. }))
                .count()
        };
        assert_eq!(lists(&transcript), 1);
        transcript.apply(FrontendEvent::ChecklistUpdated {
            items: listed.clone(),
        });
        assert_eq!(lists(&transcript), 1, "{:?}", items(&transcript));
        let mut moved = listed;
        moved[0].status = crate::checklist::ChecklistStatus::InProgress;
        transcript.apply(FrontendEvent::ChecklistUpdated { items: moved });
        assert_eq!(lists(&transcript), 2);
    }

    /// A reset lets go of what a replay of the log draws again in its own words, a nudge's stage
    /// direction among it, and keeps what the host and the agent said that the log does not: a
    /// warning, and a `!command` with its output.
    #[test]
    fn a_reset_keeps_what_the_log_does_not_say() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::Nudged {
            kind: crate::conversation::NudgeKind::Checklist,
            text: "[Checklist: items are still open.]".into(),
        });
        transcript.apply(FrontendEvent::Notice(Notice::checklist_nudged(2, 1, 3)));
        transcript.apply(FrontendEvent::Notice(Notice::checklist_left_open(
            2, 3, false,
        )));
        transcript.push(Item::Prompt {
            text: "!ls".into(),
            class: None,
            inbox_id: None,
            delivered: true,
            ran_under: RanUnder::Shell,
        });
        transcript.push(Item::ShellOutput("total 0".into()));
        transcript.reset(None);
        let kept: Vec<String> = items(&transcript)
            .into_iter()
            .map(|item| match item {
                Item::Notice { text, .. } => text.clone(),
                Item::Prompt { text, .. } => text.clone(),
                Item::ShellOutput(text) => text.clone(),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(kept, vec![
            "Checklist: 2 items left open after 3 nudges".to_string(),
            "!ls".to_string(),
            "total 0".to_string(),
        ]);
    }

    /// A thinking block with no words is no step, live or on a replay of the log: the live run
    /// lets its counter go as it ends, a whole block that comes with no words is not drawn, and
    /// a replay draws none, so the folds count alike.
    #[test]
    fn a_thinking_block_with_no_words_is_no_step() {
        let mut live = transcript();
        live.apply(FrontendEvent::ThinkingProgress {
            estimated_tokens: Some(40),
        });
        live.apply(FrontendEvent::ThinkingEnded);
        live.apply(FrontendEvent::ThinkingBlock {
            content: String::new(),
        });
        assert!(items(&live).is_empty(), "{:?}", items(&live));
        let mut replayed = transcript();
        replayed.load_page_above(page(
            vec![Event::Append(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Thinking {
                    thinking: String::new(),
                    opaque: None,
                }],
            })],
            1,
            false,
        ));
        assert!(items(&replayed).is_empty(), "{:?}", items(&replayed));
    }

    /// What the log never holds is not left drawn live: answer text of spaces alone once its
    /// stream closes, and a call still being written when its turn ended.
    #[test]
    fn what_the_log_never_holds_is_not_left_drawn() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::AssistantTextDelta("  \n".into()));
        transcript.apply(FrontendEvent::ToolCallComposing {
            id: "c1".into(),
            name: "shell_execute".into(),
        });
        assert!(
            matches!(items(&transcript)[..], [Item::ToolCall { .. }]),
            "{:?}",
            items(&transcript)
        );
        transcript.apply(FrontendEvent::TurnEnded {
            turn_id: Uuid::new_v4(),
            ending: crate::store::turns::TurnEnding::succeeded(
                "end_turn",
                crate::store::turns::TurnUsage::default(),
            ),
        });
        assert!(
            !items(&transcript)
                .iter()
                .any(|item| matches!(item, Item::ToolCall { .. })),
            "{:?}",
            items(&transcript)
        );
    }

    /// A result read on a page below goes to the last call with its id on the page above that
    /// has none, which is its own where a provider reuses call ids round to round.
    #[test]
    fn a_result_from_the_page_below_goes_to_its_own_call() {
        let call = || {
            Event::Append(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_0".into(),
                    name: "shell_execute".into(),
                    input: serde_json::json!({"command": "ls"}),
                }],
            })
        };
        let result = |text: &str| {
            Event::Append(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_0".into(),
                    content: vec![ToolResultContent::Text { text: text.into() }],
                    is_error: false,
                    display: None,
                }],
            })
        };
        let results = |above: Vec<Event>| {
            let rows = above.len() as i64;
            let mut transcript = transcript();
            transcript.load_page_above(page(vec![result("second")], rows + 1, true));
            transcript.load_page_above(page(above, 1, false));
            items(&transcript)
                .into_iter()
                .filter_map(|item| match item {
                    Item::ToolCall { result, .. } => Some(result.clone()),
                    _ => None,
                })
                .collect::<Vec<Option<String>>>()
        };
        assert_eq!(
            results(vec![call(), result("first"), call()]),
            vec![Some("first".to_string()), Some("second".to_string())],
            "the result goes to the call that has none"
        );
        assert_eq!(
            results(vec![call(), call(), result("first")]),
            vec![Some("second".to_string()), Some("first".to_string())],
            "a call that has its result keeps it"
        );
        assert_eq!(
            results(vec![call(), call()]),
            vec![None, Some("second".to_string())],
            "of two calls with none, the later is the one the result follows"
        );
    }

    /// What the screen draws for an event that comes before the log has a row for it stands
    /// where the log puts it: a compaction before the turn opens, and a report its opening
    /// carries ahead of the words, above the prompt typed for it.
    #[test]
    fn what_comes_before_a_turn_s_words_stands_above_its_prompt() {
        let pending = Item::Prompt {
            text: "typed".into(),
            class: None,
            inbox_id: None,
            delivered: true,
            ran_under: RanUnder::Pending,
        };
        let shape = |transcript: &Transcript| -> Vec<&'static str> {
            items(transcript)
                .into_iter()
                .map(|item| match item {
                    Item::Prompt { .. } => "prompt",
                    Item::Compaction { .. } => "compaction",
                    Item::Notice { .. } => "note",
                    _ => "other",
                })
                .collect()
        };
        let mut compacted = transcript();
        compacted.push(pending.clone());
        compacted.apply(FrontendEvent::TurnStarted {
            turn_id: Uuid::new_v4(),
        });
        compacted.apply(FrontendEvent::Compacted {
            id: "c1".into(),
            record: crate::conversation::CompactionRecord::for_test(),
            generation: 1,
        });
        assert_eq!(shape(&compacted), vec!["compaction", "prompt"]);
        let mut reported = transcript();
        reported.push(pending);
        reported.apply(FrontendEvent::TurnStarted {
            turn_id: Uuid::new_v4(),
        });
        reported.apply(FrontendEvent::TurnOpened {
            permission: crate::permission::Permission::Read,
        });
        reported.apply(FrontendEvent::MessageAppended {
            message: Message {
                role: Role::User,
                content: vec![
                    ContentBlock::BackgroundReport {
                        task_ids: Vec::new(),
                        text: "[Background task reporting]".into(),
                    },
                    ContentBlock::Text {
                        text: "typed".into(),
                    },
                ],
            },
            outcomes: Vec::new(),
            late: None,
        });
        assert_eq!(shape(&reported), vec!["note", "prompt"]);
    }

    /// A call still being written when its turn ended is let go without moving the answer the
    /// turn was streaming off its place, so the answer still closes when the turn's last row
    /// comes.
    #[test]
    fn letting_a_half_written_call_go_keeps_the_answer_open_where_it_is() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::ToolCallComposing {
            id: "c1".into(),
            name: "shell_execute".into(),
        });
        transcript.apply(FrontendEvent::AssistantTextDelta("- one\n  two".into()));
        transcript.apply(FrontendEvent::TurnEnded {
            turn_id: Uuid::new_v4(),
            ending: crate::store::turns::TurnEnding::failed(
                &crate::error::MekaError::Interrupted,
                crate::store::turns::TurnUsage::default(),
            ),
        });
        transcript.push(Item::Error("Stopped".into()));
        assert!(
            matches!(items(&transcript)[0], Item::Text {
                streaming: false,
                ..
            }),
            "{:?}",
            items(&transcript)
        );
    }

    /// A checklist nudge's counted direction stands for the nudge, which a replay of the log says
    /// in its own words, so a reset lets it go; a nudge that says nothing marks no notice, and the
    /// warning after it stays.
    #[test]
    fn only_a_checklist_nudge_s_direction_is_the_log_s() {
        let notes = |transcript: &Transcript| -> Vec<String> {
            items(transcript)
                .into_iter()
                .filter_map(|item| match item {
                    Item::Notice { text, .. } => Some(text.clone()),
                    _ => None,
                })
                .collect()
        };
        let mut nudged = transcript();
        nudged.apply(FrontendEvent::Nudged {
            kind: crate::conversation::NudgeKind::Checklist,
            text: "[Checklist]".into(),
        });
        nudged.apply(FrontendEvent::Notice(Notice::checklist_nudged(2, 1, 3)));
        assert_eq!(notes(&nudged), vec![
            "Checklist: 2 open items, continuing, nudge 1 of 3".to_string()
        ]);
        nudged.reset(None);
        assert!(notes(&nudged).is_empty(), "the log says it");
        let mut quiet = transcript();
        quiet.apply(FrontendEvent::Nudged {
            kind: crate::conversation::NudgeKind::VisibleReply,
            text: "[Answer]".into(),
        });
        quiet.apply(FrontendEvent::Notice(Notice::checklist_left_open(
            1, 3, false,
        )));
        assert_eq!(notes(&quiet), vec![
            "Checklist: 1 item left open after 3 nudges".to_string()
        ]);
        quiet.reset(None);
        assert_eq!(notes(&quiet).len(), 1, "the log does not say it");
    }

    /// A prompt sent for a turn that never opened, one refused before it began, is drawn as no
    /// turn's once the host has answered, and a later turn's opening does not take it.
    #[test]
    fn a_refused_prompt_takes_no_later_turn_s_level() {
        let mut transcript = transcript();
        transcript.push(Item::Prompt {
            text: "first".into(),
            class: None,
            inbox_id: None,
            delivered: true,
            ran_under: RanUnder::Pending,
        });
        transcript.settle_unopened_prompts();
        transcript.apply(FrontendEvent::TurnOpened {
            permission: crate::permission::Permission::Workspace,
        });
        assert!(
            matches!(items(&transcript)[0], Item::Prompt {
                ran_under: RanUnder::Unrecorded,
                ..
            }),
            "{:?}",
            items(&transcript)
        );
    }

    /// Switching to another session lets go of everything the screen showed of the last, its
    /// notes and its shell's output among them.
    #[test]
    fn switching_sessions_lets_the_last_one_go() {
        let mut transcript = transcript();
        transcript.push(Item::note("Connected to 'exa'"));
        transcript.push(Item::ShellOutput("marker".into()));
        transcript.forget_session();
        assert!(items(&transcript).is_empty(), "{:?}", items(&transcript));
    }

    /// Another client's message that lands at a round boundary is drawn as a prompt, and one the
    /// screen enqueued itself is not drawn twice.
    #[test]
    fn another_client_s_message_is_drawn_where_it_lands() {
        let mine = Uuid::new_v4();
        let mut transcript = transcript();
        transcript.push(Item::Prompt {
            text: "mine".into(),
            class: Some(InboxClass::Steer),
            inbox_id: Some(mine),
            delivered: false,
            ran_under: RanUnder::Pending,
        });
        let block = |id: Uuid, body: &str| ContentBlock::Inbox {
            item_id: Some(id),
            class: Some(InboxClass::Steer),
            source: None,
            body: body.into(),
            text: body.into(),
        };
        transcript.apply(FrontendEvent::InboxAppended {
            blocks: vec![block(mine, "mine"), block(Uuid::new_v4(), "from afar")],
        });
        let prompts: Vec<&str> = items(&transcript)
            .into_iter()
            .filter_map(|item| match item {
                Item::Prompt { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(prompts, vec!["mine", "from afar"]);
    }

    fn succeeded() -> FrontendEvent {
        FrontendEvent::TurnEnded {
            turn_id: Uuid::new_v4(),
            ending: crate::store::turns::TurnEnding::succeeded(
                "end_turn",
                crate::store::turns::TurnUsage::default(),
            ),
        }
    }

    /// A batch's checklist is replayed where the live screen drew it: after the batch's results
    /// and ahead of a report and a message that rode the same message, which live are announced
    /// after the list.
    #[test]
    fn a_replayed_batch_s_checklist_comes_before_what_arrived_behind_it() {
        use crate::checklist::{ChecklistItem, ChecklistStatus};
        let list = vec![ChecklistItem {
            id: 1,
            text: "first thing".into(),
            status: ChecklistStatus::Pending,
            reason: None,
            task: None,
        }];
        let events = vec![
            Event::Append(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "a".into(),
                    name: "checklist_add".into(),
                    input: serde_json::json!({"items": ["first thing"]}),
                }],
            }),
            Event::Append(Message {
                role: Role::User,
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "a".into(),
                        content: vec![ToolResultContent::Text { text: "ok".into() }],
                        is_error: false,
                        display: Some(ResultDisplay {
                            checklist: Some(list.clone()),
                            ..Default::default()
                        }),
                    },
                    ContentBlock::BackgroundReport {
                        task_ids: vec!["7f3a1c22".into()],
                        text: "[Background task reporting]".into(),
                    },
                    ContentBlock::Inbox {
                        item_id: None,
                        class: Some(InboxClass::Steer),
                        source: None,
                        body: "look here".into(),
                        text: "look here".into(),
                    },
                ],
            }),
        ];
        let mut transcript = transcript();
        transcript.load_page_above(page(events, 1, false));
        let kinds: Vec<&str> = items(&transcript)
            .into_iter()
            .map(|item| match item {
                Item::ToolCall { .. } => "call",
                Item::Checklist { .. } => "checklist",
                Item::Notice { .. } => "report",
                Item::Prompt { .. } => "prompt",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, ["call", "checklist", "report", "prompt"]);
    }

    /// A call the turn never ran, written whole and cut before it was run, goes when the turn
    /// ends, as one still being written does: the log holds neither.
    #[test]
    fn a_call_the_turn_never_ran_goes_when_it_ends() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::ToolCallStarted {
            id: "ran".into(),
            name: "file_read".into(),
            input: serde_json::json!({"path": "a.txt"}),
            display_summary: Some("a.txt".into()),
        });
        transcript.apply(FrontendEvent::ToolCallCompleted {
            id: "ran".into(),
            name: "file_read".into(),
            is_error: false,
            content: Vec::new(),
            metadata: None,
            display: None,
        });
        transcript.apply(FrontendEvent::ToolCallStarted {
            id: "cut".into(),
            name: "file_read".into(),
            input: serde_json::json!({"path": "b.txt"}),
            display_summary: Some("b.txt".into()),
        });
        transcript.apply(succeeded());
        let calls: Vec<&str> = items(&transcript)
            .into_iter()
            .filter_map(|item| match item {
                Item::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, ["ran"]);
    }

    /// A call its round dropped goes at once, while the turn goes on, and a later call under the
    /// same id keeps its row: the log holds none of the dropped one.
    #[test]
    fn a_call_its_round_dropped_goes_while_the_turn_goes_on() {
        let mut transcript = transcript();
        let started = |path: &str| FrontendEvent::ToolCallStarted {
            id: "call_0".into(),
            name: "file_read".into(),
            input: serde_json::json!({ "path": path }),
            display_summary: Some(path.to_string()),
        };
        transcript.apply(started("a.txt"));
        transcript.apply(FrontendEvent::ToolCallDropped {
            id: "call_0".into(),
            started: true,
        });
        transcript.apply(started("b.txt"));
        let calls: Vec<String> = items(&transcript)
            .into_iter()
            .filter_map(|item| match item {
                Item::ToolCall { summary, .. } => Some(format!("{summary:?}")),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert!(calls[0].contains("b.txt"), "{calls:?}");
    }

    /// A drop names the call by its state as well as its id: one whose arguments never finished
    /// goes, while a running call under another drop's id stays.
    #[test]
    fn a_dropped_call_goes_by_its_state() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::ToolCallComposing {
            id: "a".into(),
            name: "file_write".into(),
        });
        transcript.apply(FrontendEvent::ToolCallDropped {
            id: "a".into(),
            started: false,
        });
        transcript.apply(FrontendEvent::ToolCallStarted {
            id: "b".into(),
            name: "file_read".into(),
            input: serde_json::json!({"path": "b.txt"}),
            display_summary: Some("b.txt".into()),
        });
        transcript.apply(FrontendEvent::ToolCallDropped {
            id: "b".into(),
            started: false,
        });
        let calls: Vec<&str> = items(&transcript)
            .into_iter()
            .filter_map(|item| match item {
                Item::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, ["b"]);
    }

    /// A followup typed while a turn ran is drawn again at the end once it is sent, where the
    /// turn it opens begins, as a replay draws it; the copy that waited where it was typed goes.
    #[test]
    fn a_sent_followup_is_drawn_where_its_turn_begins() {
        let mut transcript = transcript();
        let waiting = transcript.push(Item::Prompt {
            text: "second please".into(),
            class: Some(InboxClass::Followup),
            inbox_id: None,
            delivered: false,
            ran_under: RanUnder::Pending,
        });
        transcript.push(Item::note("Read 2 files"));
        transcript.apply(succeeded());
        transcript.send_followup(waiting, "second please".into());
        let shown = items(&transcript);
        assert!(
            matches!(shown.last(), Some(Item::Prompt { text, class: None, delivered: true, .. })
                if text == "second please"),
            "{shown:?}"
        );
        let prompts = shown
            .iter()
            .filter(|item| matches!(item, Item::Prompt { .. }))
            .count();
        assert_eq!(prompts, 1, "{shown:?}");
    }

    /// A reset lets go of what was said while a turn ran, whose place the replay cannot give back,
    /// and keeps what the host said between turns and the shell's commands.
    #[test]
    fn a_reset_lets_go_of_what_a_turn_said_and_keeps_what_was_said_between_turns() {
        let mut transcript = transcript();
        transcript.push(Item::note("Connected to 'exa'"));
        transcript.apply(FrontendEvent::TurnStarted {
            turn_id: Uuid::new_v4(),
        });
        transcript.push(Item::Notice {
            level: NoticeLevel::Warn,
            text: "Interrupted".into(),
        });
        transcript.push(Item::Error("provider error: down".into()));
        transcript.push(Item::Prompt {
            text: "ls".into(),
            class: None,
            inbox_id: None,
            delivered: true,
            ran_under: RanUnder::Shell,
        });
        transcript.apply(succeeded());
        transcript.reset(None);
        let shown = items(&transcript);
        assert!(
            matches!(shown[..], [Item::Notice { .. }, Item::Prompt { .. }]),
            "{shown:?}"
        );
        assert!(
            matches!(shown[0], Item::Notice { text, .. } if text == "Connected to 'exa'"),
            "{shown:?}"
        );
    }

    /// A steer the screen drew that its turn took ends delivered, in the turn's level, when the
    /// turn ends before the provider read it, as a replay of the log draws it.
    #[test]
    fn a_drawn_steer_its_turn_took_is_delivered_when_the_turn_ends() {
        use crate::permission::Permission;
        let mine = Uuid::new_v4();
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::TurnOpened {
            permission: Permission::Workspace,
        });
        transcript.push(Item::Prompt {
            text: "mine".into(),
            class: Some(InboxClass::Steer),
            inbox_id: Some(mine),
            delivered: false,
            ran_under: RanUnder::Pending,
        });
        transcript.apply(FrontendEvent::InboxAppended {
            blocks: vec![ContentBlock::Inbox {
                item_id: Some(mine),
                class: Some(InboxClass::Steer),
                source: None,
                body: "mine".into(),
                text: "mine".into(),
            }],
        });
        transcript.apply(FrontendEvent::TurnEnded {
            turn_id: Uuid::new_v4(),
            ending: crate::store::turns::TurnEnding::failed(
                &crate::error::MekaError::Usage("down".into()),
                crate::store::turns::TurnUsage::default(),
            ),
        });
        assert!(
            matches!(items(&transcript)[0], Item::Prompt {
                delivered: true,
                ran_under: RanUnder::Level(Permission::Workspace),
                ..
            }),
            "{:?}",
            items(&transcript)
        );
    }

    /// The checklist a resumed session is announced with while its log is still being read waits
    /// for the first page, whichever door resumed it, and lands where it was announced, never
    /// inside a turn that streams meanwhile; one the page draws last is not drawn again.
    #[test]
    fn a_resumed_session_s_checklist_waits_for_its_first_page() {
        use crate::checklist::{ChecklistItem, ChecklistStatus};
        let list = vec![ChecklistItem {
            id: 1,
            text: "first thing".into(),
            status: ChecklistStatus::Pending,
            reason: None,
            task: None,
        }];
        let mut transcript = transcript();
        transcript.push(Item::note("from the session before"));
        transcript.forget_session();
        transcript.apply(FrontendEvent::ChecklistUpdated {
            items: list.clone(),
        });
        transcript.apply(FrontendEvent::AssistantTextDelta("an answer ".into()));
        assert!(
            !items(&transcript)
                .iter()
                .any(|item| matches!(item, Item::Checklist { .. })),
            "held for the page: {:?}",
            items(&transcript)
        );
        transcript.load_page_above(page(vec![said("before")], 1, false));
        transcript.apply(FrontendEvent::AssistantTextDelta("goes on".into()));
        let shown = items(&transcript);
        assert!(
            matches!(shown[..], [Item::Text { .. }, Item::Checklist { .. }, Item::Text { text, .. }]
                if text == "an answer goes on"),
            "{shown:?}"
        );
    }

    /// Another notice between a checklist nudge and its stage direction leaves the nudge waiting
    /// for it, so the direction is still the one a reload lets go of for the replay's.
    #[test]
    fn a_notice_between_a_nudge_and_its_direction_leaves_the_nudge_waiting() {
        let mut transcript = transcript();
        transcript.apply(FrontendEvent::Nudged {
            kind: crate::conversation::NudgeKind::Checklist,
            text: "items are open".into(),
        });
        transcript.apply(FrontendEvent::Notice(Notice::info("Connected to 'exa'")));
        transcript.apply(FrontendEvent::Notice(Notice::checklist_nudged(1, 1, 3)));
        transcript.reset(None);
        let shown = items(&transcript);
        assert!(
            matches!(shown[..], [Item::Notice { text, .. }] if text == "Connected to 'exa'"),
            "{shown:?}"
        );
    }

    /// A report riding a prompt of images alone stands ahead of the images, as a replay of the log
    /// draws the message.
    #[test]
    fn a_report_riding_an_image_prompt_stands_ahead_of_it() {
        use crate::permission::Permission;
        let mut transcript = transcript();
        transcript.push(Item::Image(crate::render::tui::Picture::unread()));
        transcript.apply(FrontendEvent::TurnStarted {
            turn_id: Uuid::new_v4(),
        });
        transcript.apply(FrontendEvent::TurnOpened {
            permission: Permission::Read,
        });
        transcript.apply(FrontendEvent::MessageAppended {
            message: Message {
                role: Role::User,
                content: vec![
                    ContentBlock::BackgroundReport {
                        task_ids: vec!["7f3a1c22".into()],
                        text: "[Background task reporting]".into(),
                    },
                    ContentBlock::Image {
                        source: crate::image::ImageSource::Base64 {
                            media_type: "image/png".into(),
                            data: String::new(),
                        },
                    },
                ],
            },
            outcomes: Vec::new(),
            late: None,
        });
        assert!(
            matches!(items(&transcript)[..], [
                Item::Notice { .. },
                Item::Image(_)
            ]),
            "{:?}",
            items(&transcript)
        );
    }

    /// An item is flipped by its id, wherever a page loaded above has moved it since the frame
    /// it was picked in.
    #[test]
    fn an_item_is_toggled_by_its_id_wherever_it_has_moved() {
        let mut transcript = transcript();
        transcript.set_log_end(Some(1));
        transcript.apply(FrontendEvent::ThinkingDelta("a thought".into()));
        transcript.apply(FrontendEvent::AssistantTextDelta("then words".into()));
        let id = transcript
            .drawn()
            .find(|entry| matches!(entry.item, Item::Thinking { .. }))
            .map(|entry| entry.id)
            .expect("the thought");
        let before = transcript
            .drawn()
            .find(|entry| entry.id == id)
            .map(|entry| entry.expanded);
        transcript.load_page_above(page(vec![said("older")], 1, false));
        assert!(transcript.toggle_item(id));
        let after = transcript
            .drawn()
            .find(|entry| entry.id == id)
            .map(|entry| entry.expanded);
        assert_eq!(after, before.map(|expanded| !expanded));
    }

    /// A call announced without a composing first, as a blocking reply announces it, is a call
    /// of its own even where an earlier round's call had the same id.
    #[test]
    fn a_started_call_never_takes_an_earlier_call_with_its_id() {
        let mut transcript = transcript();
        for command in ["echo alpha", "echo beta"] {
            transcript.apply(FrontendEvent::ToolCallStarted {
                id: "call_0".into(),
                name: "shell_execute".into(),
                input: serde_json::json!({"command": command}),
                display_summary: Some(command.into()),
            });
            transcript.apply(FrontendEvent::ToolCallCompleted {
                id: "call_0".into(),
                name: "shell_execute".into(),
                is_error: false,
                content: Vec::new(),
                metadata: None,
                display: None,
            });
        }
        let calls = items(&transcript)
            .into_iter()
            .filter(|item| matches!(item, Item::ToolCall { .. }))
            .count();
        assert_eq!(calls, 2);
    }
}
