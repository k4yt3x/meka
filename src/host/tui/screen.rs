//! The screen: the task that owns the terminal, drains what the host and the agent send, reads the
//! keyboard and the mouse, and draws a frame when something changed.

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use base64::Engine as _;
use crossterm::event::{Event, KeyCode, KeyEvent, MouseButton, MouseEvent, MouseEventKind};
use futures::StreamExt;
use ratatui::{
    DefaultTerminal, Frame,
    layout::Rect,
    style::Style,
    text::{Line, Span, Text},
    widgets::Paragraph,
};

use super::{
    commands::{Answerer, SlashCommand, parse_slash_command, unknown_command_message},
    frontend::{ApprovalDecision, ApprovalRequest, HostEvent, UiEvent},
    input::{
        Action, Completer, Composer, ComposerState, Focus, action_for, handle_cd,
        shorten_path_with_tilde,
    },
    model::{Item, Transcript},
    panel::{Choice, Panel},
    view::{self, StatusFacts, Viewport},
};
use crate::{
    frontend::{ElicitationKind, ElicitationPrompt, ElicitationResponse},
    host::terminal::{Escalation, INTERRUPT_RELAY, escalation_for},
    permission::SharedPermission,
    render::tui::{self as draw, TEXT_EDGE, gray},
    store::inbox::InboxClass,
    streams::capture,
};

/// Log lines kept for `/log`: enough to read back a turn's worth, little enough to hold.
const LOG_RING_LINES: usize = 500;

/// How often the screen redraws while something is changing.
const FRAME_INTERVAL: Duration = Duration::from_millis(33);
/// How long a hint stays in the status row.
const HINT_LIFETIME: Duration = Duration::from_secs(6);
/// The composer grows up to this share of the screen before it scrolls inside.
const COMPOSER_SHARE: u16 = 3;
/// A panel takes up to this share of the screen.
const PANEL_SHARE: u16 = 2;
/// A drag over the transcript: where it started and where the pointer is, in the frame's
/// coordinates, and whether it moved at all, which is what tells a drag from a click.
#[derive(Debug, Clone, Copy)]
struct Selection {
    anchor: (u16, u16),
    head: (u16, u16),
    dragged: bool,
}

impl Selection {
    /// The selection's ends in reading order: the earlier row first, the earlier column on one row.
    fn ordered(self) -> ((u16, u16), (u16, u16)) {
        let (a, b) = (self.anchor, self.head);
        if (a.1, a.0) <= (b.1, b.0) {
            (a, b)
        } else {
            (b, a)
        }
    }
}
/// What the status row says when a question is taken off the screen unanswered.
const QUESTION_WITHDRAWN_HINT: &str = "the question was withdrawn";
/// How often the status row's count of running work is read back from the store.
const WORK_POLL_INTERVAL: Duration = Duration::from_secs(1);
/// The sessions the picker offers: the newest, since an older one is reached by `/resume <id>`.
const SESSION_PICKER_ROWS: u32 = 50;
/// The columns a session's title keeps on a picker row however wide what follows it is.
const SESSION_TITLE_FLOOR: usize = 16;
/// The most of a picker row the directory column may take, as a share of its width; a longer
/// directory is cut from its start, the end that repeats from row to row.
const SESSION_DIRECTORY_SHARE: usize = 4;
/// How long a finished `!command` is given to hand over the rest of its output: a process it
/// started and left behind can hold the pipe open past its own exit.
const SHELL_DRAIN_GRACE: Duration = Duration::from_secs(2);
/// What a `!command` prints is kept to this much per stream, the tail, so a command that never
/// stops printing cannot grow the screen's memory with it.
const SHELL_OUTPUT_KEEP_BYTES: usize = crate::text::MIB;
/// And to this many lines, since every line is a row of layout whatever its length.
const SHELL_OUTPUT_KEEP_LINES: usize = 10_000;
/// A question takes no answer this soon after opening: a key already on its way to the composer,
/// Enter above all, would otherwise land on the first choice, which allows.
const QUESTION_GRACE: Duration = Duration::from_millis(350);

/// Whether a key is held back because the question it would reach, when `asking` says one is on
/// screen, opened less than [`QUESTION_GRACE`] ago. The safe keys pass. So does text typed into a
/// form's field: it is not an answer, it shows in the field and can be taken back, where a key
/// swallowed here is gone, and the Enter that would send it is the one that waits. A panel's
/// letters are its picks, so on a panel every key but the safe ones waits.
fn held_by_question_grace(
    asking: bool,
    opened: Option<Instant>,
    focus: Focus,
    action: Action,
) -> bool {
    asking
        && opened.is_some_and(|opened| opened.elapsed() < QUESTION_GRACE)
        && !matches!(action, Action::Escape | Action::Interrupt | Action::Redraw)
        && !matches!((focus, action), (Focus::Question, Action::Edit))
}
/// A listing shown in a panel is cut to its last bytes past this; a task's outcome is the one
/// that can run to its 64 MiB bound.
const PANEL_TEXT_KEEP_BYTES: usize = crate::text::MIB;

/// Everything the screen is launched with. Built by the host loop from the session it has just
/// assembled; the screen owns nothing here that the host does not also hold a handle to.
pub(crate) struct ScreenLaunch {
    pub(crate) ui_receiver: tokio::sync::mpsc::UnboundedReceiver<UiEvent>,
    /// The screen's own sending half, for the pages of the log it reads for itself.
    pub(crate) ui: super::frontend::Ui,
    /// The log's last row as the screen opens, which the first page is read down from; `None`
    /// for a session with no rows yet.
    pub(crate) log_end: Option<i64>,
    pub(crate) host: tokio::sync::mpsc::UnboundedSender<HostEvent>,
    /// `tracing` lines, routed here by the relay so none lands on the terminal.
    pub(crate) logs: crate::relay::ScreenLogs,
    pub(crate) transcript: Transcript,
    pub(crate) shared_permission: SharedPermission,
    pub(crate) sandbox_state: crate::sandbox::SandboxState,
    pub(crate) cwd: crate::workspace::SharedCwd,
    /// Where meka was started, which is what a bare `/cd` returns to.
    pub(crate) launch_cwd: PathBuf,
    /// What the status row shows, in order.
    pub(crate) status_row: Vec<crate::config::StatusFact>,
    /// The context gauge's two cells: tokens used and the window, when the status row shows it.
    pub(crate) context: Option<(Arc<AtomicU64>, Arc<AtomicU64>)>,
    pub(crate) history: Option<crate::store::history::HistoryStore>,
    pub(crate) completer: Completer,
    pub(crate) skill_roots: Vec<PathBuf>,
    pub(crate) current_profile: Arc<std::sync::RwLock<String>>,
    pub(crate) configured_profiles: Vec<crate::config::ProfileSummary>,
    pub(crate) schedule: crate::config::ResolvedScheduleConfig,
    pub(crate) store: crate::store::Store,
    pub(crate) agent: Arc<crate::agent::Agent>,
    /// The sub-agents running in this process, for the status row.
    pub(crate) running_subagents: crate::session::RunningSubagents,
    pub(crate) cancel: crate::host::CancelCell,
    pub(crate) send_class: InboxClass,
    pub(crate) mouse: bool,
    /// Whether a turn is already running when the screen opens: a prompt given on the command
    /// line is the first turn's input.
    pub(crate) busy: bool,
    /// Open on the session picker: `meka -r` with no session named.
    pub(crate) pick_session: bool,
    /// `--project` beside a bare `-r`: the picker opens over that project's sessions.
    pub(crate) pick_project: Option<String>,
}

/// What a panel's pick does.
enum PanelKind {
    /// Something to read; Enter and Esc close it.
    Info,
    /// The levels offered, in the order drawn.
    Permission(Vec<crate::permission::Permission>),
    Approvals,
    /// The profile names offered, in the order drawn.
    Profile(Vec<String>),
    /// The projects offered, in the order drawn, the first row being "No project".
    Project(Vec<String>),
    /// A tool call waiting on the answer.
    Approval(ApprovalRequest),
    /// The command line each choice runs, in the order drawn: a skill to run, a memory, a task
    /// or a job to show.
    Run(Vec<String>),
    /// The configured MCP servers offered, in the order drawn; the pick offers what to do with
    /// one.
    McpServers(Vec<String>),
    /// A server asking for a URL to be opened: the pick opens it, skips it or declines.
    OpenUrl(UrlQuestion),
}

/// A URL a server wants opened, and whom to tell what was done with it.
struct UrlQuestion {
    url: String,
    responder: tokio::sync::oneshot::Sender<ElicitationResponse>,
}

/// What the URL picker offers, in the order drawn, and what each answers the server with.
const URL_CHOICES: [(&str, Option<char>); 3] = [
    ("Open in the browser", Some('y')),
    ("Skip", Some('s')),
    ("Decline", Some('n')),
];

/// A question from the agent's side waiting for the one on screen to be answered: two tool calls
/// dispatched together can both ask, and a form can arrive while an approval is open, and an
/// answer meant for one must not land on another.
enum Question {
    Approval(ApprovalRequest),
    Elicitation {
        prompt: ElicitationPrompt,
        responder: tokio::sync::oneshot::Sender<ElicitationResponse>,
    },
}

impl Question {
    /// Answer the question the safe way without putting it, for a turn the user has stopped.
    fn withdraw(self) {
        let answered = match self {
            Question::Approval(request) => request.responder.send(ApprovalDecision::Deny).is_ok(),
            Question::Elicitation { responder, .. } => {
                responder.send(ElicitationResponse::Decline).is_ok()
            }
        };
        if !answered {
            tracing::trace!("the turn that asked has already moved on");
        }
    }

    /// Whether the turn that asked has moved on, so the question is not worth putting.
    fn is_withdrawn(&self) -> bool {
        match self {
            Question::Approval(request) => {
                request.cancellation.is_cancelled() || request.responder.is_closed()
            }
            Question::Elicitation { responder, .. } => responder.is_closed(),
        }
    }
}

/// A `!command` in flight: when it started, for the status row; the token that ends it; and its
/// process id, for the kill that cannot wait on a task, the one leaving makes.
struct ShellRun {
    id: u64,
    started: Instant,
    stop: tokio_util::sync::CancellationToken,
    pid: Option<u32>,
    /// Set once the command has ended or been killed, so letting go of the run does nothing.
    ended: bool,
}

impl ShellRun {
    /// Kill the command and what it started, now. The token goes first, so the task waiting on
    /// the command reads the death that follows as a stop rather than an exit; the group is
    /// then signaled from here as well, for the leave that may not poll the task again.
    fn kill(&mut self) {
        self.ended = true;
        self.stop.cancel();
        signal_shell_group(self.pid);
    }
}

impl Drop for ShellRun {
    /// A run let go of while the command still runs is killed: the screen is going, by a key,
    /// by the host or by a panic, and nothing shows the command once it is gone.
    fn drop(&mut self) {
        if !self.ended {
            self.kill();
        }
    }
}

/// What is drawn in place of the composer and the status row.
enum Overlay {
    Panel { panel: Panel, kind: PanelKind },
    Elicitation(Elicitation),
}

/// One field of an elicitation form.
struct Field {
    name: String,
    hint: String,
    kind: String,
}

/// A server's form, asked one field at a time in the composer's place.
struct Elicitation {
    responder: tokio::sync::oneshot::Sender<ElicitationResponse>,
    /// What the server asks, as it said it.
    body: Vec<Line<'static>>,
    fields: Vec<Field>,
    /// The field being asked.
    index: usize,
    filled: serde_json::Map<String, serde_json::Value>,
    answer: String,
}

impl Elicitation {
    fn question(&self) -> String {
        self.fields
            .get(self.index)
            .map(|field| {
                format!(
                    "{} ({}): ",
                    one_line(&field.name, 64),
                    one_line(&field.hint, 160)
                )
            })
            .unwrap_or_default()
    }
}

/// A server-supplied string on one row of meka's own, cut at `columns` the way every cut on the
/// screen is made: at the end, with the marker.
fn one_line(text: &str, columns: usize) -> String {
    draw::cut_text(&crate::text::sanitize_to_line(text, usize::MAX), columns)
}

/// A picker row's description: the first line of what something says about itself, or nothing
/// when it says nothing.
fn description_of(text: &str) -> Option<String> {
    let line = one_line(text, 120);
    (!line.trim().is_empty()).then_some(line)
}

/// A form field's answer as the value its type asks for; anything that does not parse stays a
/// string, which is what the server sees and can refuse.
fn field_value(kind: &str, value: String) -> serde_json::Value {
    match kind {
        "boolean" => match value.to_ascii_lowercase().as_str() {
            "true" | "yes" | "y" => serde_json::Value::Bool(true),
            "false" | "no" | "n" => serde_json::Value::Bool(false),
            _ => serde_json::Value::String(value),
        },
        "integer" | "number" => value
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::String(value)),
        _ => serde_json::Value::String(value),
    }
}

/// The four answers an approval prompt offers, in the order drawn.
const APPROVAL_CHOICES: [(&str, Option<char>, ApprovalDecision); 4] = [
    ("Allow", Some('y'), ApprovalDecision::Allow),
    (
        "Allow for the session",
        Some('a'),
        ApprovalDecision::AllowAlways,
    ),
    ("Deny", Some('n'), ApprovalDecision::Deny),
    ("Deny for the session", None, ApprovalDecision::DenyAlways),
];

/// The screen's state between frames.
struct Screen {
    launch: ScreenLaunch,
    viewport: Viewport,
    composer: Composer,
    overlay: Option<Overlay>,
    skill_watch: crate::skills::SkillNameWatch,
    /// What was logged lately, at every level the screen's layer keeps, for `/log`.
    log_ring: std::collections::VecDeque<crate::relay::LogLine>,
    /// Whether the host is answering something the screen sent it, and since when.
    busy: Option<Instant>,
    /// What the agent is doing, as the stream last said; the working row's word.
    activity: view::Activity,
    /// A compaction is running: the word while the stream says nothing more specific, since the
    /// checkpoint turn's own thinking and calls say what they are, and the summarizer says
    /// nothing.
    compacting: bool,
    /// The seconds and the spinner frame the working row last showed, so a change redraws it.
    shown_working: (u64, usize),
    /// A wake arrived while the host was busy; handed over when it is not.
    wake_pending: bool,
    /// The questions waiting behind the one on screen.
    questions: std::collections::VecDeque<Question>,
    /// The `!command` running, if one is.
    shell: Option<ShellRun>,
    next_shell_run: u64,
    /// Followups typed during a turn, each with the id of the prompt item it is shown as.
    followups: std::collections::VecDeque<(u64, String)>,
    hint: Option<(String, Instant)>,
    /// When the open question was asked, for the grace it takes no answer in.
    question_opened: Option<Instant>,
    /// Bumped when the window is thrown away (a reload, a jump back to the bottom), so a page
    /// read for the old window is dropped when it lands rather than put into the new one.
    history_generation: u64,
    /// The drag in progress or just finished over the transcript.
    selection: Option<Selection>,
    /// Whether the pointer is on the stuck prompt's bar, which lights it.
    sticky_hovered: bool,
    /// The last frame as drawn, which is what a selection copies: the text as shown, whatever
    /// item it came from. Kept from every frame, since a press, a drag and a release can all
    /// arrive between two frames.
    drawn: ratatui::buffer::Buffer,
    /// The transcript's area in the last frame, for mapping a click.
    transcript_area: Rect,
    /// The back-to-bottom pill in the last frame, when one was drawn.
    pill: Option<Rect>,
    /// The overlay's area in the last frame, when one was drawn.
    overlay_area: Option<Rect>,
    /// A page of the log is on its way; one read at a time, so two scrolls do not ask twice.
    history_loading: bool,
    /// The background tasks and sub-agents running for this session, as last read.
    work: view::RunningWork,
    work_polled: Option<Instant>,
    /// Where the last frame drew multi-column emoji, so the next frame rewrites the cells after
    /// them; see [`view::wide_symbols`].
    wide_cells: Vec<(u16, u16, u16)>,
    dirty: bool,
    leaving: bool,
}

impl Screen {
    fn new(mut launch: ScreenLaunch) -> Self {
        let skill_watch = crate::skills::SkillNameWatch::new(launch.skill_roots.clone());
        let history = launch.history.take();
        let busy = launch.busy.then(Instant::now);
        let mut screen = Self {
            launch,
            viewport: Viewport::new(),
            composer: Composer::new(history),
            overlay: None,
            skill_watch,
            log_ring: std::collections::VecDeque::with_capacity(LOG_RING_LINES),
            busy,
            activity: view::Activity::Working,
            compacting: false,
            shown_working: (0, 0),
            wake_pending: false,
            questions: std::collections::VecDeque::new(),
            shell: None,
            next_shell_run: 0,
            followups: std::collections::VecDeque::new(),
            hint: None,
            question_opened: None,
            history_generation: 0,
            selection: None,
            sticky_hovered: false,
            drawn: ratatui::buffer::Buffer::default(),
            transcript_area: Rect::default(),
            pill: None,
            overlay_area: None,
            history_loading: false,
            work: view::RunningWork::default(),
            work_polled: None,
            wide_cells: Vec::new(),
            dirty: true,
            leaving: false,
        };
        let log_end = screen.launch.log_end;
        screen.launch.transcript.set_log_end(log_end);
        screen.refresh_skill_names();
        // The last page of the log is what the screen opens on.
        screen.request_page_above();
        screen
    }

    /// Read the page of the log above the window, or its last page into an empty window.
    fn request_page_above(&mut self) {
        if self.history_loading || !self.launch.transcript.has_more_above() {
            return;
        }
        let Some(session_id) = self.launch.agent.session_id() else {
            return;
        };
        let before = match self.launch.transcript.window_floor() {
            Some(floor) => Some(floor),
            None => self.launch.transcript.log_end().map(|end| end + 1),
        };
        let read_rewound = !self.launch.transcript.knows_rewound();
        self.history_loading = true;
        let generation = self.history_generation;
        let store = self.launch.store.clone();
        let ui = self.launch.ui.clone();
        tokio::spawn(async move {
            let page = match store
                .log_page_before(session_id, before, super::model::HISTORY_PAGE_ROWS)
                .await
            {
                Ok(page) => page,
                Err(error) => {
                    tracing::warn!("failed to read the log: {error}");
                    crate::store::LogPage::default()
                }
            };
            let tasks = match store
                .background_store()
                .list_background_tasks(session_id)
                .await
            {
                Ok(tasks) => tasks,
                Err(error) => {
                    tracing::warn!("failed to read the background tasks for the log: {error}");
                    Vec::new()
                }
            };
            // Read here, off the screen's path, since a log a rewind is in is read whole for it.
            let rewound = if read_rewound {
                Some(store.rewound(session_id).await.unwrap_or_else(|error| {
                    tracing::warn!("failed to read what the session's rewinds took: {error}");
                    Vec::new()
                }))
            } else {
                None
            };
            ui.send(UiEvent::HistoryPage {
                above: true,
                page,
                tasks,
                rewound,
                generation,
            });
        });
    }

    /// Read the page of the log below the window, toward the row the screen opened at.
    fn request_page_below(&mut self) {
        if self.history_loading || self.launch.transcript.is_attached() {
            return;
        }
        let (Some(session_id), Some(ceiling), Some(through)) = (
            self.launch.agent.session_id(),
            self.launch.transcript.window_ceiling(),
            self.launch.transcript.log_end(),
        ) else {
            return;
        };
        self.history_loading = true;
        let generation = self.history_generation;
        let store = self.launch.store.clone();
        let ui = self.launch.ui.clone();
        tokio::spawn(async move {
            let page = match store
                .log_page_after(
                    session_id,
                    ceiling,
                    through,
                    super::model::HISTORY_PAGE_ROWS,
                )
                .await
            {
                Ok(page) => page,
                Err(error) => {
                    tracing::warn!("failed to read the log: {error}");
                    crate::store::LogPage::default()
                }
            };
            let tasks = match store
                .background_store()
                .list_background_tasks(session_id)
                .await
            {
                Ok(tasks) => tasks,
                Err(error) => {
                    tracing::warn!("failed to read the background tasks for the log: {error}");
                    Vec::new()
                }
            };
            ui.send(UiEvent::HistoryPage {
                above: false,
                page,
                tasks,
                rewound: None,
                generation,
            });
        });
    }

    /// The log changed under the screen: read its end again and show the last page.
    fn reload(&mut self) {
        let Some(session_id) = self.launch.agent.session_id() else {
            self.launch.transcript.reset(None);
            self.viewport.scroll_to_bottom();
            return;
        };
        self.history_generation += 1;
        self.history_loading = true;
        let generation = self.history_generation;
        let store = self.launch.store.clone();
        let ui = self.launch.ui.clone();
        tokio::spawn(async move {
            let (log_end, page) = match store.last_log_row(session_id).await {
                Ok(end) => {
                    let page = store
                        .log_page_before(
                            session_id,
                            end.map(|end| end + 1),
                            super::model::HISTORY_PAGE_ROWS,
                        )
                        .await
                        .unwrap_or_else(|error| {
                            tracing::warn!("failed to read the log: {error}");
                            crate::store::LogPage::default()
                        });
                    (end, page)
                }
                Err(error) => {
                    tracing::warn!("failed to read the log's end: {error}");
                    (None, crate::store::LogPage::default())
                }
            };
            let turns = store
                .turn_store()
                .load_turns(session_id)
                .await
                .unwrap_or_else(|error| {
                    tracing::warn!("failed to read the session's turns: {error}");
                    Vec::new()
                });
            let rewound = store.rewound(session_id).await.unwrap_or_else(|error| {
                tracing::warn!("failed to read what the session's rewinds took: {error}");
                Vec::new()
            });
            ui.send(UiEvent::Reloaded {
                log_end,
                page,
                turns,
                rewound,
                generation,
            });
        });
    }

    /// Ask for the page the view is about to need, if any.
    fn pump_history(&mut self) {
        if self.viewport.is_near_top() {
            self.request_page_above();
        }
        if !self.launch.transcript.is_attached() && self.viewport.is_near_bottom() {
            self.request_page_below();
        }
    }

    /// Return to the bottom: the tail under the last page of the log, following new output.
    fn back_to_bottom(&mut self) {
        if !self.launch.transcript.is_attached() {
            self.launch.transcript.drop_window();
            // A page still being read was for the window just dropped.
            self.history_generation += 1;
            self.history_loading = false;
            self.request_page_above();
        }
        self.viewport.scroll_to_bottom();
        self.dirty = true;
    }

    fn refresh_skill_names(&mut self) {
        if let Some(discovered) = self.skill_watch.refresh() {
            *crate::sync::write(&self.launch.completer.skill_names) = discovered;
        }
    }

    /// The list follows the line: open as `/` is typed, narrowed with every key. Skill names are
    /// re-read only when the line asks for them, since the check stats every skill root.
    fn suggest(&mut self) {
        if self.composer.text().starts_with("/skill") {
            self.refresh_skill_names();
        }
        self.composer.suggest(&self.launch.completer);
    }

    /// What a command has to say back that is not a fact of the session: a refusal, a failure,
    /// an empty list. A panel until Esc, and nothing on the transcript.
    fn reply(&mut self, message: impl Into<String>) {
        self.open_panel(Panel::with_text(&message.into()), PanelKind::Info);
    }

    /// A picker whose every choice is a command line to run: the label and what it is, and the
    /// line Enter runs for it.
    fn open_run_picker(&mut self, rows: Vec<(String, Option<String>, String)>) {
        let lines = rows.iter().map(|(_, _, line)| line.clone()).collect();
        let choices = rows
            .into_iter()
            .map(|(label, description, _)| Choice {
                label,
                key: None,
                description,
            })
            .collect();
        self.open_panel(
            Panel::with_choices(Vec::new(), choices, 0),
            PanelKind::Run(lines),
        );
    }

    /// `/resume` with nothing after it: the sessions this screen can switch to, newest first,
    /// filtered as text is typed; the pick runs `/resume <id>`. A table: the title, then the
    /// directory, the profile and the age in columns of their own at the right edge, sized by
    /// [`SessionColumns::fit`].
    async fn open_session_picker(&mut self, project: Option<String>) {
        // Before the first frame the transcript has no width yet, as at `meka -r`; the terminal
        // itself says how wide the picker can be, and so how much of a title a row can hold.
        let width = match self.transcript_area.width {
            0 => crossterm::terminal::size().map_or(80, |(columns, _)| columns),
            width => width,
        } as usize;
        let sessions = match self
            .launch
            .store
            .list_sessions_titled_within(
                SESSION_PICKER_ROWS,
                None,
                &crate::store::SessionFilter {
                    project,
                    ..Default::default()
                },
                width,
            )
            .await
        {
            Ok((sessions, _)) => sessions,
            Err(error) => {
                self.reply(format!("Failed to list sessions: {error}"));
                return;
            }
        };
        let current = self.launch.agent.session_id();
        let now = chrono::Utc::now();
        let offered: Vec<(&crate::store::SessionSummary, String, String, String)> = sessions
            .iter()
            .filter(|session| Some(session.id) != current)
            .map(|session| {
                let age = chrono::DateTime::parse_from_rfc3339(&session.updated_at)
                    .map(|at| crate::text::format_age(at.with_timezone(&chrono::Utc), now))
                    .unwrap_or_else(|_| session.updated_at.clone());
                let directory = session
                    .cwd
                    .as_deref()
                    .map(super::input::shorten_path_with_tilde)
                    .unwrap_or_default();
                let title = crate::text::sanitize_to_line(&session.title, usize::MAX);
                let title = if title.is_empty() {
                    "(untitled)".to_string()
                } else {
                    title
                };
                (session, title, directory, age)
            })
            .collect();
        let longest_directory = offered
            .iter()
            .map(|(_, _, directory, _)| crate::text::display_width(directory))
            .max()
            .unwrap_or(0);
        let longest_profile = offered
            .iter()
            .map(|(session, ..)| crate::text::display_width(&session.profile))
            .max()
            .unwrap_or(0);
        let longest_age = offered
            .iter()
            .map(|(_, _, _, age)| crate::text::display_width(age))
            .max()
            .unwrap_or(0);
        let columns = SessionColumns::fit(width, longest_directory, longest_profile, longest_age);
        let mut lines = Vec::new();
        let mut choices = Vec::new();
        for (session, title, directory, age) in offered {
            lines.push(format!("/resume {}", session.id));
            choices.push(Choice {
                label: draw::cut_text(&title, columns.title),
                key: None,
                description: Some(columns.facts(&directory, &session.profile, &age)),
            });
        }
        if choices.is_empty() {
            self.reply("No sessions to resume.");
            return;
        }
        self.open_panel(
            Panel::with_choices(Vec::new(), choices, 0)
                .with_filter()
                .with_facts_at_the_edge(),
            PanelKind::Run(lines),
        );
    }

    /// What can be done with one MCP server, under its name: the three verbs `/mcp` takes.
    fn open_mcp_actions(&mut self, server: String) {
        let actions = [
            ("Reconnect", "Connect to the server again", "reconnect"),
            ("Login", "Run the OAuth flow", "login"),
            ("Logout", "Clear the stored credentials", "logout"),
        ];
        let lines = actions
            .iter()
            .map(|(_, _, verb)| format!("/mcp {verb} {server}"))
            .collect();
        let choices = actions
            .iter()
            .map(|(label, description, _)| Choice {
                label: (*label).to_string(),
                key: None,
                description: Some((*description).to_string()),
            })
            .collect();
        self.open_panel(
            Panel::with_choices(Vec::new(), choices, 0)
                .with_heading(Line::from(Span::styled(server, gray()))),
            PanelKind::Run(lines),
        );
    }

    fn show_hint(&mut self, hint: impl Into<String>) {
        self.hint = Some((hint.into(), Instant::now()));
        self.dirty = true;
    }

    fn push(&mut self, item: Item) {
        self.launch.transcript.push(item);
        self.dirty = true;
    }

    fn open_panel(&mut self, panel: Panel, kind: PanelKind) {
        self.overlay = Some(Overlay::Panel { panel, kind });
        self.dirty = true;
    }

    fn close_overlay(&mut self) {
        self.overlay = None;
        self.overlay_area = None;
        self.dirty = true;
    }

    fn send_host(&mut self, event: HostEvent) {
        if self.launch.host.send(event).is_err() {
            // The host loop is gone, so there is nothing left to answer anything.
            self.push(Item::Error("the agent stopped; leaving".to_string()));
            self.leaving = true;
        }
    }

    /// Hand the host something to answer, and wait for its `Done` before sending the next.
    fn ask_host(&mut self, event: HostEvent) {
        if self.busy.is_none() {
            self.busy = Some(Instant::now());
        }
        self.send_host(event);
    }

    fn is_busy(&self) -> bool {
        self.busy.is_some()
    }

    fn context_gauge(&self) -> Option<String> {
        let (tokens, window) = self.launch.context.as_ref()?;
        super::view::context_gauge_text(
            tokens.load(std::sync::atomic::Ordering::Relaxed),
            window.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// The profile the session runs on, by name, for the status row.
    fn profile_fact(&self) -> Option<String> {
        let name = crate::sync::read(&self.launch.current_profile).clone();
        (!name.is_empty()).then_some(name)
    }

    /// The model the session's profile runs, for the status row; `None` for a profile whose
    /// model the config leaves to the backend.
    fn model_fact(&self) -> Option<String> {
        let name = crate::sync::read(&self.launch.current_profile).clone();
        self.launch
            .configured_profiles
            .iter()
            .find(|profile| profile.name == name)
            .and_then(|profile| profile.model.clone())
    }

    fn focus(&self) -> Focus {
        match &self.overlay {
            Some(Overlay::Panel { panel, .. }) if panel.is_typing() => Focus::Filtering,
            Some(Overlay::Panel { .. }) => Focus::Panel,
            Some(Overlay::Elicitation(_)) => Focus::Question,
            None => self.composer.focus(),
        }
    }

    fn composer_state(&self) -> ComposerState {
        ComposerState {
            cursor: self.viewport.cursor().is_some(),
            busy: self.working_since().is_some(),
            ..self.composer.state()
        }
    }

    /// Since when something the screen started has been running: the host's answer, or a
    /// `!command`. What the status row counts and what Esc stops.
    fn working_since(&self) -> Option<Instant> {
        self.busy
            .or_else(|| self.shell.as_ref().map(|run| run.started))
    }

    /// Whether what is open is a question from the agent's side, which another waits behind.
    fn overlay_is_question(&self) -> bool {
        matches!(
            &self.overlay,
            Some(Overlay::Panel {
                kind: PanelKind::Approval(_) | PanelKind::OpenUrl(_),
                ..
            }) | Some(Overlay::Elicitation(_))
        )
    }

    /// Put a question to the user, or hold it until the one on screen is answered.
    fn ask(&mut self, question: Question) {
        if self.overlay_is_question() {
            self.questions.push_back(question);
        } else {
            self.open_question(question);
        }
    }

    fn open_question(&mut self, question: Question) {
        match question {
            Question::Approval(request) => self.open_approval(request),
            Question::Elicitation { prompt, responder } => {
                self.open_elicitation(prompt, responder);
            }
        }
        self.question_opened = Some(Instant::now());
    }

    /// Open the next question waiting, once nothing is being asked; one whose turn has moved on
    /// is dropped unasked.
    fn open_pending_question(&mut self) {
        while !self.overlay_is_question() {
            let Some(question) = self.questions.pop_front() else {
                return;
            };
            if question.is_withdrawn() {
                continue;
            }
            self.open_question(question);
            self.dirty = true;
        }
    }

    fn handle_ui(&mut self, event: UiEvent) {
        self.dirty = true;
        match event {
            UiEvent::Agent(event) => {
                match &event {
                    crate::frontend::FrontendEvent::CompactionStarted { .. } => {
                        self.compacting = true;
                    }
                    crate::frontend::FrontendEvent::Compacted { .. }
                    | crate::frontend::FrontendEvent::CompactionFailed { .. } => {
                        self.compacting = false;
                    }
                    _ => {}
                }
                self.activity.note(&event);
                self.launch.transcript.apply(event);
            }
            UiEvent::Approval(request) => self.ask(Question::Approval(request)),
            UiEvent::Elicitation { prompt, responder } => {
                self.ask(Question::Elicitation { prompt, responder });
            }
            UiEvent::ShellFinished {
                run,
                output,
                stopped,
            } => self.finish_shell(run, output, stopped),
            UiEvent::Item(item) => self.push(item),
            UiEvent::McpServers(servers) => {
                let names = servers.iter().map(|server| server.name.clone()).collect();
                let choices = servers
                    .into_iter()
                    .map(|server| Choice {
                        label: server.name,
                        key: None,
                        description: Some(server.state),
                    })
                    .collect();
                self.open_panel(
                    Panel::with_choices(Vec::new(), choices, 0),
                    PanelKind::McpServers(names),
                );
            }
            UiEvent::Panel(body) => self.open_panel(Panel::with_text(&body), PanelKind::Info),
            UiEvent::Done => {
                self.launch.transcript.flush_turn_end();
                self.busy = None;
                self.activity = view::Activity::Working;
                self.compacting = false;
                if self.wake_pending {
                    self.wake_pending = false;
                    self.ask_host(HostEvent::Wake);
                } else if let Some((id, text)) = self.followups.pop_front() {
                    // A reload between the typing and the sending (a `/rewind` or `/fork` run
                    // between them) keeps nothing of the tail but the host's own rows, so the
                    // prompt is shown again rather than sent under nothing.
                    if !self.launch.transcript.mark_sent(id) {
                        self.push(Item::Prompt {
                            text: text.clone(),
                            class: None,
                            inbox_id: None,
                            delivered: true,
                        });
                    }
                    self.ask_host(HostEvent::UserInput {
                        parts: vec![crate::conversation::ContentBlock::Text { text }],
                        images: Vec::new(),
                    });
                }
            }
            UiEvent::Wake => {
                if self.is_busy() {
                    self.wake_pending = true;
                } else {
                    self.ask_host(HostEvent::Wake);
                }
            }
            UiEvent::Leave => self.leaving = true,
            UiEvent::Reload => self.reload(),
            UiEvent::Reloaded {
                log_end,
                page,
                turns,
                rewound,
                generation,
            } => {
                if generation != self.history_generation {
                    return;
                }
                self.history_loading = false;
                self.launch.transcript.reset(log_end);
                self.launch.transcript.set_turn_records(turns);
                self.launch.transcript.set_rewound(rewound);
                self.launch.transcript.load_page_above(page);
                self.viewport.scroll_to_bottom();
            }
            UiEvent::HistoryPage {
                above,
                page,
                tasks,
                rewound,
                generation,
            } => {
                if generation != self.history_generation {
                    return;
                }
                self.history_loading = false;
                self.launch.transcript.remember_background_tasks(tasks);
                if let Some(rewound) = rewound {
                    self.launch.transcript.set_rewound(rewound);
                }
                if above {
                    self.launch.transcript.load_page_above(page);
                } else {
                    self.launch.transcript.load_page_below(page);
                }
            }
        }
    }

    /// The approval prompt as a panel: the call's row as its heading, its arguments as a tree
    /// under it, the four answers below.
    fn open_approval(&mut self, request: ApprovalRequest) {
        // The call's own row heads the panel and its arguments stand under it as a tree, the
        // way the transcript draws the call once it is expanded: with the arguments shown, the
        // row carries no summary, which would say the same thing twice.
        let width = self.transcript_area.width.max(20) as usize;
        let title = draw::tool_row(&request.tool_name, None, None, None, width);
        let body = draw::approval_rows(&request.tool_name, &request.input, width);
        let choices = APPROVAL_CHOICES
            .iter()
            .map(|(label, key, _)| Choice {
                label: (*label).to_string(),
                key: *key,
                description: None,
            })
            .collect();
        self.open_panel(
            Panel::with_choices(body, choices, 0).with_heading(title),
            PanelKind::Approval(request),
        );
    }

    fn open_elicitation(
        &mut self,
        prompt: ElicitationPrompt,
        responder: tokio::sync::oneshot::Sender<ElicitationResponse>,
    ) {
        // Server-controlled strings are flattened to one line before they wrap: a newline in the
        // message could paint rows that look like meka's own, enough to forge an approval block.
        let width = self.transcript_area.width.max(20) as usize;
        let mut body = draw::glyph_text_rows(
            draw::ACT_GLYPH,
            gray(),
            &format!(
                "{}: {}",
                one_line(&prompt.server_name, 64),
                crate::text::sanitize_to_line(&prompt.message, usize::MAX)
            ),
            Style::default(),
            width,
        );
        let fields = match prompt.kind {
            // A URL is a pick, not a typed answer: the three things that can be done with it are
            // the choices, the way the approval prompt's are. The URL is wrapped whole, since what
            // is cut from a URL is what the browser is about to open.
            ElicitationKind::Url { url } => {
                body.extend(draw::edge_rows(draw::plain_lines(
                    &crate::text::sanitize_to_line(&url, usize::MAX),
                    width.saturating_sub(TEXT_EDGE),
                    gray(),
                )));
                let choices = URL_CHOICES
                    .iter()
                    .map(|(label, key)| Choice {
                        label: (*label).to_string(),
                        key: *key,
                        description: None,
                    })
                    .collect();
                self.open_panel(
                    Panel::with_choices(body, choices, 0),
                    PanelKind::OpenUrl(UrlQuestion { url, responder }),
                );
                return;
            }
            ElicitationKind::Form { schema } => {
                let fields: Vec<Field> = schema
                    .get("properties")
                    .and_then(|properties| properties.as_object())
                    .map(|properties| {
                        properties
                            .iter()
                            .map(|(name, field)| {
                                let description = field
                                    .get("description")
                                    .and_then(|value| value.as_str())
                                    .unwrap_or("");
                                let kind = field
                                    .get("type")
                                    .and_then(|value| value.as_str())
                                    .unwrap_or("string")
                                    .to_string();
                                Field {
                                    name: name.clone(),
                                    hint: if description.is_empty() {
                                        kind.clone()
                                    } else {
                                        description.to_string()
                                    },
                                    kind,
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                // A form with nothing to fill in asks the user nothing, so there is no answer to
                // send back and accepting it would consent on the user's behalf.
                if fields.is_empty() {
                    if responder.send(ElicitationResponse::Decline).is_err() {
                        tracing::trace!("the turn that asked has already moved on");
                    }
                    return;
                }
                fields
            }
        };
        self.overlay = Some(Overlay::Elicitation(Elicitation {
            responder,
            body,
            fields,
            index: 0,
            filled: serde_json::Map::new(),
            answer: String::new(),
        }));
    }

    /// Enter on the elicitation prompt.
    fn answer_elicitation(&mut self) {
        let Some(Overlay::Elicitation(mut elicitation)) = self.overlay.take() else {
            return;
        };
        self.overlay_area = None;
        let answer = std::mem::take(&mut elicitation.answer);
        let value = answer.trim().to_string();
        if !value.is_empty()
            && let Some(field) = elicitation.fields.get(elicitation.index)
        {
            elicitation
                .filled
                .insert(field.name.clone(), field_value(&field.kind, value));
        }
        elicitation.index += 1;
        if elicitation.index < elicitation.fields.len() {
            self.overlay = Some(Overlay::Elicitation(elicitation));
            return;
        }
        let content = serde_json::Value::Object(std::mem::take(&mut elicitation.filled));
        if elicitation
            .responder
            .send(ElicitationResponse::Accept {
                content: Some(content),
            })
            .is_err()
        {
            tracing::trace!("the turn that asked has already moved on");
        }
    }

    /// What was picked for a URL a server wants opened.
    fn answer_url(question: UrlQuestion, index: usize) {
        let response = match index {
            0 => {
                if let Err(error) = open::that(&question.url) {
                    // The URL is on screen; a launch failure on a headless host is expected
                    // noise.
                    tracing::debug!("failed to open the browser: {error}");
                }
                ElicitationResponse::Accept { content: None }
            }
            1 => ElicitationResponse::Cancel,
            _ => ElicitationResponse::Decline,
        };
        if question.responder.send(response).is_err() {
            tracing::trace!("the turn that asked has already moved on");
        }
    }

    /// Enter on a panel: apply the pick, or close what is only there to be read.
    async fn pick_panel(&mut self) {
        let Some(Overlay::Panel { panel, .. }) = &self.overlay else {
            return;
        };
        match panel.picked() {
            Some(index) => self.apply_pick(index).await,
            None => self.close_overlay(),
        }
    }

    /// What picking the `index`th choice of the open panel does.
    async fn apply_pick(&mut self, index: usize) {
        let Some(Overlay::Panel { kind, .. }) = self.overlay.take() else {
            return;
        };
        self.overlay_area = None;
        self.dirty = true;
        match kind {
            PanelKind::Info => {}
            PanelKind::Permission(levels) => {
                if let Some(level) = levels.get(index).copied() {
                    self.set_permission(level);
                }
            }
            PanelKind::Approvals => self.set_approvals(index == 0),
            PanelKind::Profile(names) => {
                if let Some(name) = names.get(index) {
                    self.run_command_line(format!("/profile {name}")).await;
                }
            }
            PanelKind::Project(names) => match (index, names.get(index)) {
                (0, _) => self.run_command_line("/project close".to_string()).await,
                (_, Some(name)) => {
                    self.run_command_line(format!("/project open {name}")).await;
                }
                (_, None) => {}
            },
            PanelKind::Approval(request) => {
                if let Some((_, _, decision)) = APPROVAL_CHOICES.get(index) {
                    self.decide(request, *decision);
                }
            }
            PanelKind::Run(lines) => {
                if let Some(line) = lines.get(index) {
                    self.run_command_line(line.clone()).await;
                }
            }
            PanelKind::McpServers(names) => {
                if let Some(name) = names.get(index) {
                    self.open_mcp_actions(name.clone());
                }
            }
            PanelKind::OpenUrl(question) => Self::answer_url(question, index),
        }
    }

    fn set_permission(&mut self, level: crate::permission::Permission) {
        match self.launch.shared_permission.try_set(level) {
            Ok(()) => {
                self.push(Item::note(format!("Permission level set to: {level}")));
                self.send_host(HostEvent::PermissionChanged(level));
            }
            Err(error) => self.reply(error.to_string()),
        }
    }

    fn set_approvals(&mut self, approvals: bool) {
        self.launch.shared_permission.set_approvals(approvals);
        self.push(Item::note(format!(
            "Approvals set to: {}",
            if approvals { "on" } else { "off" }
        )));
        self.send_host(HostEvent::ApprovalsChanged(approvals));
    }

    fn decide(&mut self, request: ApprovalRequest, decision: ApprovalDecision) {
        // The frontend drops its receiver when the turn is stopped mid-prompt, so an answer typed
        // after that is to a question nobody is asking any more. Said rather than swallowed,
        // because the user just typed something and is owed an account of where it went.
        if request.responder.is_closed() {
            self.show_hint("discarded: the turn had already been stopped");
            return;
        }
        let word = match decision {
            ApprovalDecision::Allow => "allowed",
            ApprovalDecision::AllowAlways => "allowed for the session",
            ApprovalDecision::Deny => "denied",
            ApprovalDecision::DenyAlways => "denied for the session",
        };
        self.push(Item::note(format!(
            "{} {word}",
            one_line(&request.tool_name, 64)
        )));
        if request.responder.send(decision).is_err() {
            tracing::warn!("failed to send the approval answer: the agent is gone");
        }
    }

    /// Esc while a filter is being typed: the keys are the panel's again, and what was typed
    /// stays and keeps filtering.
    fn stop_filtering(&mut self) {
        if let Some(Overlay::Panel { panel, .. }) = self.overlay.as_mut() {
            panel.stop_filtering();
            self.dirty = true;
        }
    }

    /// Esc on an overlay: close it, which is the safe answer where one is owed.
    fn dismiss_overlay(&mut self) {
        let Some(overlay) = self.overlay.take() else {
            return;
        };
        self.overlay_area = None;
        self.dirty = true;
        match overlay {
            Overlay::Panel {
                kind: PanelKind::Approval(request),
                ..
            } => self.decide(request, ApprovalDecision::Deny),
            Overlay::Panel {
                kind: PanelKind::OpenUrl(question),
                ..
            } => Self::answer_url(question, 2),
            Overlay::Panel { .. } => {}
            Overlay::Elicitation(elicitation) => {
                if elicitation
                    .responder
                    .send(ElicitationResponse::Decline)
                    .is_err()
                {
                    tracing::trace!("the turn that asked has already moved on");
                }
            }
        }
    }

    /// A key typed into the elicitation's answer.
    fn edit_question(&mut self, key: KeyEvent) {
        let Some(Overlay::Elicitation(elicitation)) = self.overlay.as_mut() else {
            return;
        };
        match key.code {
            KeyCode::Backspace => {
                elicitation.answer.pop();
            }
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::CONTROL) =>
            {
                elicitation.answer.push(character);
            }
            _ => {}
        }
    }

    /// A key typed at a panel: text for the filter while one is being typed, `/` to start typing
    /// one, else the letter or digit that picks a choice.
    async fn edit_panel(&mut self, key: KeyEvent) {
        if let Some(Overlay::Panel { panel, .. }) = self.overlay.as_mut() {
            if panel.is_typing() {
                match key.code {
                    KeyCode::Backspace => panel.filter_erase(),
                    KeyCode::Char(character)
                        if !key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL) =>
                    {
                        panel.filter_type(character);
                    }
                    _ => {}
                }
                return;
            }
            if key.code == KeyCode::Char('/') && panel.can_filter() {
                panel.start_filtering();
                return;
            }
        }
        let KeyCode::Char(character) = key.code else {
            return;
        };
        if key
            .modifiers
            .contains(crossterm::event::KeyModifiers::CONTROL)
        {
            return;
        }
        let Some(Overlay::Panel { panel, .. }) = &self.overlay else {
            return;
        };
        if let Some(index) = panel.choice_for_key(character) {
            self.apply_pick(index).await;
        }
    }

    /// Every question open or waiting is withdrawn with its safe answer: what a stop does for
    /// the questions the turn's own cancellation does not reach, an MCP form or URL among them.
    fn withdraw_questions(&mut self) {
        if self.overlay_is_question()
            && let Some(overlay) = self.overlay.take()
        {
            self.overlay_area = None;
            match overlay {
                Overlay::Panel {
                    kind: PanelKind::Approval(request),
                    ..
                } => Question::Approval(request).withdraw(),
                Overlay::Panel {
                    kind: PanelKind::OpenUrl(question),
                    ..
                } => Self::answer_url(question, 2),
                Overlay::Elicitation(elicitation) => Question::Elicitation {
                    prompt: ElicitationPrompt {
                        server_name: String::new(),
                        kind: ElicitationKind::Form {
                            schema: serde_json::Value::Null,
                        },
                        message: String::new(),
                        deadline: tokio::time::Instant::now(),
                        subagent_id: None,
                    },
                    responder: elicitation.responder,
                }
                .withdraw(),
                Overlay::Panel { .. } => {}
            }
            self.show_hint(QUESTION_WITHDRAWN_HINT);
        }
        while let Some(question) = self.questions.pop_front() {
            question.withdraw();
        }
    }

    /// A question whose turn has been stopped is withdrawn.
    fn withdraw_stopped_questions(&mut self) {
        let withdrawn = match &self.overlay {
            Some(Overlay::Panel {
                kind: PanelKind::Approval(request),
                ..
            }) => request.cancellation.is_cancelled() || request.responder.is_closed(),
            Some(Overlay::Panel {
                kind: PanelKind::OpenUrl(question),
                ..
            }) => question.responder.is_closed(),
            Some(Overlay::Elicitation(elicitation)) => elicitation.responder.is_closed(),
            _ => false,
        };
        if withdrawn {
            self.close_overlay();
            self.show_hint(QUESTION_WITHDRAWN_HINT);
        }
    }

    async fn handle_key(&mut self, key: KeyEvent, terminal: &mut DefaultTerminal) {
        if key.kind == crossterm::event::KeyEventKind::Release {
            return;
        }
        self.dirty = true;
        self.selection = None;
        let focus = self.focus();
        let action = action_for(&key, focus, self.composer_state());
        if held_by_question_grace(
            self.overlay_is_question(),
            self.question_opened,
            focus,
            action,
        ) {
            return;
        }
        match action {
            Action::Submit => match focus {
                Focus::Panel | Focus::Filtering => self.pick_panel().await,
                Focus::Question => self.answer_elicitation(),
                Focus::Search => self.composer.accept_search(),
                // Enter on the list runs the pick, as the command palettes people know do; Tab
                // is the key that fills it in to go on typing.
                Focus::Completion => {
                    self.composer.accept_completion();
                    self.submit().await;
                }
                Focus::Composer => self.submit().await,
            },
            Action::Newline => self.composer.newline(),
            Action::CyclePermission => self.cycle_permission(),
            Action::Interrupt => {
                if !self.is_busy() && matches!(self.overlay, Some(Overlay::Panel { .. })) {
                    self.dismiss_overlay();
                } else {
                    self.interrupt().await;
                }
            }
            Action::Leave => self.leave(),
            Action::Escape => match focus {
                Focus::Panel | Focus::Question => self.dismiss_overlay(),
                Focus::Filtering => self.stop_filtering(),
                Focus::Search => self.composer.close_search(),
                Focus::Completion => self.composer.close_completion(),
                Focus::Composer => {
                    if self.viewport.cursor().is_some() {
                        self.viewport.set_cursor(None);
                    } else {
                        self.back_to_bottom();
                    }
                }
            },
            Action::HistoryPrevious => self.composer.history_previous(),
            Action::HistoryNext => self.composer.history_next(),
            Action::HistorySearch => self.composer.history_search(),
            Action::PasteImage => self.attach_clipboard_image(),
            Action::Complete => {
                self.refresh_skill_names();
                self.composer.complete(&self.launch.completer);
            }
            Action::PickPrevious => match self.overlay.as_mut() {
                Some(Overlay::Panel { panel, .. }) => panel.pick_previous(),
                _ => self.composer.pick_previous(),
            },
            Action::PickNext => match self.overlay.as_mut() {
                Some(Overlay::Panel { panel, .. }) => panel.pick_next(),
                _ => self.composer.pick_next(),
            },
            Action::ScrollUp => match self.overlay.as_mut() {
                Some(Overlay::Panel { panel, .. }) => {
                    panel.page_previous(self.viewport.page().unsigned_abs());
                }
                _ => self.viewport.scroll_by(-self.viewport.page()),
            },
            Action::ScrollDown => match self.overlay.as_mut() {
                Some(Overlay::Panel { panel, .. }) => {
                    panel.page_next(self.viewport.page().unsigned_abs());
                }
                _ => self.viewport.scroll_by(self.viewport.page()),
            },
            Action::ScrollTop => match self.overlay.as_mut() {
                Some(Overlay::Panel { panel, .. }) => panel.pick_first(),
                _ => self.viewport.scroll_to_top(),
            },
            Action::ScrollBottom => match self.overlay.as_mut() {
                Some(Overlay::Panel { panel, .. }) => panel.pick_last(),
                _ => self.back_to_bottom(),
            },
            Action::CursorPrevious => {
                let next = self
                    .launch
                    .transcript
                    .collapsible_before(self.viewport.cursor());
                if next.is_some() {
                    self.viewport.set_cursor(next);
                }
            }
            Action::CursorNext => {
                if let Some(current) = self.viewport.cursor() {
                    let next = self.launch.transcript.collapsible_after(current);
                    self.viewport.set_cursor(next);
                }
            }
            Action::ToggleItem => {
                if let Some(index) = self.viewport.cursor() {
                    self.launch.transcript.toggle(index);
                }
            }
            Action::ToggleThinking => {
                let shown = !self.launch.transcript.thinking_shown();
                self.launch.transcript.set_thinking_shown(shown);
            }
            Action::ToggleTools => {
                let expanded = !self.launch.transcript.tools_expanded();
                self.launch.transcript.set_tools_expanded(expanded);
            }
            Action::CycleSendClass => {
                self.launch.send_class = match self.launch.send_class {
                    InboxClass::Steer => InboxClass::Followup,
                    InboxClass::Followup => InboxClass::Interrupt,
                    InboxClass::Interrupt => InboxClass::Steer,
                };
                // The composer's placeholder says what the new class does.
            }
            Action::Redraw => {
                // Not ratatui's own `clear`: it first asks the terminal where the cursor is and
                // waits for the answer, and a terminal that gives none (a harness, some links)
                // leaves the screen stuck until the next key. The terminal is cleared outright
                // and a blank frame drawn, so the next frame rewrites every cell.
                use ratatui::backend::Backend as _;
                let cleared = terminal
                    .backend_mut()
                    .clear_region(ratatui::backend::ClearType::All)
                    .and_then(|()| terminal.draw(|_| {}).map(|_| ()));
                if let Err(error) = cleared {
                    tracing::debug!("failed to clear the screen: {error}");
                }
            }
            Action::Edit => match focus {
                Focus::Panel | Focus::Filtering => self.edit_panel(key).await,
                Focus::Question => self.edit_question(key),
                Focus::Search => self.composer.edit_search(key),
                Focus::Completion | Focus::Composer => {
                    self.viewport.set_cursor(None);
                    self.composer.edit(key);
                    self.suggest();
                }
            },
        }
    }

    fn cycle_permission(&mut self) {
        let level = self.launch.shared_permission.cycle();
        tracing::debug!("permission cycled to {level}");
        // Recorded on the row, not just in this process's cell: a scheduled gate is re-checked at
        // fire time against the row, so a `meka serve` sharing the data directory would keep
        // firing a gate the user has just withdrawn here.
        self.send_host(HostEvent::PermissionChanged(level));
        // Re-said at the moment the user enters the read level, so a misconfigured sandbox
        // surfaces now instead of at the first `shell_execute` failure.
        if level == crate::permission::Permission::Read {
            crate::sandbox::warn_if_sandbox_issues(
                &self.launch.sandbox_state,
                crate::sandbox::WarnContext::ReadModeEntry,
            );
        }
    }

    /// Esc or Ctrl+C while something runs: climb the ladder a turn climbs, so the second press
    /// stops the background tasks and the third leaves. Ctrl+C when idle clears a composer with
    /// text in it.
    async fn interrupt(&mut self) {
        // A running command is the first thing a stop reaches; the ladder is the turn's.
        if let Some(mut run) = self.shell.take() {
            run.kill();
            return;
        }
        if !self.is_busy() && !self.composer.is_empty() && self.overlay.is_none() {
            self.composer.clear();
            return;
        }
        let press = INTERRUPT_RELAY.count_press(&self.launch.cancel);
        match escalation_for(press) {
            // `count_press` has already fired the turn's token, when there was one; the
            // questions go with it, since the token reaches the approval prompt alone.
            Escalation::CancelTurn => {
                self.withdraw_questions();
                // Nothing ran, so the press stopped nothing; what the next ones do is said.
                if !self.is_busy() {
                    self.show_hint("Ctrl+C again stops background tasks, a third leaves");
                }
            }
            Escalation::CancelBackgroundTasks => {
                let stopped = self
                    .launch
                    .agent
                    .background_tasks()
                    .stop_all(
                        &self.launch.agent.store().background_store(),
                        crate::store::background::TaskStatus::Canceled,
                    )
                    .await;
                if let Some(notice) = crate::host::terminal::background_cancellation_notice(stopped)
                {
                    self.push(Item::note(notice));
                }
            }
            Escalation::Leave => self.leave_with(true),
        }
    }

    /// Every way out. A turn still running is stopped first: the host reads the exit only between
    /// turns, and once the screen is gone nothing shows what the turn is doing.
    fn leave(&mut self) {
        self.leave_with(false);
    }

    /// The terminal is gone, noticed by a read or a write that failed rather than by the signal
    /// the host listens for: macOS reports a closed pty to the write first, and a terminal held
    /// through another process sends no signal at all. Every way of noticing ends the same way,
    /// as the signal does: the turn stopped and the interrupted status.
    fn hang_up(&mut self) {
        self.leave_with(true);
    }

    /// Leave, as an interrupt when the ladder's last press asked. A `!command` still running is
    /// killed too: nothing shows it once the screen is gone, and the process is about to exit.
    fn leave_with(&mut self, interrupted: bool) {
        self.launch
            .cancel
            .cancel(crate::store::turns::CancelReason::Client);
        if let Some(mut run) = self.shell.take() {
            run.kill();
        }
        self.send_host(HostEvent::Exit { interrupted });
        self.leaving = true;
    }

    /// Enter on the composer.
    async fn submit(&mut self) {
        let text = self.composer.text();
        let trimmed = text.trim().to_string();
        if trimmed.is_empty() && !self.composer.has_attachments() {
            return;
        }
        // The shell mode with nothing typed yet: Enter runs nothing and leaves the mode on.
        if trimmed == "!" {
            return;
        }
        // An image rides on a prompt that opens a turn; the inbox a busy turn reads carries
        // words alone.
        if self.is_busy() && self.composer.has_attachments() {
            self.reply("An image goes with the next prompt; wait for the turn to end.");
            return;
        }
        // Asked ahead of taking anything, so a command that takes no image leaves the line and the
        // image where they were.
        let routed = if self.composer.has_attachments() && trimmed.starts_with('/') {
            let images: Vec<crate::image::ImageSource> = self
                .composer
                .attachments()
                .iter()
                .map(|attachment| attachment.source.clone())
                .collect();
            match super::commands::with_images(&trimmed, images) {
                super::commands::WithImages::TakesNone => {
                    self.reply(
                        "An image goes with a prompt, a skill or an MCP prompt; send this command \
                         without it.",
                    );
                    return;
                }
                super::commands::WithImages::Invocation(command) => Some(command),
                super::commands::WithImages::Words => None,
            }
        } else {
            None
        };
        let attachments = self.composer.take_attachments();
        if !trimmed.is_empty() {
            self.composer.remember(&trimmed);
        }
        self.composer.clear();
        self.back_to_bottom();
        // Not echoed here, as no command line is: the appended message says it.
        if let Some(command) = routed {
            self.ask_host(HostEvent::Command(command));
            return;
        }
        if !attachments.is_empty() {
            if !trimmed.is_empty() {
                self.push(Item::Prompt {
                    text: trimmed.clone(),
                    class: None,
                    inbox_id: None,
                    delivered: true,
                });
            }
            for _ in &attachments {
                self.push(Item::Image);
            }
            self.ask_host(HostEvent::UserInput {
                parts: if trimmed.is_empty() {
                    Vec::new()
                } else {
                    vec![crate::conversation::ContentBlock::Text { text: trimmed }]
                },
                images: attachments
                    .into_iter()
                    .map(|attachment| attachment.source)
                    .collect(),
            });
            return;
        }

        if trimmed.starts_with('/') {
            self.run_command_line(trimmed).await;
            return;
        }
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            self.leave();
            return;
        }
        if let Some(command) = trimmed.strip_prefix('!') {
            if !command.trim().is_empty() {
                self.run_shell(command);
            }
            return;
        }
        if self.is_busy() {
            self.send_while_busy(trimmed).await;
            return;
        }
        self.push(Item::Prompt {
            text: trimmed.clone(),
            class: None,
            inbox_id: None,
            delivered: true,
        });
        self.ask_host(HostEvent::UserInput {
            parts: vec![crate::conversation::ContentBlock::Text { text: trimmed }],
            images: Vec::new(),
        });
    }

    /// A slash command line, typed or picked. Not echoed as a prompt: a command is not something
    /// said to the agent, and what it shows or does is its own record. The door every command line
    /// passes but an invocation typed with an image, which `submit` routes after the same busy
    /// check, so the rule for a busy host is asked here: a command the host answers would wait on
    /// the turn, and a wait that ran a `/rewind` after the turn it was typed during is a surprise,
    /// so it is refused instead.
    async fn run_command_line(&mut self, line: String) {
        match parse_slash_command(&line) {
            Some(command) => match command.answered_by() {
                Answerer::Screen => self.answer_locally(command).await,
                Answerer::Host if self.is_busy() => {
                    self.reply(SlashCommand::busy_refusal(&line));
                }
                Answerer::Host => self.ask_host(HostEvent::Command(command)),
            },
            None => self.reply(unknown_command_message(&line)),
        }
    }

    /// Enter while a turn runs: a steer or an interrupt goes to the inbox, where the loop reads
    /// it; a followup waits here for the turn to end.
    async fn send_while_busy(&mut self, text: String) {
        let class = self.launch.send_class;
        if class == InboxClass::Followup {
            let id = self.launch.transcript.push(Item::Prompt {
                text: text.clone(),
                class: Some(class),
                inbox_id: None,
                delivered: false,
            });
            self.followups.push_back((id, text));
            return;
        }
        let Some(session_id) = self.launch.agent.session_id() else {
            self.reply("No active session yet.");
            return;
        };
        let item = match crate::store::inbox::NewInboxItem::from_parts(
            session_id,
            class,
            Some("terminal".to_string()),
            text.as_str(),
        ) {
            Ok(item) => item,
            Err(error) => {
                self.reply(error.to_string());
                return;
            }
        };
        match self.launch.store.inbox_store().enqueue(item).await {
            Ok(enqueued) => {
                self.push(Item::Prompt {
                    text,
                    class: Some(class),
                    inbox_id: Some(enqueued.id),
                    delivered: false,
                });
            }
            Err(error) => self.reply(error.to_string()),
        }
    }

    /// A listing a CLI handler prints, as a panel titled with the command that asked for it. A
    /// failure is part of the body: it is what the command has to show.
    fn listing_panel(&mut self, printed: String, outcome: Result<(), impl std::fmt::Display>) {
        let mut body = printed;
        if body.len() > PANEL_TEXT_KEEP_BYTES {
            super::model::keep_tail(&mut body, PANEL_TEXT_KEEP_BYTES);
            body.insert_str(
                0,
                &format!(
                    "(cut to the last {})\n",
                    crate::text::format_size(PANEL_TEXT_KEEP_BYTES)
                ),
            );
        }
        if let Err(error) = outcome {
            if !body.is_empty() && !body.ends_with('\n') {
                body.push('\n');
            }
            body.push_str(&error.to_string());
        }
        self.open_panel(Panel::with_text(&body), PanelKind::Info);
    }

    /// The commands the screen answers itself: what it holds the answer to, and the lists the
    /// store answers, which are reachable mid-turn because the host is not needed for them.
    async fn answer_locally(&mut self, command: SlashCommand) {
        let session_id = self.launch.agent.session_id();
        match command {
            SlashCommand::Exit => self.leave(),
            SlashCommand::Log => self.open_log_panel(),
            SlashCommand::Help => self.open_panel(
                Panel::with_text(&super::commands::help_text()),
                PanelKind::Info,
            ),
            SlashCommand::Session => {
                self.reply(match session_id {
                    Some(id) => format!("Current session: {id}"),
                    None => "No active session yet.".to_string(),
                });
            }
            SlashCommand::Resume(None) => self.open_session_picker(None).await,
            SlashCommand::Profile(None) => {
                let current = crate::sync::read(&self.launch.current_profile).clone();
                let names: Vec<String> = self
                    .launch
                    .configured_profiles
                    .iter()
                    .map(|profile| profile.name.clone())
                    .collect();
                let choices = self
                    .launch
                    .configured_profiles
                    .iter()
                    .map(|profile| Choice {
                        label: profile.name.clone(),
                        key: None,
                        description: Some(format!(
                            "{} on {}",
                            profile.model.as_deref().unwrap_or("-"),
                            profile.account
                        )),
                    })
                    .collect();
                let picked = names.iter().position(|name| *name == current).unwrap_or(0);
                self.open_panel(
                    Panel::with_choices(Vec::new(), choices, picked),
                    PanelKind::Profile(names),
                );
            }
            SlashCommand::Project(super::commands::ProjectVerb::Picker) => {
                let current = self.launch.agent.project();
                // The first row leaves the project: a row rather than a name, since no project is
                // named by nothing. Read from the store now, so a project added since the screen
                // opened is offered.
                let mut names = vec![String::new()];
                match self.launch.store.project_names().await {
                    Ok(projects) => names.extend(projects),
                    Err(error) => {
                        self.reply(error.to_string());
                        return;
                    }
                }
                let choices = names
                    .iter()
                    .map(|name| Choice {
                        label: if name.is_empty() {
                            "No project".to_string()
                        } else {
                            name.clone()
                        },
                        key: None,
                        description: None,
                    })
                    .collect();
                let picked = current
                    .and_then(|current| names.iter().position(|name| *name == current))
                    .unwrap_or(0);
                self.open_panel(
                    Panel::with_choices(Vec::new(), choices, picked),
                    PanelKind::Project(names),
                );
            }
            SlashCommand::Permission(None) => {
                let current = self.launch.shared_permission.get();
                let levels: Vec<crate::permission::Permission> =
                    self.launch.shared_permission.enabled().iter().collect();
                let choices = levels
                    .iter()
                    .map(|level| Choice {
                        label: level.to_string(),
                        key: None,
                        description: None,
                    })
                    .collect();
                let picked = levels
                    .iter()
                    .position(|level| *level == current)
                    .unwrap_or(0);
                self.open_panel(
                    Panel::with_choices(Vec::new(), choices, picked),
                    PanelKind::Permission(levels),
                );
            }
            SlashCommand::Permission(Some(level)) => {
                match level.parse::<crate::permission::Permission>() {
                    Ok(permission) => self.set_permission(permission),
                    Err(error) => self.reply(error),
                }
            }
            SlashCommand::Approvals(argument) => match argument.as_deref().map(str::trim) {
                None | Some("") => {
                    let picked = if self.launch.shared_permission.approvals() {
                        0
                    } else {
                        1
                    };
                    let choices = vec![
                        Choice {
                            label: "on".to_string(),
                            key: None,
                            description: None,
                        },
                        Choice {
                            label: "off".to_string(),
                            key: None,
                            description: None,
                        },
                    ];
                    self.open_panel(
                        Panel::with_choices(Vec::new(), choices, picked),
                        PanelKind::Approvals,
                    );
                }
                Some(word @ ("on" | "off")) => self.set_approvals(word == "on"),
                Some(other) => self.reply(format!(
                    "'{other}' is not a setting: `/approvals` takes on or off"
                )),
            },
            SlashCommand::Cd(argument) => {
                match handle_cd(
                    &self.launch.cwd,
                    &self.launch.launch_cwd,
                    argument.as_deref().unwrap_or(""),
                ) {
                    Ok(moved) => {
                        self.push(Item::note(format!(
                            "Working directory set to: {}",
                            shorten_path_with_tilde(&moved)
                        )));
                        self.send_host(HostEvent::CwdChanged(moved));
                    }
                    Err(message) => self.reply(message),
                }
            }
            // Pickers rather than listings: what each thing is, in a row, and the pick runs or
            // shows it, which is what a person who typed the command is there for. The columns a
            // terminal table carried (priority, age, tags, elapsed) are the show panel's.
            SlashCommand::SkillList => {
                let index = crate::skills::discover_skills_in_roots(&self.launch.skill_roots);
                if index.skills.is_empty() {
                    self.reply("No skills.");
                } else {
                    let rows = index
                        .skills
                        .iter()
                        .map(|skill| {
                            (
                                skill.name.clone(),
                                description_of(&skill.description),
                                format!("/skill {}", skill.name),
                            )
                        })
                        .collect();
                    self.open_run_picker(rows);
                }
            }
            SlashCommand::MemoryList => {
                let store = Arc::clone(self.launch.agent.memories());
                let view = crate::store::memory::MemoryView::for_session(
                    self.launch.agent.project().as_deref(),
                );
                match store.index(&view).await {
                    Ok(index) if index.is_empty() => self.reply("No memories."),
                    Ok(index) => {
                        let rows = index
                            .iter()
                            .map(|memory| {
                                (
                                    memory.name.clone(),
                                    description_of(&memory.description),
                                    format!("/memory {}", memory.name),
                                )
                            })
                            .collect();
                        self.open_run_picker(rows);
                    }
                    Err(error) => self.reply(error.to_string()),
                }
            }
            SlashCommand::MemoryShow { name } => {
                let store = Arc::clone(self.launch.agent.memories());
                let view = crate::store::memory::MemoryView::for_session(
                    self.launch.agent.project().as_deref(),
                );
                let (outcome, printed) =
                    capture(crate::cli::memory::run_show(&store, &name, &view)).await;
                self.listing_panel(printed, outcome);
            }
            SlashCommand::ScheduleList => match session_id {
                Some(id) => match self
                    .launch
                    .store
                    .schedule_store()
                    .list_scheduled_jobs(id)
                    .await
                {
                    Ok(jobs) if jobs.is_empty() => self.reply("No scheduled jobs."),
                    Ok(jobs) => {
                        let shown =
                            crate::text::unique_prefix_len(jobs.iter().map(|job| job.id.as_str()));
                        let jobs = crate::cli::schedule::with_levels(
                            &self.launch.store,
                            &self.launch.schedule,
                            jobs,
                        )
                        .await;
                        let memory = self.launch.store.scheduler_memory();
                        let rows = jobs
                            .iter()
                            .map(|(job, level)| {
                                let id = job.id.get(..shown).unwrap_or(&job.id).to_string();
                                // The one fact a row carries beyond what the job is: whether it
                                // can fire, since a job that cannot is the one to look at.
                                let held = match crate::schedule::job_withheld(
                                    memory, job, *level, None,
                                ) {
                                    crate::schedule::Withheld::Yes(_) => "  held",
                                    crate::schedule::Withheld::Undetermined => "  held?",
                                    crate::schedule::Withheld::No => "",
                                };
                                (
                                    id.clone(),
                                    Some(format!(
                                        "{}  {}{held}",
                                        job.schedule.describe(),
                                        one_line(&job.prompt, 120)
                                    )),
                                    format!("/schedule show {id}"),
                                )
                            })
                            .collect();
                        self.open_run_picker(rows);
                    }
                    Err(error) => self.reply(error.to_string()),
                },
                None => self.reply("No active session yet."),
            },
            SlashCommand::ScheduleShow { id } => match session_id {
                Some(session) => {
                    let (outcome, printed) = capture(crate::cli::schedule::show(
                        &self.launch.store,
                        &id,
                        &self.launch.schedule,
                        Some(session),
                    ))
                    .await;
                    self.listing_panel(printed, outcome);
                }
                None => self.reply("No active session yet."),
            },
            SlashCommand::TaskList => match session_id {
                Some(id) => match self
                    .launch
                    .store
                    .background_store()
                    .list_background_tasks(id)
                    .await
                {
                    Ok(tasks) if tasks.is_empty() => self.reply("No background tasks."),
                    Ok(tasks) => {
                        let shown = crate::text::unique_prefix_len(
                            tasks.iter().map(|task| task.id.as_str()),
                        );
                        let rows = tasks
                            .iter()
                            .map(|task| {
                                let id = task.id.get(..shown).unwrap_or(&task.id).to_string();
                                (
                                    id.clone(),
                                    Some(format!(
                                        "{}  {}  {}",
                                        task.status.name(),
                                        task.tool,
                                        one_line(&task.label, 120)
                                    )),
                                    format!("/task show {id}"),
                                )
                            })
                            .collect();
                        self.open_run_picker(rows);
                    }
                    Err(error) => self.reply(error.to_string()),
                },
                None => self.reply("No active session yet."),
            },
            SlashCommand::ScheduleCancel { id } => match session_id {
                Some(session) => {
                    match self
                        .launch
                        .store
                        .schedule_store()
                        .cancel_scheduled_job(session, &id)
                        .await
                    {
                        Ok(Some(canceled)) => self.push(Item::note(format!(
                            "Canceled job {}.",
                            &canceled[..8.min(canceled.len())]
                        ))),
                        Ok(None) => self.reply(format!("No scheduled job matching '{id}'.")),
                        Err(error) => self.reply(error.to_string()),
                    }
                }
                None => self.reply("No active session yet."),
            },
            // Recorded first, then signaled: `finish_background_task` only overwrites a `running`
            // row, so a task finishing in the same instant cannot report success after the user
            // was told it stopped. Answered here so a build can be stopped while a turn runs.
            SlashCommand::TaskCancel { id } => match session_id {
                Some(session) => {
                    match crate::cli::background::cancel(&self.launch.store, session, id.as_deref())
                        .await
                    {
                        Ok(canceled) if canceled.is_empty() => {
                            self.reply("No running background tasks.");
                        }
                        Ok(canceled) => {
                            for task_id in &canceled {
                                self.launch.agent.background_tasks().cancel(task_id).await;
                            }
                            self.push(Item::note(format!(
                                "Canceling {}",
                                crate::text::counted(
                                    canceled.len() as u64,
                                    "background task",
                                    "background tasks"
                                )
                            )));
                        }
                        Err(error) => self.reply(error.to_string()),
                    }
                }
                None => self.reply("No active session yet."),
            },
            SlashCommand::TaskShow { id } => match session_id {
                Some(session) => {
                    let (outcome, printed) = capture(crate::cli::background::show(
                        &self.launch.store,
                        session,
                        &id,
                        crate::render::Stream::Stderr,
                    ))
                    .await;
                    self.listing_panel(printed, outcome);
                }
                None => self.reply("No active session yet."),
            },
            other => {
                debug_assert!(
                    other.answered_by() == Answerer::Screen,
                    "the host answers this command"
                );
            }
        }
    }

    /// `!command`: run it in the session's directory and show what it printed. Captured rather
    /// than given the terminal, so the output is part of the transcript and scrolls with it; a
    /// command that needs a terminal of its own belongs in a shell. It runs beside the screen,
    /// which keeps drawing and reading keys, and Esc or Ctrl+C kills it with everything it
    /// spawned.
    fn run_shell(&mut self, command: &str) {
        if self.shell.is_some() {
            self.reply("A command is still running; Esc stops it.");
            return;
        }
        self.push(Item::Prompt {
            text: format!("!{command}"),
            class: None,
            inbox_id: None,
            delivered: true,
        });
        let child = match spawn_shell_command(command, &self.launch.cwd.get()) {
            Ok(child) => child,
            Err(error) => {
                self.reply(format!("Failed to execute command: {error}"));
                return;
            }
        };
        let run = self.next_shell_run;
        self.next_shell_run += 1;
        let stop = tokio_util::sync::CancellationToken::new();
        self.shell = Some(ShellRun {
            id: run,
            started: Instant::now(),
            stop: stop.clone(),
            pid: child.id(),
            ended: false,
        });
        let ui = self.launch.ui.clone();
        tokio::spawn(async move {
            let (output, stopped) = wait_for_shell_command(child, stop).await;
            ui.send(UiEvent::ShellFinished {
                run,
                output,
                stopped,
            });
        });
    }

    /// What a `!command` came to: its output on the transcript, and a row for how it ended when
    /// it did not succeed.
    fn finish_shell(
        &mut self,
        run: u64,
        output: Result<super::frontend::ShellOutput, String>,
        stopped: bool,
    ) {
        if self.shell.as_ref().is_some_and(|running| running.id == run) {
            if let Some(running) = self.shell.as_mut() {
                running.ended = true;
            }
            self.shell = None;
        }
        match output {
            Ok(output) => {
                let mut printed = String::from_utf8_lossy(&output.stdout).into_owned();
                let errors = String::from_utf8_lossy(&output.stderr);
                if !errors.trim().is_empty() {
                    if !printed.is_empty() && !printed.ends_with('\n') {
                        printed.push('\n');
                    }
                    printed.push_str(&errors);
                }
                let lines_cut = keep_last_lines(&mut printed, SHELL_OUTPUT_KEEP_LINES);
                if lines_cut || output.dropped_bytes > 0 {
                    let kept = if lines_cut {
                        format!("{SHELL_OUTPUT_KEEP_LINES} lines")
                    } else {
                        crate::text::format_size(SHELL_OUTPUT_KEEP_BYTES)
                    };
                    self.push(Item::note(format!("Output cut to its last {kept}")));
                }
                if !printed.trim().is_empty() {
                    self.push(Item::ShellOutput(printed));
                }
                if stopped {
                    self.push(Item::interrupted());
                } else if !output.status.success()
                    && let Some(code) = output.status.code()
                {
                    self.push(Item::note(format!("Command exited with status {code}")));
                }
            }
            Err(error) => self.reply(format!("Failed to execute command: {error}")),
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        let over_overlay = self
            .overlay_area
            .is_some_and(|area| mouse.row >= area.y && mouse.row < area.y + area.height);
        // Every mouse event says where the pointer is; the item under it lights up, and only a
        // change of item is worth a frame, since motion arrives many times a second.
        let area = self.transcript_area;
        let over_transcript = (mouse.row >= area.y && mouse.row < area.y + area.height)
            .then(|| usize::from(mouse.row - area.y));
        if self.viewport.set_hover_row(over_transcript) {
            self.dirty = true;
        }
        let on_sticky = over_transcript == Some(0) && self.viewport.sticky_prompt().is_some();
        if on_sticky != self.sticky_hovered {
            self.sticky_hovered = on_sticky;
            self.dirty = true;
        }
        if matches!(mouse.kind, MouseEventKind::Moved) {
            return;
        }
        self.dirty = true;
        match mouse.kind {
            MouseEventKind::ScrollUp => match self.overlay.as_mut() {
                Some(Overlay::Panel { panel, .. }) if over_overlay => panel.scroll_by(-3),
                _ => self.viewport.scroll_by(-3),
            },
            MouseEventKind::ScrollDown => match self.overlay.as_mut() {
                Some(Overlay::Panel { panel, .. }) if over_overlay => panel.scroll_by(3),
                _ => self.viewport.scroll_by(3),
            },
            // A press starts a selection; whether it was a click is known at the release.
            MouseEventKind::Down(MouseButton::Left) => {
                if let Some(pill) = self.pill
                    && mouse.row == pill.y
                    && mouse.column >= pill.x
                    && mouse.column < pill.x + pill.width
                {
                    self.back_to_bottom();
                    return;
                }
                // The bar at the top: a click brings its prompt to the row under it, and the
                // prompt before takes the bar, so clicking on walks back prompt by prompt.
                if on_sticky && let Some((_, start)) = self.viewport.sticky_prompt() {
                    self.viewport.scroll_to(start.saturating_sub(1));
                    self.selection = None;
                    return;
                }
                self.selection = over_transcript.map(|_| Selection {
                    anchor: (mouse.column, mouse.row),
                    head: (mouse.column, mouse.row),
                    dragged: false,
                });
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some(selection) = self.selection.as_mut() {
                    let last_row = (area.y + area.height).saturating_sub(1).max(area.y);
                    selection.head = (mouse.column, mouse.row.clamp(area.y, last_row));
                    selection.dragged = true;
                }
            }
            // Released after a drag: the selection is copied and stays lit until the next key
            // or press. Released where it was pressed: the click toggles the item under it.
            MouseEventKind::Up(MouseButton::Left) => {
                let Some(selection) = self.selection else {
                    return;
                };
                if selection.dragged {
                    let text = self.selected_text(selection);
                    let rows = text.lines().count();
                    if rows > 0 {
                        copy_to_clipboard(&text);
                        self.show_hint(format!(
                            "copied {rows} row{}",
                            if rows == 1 { "" } else { "s" }
                        ));
                    }
                } else {
                    self.selection = None;
                    if let Some(row) = over_transcript
                        && let Some(index) = self.viewport.item_at_row(row)
                    {
                        self.launch.transcript.toggle(index);
                    }
                }
            }
            _ => {}
        }
    }

    /// The text under a selection, from the cells of the last frame: whole rows between the
    /// ends, the ends cut at their columns, trailing blanks dropped.
    fn selected_text(&self, selection: Selection) -> String {
        let area = self.transcript_area;
        let (from, to) = selection.ordered();
        let column_of = |column: u16| usize::from(column.saturating_sub(area.x));
        let cells: Vec<Vec<String>> = (from.1..=to.1)
            .filter(|&y| y >= area.y && y < area.y + area.height)
            .map(|y| {
                (area.x..area.x + area.width)
                    .map(|x| {
                        self.drawn
                            .cell((x, y))
                            .map(|cell| cell.symbol().to_string())
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .collect();
        if cells.is_empty() {
            return String::new();
        }
        text_between(
            &cells,
            (0, column_of(from.0)),
            (cells.len() - 1, column_of(to.0)),
        )
    }

    /// Keep the frame just drawn, for the selection that may follow.
    fn remember_drawn(&mut self, buffer: &ratatui::buffer::Buffer) {
        self.drawn = buffer.clone();
    }

    /// Light the selection on the frame: the cells between its ends, reversed.
    fn light_selection(&self, frame: &mut Frame<'_>) {
        let Some(selection) = self.selection.filter(|selection| selection.dragged) else {
            return;
        };
        let area = self.transcript_area;
        let (from, to) = selection.ordered();
        let right = (area.x + area.width).saturating_sub(1);
        for y in from.1..=to.1 {
            if y < area.y || y >= area.y + area.height {
                continue;
            }
            let first = if y == from.1 {
                from.0.max(area.x)
            } else {
                area.x
            };
            let last = if y == to.1 { to.0.min(right) } else { right };
            for x in first..=last {
                if let Some(cell) = frame.buffer_mut().cell_mut((x, y)) {
                    cell.modifier |= ratatui::style::Modifier::REVERSED;
                }
            }
        }
    }

    /// Take what was logged since the last frame: a row for what the user's filter shows, and
    /// a place in the ring for everything, which is what `/log` reads.
    fn drain_logs(&mut self) {
        while let Some(line) = self.launch.logs.try_recv() {
            if line.shown {
                self.push(log_item(line.clone()));
            }
            if self.log_ring.len() == LOG_RING_LINES {
                self.log_ring.pop_front();
            }
            self.log_ring.push_back(line);
        }
    }

    /// Ctrl+V: the image on the clipboard, attached to the next prompt. Read from the process's
    /// own clipboard, since a terminal pastes text alone; over SSH, or without a display, there
    /// is none to read and the reply says so.
    fn attach_clipboard_image(&mut self) {
        match clipboard_image() {
            Ok(Some((label, bytes))) => self.attach_image(&bytes, label),
            Ok(None) => self.reply("No image on the clipboard."),
            Err(error) => self.reply(format!("Cannot read the clipboard: {error}")),
        }
    }

    /// Attach image bytes to the next prompt, through the same door the HTTP API's images take,
    /// so the limits and the format rules are one set.
    fn attach_image(&mut self, bytes: &[u8], label: String) {
        if !self.launch.agent.cells().profile.current().vision {
            self.reply(format!(
                "The profile '{}' does not accept images.",
                crate::sync::read(&self.launch.current_profile)
            ));
            return;
        }
        match crate::image::prepare_image_source(crate::image::classify_bytes(bytes), bytes) {
            Ok(source) => {
                self.composer
                    .attach(super::input::Attachment { source, label });
                self.dirty = true;
            }
            Err(error) => self.reply(error),
        }
    }

    /// A paste. A single path to an image file attaches the file, which is how a drag from a
    /// file manager arrives; anything else is text for the composer.
    fn paste(&mut self, text: String) {
        // A terminal pastes a line break as the carriage return Enter sends, inside a bracketed
        // paste as much as outside one; the composer's lines end in `\n`.
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        // A form's field takes the paste as one line, and so does a picker's filter while one is
        // being typed, as a session id pasted into `/resume`'s; a panel otherwise has nowhere to
        // put one.
        match self.overlay.as_mut() {
            Some(Overlay::Elicitation(elicitation)) => {
                elicitation
                    .answer
                    .push_str(&crate::text::sanitize_to_line(&text, usize::MAX));
                self.dirty = true;
                return;
            }
            Some(Overlay::Panel { panel, .. }) => {
                if panel.is_typing() {
                    for character in crate::text::sanitize_to_line(&text, usize::MAX).chars() {
                        panel.filter_type(character);
                    }
                    self.dirty = true;
                }
                return;
            }
            None => {}
        }
        self.dirty = true;
        if let Some(path) = pasted_image_path(&text) {
            let label = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            match std::fs::read(&path) {
                Ok(bytes) => self.attach_image(&bytes, label),
                Err(error) => {
                    self.reply(format!("Cannot read '{}': {error}", path.display()));
                }
            }
            return;
        }
        self.composer.insert(&text);
        self.suggest();
    }

    /// `/log`: the lines kept, newest last, each with its time, level and target.
    fn open_log_panel(&mut self) {
        if self.log_ring.is_empty() {
            self.reply("No log lines yet.");
            return;
        }
        let body: String = self
            .log_ring
            .iter()
            .map(|line| {
                format!(
                    "{} {:<5} {}: {}\n",
                    crate::text::format_timestamp(line.at, crate::text::Precision::Seconds),
                    line.level,
                    line.target,
                    line.message
                )
            })
            .collect();
        self.open_panel(Panel::with_text(&body).scrolled_to_end(), PanelKind::Info);
    }

    fn expire_hint(&mut self) {
        if let Some((_, since)) = &self.hint
            && since.elapsed() > HINT_LIFETIME
        {
            self.hint = None;
            self.dirty = true;
        }
    }

    /// Read back how much work runs beside the turn, once a second: the session's background
    /// tasks from the store, the sub-agents from the process. A change is a reason to draw.
    async fn poll_work(&mut self) {
        if self
            .work_polled
            .is_some_and(|at| at.elapsed() < WORK_POLL_INTERVAL)
        {
            return;
        }
        self.work_polled = Some(Instant::now());
        let tasks = match self.launch.agent.session_id() {
            Some(session_id) => match self
                .launch
                .store
                .background_store()
                .list_running_background_tasks(session_id)
                .await
            {
                // A sub-agent run in the background is a task row too; it is counted as the
                // sub-agent it is.
                Ok(rows) => rows
                    .iter()
                    .filter(|task| task.tool != "agent_spawn")
                    .count(),
                Err(error) => {
                    tracing::debug!("failed to count the running tasks: {error}");
                    self.work.tasks
                }
            },
            None => 0,
        };
        let work = view::RunningWork {
            tasks,
            subagents: self.launch.running_subagents.running_ids().len(),
        };
        if work != self.work {
            self.work = work;
            self.dirty = true;
        }
    }

    /// The status row counts the seconds of work; a new second is a reason to draw.
    /// The working row's word and how long the work has run, while something runs: the turn's
    /// activity, stilled to a wait while a question is open, or the `!command`.
    fn working_row(&self) -> Option<(view::Activity, Duration)> {
        let since = self.working_since()?;
        let activity = if self.busy.is_none() {
            view::Activity::Shell
        } else if self.overlay_is_question() {
            view::Activity::Waiting
        } else if self.compacting && self.activity == view::Activity::Working {
            view::Activity::Compacting
        } else {
            self.activity
        };
        Some((activity, since.elapsed()))
    }

    fn tick_working(&mut self) {
        if let Some((activity, elapsed)) = self.working_row() {
            let spinner = if activity.is_waiting() {
                self.shown_working.1
            } else {
                view::spinner_frame(elapsed)
            };
            let shown = (elapsed.as_secs(), spinner);
            if shown != self.shown_working {
                self.shown_working = shown;
                self.dirty = true;
            }
        }
    }

    fn draw(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        if area.height < 4 || area.width < 4 {
            return;
        }
        let status_height = 1;
        // A blank row between the transcript and what sits under it, always, so neither the field
        // nor the working row sits flush against the last line of the answer; the back-to-bottom
        // pill lives on the blank row, and the working row has one of its own under it while the
        // agent works.
        let working = self.working_row();
        let gap = 1 + u16::from(working.is_some());
        let most_composer = (area.height / COMPOSER_SHARE).max(1);
        let most_panel = (area.height / PANEL_SHARE).max(2);
        let bottom_height = match &mut self.overlay {
            Some(Overlay::Panel { panel, .. }) => panel.height(area.width as usize, most_panel),
            Some(Overlay::Elicitation(elicitation)) => u16::try_from(elicitation.body.len())
                .unwrap_or(u16::MAX)
                .saturating_add(1)
                .min(most_panel)
                .max(2),
            None if self.composer.search.is_some() => 1 + status_height,
            None => {
                self.composer
                    .height(area.width.saturating_sub(TEXT_EDGE as u16), most_composer)
                    + status_height
            }
        };
        let transcript_height = area.height.saturating_sub(bottom_height + gap).max(1);
        let transcript_area = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: transcript_height,
        };
        let gap_area = Rect {
            x: area.x,
            y: area.y + transcript_height,
            width: area.width,
            height: gap,
        };
        let bottom_area = Rect {
            x: area.x,
            y: area.y + transcript_height + gap,
            width: area.width,
            height: bottom_height,
        };
        self.transcript_area = transcript_area;

        self.viewport
            .draw(frame, transcript_area, &self.launch.transcript);
        if let Some((line, _)) = self.viewport.sticky_prompt() {
            view::draw_sticky_prompt(
                frame,
                Rect {
                    height: 1.min(transcript_area.height),
                    ..transcript_area
                },
                line,
                self.sticky_hovered,
            );
        }
        // The pill sits on the blank row, or on the working row while there is one, so it never
        // hangs in the air above the words that say the agent is at work.
        let mut pill_row = Rect {
            height: 1,
            ..gap_area
        };
        let mut taken = 0;
        if let Some((activity, elapsed)) = &working {
            pill_row.y = gap_area.y + 1;
            taken =
                view::draw_working_row(frame, pill_row, activity, *elapsed, self.shown_working.1);
        }
        self.pill = (!self.viewport.is_following())
            .then(|| view::draw_back_to_bottom(frame, pill_row, taken));

        match &mut self.overlay {
            Some(Overlay::Panel { panel, .. }) => {
                panel.draw(frame, bottom_area);
                self.overlay_area = Some(bottom_area);
            }
            Some(Overlay::Elicitation(elicitation)) => {
                view::draw_question(
                    frame,
                    bottom_area,
                    &elicitation.body,
                    &elicitation.question(),
                    &elicitation.answer,
                );
                self.overlay_area = Some(bottom_area);
            }
            None => {
                self.overlay_area = None;
                let composer_area = Rect {
                    height: bottom_height - status_height,
                    ..bottom_area
                };
                let status_area = Rect {
                    y: bottom_area.y + composer_area.height,
                    height: status_height,
                    ..bottom_area
                };
                if let Some(search) = &self.composer.search {
                    let shown = search.picked_match().unwrap_or("");
                    let line = Line::from(vec![
                        Span::styled(format!("(reverse-i-search) '{}': ", search.query), gray()),
                        Span::raw(shown.to_string()),
                    ]);
                    frame.render_widget(Paragraph::new(Text::from(line)), composer_area);
                } else {
                    // The block is the composer's gray across the row; the lead is the prompt
                    // glyph dimmed so the words outweigh it, and the text sits at the edge so a
                    // second line starts flush under the first. The permission level is the status
                    // row's to say, so the lead says nothing twice.
                    let lead = TEXT_EDGE as u16;
                    let gutter = Rect {
                        width: lead.min(composer_area.width),
                        ..composer_area
                    };
                    let field = Rect {
                        x: composer_area.x + lead,
                        width: composer_area.width.saturating_sub(lead),
                        ..composer_area
                    };
                    let block = Style::default().bg(super::input::COMPOSER_BACKGROUND);
                    frame.render_widget(ratatui::widgets::Block::new().style(block), composer_area);
                    // While the agent works, the empty composer says what Enter does with the
                    // next message, which is where a person about to type looks.
                    self.composer.set_placeholder(if self.busy.is_some() {
                        send_class_placeholder(self.launch.send_class)
                    } else {
                        ""
                    });
                    // The glyph sits on the text's first row, under the chips when there are any:
                    // the prompt's `>`, or the shell mode's red `!`.
                    let glyph_row = self.composer.attachment_rows();
                    let (glyph, glyph_color) = if self.composer.is_shell_mode() {
                        (draw::SHELL_GLYPH, ratatui::style::Color::Red)
                    } else {
                        (draw::PROMPT_GLYPH, draw::PROMPT_GLYPH_COLOR)
                    };
                    let bar: Vec<Line<'static>> = (0..gutter.height)
                        .map(|row| {
                            if row == glyph_row {
                                Line::from(Span::styled(
                                    format!("{glyph:<TEXT_EDGE$}"),
                                    block.fg(glyph_color),
                                ))
                            } else {
                                Line::from(Span::styled("  ".to_string(), block))
                            }
                        })
                        .collect();
                    frame.render_widget(Paragraph::new(Text::from(bar)), gutter);
                    self.composer.draw(frame, field);
                    if let Some(completion) = &self.composer.completion {
                        let rows: Vec<view::ListRow> = completion
                            .rows
                            .iter()
                            .map(|row| view::ListRow {
                                value: row.value.clone(),
                                description: row.description.clone(),
                            })
                            .collect();
                        // Over the gap row too, so the list sits flush on the composer.
                        let above = Rect {
                            height: transcript_area.height + gap_area.height,
                            ..transcript_area
                        };
                        view::draw_completions(frame, above, &rows, completion.picked);
                    }
                }
                let facts = StatusFacts {
                    shown: &self.launch.status_row,
                    permission: self.launch.shared_permission.get(),
                    path: Some(shorten_path_with_tilde(&self.launch.cwd.get())),
                    project: self.launch.agent.project(),
                    profile: self.profile_fact(),
                    model: self.model_fact(),
                    context: self.context_gauge(),
                    work: self.work,
                    hint: self.hint.as_ref().map(|(hint, _)| hint.as_str()),
                    shell_mode: self.composer.is_shell_mode(),
                };
                frame.render_widget(
                    Paragraph::new(Text::from(view::status_line(
                        &facts,
                        status_area.width as usize,
                    )))
                    .style(Style::default().bg(view::STATUS_BACKGROUND)),
                    status_area,
                );
            }
        }
        self.light_selection(frame);
        view::refresh_after_wide_symbols(frame.buffer_mut(), &self.wide_cells);
        self.wide_cells = view::wide_symbols(frame.buffer_mut());
    }
}

/// The widths of the session picker's columns, the same on every row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SessionColumns {
    /// The title's, the rest of the row after the facts.
    title: usize,
    /// The directory column's; 0 when no row has a directory or the row is too narrow for one.
    directory: usize,
    /// The profile column's; 0 on a row too narrow to carry one.
    profile: usize,
    /// The age column's, right-aligned on the edge.
    age: usize,
}

impl SessionColumns {
    /// The columns of a row `width` wide, from the longest value each fact column holds: the
    /// directory column takes at most its share of the row, the profile column at most a name's
    /// width, the age column what its longest needs, and the title the rest. A title under its
    /// floor drops the directory column, then the profile; the age stays, so a row always says
    /// when.
    fn fit(
        width: usize,
        longest_directory: usize,
        longest_profile: usize,
        longest_age: usize,
    ) -> Self {
        let mut columns = Self {
            title: 0,
            directory: longest_directory.min(width / SESSION_DIRECTORY_SHARE),
            profile: longest_profile.min(crate::text::NAME_WIDTH),
            age: longest_age,
        };
        columns.title = columns.title_room(width);
        if columns.title < SESSION_TITLE_FLOOR {
            columns.directory = 0;
            columns.title = columns.title_room(width);
        }
        if columns.title < SESSION_TITLE_FLOOR {
            columns.profile = 0;
            columns.title = columns.title_room(width).max(SESSION_TITLE_FLOOR);
        }
        columns
    }

    /// What the title has on a row `width` wide: the rest after the text edge, the two columns
    /// the facts stand off by, and the one they keep clear of the right edge.
    fn title_room(&self, width: usize) -> usize {
        width.saturating_sub(TEXT_EDGE + 2 + self.facts_width() + 1)
    }

    /// The columns the facts take together: the ones present, two apart.
    fn facts_width(&self) -> usize {
        let present = [self.directory, self.profile, self.age]
            .into_iter()
            .filter(|columns| *columns > 0);
        present.clone().sum::<usize>() + 2 * present.count().saturating_sub(1)
    }

    /// One row's facts in their columns: the directory padded to its column, or cut from its
    /// start so its tail stays; the profile padded or cut at its end; the age right-aligned.
    fn facts(&self, directory: &str, profile: &str, age: &str) -> String {
        let blanks = |text: &str, columns: usize| {
            " ".repeat(columns.saturating_sub(crate::text::display_width(text)))
        };
        let mut cells = Vec::new();
        if self.directory > 0 {
            let directory = draw::cut_text_start(directory, self.directory);
            cells.push(format!("{directory}{}", blanks(&directory, self.directory)));
        }
        if self.profile > 0 {
            let profile = draw::cut_text(profile, self.profile);
            cells.push(format!("{profile}{}", blanks(&profile, self.profile)));
        }
        cells.push(format!("{}{age}", blanks(age, self.age)));
        cells.join("  ")
    }
}

/// What the empty composer says while a turn runs: what Enter does with the next message under
/// the send class in force.
fn send_class_placeholder(class: InboxClass) -> &'static str {
    match class {
        InboxClass::Steer => "Steer the agent",
        InboxClass::Followup => "Queue a followup",
        InboxClass::Interrupt => "Interrupt the agent",
    }
}

/// The text between two cells of a grid, as (row, column) pairs in reading order: whole rows
/// between the ends, the end rows cut at their columns, trailing blanks dropped from each row.
fn text_between(cells: &[Vec<String>], from: (usize, usize), to: (usize, usize)) -> String {
    let mut rows = Vec::new();
    for (row, line) in cells.iter().enumerate().take(to.0 + 1).skip(from.0) {
        let first = if row == from.0 { from.1 } else { 0 };
        let last = if row == to.0 {
            to.1.min(line.len().saturating_sub(1))
        } else {
            line.len().saturating_sub(1)
        };
        let text: String = line
            .get(first..=last)
            .map(|cells| cells.concat())
            .unwrap_or_default();
        rows.push(text.trim_end().to_string());
    }
    rows.join("\n")
}

/// Put `text` on the clipboard, both ways there are: an OSC 52 sequence, which the terminal
/// applies to its own clipboard and which carries over SSH and through tmux where they allow it,
/// and the operating system's clipboard where this process can reach it.
fn copy_to_clipboard(text: &str) {
    use std::io::Write as _;
    let encoded = base64::engine::general_purpose::STANDARD.encode(text);
    let mut out = std::io::stdout();
    if let Err(error) = out
        .write_all(format!("\x1b]52;c;{encoded}\x07").as_bytes())
        .and_then(|()| out.flush())
    {
        tracing::debug!("failed to write the clipboard sequence: {error}");
    }
    match arboard::Clipboard::new().and_then(|mut clipboard| clipboard.set_text(text.to_string())) {
        Ok(()) => {}
        Err(error) => tracing::debug!("the system clipboard did not take the text: {error}"),
    }
}

/// The image on the clipboard as PNG bytes with its pixel size, `None` when the clipboard holds
/// no image. The terminal cannot paste one: a paste is text, so this asks the operating system.
fn clipboard_image() -> Result<Option<(String, Vec<u8>)>, String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|error| error.to_string())?;
    let image = match clipboard.get_image() {
        Ok(image) => image,
        Err(arboard::Error::ContentNotAvailable) => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    let (width, height) = (image.width, image.height);
    let label = format!("{width}x{height}");
    let png = png_from_rgba(width, height, image.bytes.into_owned())?;
    Ok(Some((label, png)))
}

/// RGBA pixels as a PNG, the one format every provider takes.
fn png_from_rgba(width: usize, height: usize, pixels: Vec<u8>) -> Result<Vec<u8>, String> {
    let (width, height) = (
        u32::try_from(width).map_err(|_| "image too wide".to_string())?,
        u32::try_from(height).map_err(|_| "image too tall".to_string())?,
    );
    let image = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| "the clipboard image's size does not match its pixels".to_string())?;
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut png, image::ImageFormat::Png)
        .map_err(|error| error.to_string())?;
    Ok(png.into_inner())
}

/// The image file a paste names, when the whole paste is one path to one: a drag from a file
/// manager lands as the path, quoted or escaped where it has spaces, and a typed path is the
/// same thing on purpose.
fn pasted_image_path(text: &str) -> Option<PathBuf> {
    let text = text.trim();
    if text.is_empty() || text.contains('\n') {
        return None;
    }
    let unquoted = text
        .strip_prefix('\'')
        .and_then(|rest| rest.strip_suffix('\''))
        .or_else(|| {
            text.strip_prefix('"')
                .and_then(|rest| rest.strip_suffix('"'))
        })
        .unwrap_or(text)
        .replace("\\ ", " ");
    let expanded = match unquoted.strip_prefix("~/") {
        Some(rest) => dirs::home_dir()?.join(rest),
        None => PathBuf::from(unquoted),
    };
    let extension = expanded.extension()?.to_str()?.to_ascii_lowercase();
    if !matches!(
        extension.as_str(),
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp"
    ) {
        return None;
    }
    expanded.is_file().then_some(expanded)
}

/// A log line as the row its level asks for: a warning and an error in their glyphs, the rest,
/// which only `-v` lets through, dimmed with the module that wrote them.
fn log_item(line: crate::relay::LogLine) -> Item {
    match line.level {
        tracing::Level::ERROR => Item::Error(line.message),
        tracing::Level::WARN => Item::Notice {
            level: crate::frontend::NoticeLevel::Warn,
            text: line.message,
        },
        _ => Item::note(format!("{}: {}", line.target, line.message)),
    }
}

/// Start `command` through the shell in `working_dir`, its output captured and its input closed:
/// the terminal is the screen's, so the command gets no stdin to read from it.
fn spawn_shell_command(
    command: &str,
    working_dir: &Path,
) -> std::io::Result<crate::process::Spawned<tokio::process::Child>> {
    #[cfg(windows)]
    let mut shell = {
        let mut shell = tokio::process::Command::new("powershell");
        shell.arg("-Command");
        shell
    };
    #[cfg(not(windows))]
    let mut shell = {
        let mut shell = tokio::process::Command::new("sh");
        shell.arg("-c");
        shell
    };
    shell
        .arg(command)
        .current_dir(working_dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    // A group of its own, so a stop reaches what the command started and not only the shell.
    #[cfg(unix)]
    shell.process_group(0);
    crate::process::spawn(&mut shell)
}

/// Wait for a started command to exit, or for `stop` to fire, which kills it with every process
/// it started; then what it printed. The second value says whether it was stopped.
async fn wait_for_shell_command(
    mut child: crate::process::Spawned<tokio::process::Child>,
    stop: tokio_util::sync::CancellationToken,
) -> (Result<super::frontend::ShellOutput, String>, bool) {
    let mut stdout = tokio::spawn(read_tail(child.stdout.take()));
    let mut stderr = tokio::spawn(read_tail(child.stderr.take()));
    let (status, stopped) = tokio::select! {
        biased;
        () = stop.cancelled() => {
            signal_shell_group(child.id());
            #[cfg(windows)]
            if let Err(error) = child.terminate_tree() {
                tracing::debug!("failed to terminate the command's job: {error}");
            }
            if let Err(error) = child.kill().await {
                tracing::debug!("failed to kill the command: {error}");
            }
            (child.wait().await, true)
        }
        status = child.wait() => (status, false),
    };
    let status = match status {
        Ok(status) => status,
        Err(error) => return (Err(error.to_string()), stopped),
    };
    let ((stdout, stdout_dropped), (stderr, stderr_dropped)) =
        tokio::join!(drained_within(&mut stdout), drained_within(&mut stderr));
    (
        Ok(super::frontend::ShellOutput {
            status,
            stdout,
            stderr,
            dropped_bytes: stdout_dropped + stderr_dropped,
        }),
        stopped,
    )
}

/// Cut `text` to its last `keep` lines; whether anything was cut.
fn keep_last_lines(text: &mut String, keep: usize) -> bool {
    let body = text.len() - usize::from(text.ends_with('\n'));
    let Some((at, _)) = text[..body].rmatch_indices('\n').nth(keep.max(1) - 1) else {
        return false;
    };
    text.drain(..=at);
    true
}

/// The last [`SHELL_OUTPUT_KEEP_BYTES`] of what `pipe` has to give and how many bytes ahead of
/// them were let go, or nothing for a pipe the child did not get.
async fn read_tail(pipe: Option<impl tokio::io::AsyncRead + Unpin>) -> (Vec<u8>, u64) {
    use tokio::io::AsyncReadExt as _;
    let mut kept = Vec::new();
    let mut dropped = 0u64;
    let Some(mut pipe) = pipe else {
        return (kept, dropped);
    };
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        match pipe.read(&mut chunk).await {
            Ok(0) => break,
            Ok(read) => {
                kept.extend(chunk.iter().take(read));
                // Trimmed once the kept part has doubled, so a long output is moved a few
                // times rather than once per read.
                if kept.len() > 2 * SHELL_OUTPUT_KEEP_BYTES {
                    let cut = kept.len() - SHELL_OUTPUT_KEEP_BYTES;
                    kept.drain(..cut);
                    dropped += cut as u64;
                }
            }
            Err(error) => {
                tracing::debug!("failed to read the command's output: {error}");
                break;
            }
        }
    }
    if kept.len() > SHELL_OUTPUT_KEEP_BYTES {
        let cut = kept.len() - SHELL_OUTPUT_KEEP_BYTES;
        kept.drain(..cut);
        dropped += cut as u64;
    }
    (kept, dropped)
}

/// What a drain has read once the command has ended, given [`SHELL_DRAIN_GRACE`] to finish: a
/// process the command left behind can hold the pipe open, and the screen does not wait on it.
async fn drained_within(drain: &mut tokio::task::JoinHandle<(Vec<u8>, u64)>) -> (Vec<u8>, u64) {
    match tokio::time::timeout(SHELL_DRAIN_GRACE, &mut *drain).await {
        Ok(Ok(read)) => read,
        Ok(Err(error)) => {
            tracing::debug!("the command's output drain ended early: {error}");
            (Vec::new(), 0)
        }
        Err(_) => {
            drain.abort();
            tracing::debug!(
                "gave up on the rest of the command's output after {seconds}s; something it \
                 started still holds the pipe",
                seconds = SHELL_DRAIN_GRACE.as_secs()
            );
            (Vec::new(), 0)
        }
    }
}

/// Kill everything a command started along with it: on Unix the process group it was given,
/// named by its own pid. Elsewhere the job the child runs in is terminated by whoever holds it.
fn signal_shell_group(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        // SAFETY: `kill(2)` has no preconditions; the group is the child's own, so `-pid` names
        // its tree and nothing else.
        let killed = unsafe { libc::kill(-(pid as libc::pid_t), libc::SIGKILL) };
        if killed != 0 {
            let error = std::io::Error::last_os_error();
            tracing::debug!("failed to kill the command's process group {pid}: {error}");
        }
    }
    #[cfg(not(unix))]
    tracing::trace!(
        "no process group to signal for {pid:?}; the child's job is ended by its holder"
    );
}

/// The modes the screen takes beyond raw mode and the alternate screen, which ratatui gives back
/// itself. Process-wide, because the panic hook has to give them back too: a panic that left the
/// mouse captured would type every movement of it into the shell.
static MOUSE_TAKEN: AtomicBool = AtomicBool::new(false);
static KEYBOARD_ENHANCED: AtomicBool = AtomicBool::new(false);

/// Draw the screen as one frame, inside a synchronized update, so a terminal that honors one shows
/// the whole redraw at once rather than tearing between the rows.
fn draw_screen(screen: &mut Screen, terminal: &mut DefaultTerminal) {
    screen.dirty = false;
    // The frame's slow part, laying the transcript out at the width it is drawn at, is done before
    // the update opens. At a new size ratatui clears the terminal ahead of the frame, inside the
    // update, and a terminal holds an update only so long before it shows what it has, which
    // would be the cleared screen while the layout ran.
    match terminal.size() {
        Ok(size) => screen
            .viewport
            .lay_out_all(&screen.launch.transcript, usize::from(size.width)),
        Err(error) => tracing::debug!("failed to read the terminal's size: {error}"),
    }
    let mut out = std::io::stdout();
    let synchronized =
        crossterm::execute!(out, crossterm::terminal::BeginSynchronizedUpdate).is_ok();
    let drawn = terminal.draw(|frame| screen.draw(frame));
    if let Ok(completed) = &drawn {
        screen.remember_drawn(completed.buffer);
    }
    if synchronized
        && let Err(error) = crossterm::execute!(out, crossterm::terminal::EndSynchronizedUpdate)
    {
        tracing::debug!("failed to end the synchronized update: {error}");
    }
    if let Err(error) = drawn {
        tracing::error!("failed to draw the screen: {error}");
        screen.hang_up();
    }
    // After the draw, so the page asked for is the one the frame on screen is next to, not the
    // one a stale layout would have named.
    screen.pump_history();
}

/// Give back what [`enter_terminal`] took beyond what ratatui restores. Safe to call twice: a
/// mode is given back once.
fn give_back_terminal_modes() {
    let mut out = std::io::stdout();
    if KEYBOARD_ENHANCED.swap(false, Ordering::SeqCst)
        && let Err(error) = crossterm::execute!(out, crossterm::event::PopKeyboardEnhancementFlags)
    {
        tracing::debug!("failed to disable the keyboard enhancement: {error}");
    }
    if MOUSE_TAKEN.swap(false, Ordering::SeqCst)
        && let Err(error) = crossterm::execute!(out, crossterm::event::DisableMouseCapture)
    {
        tracing::debug!("failed to disable mouse capture: {error}");
    }
    if let Err(error) = crossterm::execute!(out, crossterm::event::DisableBracketedPaste) {
        tracing::debug!("failed to disable bracketed paste: {error}");
    }
}

/// Take the terminal: raw mode, the alternate screen, a panic hook that gives it all back, and
/// the mouse and paste modes the screen reads.
fn enter_terminal(mouse: bool) -> std::io::Result<std::mem::ManuallyDrop<DefaultTerminal>> {
    // ratatui's own `init` is not used: it installs a panic hook that reports a failed restore
    // with a macro that panics when stderr is gone, as it is after a hangup, which turns the
    // leave into an abort. The three steps it takes are taken here, with a hook of this one's.
    crossterm::terminal::enable_raw_mode()?;
    let terminal =
        crossterm::execute!(std::io::stdout(), crossterm::terminal::EnterAlternateScreen).and_then(
            |()| ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(std::io::stdout())),
        );
    let terminal = match terminal {
        Ok(terminal) => terminal,
        // Raw mode is given back before the failure is reported, or the shell this returns to
        // gets no echo and no line editing.
        Err(error) => {
            if let Err(restore) = ratatui::try_restore() {
                tracing::debug!("failed to give the terminal back: {restore}");
            }
            return Err(error);
        }
    };
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        give_back_terminal_modes();
        if let Err(error) = ratatui::try_restore() {
            tracing::debug!("failed to give the terminal back: {error}");
        }
        previous(info);
    }));
    let mut out = std::io::stdout();
    if let Err(error) = crossterm::execute!(out, crossterm::event::EnableBracketedPaste) {
        tracing::debug!("failed to enable bracketed paste: {error}");
    }
    if mouse {
        match crossterm::execute!(out, crossterm::event::EnableMouseCapture) {
            Ok(()) => MOUSE_TAKEN.store(true, Ordering::SeqCst),
            Err(error) => tracing::debug!("failed to enable mouse capture: {error}"),
        }
    }
    // The kitty keyboard protocol tells Shift+Enter from Enter, and a lone Esc from the start of
    // a sequence. Asked for only where the terminal says it speaks it: the legacy codes stay
    // everywhere else, and a terminal that never answers is treated as one that does not.
    match crossterm::terminal::supports_keyboard_enhancement() {
        Ok(true) => match crossterm::execute!(
            out,
            crossterm::event::PushKeyboardEnhancementFlags(
                crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        ) {
            Ok(()) => KEYBOARD_ENHANCED.store(true, Ordering::SeqCst),
            Err(error) => tracing::debug!("failed to enable the keyboard enhancement: {error}"),
        },
        Ok(false) => {}
        Err(error) => tracing::debug!("failed to ask about the keyboard enhancement: {error}"),
    }
    Ok(std::mem::ManuallyDrop::new(terminal))
}

/// Give the terminal back, in the reverse order it was taken.
fn leave_terminal(mut terminal: std::mem::ManuallyDrop<DefaultTerminal>) {
    give_back_terminal_modes();
    match terminal.show_cursor() {
        // Nothing is left for the terminal's drop to do, so it is let run.
        Ok(()) => drop(std::mem::ManuallyDrop::into_inner(terminal)),
        // The drop would show the cursor again and report the failure with a macro that panics
        // when stderr is gone, as it is after a hangup; a terminal leaked here is a few buffers
        // in a process about to exit.
        Err(error) => tracing::debug!("failed to show the cursor: {error}"),
    }
    if let Err(error) = ratatui::try_restore() {
        tracing::debug!("failed to give the terminal back: {error}");
    }
}

/// Run the screen until the user leaves or the host loop ends.
pub(crate) async fn run_screen(launch: ScreenLaunch) -> anyhow::Result<()> {
    let mouse = launch.mouse;
    let mut screen = Screen::new(launch);
    let mut terminal = enter_terminal(mouse)?;
    // `meka -r` with no session named: the picker is the first thing on the screen.
    if screen.launch.pick_session {
        let project = screen.launch.pick_project.clone();
        screen.open_session_picker(project).await;
    }
    let mut events = crossterm::event::EventStream::new();
    let mut tick = tokio::time::interval(FRAME_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    while !screen.leaving {
        tokio::select! {
            event = events.next() => match event {
                Some(Ok(Event::Key(key))) => screen.handle_key(key, &mut terminal).await,
                Some(Ok(Event::Mouse(mouse))) => screen.handle_mouse(mouse),
                Some(Ok(Event::Paste(text))) => screen.paste(text),
                // Drawn at once rather than at the next tick, while the terminal shows its own
                // idea of the old rows at the new size; but only at the size the window has: a
                // drag sends one event a step, and a step it has moved past is left to the tick.
                Some(Ok(Event::Resize(columns, rows))) => {
                    screen.dirty = true;
                    if crossterm::terminal::size().is_ok_and(|size| size == (columns, rows)) {
                        draw_screen(&mut screen, &mut terminal);
                    }
                }
                Some(Ok(_)) => {}
                Some(Err(error)) => {
                    tracing::error!("failed to read the terminal: {error}");
                    screen.hang_up();
                }
                None => screen.hang_up(),
            },
            event = screen.launch.ui_receiver.recv() => match event {
                Some(event) => screen.handle_ui(event),
                None => screen.leaving = true,
            },
            _ = tick.tick() => {
                screen.drain_logs();
                screen.expire_hint();
                screen.tick_working();
                screen.poll_work().await;
                screen.withdraw_stopped_questions();
                screen.open_pending_question();
                if screen.dirty {
                    draw_screen(&mut screen, &mut terminal);
                }
            }
        }
    }
    leave_terminal(terminal);
    // The log queue goes back to stderr with the screen, as `launch` drops: a turn the exit cut
    // short logs after this, and the screen that would have shown the line is gone.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Within the grace a question takes no answer, but a form's field takes its text: a key
    /// swallowed there is gone, while the Enter that would send it is the one that waits. A
    /// panel's letters are picks, and wait like the Enter. With no question on screen, nothing
    /// is held.
    #[test]
    fn the_grace_holds_a_question_s_answer_and_not_a_form_s_text() {
        let just_opened = Some(Instant::now());
        let held = |focus, action| held_by_question_grace(true, just_opened, focus, action);
        assert!(held(Focus::Question, Action::Submit));
        assert!(!held(Focus::Question, Action::Edit));
        assert!(held(Focus::Panel, Action::Edit));
        assert!(held(Focus::Panel, Action::Submit));
        assert!(!held(Focus::Question, Action::Escape));
        assert!(!held(Focus::Panel, Action::Interrupt));
        assert!(!held_by_question_grace(
            false,
            just_opened,
            Focus::Question,
            Action::Submit
        ));

        let long_ago = Some(Instant::now() - QUESTION_GRACE * 2);
        assert!(!held_by_question_grace(
            true,
            long_ago,
            Focus::Question,
            Action::Submit
        ));
        assert!(!held_by_question_grace(
            true,
            None,
            Focus::Panel,
            Action::Submit
        ));
    }

    /// A paste that is one path to an existing image file names the file, quoted or escaped as
    /// a file manager drops it; words, a missing file or a file that is not an image do not.
    #[test]
    fn a_paste_that_is_one_image_path_names_the_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let png = temp.path().join("shot one.png");
        std::fs::write(&png, b"bytes").expect("write");
        let shown = png.display().to_string();
        assert_eq!(pasted_image_path(&format!("'{shown}'")), Some(png.clone()));
        assert_eq!(
            pasted_image_path(&shown.replace(' ', "\\ ")),
            Some(png.clone())
        );
        assert_eq!(
            pasted_image_path(&format!("{shown}\n")),
            Some(png.clone()),
            "the newline a paste ends in is trimmed"
        );
        assert_eq!(pasted_image_path("some words"), None);
        assert_eq!(
            pasted_image_path(&temp.path().join("missing.png").display().to_string()),
            None
        );
        let notes = temp.path().join("notes.txt");
        std::fs::write(&notes, "x").expect("write");
        assert_eq!(pasted_image_path(&notes.display().to_string()), None);
    }

    /// A picker's fact columns are as wide as their longest values, the directory capped at its
    /// share of the row and the profile at a name's width, and the title has the rest; a row too
    /// narrow for the title's floor drops the directory column, then the profile.
    #[test]
    fn a_picker_s_columns_fit_the_row() {
        let columns = SessionColumns::fit(180, 20, 17, 6);
        assert_eq!(
            (columns.directory, columns.profile, columns.age),
            (20, 17, 6)
        );
        assert_eq!(
            columns.title,
            180 - TEXT_EDGE - 2 - (20 + 2 + 17 + 2 + 6) - 1
        );
        assert_eq!(
            SessionColumns::fit(180, 80, 17, 6).directory,
            45,
            "a long directory has a quarter of the row"
        );
        assert_eq!(
            SessionColumns::fit(180, 20, 40, 6).profile,
            crate::text::NAME_WIDTH
        );
        let narrow = SessionColumns::fit(60, 20, 17, 6);
        assert_eq!(narrow.directory, 0, "too narrow for a directory column");
        assert_eq!(narrow.title, 60 - TEXT_EDGE - 2 - (17 + 2 + 6) - 1);
        let narrower = SessionColumns::fit(30, 20, 17, 6);
        assert_eq!((narrower.directory, narrower.profile), (0, 0));
        assert_eq!(narrower.title, 30 - TEXT_EDGE - 2 - 6 - 1);
        assert_eq!(
            columns.facts(
                "~/.config/config/usr/local/bin/i3blocks",
                "claude-opus-5-5",
                "1w ago"
            ),
            "…/local/bin/i3blocks  claude-opus-5-5    1w ago"
        );
        assert_eq!(
            narrow.facts("~/.config/config", "claude-opus-5-5", "1w ago"),
            "claude-opus-5-5    1w ago",
            "a dropped column leaves no blank behind"
        );
        assert_eq!(
            columns.facts("", "default", "just now"),
            format!("{:20}  {:17}  just now", "", "default"),
            "a session without a directory keeps its column blank"
        );
    }

    /// A selection copies the rows between its ends, the end rows cut at their columns, without
    /// the blanks a row was padded with.
    #[test]
    fn a_selection_is_the_text_between_its_ends() {
        let grid = |rows: &[&str]| -> Vec<Vec<String>> {
            rows.iter()
                .map(|row| row.chars().map(String::from).collect())
                .collect()
        };
        let cells = grid(&["> hi      ", "          ", "  First   ", "  answer. "]);
        assert_eq!(text_between(&cells, (2, 2), (2, 6)), "First");
        assert_eq!(text_between(&cells, (0, 2), (3, 3)), "hi\n\n  First\n  an");
        assert_eq!(
            text_between(&cells, (2, 4), (2, 40)),
            "rst",
            "a column past the row's end"
        );
    }

    /// Clipboard pixels become a PNG the image door recognizes.
    #[test]
    fn clipboard_pixels_become_a_png() {
        let png = png_from_rgba(2, 1, vec![255, 0, 0, 255, 0, 0, 255, 255]).expect("a png");
        assert!(matches!(
            crate::image::classify_bytes(&png),
            crate::image::ImageHandling::PassThrough(image::ImageFormat::Png)
        ));
        assert!(
            png_from_rgba(2, 2, vec![0; 4]).is_err(),
            "a size its pixels do not fill"
        );
    }

    fn line(level: tracing::Level, message: &str) -> crate::relay::LogLine {
        crate::relay::LogLine {
            at: chrono::Utc::now(),
            level,
            target: "meka::agent::turn".to_string(),
            message: message.to_string(),
            shown: true,
        }
    }

    /// An error is an error row, a warning a warning, and anything quieter a dimmed row that
    /// names the module, since those only show under `-v`.
    #[test]
    fn a_log_line_becomes_the_row_its_level_asks_for() {
        assert_eq!(
            log_item(line(tracing::Level::ERROR, "stream error")),
            Item::Error("stream error".to_string())
        );
        assert_eq!(
            log_item(line(tracing::Level::WARN, "retrying")),
            Item::Notice {
                level: crate::frontend::NoticeLevel::Warn,
                text: "retrying".to_string(),
            }
        );
        assert_eq!(
            log_item(line(tracing::Level::INFO, "turn started")),
            Item::note("meka::agent::turn: turn started")
        );
    }

    /// A command that prints without end cannot grow the screen with it: each stream keeps its
    /// last bytes and counts what it let go.
    #[tokio::test]
    async fn a_command_s_output_keeps_its_tail() {
        let (mut writer, reader) = tokio::io::duplex(64 * 1024);
        let writing = tokio::spawn(async move {
            use tokio::io::AsyncWriteExt as _;
            for byte in *b"abc" {
                writer
                    .write_all(&vec![byte; SHELL_OUTPUT_KEEP_BYTES])
                    .await
                    .expect("write");
            }
        });
        let (kept, dropped) = read_tail(Some(reader)).await;
        writing.await.expect("the writer");
        assert_eq!(kept.len(), SHELL_OUTPUT_KEEP_BYTES);
        assert_eq!(dropped, 2 * SHELL_OUTPUT_KEEP_BYTES as u64);
        assert!(kept.iter().all(|&byte| byte == b'c'));
    }

    /// Lines past the cap go from the front, whole lines only, trailing newline or not.
    #[test]
    fn a_command_s_output_keeps_its_last_lines() {
        let mut text = "a\nb\nc\n".to_string();
        assert!(keep_last_lines(&mut text, 2));
        assert_eq!(text, "b\nc\n");
        let mut text = "a\nb\nc".to_string();
        assert!(keep_last_lines(&mut text, 2));
        assert_eq!(text, "b\nc");
        let mut text = "a\nb".to_string();
        assert!(!keep_last_lines(&mut text, 5));
        assert_eq!(text, "a\nb");
    }
}
