//! The model's reading of the store's conversations: `conversation_search` / `conversation_read`
//! over the full log of one or more sessions, and `session_list` / `session_search` over the
//! sessions themselves.
//!
//! The conversation tools read a session's whole log, including turns that compaction summarized
//! away and removed from the model's context. Compaction never deletes (it appends a boundary,
//! [`crate::conversation`]), so every turn is still on disk in the `messages` table; these tools
//! give the model a way back to detail the summary may have dropped. Which sessions a call may
//! reach is one rule, [`readable_sessions`]: the caller's own and its direct children, or every
//! session at `unrestricted`.
//!
//! A rewind is the one thing that does remove turns from this view. That is deliberate: `/rewind`
//! exists to take a turn back, and a search that could hand it straight back to the model would
//! defeat it. Those turns remain on disk and in `meka session export`.

use async_trait::async_trait;
use uuid::Uuid;

use super::{
    Tool, ToolOutput,
    util::{MAX_SEARCH_MATCHES, compile_user_regex, require_str, resolve_session_id},
};
use crate::{
    conversation::{ContentBlock, Event, Message, Role},
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
    store::{SessionFilter, SessionReach, Store},
};

/// Default number of matches `conversation_search` returns when the caller doesn't set `limit`.
const DEFAULT_CONVERSATION_SEARCH_LIMIT: usize = 20;
/// Hard cap on how many messages `conversation_read` returns in one call.
const MAX_CONVERSATION_READ_MESSAGES: usize = 20;
/// Each `conversation_search` match line is truncated to this many characters so the result stays
/// compact.
const SNIPPET_CHARS: usize = 200;
/// Rows `session_list` shows at most. What is older is `session_search`'s to find.
const SESSION_LIST_ROWS: u32 = 50;
/// Characters of a title a listing shows the model: enough to recognize a conversation by.
const TITLE_EXCERPT_CHARS: usize = 80;
/// Default number of sessions `session_search` returns when the caller doesn't set `limit`.
const DEFAULT_SESSION_SEARCH_LIMIT: usize = 10;

/// How many sessions one `conversation_search` call may name. Each is a whole log loaded and
/// scanned; the store-wide question belongs to `session_search`'s index.
const MAX_SEARCHED_SESSIONS: usize = 20;

/// The sessions a call from `site` may read, by the one rule every session-reading tool applies:
/// the caller's own and its direct children, or every session at `unrestricted`, the level defined
/// as no boundary and at which the shell reaches the store anyway. Read live, so a sub-agent's
/// reach moves with its parent's level. `include_children` is the store-wide switch and moot
/// within a tree, whose children are the point.
fn readable_sessions(
    site: &crate::session::ToolSite,
    include_children: bool,
    tool_name: &str,
) -> Result<SessionReach> {
    let own = resolve_session_id(&site.session_id, tool_name)?;
    // The level itself, not `allows`: `workspace` admits a tool that requires `unrestricted` to
    // the gate, and this is a boundary question, which only the top level disclaims.
    Ok(if site.permission.get() == Permission::Unrestricted {
        SessionReach::Store { include_children }
    } else {
        SessionReach::Tree(own)
    })
}

/// What a model passed as a session, resolved within `reach`: a full id, or a prefix exactly one
/// session there starts with.
///
/// Within a tree an id that is not there gets one answer whether or not it exists, so a probe
/// learns nothing about the sessions it may not read.
async fn resolve_readable_session(
    store: &Store,
    reach: SessionReach,
    wanted: &str,
    tool_name: &str,
) -> Result<Uuid> {
    let wanted = wanted.trim();
    let root = match reach {
        SessionReach::Store { .. } => {
            return store.resolve_session_id(wanted).await.map_err(|error| {
                MekaError::ToolExecution {
                    tool_name: tool_name.to_string(),
                    message: error.to_string(),
                }
            });
        }
        SessionReach::Tree(root) => root,
    };
    let candidates: Vec<Uuid> = store
        .load_session_tree(root)
        .await?
        .into_iter()
        .filter(|row| row.id == root || row.parent_id == Some(root))
        .map(|row| row.id)
        .collect();
    let matches: Vec<Uuid> = if let Ok(id) = Uuid::parse_str(wanted) {
        candidates
            .into_iter()
            .filter(|candidate| *candidate == id)
            .collect()
    } else if crate::text::is_usable_id_prefix(wanted) {
        let prefix = crate::text::id_prefix_for_matching(wanted);
        candidates
            .into_iter()
            .filter(|candidate| candidate.to_string().starts_with(&prefix))
            .collect()
    } else {
        Vec::new()
    };
    match matches.as_slice() {
        [one] => Ok(*one),
        [] => Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!(
                "no readable session '{wanted}': your own and your sub-agents' are readable, and \
                 any other at `unrestricted`"
            ),
        }),
        many => Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!(
                "'{wanted}' names {} readable sessions: {}. Pass more of the id.",
                many.len(),
                many.iter()
                    .map(Uuid::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }),
    }
}

fn role_label(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// The `Append` messages in log order. `CompactBoundary` events are skipped: their synthetic
/// summary is already in the model's context, and skipping them keeps message indices stable across
/// compactions (the log only ever grows).
fn append_messages(events: &[Event]) -> Vec<&Message> {
    let mut messages: Vec<&Message> = Vec::new();
    for event in events {
        match event {
            Event::Append(message) => messages.push(message),
            Event::CompactBoundary { .. } => {}
            // Skipped: it changes an image into a note, and images are not searchable either way.
            Event::Redact { .. } => {}
            // Applied rather than skipped: a repaired message is what the model actually saw, so
            // searching the superseded original would surface content that is no longer in the
            // conversation. Unlike a boundary this can shift indices, but only across a repair,
            // which is rare and confined to a single run of trailing messages.
            Event::Repair {
                replaced_count,
                messages: replacements,
            } => {
                let truncate_to = messages.len().saturating_sub(*replaced_count);
                messages.truncate(truncate_to);
                messages.extend(replacements.iter());
            }
        }
    }
    messages
}

/// Flatten a message into searchable lines: every block's [`ContentBlock::search_text`], which
/// labels the non-text ones so a match is identifiable and skips what holds no words.
fn searchable_text(message: &Message) -> String {
    message
        .content
        .iter()
        .filter_map(ContentBlock::search_text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Full, untruncated rendering of a message for `conversation_read`.
fn render_message_full(message: &Message) -> String {
    let mut segments: Vec<String> = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text } => segments.push(text.clone()),
            // meka's own per-turn preamble, restated every turn; not part of the conversation.
            ContentBlock::TurnContext { .. } => {}
            // What the model was sent back to work with, and what arrived for it, as it read them.
            ContentBlock::Nudge { text, .. }
            | ContentBlock::Inbox { text, .. }
            | ContentBlock::Scheduled { text, .. }
            | ContentBlock::BackgroundReport { text, .. } => segments.push(text.clone()),
            ContentBlock::Thinking { thinking, .. } => {
                segments.push(format!("[thinking]\n{thinking}"))
            }
            ContentBlock::RedactedThinking { .. } => {
                segments.push("[redacted thinking]".to_string())
            }
            ContentBlock::ToolUse { name, input, .. } => {
                let pretty =
                    serde_json::to_string_pretty(input).unwrap_or_else(|_| input.to_string());
                segments.push(format!("[tool call: {name}]\n{pretty}"));
            }
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                let label = if *is_error {
                    "[tool result (error)]"
                } else {
                    "[tool result]"
                };
                segments.push(format!(
                    "{label}\n{}",
                    ContentBlock::tool_result_text_content(content)
                ));
            }
            ContentBlock::Image { .. } => segments.push("[image]".to_string()),
        }
    }
    segments.join("\n")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        let head: String = text.chars().take(max_chars).collect();
        format!("{head}…")
    } else {
        text.to_string()
    }
}

/// Literal-substring (case-insensitive) or regex matcher for one query.
enum Matcher {
    Substring(String),
    Regex(regex::Regex),
}

impl Matcher {
    fn build(query: &str, use_regex: bool) -> Result<Self> {
        if use_regex {
            Ok(Matcher::Regex(compile_user_regex(
                query,
                "conversation_search",
            )?))
        } else {
            Ok(Matcher::Substring(query.to_lowercase()))
        }
    }

    fn is_match(&self, line: &str) -> bool {
        match self {
            Matcher::Substring(needle) => line.to_lowercase().contains(needle.as_str()),
            Matcher::Regex(regex) => regex.is_match(line),
        }
    }
}

/// What a scan of one log found: the lines shown, how many matched in all, the index of the first
/// match, and how many messages were scanned.
struct Found {
    shown: Vec<String>,
    total: usize,
    first_index: Option<usize>,
    messages: usize,
}

/// Scan one log for `matcher`, showing at most `room` lines.
fn find_in(events: &[Event], matcher: &Matcher, room: usize) -> Found {
    let messages = append_messages(events);
    let mut found = Found {
        shown: Vec::new(),
        total: 0,
        first_index: None,
        messages: messages.len(),
    };
    for (offset, message) in messages.iter().enumerate() {
        let index = offset + 1;
        let label = role_label(&message.role);
        let text = searchable_text(message);
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || !matcher.is_match(line) {
                continue;
            }
            found.total += 1;
            found.first_index.get_or_insert(index);
            if found.shown.len() < room {
                found.shown.push(format!(
                    "#{index} [{label}] {}",
                    truncate_chars(line, SNIPPET_CHARS)
                ));
            }
        }
    }
    found
}

fn search_events(events: &[Event], query: &str, use_regex: bool, limit: usize) -> Result<String> {
    use std::fmt::Write;

    let matcher = Matcher::build(query, use_regex)?;
    let cap = limit.clamp(1, MAX_SEARCH_MATCHES);
    let found = find_in(events, &matcher, cap);
    let total_messages = found.messages;

    if found.total == 0 {
        return Ok(format!(
            "No matches for {query:?} in {total_messages} message(s)."
        ));
    }

    let mut output = String::new();
    if found.total > found.shown.len() {
        writeln!(
            output,
            "{} matches for {query:?} across {total_messages} message(s) (showing first {}):\n",
            found.total,
            found.shown.len()
        )
        .ok();
    } else {
        writeln!(
            output,
            "{} match(es) for {query:?} across {total_messages} message(s):\n",
            found.shown.len()
        )
        .ok();
    }
    for line in &found.shown {
        writeln!(output, "{line}").ok();
    }
    if let Some(index) = found.first_index {
        writeln!(
            output,
            "\nRead a full turn with `conversation_read` (e.g. {{\"start\": {index}}})."
        )
        .ok();
    }
    Ok(output)
}

/// [`search_events`] over the logs a call named, one group per session that holds a match, under
/// one shared cap. The hint names the session beside the index, because an index means something
/// only within its log.
fn search_named(
    logs: &[(Uuid, Vec<Event>)],
    query: &str,
    use_regex: bool,
    limit: usize,
) -> Result<String> {
    use std::fmt::Write;

    let matcher = Matcher::build(query, use_regex)?;
    let cap = limit.clamp(1, MAX_SEARCH_MATCHES);
    let mut groups: Vec<(Uuid, Found)> = Vec::new();
    let mut total_found = 0usize;
    let mut shown = 0usize;
    let mut total_messages = 0usize;
    for (session, events) in logs {
        let found = find_in(events, &matcher, cap - shown);
        total_messages += found.messages;
        total_found += found.total;
        shown += found.shown.len();
        // A log whose matches all fell outside the shared cap is counted, not listed: an empty
        // group would read as a session with nothing in it.
        if !found.shown.is_empty() {
            groups.push((*session, found));
        }
    }

    if total_found == 0 {
        return Ok(format!(
            "No matches for {query:?} in {} session(s) ({total_messages} message(s)).",
            logs.len()
        ));
    }

    let mut output = String::new();
    if total_found > shown {
        writeln!(
            output,
            "{total_found} matches for {query:?} across {} session(s) (showing first {shown}):",
            logs.len()
        )
        .ok();
    } else {
        writeln!(
            output,
            "{shown} match(es) for {query:?} across {} session(s):",
            logs.len()
        )
        .ok();
    }
    for (session, found) in &groups {
        writeln!(
            output,
            "\nSession {session} ({} message(s)):",
            found.messages
        )
        .ok();
        for line in &found.shown {
            writeln!(output, "{line}").ok();
        }
    }
    if let Some((session, Some(index))) = groups
        .first()
        .map(|(session, found)| (session, found.first_index))
    {
        writeln!(
            output,
            "\nRead a full turn with `conversation_read` (e.g. {{\"session\": \"{session}\", \
             \"start\": {index}}})."
        )
        .ok();
    }
    Ok(output)
}

/// `count` messages from `start`, headed by which session they are from when the call named one.
fn read_messages(
    events: &[Event],
    start: usize,
    count: usize,
    session: Option<Uuid>,
) -> Result<String> {
    use std::fmt::Write;

    let messages = append_messages(events);
    let total = messages.len();
    if start == 0 || start > total {
        return Err(MekaError::ToolExecution {
            tool_name: "conversation_read".to_string(),
            message: format!("start {start} is out of range (valid: 1..={total})"),
        });
    }
    let count = count.clamp(1, MAX_CONVERSATION_READ_MESSAGES);
    let end = (start - 1 + count).min(total);

    let mut output = String::new();
    match session {
        Some(session) => writeln!(
            output,
            "Messages #{start}..#{end} of {total} in session {session}:\n"
        )
        .ok(),
        None => writeln!(output, "Messages #{start}..#{end} of {total}:\n").ok(),
    };
    for (position, message) in messages
        .iter()
        .enumerate()
        .skip(start - 1)
        .take(end - start + 1)
    {
        let index = position + 1;
        writeln!(output, "#{index} [{}]", role_label(&message.role)).ok();
        output.push_str(&render_message_full(message));
        output.push_str("\n\n");
    }
    Ok(output)
}

/// The session ids a `conversation_search` call named: a list, or one bare string the way
/// `memory_search` takes a bare phrasing. `None` when it named nothing, which means the caller's
/// own session.
fn named_sessions(input: &serde_json::Value) -> Result<Option<Vec<String>>> {
    let ids: Vec<String> = match input.get("sessions") {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::String(one)) => vec![one.clone()],
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| MekaError::ToolExecution {
                        tool_name: "conversation_search".to_string(),
                        message: format!("'sessions' holds {value}, which is not a session id"),
                    })
            })
            .collect::<Result<_>>()?,
        Some(other) => {
            return Err(MekaError::ToolExecution {
                tool_name: "conversation_search".to_string(),
                message: format!("'sessions' must be a list of session ids, got {other}"),
            });
        }
    };
    if ids.len() > MAX_SEARCHED_SESSIONS {
        return Err(MekaError::ToolExecution {
            tool_name: "conversation_search".to_string(),
            message: format!(
                "'sessions' names {} sessions; at most {MAX_SEARCHED_SESSIONS} per call",
                ids.len()
            ),
        });
    }
    Ok((!ids.is_empty()).then_some(ids))
}

pub(super) struct ConversationSearchTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
}

#[async_trait]
impl Tool for ConversationSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "conversation_search".to_string(),
            description: format!(
                "Search the full conversation history of this session, or of the sessions \
                 `sessions` names, including earlier turns that compaction summarized away and \
                 removed from your context. Returns matching lines, each tagged with its message \
                 index (#N) and role; follow up with `conversation_read` to read a full turn. Use \
                 this to recover a detail the compaction summary may have omitted, or to read \
                 what a sub-agent found. Substring matching is case-insensitive; set `is_regex: \
                 true` to match `query` as a case-sensitive regular expression. Large tool \
                 outputs appear as <large-output> references here; read their full content with \
                 `scratchpad_read`. Returns at most {MAX_SEARCH_MATCHES} matches."
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for (a literal substring unless `is_regex` is true)."
                    },
                    "sessions": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": format!("Session ids to search instead of this session. At most {MAX_SEARCHED_SESSIONS}.")
                    },
                    "is_regex": {
                        "type": "boolean",
                        "default": false,
                        "description": "Treat `query` as a case-sensitive regular expression instead of a case-insensitive literal substring. Default: false."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_MATCHES,
                        "default": DEFAULT_CONVERSATION_SEARCH_LIMIT,
                        "description": format!("Maximum matches to return (max {MAX_SEARCH_MATCHES}). Default: {DEFAULT_CONVERSATION_SEARCH_LIMIT}.")
                    }
                },
                "required": ["query"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let query = require_str(&input, "query", "conversation_search")?;
        let use_regex = input
            .get("is_regex")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let limit = input
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_CONVERSATION_SEARCH_LIMIT);

        let Some(named) = named_sessions(&input)? else {
            let session_id = resolve_session_id(&self.site.session_id, "conversation_search")?;
            let events = self.store.load_events(session_id).await?;
            return Ok(ToolOutput::text(
                search_events(&events, &query, use_regex, limit)?,
                false,
            ));
        };
        // Every id is resolved before any log is loaded, so one the caller may not read refuses
        // the call with nothing scanned.
        let reach = readable_sessions(&self.site, true, "conversation_search")?;
        let mut sessions: Vec<Uuid> = Vec::new();
        for wanted in &named {
            let id =
                resolve_readable_session(&self.store, reach, wanted, "conversation_search").await?;
            if !sessions.contains(&id) {
                sessions.push(id);
            }
        }
        let mut logs: Vec<(Uuid, Vec<Event>)> = Vec::with_capacity(sessions.len());
        for session in sessions {
            logs.push((session, self.store.load_events(session).await?));
        }
        Ok(ToolOutput::text(
            search_named(&logs, &query, use_regex, limit)?,
            false,
        ))
    }
}

pub(super) struct ConversationReadTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
}

#[async_trait]
impl Tool for ConversationReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "conversation_read".to_string(),
            description: format!(
                "Read the full content of conversation turns by message index, including turns \
                 compaction removed from your context. Use the #N indices reported by `conversation_search`. \
                 Reads `limit` consecutive messages starting at `start`, at most \
                 {MAX_CONVERSATION_READ_MESSAGES} per call."
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "start": {
                        "type": "integer",
                        "description": "1-based message index to start reading from (the #N from `conversation_search`)."
                    },
                    "session": {
                        "type": "string",
                        "description": "Session id to read instead of this session."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_CONVERSATION_READ_MESSAGES,
                        "default": 1,
                        "description": format!("Number of consecutive messages to read (max {MAX_CONVERSATION_READ_MESSAGES}). Default: 1.")
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["start"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let start = input
            .get("start")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "conversation_read".to_string(),
                message: "missing or invalid 'start' parameter".to_string(),
            })?;
        let limit = input
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(1);

        let named = match input.get("session").and_then(serde_json::Value::as_str) {
            Some(wanted) if !wanted.trim().is_empty() => {
                let reach = readable_sessions(&self.site, true, "conversation_read")?;
                Some(
                    resolve_readable_session(&self.store, reach, wanted, "conversation_read")
                        .await?,
                )
            }
            _ => None,
        };
        let session_id = match named {
            Some(session) => session,
            None => resolve_session_id(&self.site.session_id, "conversation_read")?,
        };
        let events = self.store.load_events(session_id).await?;
        Ok(ToolOutput::text(
            read_messages(&events, start, limit, named)?,
            false,
        ))
    }
}

/// The parent a listing shows for a row: a sub-agent's parent id, or `-` for a root session.
fn parent_cell(session: &crate::store::SessionSummary) -> String {
    session
        .parent_id
        .map(|parent| parent.to_string())
        .unwrap_or_else(|| "-".to_string())
}

pub(super) struct SessionListTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
}

#[async_trait]
impl Tool for SessionListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "session_list".to_string(),
            description: "List sessions, newest first, with each one's id, last update, parent \
                          for a sub-agent, and title. Pass an id to `conversation_search` or \
                          `conversation_read`; `session_search` finds a session by its words."
                .to_string(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let own = resolve_session_id(&self.site.session_id, "session_list")?;
        let (sessions, more) = match readable_sessions(&self.site, false, "session_list")? {
            SessionReach::Tree(root) => {
                // The caller first, then its children in the listing's own order.
                let mut sessions: Vec<crate::store::SessionSummary> =
                    self.store.session_info(root).await?.into_iter().collect();
                let (children, more) = self
                    .store
                    .list_sessions(SESSION_LIST_ROWS, None, &SessionFilter {
                        parent: Some(root),
                        ..SessionFilter::default()
                    })
                    .await?;
                sessions.extend(children);
                (sessions, more)
            }
            SessionReach::Store { .. } => {
                self.store
                    .list_sessions(SESSION_LIST_ROWS, None, &SessionFilter::default())
                    .await?
            }
        };
        let rows: Vec<Vec<String>> = sessions
            .iter()
            .map(|session| {
                vec![
                    session.id.to_string(),
                    session.updated_at.clone(),
                    parent_cell(session),
                    truncate_chars(
                        &crate::text::prose_cell(&session.title),
                        TITLE_EXCERPT_CHARS,
                    ),
                ]
            })
            .collect();
        let mut rendered = if rows.is_empty() {
            "No sessions.".to_string()
        } else {
            crate::text::format_columns(&["ID", "Updated", "Parent", "Title"], &rows)
        };
        if more.is_some() {
            rendered.push_str(&format!(
                "\nOnly the {SESSION_LIST_ROWS} most recent are listed; `session_search` reaches \
                 the rest."
            ));
        }
        // The one place the model learns its own id, which the rows above are read against.
        rendered.push_str(&format!("\nThis session: {own}"));
        Ok(ToolOutput::text(rendered, false))
    }
}

pub(super) struct SessionSearchTool {
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
}

#[async_trait]
impl Tool for SessionSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "session_search".to_string(),
            description: "Find sessions by what was said in them. Matches user and assistant \
                          text, not tool calls or results; a session whose title holds every word \
                          is listed first, the rest by their best match. Each row carries the id, \
                          the last update, the title, and the line it was found by. Pass an id to \
                          `conversation_search` for the exact lines."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Words to find. Every word must appear, as a stem or a prefix."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_MATCHES,
                        "default": DEFAULT_SESSION_SEARCH_LIMIT,
                        "description": format!("Maximum sessions to return (max {MAX_SEARCH_MATCHES}). Default: {DEFAULT_SESSION_SEARCH_LIMIT}.")
                    },
                    "include_children": {
                        "type": "boolean",
                        "default": false,
                        "description": "Search sub-agents' sessions too. Default: false."
                    }
                },
                "required": ["query"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let query = require_str(&input, "query", "session_search")?;
        let limit = input
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_SESSION_SEARCH_LIMIT)
            .clamp(1, MAX_SEARCH_MATCHES);
        let include_children = input
            .get("include_children")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let reach = readable_sessions(&self.site, include_children, "session_search")?;
        let found = self.store.search_sessions(&query, limit, reach).await?;
        if found.is_empty() {
            return Ok(ToolOutput::text(
                format!("No sessions match {query:?}."),
                false,
            ));
        }
        let rows: Vec<Vec<String>> = found
            .iter()
            .map(|found| {
                vec![
                    found.session.id.to_string(),
                    found.session.updated_at.clone(),
                    truncate_chars(
                        &crate::text::prose_cell(&found.session.title),
                        crate::text::NAME_WIDTH,
                    ),
                    found
                        .excerpt
                        .clone()
                        .unwrap_or_else(|| "(title)".to_string()),
                ]
            })
            .collect();
        Ok(ToolOutput::text(
            crate::text::format_columns(&["ID", "Updated", "Title", "Match"], &rows),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn boundary(replaced: usize) -> Event {
        Event::CompactBoundary {
            summary: Message::user("[summary]"),
            replaced_count: replaced,
            record: crate::conversation::CompactionRecord::for_test(),
            loaded_tools_snapshot: HashSet::new(),
            checklist_snapshot: Default::default(),
        }
    }

    /// A log where the first two turns were compacted away (a boundary sits after them) and one
    /// turn follows. `conversation_search` must still see the pre-boundary turns.
    fn compacted_log() -> Vec<Event> {
        vec![
            Event::Append(Message::user("the auth token expired unexpectedly")),
            Event::Append(Message::assistant_text(
                "I'll investigate the token refresh path",
            )),
            boundary(2),
            Event::Append(Message::user("now let's talk about something unrelated")),
        ]
    }

    #[test]
    fn search_finds_pre_compaction_turn_case_insensitively() {
        let events = compacted_log();
        let out = search_events(&events, "AUTH Token", false, 20).expect("search");
        // Message #1 is before the boundary (hidden from the model) but still searchable.
        assert!(out.contains("#1 [user]"), "expected a hit on #1: {out}");
        assert!(out.contains("auth token expired"), "{out}");
    }

    #[test]
    fn search_regex_mode_matches() {
        let events = compacted_log();
        let out = search_events(&events, "tok.n refresh", true, 20).expect("search");
        assert!(out.contains("#2 [assistant]"), "{out}");
    }

    #[test]
    fn search_no_match_reports_zero() {
        let events = compacted_log();
        let out = search_events(&events, "nonexistent phrase", false, 20).expect("search");
        assert!(out.starts_with("No matches"), "{out}");
    }

    #[test]
    fn boundary_summary_is_not_indexed() {
        // The synthetic summary message must not get an index; #3 is the post-boundary user turn.
        let events = compacted_log();
        let out = search_events(&events, "unrelated", false, 20).expect("search");
        assert!(out.contains("#3 [user]"), "{out}");
    }

    #[test]
    fn read_messages_returns_full_turns_by_index() {
        let events = compacted_log();
        let out = read_messages(&events, 1, 2, None).expect("read");
        assert!(
            out.contains("#1 [user]") && out.contains("auth token expired"),
            "{out}"
        );
        assert!(
            out.contains("#2 [assistant]") && out.contains("token refresh path"),
            "{out}"
        );
    }

    #[test]
    fn read_messages_out_of_range_errors() {
        let events = compacted_log();
        assert!(read_messages(&events, 0, 1, None).is_err());
        assert!(read_messages(&events, 99, 1, None).is_err());
    }

    /// A parent, its child, the child's child, and an unrelated root, each saying the same word so
    /// a search tells reach apart and nothing else.
    struct Family {
        store: Store,
        parent: Uuid,
        child: Uuid,
        grandchild: Uuid,
        stranger: Uuid,
    }

    async fn family() -> Family {
        let store = Store::for_test().await;
        let parent = store
            .create_session(None, "test-profile")
            .await
            .expect("parent");
        let spawn = |parent: Uuid| {
            let store = store.clone();
            async move {
                let (id, _lock) = store
                    .create_child_session_for_test(
                        parent,
                        None,
                        Vec::new(),
                        None,
                        "read".to_string(),
                        "test-profile".to_string(),
                    )
                    .await
                    .expect("child");
                id
            }
        };
        let child = spawn(parent).await;
        let grandchild = spawn(child).await;
        let stranger = store
            .create_session(None, "test-profile")
            .await
            .expect("stranger");
        for (id, line) in [
            (parent, "the parent planned the pelicans survey"),
            (child, "the child counted forty pelicans"),
            (grandchild, "the grandchild saw one of the pelicans"),
            (stranger, "a stranger also likes pelicans"),
        ] {
            store
                .save_event(id, &Event::Append(Message::user(line)), None)
                .await
                .expect("save");
        }
        Family {
            store,
            parent,
            child,
            grandchild,
            stranger,
        }
    }

    static UUID_LINE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
            .expect("a fixed pattern")
    });

    fn site(permission: Permission, session: Uuid) -> crate::session::ToolSite {
        crate::session::ToolSite::for_test()
            .with_permission(crate::permission::SharedPermission::new(
                permission,
                crate::permission::EnabledPermissions::ALL,
            ))
            .with_session_id(crate::session::SharedSessionId::new(Some(session)))
    }

    async fn run(tool: &dyn Tool, input: serde_json::Value) -> Result<String> {
        let output = tool
            .execute(
                input,
                crate::tools::ToolContext::detached(tokio_util::sync::CancellationToken::new()),
            )
            .await?;
        Ok(output
            .content
            .iter()
            .filter_map(|block| match block {
                crate::conversation::ToolResultContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"))
    }

    #[test]
    fn a_search_over_named_logs_groups_by_session_and_names_it_in_the_hint() {
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let logs = vec![
            (first, vec![Event::Append(Message::user(
                "pelicans at dawn",
            ))]),
            (second, vec![
                Event::Append(Message::user("nothing here")),
                Event::Append(Message::assistant_text("pelicans at dusk")),
            ]),
        ];
        let out = search_named(&logs, "pelicans", false, 20).expect("search");
        assert!(
            out.contains(&format!(
                "Session {first} (1 message(s)):\n#1 [user] pelicans at dawn"
            )),
            "{out}"
        );
        assert!(
            out.contains(&format!(
                "Session {second} (2 message(s)):\n#2 [assistant] pelicans at dusk"
            )),
            "{out}"
        );
        assert!(
            out.contains(&format!("{{\"session\": \"{first}\", \"start\": 1}}")),
            "the hint names the session: {out}"
        );

        // One cap across every log, so the second log shows nothing once the first used it up.
        let out = search_named(&logs, "pelicans", false, 1).expect("search");
        assert!(
            out.starts_with("2 matches for \"pelicans\" across 2 session(s) (showing first 1):"),
            "{out}"
        );
        assert!(!out.contains(&format!("Session {second}")), "{out}");

        let out = search_named(&logs, "walrus", false, 20).expect("search");
        assert_eq!(
            out,
            "No matches for \"walrus\" in 2 session(s) (3 message(s))."
        );
    }

    /// Below `unrestricted` a session reads its own log and its direct children's, by full id or
    /// by prefix, and nothing else gets one answer whether or not it exists.
    #[tokio::test]
    async fn a_parent_reads_its_child_s_log_and_nothing_else_below_unrestricted() {
        let family = family().await;
        let read = ConversationReadTool {
            store: family.store.clone(),
            site: site(Permission::Workspace, family.parent),
        };
        let prefix = family.child.to_string()[..8].to_string();
        let out = run(&read, serde_json::json!({"start": 1, "session": prefix}))
            .await
            .expect("a child by prefix");
        assert!(
            out.contains(&format!("in session {}", family.child)) && out.contains("forty pelicans"),
            "{out}"
        );
        let out = run(
            &read,
            serde_json::json!({"start": 1, "session": family.parent.to_string()}),
        )
        .await
        .expect("itself by id");
        assert!(out.contains("pelicans survey"), "{out}");

        let mut refusals = Vec::new();
        for wanted in [family.grandchild, family.stranger, Uuid::new_v4()] {
            let error = run(
                &read,
                serde_json::json!({"start": 1, "session": wanted.to_string()}),
            )
            .await
            .expect_err("refused")
            .to_string();
            refusals.push(error.replace(&wanted.to_string(), "<id>"));
        }
        assert!(
            refusals[0].contains("no readable session '<id>'")
                && refusals[0].contains("`unrestricted`"),
            "{refusals:?}"
        );
        assert!(
            refusals.iter().all(|refusal| refusal == &refusals[0]),
            "one text whether or not the session exists: {refusals:?}"
        );
    }

    #[tokio::test]
    async fn at_unrestricted_every_session_is_readable() {
        let family = family().await;
        let read = ConversationReadTool {
            store: family.store.clone(),
            site: site(Permission::Unrestricted, family.parent),
        };
        for (session, words) in [
            (family.stranger, "stranger also likes"),
            (family.grandchild, "grandchild saw"),
        ] {
            let out = run(
                &read,
                serde_json::json!({"start": 1, "session": session.to_string()}),
            )
            .await
            .expect("readable");
            assert!(out.contains(words), "{out}");
        }
        // A sub-agent at the level reads its parent: the level, not the tree, decides.
        let read = ConversationReadTool {
            store: family.store.clone(),
            site: site(Permission::Unrestricted, family.child),
        };
        let out = run(
            &read,
            serde_json::json!({"start": 1, "session": family.parent.to_string()}),
        )
        .await
        .expect("the parent");
        assert!(out.contains("pelicans survey"), "{out}");
        assert!(
            run(
                &read,
                serde_json::json!({"start": 1, "session": Uuid::new_v4().to_string()}),
            )
            .await
            .is_err(),
            "an unknown id is still unknown"
        );
    }

    #[tokio::test]
    async fn a_search_across_named_sessions_is_grouped_and_bounded() {
        let family = family().await;
        let search = ConversationSearchTool {
            store: family.store.clone(),
            site: site(Permission::Read, family.parent),
        };
        let out = run(
            &search,
            serde_json::json!({
                "query": "pelicans",
                "sessions": [family.child.to_string(), family.parent.to_string()],
            }),
        )
        .await
        .expect("search");
        let child_at = out
            .find(&format!("Session {}", family.child))
            .expect("the child's group");
        let parent_at = out
            .find(&format!("Session {}", family.parent))
            .expect("the parent's group");
        assert!(child_at < parent_at, "groups in the order given: {out}");

        let out = run(
            &search,
            serde_json::json!({"query": "pelicans", "sessions": family.child.to_string()}),
        )
        .await
        .expect("a bare string");
        assert!(out.contains("forty pelicans"), "{out}");

        assert!(
            run(
                &search,
                serde_json::json!({
                    "query": "pelicans",
                    "sessions": [family.child.to_string(), family.stranger.to_string()],
                }),
            )
            .await
            .is_err(),
            "one id out of reach refuses the whole call"
        );

        let too_many: Vec<String> = (0..=MAX_SEARCHED_SESSIONS)
            .map(|_| family.child.to_string())
            .collect();
        let error = run(
            &search,
            serde_json::json!({"query": "pelicans", "sessions": too_many}),
        )
        .await
        .expect_err("over the cap")
        .to_string();
        assert!(
            error.contains(&format!("at most {MAX_SEARCHED_SESSIONS}")),
            "{error}"
        );

        let out = run(&search, serde_json::json!({"query": "pelicans"}))
            .await
            .expect("own session");
        assert!(
            out.contains("#1 [user] the parent planned") && !out.contains("Session "),
            "omitted means this session, in the single-log shape: {out}"
        );
    }

    #[tokio::test]
    async fn session_list_shows_the_tree_below_unrestricted_and_the_store_s_roots_at_it() {
        let family = family().await;
        let list = SessionListTool {
            store: family.store.clone(),
            site: site(Permission::Workspace, family.parent),
        };
        let out = run(&list, serde_json::json!({})).await.expect("list");
        let rows: Vec<&str> = out.lines().collect();
        assert!(rows[0].starts_with("ID"), "{out}");
        assert!(
            rows[1].starts_with(&family.parent.to_string()),
            "the caller first: {out}"
        );
        let child_row = rows
            .iter()
            .find(|row| row.starts_with(&family.child.to_string()))
            .expect("the child's row");
        assert!(
            child_row.contains(&family.parent.to_string())
                && child_row.contains("the child counted forty pelicans"),
            "{child_row}"
        );
        assert!(
            !out.contains(&family.grandchild.to_string())
                && !out.contains(&family.stranger.to_string()),
            "{out}"
        );
        assert!(
            out.ends_with(&format!("This session: {}", family.parent)),
            "{out}"
        );

        let list = SessionListTool {
            store: family.store.clone(),
            site: site(Permission::Unrestricted, family.parent),
        };
        let out = run(&list, serde_json::json!({})).await.expect("list");
        assert!(
            out.contains(&family.stranger.to_string())
                && out
                    .lines()
                    .any(|row| row.starts_with(&family.parent.to_string())),
            "{out}"
        );
        assert!(
            !out.lines()
                .any(|row| row.starts_with(&family.child.to_string())),
            "roots only: {out}"
        );
        assert!(
            out.ends_with(&format!("This session: {}", family.parent)),
            "{out}"
        );
    }

    #[tokio::test]
    async fn session_list_says_when_it_cut_the_listing() {
        let family = family().await;
        let list = SessionListTool {
            store: family.store.clone(),
            site: site(Permission::Unrestricted, family.parent),
        };
        let out = run(&list, serde_json::json!({})).await.expect("list");
        assert!(!out.contains("most recent are listed"), "{out}");
        for _ in 0..SESSION_LIST_ROWS {
            family
                .store
                .create_session(None, "test-profile")
                .await
                .expect("one more root");
        }
        let out = run(&list, serde_json::json!({})).await.expect("list");
        assert_eq!(
            out.lines().filter(|line| UUID_LINE.is_match(line)).count(),
            SESSION_LIST_ROWS as usize,
            "{out}"
        );
        assert!(
            out.contains(&format!(
                "Only the {SESSION_LIST_ROWS} most recent are listed"
            )),
            "{out}"
        );
        assert!(
            out.ends_with(&format!("This session: {}", family.parent)),
            "{out}"
        );
    }

    #[tokio::test]
    async fn session_search_finds_within_reach() {
        let family = family().await;
        let search = SessionSearchTool {
            store: family.store.clone(),
            site: site(Permission::Read, family.parent),
        };
        let out = run(&search, serde_json::json!({"query": "pelicans"}))
            .await
            .expect("search");
        assert!(
            out.contains(&family.parent.to_string()) && out.contains(&family.child.to_string()),
            "{out}"
        );
        assert!(
            !out.contains(&family.grandchild.to_string())
                && !out.contains(&family.stranger.to_string()),
            "{out}"
        );

        let search = SessionSearchTool {
            store: family.store.clone(),
            site: site(Permission::Unrestricted, family.parent),
        };
        let out = run(&search, serde_json::json!({"query": "pelicans"}))
            .await
            .expect("search");
        assert!(
            out.contains(&family.stranger.to_string())
                && out.contains(&family.parent.to_string())
                && !out.contains(&family.child.to_string()),
            "roots only without `include_children`: {out}"
        );
        let out = run(
            &search,
            serde_json::json!({"query": "pelicans", "include_children": true}),
        )
        .await
        .expect("search");
        assert!(
            out.contains(&family.child.to_string()) && out.contains(&family.grandchild.to_string()),
            "{out}"
        );
        let out = run(&search, serde_json::json!({"query": "walrus"}))
            .await
            .expect("search");
        assert_eq!(out, "No sessions match \"walrus\".");
    }
}
