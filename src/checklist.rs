//! The checklist the agent keeps for multi-step work: the items, the states an item sits in, the
//! operations the `checklist_*` tools apply, and the shared handle every collaborator reads. The
//! tools are `tools::checklist`; this module is the vocabulary, so the turn loop and the frontends
//! can hold a list without holding a tool.
//!
//! The list is the model's own commitments, and a turn cannot end while one is open. The turn loop
//! enforces that; this module only says what open means, in [`is_open`], so the loop, the display
//! and the session record cannot disagree about it.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// The tool that appends items.
pub(crate) const ADD_TOOL_NAME: &str = "checklist_add";
/// The tool that changes one item.
pub(crate) const EDIT_TOOL_NAME: &str = "checklist_edit";
/// The tool that returns the list.
pub(crate) const READ_TOOL_NAME: &str = "checklist_read";

/// Whether `name` is one of the three.
pub(crate) fn is_tool(name: &str) -> bool {
    name == ADD_TOOL_NAME || name == EDIT_TOOL_NAME || name == READ_TOOL_NAME
}

/// A background task's id as the list shows it: the prefix `task_list` shows, whether the model
/// named that or the whole id.
pub(crate) fn short_task_id(task: &str) -> &str {
    task.get(..crate::text::ID_PREFIX).unwrap_or(task)
}

/// A state an item sits in while it is on the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub(crate) enum ChecklistStatus {
    Pending,
    InProgress,
    /// Waiting on something the model cannot move: a person, an event, a background task.
    Deferred,
}

impl ChecklistStatus {
    /// The state's one spelling, on every surface.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Deferred => "deferred",
        }
    }
}

/// What a model may set an item to: a state it then sits in, or a disposition that takes it off
/// the list. The aliases are spellings a model emits for the same word, accepted so a near miss
/// does not cost a round; meka writes one spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StatusWord {
    Pending,
    #[serde(
        alias = "wip",
        alias = "in-progress",
        alias = "in progress",
        alias = "started"
    )]
    InProgress,
    #[serde(
        alias = "waiting",
        alias = "blocked",
        alias = "later",
        alias = "paused"
    )]
    Deferred,
    #[serde(alias = "done", alias = "complete", alias = "finished")]
    Completed,
    #[serde(
        alias = "cancelled",
        alias = "skipped",
        alias = "dropped",
        alias = "wontfix"
    )]
    Canceled,
}

impl StatusWord {
    /// The word's one spelling.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Deferred => "deferred",
            Self::Completed => "completed",
            Self::Canceled => "canceled",
        }
    }

    /// The state an item sits in under this word, or `None` for a disposition.
    const fn state(self) -> Option<ChecklistStatus> {
        match self {
            Self::Pending => Some(ChecklistStatus::Pending),
            Self::InProgress => Some(ChecklistStatus::InProgress),
            Self::Deferred => Some(ChecklistStatus::Deferred),
            Self::Completed | Self::Canceled => None,
        }
    }
}

/// One item on the list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct ChecklistItem {
    pub(crate) id: u64,
    pub(crate) text: String,
    pub(crate) status: ChecklistStatus,
    /// Why the item waits; present on every deferred item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
    /// The background task a deferred item waits on, as the model named it: the full id or the
    /// short prefix `task_list` shows. Kept as named rather than resolved, because the list is
    /// recovered from the model's calls and a resolution the call does not carry could not be.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) task: Option<String>,
}

/// The list: open items only, with the id the next one takes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ChecklistState {
    /// Never reused: an edit names an id, and a reused one would land on a different item than
    /// the model meant.
    next_id: u64,
    pub(crate) items: Vec<ChecklistItem>,
}

impl Default for ChecklistState {
    fn default() -> Self {
        Self {
            next_id: 1,
            items: Vec::new(),
        }
    }
}

/// A background task whose outcome the model has been given, as the turn loop and the display
/// hand it to [`is_open`]: the id and how it ended, in the words the report led with. A task that
/// has merely exited is not one of these: its report arrives as a turn of its own, and until then
/// the model has nothing to act on, so an item deferred on it keeps waiting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReportedTask {
    pub(crate) id: String,
    /// "finished", "failed", "was canceled" or "was interrupted": a predicate for a sentence
    /// about the task.
    pub(crate) ended: &'static str,
}

/// The task under `item`, among `reported`, when the item waits on one that has reported.
///
/// Matched by prefix the way `task_cancel` resolves the same argument, folded the same way: the
/// model may have named the short id in either case, and the tool that accepted it checked that
/// it named one task of the session through that resolver, so this has to agree with it or an
/// item accepted as deferred on a task never reopens.
pub(crate) fn reported_task<'a>(
    item: &ChecklistItem,
    reported: &'a [ReportedTask],
) -> Option<&'a ReportedTask> {
    let named = crate::text::id_prefix_for_matching(item.task.as_deref()?);
    reported.iter().find(|task| task.id.starts_with(&named))
}

/// Whether `item` still binds the turn: pending, in progress, or deferred on a task whose outcome
/// has since been reported, so the model has to finish it, cancel it or defer it again.
pub(crate) fn is_open(item: &ChecklistItem, reported: &[ReportedTask]) -> bool {
    match item.status {
        ChecklistStatus::Pending | ChecklistStatus::InProgress => true,
        ChecklistStatus::Deferred => reported_task(item, reported).is_some(),
    }
}

/// An item to add: validated text and the state it starts in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NewItem {
    pub(crate) text: String,
    pub(crate) status: ChecklistStatus,
}

/// One edit, as `checklist_edit` and the replay both read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ItemEdit {
    pub(crate) id: u64,
    pub(crate) status: Option<StatusWord>,
    pub(crate) reason: Option<String>,
    pub(crate) task: Option<String>,
    pub(crate) text: Option<String>,
}

/// What an edit did to its item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EditOutcome {
    /// The item stays, in `status`.
    Kept(ChecklistStatus),
    /// The item left the list under this disposition.
    Removed(StatusWord),
}

/// What an item becomes under an edit the rules allow.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Change {
    Keep {
        status: ChecklistStatus,
        reason: Option<String>,
        task: Option<String>,
    },
    Remove(StatusWord),
}

/// An edit checked against the rules and ready to apply. Planning and applying are two steps so
/// a caller with a question the list cannot answer, whether the task the item would wait on is
/// one it may wait on, can ask it between them, ahead of any change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedEdit {
    index: usize,
    change: Change,
    text: Option<String>,
}

impl PlannedEdit {
    /// The background task the item waits on once this is applied, named or kept.
    pub(crate) fn task(&self) -> Option<&str> {
        match &self.change {
            Change::Keep {
                task: Some(task), ..
            } => Some(task),
            _ => None,
        }
    }
}

/// The object form of an item to add. Unknown keys are refused rather than dropped: a `reason`
/// or a `task` here is a deferral the model meant to record, and an add that silently kept the
/// item pending would lose it without a word.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ItemObject {
    text: String,
    #[serde(default)]
    status: Option<StatusWord>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AddInput {
    /// Each a bare string or an [`ItemObject`], told apart below rather than by an untagged
    /// enum, whose refusal names neither the key nor the value that was wrong.
    items: Vec<serde_json::Value>,
    #[serde(default, rename = "scratchpad")]
    _scratchpad: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditInput {
    id: u64,
    #[serde(default)]
    status: Option<StatusWord>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "scratchpad")]
    _scratchpad: Option<String>,
}

/// The items a `checklist_add` call names, or why it names none. One parser for the tool and
/// the replay, so a call the tool accepted is a call the replay applies.
pub(crate) fn parse_add(input: &serde_json::Value) -> Result<Vec<NewItem>, String> {
    let parsed = AddInput::deserialize(input).map_err(|error| error.to_string())?;
    if parsed.items.is_empty() {
        return Err("`items` must name at least one item".to_string());
    }
    parsed
        .items
        .into_iter()
        .map(|item| {
            let (text, status) = match item {
                serde_json::Value::String(text) => (text, None),
                serde_json::Value::Object(_) => {
                    let ItemObject { text, status } =
                        ItemObject::deserialize(item).map_err(|error| error.to_string())?;
                    (text, status)
                }
                _ => {
                    return Err(
                        "each entry of `items` is a text or an object {text, status}".to_string(),
                    );
                }
            };
            let text = clean_text(&text)?;
            // Only the two states that need nothing else: a deferral carries a reason, and a
            // disposition ends an item, so both go through `checklist_edit`, which asks for what
            // they need.
            let status = match status {
                None | Some(StatusWord::Pending) => ChecklistStatus::Pending,
                Some(StatusWord::InProgress) => ChecklistStatus::InProgress,
                Some(word) => {
                    return Err(format!(
                        "an item is added pending or in_progress; set `{}` with checklist_edit",
                        word.name()
                    ));
                }
            };
            Ok(NewItem { text, status })
        })
        .collect()
}

/// The edit a `checklist_edit` call names, or why it names none. The same one parser.
pub(crate) fn parse_edit(input: &serde_json::Value) -> Result<ItemEdit, String> {
    let parsed = EditInput::deserialize(input).map_err(|error| error.to_string())?;
    Ok(ItemEdit {
        id: parsed.id,
        status: parsed.status,
        reason: parsed.reason,
        task: parsed.task,
        text: parsed.text,
    })
}

/// A text or a reason as the one line it is shown as. The list is read a line per item, by the
/// model and on the screen alike, so a newline in either would fabricate a line that reads as an
/// item of its own; flattened here, at the one door both tools and the replay read through, so
/// the stored words are the shown words.
fn clean_line(text: &str) -> String {
    crate::text::sanitize_to_line(text, usize::MAX)
        .trim()
        .to_string()
}

fn clean_text(text: &str) -> Result<String, String> {
    let text = clean_line(text);
    if text.is_empty() {
        return Err("an item's `text` must not be blank".to_string());
    }
    Ok(text)
}

fn clean_reason(reason: Option<String>, word: StatusWord) -> Result<String, String> {
    reason
        .as_deref()
        .map(clean_line)
        .filter(|reason| !reason.is_empty())
        .ok_or_else(|| format!("`{}` needs a `reason`", word.name()))
}

impl ChecklistState {
    /// Append `items`, returning the ids they took, in order.
    pub(crate) fn add(&mut self, items: Vec<NewItem>) -> Vec<u64> {
        let mut ids = Vec::with_capacity(items.len());
        for item in items {
            let id = self.next_id;
            self.next_id += 1;
            ids.push(id);
            self.items.push(ChecklistItem {
                id,
                text: item.text,
                status: item.status,
                reason: None,
                task: None,
            });
        }
        ids
    }

    /// Check `edit` against the rules, or refuse it whole with the reason.
    ///
    /// A disposition takes the item off the list; `canceled` needs a reason, because it is
    /// otherwise the cheapest way out of a commitment, and `completed` has the work as its
    /// evidence. A deferral needs a reason and may name the task it waits on. A reason or a task
    /// on any other state would describe nothing, so it is refused rather than dropped.
    ///
    /// A deferral states what the item waits on, so a new reason states it afresh and a task not
    /// named with it is dropped; a re-deferral that gives neither keeps both. That is what lets
    /// one call move an item off a task that has reported, and it is a rule about the call alone,
    /// so the replay reads the same list from the same calls.
    pub(crate) fn plan(&self, edit: ItemEdit) -> Result<PlannedEdit, String> {
        let index = self
            .items
            .iter()
            .position(|item| item.id == edit.id)
            .ok_or_else(|| {
                format!(
                    "no open item has id {}; call checklist_read for the ids",
                    edit.id
                )
            })?;
        let text = edit.text.as_deref().map(clean_text).transpose()?;
        let item = &self.items[index];
        let current = item.status;
        let word = edit.status.unwrap_or(match current {
            ChecklistStatus::Pending => StatusWord::Pending,
            ChecklistStatus::InProgress => StatusWord::InProgress,
            ChecklistStatus::Deferred => StatusWord::Deferred,
        });
        let change = match word.state() {
            None => {
                if word == StatusWord::Canceled {
                    clean_reason(edit.reason, word)?;
                } else if edit.reason.is_some() {
                    return Err("a `reason` belongs to `deferred` or `canceled`".to_string());
                }
                if edit.task.is_some() {
                    return Err("a `task` belongs to `deferred`".to_string());
                }
                Change::Remove(word)
            }
            Some(ChecklistStatus::Deferred) => {
                let task = match edit.task {
                    Some(task) => {
                        let task = task.trim().to_string();
                        if task.is_empty() {
                            return Err("`task` must name a background task".to_string());
                        }
                        Some(task)
                    }
                    None if edit.reason.is_none() && current == ChecklistStatus::Deferred => {
                        item.task.clone()
                    }
                    None => None,
                };
                let reason = match edit.reason {
                    Some(reason) => clean_reason(Some(reason), word)?,
                    None => item
                        .reason
                        .clone()
                        .filter(|_| current == ChecklistStatus::Deferred)
                        .ok_or_else(|| format!("`{}` needs a `reason`", word.name()))?,
                };
                Change::Keep {
                    status: ChecklistStatus::Deferred,
                    reason: Some(reason),
                    task,
                }
            }
            Some(status) => {
                if edit.reason.is_some() {
                    return Err("a `reason` belongs to `deferred` or `canceled`".to_string());
                }
                if edit.task.is_some() {
                    return Err("a `task` belongs to `deferred`".to_string());
                }
                Change::Keep {
                    status,
                    reason: None,
                    task: None,
                }
            }
        };
        Ok(PlannedEdit {
            index,
            change,
            text,
        })
    }

    /// Apply an edit [`Self::plan`] allowed, on the state it was planned against.
    pub(crate) fn apply(&mut self, planned: PlannedEdit) -> EditOutcome {
        let outcome = match planned.change {
            Change::Remove(word) => {
                self.items.remove(planned.index);
                return EditOutcome::Removed(word);
            }
            Change::Keep {
                status,
                reason,
                task,
            } => {
                let item = &mut self.items[planned.index];
                item.status = status;
                item.reason = reason;
                item.task = task;
                EditOutcome::Kept(status)
            }
        };
        if let Some(text) = planned.text {
            self.items[planned.index].text = text;
        }
        outcome
    }

    /// Plan and apply `edit`, or refuse it whole with the reason, leaving the list untouched.
    pub(crate) fn edit(&mut self, edit: ItemEdit) -> Result<EditOutcome, String> {
        let planned = self.plan(edit)?;
        Ok(self.apply(planned))
    }

    /// Whether a surface showing `shown` has anything new to show in `self`: the items, not the
    /// counter, which moves without a visible change. The one question both announcers ask, the
    /// dispatcher after a round and the hydration that reads the list back.
    pub(crate) fn shows_the_same_items(&self, shown: &Self) -> bool {
        self.items == shown.items
    }

    /// The items that still bind the turn; see [`is_open`].
    pub(crate) fn open_items<'a>(&'a self, reported: &[ReportedTask]) -> Vec<&'a ChecklistItem> {
        self.items
            .iter()
            .filter(|item| is_open(item, reported))
            .collect()
    }
}

/// The list of one session, shared by handle between the `checklist_*` tools that edit it, the
/// turn loop that reads it at the end of a turn, and the frontend that draws it.
///
/// A `std` lock rather than a `tokio` one: the tool edits the state in place inside
/// [`Self::update`] with no `.await` in reach, and every other holder takes a copy. Poisoning is
/// recovered through [`crate::sync`] like every other `std` lock in the tree.
#[derive(Clone, Default)]
pub(crate) struct SharedChecklist {
    state: Arc<std::sync::RwLock<ChecklistState>>,
    /// Held by each `checklist_*` call for its whole run. The dispatcher runs the calls of one
    /// assistant message concurrently, and an edit that names a task awaits the store while one
    /// that does not finishes at once; the replay applies the recorded calls in the order the
    /// model wrote them, so the live list has to be built in that order too. A fair async lock
    /// taken first thing does it: the calls are polled in source order, so they queue in it.
    turn_order: Arc<tokio::sync::Mutex<()>>,
}

impl SharedChecklist {
    /// A copy of the current state.
    pub(crate) fn get(&self) -> ChecklistState {
        crate::sync::read(&self.state).clone()
    }

    /// The place in line a call takes before it reads or writes; see the field.
    pub(crate) async fn in_turn_order(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.turn_order.lock().await
    }

    /// Edit the state in place. One write lock spans the whole edit, so a reader never sees a
    /// half-applied one.
    pub(crate) fn update<R>(&self, edit: impl FnOnce(&mut ChecklistState) -> R) -> R {
        edit(&mut crate::sync::write(&self.state))
    }

    /// Replace the state whole, with what the conversation records.
    pub(crate) fn replace(&self, state: ChecklistState) {
        *crate::sync::write(&self.state) = state;
    }
}

/// One item as a line: the id, a tag in parentheses with the state in the word `checklist_edit`
/// takes and what a deferred item waits on, then the text. Plain text for the model alone, in the
/// tool echo, the nudge and the compaction copy; the terminal rendering is
/// `render::render_checklist`. The tag stands between the id and the text so what meka says
/// about the item comes ahead of the words the model wrote, which may say anything, and the id is
/// delimited whatever the text opens with.
pub(crate) fn format_item(item: &ChecklistItem) -> String {
    let tag = match item.status {
        ChecklistStatus::Pending | ChecklistStatus::InProgress => item.status.name().to_string(),
        ChecklistStatus::Deferred => match &item.task {
            Some(task) => format!(
                "deferred until task {} reports: {}",
                short_task_id(task),
                reason_of(item)
            ),
            None => format!("deferred: {}", reason_of(item)),
        },
    };
    tagged_line(item, &tag)
}

/// [`format_item`] for an item whose task has reported, as the nudge lists it: the tag says what
/// reopened it and keeps the reason it was deferred for, ahead of the text like any other tag.
pub(crate) fn format_reopened_item(item: &ChecklistItem, task: &ReportedTask) -> String {
    tagged_line(
        item,
        &format!(
            "task {} {}, so this item is open again; was deferred: {}",
            short_task_id(&task.id),
            task.ended,
            reason_of(item)
        ),
    )
}

fn reason_of(item: &ChecklistItem) -> &str {
    item.reason.as_deref().unwrap_or_default()
}

fn tagged_line(item: &ChecklistItem, tag: &str) -> String {
    format!("- {} ({tag}): {}", item.id, item.text)
}

/// The whole list as plain text, one line per item.
pub(crate) fn format_checklist(state: &ChecklistState) -> String {
    if state.items.is_empty() {
        return "(no open items)\n".to_string();
    }
    let mut output = String::new();
    for item in &state.items {
        output.push_str(&format_item(item));
        output.push('\n');
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(texts: &[&str]) -> Vec<NewItem> {
        texts
            .iter()
            .map(|text| NewItem {
                text: (*text).to_string(),
                status: ChecklistStatus::Pending,
            })
            .collect()
    }

    fn edit(id: u64, status: StatusWord) -> ItemEdit {
        ItemEdit {
            id,
            status: Some(status),
            reason: None,
            task: None,
            text: None,
        }
    }

    fn with_reason(mut edit: ItemEdit, reason: &str) -> ItemEdit {
        edit.reason = Some(reason.to_string());
        edit
    }

    #[test]
    fn ids_count_up_and_are_never_reused() {
        let mut state = ChecklistState::default();
        assert_eq!(state.add(items(&["one", "two"])), vec![1, 2]);
        state
            .edit(edit(1, StatusWord::Completed))
            .expect("completes");
        assert_eq!(state.add(items(&["three"])), vec![3]);
        assert_eq!(
            state.items.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn a_disposition_takes_the_item_off_the_list() {
        let mut state = ChecklistState::default();
        state.add(items(&["one", "two"]));
        assert_eq!(
            state.edit(edit(1, StatusWord::Completed)),
            Ok(EditOutcome::Removed(StatusWord::Completed))
        );
        assert_eq!(
            state.edit(with_reason(edit(2, StatusWord::Canceled), "superseded")),
            Ok(EditOutcome::Removed(StatusWord::Canceled))
        );
        assert!(state.items.is_empty());
        assert!(
            state.edit(edit(1, StatusWord::Pending)).is_err(),
            "a removed id is gone"
        );
    }

    #[test]
    fn a_cancel_and_a_deferral_need_a_reason_and_nothing_else_takes_one() {
        let mut state = ChecklistState::default();
        state.add(items(&["one"]));
        let refused = state
            .edit(edit(1, StatusWord::Canceled))
            .expect_err("no reason");
        assert!(refused.contains("reason"), "{refused}");
        let refused = state
            .edit(edit(1, StatusWord::Deferred))
            .expect_err("no reason");
        assert!(refused.contains("reason"), "{refused}");
        let refused = state
            .edit(with_reason(edit(1, StatusWord::InProgress), "because"))
            .expect_err("a reason on an open state describes nothing");
        assert!(refused.contains("reason"), "{refused}");
        assert_eq!(
            state.items[0].status,
            ChecklistStatus::Pending,
            "refused whole"
        );
        assert_eq!(
            state.edit(with_reason(
                edit(1, StatusWord::Deferred),
                " waiting on Sam "
            )),
            Ok(EditOutcome::Kept(ChecklistStatus::Deferred))
        );
        assert_eq!(state.items[0].reason.as_deref(), Some("waiting on Sam"));
    }

    #[test]
    fn a_task_rides_only_a_deferral_and_leaves_with_it() {
        let mut state = ChecklistState::default();
        state.add(items(&["build"]));
        let mut deferred = with_reason(edit(1, StatusWord::Deferred), "building");
        deferred.task = Some("1a2b3c4d".to_string());
        state.edit(deferred).expect("defers on the task");
        assert_eq!(state.items[0].task.as_deref(), Some("1a2b3c4d"));

        let mut again = edit(1, StatusWord::Deferred);
        again.reason = None;
        assert_eq!(
            state.edit(again),
            Ok(EditOutcome::Kept(ChecklistStatus::Deferred)),
            "a bare re-deferral keeps the reason and the task"
        );
        assert_eq!(state.items[0].reason.as_deref(), Some("building"));
        assert_eq!(state.items[0].task.as_deref(), Some("1a2b3c4d"));

        let afresh = with_reason(edit(1, StatusWord::Deferred), "waiting on Sam now");
        assert_eq!(
            state.plan(afresh.clone()).expect("plans").task(),
            None,
            "a new reason states the deferral afresh, and a task not named with it is dropped"
        );
        state.edit(afresh).expect("re-defers");
        assert_eq!(state.items[0].reason.as_deref(), Some("waiting on Sam now"));
        assert_eq!(state.items[0].task, None);
        let mut named_again = with_reason(edit(1, StatusWord::Deferred), "building again");
        named_again.task = Some("5e6f7a8b".to_string());
        assert_eq!(
            state.plan(named_again.clone()).expect("plans").task(),
            Some("5e6f7a8b")
        );
        state.edit(named_again).expect("re-defers on a task");
        assert_eq!(state.items[0].task.as_deref(), Some("5e6f7a8b"));

        let mut on_pending = edit(1, StatusWord::InProgress);
        on_pending.task = Some("1a2b3c4d".to_string());
        assert!(state.edit(on_pending).is_err());
        state
            .edit(edit(1, StatusWord::InProgress))
            .expect("resumes");
        assert_eq!(state.items[0].reason, None);
        assert_eq!(state.items[0].task, None);
    }

    #[test]
    fn open_means_pending_in_progress_or_deferred_on_an_ended_task() {
        let mut state = ChecklistState::default();
        state.add(items(&["a", "b", "c", "d"]));
        state
            .edit(edit(2, StatusWord::InProgress))
            .expect("in progress");
        state
            .edit(with_reason(edit(3, StatusWord::Deferred), "waiting on Sam"))
            .expect("deferred");
        let mut on_task = with_reason(edit(4, StatusWord::Deferred), "building");
        on_task.task = Some("1a2b3c4d".to_string());
        state.edit(on_task).expect("deferred on a task");

        let none: Vec<ReportedTask> = Vec::new();
        let open: Vec<u64> = state.open_items(&none).iter().map(|item| item.id).collect();
        assert_eq!(open, vec![1, 2], "a deferral does not bind the turn");

        let reported = vec![ReportedTask {
            id: "1a2b3c4d-full-id".to_string(),
            ended: "finished",
        }];
        let open: Vec<u64> = state
            .open_items(&reported)
            .iter()
            .map(|item| item.id)
            .collect();
        assert_eq!(
            open,
            vec![1, 2, 4],
            "until the task it waits on has reported"
        );
        state.items[3].task = Some("1A2B3C4D".to_string());
        assert_eq!(
            state.open_items(&reported).len(),
            3,
            "named in either case, as the resolver that accepted it allows"
        );
        assert_eq!(
            reported_task(&state.items[3], &reported).map(|task| task.ended),
            Some("finished")
        );
    }

    #[test]
    fn parsing_accepts_strings_objects_and_the_models_spellings() {
        let parsed = parse_add(&serde_json::json!({
            "items": ["one", {"text": " two ", "status": "wip"}],
            "scratchpad": "ignored"
        }))
        .expect("parses");
        assert_eq!(parsed, vec![
            NewItem {
                text: "one".to_string(),
                status: ChecklistStatus::Pending
            },
            NewItem {
                text: "two".to_string(),
                status: ChecklistStatus::InProgress
            }
        ]);
        let edit = parse_edit(&serde_json::json!({"id": 3, "status": "done"})).expect("parses");
        assert_eq!(edit.status, Some(StatusWord::Completed));
        assert_eq!(
            serde_json::to_value(StatusWord::Canceled).expect("serializes"),
            serde_json::json!("canceled"),
            "meka writes one spelling"
        );
    }

    #[test]
    fn parsing_refuses_an_empty_add_a_blank_text_a_disposition_on_add_and_an_unknown_field() {
        assert!(parse_add(&serde_json::json!({"items": []})).is_err());
        assert!(parse_add(&serde_json::json!({"items": ["  "]})).is_err());
        let refused = parse_add(&serde_json::json!({"items": [{"text": "x", "status": "done"}]}))
            .expect_err("a disposition on add");
        assert!(refused.contains("checklist_edit"), "{refused}");
        let refused =
            parse_add(&serde_json::json!({"items": [{"text": "x", "status": "deferred"}]}))
                .expect_err("a deferral on add would carry no reason");
        assert!(refused.contains("checklist_edit"), "{refused}");
        let refused = parse_add(&serde_json::json!({
            "items": [{"text": "x", "reason": "waiting", "task": "1a2b3c4d"}]
        }))
        .expect_err("a deferral's fields on an add are refused, never dropped");
        assert!(
            refused.contains("unknown field `reason`") && refused.contains("`status`"),
            "the refusal names the key and what was expected: {refused}"
        );
        let refused = parse_add(&serde_json::json!({"items": [{"text": "x", "status": "bogus"}]}))
            .expect_err("an unknown status word");
        assert!(refused.contains("bogus"), "{refused}");
        let refused = parse_add(&serde_json::json!({"items": [7]})).expect_err("a number");
        assert!(refused.contains("a text or an object"), "{refused}");
        assert!(parse_add(&serde_json::json!({"title": "T", "items": ["x"]})).is_err());
        assert!(parse_edit(&serde_json::json!({"set": {"1": "completed"}})).is_err());
        assert!(parse_edit(&serde_json::json!({"id": 1, "status": "bogus"})).is_err());
    }

    /// The list is read a line per item, so a text or a reason is one line however the model
    /// wrote it: a newline in either would fabricate a line that reads as an item of its own.
    #[test]
    fn a_text_or_a_reason_is_one_line_however_it_was_written() {
        let parsed =
            parse_add(&serde_json::json!({"items": ["a\n- 99 (pending): b\r\n"]})).expect("parses");
        assert_eq!(parsed[0].text, "a - 99 (pending): b");
        let mut state = ChecklistState::default();
        state.add(parsed);
        let edit = parse_edit(&serde_json::json!({
            "id": 1, "status": "deferred", "reason": "waiting\non Sam", "text": "ask\tSam"
        }))
        .expect("parses");
        state.edit(edit).expect("defers");
        assert_eq!(state.items[0].reason.as_deref(), Some("waiting on Sam"));
        assert_eq!(state.items[0].text, "ask    Sam");
        assert_eq!(format_checklist(&state).lines().count(), 1);
    }

    #[test]
    fn the_plain_rendering_names_what_each_item_waits_on() {
        let mut state = ChecklistState::default();
        assert_eq!(format_checklist(&state), "(no open items)\n");
        state.add(items(&["a", "b", "c", "d"]));
        state
            .edit(edit(2, StatusWord::InProgress))
            .expect("in progress");
        state
            .edit(with_reason(edit(3, StatusWord::Deferred), "waiting on Sam"))
            .expect("deferred");
        let mut on_task = with_reason(edit(4, StatusWord::Deferred), "building");
        on_task.task = Some("1a2b3c4d".to_string());
        state.edit(on_task).expect("deferred on a task");
        assert_eq!(
            format_checklist(&state),
            "- 1 (pending): a\n- 2 (in_progress): b\n- 3 (deferred: waiting on Sam): c\n\
             - 4 (deferred until task 1a2b3c4d reports: building): d\n"
        );
        let reported = ReportedTask {
            id: "1a2b3c4d-0000-4000-8000-000000000000".to_string(),
            ended: "finished",
        };
        assert_eq!(
            format_reopened_item(&state.items[3], &reported),
            "- 4 (task 1a2b3c4d finished, so this item is open again; was deferred: building): d"
        );
        state.items[3].task = Some("1a2b3c4d-0000-4000-8000-000000000000".to_string());
        assert!(
            format_item(&state.items[3])
                == "- 4 (deferred until task 1a2b3c4d reports: building): d",
            "a task named in full is shown by the prefix task_list shows"
        );
    }
}
