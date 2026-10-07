//! The `checklist_*` family: `checklist_add` appends items, `checklist_edit` changes one by id,
//! `checklist_read` returns the list. Every call echoes the live list back, so the model always
//! holds the ids its next edit needs. The list's recovery from the conversation lives here too,
//! beside the tools that write it, the way the loaded tool set is recovered beside `tool_load`.

use std::collections::HashMap;

use async_trait::async_trait;

use super::{Tool, ToolOutput, util::resolve_session_id};
use crate::{
    checklist::{
        ADD_TOOL_NAME, ChecklistState, EDIT_TOOL_NAME, EditOutcome, ItemEdit, NewItem,
        READ_TOOL_NAME, SharedChecklist, format_checklist, parse_add, parse_edit,
    },
    conversation::{ContentBlock, Conversation},
    error::Result,
    permission::Permission,
    provider::ToolDefinition,
    store::Store,
};

/// Whether a call of `name` can change the list: what the dispatcher watches to announce a change
/// to the display. `checklist_read` cannot, so it is not here.
pub(crate) fn changes_the_list(name: &str) -> bool {
    name == ADD_TOOL_NAME || name == EDIT_TOOL_NAME
}

/// The `scratchpad` property every tool advertises, so a call carrying it is not refused for an
/// unknown key; the list is state rather than output, so there is nothing to redirect.
fn scratchpad_property() -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "description": "Accepted for uniformity with every other tool; the list is kept as state, \
                        so nothing is redirected."
    })
}

const STATUS_WORDS: [&str; 5] = [
    "pending",
    "in_progress",
    "deferred",
    "completed",
    "canceled",
];

/// The rule, in the words every tool description repeats: what binds a turn and the ways out.
const RULE: &str = "A turn cannot end while an item is pending or in progress: finish it, or set \
                    it completed, canceled (with a reason) or deferred (with a reason, and the \
                    `task` it waits on when that is a background task; the item reopens when the \
                    task's result is reported). Defer only what cannot proceed now because it \
                    waits on a person, an event or an explicit later, never because it is hard or \
                    tedious. Reasons are recorded. Completed and canceled items leave the list.";

/// The result every member returns, with what the call did ahead of the list.
fn echo(did: &str, state: &ChecklistState) -> ToolOutput {
    ToolOutput::text(format!("{did}\n\n{}", format_checklist(state)), false)
}

pub(crate) struct ChecklistAddTool {
    pub(crate) checklist: SharedChecklist,
}

#[async_trait]
impl Tool for ChecklistAddTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: ADD_TOOL_NAME.to_string(),
            description: format!(
                "Add items to your checklist, the things you have committed to do before this turn \
                 ends, as you commit to them. Each entry is a text or an object {{\"text\":..., \
                 \"status\":...}} with `in_progress` for the one you are starting on. Returns the \
                 ids assigned and the whole list. {RULE}"
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "scratchpad": scratchpad_property(),
                    "items": {
                        "type": "array",
                        "description": "The items to add, in order. Each is a text (added \
                                        pending) or an object {text, status} with status \
                                        pending or in_progress.",
                        "items": {
                            "anyOf": [
                                { "type": "string" },
                                {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "What needs to be done."
                                        },
                                        "status": {
                                            "type": "string",
                                            "enum": ["pending", "in_progress"]
                                        }
                                    },
                                    "required": ["text"]
                                }
                            ]
                        }
                    }
                },
                "required": ["items"]
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
        let _in_order = self.checklist.in_turn_order().await;
        let items = match parse_add(&input) {
            Ok(items) => items,
            Err(error) => {
                return Ok(ToolOutput::text(
                    format!("invalid {ADD_TOOL_NAME} input: {error}"),
                    true,
                ));
            }
        };
        Ok(self.checklist.update(|state| {
            let ids = state.add(items);
            let ids = ids
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            echo(&format!("Added with ids {ids}."), state)
        }))
    }
}

pub(crate) struct ChecklistEditTool {
    pub(crate) checklist: SharedChecklist,
    pub(crate) store: Store,
    pub(crate) site: crate::session::ToolSite,
}

#[async_trait]
impl Tool for ChecklistEditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: EDIT_TOOL_NAME.to_string(),
            description: format!(
                "Change one checklist item by `id`: its `status`, its `reason` (required for \
                 deferred and canceled), the background `task` a deferred item waits on, or its \
                 `text`. The whole call is refused if any part of it is wrong, and the list is \
                 returned. {RULE}"
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "scratchpad": scratchpad_property(),
                    "id": {
                        "type": "integer",
                        "description": "The item's id, as the list shows it."
                    },
                    "status": {
                        "type": "string",
                        "enum": STATUS_WORDS,
                        "description": "The state to move the item to. `completed` and \
                                        `canceled` take it off the list."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the item is deferred or canceled. Required with \
                                        either; refused with anything else."
                    },
                    "task": {
                        "type": "string",
                        "description": "With `deferred` only: the background task the item \
                                        waits on, by the id `task_list` shows. The item reopens \
                                        when the task's result is reported; a task that has \
                                        reported is refused. A new `reason` states the deferral \
                                        afresh, so name the task again to keep waiting on it."
                    },
                    "text": {
                        "type": "string",
                        "description": "New wording for the item."
                    }
                },
                "required": ["id"]
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
        let _in_order = self.checklist.in_turn_order().await;
        let edit = match parse_edit(&input) {
            Ok(edit) => edit,
            Err(error) => {
                return Ok(ToolOutput::text(
                    format!("invalid {EDIT_TOOL_NAME} input: {error}"),
                    true,
                ));
            }
        };
        let id = edit.id;
        // Planned against a copy, so the task the item would wait on, named or kept from an
        // earlier deferral, can be checked against the store before anything is written. It is
        // checked here and never on replay: the list is recovered from the calls as the model made
        // them, and the result's error flag is what tells the replay whether this one counted.
        let planned = match self.checklist.get().plan(edit.clone()) {
            Ok(planned) => planned,
            Err(error) => {
                return Ok(ToolOutput::text(format!("{EDIT_TOOL_NAME}: {error}"), true));
            }
        };
        if let Some(task) = planned.task() {
            let session_id = resolve_session_id(&self.site.session_id, EDIT_TOOL_NAME)?;
            let known = self
                .store
                .background_store()
                .resolve_background_task(session_id, task)
                .await?;
            match known {
                None => {
                    return Ok(ToolOutput::text(
                        format!(
                            "{EDIT_TOOL_NAME}: no background task in this session matches \
                             '{task}'. Call `task_list` for the current ids."
                        ),
                        true,
                    ));
                }
                // An item deferred on a task that has reported would be open the moment it was
                // deferred: the report is what reopens it, and it has already arrived.
                Some(known) if known.delivered_at.is_some() => {
                    return Ok(ToolOutput::text(
                        format!(
                            "{EDIT_TOOL_NAME}: task '{task}' {} and has reported, so nothing is \
                             left to wait for; finish item {id}, cancel it, or defer it with a \
                             new reason and, if it now waits on another task, that task.",
                            known.status.headline()
                        ),
                        true,
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(self.checklist.update(|state| match state.edit(edit) {
            Ok(EditOutcome::Kept(status)) => {
                echo(&format!("Item {id} is {}.", status.name()), state)
            }
            Ok(EditOutcome::Removed(word)) => echo(&format!("Item {id} {}.", word.name()), state),
            Err(error) => ToolOutput::text(format!("{EDIT_TOOL_NAME}: {error}"), true),
        }))
    }
}

pub(crate) struct ChecklistReadTool {
    pub(crate) checklist: SharedChecklist,
}

#[async_trait]
impl Tool for ChecklistReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: READ_TOOL_NAME.to_string(),
            description: "Return your checklist: one line per open item with its id, its state, \
                          what a deferred one waits on, and its text. Each add or edit already \
                          returns it, so this is for a look between them."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "scratchpad": scratchpad_property()
                }
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
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ReadInput {
            #[serde(default, rename = "scratchpad")]
            _scratchpad: Option<String>,
        }
        let _in_order = self.checklist.in_turn_order().await;
        if let Err(error) = serde_json::from_value::<ReadInput>(input) {
            return Ok(ToolOutput::text(
                format!("invalid {READ_TOOL_NAME} input: {error}"),
                true,
            ));
        }
        Ok(ToolOutput::text(
            format_checklist(&self.checklist.get()),
            false,
        ))
    }
}

/// A call waiting for its result during a replay.
enum Pending {
    Add(Vec<NewItem>),
    Edit(ItemEdit),
}

/// The list as the conversation records it: the snapshot the last compaction boundary carries,
/// then every `checklist_add` and `checklist_edit` call after it whose result was not an error,
/// in order. The one door for recovering the list, so resume, rewind, fork and import all read
/// the same answer from the same rows.
///
/// Read from the view rather than the raw events, so a repair that dropped a turn drops the
/// items that turn added; the boundary's snapshot stands in for the rows the view no longer
/// holds. A recorded call is read as the tool read it, with the `background` flag the dispatcher
/// strips gone. A call whose input then does not parse under the current schema is skipped: this
/// build could not have applied it, so a list that counted it would be one this build never
/// showed.
pub(crate) fn replay_checklist(conversation: &Conversation) -> ChecklistState {
    let (state, messages) = conversation.checklist_baseline();
    replay_checklist_over(state, messages)
}

/// [`replay_checklist`] from `state` over `messages` alone: what a compaction records on its
/// boundary, replayed over the rows it summarizes, so the rows it keeps are applied once, by the
/// replay that starts from the boundary, rather than twice.
pub(crate) fn replay_checklist_over(
    mut state: ChecklistState,
    messages: &[crate::conversation::Message],
) -> ChecklistState {
    let mut pending: HashMap<&str, Pending> = HashMap::new();
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolUse { id, name, input } if name == ADD_TOOL_NAME => {
                    if let Ok(items) = parse_add(&crate::tools::without_background_flag(input)) {
                        pending.insert(id, Pending::Add(items));
                    }
                }
                ContentBlock::ToolUse { id, name, input } if name == EDIT_TOOL_NAME => {
                    if let Ok(edit) = parse_edit(&crate::tools::without_background_flag(input)) {
                        pending.insert(id, Pending::Edit(edit));
                    }
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    is_error,
                    ..
                } => {
                    let Some(call) = pending.remove(tool_use_id.as_str()) else {
                        continue;
                    };
                    if *is_error {
                        continue;
                    }
                    match call {
                        Pending::Add(items) => {
                            state.add(items);
                        }
                        // The tool refused what the state refuses, so an error here is a call
                        // whose result lied; the list is what the tool's own rules allow.
                        Pending::Edit(edit) => {
                            if let Err(error) = state.edit(edit) {
                                tracing::warn!(
                                    "a recorded checklist_edit does not apply on replay: {error}"
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    state
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::{
        checklist::ChecklistStatus,
        conversation::{Event, Message, Role},
    };

    struct Family {
        add: ChecklistAddTool,
        read: ChecklistReadTool,
        list: SharedChecklist,
    }

    fn family() -> Family {
        let list = SharedChecklist::default();
        Family {
            add: ChecklistAddTool {
                checklist: list.clone(),
            },
            read: ChecklistReadTool {
                checklist: list.clone(),
            },
            list,
        }
    }

    async fn edit_tool(list: &SharedChecklist) -> ChecklistEditTool {
        // A session to resolve a task against; it has no tasks, which is the point of the one
        // test that names one.
        let site = crate::session::ToolSite::for_test();
        site.session_id.set(uuid::Uuid::new_v4());
        ChecklistEditTool {
            checklist: list.clone(),
            store: Store::for_test().await,
            site,
        }
    }

    async fn call(tool: &dyn Tool, input: serde_json::Value) -> ToolOutput {
        tool.execute(
            input,
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("a checklist call answers with a result, never an Err")
    }

    #[tokio::test]
    async fn a_read_takes_no_arguments() {
        let family = family();
        let result = call(&family.read, serde_json::json!({})).await;
        assert!(!result.is_error);
        assert!(result.text_content().contains("(no open items)"));
        let refused = call(&family.read, serde_json::json!({"id": 1})).await;
        assert!(refused.is_error, "{}", refused.text_content());
    }

    /// `scratchpad` is the parameter every tool takes, and `deny_unknown_fields` on the inputs
    /// would refuse a call that carries it unless each input names it.
    #[tokio::test]
    async fn the_universal_scratchpad_parameter_is_accepted() {
        let family = family();
        let result = call(
            &family.add,
            serde_json::json!({ "items": ["One"], "scratchpad": "plan" }),
        )
        .await;
        assert!(!result.is_error, "{}", result.text_content());
        assert_eq!(family.list.get().items.len(), 1);
        let result = call(&family.read, serde_json::json!({ "scratchpad": "plan" })).await;
        assert!(!result.is_error, "{}", result.text_content());
    }

    #[tokio::test]
    async fn an_add_appends_and_answers_with_the_ids() {
        let family = family();
        let result = call(
            &family.add,
            serde_json::json!({ "items": ["First", {"text": "Second", "status": "in_progress"}] }),
        )
        .await;
        assert!(!result.is_error, "{}", result.text_content());
        assert!(
            result.text_content().starts_with("Added with ids 1, 2."),
            "{}",
            result.text_content()
        );
        let result = call(&family.add, serde_json::json!({ "items": ["Third"] })).await;
        assert!(
            result.text_content().starts_with("Added with ids 3."),
            "an add appends, never replaces: {}",
            result.text_content()
        );
        let state = family.list.get();
        assert_eq!(state.items.len(), 3);
        assert_eq!(state.items[1].status, ChecklistStatus::InProgress);

        let refused = call(&family.add, serde_json::json!({ "items": [] })).await;
        assert!(refused.is_error);
        assert_eq!(
            family.list.get().items.len(),
            3,
            "a refused add leaves the list alone"
        );
    }

    #[tokio::test]
    async fn an_edit_moves_one_item_and_refuses_a_bad_one_whole() {
        let family = family();
        let edit = edit_tool(&family.list).await;
        call(&family.add, serde_json::json!({ "items": ["A", "B"] })).await;

        let result = call(&edit, serde_json::json!({ "id": 1, "status": "completed" })).await;
        assert!(!result.is_error, "{}", result.text_content());
        assert!(
            result.text_content().starts_with("Item 1 completed."),
            "{}",
            result.text_content()
        );
        assert_eq!(family.list.get().items.len(), 1);

        let refused = call(&edit, serde_json::json!({ "id": 2, "status": "canceled" })).await;
        assert!(refused.is_error, "a cancel needs a reason");
        assert!(
            refused.text_content().contains("reason"),
            "{}",
            refused.text_content()
        );
        let refused = call(&edit, serde_json::json!({ "id": 9, "status": "completed" })).await;
        assert!(refused.is_error);
        assert!(
            refused.text_content().contains("no open item"),
            "{}",
            refused.text_content()
        );
        assert_eq!(family.list.get().items.len(), 1, "refused whole");

        let result = call(
            &edit,
            serde_json::json!({ "id": 2, "status": "deferred", "reason": "waiting on Sam", "text": "B, reworded" }),
        )
        .await;
        assert!(!result.is_error, "{}", result.text_content());
        let item = &family.list.get().items[0];
        assert_eq!(item.status, ChecklistStatus::Deferred);
        assert_eq!(item.text, "B, reworded");
    }

    /// A task the session does not have is refused at the tool, the one place the store is in
    /// reach; the replay trusts that check through the result's error flag.
    #[tokio::test]
    async fn a_deferral_on_an_unknown_task_is_refused() {
        let family = family();
        let edit = edit_tool(&family.list).await;
        call(&family.add, serde_json::json!({ "items": ["A"] })).await;
        let refused = call(
            &edit,
            serde_json::json!({ "id": 1, "status": "deferred", "reason": "building", "task": "deadbeef" }),
        )
        .await;
        assert!(refused.is_error, "{}", refused.text_content());
        assert!(
            refused.text_content().contains("task_list"),
            "{}",
            refused.text_content()
        );
        assert_eq!(family.list.get().items[0].status, ChecklistStatus::Pending);
    }

    /// A task that has reported has nothing left to wait for, so an item cannot be deferred on
    /// it, named or kept from an earlier deferral: it would be open the moment it was deferred,
    /// and a model answering the nudge by deferring it on the same task again would never run
    /// out of nudges, each edit being a tool round. A task that is running, or has exited but
    /// not reported, is one the report will reopen the item for, so it is allowed.
    #[tokio::test]
    async fn a_deferral_on_a_task_that_has_already_reported_is_refused() {
        use crate::store::background::{BackgroundTask, TaskStatus};

        let family = family();
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("a session");
        let site = crate::session::ToolSite::for_test();
        site.session_id.set(session_id);
        let edit = ChecklistEditTool {
            checklist: family.list.clone(),
            store: store.clone(),
            site,
        };
        call(&family.add, serde_json::json!({ "items": ["A", "B"] })).await;
        let task_id = "1a2b3c4d-0000-4000-8000-000000000000";
        store
            .background_store()
            .start_background_task(&BackgroundTask {
                id: task_id.to_string(),
                session_id,
                tool: "shell_execute".to_string(),
                label: "make".to_string(),
                status: TaskStatus::Running,
                outcome: None,
                scratchpad_entry: None,
                started_at: chrono::Utc::now(),
                finished_at: None,
                announced_at: None,
                delivered_at: None,
                subagent_id: None,
            })
            .await
            .expect("a task");

        let result = call(
            &edit,
            serde_json::json!({ "id": 1, "status": "deferred", "reason": "building", "task": "1A2B3C4D" }),
        )
        .await;
        assert!(!result.is_error, "running: {}", result.text_content());
        store
            .background_store()
            .finish_background_task(task_id, TaskStatus::Completed, None, None)
            .await
            .expect("the task ends");
        let result = call(
            &edit,
            serde_json::json!({ "id": 2, "status": "deferred", "reason": "building", "task": task_id }),
        )
        .await;
        assert!(
            !result.is_error,
            "exited but unreported, the report is still to come: {}",
            result.text_content()
        );
        call(&edit, serde_json::json!({ "id": 2, "status": "pending" })).await;

        store
            .background_store()
            .mark_background_tasks_delivered(&[task_id.to_string()])
            .await
            .expect("reported");
        let refused = call(
            &edit,
            serde_json::json!({ "id": 2, "status": "deferred", "reason": "building", "task": "1a2b3c4d" }),
        )
        .await;
        assert!(refused.is_error, "{}", refused.text_content());
        assert!(
            refused.text_content().contains("finished and has reported"),
            "{}",
            refused.text_content()
        );
        assert_eq!(
            family.list.get().items[1].status,
            ChecklistStatus::Pending,
            "refused whole"
        );
        let refused = call(&edit, serde_json::json!({ "id": 1, "status": "deferred" })).await;
        assert!(
            refused.is_error,
            "a bare re-deferral keeps the task, and that task has reported: {}",
            refused.text_content()
        );
        assert_eq!(family.list.get().items[0].task.as_deref(), Some("1A2B3C4D"));
        let result = call(
            &edit,
            serde_json::json!({ "id": 1, "status": "deferred", "reason": "waiting on Sam now" }),
        )
        .await;
        assert!(
            !result.is_error,
            "a new reason states the deferral afresh, off the task: {}",
            result.text_content()
        );
        assert_eq!(family.list.get().items[0].task, None);
    }

    /// Each tool tells the model what it is for, names its parameters with the `scratchpad` every
    /// tool takes, and repeats the rule the turn is held to.
    #[test]
    fn every_member_describes_itself_and_its_parameters() {
        let family = family();
        let store_free_edit = ChecklistEditTool {
            checklist: family.list.clone(),
            store: {
                let runtime = tokio::runtime::Runtime::new().expect("a runtime");
                runtime.block_on(Store::for_test())
            },
            site: crate::session::ToolSite::for_test(),
        };
        let definitions = [
            (family.add.definition(), &["items", "scratchpad"][..]),
            (
                store_free_edit.definition(),
                &["id", "status", "reason", "task", "text", "scratchpad"][..],
            ),
            (family.read.definition(), &["scratchpad"][..]),
        ];
        for (definition, parameters) in definitions {
            assert!(!definition.description.is_empty(), "{}", definition.name);
            let properties = definition.parameters["properties"]
                .as_object()
                .unwrap_or_else(|| panic!("{} names its parameters", definition.name));
            for parameter in parameters {
                assert!(
                    properties.contains_key(*parameter),
                    "{} takes `{parameter}`",
                    definition.name
                );
            }
            assert_eq!(properties.len(), parameters.len(), "{}", definition.name);
        }
        assert!(
            family.add.definition().description.contains(RULE)
                && store_free_edit.definition().description.contains(RULE),
            "the rule rides every tool that changes the list"
        );
    }

    #[test]
    fn only_the_writing_members_change_the_list() {
        assert!(changes_the_list(ADD_TOOL_NAME));
        assert!(changes_the_list(EDIT_TOOL_NAME));
        assert!(!changes_the_list(READ_TOOL_NAME));
        assert!(!changes_the_list("file_read"));
    }

    fn call_block(id: &str, name: &str, input: serde_json::Value) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input,
        }
    }

    fn result_block(id: &str, is_error: bool) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: id.to_string(),
            content: vec![crate::conversation::ToolResultContent::Text {
                text: "ok".to_string(),
            }],
            is_error,
        }
    }

    fn round(calls: Vec<ContentBlock>, results: Vec<ContentBlock>) -> [Event; 2] {
        [
            Event::Append(Message {
                role: Role::Assistant,
                content: calls,
            }),
            Event::Append(Message {
                role: Role::User,
                content: results,
            }),
        ]
    }

    /// The replay applies what the tools applied, skipping a call that errored and one whose
    /// input this build does not read, which is how a call under a retired shape counts for
    /// nothing.
    #[test]
    fn the_replay_applies_successful_calls_in_order_and_nothing_else() {
        let mut events = vec![Event::Append(Message::user("plan"))];
        events.extend(round(
            vec![call_block(
                "a",
                ADD_TOOL_NAME,
                serde_json::json!({"items": ["one", "two"]}),
            )],
            vec![result_block("a", false)],
        ));
        events.extend(round(
            vec![
                call_block(
                    "b",
                    EDIT_TOOL_NAME,
                    serde_json::json!({"id": 1, "status": "completed"}),
                ),
                call_block(
                    "c",
                    EDIT_TOOL_NAME,
                    serde_json::json!({"id": 2, "status": "canceled"}),
                ),
                call_block(
                    "d",
                    ADD_TOOL_NAME,
                    serde_json::json!({"title": "T", "items": ["retired shape"]}),
                ),
            ],
            vec![
                result_block("b", false),
                result_block("c", true),
                result_block("d", false),
            ],
        ));
        let mut state = replay_checklist(&Conversation::from_events(events));
        let ids: Vec<u64> = state.items.iter().map(|item| item.id).collect();
        assert_eq!(ids, vec![2], "{state:?}");
        assert_eq!(
            state.add(vec![NewItem {
                text: "next".to_string(),
                status: ChecklistStatus::Pending,
            }]),
            vec![3],
            "the counter continued past the errored add's id"
        );
    }

    /// `background` is offered on every detachable tool and stripped before any tool runs, so a
    /// model that passes it on a list edit has the edit applied live while the recorded call
    /// still carries the key; the replay has to read the call as the tool did, or the next
    /// hydration drops what the model added.
    #[test]
    fn the_replay_reads_a_recorded_call_as_the_tool_did() {
        let mut events = vec![Event::Append(Message::user("plan"))];
        events.extend(round(
            vec![call_block(
                "a",
                ADD_TOOL_NAME,
                serde_json::json!({"items": ["kept"], "background": false}),
            )],
            vec![result_block("a", false)],
        ));
        events.extend(round(
            vec![call_block(
                "b",
                EDIT_TOOL_NAME,
                serde_json::json!({"id": 1, "status": "in_progress", "background": false}),
            )],
            vec![result_block("b", false)],
        ));
        let state = replay_checklist(&Conversation::from_events(events));
        assert_eq!(state.items.len(), 1, "{state:?}");
        assert_eq!(state.items[0].status, ChecklistStatus::InProgress);
        assert!(
            !crate::tools::detachable(ADD_TOOL_NAME) && !crate::tools::detachable(EDIT_TOOL_NAME),
            "and the flag is never offered on a list edit, whose result the model needs at once"
        );
    }

    /// A rewind drops the turn's rows from the view, and the items it added with them.
    #[test]
    fn the_replay_follows_the_view_so_a_rewind_drops_what_its_turn_added() {
        let mut events = vec![Event::Append(Message::user("plan"))];
        events.extend(round(
            vec![call_block(
                "a",
                ADD_TOOL_NAME,
                serde_json::json!({"items": ["kept"]}),
            )],
            vec![result_block("a", false)],
        ));
        events.push(Event::Append(Message::assistant_text("done")));
        events.push(Event::Append(Message::user("more")));
        events.extend(round(
            vec![call_block(
                "b",
                ADD_TOOL_NAME,
                serde_json::json!({"items": ["dropped"]}),
            )],
            vec![result_block("b", false)],
        ));
        events.push(Event::Append(Message::assistant_text("done again")));
        let mut conversation = Conversation::from_events(events);
        assert_eq!(replay_checklist(&conversation).items.len(), 2);
        conversation.rewind(1).expect("one turn to drop");
        let state = replay_checklist(&conversation);
        assert_eq!(
            state
                .items
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>(),
            vec!["kept"]
        );
    }

    /// The boundary's snapshot stands in for the rows a compaction replaced, and the calls after
    /// it continue from there with the ids it recorded.
    #[test]
    fn the_replay_starts_from_the_last_boundarys_snapshot() {
        let mut before = ChecklistState::default();
        before.add(vec![NewItem {
            text: "from before".to_string(),
            status: ChecklistStatus::Pending,
        }]);
        let mut events = vec![
            Event::Append(Message::user("plan")),
            Event::Append(Message::assistant_text("working")),
            Event::CompactBoundary {
                summary: Message::user("summary"),
                replaced_count: 2,
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: before,
            },
        ];
        events.extend(round(
            vec![call_block(
                "a",
                ADD_TOOL_NAME,
                serde_json::json!({"items": ["after"]}),
            )],
            vec![result_block("a", false)],
        ));
        let state = replay_checklist(&Conversation::from_events(events));
        assert_eq!(
            state
                .items
                .iter()
                .map(|item| (item.id, item.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "from before"), (2, "after")]
        );
    }
}
