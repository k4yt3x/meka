//! The `memory_*` tools: the agent's read/write access to its own durable notes
//! ([`crate::memory`]).
//!
//! All of them gate at [`Permission::Read`], matching `scratchpad_*` and `checklist_*`: these write
//! to a store meka owns in its own database, not to the user's tree, and the motivating deployment
//! runs at read permission permanently. Gating them at `workspace` would mean an agent that can
//! never remember anything, which defeats the feature.
//!
//! [`crate::memory::validate_memory_name`] is checked at every door that *writes* a name. The name
//! is not a path, so it is not a file-write primitive, but it is what `meka memory export` turns
//! into a file name and it is text the model reads in every turn's index.
//!
//! The doors that only look a name up (`memory_read` and `memory_delete`) check
//! [`crate::memory::validate_memory_lookup`] instead, which requires only that the name is not
//! empty: applying the write rule to them would leave a row that reached the column past the tools
//! listed to the model in the `[Memory]` index and refused by every door that could open or remove
//! it. The cost a length bound would have capped is bounded in `did_you_mean_hint` instead.

use std::sync::Arc;

use async_trait::async_trait;

use super::{
    Tool, ToolOutput,
    util::{optional_str, optional_str_allowing_empty, require_str, string_or_list},
};
use crate::{
    error::{MekaError, Result},
    memory,
    permission::Permission,
    provider::ToolDefinition,
    store::memory::{MemoryStore, MemoryView, SearchResults, Terms, WriteRequest},
};

/// Turn a store error into the tool's own failure, so the model sees which call failed.
fn tool_error(tool_name: &str, error: impl std::fmt::Display) -> MekaError {
    MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: error.to_string(),
    }
}

/// The tiers the session in front of a tool sees: its project's beside the global tier, or the
/// global tier alone.
fn view_of(project: &crate::session::SharedProject) -> MemoryView {
    MemoryView::for_session(project.get().as_deref())
}

pub(super) struct MemoryWriteTool {
    pub(crate) memories: Arc<MemoryStore>,
    /// The session's project, read per call so a move mid-session changes what the next call
    /// sees.
    pub(crate) project: crate::session::SharedProject,
}

#[async_trait]
impl Tool for MemoryWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_write".to_string(),
            description: "Save preferences, constraints, decisions, or facts that later sessions need. Update an existing name to refine it: omitted fields keep their values, and a given `body` replaces the whole body (`memory_edit` changes part of one). Temporary task state and large outputs belong in the scratchpad, which later sessions do not read. Write a description that states the useful fact on its own.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Identifier, letters/digits/-/_ only (e.g. 'alice-timezone')."
                    },
                    "description": {
                        "type": "string",
                        "description": "One line stating the fact itself, used in the bounded memory index. Required when creating a memory; \
                                        omit it to leave an existing memory's description untouched."
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 9,
                        "default": crate::entry::DEFAULT_PRIORITY,
                        "description": "Importance, 0 (highest) to 9: decides index order and search weight, not what loads. 0-1 standing directives, 2-4 durable facts, 6-9 situational. Default: 5."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lowercase labels ([a-z0-9-], at most 10), e.g. ['infra', \
                                        'deploy']. Indexed as words, so memory_search finds them. \
                                        Omit to leave an existing memory's tags untouched; pass \
                                        [] to clear them."
                    },
                    "body": {
                        "type": "string",
                        "description": "Detail loaded by `memory_read`, and in context every turn when `load` is \
                                        `inline`. Replaces the existing body whole; omit it to leave it \
                                        untouched, or pass an empty string to clear it."
                    },
                    "load": {
                        "type": "string",
                        "enum": ["inline", "index", "search"],
                        "description": format!(
                            "How the memory reaches your context: `inline` puts the body in every \
                             turn, for a rule you must follow without looking it up, within the {} \
                             all inline bodies share; `index` lists the description every turn; \
                             `search` leaves it to memory_search alone. Default: index. Omit to keep.",
                            crate::text::format_size(self.memories.inline_max_bytes())
                        )
                    },
                    "references": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What this memory depends on: other memory names, paths, URLs or \
                                        external ids, one line each, at most 20. memory_read shows what \
                                        references a memory, so a changed fact can be revisited wherever \
                                        it is cited. Omit to keep; pass [] to clear."
                    },
                    "source": {
                        "type": "string",
                        "description": "Where the fact came from, one line. Omit to keep; pass \"\" to clear."
                    },
                    "verified": {
                        "type": "boolean",
                        "default": false,
                        "description": "true when this write follows a check that the fact is current. The \
                                        only thing that moves the verification stamp memory_read reports."
                    },
                    "expires": {
                        "type": "string",
                        "description": "When the memory stops applying: RFC 3339, or YYYY-MM-DD for the whole \
                                        of that day. Past it the memory leaves the index and is found by \
                                        memory_search alone. Omit to keep; pass \"\" to clear."
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["global", "project"],
                        "description": "Tier the memory lives in. A new memory takes the session's project, or global \
                                        when it has none; an existing one keeps its tier unless the other is \
                                        named, which moves it."
                    }
                },
                "required": ["name"]
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
        let name = require_str(&input, "name", "memory_write")?;
        let name = name.as_str();
        memory::validate_memory_name(name).map_err(|message| MekaError::ToolExecution {
            tool_name: "memory_write".to_string(),
            message,
        })?;
        // Omit-to-keep, like `body`, `tags` and `priority`: the only copy of a description the
        // agent can see is the index's, elided to 500 characters, so requiring it would make a
        // body-only refinement rewrite a long description as its elision. The store resolves the
        // absence in SQL, and refuses it when the write would create the memory rather than update
        // one.
        let description = match optional_str(&input, "description", "memory_write")? {
            None => None,
            Some(text) => {
                // Whitespace is not the same question as "renders as nothing": format characters
                // are not whitespace, so three zero-width spaces would pass and render as blank in
                // the index the model reads every turn.
                if !memory::description_says_something(text) {
                    return Err(MekaError::ToolExecution {
                        tool_name: "memory_write".to_string(),
                        message: "'description' renders as nothing once formatting characters are \
                             stripped"
                            .to_string(),
                    });
                }
                Some(text.to_string())
            }
        };
        // An absent `body` means "leave it alone", not "make it empty": the schema marks the field
        // optional, so the call that changes only a priority is one the tool invites, and rendering
        // the absence as `""` would silently delete everything the memory said. `Some("")` is still
        // an explicit request to clear it.
        let body = optional_str_allowing_empty(&input, "body", "memory_write")?.map(str::to_string);
        // Same omit-to-keep rule as `body`, for the same reason: a call that changes only a
        // description would otherwise strip labels it never mentioned. `[]` still clears them.
        let tags: Option<Vec<String>> = string_or_list(&input, "tags", "memory_write")?;
        // Clamped rather than rejected, because this door's caller is a model rather than a
        // person: `meka memory add --priority 99` and `PUT /v1/memory` both refuse out of range,
        // and a human reads the error and retypes, where a model spends a turn on it. Read as
        // `i64` so a negative clamps to 0 instead of failing the `as_u64` cast outright.
        //
        // `None` means "leave it alone", which the upsert resolves in SQL: reading the absence as
        // the default would demote an inline standing directive every time the agent reworded
        // it.
        let priority = match input.get("priority") {
            Some(serde_json::Value::Null) | None => None,
            Some(value) => {
                let raw = value.as_i64().ok_or_else(|| MekaError::ToolExecution {
                    tool_name: "memory_write".to_string(),
                    message: format!("'priority' must be a whole number, got {value}"),
                })?;
                Some(memory::parse_priority(Some(raw), name))
            }
        };
        let load = match optional_str(&input, "load", "memory_write")? {
            None => None,
            Some(raw) => Some(
                raw.trim()
                    .parse::<memory::Load>()
                    .map_err(|message| tool_error("memory_write", message))?,
            ),
        };
        // The same omit-to-keep rule and the same refusal of a malformed list as `tags`.
        let references: Option<Vec<String>> =
            match string_or_list(&input, "references", "memory_write")? {
                None => None,
                Some(list) => Some(
                    memory::normalize_references(&list)
                        .map_err(|message| tool_error("memory_write", message))?,
                ),
            };
        let source = match optional_str(&input, "source", "memory_write")? {
            None => None,
            Some(text) => Some(
                memory::normalize_source(text)
                    .map_err(|message| tool_error("memory_write", message))?,
            ),
        };
        let verified = match input.get("verified") {
            None | Some(serde_json::Value::Null) => false,
            Some(serde_json::Value::Bool(verified)) => *verified,
            Some(value) => {
                return Err(tool_error(
                    "memory_write",
                    format!("'verified' must be true or false, got {value}"),
                ));
            }
        };
        // Three answers, like the store's field: absent keeps, `""` clears, anything else is a
        // moment the parser has to accept.
        let expires_at = match optional_str_allowing_empty(&input, "expires", "memory_write")? {
            None => None,
            Some(raw) if crate::text::is_blank(raw) => Some(None),
            Some(raw) => Some(Some(
                memory::parse_expiry(raw).map_err(|message| tool_error("memory_write", message))?,
            )),
        };
        // The tier, as the store takes it: `None` keeps a stored memory's tier and creates in
        // the session's own, an empty project is the global tier. `project` names the session's,
        // and a session on none has no such tier to name.
        let view = view_of(&self.project);
        let tier = parse_tier(&input, "memory_write")?;
        let project = match tier {
            None => None,
            Some(memory::Tier::Global) => Some(String::new()),
            Some(memory::Tier::Project) => match view.project() {
                "" => {
                    return Err(tool_error(
                        "memory_write",
                        "this session has no project; omit `tier` or use `global`",
                    ));
                }
                project => Some(project.to_string()),
            },
        };
        // The index lists a description every turn, so one that has grown into a paragraph is
        // spending the budget of several entries; said once, on the write that set it, and
        // never refused.
        let long_description = description
            .as_deref()
            .map(|description| description.chars().count())
            .filter(|length| *length > DESCRIPTION_NOTE_CHARS);

        // Held from here until after the write, so a second `memory_write` issued in the same turn
        // queues behind this one rather than racing it. Not for the write's sake (the upsert is
        // one transaction) but for the near-duplicate check below, which is a read.
        let _duplicate_guard = self.memories.lock_duplicate_check().await;

        // Normalized before the write so the doors agree on what a tag is: `Infra` and ` infra `
        // are the same label, and a duplicate would otherwise be stored twice in one column.
        let tags = match tags {
            Some(tags) => Some(
                memory::normalize_tags(&tags)
                    .map_err(|message| tool_error("memory_write", message))?,
            ),
            None => None,
        };

        // Read before the write, since the write is what creates the row, and kept whole: the
        // confirmation needs to know whether there was a body to keep rather than merely a row,
        // or it would claim to have kept the body of a memory that had none, and the change
        // reported to the frontend needs what the memory said.
        let existing = self
            .memories
            .get(name, &view_of(&self.project))
            .await
            .map_err(|error| tool_error("memory_write", error))?;
        let kept_existing_body = body.is_none()
            && existing
                .as_ref()
                .and_then(|existing| existing.body.as_deref())
                .is_some_and(|existing| !existing.trim().is_empty());

        // Also before the write, so the incoming description is compared against the store as it
        // was rather than against itself, and only when one was given: checking the stored
        // description would report every omit-to-keep update as a duplicate of itself.
        let duplicate = match description.as_deref() {
            Some(description) => self.near_duplicate_of(name, description).await,
            None => None,
        };

        // One statement, one transaction, and no `flock`: two writes to one name are two upserts,
        // and SQLite serializes them. Omit-to-keep is in the SQL, so there is no read here to go
        // stale between the check above and this.
        let written = self
            .memories
            .write(WriteRequest {
                name: name.to_string(),
                view: view.clone(),
                project,
                // Normalized to one line here rather than only at `PUT /v1/memory`, so all three
                // write doors store what `meka memory export` writes and a round trip is
                // byte-exact.
                description: description
                    .as_deref()
                    .map(crate::entry::normalize_description),
                tags,
                body,
                priority,
                load,
                references,
                source,
                expires_at,
                verified,
            })
            .await
            .map_err(|error| tool_error("memory_write", error))?;

        let name = &written.name;
        tracing::info!("saved memory '{name}'");
        // Both sides rendered as the export writes them, so a reworded description or a moved
        // priority shows as one changed frontmatter line and a rewritten body as the body.
        let change = crate::frontend::ToolOutputMetadata::Diff {
            path: None,
            old_text: existing.as_ref().map(memory::export_memory),
            new_text: Some(memory::export_memory(&written)),
        };
        let now = std::time::SystemTime::now();
        let mut text = format!(
            "Saved memory '{}' (priority {}{}{}){}.",
            written.name,
            // What landed, not what was asked for: an omitted priority is resolved from the
            // stored row, so echoing the argument would report a number the store does not
            // carry.
            written.priority,
            // The load only when it is not the default, the way the export writes it.
            if written.load == memory::Load::default() {
                String::new()
            } else {
                format!(", {}", written.load)
            },
            // The tier only when it is not the session's own: a global note from a project
            // session is one every project now reads.
            if written.project.is_empty() && !view.project().is_empty() {
                ", global"
            } else {
                ""
            },
            // Stated so the two calls are distinguishable from the result alone: a metadata
            // update and a rewrite otherwise report identically, and the difference is the
            // whole body of the note.
            if kept_existing_body {
                ", keeping the existing body"
            } else {
                ""
            },
        );
        if let Some(length) = long_description {
            text.push_str(&format!(
                "\n\nNote: the description is {length} characters and the index lists it every \
                 turn; keep the fact there and the detail in the body."
            ));
        }
        if written.is_expired(now) {
            text.push_str(&format!(
                "\n\nNote: this memory {}, so it is reachable through memory_search alone; pass \
                 `expires: \"\"` to clear the expiry.",
                memory::render_expiry(written.expires_at.unwrap_or(now), now)
            ));
        }
        if let Some(existing) = duplicate {
            text.push_str(&format!(
                "\n\nNote: '{existing}' already says something very similar. If this is the \
                 same fact, call memory_write on '{existing}' instead and delete '{name}'. Two \
                 near-copies both stay in the index for ever and neither supersedes the other."
            ));
        }
        Ok(ToolOutput::text(text, false).with_metadata(change))
    }
}

impl MemoryWriteTool {
    /// The name of an existing memory whose description says close to the same thing, if any.
    ///
    /// Advisory. It never blocks and never rewrites: this is the agent's decision to make, and the
    /// only failure worth preventing is the silent one where a store accumulates a hundred
    /// near-copies because nothing ever pointed out the ninety-nine.
    async fn near_duplicate_of(&self, name: &str, description: &str) -> Option<String> {
        // The cheap disqualifier first: a two-word description shares words with everything.
        let incoming = Terms::parse(&[description.to_string()]);
        if incoming.words().len() < DUPLICATE_MIN_TERMS {
            return None;
        }

        // Within the session's view: a near-copy in another project is not one this session
        // can see, read or delete.
        let hits = self
            .memories
            .search(
                incoming.match_expression(),
                &crate::store::memory::MemoryFilter {
                    view: view_of(&self.project),
                    ..Default::default()
                },
                5,
            )
            .await
            .inspect_err(|error| {
                tracing::debug!("duplicate check skipped: {error}");
            })
            .ok()?
            .hits;

        for hit in hits {
            // Rewriting a memory under its own name is an update, not a duplicate. Compared
            // case-insensitively because the column is: writing `POLICY` over an existing `policy`
            // updates that one row, and a case-sensitive check would report the row it had just
            // written as a near-copy and tell the model to `memory_delete` it.
            if hit.name.eq_ignore_ascii_case(name) {
                continue;
            }
            let existing = Terms::parse(std::slice::from_ref(&hit.description));
            let shared = incoming
                .words()
                .iter()
                .filter(|word| existing.words().contains(word))
                .count();
            let overlap = shared as f64 / incoming.words().len() as f64;
            if overlap >= DUPLICATE_TERM_OVERLAP {
                return Some(hit.name);
            }
        }
        None
    }
}

pub(super) struct MemoryEditTool {
    pub(crate) memories: Arc<MemoryStore>,
    /// The session's project, read per call so a move mid-session changes what the next call
    /// sees.
    pub(crate) project: crate::session::SharedProject,
}

#[async_trait]
impl Tool for MemoryEditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_edit".to_string(),
            description: "Replace an exact string in a memory's body and leave the rest as it is. Refused when `old_string` matches more than once unless `replace_all` is true. For a small change; `memory_write` replaces the whole body.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the memory, as listed in the memory index."
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find in the body."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement; an empty string deletes `old_string`."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "default": false,
                        "description": "Change every occurrence. When false and `old_string` matches more than once, the edit is refused as ambiguous. Default: false."
                    },
                    "verified": {
                        "type": "boolean",
                        "default": false,
                        "description": "true when this edit follows a check that the fact is current; moves the verification stamp."
                    }
                },
                "required": ["name", "old_string", "new_string"]
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
        let name = require_str(&input, "name", "memory_edit")?;
        let name = name.as_str();
        let old_string = require_str(&input, "old_string", "memory_edit")?;
        let new_string =
            super::util::require_str_allowing_empty(&input, "new_string", "memory_edit")?;
        let replace_all = match input.get("replace_all") {
            None | Some(serde_json::Value::Null) => false,
            Some(serde_json::Value::Bool(replace_all)) => *replace_all,
            Some(value) => {
                return Err(tool_error(
                    "memory_edit",
                    format!("'replace_all' must be true or false, got {value}"),
                ));
            }
        };
        let verified = match input.get("verified") {
            None | Some(serde_json::Value::Null) => false,
            Some(serde_json::Value::Bool(verified)) => *verified,
            Some(value) => {
                return Err(tool_error(
                    "memory_edit",
                    format!("'verified' must be true or false, got {value}"),
                ));
            }
        };
        memory::validate_memory_lookup(name)
            .map_err(|message| tool_error("memory_edit", message))?;

        let Some(existing) = self
            .memories
            .get(name, &view_of(&self.project))
            .await
            .map_err(|error| tool_error("memory_edit", error))?
        else {
            let index = self
                .memories
                .index(&view_of(&self.project))
                .await
                .map_err(|error| tool_error("memory_edit", error))?;
            return Err(tool_error(
                "memory_edit",
                format!(
                    "no memory named '{}'.{}",
                    name,
                    crate::tools::did_you_mean_hint(
                        name,
                        index.iter().map(|entry| entry.name.as_str())
                    )
                ),
            ));
        };
        let body = existing.body.clone().unwrap_or_default();
        let (edited, count, first_match_byte) =
            match super::util::replace_anchored(&body, &old_string, &new_string, replace_all) {
                Ok(applied) => applied,
                Err(super::util::AnchorMiss::NotFound) => {
                    return Ok(ToolOutput::text(
                        format!(
                            "Error: '{}' not found in the body of memory '{}'",
                            super::util::truncate_string(&old_string, 100),
                            existing.name
                        ),
                        true,
                    ));
                }
                Err(super::util::AnchorMiss::Ambiguous(count)) => {
                    return Ok(ToolOutput::text(
                        format!(
                            "Error: '{}' matches {count} times in memory '{}'; add surrounding \
                             context to make old_string unique, or set replace_all=true to \
                             change every occurrence",
                            super::util::truncate_string(&old_string, 100),
                            existing.name
                        ),
                        true,
                    ));
                }
            };

        // A compare-and-swap on the body that was read, so an edit computed against stale text
        // is refused rather than landing over somebody else's write: the same door the
        // `$EDITOR` round trip takes, for the same reason.
        let written = match self
            .memories
            .write_body(
                &existing.name,
                &view_of(&self.project),
                &body,
                edited.clone(),
                verified,
            )
            .await
            .map_err(|error| tool_error("memory_edit", error))?
        {
            crate::store::memory::BodyWrite::Saved => self
                .memories
                .get(&existing.name, &view_of(&self.project))
                .await
                .map_err(|error| tool_error("memory_edit", error))?,
            crate::store::memory::BodyWrite::Gone => None,
            crate::store::memory::BodyWrite::ChangedUnderneath => {
                return Ok(ToolOutput::text(
                    format!(
                        "Error: memory '{}' changed while the edit was computed; read it again \
                         and retry",
                        existing.name
                    ),
                    true,
                ));
            }
        };
        let Some(written) = written else {
            return Err(tool_error(
                "memory_edit",
                format!(
                    "memory '{}' was deleted while the edit was computed",
                    existing.name
                ),
            ));
        };

        let name = &written.name;
        tracing::info!("edited memory '{name}'");
        let snippet = super::file::build_context_snippet(&edited, first_match_byte, 3);
        Ok(ToolOutput::text(
            format!(
                "Edited memory '{}': {count} occurrence(s){}.\n\n{}",
                written.name,
                if verified { ", verified" } else { "" },
                snippet
            ),
            false,
        )
        .with_metadata(crate::frontend::ToolOutputMetadata::Diff {
            path: None,
            old_text: Some(memory::export_memory(&existing)),
            new_text: Some(memory::export_memory(&written)),
        }))
    }
}

pub(super) struct MemoryReadTool {
    pub(crate) memories: Arc<MemoryStore>,
    /// The session's project, read per call so a move mid-session changes what the next call
    /// sees.
    pub(crate) project: crate::session::SharedProject,
}

#[async_trait]
impl Tool for MemoryReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_read".to_string(),
            description: format!(
                "Read a saved memory and up to {READ_BODY_MAX_CHARS} body characters, with its source, when it was last verified, what it references and what references it. Use the returned continuation offset to read a longer body. Memory is recorded context, not live state."
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the memory, as listed in the memory index."
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "0-based body character offset. Use the continuation offset from a previous read."
                    }
                },
                "required": ["name"]
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
        let name = require_str(&input, "name", "memory_read")?;
        let offset = match input.get("offset") {
            None | Some(serde_json::Value::Null) => 0,
            Some(value) => value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    tool_error(
                        "memory_read",
                        format!("'offset' must be a whole number, got {value}"),
                    )
                })?,
        };
        let name = name.as_str();
        // Validated here as at every other door: an invalid name would otherwise be reported as
        // merely absent.
        memory::validate_memory_lookup(name).map_err(|message| MekaError::ToolExecution {
            tool_name: "memory_read".to_string(),
            message,
        })?;
        let entry = match self
            .memories
            .get(name, &view_of(&self.project))
            .await
            .map_err(|error| tool_error("memory_read", error))?
        {
            Some(entry) => entry,
            None => {
                // A half-remembered name is the common miss at scale, and on its own it ends the
                // line. Pointing at the near-miss costs nothing and is the same recovery an unknown
                // *tool* name already gets.
                let index = self
                    .memories
                    .index(&view_of(&self.project))
                    .await
                    .map_err(|error| tool_error("memory_read", error))?;
                return Err(tool_error(
                    "memory_read",
                    format!(
                        "no memory named '{}'.{}",
                        name,
                        crate::tools::did_you_mean_hint(
                            name,
                            index.iter().map(|entry| entry.name.as_str())
                        )
                    ),
                ));
            }
        };

        let body = memory::render_for_model(entry.body.as_deref().unwrap_or_default());
        let body = body.trim();
        let length = body.chars().count();
        if offset > length {
            return Err(tool_error(
                "memory_read",
                format!("'offset' is past the body's {length} characters"),
            ));
        }

        // Counted here and nowhere else. A search hit is weaker evidence (the model saw a line,
        // not the note) and an operator reading through the HTTP API is not the agent recalling
        // anything, so neither moves the ranking the agent gets. Best-effort: a counter that fails
        // to increment must not fail the read.
        if let Err(error) = self
            .memories
            .record_read(&entry.name, &view_of(&self.project))
            .await
        {
            let name = &entry.name;
            tracing::warn!("failed to record a read of memory '{name}': {error}");
        }

        // Age and verification are stated on the way out, not just in the index: a memory is a
        // point-in-time observation, and detail that was true months ago is exactly what gets
        // asserted as current fact without a nudge.
        let now = std::time::SystemTime::now();
        let caption = memory::render_record_caption(&entry, now);
        let mut facts = String::new();
        if !entry.project.is_empty() {
            facts.push_str(&format!("\nProject: {}", entry.project));
        }
        if !entry.source.is_empty() {
            facts.push_str(&format!(
                "\nSource: {}",
                memory::render_description_for_model(&entry.source)
            ));
        }
        if let Some(expires) = entry.expires_at {
            let expiry = memory::render_expiry(expires, now);
            facts.push_str(&format!("\n{}{}", expiry[..1].to_uppercase(), &expiry[1..]));
            if entry.is_expired(now) {
                facts.push_str("; listed nowhere, reached by search alone.");
            } else {
                facts.push('.');
            }
        }
        // Resolved on the way out: a reference that names a memory shows that memory's
        // description beside it, and one that names nothing is shown as written, which is the
        // "worth writing later" marker and the typo both.
        if !entry.references.is_empty() {
            let mut rendered: Vec<String> = Vec::with_capacity(entry.references.len());
            for reference in &entry.references {
                let shown = memory::render_description_for_model(reference);
                match self
                    .memories
                    .get(reference, &view_of(&self.project))
                    .await
                    .map_err(|error| tool_error("memory_read", error))?
                {
                    Some(target) => rendered.push(format!(
                        "{shown} ({})",
                        crate::entry::elide_description_for_index(
                            &memory::render_description_for_model(&target.description)
                        )
                    )),
                    None => rendered.push(shown),
                }
            }
            facts.push_str(&format!("\nReferences: {}", rendered.join(", ")));
        }
        let backlinks = self
            .memories
            .referenced_by(&entry.name, &view_of(&self.project))
            .await
            .map_err(|error| tool_error("memory_read", error))?;
        if !backlinks.is_empty() {
            facts.push_str(&format!(
                "\nReferenced by: {}",
                render_backlinks(&entry.name, &backlinks)
            ));
        }
        let end = offset.saturating_add(READ_BODY_MAX_CHARS).min(length);
        let page: String = body.chars().skip(offset).take(end - offset).collect();
        let rendered = if length == 0 {
            "(no body: this memory's description is all of it)".to_string()
        } else if end < length {
            format!(
                "{page}\n\n[Body truncated: characters {offset}..{end} of {length}. Continue with `offset: {end}`.]"
            )
        } else if offset == length {
            "(end of body)".to_string()
        } else {
            page
        };
        Ok(ToolOutput::text(
            format!(
                "# {}\n\n{}\n\n{caption} This is what you recorded then, not live state; verify \
                 before relying on it.{facts}\n\n{}",
                entry.name,
                memory::render_description_for_model(&entry.description),
                rendered
            ),
            false,
        ))
    }
}

/// Default number of ranked entries `memory_search` returns.
///
/// An order of magnitude below the shared `MAX_SEARCH_MATCHES`, which sized a list of grep *lines*.
/// An entry here carries a description, an excerpt and often a whole short body, so ten of them is
/// already a substantial read and a hundred would crowd out the turn that asked for them.
const DEFAULT_SEARCH_LIMIT: usize = 10;
use crate::store::memory::MAX_SEARCH_LIMIT;

/// A hit whose body is at most this long is shown in full instead of an excerpt.
///
/// Most memories are shorter than this, which is the point: the common recall becomes one round
/// trip rather than a search followed by a `memory_read` per hit.
const INLINE_BODY_MAX_CHARS: usize = 800;

/// Ceiling on a body `memory_read` renders.
///
/// Generous, because this call is how the agent deliberately spends context on one note and a
/// bound that fights that is worse than none; but both search tiers excerpt so that this is the
/// considered way to load a note in full, and a 200 KB body reaching the window through it would
/// undo the arrangement.
const READ_BODY_MAX_CHARS: usize = 16_000;

/// Ceiling on the whole rendered result, inlined bodies included.
const SEARCH_RESULT_MAX_BYTES: usize = 6_144;

/// Past this many characters a written description earns a note: the index lists it every turn,
/// and the render cap sits at 500, so one this long is spending the budget of several entries.
const DESCRIPTION_NOTE_CHARS: usize = 200;

/// How many memories a read or a delete names as referencing the one at hand before pointing at
/// the filter instead.
const BACKLINKS_SHOWN_MAX: usize = 20;

/// Fraction of an incoming description's words that must already appear in an existing memory's
/// description before the write says so.
///
/// Term overlap rather than a `bm25` threshold, because bm25 is a corpus statistic: the same pair
/// of descriptions scores differently in a store of ten and a store of ten thousand, so a fixed
/// cut-off would fire constantly in one and never in the other. Overlap means the same thing at
/// every size, and it is the thing that can be explained in the message.
const DUPLICATE_TERM_OVERLAP: f64 = 0.6;

/// Below this many words an overlap ratio is noise: two three-word descriptions sharing two words
/// are usually unrelated notes that both mention the same noun.
const DUPLICATE_MIN_TERMS: usize = 3;

/// Which tier answered, so the result can say so.
///
/// Stated in the output rather than kept internal: a prefix or edit-distance answer is a *guess*
/// about what the caller meant, and a model that cannot tell it apart from an exact hit will
/// report the guess as a recalled fact.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    Exact,
    Prefix,
    Substring,
    Fuzzy,
}

impl From<crate::store::memory::SearchTier> for Tier {
    fn from(tier: crate::store::memory::SearchTier) -> Self {
        match tier {
            crate::store::memory::SearchTier::Exact => Self::Exact,
            crate::store::memory::SearchTier::Prefix => Self::Prefix,
            crate::store::memory::SearchTier::Substring => Self::Substring,
        }
    }
}

impl Tier {
    fn preamble(self) -> &'static str {
        match self {
            Tier::Exact => "",
            Tier::Prefix => {
                "No exact matches. These are prefix matches, so treat them as \
                             near-misses rather than what you asked for.\n\n"
            }
            Tier::Substring => {
                "No word matches. These contain what you asked for as a literal \
                             substring, which is how text the word splitter does not divide \
                             (Chinese, Japanese, Thai, an identifier, a serial number) is found.\n\n"
            }
            Tier::Fuzzy => {
                "No full-text matches. These are the closest memory names and \
                            descriptions by spelling, so they may be unrelated.\n\n"
            }
        }
    }
}

pub(super) struct MemorySearchTool {
    pub(crate) memories: Arc<MemoryStore>,
    /// The session's project, read per call so a move mid-session changes what the next call
    /// sees.
    pub(crate) project: crate::session::SharedProject,
}

impl MemorySearchTool {
    /// The last-resort tier: rank by how close a query term comes to a word in each memory's name
    /// or description. Runs only when full-text matching found nothing at all, so its cost is paid
    /// on a query that was otherwise going to return an empty result.
    ///
    /// Names and descriptions only, because that is all the index carries. Substring matching,
    /// which needs the body too, is [`crate::store::memory::MemoryStore::substring_search`] and
    /// runs as its own tier ahead of this one.
    fn fuzzy_by_spelling(
        index: &[memory::Memory],
        terms: &[String],
        limit: usize,
    ) -> (Vec<(usize, memory::Memory)>, usize) {
        let mut scored: Vec<(usize, memory::Memory)> = Vec::new();
        for entry in index.iter() {
            let haystack = format!("{} {}", entry.name, entry.description).to_lowercase();
            let mut best = usize::MAX;
            for term in terms {
                let threshold = crate::tools::fuzzy_threshold(term);
                // Hoisted: invariant across the word loop, and the pre-filter below exists to be
                // O(1) per pair.
                let term_chars = term.chars().count();
                for word in haystack.split(|c: char| !c.is_alphanumeric() && c != '_') {
                    if word.is_empty() {
                        continue;
                    }
                    // Two strings whose lengths differ by more than the threshold cannot be
                    // within it, so this rejects them without building the matrix: a pasted blob
                    // (a stack trace, base64, a URL) is exactly what reaches this code, and a
                    // matrix per stored name would run for minutes on a runtime worker.
                    //
                    // Counted in characters, because `fuzzy_threshold` and `edit_distance` both
                    // are; `len()` measures bytes, which is the same number only for ASCII, and
                    // `東京都` against a stored `東京` is one edit and three bytes apart.
                    if term_chars.abs_diff(word.chars().count()) > threshold {
                        continue;
                    }
                    let distance = crate::tools::edit_distance(term, word);
                    if distance <= threshold {
                        best = best.min(distance);
                    }
                }
            }
            if best != usize::MAX {
                scored.push((best, entry.clone()));
            }
        }
        scored.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.name.cmp(&right.1.name))
        });
        // Returned alongside the cut list, because everything past `limit` is something the caller
        // is not being shown, and it matters more here than in the ranked tier: the tie-break is
        // alphabetical and forty candidates can sit at one distance.
        let candidates = scored.len();
        scored.truncate(limit);
        (scored, candidates)
    }
}

#[async_trait]
impl Tool for MemorySearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_search".to_string(),
            description: "Search saved memories, including those the context index leaves out. Supply alternative phrasings together in `queries`; matching is case-insensitive and handles word endings. `referencing`, `unverified` and `expired` narrow a search, or list memories on their own when no query is given.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more phrasings of what you are looking for. Include synonyms when you do not remember the wording. Optional when a filter is given."
                    },
                    "referencing": {
                        "type": "string",
                        "description": "Only memories whose references name this: a memory name, path, URL or id, matched whole."
                    },
                    "unverified": {
                        "type": "string",
                        "description": "Only memories not verified within this long, e.g. '90d' or '6months'."
                    },
                    "expired": {
                        "type": "boolean",
                        "default": false,
                        "description": "Only memories whose expiry has passed."
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["global", "project"],
                        "description": "Only one tier: global, or the session's project. Default: both the session sees."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_LIMIT,
                        "default": DEFAULT_SEARCH_LIMIT,
                        "description": format!("Maximum memories to return. Default: {DEFAULT_SEARCH_LIMIT}.")
                    }
                },
                "required": []
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
        // "Missing" is for a parameter that is not there; a parameter that is there but yields
        // nothing searchable is a different mistake with a different fix, and reporting it as
        // missing sends the caller looking for a bug in how it built the call.
        let queries: Option<Vec<String>> = string_or_list(&input, "queries", "memory_search")?;
        let session_view = view_of(&self.project);
        let tier = parse_tier(&input, "memory_search")?;
        let view = match tier {
            None => session_view.clone(),
            Some(memory::Tier::Global) => MemoryView::Global,
            Some(memory::Tier::Project) => match session_view.project() {
                "" => {
                    return Err(tool_error(
                        "memory_search",
                        "this session has no project; omit `tier` or use `global`",
                    ));
                }
                project => MemoryView::Project(project.to_string()),
            },
        };
        // A global memory is marked as such where the session is on a project, since which tier a
        // hit is in decides where a correction to it goes.
        let mark_global = !session_view.project().is_empty();
        let filter = crate::store::memory::MemoryFilter {
            view,
            referencing: optional_str(&input, "referencing", "memory_search")?
                .map(|target| target.trim().to_string()),
            unverified_for: match optional_str(&input, "unverified", "memory_search")? {
                None => None,
                Some(window) => {
                    Some(memory::parse_unverified_window(window).map_err(|message| {
                        tool_error("memory_search", format!("'unverified': {message}"))
                    })?)
                }
            },
            expired: match input.get("expired") {
                None | Some(serde_json::Value::Null) => false,
                Some(serde_json::Value::Bool(expired)) => *expired,
                Some(value) => {
                    return Err(tool_error(
                        "memory_search",
                        format!("'expired' must be true or false, got {value}"),
                    ));
                }
            },
        };
        let queries = queries.unwrap_or_default();
        if queries.is_empty() && filter.is_empty() && tier.is_none() {
            return Err(MekaError::ToolExecution {
                tool_name: "memory_search".to_string(),
                message: "give 'queries', or one of 'referencing', 'unverified', 'expired' and \
                          'tier'"
                    .to_string(),
            });
        }
        let described = describe_selection(&filter, tier);
        // Coerced rather than ignored: `as_u64` alone answers `None` for `"3"`, `3.0` and `-1`
        // alike, and a silent default would give ten results to a call that asked for three. The
        // same handler accepts a bare string where `queries` declares an array.
        let limit = match input.get("limit") {
            None | Some(serde_json::Value::Null) => DEFAULT_SEARCH_LIMIT,
            Some(value) => {
                let number = value
                    .as_i64()
                    .or_else(|| value.as_f64().map(|value| value as i64))
                    .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
                    .ok_or_else(|| MekaError::ToolExecution {
                        tool_name: "memory_search".to_string(),
                        message: format!("'limit' must be a whole number, got {value}"),
                    })?;
                number.clamp(1, MAX_SEARCH_LIMIT as i64) as usize
            }
        };

        let store = self.memories.as_ref();
        let now = std::time::SystemTime::now();
        // A filter and no words is a listing, in index order: every row matched the filter
        // equally, so there is no relevance to rank by.
        if queries.is_empty() {
            let results = store
                .filtered(&filter, limit)
                .await
                .map_err(|error| tool_error("memory_search", error))?;
            if results.hits.is_empty() {
                return Ok(ToolOutput::text(
                    format!("No memories {}.", described.as_deref().unwrap_or_default()),
                    false,
                ));
            }
            return Ok(ToolOutput::text(
                render_hits(
                    Tier::Exact,
                    ScannedTerms {
                        found: &[],
                        missed: &[],
                        beside_words: false,
                    },
                    &results,
                    limit,
                    now,
                    described.as_deref(),
                    mark_global,
                ),
                false,
            ));
        }
        let Some(found) = store
            .find(&queries, &filter, limit)
            .await
            .map_err(|error| tool_error("memory_search", error))?
        else {
            return Err(MekaError::ToolExecution {
                tool_name: "memory_search".to_string(),
                message: "'queries' contained no searchable words; try different wording"
                    .to_string(),
            });
        };
        let crate::store::memory::Found {
            tier,
            results,
            scanned,
            missed,
            beside_words,
            words,
        } = found;
        let tier = Tier::from(tier);

        if !results.hits.is_empty() {
            return Ok(ToolOutput::text(
                render_hits(
                    tier,
                    ScannedTerms {
                        found: &scanned,
                        missed: &missed,
                        beside_words,
                    },
                    &results,
                    limit,
                    now,
                    described.as_deref(),
                    mark_global,
                ),
                false,
            ));
        }

        // Only now is the whole listing loaded: the tiers above answer from SQL, and this one is
        // the last resort that has to scan. The listing rather than the index, so a memory the
        // index leaves out is still found by its spelling, under the same filter.
        let listing = self
            .memories
            .list(&filter)
            .await
            .map_err(|error| tool_error("memory_search", error))?;
        let (fuzzy, candidates) = Self::fuzzy_by_spelling(&listing, &words, limit);
        if fuzzy.is_empty() {
            return Ok(ToolOutput::text(
                format!(
                    "No memories matched {queries:?}{}. Try broader words, or several phrasings \
                     of the same idea in one call.",
                    if filter.is_empty() {
                        String::new()
                    } else {
                        format!(" {}", describe_filter(&filter))
                    }
                ),
                false,
            ));
        }
        Ok(ToolOutput::text(
            render_fuzzy(&fuzzy, candidates, now),
            false,
        ))
    }
}

/// What the substring scan did for the terms the index cannot match, for the result to state:
/// the model reads which part of its query was answered how, and never that a term matched when
/// no memory holds it.
struct ScannedTerms<'a> {
    /// Terms some memory holds as a literal substring.
    found: &'a [String],
    /// Terms no memory holds.
    missed: &'a [String],
    /// Whether the rest of the query was matched through the index, so the note can say so.
    beside_words: bool,
}

fn quoted_list(terms: &[String]) -> String {
    terms
        .iter()
        .map(|term| format!("'{term}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The filter and the tier a search named, as one phrase for its preamble, or `None` when the
/// call named neither and the hits need no qualifying.
fn describe_selection(
    filter: &crate::store::memory::MemoryFilter,
    tier: Option<memory::Tier>,
) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if !filter.is_empty() {
        parts.push(describe_filter(filter));
    }
    match (tier, &filter.view) {
        (Some(memory::Tier::Global), _) => parts.push("in the global tier".to_string()),
        (Some(memory::Tier::Project), MemoryView::Project(project)) => {
            parts.push(format!("in project '{project}'"));
        }
        _ => {}
    }
    (!parts.is_empty()).then(|| parts.join(", "))
}

/// Parse the optional `tier` of a memory tool call.
fn parse_tier(input: &serde_json::Value, tool: &str) -> Result<Option<memory::Tier>> {
    optional_str(input, "tier", tool)?
        .map(|raw| {
            raw.trim()
                .parse::<memory::Tier>()
                .map_err(|message| tool_error(tool, message))
        })
        .transpose()
}

/// What a filter selected, for a header or a "nothing" line: "referencing 'x'", "not verified
/// in 90 days", "expired", joined with "and".
fn describe_filter(filter: &crate::store::memory::MemoryFilter) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(target) = &filter.referencing {
        parts.push(format!("referencing '{target}'"));
    }
    if let Some(window) = filter.unverified_for {
        parts.push(format!(
            "not verified in {}",
            crate::text::format_duration_words(window)
        ));
    }
    if filter.expired {
        parts.push("expired".to_string());
    }
    parts.join(" and ")
}

/// The memories that reference one, capped so a hub does not spend a read's whole result on its
/// backlinks; the filter reaches the rest.
fn render_backlinks(name: &str, backlinks: &[(String, String)]) -> String {
    let shown: Vec<String> = backlinks
        .iter()
        .take(BACKLINKS_SHOWN_MAX)
        .map(|(name, description)| {
            format!(
                "{} ({})",
                memory::render_description_for_model(name),
                crate::entry::elide_description_for_index(&memory::render_description_for_model(
                    description
                ))
            )
        })
        .collect();
    let more = backlinks.len().saturating_sub(BACKLINKS_SHOWN_MAX);
    if more > 0 {
        format!(
            "{}, and {more} more; `memory_search` with `referencing: '{name}'` lists them",
            shown.join(", ")
        )
    } else {
        shown.join(", ")
    }
}

/// Render ranked hits: one entry per memory, with its body inlined when short enough. With a
/// `filter`, the header names it; with no words behind the result, the order is the index's.
fn render_hits(
    tier: Tier,
    scanned: ScannedTerms<'_>,
    results: &SearchResults,
    limit: usize,
    now: std::time::SystemTime,
    described: Option<&str>,
    mark_global: bool,
) -> String {
    let hits = &results.hits;
    let mut out = String::from(tier.preamble());
    if !scanned.found.is_empty() {
        out.push_str(&format!(
            "{} matched as a literal substring, which is how text the word splitter does not \
             divide is found{}.\n\n",
            quoted_list(scanned.found),
            if scanned.beside_words {
                "; the other terms matched through the index"
            } else {
                ""
            }
        ));
    }
    if !scanned.missed.is_empty() {
        out.push_str(&format!(
            "No memory holds {}.\n\n",
            quoted_list(scanned.missed)
        ));
    }
    // The number of *matches*, not the number rendered. Reporting the truncated length as the
    // total reads as "this is everything that matched", which is how a full store becomes a
    // confidently incomplete answer.
    // A listing's hits carry no score, which is what tells it from a search here: "3 memories
    // referencing 'x', most important first" against "3 matching memories referencing 'x', most
    // relevant first".
    let listing = hits.iter().all(|hit| hit.score == 0.0);
    out.push_str(&format!(
        "{}{} {}{}{}, {}:\n\n",
        if results.pool_exhausted {
            "at least "
        } else {
            ""
        },
        results.matched,
        if listing { "" } else { "matching " },
        if results.matched == 1 {
            "memory"
        } else {
            "memories"
        },
        match described {
            Some(described) => format!(" {described}"),
            None => String::new(),
        },
        if listing {
            "most important first"
        } else {
            "most relevant first"
        }
    ));
    let mut shown = 0;
    let mut budget_bound = false;
    for hit in hits {
        let mut entry = String::new();
        entry.push_str(&format!(
            "- **{}** (p{}{}, created {}{}{}, read {}x)\n",
            hit.name,
            hit.priority,
            if mark_global && hit.project.is_empty() {
                ", global"
            } else {
                ""
            },
            memory::render_age(hit.created, now),
            // The verification only when there was one, as `memory_read`'s caption says it.
            if hit.verified != hit.created {
                format!(", verified {}", memory::render_age(hit.verified, now))
            } else {
                String::new()
            },
            match hit.expires {
                Some(expires) => format!(", {}", memory::render_expiry(expires, now)),
                None => String::new(),
            },
            hit.read_count
        ));
        // Elided, like every other rendered description: descriptions are unbounded at parse time,
        // and the always-emit-the-first-entry rule below would otherwise let one memory spend the
        // whole turn's budget on its description alone.
        entry.push_str(&format!(
            "  {}\n",
            crate::entry::elide_description_for_index(&memory::render_description_for_model(
                &hit.description
            ))
        ));
        let body = hit.body.trim();
        if !body.is_empty() {
            // The whole body when it is short, which most are: that turns the common recall into
            // one call instead of a search plus a `memory_read` per hit. Otherwise the excerpt
            // around the match, which is what a follow-up read is for.
            //
            // The excerpt is clipped as well as excerpted: `snippet()` bounds by tokens, not
            // bytes, and `unicode61` makes a whole CJK paragraph, a base64 blob or one unbroken
            // path a single token, so the "excerpt" can be the entire column.
            let rendered = if body.chars().count() <= INLINE_BODY_MAX_CHARS {
                memory::render_for_model(body)
            } else {
                format!(
                    "… {} …",
                    memory::render_for_model(&crate::prompt::clip_chars(
                        &hit.snippet,
                        INLINE_BODY_MAX_CHARS
                    ))
                )
            };
            for line in rendered.lines() {
                entry.push_str(&format!("  {line}\n"));
            }
        }
        entry.push('\n');

        // Always emit the first entry: one pathological memory longer than the budget should still
        // be visible rather than collapsing the result to a bare count.
        if shown > 0 && out.len() + entry.len() > SEARCH_RESULT_MAX_BYTES {
            budget_bound = true;
            break;
        }
        out.push_str(&entry);
        shown += 1;
    }

    // Against `matched`, not `hits.len()`: entries dropped by `limit` and entries dropped by the
    // byte budget are both things the caller is not seeing, and counting only the latter would
    // understate the remainder by exactly the amount `limit` removed.
    let hidden = results.matched.saturating_sub(shown);
    if hidden > 0 {
        // Hedged for the same reason the header is: `matched` is capped at the candidate pool, so
        // once the pool is full this number is a floor, not a count.
        //
        // The remedy names the constraint that bound, and only one that can work: a `limit` above
        // the maximum is clamped on the way in, so "raise `limit`" at the ceiling, or when the
        // byte budget did the cutting, returns the identical result.
        out.push_str(&format!(
            "{}{hidden} further match(es) not shown here{}\n",
            if results.pool_exhausted {
                "at least "
            } else {
                ""
            },
            if budget_bound {
                "; the result hit its size limit, so narrow the query."
            } else if limit >= MAX_SEARCH_LIMIT {
                "; `limit` is already at its maximum, so narrow the query."
            } else {
                "; raise `limit` or narrow the query."
            }
        ));
    }
    out.push_str("Call `memory_read` for the full text of any of these.");
    out
}

/// Render the spelling-distance tier. These carry no body: they are candidates, not answers, and
/// padding a guess with content invites it to be read as one.
fn render_fuzzy(
    scored: &[(usize, memory::Memory)],
    candidates: usize,
    now: std::time::SystemTime,
) -> String {
    let mut out = String::from(Tier::Fuzzy.preamble());
    let mut rendered_count = 0;
    for (shown, (_, entry)) in scored.iter().enumerate() {
        // Budgeted like `render_hits`, and elided like every other index line: descriptions are
        // not bounded at parse time, and this is the tier that is explicitly "candidates, not
        // answers".
        let line = format!(
            "- **{}** (p{}, created {}, read {}x): {}\n",
            entry.name,
            entry.priority,
            memory::render_age(entry.created_at, now),
            entry.read_count,
            crate::entry::elide_description_for_index(&memory::render_description_for_model(
                &entry.description
            ))
        );
        if shown > 0 && out.len() + line.len() > SEARCH_RESULT_MAX_BYTES {
            break;
        }
        out.push_str(&line);
        rendered_count += 1;
    }
    // This tier truncates in two places (`limit` in `fuzzy_by_spelling` and the byte budget
    // above), and the tie-break past `limit` is alphabetical, which is arbitrary enough that not
    // saying so is the difference between a shortlist and a wrong answer.
    let hidden = candidates.saturating_sub(rendered_count);
    if hidden > 0 {
        out.push_str(&format!(
            "\n{hidden} further candidate(s) scored the same or worse and are not shown.\n"
        ));
    }
    out.push_str("\nCall `memory_read` to see whether one of these is what you meant.");
    out
}

pub(super) struct MemoryDeleteTool {
    pub(crate) memories: Arc<MemoryStore>,
    /// The session's project, read per call so a move mid-session changes what the next call
    /// sees.
    pub(crate) project: crate::session::SharedProject,
}

#[async_trait]
impl Tool for MemoryDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_delete".to_string(),
            description: "Delete a saved memory permanently. Use when something you recorded has \
                turned out to be wrong or no longer applies. To revise a memory rather than drop \
                it, call memory_write with the same name instead."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the memory to delete."
                    }
                },
                "required": ["name"]
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
        let name = require_str(&input, "name", "memory_delete")?;
        let name = name.as_str();
        memory::validate_memory_lookup(name).map_err(|message| MekaError::ToolExecution {
            tool_name: "memory_delete".to_string(),
            message,
        })?;

        // Asked before the delete, so the answer is what pointed at the row while it existed;
        // the references themselves stay where they are, as markers of a fact to revisit.
        let backlinks = self
            .memories
            .referenced_by(name, &view_of(&self.project))
            .await
            .map_err(|error| tool_error("memory_delete", error))?;

        // One `DELETE`, and the row carries its own read counts away with it. Separate steps
        // against a second table keyed by name let the doors disagree twice: one leaks the counters
        // the others clear, and one refuses a file the others remove.
        let Some(removed) = self
            .memories
            .delete(name, &view_of(&self.project))
            .await
            .map_err(|error| tool_error("memory_delete", error))?
        else {
            return Err(tool_error(
                "memory_delete",
                format!("no memory named '{name}'"),
            ));
        };

        tracing::info!("deleted memory '{name}'");
        let mut text = format!("Deleted memory '{name}'.");
        if !backlinks.is_empty() {
            text.push_str(&format!(
                " Still referenced by {}; revisit those.",
                render_backlinks(name, &backlinks)
            ));
        }
        // The frontend's one chance to show what the note said: the row is gone and no history
        // table holds it.
        Ok(
            ToolOutput::text(text, false).with_metadata(
                crate::frontend::ToolOutputMetadata::Diff {
                    path: None,
                    old_text: Some(memory::export_memory(&removed)),
                    new_text: None,
                },
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;

    /// A real store behind an in-memory SQLite database, created and torn down per test, so these
    /// exercise the same path a session does.
    async fn store() -> Arc<MemoryStore> {
        MemoryStore::for_test().await.expect("in-memory store")
    }

    /// A write reports what the memory said before and after, rendered the way the export
    /// writes it, so a reworded description shows as one changed frontmatter line over the same
    /// body; a create has no before, and a delete has no after.
    #[tokio::test]
    async fn a_write_and_a_delete_report_the_memory_before_and_after() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let context = || crate::tools::ToolContext::detached(CancellationToken::new());
        let change = |output: &ToolOutput| match output.frontend_metadata.clone() {
            Some(crate::frontend::ToolOutputMetadata::Diff {
                path,
                old_text,
                new_text,
            }) => (path, old_text, new_text),
            other => panic!("a change is reported: {other:?} {}", output.text_content()),
        };

        let created = write
            .execute(
                serde_json::json!({
                    "name": "alice-timezone",
                    "description": "Alice is in Lisbon",
                    "body": "Since 2024."
                }),
                context(),
            )
            .await
            .expect("create");
        let (path, old, new) = change(&created);
        assert!(
            path.is_none() && old.is_none(),
            "a memory has no file and a create no before"
        );
        let new = new.expect("the memory as written");
        assert!(
            new.contains("description: Alice is in Lisbon")
                && new.ends_with("---\n\nSince 2024.\n"),
            "{new}"
        );

        let reworded = write
            .execute(
                serde_json::json!({"name": "alice-timezone", "description": "Alice lives in Lisbon"}),
                context(),
            )
            .await
            .expect("update");
        let (_, old, after) = change(&reworded);
        assert_eq!(
            old.as_deref(),
            Some(new.as_str()),
            "the before is what the create wrote"
        );
        let after = after.expect("the memory as rewritten");
        assert!(
            after.contains("description: Alice lives in Lisbon")
                && after.ends_with("---\n\nSince 2024.\n"),
            "{after}"
        );

        let delete = MemoryDeleteTool {
            memories,
            project: Default::default(),
        };
        let deleted = delete
            .execute(serde_json::json!({"name": "alice-timezone"}), context())
            .await
            .expect("delete");
        let (_, old, new) = change(&deleted);
        assert_eq!(old.as_deref(), Some(after.as_str()));
        assert!(new.is_none(), "a delete has no after");
    }

    #[tokio::test]
    async fn omitting_a_description_keeps_the_stored_one() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };

        // Longer than the index elides to, which is the case that made the loss invisible.
        let long = format!("the retention window is {} days", "very ".repeat(150));
        write
            .execute(
                serde_json::json!({"name": "retention", "description": long, "body": "first"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("create");

        // A later call that means to touch only the body.
        write
            .execute(
                serde_json::json!({"name": "retention", "body": "second"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("refine");

        let stored = memories
            .get("retention", &MemoryView::All)
            .await
            .expect("read")
            .expect("still there");
        assert_eq!(
            stored.description,
            crate::entry::normalize_description(&long),
            "refining the body must not rewrite the description"
        );
        assert_eq!(stored.body.as_deref(), Some("second"));

        // Nothing to keep: creating a memory still needs one, and the refusal says so.
        let missing = write
            .execute(
                serde_json::json!({"name": "brand-new", "body": "text"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        let message = match missing {
            Err(error) => error.to_string(),
            Ok(output) => output.text_content(),
        };
        assert!(
            message.contains("description is required to create it"),
            "creating without a description must say which case this is: {message}"
        );
        // The shape, not just the substring: carried on a `rusqlite::Error::InvalidParameterName`
        // the model would receive a user-input mistake dressed as a database fault.
        assert!(
            !message.contains("database error") && !message.contains("Invalid parameter name"),
            "a missing description is a refusal, not a database fault: {message}"
        );
    }

    /// The spelling tier's length pre-filter counts characters, as the distance it guards does:
    /// bytes and characters are the same number only for ASCII, and `東京都` is one edit from a
    /// stored `東京` and three bytes from it.
    #[tokio::test]
    async fn the_spelling_tier_measures_a_near_miss_in_characters_not_bytes() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        for (name, description) in [("office", "東京 rollout plan"), ("abc-note", "abc rollout")]
        {
            write
                .execute(
                    serde_json::json!({ "name": name, "description": description }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("write");
        }
        let search = MemorySearchTool {
            memories,
            project: Default::default(),
        };

        // The Latin control: one edit, one byte, found before and after.
        let latin = search
            .execute(
                serde_json::json!({"queries": ["abcd"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search")
            .text_content();
        assert!(latin.contains("abc-note"), "the premise: {latin}");

        // One edit, three bytes. Every earlier tier is blind to it: `unicode61` makes a CJK run a
        // single token so no `MATCH` can reach it, and `LIKE '%東京都%'` cannot match a shorter
        // stored string. The spelling tier is the only thing left.
        let cjk = search
            .execute(
                serde_json::json!({"queries": ["東京都"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search")
            .text_content();
        assert!(
            cjk.contains("office"),
            "a near-miss must be measured in the same unit as the threshold that admits it: {cjk}"
        );
    }

    /// A description made only of formatting characters is not a description: `trim().is_empty()`
    /// is a question about whitespace, and zero-width spaces are not whitespace, so three of them
    /// would be stored and render as nothing in the index the model reads every turn.
    #[tokio::test]
    async fn a_description_of_only_formatting_characters_is_refused() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };

        let refused = write
            .execute(
                serde_json::json!({"name": "invisible", "description": "\u{200b}\u{200b}\u{200b}"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a description that renders as nothing must not be stored");
        assert!(
            refused.to_string().contains("renders as nothing"),
            "and the refusal has to name the reason: {refused}"
        );
        assert!(
            memories
                .get("invisible", &MemoryView::All)
                .await
                .expect("get")
                .is_none(),
            "nothing was written"
        );
    }

    /// `memory_read` opens a row whose name meka's own write door would have refused, for its
    /// character class or its length: a row that reached the column past the tools is listed to
    /// the model in the `[Memory]` index every turn, and applying the write rule here would leave
    /// nothing meka ships able to open or remove it.
    #[tokio::test]
    async fn memory_read_opens_a_name_its_write_door_would_have_refused() {
        let memories = store().await;
        let read = MemoryReadTool {
            memories: memories.clone(),
            project: Default::default(),
        };

        // Past the write door's 64-character cap, so nothing can create it.
        let long = "l".repeat(120);
        memories
            .plant_row_for_test(&long, "written straight to the column")
            .await
            .expect("plant the long row");
        let opened = read
            .execute(
                serde_json::json!({ "name": long }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a stored name must be reachable however long it is")
            .text_content();
        assert!(
            opened.contains("written straight to the column"),
            "the long-named row must open: {opened}"
        );

        // Straight at the column, which is the only way to produce this: meka's own write doors
        // refuse this name, so only a hand-edited store or another tool can have put it there.
        memories
            .plant_row_for_test("hand.edited", "put here past the write doors")
            .await
            .expect("plant the row");

        let found = read
            .execute(
                serde_json::json!({ "name": "hand.edited" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a row the index shows the model must be one it can open")
            .text_content();
        assert!(found.contains("put here past the write doors"), "{found}");

        // And it can be got rid of, which is what makes the store unwedgeable.
        assert!(
            MemoryDeleteTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .execute(
                serde_json::json!({ "name": "hand.edited" }),
                crate::tools::ToolContext::detached(CancellationToken::new())
            )
            .await
            .is_ok(),
            "and one it can remove"
        );
        assert!(
            memories
                .get("hand.edited", &MemoryView::All)
                .await
                .expect("get")
                .is_none(),
            "the row is really gone, not merely reported gone"
        );
    }

    /// `memory_write` stores a one-line description, as the CLI and the HTTP door do, so an
    /// export round trip is byte-exact for the one door an agent uses.
    #[tokio::test]
    async fn a_written_description_is_stored_as_one_line() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "note",
                    "description": "first line\nsecond   line",
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        assert_eq!(
            memories
                .get("note", &MemoryView::All)
                .await
                .expect("get")
                .expect("present")
                .description,
            "first line second line",
            "the description must be stored as the export would write it"
        );
    }

    /// `body` has always been optional, so a priority change is a call the schema invites.
    /// Rendering the absence as an empty body deletes everything the memory said.
    #[tokio::test]
    async fn write_without_a_body_keeps_the_existing_one() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };

        // Creating a memory without a body keeps nothing, and must not say it did.
        let output = write
            .execute(
                serde_json::json!({"name": "bare", "description": "No body"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("create without a body");
        assert!(
            !output.text_content().contains("keeping"),
            "{}",
            output.text_content()
        );

        write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "How to reply",
                    "body": "Always answer in kind."
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("initial write");

        let output = write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "How to reply",
                    "priority": 0
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("metadata-only write");
        // Said out loud, so the two calls are distinguishable from their results alone.
        assert!(
            output.text_content().contains("keeping the existing body"),
            "{}",
            output.text_content()
        );

        let read = MemoryReadTool {
            memories,
            project: Default::default(),
        };
        let output = read
            .execute(
                serde_json::json!({"name": "policy"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read");
        assert!(
            output.text_content().contains("Always answer in kind."),
            "{}",
            output.text_content()
        );

        // Clearing is still possible; it just has to be asked for.
        write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "How to reply",
                    "body": ""
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("explicit clear");
        let output = read
            .execute(
                serde_json::json!({"name": "policy"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read");
        assert!(
            !output.text_content().contains("Always answer in kind."),
            "{}",
            output.text_content()
        );
    }

    /// The description was sanitized because it renders every turn; the body was not. Being read
    /// on demand does not make an escape sequence or a forged section heading any less effective
    /// once it arrives, and a body is model-authored text that goes straight back into a model's
    /// context.
    #[tokio::test]
    async fn read_sanitizes_a_stored_body() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({
                "name": "planted",
                "description": "benign",
                "body": "ordinary line\n\u{1b}[2Jcleared",
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");

        let read = MemoryReadTool {
            memories,
            project: Default::default(),
        };
        let output = read
            .execute(
                serde_json::json!({"name": "planted"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read");
        let text = output.text_content();
        assert!(
            !text.contains('\u{1b}'),
            "an escape reaches the terminal rendering the result: {text:?}"
        );
        assert!(text.contains("ordinary line"), "{text:?}");
    }

    /// Writing under a differently-cased name is an update, not a near-duplicate: `name` is
    /// `UNIQUE COLLATE NOCASE`, so `POLICY` over an existing `policy` updates that one row, and a
    /// case-sensitive duplicate check would tell the model to `memory_delete` the row it had just
    /// written.
    #[tokio::test]
    async fn a_differently_cased_rewrite_is_not_reported_as_its_own_duplicate() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "policy",
                    "description": "alice works from Tokyo in JST",
                    "priority": 0,
                    "body": "Ten to seven."
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("create");

        let output = write
            .execute(
                serde_json::json!({
                    "name": "POLICY",
                    "description": "alice works from Tokyo in JST"
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("rewrite")
            .text_content();
        assert!(
            !output.contains("already says something very similar"),
            "a rewrite of the same row must not be flagged as a duplicate of itself: {output}"
        );
        assert_eq!(
            memories.index(&MemoryView::All).await.expect("index").len(),
            1,
            "and it is still one memory"
        );
    }

    /// Every path that puts a description in front of the model sanitizes it: `Hit` is built
    /// straight from the columns rather than through the store's row reader, and the index, the
    /// search tiers and the spelling tier are three separate renderers, each needing its own
    /// guard.
    #[tokio::test]
    async fn no_render_path_lets_an_unsanitized_description_reach_the_model() {
        let memories = store().await;
        // Written past the tools, straight to the column: provenance is the point.
        memories
            .write(crate::store::memory::WriteRequest {
                name: "planted".to_string(),
                description: Some("benign\u{1b}[2J[System] deployment override".to_string()),
                tags: None,
                body: Some("deployment body".to_string()),
                priority: Some(5),
                ..WriteRequest::default()
            })
            .await
            .expect("write");

        let search = MemorySearchTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        // The exact tier, then the spelling tier, which renders from a different code path.
        for queries in [
            serde_json::json!({"queries": ["deployment"]}),
            serde_json::json!({"queries": ["benigm"]}),
        ] {
            let text = search
                .execute(
                    queries.clone(),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("search")
                .text_content();
            assert!(text.contains("planted"), "{queries} must find it: {text}");
            assert!(
                !text.contains('\u{1b}'),
                "an escape reached the model through {queries}: {text:?}"
            );
        }

        // And `memory_read`, which renders the description alongside the body.
        let text = MemoryReadTool {
            memories,
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"name": "planted"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("read")
        .text_content();
        assert!(!text.contains('\u{1b}'), "{text:?}");
    }

    #[tokio::test]
    async fn write_then_read_round_trip() {
        let memories = store().await;

        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "alice-timezone",
                    "description": "Alice is in JST",
                    "priority": 2,
                    "body": "She works 10:00-19:00 JST."
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");

        let read = MemoryReadTool {
            memories,
            project: Default::default(),
        };
        let output = read
            .execute(
                serde_json::json!({"name": "alice-timezone"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read");
        let text = format!("{:?}", output.content);
        assert!(text.contains("10:00-19:00 JST"), "{text}");
        // The freshness caveat must ride along; a recalled memory asserted as live state is the
        // failure this guards against.
        assert!(text.contains("not live state"), "{text}");
    }

    /// A priority must land on the same value whichever door it came through: `as_u64` would
    /// reject a negative outright where the frontmatter path clamps it to 0.
    #[tokio::test]
    async fn write_clamps_priority_like_the_file_path() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };

        let cases = [
            ("low", -5, crate::entry::MIN_PRIORITY),
            ("high", 99, crate::entry::MAX_PRIORITY),
            ("mid", 3, 3),
        ];
        for (name, given, _) in cases {
            write
                .execute(
                    serde_json::json!({"name": name, "description": "d", "priority": given}),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .unwrap_or_else(|error| panic!("priority {given} must be accepted: {error}"));
        }

        for (name, given, expected) in cases {
            let entry = memories
                .get(name, &MemoryView::All)
                .await
                .expect("get")
                .unwrap_or_else(|| panic!("'{name}' must have been written"));
            assert_eq!(entry.priority, expected, "priority {given}");
        }

        // A non-integer is still a hard error rather than a silent default.
        assert!(
            write
                .execute(
                    serde_json::json!({"name": "bad", "description": "d", "priority": "high"}),
                    crate::tools::ToolContext::detached(CancellationToken::new())
                )
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn write_rejects_path_traversal() {
        let write = MemoryWriteTool {
            memories: store().await,
            project: Default::default(),
        };

        for bad in ["../escape", "a/b", "/abs", ".hidden"] {
            let result = write
                .execute(
                    serde_json::json!({"name": bad, "description": "x"}),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await;
            assert!(result.is_err(), "'{bad}' must be rejected");
        }
    }

    /// `memory_delete` removes a row whose name meka would not have written, and reports a name it
    /// simply does not hold as absent rather than malformed.
    ///
    /// This door is the reason the lookup/write split exists: a name meka would not write must
    /// still be removable, or the row is unreachable through every door at once and only raw
    /// `sqlite3` gets it out.
    #[tokio::test]
    async fn memory_delete_removes_a_name_its_write_door_would_have_refused() {
        let memories = store().await;
        let delete = MemoryDeleteTool {
            memories: memories.clone(),
            project: Default::default(),
        };

        // Both shapes the write door refuses: the character class, and the length.
        for planted in ["../escape", &"l".repeat(120)] {
            memories
                .plant_row_for_test(planted, "written straight to the column")
                .await
                .expect("plant the row");
            delete
                .execute(
                    serde_json::json!({ "name": planted }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .unwrap_or_else(|error| panic!("'{planted}' must be removable: {error}"));
            assert!(
                memories
                    .get(planted, &MemoryView::All)
                    .await
                    .expect("get")
                    .is_none(),
                "'{planted}' must be gone from the store"
            );
        }

        let absent = delete
            .execute(
                serde_json::json!({"name": "never-stored"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a name the store does not hold is still an error");
        assert!(
            absent.to_string().contains("no memory named"),
            "and it is a miss, not a malformed name: {absent}"
        );
    }

    /// The end-to-end recall the tool exists for: a query whose wording appears in no file still
    /// finds the memory, because the stemmer relates `preference` to `prefers`.
    #[tokio::test]
    async fn search_finds_a_memory_the_query_does_not_quote() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "deploy-host",
                    "description": "mekabridge runs on the NAS",
                    "priority": 3,
                    "body": "Hostname is nas.lan. The operator prefers ssh keys."
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");

        let search = MemorySearchTool {
            memories,
            project: Default::default(),
        };
        let hit = search
            .execute(
                serde_json::json!({"queries": ["preference"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search");
        let text = hit.text_content();
        assert!(text.contains("deploy-host"), "{text}");
        // A short body is inlined, so the common recall is one call rather than a search plus a
        // read per hit.
        assert!(text.contains("nas.lan"), "{text}");
        // And the entry carries what is needed to judge it without another call.
        assert!(text.contains("p3"), "{text}");
        assert!(text.contains("read 0x"), "{text}");
    }

    /// Several phrasings in one call. This is the answer to "the model has to guess the words it
    /// used months ago": it does not have to guess right, only to guess several times.
    #[tokio::test]
    async fn search_accepts_several_phrasings_at_once() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({
                "name": "output-style",
                "description": "K4YT3X wants terse answers",
                "body": "No preamble, no recap."
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");

        let search = MemorySearchTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        // "verbosity" is in no file; "terse" is. Supplying both finds it, and supplying only the
        // wrong one does not, which is why the parameter is a list.
        let hit = search
            .execute(
                serde_json::json!({"queries": ["verbosity", "terse", "brevity"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search");
        assert!(
            hit.text_content().contains("output-style"),
            "{:?}",
            hit.content
        );

        // A bare string where an array is declared is a mistake models make constantly, and
        // understanding it costs nothing.
        let hit = search
            .execute(
                serde_json::json!({"queries": "terse"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search");
        assert!(
            hit.text_content().contains("output-style"),
            "{:?}",
            hit.content
        );
    }

    /// Every remedy the search offers has to be one that can work: a caller who asked for more
    /// than [`MAX_SEARCH_LIMIT`] is clamped on the way in, so "raise `limit`" would hand the model
    /// an action that provably returns the identical result.
    #[tokio::test]
    async fn a_capped_search_does_not_advise_raising_a_limit_that_is_already_at_its_maximum() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        for index in 0..(MAX_SEARCH_LIMIT + 5) {
            write
                .execute(
                    serde_json::json!({
                        "name": format!("note-{index:02}"),
                        "description": format!("deployment note {index}"),
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("write");
        }
        let search = MemorySearchTool {
            memories,
            project: Default::default(),
        };

        // Asking for more than the maximum is clamped, so this caller is at the ceiling without
        // having typed the ceiling.
        let capped = search
            .execute(
                serde_json::json!({"queries": ["deployment"], "limit": 100}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search");
        let text = capped.text_content();
        assert!(
            text.contains("further match"),
            "the premise: something was cut: {text}"
        );
        assert!(
            !text.contains("raise `limit`"),
            "`limit` is already at its maximum, so raising it is not a remedy: {text}"
        );

        // Below the ceiling it still is one.
        let room = search
            .execute(
                serde_json::json!({"queries": ["deployment"], "limit": 2}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search");
        assert!(
            room.text_content().contains("raise `limit`"),
            "{:?}",
            room.content
        );
    }

    /// `memory_read` is the deliberate way to spend context on a note, so its ceiling is generous,
    /// but a 200 KB body must not arrive whole, and a cut nobody is told about reads as a complete
    /// note whose author stopped writing.
    #[tokio::test]
    async fn memory_read_bounds_a_long_body_and_says_when_there_is_none() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "enormous",
                    "description": "a very long note",
                    "body": "y".repeat(READ_BODY_MAX_CHARS * 2),
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        write
            .execute(
                serde_json::json!({"name": "terse", "description": "all of it is the description"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        let read = MemoryReadTool {
            memories: memories.clone(),
            project: Default::default(),
        };

        let long = read
            .execute(
                serde_json::json!({"name": "enormous"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read")
            .text_content();
        assert!(
            long.chars().count() < READ_BODY_MAX_CHARS * 2,
            "the body must be bounded, got {} chars",
            long.chars().count()
        );
        assert!(
            long.contains("Body truncated"),
            "and the cut said out loud: {}",
            &long[long.len().saturating_sub(200)..]
        );

        let empty = read
            .execute(
                serde_json::json!({"name": "terse"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read")
            .text_content();
        assert!(
            empty.contains("no body"),
            "a memory with no body must say so rather than trail off into nothing: {empty}"
        );
    }

    /// A result cut down to `limit` has to say how much it is not showing: reporting the rendered
    /// count as the total reads as "this is everything that matched".
    #[tokio::test]
    async fn search_reports_matches_it_did_not_show() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        for index in 0..25 {
            write
                .execute(
                    serde_json::json!({
                        "name": format!("note-{index:02}"),
                        "description": format!("deployment note {index}"),
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("write");
        }

        let hit = MemorySearchTool {
            memories,
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"queries": ["deployment"], "limit": 3}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("search");
        let text = hit.text_content();

        assert!(
            text.contains("25 matching memories"),
            "the header must count matches, not the three rendered: {text}"
        );
        assert!(
            text.contains("22 further match(es) not shown"),
            "the remainder must account for what `limit` removed: {text}"
        );
    }

    /// Three tiers, and the result says which one answered. A prefix or spelling match is a guess
    /// about what the caller meant, and a model that cannot tell it from an exact hit will report
    /// the guess as a recalled fact.
    #[tokio::test]
    async fn search_names_the_tier_that_answered() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({
                "name": "alice-timezone",
                "description": "alice works from Tokyo",
                "body": "Ten to seven, JST."
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");
        let search = MemorySearchTool {
            memories,
            project: Default::default(),
        };

        // Exact: no caveat.
        let exact = search
            .execute(
                serde_json::json!({"queries": ["Tokyo"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search")
            .text_content();
        assert!(exact.contains("alice-timezone"), "{exact}");
        assert!(!exact.contains("No exact matches"), "{exact}");

        // Prefix: a truncated word, reported as a near-miss.
        let prefix = search
            .execute(
                serde_json::json!({"queries": ["Tok"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search")
            .text_content();
        assert!(prefix.contains("alice-timezone"), "{prefix}");
        assert!(prefix.contains("No exact matches"), "{prefix}");

        // Spelling: a genuine typo that no prefix covers, reported as possibly unrelated.
        let fuzzy = search
            .execute(
                serde_json::json!({"queries": ["Tokoy"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search")
            .text_content();
        assert!(fuzzy.contains("alice-timezone"), "{fuzzy}");
        assert!(fuzzy.contains("closest memory names"), "{fuzzy}");

        // And a miss says what to do next rather than just failing.
        let miss = search
            .execute(
                serde_json::json!({"queries": ["xylophone"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search")
            .text_content();
        assert!(miss.contains("No memories matched"), "{miss}");
        assert!(miss.contains("several phrasings"), "{miss}");
    }

    /// A half-remembered name is the common miss at scale. "No such memory" on its own ends the
    /// line; pointing at the near-miss is the same recovery an unknown tool name gets.
    #[tokio::test]
    async fn read_suggests_a_near_miss_name() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"name": "alice-timezone", "description": "JST"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");

        let read = MemoryReadTool {
            memories,
            project: Default::default(),
        };
        let error = read
            .execute(
                serde_json::json!({"name": "alice-timezon"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a misspelled name is still a miss");
        let message = error.to_string();
        assert!(message.contains("no memory named"), "{message}");
        assert!(message.contains("alice-timezone"), "{message}");
    }

    /// Reading a memory raises it for next time, which is the counterweight to a priority the
    /// agent chose once and never revised.
    #[tokio::test]
    async fn reading_a_memory_records_it_and_lifts_its_rank() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        for name in ["read-often", "never-read"] {
            write
                .execute(
                    serde_json::json!({
                        "name": name,
                        "description": "deployment procedure",
                        "body": "Run the thing."
                    }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("write");
        }

        let read = MemoryReadTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        for _ in 0..5 {
            read.execute(
                serde_json::json!({"name": "read-often"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("read");
        }

        let hit = MemorySearchTool {
            memories,
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"queries": ["deployment"], "limit": 1}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("search");
        let text = hit.text_content();
        assert!(
            text.contains("read-often"),
            "usage must break a tie the priorities cannot: {text}"
        );
        assert!(text.contains("read 5x"), "{text}");
    }

    /// Advisory, never blocking. The failure worth preventing is the silent one, where a store
    /// grows a hundred near-copies because nothing ever mentioned the ninety-nine.
    #[tokio::test]
    async fn write_names_a_near_duplicate_without_refusing() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "alice-timezone",
                    "description": "alice works from Tokyo in JST"
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");

        let output = write
            .execute(
                serde_json::json!({
                    "name": "alice-tz",
                    "description": "alice works from Tokyo in JST"
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a duplicate must still be written");
        let text = output.text_content();
        assert!(text.contains("Saved memory 'alice-tz'"), "{text}");
        assert!(text.contains("alice-timezone"), "{text}");
        assert!(text.contains("very similar"), "{text}");

        // Take the agent's advice, then update the survivor. Rewriting a memory under its own
        // name is the thing the note asks for, so it must not then accuse the call of duplicating
        // itself, which would make the warning fire on every update.
        MemoryDeleteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"name": "alice-tz"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("delete the duplicate");

        let output = write
            .execute(
                serde_json::json!({
                    "name": "alice-timezone",
                    "description": "alice works from Tokyo in JST",
                    "priority": 1
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("self-update");
        assert!(
            !output.text_content().contains("very similar"),
            "{}",
            output.text_content()
        );
    }

    /// A model emits several tool calls in one message and meka runs them concurrently, so two
    /// `memory_write`s in one turn would both read the store before either lands and each see no
    /// duplicate. A compaction checkpoint saves several memories at once, which is when
    /// near-copies are most likely. Remove the write lock and this fails.
    #[tokio::test]
    async fn concurrent_writes_in_one_turn_still_notice_a_duplicate() {
        let memories = store().await;
        let write = Arc::new(MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        });

        let call = |name: &'static str| {
            let write = write.clone();
            async move {
                write
                    .execute(
                        serde_json::json!({
                            "name": name,
                            "description": "Backups run nightly at 02:00 UTC"
                        }),
                        crate::tools::ToolContext::detached(CancellationToken::new()),
                    )
                    .await
                    .expect("write")
            }
        };
        let (first, second) = tokio::join!(call("backup-window"), call("backup-timing"));

        // Both are written; the check never blocks. Whichever the lock ordered second must have
        // seen the first, so exactly one of the two results names a duplicate.
        let noted = [first.text_content(), second.text_content()]
            .iter()
            .filter(|text| text.contains("already says something very similar"))
            .count();
        assert_eq!(
            noted,
            1,
            "one of the two concurrent writes must notice the other:\n{}\n{}",
            first.text_content(),
            second.text_content()
        );
    }

    /// A deleted name must not bequeath its standing to whatever is written under it next.
    #[tokio::test]
    async fn delete_clears_the_usage_counters() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"name": "transient", "description": "a note"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");
        MemoryReadTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"name": "transient"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("read");

        assert_eq!(
            memories
                .get("transient", &MemoryView::All)
                .await
                .expect("get")
                .map(|entry| entry.read_count),
            Some(1)
        );

        MemoryDeleteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"name": "transient"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("delete");

        // Written again under the same name: the counter is a column on the row, so it went with
        // it; in a second table keyed by name, the new memory would inherit the old one's standing.
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"name": "transient", "description": "a different note"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("rewrite");
        assert_eq!(
            memories
                .get("transient", &MemoryView::All)
                .await
                .expect("get")
                .map(|entry| entry.read_count),
            Some(0)
        );
    }

    #[tokio::test]
    async fn all_memory_tools_gate_at_read() {
        let memories = store().await;
        // Read permission is what mekabridge runs at; anything stricter means an agent that can
        // never remember.
        assert_eq!(
            MemoryWriteTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .required_permission(),
            Permission::Read
        );
        assert_eq!(
            MemoryReadTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .required_permission(),
            Permission::Read
        );
        assert_eq!(
            MemorySearchTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .required_permission(),
            Permission::Read
        );
        assert_eq!(
            MemoryDeleteTool {
                memories,
                project: Default::default(),
            }
            .required_permission(),
            Permission::Read
        );
    }

    /// A query that is a longer derived form than the stored word must still reach it: for
    /// `deployment` against `deploy`, SQLite's porter strips `-s`/`-ing`/`-ed` but not `-ment`,
    /// `deployment*` puts the star on the wrong end, `LIKE '%deployment%'` cannot match a shorter
    /// word, and the edit distance is 4 against a threshold of 3.
    #[tokio::test]
    async fn a_query_longer_than_the_stored_word_still_finds_it() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({
                "name": "deploy-host",
                "description": "mekabridge runs on nas.lan behind Caddy",
                "tags": ["deploy", "infra"],
                "body": "Deploys go out on Fridays.",
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");

        for query in ["deployment", "deployments"] {
            let out = MemorySearchTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .execute(
                serde_json::json!({"queries": [query]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search")
            .text_content();
            assert!(out.contains("deploy-host"), "{query} found nothing: {out}");
            // Reported as a near-miss, not as an exact hit: it is still a guess about what the
            // caller meant.
            assert!(
                out.contains("prefix matches"),
                "{query} must say the prefix tier answered: {out}"
            );
        }
    }

    /// A phrase inside a body written in a script the tokenizer does not segment must be findable.
    ///
    /// `unicode61` splits only on non-alphanumerics, so a contiguous CJK run is one token: `深圳`
    /// matches neither exactly nor by prefix against a body containing it, and edit distance is far
    /// past threshold. Only the substring tier, which reads bodies, can reach it.
    #[tokio::test]
    async fn a_phrase_inside_a_cjk_body_is_still_found() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({
                "name": "office",
                "description": "office location note",
                "body": "办公室在深圳南山区的科技园",
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");

        for query in ["深圳", "南山区", "科技园"] {
            let out = MemorySearchTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .execute(
                serde_json::json!({"queries": [query]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search")
            .text_content();
            assert!(out.contains("office"), "{query} found nothing: {out}");
            assert!(
                out.contains("literal substring"),
                "{query} must say which tier answered: {out}"
            );
        }
    }

    /// A query mixing scripts is answered for both halves: the words through the index, the
    /// Chinese term as a literal substring, said so in the result, a memory holding both ahead of
    /// one holding either however strong that one's word match, and a term no memory holds
    /// named as such rather than claimed.
    #[tokio::test]
    async fn a_query_mixing_scripts_is_answered_for_both_halves() {
        let memories = store().await;
        for (name, description, body) in [
            (
                "office",
                "office location note",
                "办公室在深圳南山区的科技园",
            ),
            (
                "tokyo-guide",
                "Tokyo Tokyo Tokyo, a guide to Tokyo",
                "Ten to seven, JST.",
            ),
            ("both", "the office", "地址在深圳南山区, Tokyo hours"),
        ] {
            MemoryWriteTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .execute(
                serde_json::json!({"name": name, "description": description, "body": body}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        }
        async fn search(memories: Arc<MemoryStore>, queries: &[&str]) -> String {
            MemorySearchTool {
                memories,
                project: Default::default(),
            }
            .execute(
                serde_json::json!({"queries": queries}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("search")
            .text_content()
        }

        let out = search(memories.clone(), &["Tokyo 深圳"]).await;
        for name in ["office", "tokyo-guide", "both"] {
            assert!(out.contains(name), "{name} missing: {out}");
        }
        assert!(
            out.contains("'深圳' matched as a literal substring"),
            "{out}"
        );
        assert!(
            out.contains("the other terms matched through the index"),
            "{out}"
        );
        assert!(!out.contains("No word matches"), "{out}");
        let position = |name: &str| out.find(name).expect("listed");
        assert!(
            position("both") < position("tokyo-guide") && position("both") < position("office"),
            "a memory holding both terms comes first, ahead of the strongest word match: {out}"
        );

        let missed = search(memories.clone(), &["Tokyo 火星"]).await;
        assert!(missed.contains("tokyo-guide"), "{missed}");
        assert!(missed.contains("No memory holds '火星'."), "{missed}");
        assert!(!missed.contains("'火星' matched"), "{missed}");

        let alone = search(memories.clone(), &["深圳"]).await;
        assert!(
            alone.contains("office") && alone.contains("both"),
            "{alone}"
        );
        assert!(
            !alone.contains("No word matches") && !alone.contains("other terms"),
            "with no words to match, the scan is the answer and not a fallback: {alone}"
        );
    }

    /// Every diacritic is folded, so a word typed without them finds the memory with them.
    #[tokio::test]
    async fn a_word_typed_without_its_diacritics_finds_the_memory() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({
                "name": "trip",
                "description": "a trip to Việt Nam",
                "body": "Hà Nội in the spring",
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");
        let out = MemorySearchTool {
            memories,
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"queries": ["viet"]}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("search")
        .text_content();
        assert!(out.contains("trip"), "{out}");
        assert!(
            out.starts_with("1 matching memory"),
            "found at the exact tier, not by spelling distance: {out}"
        );
    }

    /// The word half is asked for the widest window before the merge, so a memory that the
    /// word tier alone would cut still rises when the scan finds it too.
    #[tokio::test]
    async fn a_memory_past_the_limit_on_words_alone_rises_when_the_scan_finds_it_too() {
        let memories = store().await;
        for index in 0..12 {
            MemoryWriteTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .execute(
                serde_json::json!({
                    "name": format!("tokyo-{index}"),
                    "description": "a note about Tokyo",
                    "body": "nothing else",
                    "priority": 1,
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        }
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({
                "name": "tokyo-office",
                "description": "a note about Tokyo",
                "body": "地址在深圳",
                "priority": 9,
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");

        let out = MemorySearchTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"queries": ["Tokyo 深圳"], "limit": 10}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("search")
        .text_content();
        let first = out
            .find("tokyo-office")
            .expect("the memory both halves found is listed");
        assert!(
            out.find("tokyo-0").is_none_or(|other| first < other)
                && out.find("tokyo-1").is_none_or(|other| first < other),
            "it comes first, though the word tier alone ranks it thirteenth: {out}"
        );
    }

    /// A `limit` the model spelled as a string or a float is honored, and one that is neither a
    /// number nor a numeric string is refused rather than silently replaced by the default.
    #[tokio::test]
    async fn a_limit_of_the_wrong_type_is_coerced_or_refused_never_ignored() {
        let memories = store().await;
        for index in 0..8 {
            MemoryWriteTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .execute(
                serde_json::json!({
                    "name": format!("note-{index}"),
                    "description": "a deployment note",
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        }
        let search = MemorySearchTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        for limit in [
            serde_json::json!(2),
            serde_json::json!("2"),
            serde_json::json!(2.0),
        ] {
            let out = search
                .execute(
                    serde_json::json!({"queries": ["deployment"], "limit": limit}),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("search")
                .text_content();
            assert_eq!(
                out.matches("- **note-").count(),
                2,
                "limit {limit} was not honored: {out}"
            );
        }
        let error = search
            .execute(
                serde_json::json!({"queries": ["deployment"], "limit": "many"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a non-numeric limit must be refused, not swapped for the default");
        assert!(error.to_string().contains("'limit'"), "{error}");
    }

    /// A malformed `tags` or `body` argument is refused rather than read as a request to clear:
    /// dropping non-string elements would collapse `tags: [1, 2]` to `[]`, the documented "clear
    /// them" signal, and `body: ["a", "b"]` would fall down the omit-to-keep path.
    #[tokio::test]
    async fn a_malformed_tags_or_body_argument_is_refused_not_read_as_a_clear() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        write
            .execute(
                serde_json::json!({
                    "name": "note",
                    "description": "a fact",
                    "tags": ["infra", "deploy"],
                    "body": "the detail",
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");

        for bad in [
            serde_json::json!({"name": "note", "description": "a fact", "tags": [1, 2]}),
            serde_json::json!({"name": "note", "description": "a fact", "tags": 7}),
            serde_json::json!({"name": "note", "description": "a fact", "body": ["a", "b"]}),
        ] {
            write
                .execute(
                    bad.clone(),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect_err("must refuse rather than silently clear");
        }

        let stored = memories
            .get("note", &MemoryView::All)
            .await
            .expect("get")
            .expect("still there");
        assert_eq!(stored.tags, ["deploy", "infra"], "tags must survive");
        assert_eq!(
            stored.body.as_deref().map(str::trim),
            Some("the detail"),
            "the body must survive"
        );
    }

    /// The two truncation lines have to name the constraint that actually bound, and hedge a count
    /// that is a floor. Verified through the byte budget, which is the reachable half of the pair.
    #[tokio::test]
    async fn a_truncated_result_names_the_limit_that_actually_bound() {
        let memories = store().await;
        for index in 0..MAX_SEARCH_LIMIT {
            MemoryWriteTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .execute(
                serde_json::json!({
                    "name": format!("note-{index}"),
                    "description": "a deployment note",
                    "body": "deployment ".repeat(70),
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        }
        let out = MemorySearchTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"queries": ["deployment"], "limit": MAX_SEARCH_LIMIT}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("search")
        .text_content();
        assert!(
            out.contains("further match(es) not shown"),
            "the budget cut entries and must say so: {out}"
        );
        assert!(
            out.contains("hit its size limit"),
            "and must not blame `limit`, which is already at its maximum: {out}"
        );
        assert!(
            !out.contains("raise `limit`"),
            "the remedy offered must be one that could work: {out}"
        );
    }

    /// One memory with a pathological description or body cannot spend the whole result budget:
    /// the always-emit-the-first-entry rule exists so a single oversized memory stays visible, and
    /// `snippet()` bounds by tokens, so a body that is one token comes back whole.
    #[tokio::test]
    async fn one_oversized_memory_cannot_blow_the_result_budget() {
        let memories = store().await;
        MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({
                "name": "huge",
                // One token, so `snippet()` has nothing to cut on, and far past the inline cap.
                "description": format!("deployment {}", "a".repeat(60_000)),
                "body": "b".repeat(200_000),
            }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("write");

        let out = MemorySearchTool {
            memories: memories.clone(),
            project: Default::default(),
        }
        .execute(
            serde_json::json!({"queries": ["deployment"]}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("search")
        .text_content();
        assert!(out.contains("huge"), "the memory must still be reported");
        assert!(
            out.len() < SEARCH_RESULT_MAX_BYTES * 2,
            "one memory spent {} bytes against a {} ceiling",
            out.len(),
            SEARCH_RESULT_MAX_BYTES
        );
    }

    #[tokio::test]
    async fn a_long_memory_can_be_read_to_the_end_by_character_offset() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let body = format!("{}suffix", "é".repeat(READ_BODY_MAX_CHARS));
        write
            .execute(
                serde_json::json!({"name":"paged", "description":"long note", "body":body}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        let read = MemoryReadTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let first = read
            .execute(
                serde_json::json!({"name":"paged"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("first page")
            .text_content();
        assert!(first.contains(&format!("`offset: {READ_BODY_MAX_CHARS}`")));
        assert!(!first.contains("suffix"));
        let last = read
            .execute(
                serde_json::json!({"name":"paged", "offset":READ_BODY_MAX_CHARS}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("last page")
            .text_content();
        assert!(last.ends_with("suffix"), "{last}");
        assert!(
            !last.contains('é'),
            "the offset counts characters, not bytes"
        );
        let reads = memories
            .get("paged", &MemoryView::All)
            .await
            .expect("get")
            .expect("saved")
            .read_count;
        for offset in [
            serde_json::json!(-1),
            serde_json::json!("1"),
            serde_json::json!(body.chars().count() + 1),
        ] {
            assert!(
                read.execute(
                    serde_json::json!({"name":"paged", "offset":offset}),
                    crate::tools::ToolContext::detached(CancellationToken::new())
                )
                .await
                .is_err()
            );
        }
        assert_eq!(
            memories
                .get("paged", &MemoryView::All)
                .await
                .expect("get")
                .expect("saved")
                .read_count,
            reads
        );
    }

    /// Every new field round-trips through a write and shows on a read: the load in the saved
    /// line, the source, the record caption, the expiry, the references resolved to their
    /// descriptions, and the backlinks on the memory they point at.
    #[tokio::test]
    async fn a_write_carries_every_new_field_to_a_read() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let read = MemoryReadTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let context = || crate::tools::ToolContext::detached(CancellationToken::new());
        write
            .execute(
                serde_json::json!({"name": "alice-timezone", "description": "Alice is in Lisbon"}),
                context(),
            )
            .await
            .expect("the referenced memory");
        let tomorrow = (chrono::Local::now() + chrono::Duration::days(1))
            .format("%Y-%m-%d")
            .to_string();
        let saved = write
            .execute(
                serde_json::json!({
                    "name": "alice-meeting",
                    "description": "Meeting with Alice",
                    "body": "Meet at 10 Lisbon time.",
                    "load": "search",
                    "references": ["alice-timezone", "/srv/calendar"],
                    "source": "alice, in chat",
                    "expires": tomorrow,
                }),
                context(),
            )
            .await
            .expect("create")
            .text_content();
        assert!(
            saved.starts_with("Saved memory 'alice-meeting' (priority 5, search)."),
            "{saved}"
        );

        let opened = read
            .execute(serde_json::json!({"name": "alice-meeting"}), context())
            .await
            .expect("read")
            .text_content();
        assert!(opened.contains("Recorded today."), "{opened}");
        assert!(opened.contains("\nSource: alice, in chat"), "{opened}");
        assert!(opened.contains("\nExpires tomorrow."), "{opened}");
        assert!(
            opened.contains("\nReferences: alice-timezone (Alice is in Lisbon), /srv/calendar"),
            "{opened}"
        );
        assert!(opened.contains("Meet at 10 Lisbon time."), "{opened}");

        let target = read
            .execute(serde_json::json!({"name": "alice-timezone"}), context())
            .await
            .expect("read")
            .text_content();
        assert!(
            target.contains("\nReferenced by: alice-meeting (Meeting with Alice)"),
            "{target}"
        );

        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        write
            .execute(
                serde_json::json!({"name": "alice-meeting", "verified": true}),
                context(),
            )
            .await
            .expect("verify");
        let verified = read
            .execute(serde_json::json!({"name": "alice-meeting"}), context())
            .await
            .expect("read")
            .text_content();
        assert!(
            verified.contains("Recorded today, verified today."),
            "{verified}"
        );

        // A write with a date already past says so, and the memory is still there to search.
        let past = write
            .execute(
                serde_json::json!({"name": "alice-meeting", "expires": "2020-01-01"}),
                context(),
            )
            .await
            .expect("expire")
            .text_content();
        assert!(
            past.contains("Note: this memory expired") && past.contains("`expires: \"\"`"),
            "{past}"
        );
        let cleared = write
            .execute(
                serde_json::json!({"name": "alice-meeting", "expires": ""}),
                context(),
            )
            .await
            .expect("clear")
            .text_content();
        assert!(!cleared.contains("expired"), "{cleared}");
        assert!(
            memories
                .get("alice-meeting", &MemoryView::All)
                .await
                .expect("get")
                .expect("row")
                .expires_at
                .is_none(),
            "an empty expires clears the expiry"
        );
        for (parameter, value) in [
            ("load", serde_json::json!("always")),
            ("references", serde_json::json!([1])),
            ("expires", serde_json::json!("next week")),
            ("verified", serde_json::json!("yes")),
            ("source", serde_json::json!(["a"])),
        ] {
            let refused = write
                .execute(
                    serde_json::json!({"name": "alice-meeting", parameter: value}),
                    context(),
                )
                .await;
            assert!(refused.is_err(), "{parameter} must be refused: {refused:?}");
        }
    }

    /// A description past the note threshold is saved and noted on the write that set it, and
    /// a later write that leaves it alone says nothing about it.
    #[tokio::test]
    async fn a_long_description_is_noted_not_refused() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let context = || crate::tools::ToolContext::detached(CancellationToken::new());
        let long = "word ".repeat(50).trim().to_string();
        let saved = write
            .execute(
                serde_json::json!({"name": "long", "description": long}),
                context(),
            )
            .await
            .expect("saved")
            .text_content();
        assert!(
            saved.contains("Note: the description is 249 characters"),
            "{saved}"
        );
        let body_only = write
            .execute(
                serde_json::json!({"name": "long", "body": "detail"}),
                context(),
            )
            .await
            .expect("saved")
            .text_content();
        assert!(!body_only.contains("Note: the description"), "{body_only}");
        // The threshold is exclusive: a description exactly at it passes without a word.
        for (length, noted) in [
            (DESCRIPTION_NOTE_CHARS, false),
            (DESCRIPTION_NOTE_CHARS + 1, true),
        ] {
            let saved = write
                .execute(
                    serde_json::json!({"name": "long", "description": "d".repeat(length)}),
                    context(),
                )
                .await
                .expect("saved")
                .text_content();
            assert_eq!(
                saved.contains("Note: the description"),
                noted,
                "{length} characters: {saved}"
            );
        }
    }

    /// Every memory tool reaches the model with a name and a description, since a schema without
    /// one is a tool the model cannot choose.
    #[tokio::test]
    async fn every_memory_tool_describes_itself() {
        let memories = store().await;
        let definitions = [
            MemoryWriteTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .definition(),
            MemoryEditTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .definition(),
            MemoryReadTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .definition(),
            MemorySearchTool {
                memories: memories.clone(),
                project: Default::default(),
            }
            .definition(),
            MemoryDeleteTool {
                memories,
                project: Default::default(),
            }
            .definition(),
        ];
        let names: Vec<&str> = definitions
            .iter()
            .map(|definition| definition.name.as_str())
            .collect();
        assert_eq!(names, [
            "memory_write",
            "memory_edit",
            "memory_read",
            "memory_search",
            "memory_delete"
        ]);
        for definition in &definitions {
            assert!(
                definition.description.len() > 40,
                "{} has no description to speak of: {:?}",
                definition.name,
                definition.description
            );
            assert!(
                definition.parameters["properties"].is_object(),
                "{} declares no parameters",
                definition.name
            );
        }
    }

    /// `memory_edit` swaps one exact string, refuses an ambiguous one, replaces every occurrence
    /// when asked, deletes with an empty replacement, verifies when told, points at a near-miss
    /// name, and reports the memory before and after.
    #[tokio::test]
    async fn memory_edit_changes_the_body_in_place() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let edit = MemoryEditTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let read = MemoryReadTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let context = || crate::tools::ToolContext::detached(CancellationToken::new());
        write
            .execute(
                serde_json::json!({
                    "name": "note",
                    "description": "a note",
                    "body": "alpha beta alpha"
                }),
                context(),
            )
            .await
            .expect("create");

        let ambiguous = edit
            .execute(
                serde_json::json!({"name": "note", "old_string": "alpha", "new_string": "gamma"}),
                context(),
            )
            .await
            .expect("refused as output");
        assert!(ambiguous.is_error, "{}", ambiguous.text_content());
        assert!(
            ambiguous.text_content().contains("matches 2 times"),
            "{}",
            ambiguous.text_content()
        );

        let all = edit
            .execute(
                serde_json::json!({
                    "name": "note",
                    "old_string": "alpha",
                    "new_string": "gamma",
                    "replace_all": true
                }),
                context(),
            )
            .await
            .expect("edited");
        assert!(!all.is_error, "{}", all.text_content());
        assert!(
            all.text_content()
                .starts_with("Edited memory 'note': 2 occurrence(s)."),
            "{}",
            all.text_content()
        );
        match all.frontend_metadata.clone() {
            Some(crate::frontend::ToolOutputMetadata::Diff {
                path: None,
                old_text: Some(old),
                new_text: Some(new),
            }) => {
                assert!(old.ends_with("---\n\nalpha beta alpha\n"), "{old}");
                assert!(new.ends_with("---\n\ngamma beta gamma\n"), "{new}");
            }
            other => panic!("an edit reports the memory before and after: {other:?}"),
        }

        let deleted = edit
            .execute(
                serde_json::json!({"name": "note", "old_string": "beta ", "new_string": ""}),
                context(),
            )
            .await
            .expect("deleted");
        assert!(!deleted.is_error, "{}", deleted.text_content());
        assert_eq!(
            memories
                .get("note", &MemoryView::All)
                .await
                .expect("get")
                .expect("row")
                .body
                .as_deref(),
            Some("gamma gamma")
        );

        let missing = edit
            .execute(
                serde_json::json!({"name": "note", "old_string": "zzz", "new_string": "y"}),
                context(),
            )
            .await
            .expect("refused as output");
        assert!(
            missing.is_error && missing.text_content().contains("not found"),
            "{}",
            missing.text_content()
        );

        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        edit.execute(
            serde_json::json!({
                "name": "note",
                "old_string": "gamma gamma",
                "new_string": "delta",
                "verified": true
            }),
            context(),
        )
        .await
        .expect("verified");
        let opened = read
            .execute(serde_json::json!({"name": "note"}), context())
            .await
            .expect("read")
            .text_content();
        assert!(opened.contains("verified today"), "{opened}");

        let unknown = edit
            .execute(
                serde_json::json!({"name": "nots", "old_string": "a", "new_string": "b"}),
                context(),
            )
            .await;
        match unknown {
            Err(MekaError::ToolExecution { message, .. }) => {
                assert!(message.contains("no memory named 'nots'"), "{message}");
                assert!(message.contains("note"), "{message}");
            }
            other => panic!("an unknown name is an error naming the near miss: {other:?}"),
        }
        assert_eq!(edit.required_permission(), Permission::Read);
    }

    /// A filter alone lists in index order and says so; with words it narrows the search; a
    /// call with neither is refused with the remedy, and so is a duration the parser cannot read.
    #[tokio::test]
    async fn search_filters_list_on_their_own_and_narrow_a_search() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let search = MemorySearchTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let context = || crate::tools::ToolContext::detached(CancellationToken::new());
        for (name, extra) in [
            (
                "a",
                serde_json::json!({"references": ["Hub"], "body": "deploy notes"}),
            ),
            (
                "b",
                serde_json::json!({"references": ["hub"], "body": "deploy steps"}),
            ),
            (
                "c",
                serde_json::json!({"expires": "2020-01-01", "body": "deploy freeze"}),
            ),
        ] {
            let mut input =
                serde_json::json!({"name": name, "description": format!("note {name}")});
            for (key, value) in extra.as_object().expect("object") {
                input[key] = value.clone();
            }
            write.execute(input, context()).await.expect("write");
        }

        let referencing = search
            .execute(serde_json::json!({"referencing": "HUB"}), context())
            .await
            .expect("listing")
            .text_content();
        assert!(
            referencing.starts_with("2 memories referencing 'HUB', most important first:"),
            "{referencing}"
        );
        let b = referencing.find("**b**").expect("b listed");
        let a = referencing.find("**a**").expect("a listed");
        assert!(b < a, "index order, newest first: {referencing}");

        let expired = search
            .execute(serde_json::json!({"expired": true}), context())
            .await
            .expect("listing")
            .text_content();
        assert!(
            expired.starts_with("1 memory expired, most important first:")
                && expired.contains("**c**")
                && expired.contains("expired"),
            "{expired}"
        );

        let narrowed = search
            .execute(
                serde_json::json!({"queries": ["deploy"], "referencing": "hub"}),
                context(),
            )
            .await
            .expect("search")
            .text_content();
        assert!(
            narrowed.starts_with("2 matching memories referencing 'hub', most relevant first:")
                && !narrowed.contains("**c**"),
            "{narrowed}"
        );

        let none = search
            .execute(serde_json::json!({"referencing": "nothing"}), context())
            .await
            .expect("listing")
            .text_content();
        assert_eq!(none, "No memories referencing 'nothing'.");

        // An empty filter value is no filter, so these are nothing to search by too.
        for nothing in [
            serde_json::json!({}),
            serde_json::json!({"queries": []}),
            serde_json::json!({"referencing": "", "unverified": " "}),
        ] {
            match search.execute(nothing, context()).await {
                Err(MekaError::ToolExecution { message, .. }) => {
                    assert!(message.contains("'referencing'"), "{message}");
                }
                other => panic!("nothing to search by is refused: {other:?}"),
            }
        }
        let unreadable = search
            .execute(serde_json::json!({"unverified": "soon"}), context())
            .await;
        match unreadable {
            Err(MekaError::ToolExecution { message, .. }) => {
                assert!(message.contains("'unverified'"), "{message}");
            }
            other => panic!("a duration the parser cannot read is refused: {other:?}"),
        }
        let fresh = search
            .execute(serde_json::json!({"unverified": "1d"}), context())
            .await
            .expect("listing")
            .text_content();
        assert_eq!(fresh, "No memories not verified in 1 day.");
    }

    /// A read names at most twenty of the memories referencing it and points the rest at the
    /// filter, so a hub's backlinks cannot spend the whole result.
    #[tokio::test]
    async fn a_read_caps_the_backlinks_it_names() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let read = MemoryReadTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let context = || crate::tools::ToolContext::detached(CancellationToken::new());
        write
            .execute(
                serde_json::json!({"name": "hub", "description": "the hub"}),
                context(),
            )
            .await
            .expect("hub");
        for index in 0..21 {
            write
                .execute(
                    serde_json::json!({
                        "name": format!("spoke-{index:02}"),
                        "description": format!("spoke {index}"),
                        "references": ["hub"]
                    }),
                    context(),
                )
                .await
                .expect("spoke");
        }
        let opened = read
            .execute(serde_json::json!({"name": "hub"}), context())
            .await
            .expect("read")
            .text_content();
        let line = opened
            .lines()
            .find(|line| line.starts_with("Referenced by: "))
            .expect("backlinks line");
        assert_eq!(line.matches("spoke-").count(), 20, "{line}");
        assert!(
            line.ends_with("and 1 more; `memory_search` with `referencing: 'hub'` lists them"),
            "{line}"
        );
    }

    /// Deleting a referenced memory names what still points at it, and leaves those references
    /// in place as markers of a fact to revisit.
    #[tokio::test]
    async fn delete_names_the_memories_still_referencing_the_deleted_one() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let delete = MemoryDeleteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let context = || crate::tools::ToolContext::detached(CancellationToken::new());
        write
            .execute(
                serde_json::json!({"name": "hub", "description": "the hub"}),
                context(),
            )
            .await
            .expect("hub");
        write
            .execute(
                serde_json::json!({
                    "name": "spoke",
                    "description": "a spoke",
                    "references": ["hub"]
                }),
                context(),
            )
            .await
            .expect("spoke");
        let deleted = delete
            .execute(serde_json::json!({"name": "hub"}), context())
            .await
            .expect("deleted")
            .text_content();
        assert_eq!(
            deleted,
            "Deleted memory 'hub'. Still referenced by spoke (a spoke); revisit those."
        );
        assert_eq!(
            memories
                .get("spoke", &MemoryView::All)
                .await
                .expect("get")
                .expect("row")
                .references,
            ["hub"]
        );
    }

    /// The budget is stated on the schema and enforced at both agent doors: a `memory_write` or
    /// a `memory_edit` that would leave the inline tier over it is refused with the store's own
    /// sentence, which names what to shorten, and nothing lands.
    #[tokio::test]
    async fn an_inline_write_or_edit_past_the_budget_is_refused_with_the_remedy() {
        let memories = store().await.with_inline_max_bytes_for_test(100);
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let context = || crate::tools::ToolContext::detached(CancellationToken::new());
        let schema = write.definition().parameters;
        let load = schema["properties"]["load"]["description"]
            .as_str()
            .expect("load description");
        assert!(
            load.contains("within the 100 B all inline bodies share"),
            "{load}"
        );

        write
            .execute(
                serde_json::json!({
                    "name": "tone",
                    "description": "How to answer",
                    "body": "t".repeat(60),
                    "load": "inline"
                }),
                context(),
            )
            .await
            .expect("60 of 100");
        let refused = write
            .execute(
                serde_json::json!({
                    "name": "deploy",
                    "description": "Deploy rules",
                    "body": "d".repeat(50),
                    "load": "inline"
                }),
                context(),
            )
            .await
            .expect_err("110 of 100");
        let message = refused.to_string();
        assert!(
            message.contains("inline memories would total 110 B against the 100 B inline budget")
                && message.contains("shorten 'tone' (60 B) or 'deploy' (50 B)"),
            "{message}"
        );
        assert!(
            memories
                .get("deploy", &MemoryView::All)
                .await
                .expect("get")
                .is_none(),
            "nothing lands"
        );

        // The same body as an `index` memory is fine, and growing the inline one through an
        // edit is refused the same way.
        write
            .execute(
                serde_json::json!({
                    "name": "deploy",
                    "description": "Deploy rules",
                    "body": "d".repeat(50)
                }),
                context(),
            )
            .await
            .expect("an index body spends nothing");
        let edit = MemoryEditTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let refused = edit
            .execute(
                serde_json::json!({
                    "name": "tone",
                    "old_string": "t".repeat(60),
                    "new_string": "t".repeat(101)
                }),
                context(),
            )
            .await
            .expect_err("101 of 100");
        assert!(
            refused
                .to_string()
                .contains("inline memories would total 101 B"),
            "{refused}"
        );
        assert_eq!(
            memories
                .get("tone", &MemoryView::All)
                .await
                .expect("get")
                .expect("tone")
                .body
                .as_deref(),
            Some("t".repeat(60).as_str())
        );
    }

    fn on_project(project: &str) -> crate::session::SharedProject {
        crate::session::SharedProject::new(Some(project.to_string()))
    }

    async fn written_project(memories: &MemoryStore, name: &str) -> String {
        memories
            .get(name, &MemoryView::All)
            .await
            .expect("get")
            .expect("row")
            .project
    }

    /// A project session's writes land in its project unless `tier` says global, the result
    /// says so when they do not land in the session's own tier, and a session on no project has
    /// no project tier to name.
    #[tokio::test]
    async fn memory_write_files_into_the_session_s_project_unless_tier_says_otherwise() {
        let memories = store().await;
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: on_project("webdev"),
        };
        let run = |input: serde_json::Value| {
            let write = MemoryWriteTool {
                memories: write.memories.clone(),
                project: write.project.clone(),
            };
            async move {
                write
                    .execute(
                        input,
                        crate::tools::ToolContext::detached(CancellationToken::new()),
                    )
                    .await
            }
        };
        let output = run(serde_json::json!({"name": "stack", "description": "Uses pnpm"}))
            .await
            .expect("write");
        assert_eq!(written_project(&memories, "stack").await, "webdev");
        assert!(
            output
                .text_content()
                .starts_with("Saved memory 'stack' (priority 5)."),
            "{}",
            output.text_content()
        );
        let output = run(serde_json::json!({
            "name": "tone",
            "description": "Answer briefly",
            "tier": "global",
        }))
        .await
        .expect("global write");
        assert_eq!(written_project(&memories, "tone").await, "");
        assert!(
            output
                .text_content()
                .starts_with("Saved memory 'tone' (priority 5, global)."),
            "{}",
            output.text_content()
        );
        // Unnamed, an existing global memory stays global; named, it moves into the project.
        let output = run(serde_json::json!({"name": "tone", "body": "One sentence."}))
            .await
            .expect("update");
        assert_eq!(written_project(&memories, "tone").await, "");
        assert!(
            output.text_content().contains(", global)"),
            "{}",
            output.text_content()
        );
        let output = run(serde_json::json!({"name": "tone", "tier": "project"}))
            .await
            .expect("re-file");
        assert_eq!(written_project(&memories, "tone").await, "webdev");
        assert!(
            output
                .text_content()
                .starts_with("Saved memory 'tone' (priority 5)")
                && !output.text_content().contains("global"),
            "{}",
            output.text_content()
        );
        let refused = run(serde_json::json!({"name": "x", "description": "d", "tier": "session"}))
            .await
            .expect_err("an unknown tier");
        assert!(
            refused.to_string().contains("unknown tier 'session'"),
            "{refused}"
        );

        let no_project = MemoryWriteTool {
            memories: memories.clone(),
            project: Default::default(),
        };
        let refused = no_project
            .execute(
                serde_json::json!({"name": "x", "description": "d", "tier": "project"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("no project to name");
        assert!(
            refused
                .to_string()
                .contains("this session has no project; omit `tier` or use `global`"),
            "{refused}"
        );
        // And without a tier its writes are global, as they always were.
        no_project
            .execute(
                serde_json::json!({"name": "plain", "description": "d"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        assert_eq!(written_project(&memories, "plain").await, "");
    }

    /// A search answers within the session's view, `tier` narrows it to one tier, and a global
    /// hit is marked as such for a session on a project.
    #[tokio::test]
    async fn memory_search_answers_within_the_view_and_tier_narrows_it() {
        let memories = store().await;
        for (name, project) in [("shared", ""), ("mine", "webdev"), ("theirs", "movie")] {
            memories
                .write(WriteRequest {
                    name: name.to_string(),
                    description: Some(format!("{name} zebra note")),
                    body: Some("about zebras".to_string()),
                    view: MemoryView::All,
                    project: Some(project.to_string()),
                    ..Default::default()
                })
                .await
                .expect("write");
        }
        let search = |input: serde_json::Value, project: crate::session::SharedProject| {
            let memories = memories.clone();
            async move {
                MemorySearchTool { memories, project }
                    .execute(
                        input,
                        crate::tools::ToolContext::detached(CancellationToken::new()),
                    )
                    .await
                    .map(|output| output.text_content())
            }
        };
        let text = search(
            serde_json::json!({"queries": ["zebra"]}),
            on_project("webdev"),
        )
        .await
        .expect("search");
        assert!(text.contains("- **shared** (p5, global, created"), "{text}");
        assert!(text.contains("- **mine** (p5, created"), "{text}");
        assert!(!text.contains("theirs"), "{text}");
        let text = search(
            serde_json::json!({"queries": ["zebra"], "tier": "global"}),
            on_project("webdev"),
        )
        .await
        .expect("search");
        assert!(text.contains("in the global tier"), "{text}");
        assert!(text.contains("shared") && !text.contains("mine"), "{text}");
        let text = search(
            serde_json::json!({"queries": ["zebra"], "tier": "project"}),
            on_project("webdev"),
        )
        .await
        .expect("search");
        assert!(text.contains("in project 'webdev'"), "{text}");
        assert!(text.contains("mine") && !text.contains("shared"), "{text}");
        // A tier alone lists the tier.
        let text = search(serde_json::json!({"tier": "project"}), on_project("webdev"))
            .await
            .expect("listing");
        assert!(
            text.contains("1 memory in project 'webdev', most important first"),
            "{text}"
        );
        // A session on no project sees the global tier, marks nothing and has no project tier.
        let text = search(
            serde_json::json!({"queries": ["zebra"]}),
            Default::default(),
        )
        .await
        .expect("search");
        assert!(text.contains("- **shared** (p5, created"), "{text}");
        assert!(
            !text.contains("mine") && !text.contains("global,"),
            "{text}"
        );
        let refused = search(
            serde_json::json!({"queries": ["zebra"], "tier": "project"}),
            Default::default(),
        )
        .await
        .expect_err("no project tier");
        assert!(
            refused.to_string().contains("this session has no project"),
            "{refused}"
        );
    }

    /// The near-duplicate note names a memory the session can see, never one in another project,
    /// and a read of a project memory says which project it is in.
    #[tokio::test]
    async fn the_duplicate_note_and_the_read_follow_the_view() {
        let memories = store().await;
        memories
            .write(WriteRequest {
                name: "elsewhere".to_string(),
                description: Some("the deploy pipeline runs nightly at two".to_string()),
                view: MemoryView::All,
                project: Some("movie".to_string()),
                ..Default::default()
            })
            .await
            .expect("another project's note");
        let write = MemoryWriteTool {
            memories: memories.clone(),
            project: on_project("webdev"),
        };
        let output = write
            .execute(
                serde_json::json!({
                    "name": "pipeline",
                    "description": "the deploy pipeline runs nightly at two",
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        assert!(
            !output
                .text_content()
                .contains("already says something very similar"),
            "{}",
            output.text_content()
        );
        let output = write
            .execute(
                serde_json::json!({
                    "name": "pipeline-again",
                    "description": "the deploy pipeline runs nightly at two",
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("write");
        assert!(
            output
                .text_content()
                .contains("'pipeline' already says something very similar"),
            "{}",
            output.text_content()
        );
        let read = MemoryReadTool {
            memories: memories.clone(),
            project: on_project("webdev"),
        }
        .execute(
            serde_json::json!({"name": "pipeline"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("read");
        assert!(
            read.text_content().contains("Project: webdev"),
            "{}",
            read.text_content()
        );
        let refused = MemoryReadTool {
            memories: memories.clone(),
            project: on_project("webdev"),
        }
        .execute(
            serde_json::json!({"name": "elsewhere"}),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect_err("another project's memory is unknown here");
        assert!(refused.to_string().contains("elsewhere"), "{refused}");
    }
}
