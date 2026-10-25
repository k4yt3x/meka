//! The `skill_*` tools: the agent's access to the installed skill store ([`crate::skills`]).
//!
//! All five gate at [`Permission::Read`], for the reason spelled out on [`crate::tools::memory`]:
//! `workspace` in meka means "may modify the user's tree", and these write to a store meka owns
//! under its own config directory. `agent_spawn` is a read-tier tool too, so the dispatcher
//! deployment these exist for runs at read permission permanently; gating them at `workspace` would
//! withhold them from the only configuration that wants them.
//!
//! `skill_write`, `skill_edit` and `skill_delete` are registered only when `[skills] agent_managed`
//! is on, and never for a sub-agent. The authorization lives in that flag rather than in the
//! permission tier.

use std::sync::Arc;

use async_trait::async_trait;

use super::{
    Tool, ToolOutput,
    util::{
        MAX_SEARCH_MATCHES, compile_user_regex, optional_str, optional_str_allowing_empty,
        require_str,
    },
};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
    skills::{self, SkillCache},
};

pub(super) struct SkillReadTool {
    /// Shared skill cache with the agent. Dispatch reads through `current().await` so the tool
    /// sees any auto-reloads that happened during the turn.
    pub(crate) skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill_read".to_string(),
            description: "Load the full content of a named skill. Skills are knowledge \
                          files that document procedures, tools, and non-standard \
                          knowledge. Call this tool with the skill name (as listed in \
                          the conversation context) to get its full instructions."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to load."
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
        let name = require_str(&input, "name", "skill_read")?;
        let skills = self.skills.current().await;

        let skill = match skills.find(&name) {
            Some(skill) => skill,
            // "No such skill" and "it is right there and meka cannot read it" call for opposite
            // responses: collapsed into "not found", a model handed a procedure whose file has a
            // typo in its frontmatter improvises one.
            None => {
                let hint = match skills.skip_reason(&name) {
                    Some(reason) => format!(
                        "Error: skill '{name}' exists on disk but could not be read: {reason}. Tell the user; \
                         they need to fix that file. Do not substitute your own version of it."
                    ),
                    None => {
                        let available: Vec<&str> =
                            skills.skills.iter().map(|s| s.name.as_str()).collect();
                        let hint = if available.is_empty() {
                            "No skills are installed.".to_string()
                        } else {
                            format!("Available skills: {}", available.join(", "))
                        };
                        format!("Error: skill '{name}' not found. {hint}")
                    }
                };
                return Ok(ToolOutput::text(hint, true));
            }
        };

        let body =
            skills::load_skill_body(skill)
                .await
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "skill_read".to_string(),
                    message: error,
                })?;

        Ok(ToolOutput::text(body, false))
    }
}

/// Resolve the skills root, or fail naming the cause rather than reporting an empty store. Mirrors
/// `require_root` in [`crate::tools::memory`].
fn require_root(cache: &SkillCache, tool_name: &str) -> Result<std::path::PathBuf> {
    cache
        .root()
        .map(|root| root.to_path_buf())
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: "skills are disabled, or no config directory resolved".to_string(),
        })
}

/// Refuse to touch a directory that holds a `SKILL.md` discovery could not parse.
///
/// Absent from the index is not the same as absent from disk: such a file is skipped, so neither
/// the model nor this tool can say what is in it, and its only copy is that file.
///
/// Answered from the index rather than by probing the filesystem: discovery has already read every
/// one of these files and recorded why each failed, and an `is_file()` probe could say "not a
/// valid skill" but never why.
///
/// The `meka skill remove` remedy is only offered when the file is one that command can reach; for
/// a broken skill under a read-only `additional_directories` root it answers "not found", so the
/// refusal names the path instead.
///
/// [`skills::write_skill`] refuses the same case independently; this exists so the refusal arrives
/// as a readable tool result rather than a tool error, and so `skill_delete` gets it too.
fn reject_unreadable(
    name: &str,
    installed: &skills::SkillIndex,
    native_root: &std::path::Path,
) -> Option<ToolOutput> {
    // A skill that loaded is not here, and that is [`skills::SkillIndex`]'s disjointness invariant
    // rather than a check of this function's own: a working `deploy` in meka's store beside a
    // broken `deploy/` in a read-only root would otherwise put the name in both halves, and
    // re-checking `find` here would leave the other readers of `skipped` to each remember it.
    //
    // A bare `skills/<name>/` with no `SKILL.md` is not here either, because discovery skips such a
    // directory silently rather than recording it: it is a half-finished `meka skill add` or the
    // residue of an interrupted write, and `write_skill` should finish it.
    let reason = installed.skip_reason(name)?;
    let remedy = match installed.location(name) {
        Some((root, source_dir)) if root != native_root => format!(
            "It lives at {}, which meka reads but does not write to, so ask the user to fix or \
             remove it there.",
            source_dir.display()
        ),
        _ => format!(
            "Use a different name, or ask the user to fix or remove it with \
             `meka skill remove {name}`."
        ),
    };
    Some(ToolOutput::text(
        format!(
            "Error: '{name}' exists on disk but its SKILL.md is not a valid skill ({reason}), so it is in no \
             index and its contents cannot be shown. Leaving it untouched rather than overwriting \
             something neither of us can see. {remedy}"
        ),
        true,
    ))
}

pub(super) struct SkillSearchTool {
    pub(crate) skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill_search".to_string(),
            description: "Search the full text of every installed skill by regex. Searches bodies \
                as well as frontmatter, so it finds skills whose one-line description does not \
                mention the term, and skills the index did not list."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Rust regex matched against each line of every skill."
                    }
                },
                "required": ["pattern"]
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
        let pattern = require_str(&input, "pattern", "skill_search")?;
        let regex = compile_user_regex(&pattern, "skill_search")?;
        let skills = self.skills.current().await;

        let mut matches = Vec::new();
        let mut truncated = false;
        for skill in skills.skills.iter() {
            let content = match tokio::fs::read_to_string(&skill.body_path).await {
                Ok(content) => content,
                Err(error) => {
                    let path = skill.body_path.display();
                    tracing::warn!("skill_search skipping {path}: {error}");
                    continue;
                }
            };
            for (index, line) in content.lines().enumerate() {
                if !regex.is_match(line) {
                    continue;
                }
                if matches.len() >= MAX_SEARCH_MATCHES {
                    truncated = true;
                    break;
                }
                matches.push(format!("{}:{}: {}", skill.name, index + 1, line.trim()));
            }
            if truncated {
                break;
            }
        }

        if matches.is_empty() {
            return Ok(ToolOutput::text(
                "No skills matched that pattern.".to_string(),
                false,
            ));
        }

        let mut out = matches.join("\n");
        if truncated {
            out.push_str(&format!(
                "\n\n(stopped at {MAX_SEARCH_MATCHES} matches; narrow the pattern to see the rest)"
            ));
        }
        Ok(ToolOutput::text(out, false))
    }
}

pub(super) struct SkillWriteTool {
    pub(crate) skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill_write".to_string(),
            description: "Create or update a skill: a reusable procedure a later session or \
                sub-agent can follow without being told again. Writing to an existing name \
                updates it: omitted fields keep their values, and a given `body` replaces the \
                whole body (`skill_edit` changes part of one). Prefer a skill over a memory for a \
                *method* rather than a fact, \
                especially one to hand to `agent_spawn`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Identifier: lowercase letters, digits and hyphens (e.g. 'triage-build-failure')."
                    },
                    "description": {
                        "type": "string",
                        "description": "One line stating what the skill is for, used in the bounded skill index. Required when creating a \
                                        skill; omit it to leave an existing skill's description \
                                        untouched."
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 9,
                        "default": crate::entry::DEFAULT_PRIORITY,
                        "description": "Lower sorts higher in the index and survives truncation. \
                                        0-2 procedures you reach for constantly, 5 default, 6-9 \
                                        rarely relevant. Omit it to leave an existing skill's \
                                        priority untouched."
                    },
                    "body": {
                        "type": "string",
                        "description": "The procedure itself, loaded only when skill_read is \
                                        called or the skill is spawned. Replaces the existing \
                                        body whole; omit it to leave it untouched, or pass an \
                                        empty string to clear it."
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
        let root = require_root(&self.skills, "skill_write")?;
        let name = require_str(&input, "name", "skill_write")?;
        // Before the join below, and again inside `write_skill`. Same layered guard `memory_write`
        // applies: these tools run at read permission, so the character class is what keeps this
        // from being an arbitrary-file-write primitive.
        skills::validate_skill_name(&name).map_err(|message| MekaError::ToolExecution {
            tool_name: "skill_write".to_string(),
            message,
        })?;
        // Omit-to-keep, like `body` and `priority` below: the only copy of a description the agent
        // can see is the `[Skills]` index's, elided to 500 characters, so requiring it would make a
        // body-only refinement rewrite a long description as its elision.
        let requested_description =
            optional_str(&input, "description", "skill_write")?.map(str::to_string);
        // `Some("")` is an explicit request to clear the body, as it is on `memory_write`.
        let body: Option<String> =
            optional_str_allowing_empty(&input, "body", "skill_write")?.map(str::to_string);
        let requested_priority = match input.get("priority") {
            Some(serde_json::Value::Null) | None => None,
            Some(value) => {
                let raw = value.as_i64().ok_or_else(|| MekaError::ToolExecution {
                    tool_name: "skill_write".to_string(),
                    message: format!("'priority' must be a whole number, got {value}"),
                })?;
                Some(crate::entry::parse_priority(Some(raw), "skill", &name))
            }
        };

        let installed = self.skills.current().await;
        // Omitted means "leave it alone", as `PUT /v1/skills` has it: reading the absence as the
        // default would demote a prioritized skill every time the agent refined its text, and
        // priority decides which entries the index cap drops.
        let priority = requested_priority.unwrap_or_else(|| {
            installed
                .find(&name)
                .map_or(crate::entry::DEFAULT_PRIORITY, |skill| skill.priority)
        });
        // Unreadable first, because it is the more specific answer: a file that is both foreign and
        // unparseable needs its parse error named, which the plain foreign refusal cannot carry.
        //
        // Both run before the description is resolved: `SkillIndex` keeps loaded and skipped
        // skills disjoint, so a skill whose `SKILL.md` is present but unparseable is absent from
        // `installed.find`, and resolving first would answer a body-only write with "no skill
        // named 'x' exists".
        if let Some(refusal) = reject_unreadable(&name, &installed, &root) {
            return Ok(refusal);
        }
        if let Some(refusal) = skills::refuse_foreign_write(&installed, &name, &root) {
            return Ok(ToolOutput::text(
                format!("Error: {}", refusal.where_it_lives()),
                true,
            ));
        }

        let existing = installed.find(&name);
        // Resolved against the same snapshot `priority` used, so both answer "what does the skill
        // already say" from one read. A write that creates the skill has nothing to keep, so the
        // description is required there and the refusal says which case this is.
        let description = match requested_description {
            Some(description) => description,
            None => match existing {
                // Truncated to what `write_skill` will accept, because it is being carried rather
                // than authored: `parse_skill_definition` only warns about an over-long
                // description, so a skill imported from a repository loads with one, and handing
                // it straight back would make `write_skill` refuse over a field the call never
                // mentioned. Truncated to the spec's 1024, not the index's 500, which would
                // reintroduce the silent rewrite omit-to-keep exists to prevent.
                Some(skill) => skill
                    .description
                    .char_indices()
                    .nth(skills::MAX_DESCRIPTION_CHARS)
                    .map_or_else(
                        || skill.description.clone(),
                        |(cut, _)| skill.description[..cut].to_string(),
                    ),
                None => {
                    return Err(MekaError::ToolExecution {
                        tool_name: "skill_write".to_string(),
                        message: format!(
                            "no skill named '{name}' exists, so a description is required to create it"
                        ),
                    });
                }
            },
        };
        // On the blocking pool, for the same reason `memory_write` is: the write goes through
        // `write_file_atomic`, which `fsync`s, and a `fsync` parks the calling thread for as long
        // as the filesystem takes. On a runtime worker that is every other session's turn waiting.
        let written = {
            let root = root.clone();
            let name = name.clone();
            let description = description.clone();
            let body = body.clone();
            tokio::task::spawn_blocking(move || {
                skills::write_skill(
                    &root,
                    &name,
                    &description,
                    priority,
                    Some(AGENT_AUTHOR),
                    body.as_deref(),
                )
            })
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "skill_write".to_string(),
                message: format!("write task failed: {error}"),
            })?
            .map_err(|message| MekaError::ToolExecution {
                tool_name: "skill_write".to_string(),
                message,
            })?
        };
        // The write is only visible to the next `current()` if the cache notices it, and a
        // `(mtime, size)` snapshot cannot see a same-tick rewrite of the same length; the
        // dispatcher flow writes a skill and hands it to `agent_spawn(skill:)` in the same turn.
        self.skills.invalidate().await;

        let skills::WrittenSkill {
            skill,
            previous,
            text,
        } = written;
        // From the text the write read under its lock: whether there was a body to keep rather
        // than merely a file, or the confirmation would claim to have kept the body of a skill
        // that had none.
        let kept_existing_body = body.is_none()
            && previous
                .as_deref()
                .and_then(crate::entry::split_frontmatter)
                .is_some_and(|(_, body)| !body.trim().is_empty());

        let path = skill.body_path.display();
        tracing::info!("saved skill to {path}");
        Ok(ToolOutput::text(
            // The rank the *file* now carries, read back from the bytes rather than echoed from
            // the request. Deliberately promises reachability by name rather than a place in the
            // index: the index is capped, so a low-priority skill in a large store may not be
            // listed there, and `skill_read` / `agent_spawn` work either way.
            format!(
                "Saved skill '{}' (priority {}){}. From the next turn on you can load it with \
                 skill_read, or hand it to a worker with agent_spawn(skill: \"{}\").",
                name,
                skill.priority,
                if kept_existing_body {
                    ", keeping the existing body"
                } else {
                    ""
                },
                name
            ),
            false,
        )
        .with_metadata(crate::frontend::ToolOutputMetadata::Diff {
            path: Some(skill.body_path),
            old_text: previous,
            new_text: Some(text),
        }))
    }
}

/// Stamped into the `author` frontmatter of a skill the agent *creates*.
///
/// Only on creation: [`skills::write_skill`] keeps an existing `author`, so refining a skill you
/// wrote does not quietly reassign it. Informational only, using a field skills already had. It
/// exists so `meka skill list` is legible about where an entry came from, not as a guard: nothing
/// branches on it.
const AGENT_AUTHOR: &str = "meka (agent-authored)";

pub(super) struct SkillEditTool {
    pub(crate) skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillEditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill_edit".to_string(),
            description: "Replace an exact string in a skill's body and leave the rest as it is. \
                Refused when `old_string` matches more than once unless `replace_all` is true. \
                For a small change; `skill_write` replaces the whole body."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill, as listed in the skills index."
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find in the body as stored; the header skill_read prepends is not part of it."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement; an empty string deletes `old_string`."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "default": false,
                        "description": "Change every occurrence. When false and `old_string` matches more than once, the edit is refused as ambiguous. Default: false."
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
        let root = require_root(&self.skills, "skill_edit")?;
        let name = require_str(&input, "name", "skill_edit")?;
        // Write rules, as `skill_write` applies them: an edit rewrites the file, and the name is
        // joined onto the root.
        skills::validate_skill_name(&name).map_err(|message| MekaError::ToolExecution {
            tool_name: "skill_edit".to_string(),
            message,
        })?;
        let old_string = require_str(&input, "old_string", "skill_edit")?;
        let new_string =
            super::util::require_str_allowing_empty(&input, "new_string", "skill_edit")?;
        let replace_all = match input.get("replace_all") {
            None | Some(serde_json::Value::Null) => false,
            Some(serde_json::Value::Bool(replace_all)) => *replace_all,
            Some(value) => {
                return Err(MekaError::ToolExecution {
                    tool_name: "skill_edit".to_string(),
                    message: format!("'replace_all' must be true or false, got {value}"),
                });
            }
        };

        let installed = self.skills.current().await;
        // Unreadable first, for the reason `skill_write` gives.
        if let Some(refusal) = reject_unreadable(&name, &installed, &root) {
            return Ok(refusal);
        }
        if let Some(refusal) = skills::refuse_foreign_write(&installed, &name, &root) {
            return Ok(ToolOutput::text(
                format!("Error: {}", refusal.where_it_lives()),
                true,
            ));
        }
        let Some(skill) = installed.find(&name) else {
            return Err(MekaError::ToolExecution {
                tool_name: "skill_edit".to_string(),
                message: format!(
                    "no skill named '{}'.{}",
                    name,
                    crate::tools::did_you_mean_hint(
                        &name,
                        installed.skills.iter().map(|skill| skill.name.as_str())
                    )
                ),
            });
        };
        // The body as stored rather than as `skill_read` shows it: the base-directory header is
        // a render-time line, so an anchor matched against it is one the file cannot contain.
        let body =
            skills::load_skill_source(skill)
                .await
                .map_err(|message| MekaError::ToolExecution {
                    tool_name: "skill_edit".to_string(),
                    message,
                })?;
        let (edited, count, first_match_byte) =
            match super::util::replace_anchored(&body, &old_string, &new_string, replace_all) {
                Ok(applied) => applied,
                Err(super::util::AnchorMiss::NotFound) => {
                    // Named when it is the likely cause, as `file_edit` does: a hand-written
                    // `SKILL.md` can carry CRLF, and nothing in a bare "not found" points at an
                    // invisible difference.
                    let crlf_hint = if body.contains("\r\n")
                        && old_string.contains('\n')
                        && !old_string.contains("\r\n")
                    {
                        " (this skill uses CRLF line endings; old_string uses LF)"
                    } else {
                        ""
                    };
                    return Ok(ToolOutput::text(
                        format!(
                            "Error: '{}' not found in the body of skill '{name}'{crlf_hint}",
                            super::util::truncate_string(&old_string, 100)
                        ),
                        true,
                    ));
                }
                Err(super::util::AnchorMiss::Ambiguous(count)) => {
                    return Ok(ToolOutput::text(
                        format!(
                            "Error: '{}' matches {count} times in skill '{name}'; add surrounding \
                             context to make old_string unique, or set replace_all=true to change \
                             every occurrence",
                            super::util::truncate_string(&old_string, 100)
                        ),
                        true,
                    ));
                }
            };

        // On the blocking pool for the reason `skill_write` gives, and compared against the body
        // read above under the store lock: the read happened outside it, so a write that landed in
        // between is refused rather than overwritten.
        let outcome = {
            let root = root.clone();
            let name = name.clone();
            let body = body.clone();
            let edited = edited.clone();
            tokio::task::spawn_blocking(move || {
                skills::write_skill_body(&root, &name, &body, &edited)
            })
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "skill_edit".to_string(),
                message: format!("write task failed: {error}"),
            })?
            .map_err(|message| MekaError::ToolExecution {
                tool_name: "skill_edit".to_string(),
                message,
            })?
        };
        // See the note in `skill_write`; on a refusal the store moved under the index, which is
        // one more reason to read it again.
        self.skills.invalidate().await;
        let written = match outcome {
            skills::BodyWrite::Saved(written) => *written,
            skills::BodyWrite::ChangedUnderneath => {
                return Ok(ToolOutput::text(
                    format!(
                        "Error: skill '{name}' changed while the edit was computed; read it again \
                         and retry"
                    ),
                    true,
                ));
            }
            skills::BodyWrite::Gone => {
                return Err(MekaError::ToolExecution {
                    tool_name: "skill_edit".to_string(),
                    message: format!("skill '{name}' was deleted while the edit was computed"),
                });
            }
        };

        let path = written.skill.body_path.display();
        tracing::info!("edited skill at {path}");
        // The stored body opens with the blank line the renderer leaves after the fence; the
        // snippet numbers the procedure's own lines from 1.
        let (shown, shown_match_byte) = match edited.strip_prefix('\n') {
            Some(rest) if first_match_byte > 0 => (rest, first_match_byte - 1),
            _ => (edited.as_str(), first_match_byte),
        };
        let snippet = super::file::build_context_snippet(shown, shown_match_byte, 3);
        Ok(ToolOutput::text(
            format!("Edited skill '{name}': {count} occurrence(s).\n\n{snippet}"),
            false,
        )
        .with_metadata(crate::frontend::ToolOutputMetadata::Diff {
            path: Some(written.skill.body_path),
            old_text: written.previous,
            new_text: Some(written.text),
        }))
    }
}

pub(super) struct SkillDeleteTool {
    pub(crate) skills: Arc<SkillCache>,
}

#[async_trait]
impl Tool for SkillDeleteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill_delete".to_string(),
            description: "Delete a skill permanently, including any files bundled with it. To \
                revise one instead, call skill_write or skill_edit with the same name."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to delete."
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
        let root = require_root(&self.skills, "skill_delete")?;
        let name = require_str(&input, "name", "skill_delete")?;
        // Lookup rules: a skill whose name predates the spec is still listed and still readable, so
        // it has to be removable too.
        skills::validate_addressable_name(&name).map_err(|message| MekaError::ToolExecution {
            tool_name: "skill_delete".to_string(),
            message,
        })?;

        let installed = self.skills.current().await;
        // Unreadable first, for the reason `skill_write` gives.
        if let Some(refusal) = reject_unreadable(&name, &installed, &root) {
            return Ok(refusal);
        }
        if let Some(refusal) = skills::refuse_foreign_delete(&installed, &name, &root) {
            return Ok(ToolOutput::text(
                format!("Error: {}", refusal.where_it_lives()),
                true,
            ));
        }
        if installed.find(&name).is_none() {
            return Ok(ToolOutput::text(
                format!("Error: skill '{name}' not found."),
                true,
            ));
        }

        let removed =
            skills::delete_skill(&root, &name).map_err(|message| MekaError::ToolExecution {
                tool_name: "skill_delete".to_string(),
                message,
            })?;
        // See the note in `skill_write`: the index must not keep listing a skill that is gone.
        self.skills.invalidate().await;

        let path = removed.dir.display();
        tracing::info!("deleted skill {path}");
        let output = ToolOutput::text(
            format!("Deleted skill '{name}' and everything in its directory."),
            false,
        );
        // The frontend's one chance to show what the skill said: the file is gone. A skill whose
        // file could not be read is reported removed with no change to draw.
        Ok(match removed.text {
            Some(text) => output.with_metadata(crate::frontend::ToolOutputMetadata::Diff {
                path: Some(removed.file),
                old_text: Some(text),
                new_text: None,
            }),
            None => output,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tokio_util::sync::CancellationToken;

    use super::*;

    fn write_skill(root: &Path, name: &str, body: &str) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("create dir");
        std::fs::write(dir.join("SKILL.md"), body).expect("write SKILL.md");
    }

    #[tokio::test]
    async fn skill_tool_unknown_skill() {
        let tool = SkillReadTool {
            skills: SkillCache::for_root(None),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "nonexistent-skill-xyz"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok with error output");

        assert!(result.is_error);
        let text = crate::conversation::ContentBlock::tool_result_text_content(&result.content);
        assert!(text.contains("not found"));
    }

    #[tokio::test]
    async fn skill_tool_missing_name() {
        let tool = SkillReadTool {
            skills: SkillCache::for_root(None),
        };
        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn skill_tool_prepends_context_header() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "demo",
            "---\ndescription: x\n---\nRun helper.py to do the thing.\n",
        );
        let tool = SkillReadTool {
            skills: SkillCache::for_root(Some(temp.path().to_path_buf())),
        };
        let result = tool
            .execute(
                serde_json::json!({"name": "demo"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should load");

        assert!(!result.is_error);
        let text = crate::conversation::ContentBlock::tool_result_text_content(&result.content);
        assert!(text.starts_with("Base directory for this skill and its bundled files:"));
        assert!(text.contains(&temp.path().join("demo").display().to_string()));
        assert!(text.contains("Run helper.py to do the thing."));
    }

    #[test]
    fn an_index_tells_absent_from_unreadable() {
        let skill = crate::skills::Skill {
            name: "foo".to_string(),
            source_dir: std::path::PathBuf::from("/tmp"),
            description: "desc".to_string(),
            license: None,
            compatibility: None,
            allowed_tools: None,
            priority: crate::entry::DEFAULT_PRIORITY,
            metadata: None,
            extra: yaml_serde::Mapping::new(),
            conformance: crate::skills::Conformance::default(),
            body_path: std::path::PathBuf::from("/tmp/SKILL.md"),
            root: std::path::PathBuf::from("/tmp"),
        };
        let index = crate::skills::SkillIndex {
            skills: vec![skill],
            skipped: vec![crate::skills::SkippedSkill {
                name: "broken".to_string(),
                reason: "missing YAML frontmatter".to_string(),
                root: std::path::PathBuf::from("/tmp"),
            }],
        };
        assert!(index.find("foo").is_some());
        assert!(index.find("bar").is_none());
        // The distinction the whole index exists for: a name that is absent and a name whose file
        // is unreadable are different answers, and only one of them is "no such skill".
        assert_eq!(
            index.skip_reason("broken"),
            Some("missing YAML frontmatter")
        );
        assert_eq!(index.skip_reason("bar"), None);
        assert!(index.find("broken").is_none());
    }

    #[test]
    fn write_skill_helper() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "test",
            "---\ndescription: x\nwhen_to_use: y\n---\nbody\n",
        );
        assert!(temp.path().join("test/SKILL.md").exists());
    }

    fn cache_at(temp: &tempfile::TempDir) -> Arc<SkillCache> {
        SkillCache::for_root(Some(temp.path().to_path_buf()))
    }

    /// A body-only write reaches a skill whose stored description exceeds the spec cap, and one
    /// whose file is present but unparseable is told so rather than told it does not exist.
    ///
    /// `SkillIndex` keeps loaded and skipped skills disjoint, so resolving the description above
    /// the refusal gates would report an unparseable skill absent; and `parse_skill_definition`
    /// only warns about an over-long description, so handing a stored one back to `write_skill`
    /// unchanged would refuse the write over a field the call never mentioned.
    #[tokio::test]
    async fn a_body_only_write_survives_an_awkward_stored_description() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("skills");
        std::fs::create_dir_all(&root).expect("root");

        // Longer than the spec's cap, which discovery accepts with a warning.
        let long = "d".repeat(skills::MAX_DESCRIPTION_CHARS + 76);
        let dir = root.join("imported");
        std::fs::create_dir_all(&dir).expect("skill dir");
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: imported\ndescription: {long}\n---\n\nORIGINAL\n"),
        )
        .expect("write");

        let skills_cache = SkillCache::new(Some(root.clone()), Vec::new());
        let write = SkillWriteTool {
            skills: skills_cache.clone(),
        };

        let refined = run(
            &write,
            serde_json::json!({"name": "imported", "body": "REFINED"}),
        )
        .await;
        assert!(
            !refined.is_error,
            "a body-only write must not be refused over the description it is carrying: {}",
            refined.text_content()
        );
        let stored = std::fs::read_to_string(dir.join("SKILL.md")).expect("read back");
        assert!(
            stored.contains("REFINED"),
            "the body must have landed: {stored}"
        );

        // A present-but-unparseable skill is named for what it is.
        let broken = root.join("broken");
        std::fs::create_dir_all(&broken).expect("broken dir");
        std::fs::write(
            broken.join("SKILL.md"),
            "---\ndescription: [unclosed\n---\nBODY\n",
        )
        .expect("write");
        skills_cache.invalidate().await;

        let result = run(&write, serde_json::json!({"name": "broken", "body": "new"})).await;
        let message = result.text_content();
        assert!(
            !message.contains("does not exist") && !message.contains("is required to create it"),
            "a skill whose file is on disk must not be reported as absent: {message}"
        );
    }

    /// The tools report the skill file before and after, with its path, a delete with no after,
    /// so every surface draws the change.
    #[tokio::test]
    async fn a_write_and_a_delete_report_the_file_before_and_after() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("skills");
        std::fs::create_dir_all(&root).expect("root");
        let skills = SkillCache::new(Some(root.clone()), Vec::new());
        let write = SkillWriteTool {
            skills: skills.clone(),
        };
        let change = |output: &ToolOutput| match output.frontend_metadata.clone() {
            Some(crate::frontend::ToolOutputMetadata::Diff {
                path,
                old_text,
                new_text,
            }) => (path, old_text, new_text),
            other => panic!("a change is reported: {other:?} {}", output.text_content()),
        };
        let file = root.join("demo").join("SKILL.md");

        let created = run(
            &write,
            serde_json::json!({"name": "demo", "description": "d", "body": "one"}),
        )
        .await;
        assert!(!created.is_error, "{}", created.text_content());
        let (path, old, new) = change(&created);
        assert_eq!(path.as_deref(), Some(file.as_path()));
        assert!(old.is_none(), "a create has no before");
        let new = new.expect("the file as written");
        assert_eq!(new, std::fs::read_to_string(&file).expect("read back"));

        let refined = run(&write, serde_json::json!({"name": "demo", "body": "two"})).await;
        assert!(!refined.is_error, "{}", refined.text_content());
        let (_, old, after) = change(&refined);
        assert_eq!(old.as_deref(), Some(new.as_str()));
        let after = after.expect("the file as rewritten");
        assert!(after.contains("two"), "{after}");

        let delete = SkillDeleteTool { skills };
        let deleted = run(&delete, serde_json::json!({"name": "demo"})).await;
        assert!(!deleted.is_error, "{}", deleted.text_content());
        let (path, old, new) = change(&deleted);
        assert_eq!(path.as_deref(), Some(file.as_path()));
        assert_eq!(old.as_deref(), Some(after.as_str()));
        assert!(new.is_none(), "a delete has no after");
    }

    /// A non-string `body` is refused, not read as "leave it alone": `as_str` would put
    /// `body: ["a", "b"]` down the omit-to-keep path and report success while storing nothing the
    /// caller sent.
    #[tokio::test]
    async fn a_non_string_body_is_refused_rather_than_treated_as_absent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("skills");
        std::fs::create_dir_all(&root).expect("root");
        let write = SkillWriteTool {
            skills: SkillCache::new(Some(root), Vec::new()),
        };

        let result = write
            .execute(
                serde_json::json!({
                    "name": "listy",
                    "description": "d",
                    "body": ["line one", "line two"],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        let message = match result {
            Err(error) => error.to_string(),
            Ok(output) => output.text_content(),
        };
        assert!(
            message.contains("'body' must be a string"),
            "a list body must be refused by name: {message}"
        );
    }

    /// An omitted `description` keeps the stored one, and is refused when there is none to keep.
    ///
    /// The only copy of a description an agent can see is the `[Skills]` index's, elided to 500
    /// characters, so requiring the field would make a body-only refinement resend the elision.
    #[tokio::test]
    async fn omitting_a_skill_description_keeps_the_stored_one() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("skills");
        std::fs::create_dir_all(&root).expect("root");
        let skills = SkillCache::new(Some(root.clone()), Vec::new());
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        // Longer than the index elides to, which is the case that made the loss invisible.
        let long = format!("how to triage a build failure {}", "in detail ".repeat(90));
        let created = run(
            &write,
            serde_json::json!({"name": "triage", "description": long, "body": "FIRST"}),
        )
        .await;
        assert!(!created.is_error, "{}", created.text_content());

        let refined = run(
            &write,
            serde_json::json!({"name": "triage", "body": "SECOND"}),
        )
        .await;
        assert!(!refined.is_error, "{}", refined.text_content());

        let stored = skills.current().await;
        let skill = stored.find("triage").expect("still installed");
        assert_eq!(
            skill.description,
            crate::entry::normalize_description(&long),
            "refining the body must not rewrite the description"
        );

        // Nothing to keep: creating a skill still needs one, and the refusal says so.
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
    }

    /// Both write doors refuse a skill that lives in a read-only extra root, and neither writes
    /// anything anywhere as a side effect, which is the half worth a test.
    #[tokio::test]
    async fn write_and_delete_refuse_a_skill_from_a_read_only_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let native = temp.path().join("native");
        let shared = temp.path().join("shared");
        std::fs::create_dir_all(&native).expect("native");
        write_skill(
            &shared,
            "borrowed",
            "---\ndescription: theirs\n---\nTHEIR PROCEDURE\n",
        );
        let skills = SkillCache::new(Some(native.clone()), vec![shared.clone()]);

        let write = SkillWriteTool {
            skills: skills.clone(),
        };
        let result = run(
            &write,
            serde_json::json!({"name": "borrowed", "description": "mine", "body": "MINE"}),
        )
        .await;
        assert!(result.is_error);
        assert!(
            result.text_content().contains("does not write to"),
            "{}",
            result.text_content()
        );

        let delete = SkillDeleteTool {
            skills: skills.clone(),
        };
        let result = run(&delete, serde_json::json!({"name": "borrowed"})).await;
        assert!(result.is_error);
        assert!(result.text_content().contains("does not write to"));

        // Neither refusal touched the foreign file, and neither created a shadow copy.
        assert!(
            std::fs::read_to_string(shared.join("borrowed/SKILL.md"))
                .expect("still there")
                .contains("THEIR PROCEDURE")
        );
        assert!(
            !native.join("borrowed").exists(),
            "a shadowing copy must not be created in meka's own root"
        );

        // A name meka does own is unaffected.
        let result = run(
            &write,
            serde_json::json!({"name": "ours", "description": "d", "body": "b"}),
        )
        .await;
        assert!(!result.is_error, "{}", result.text_content());
    }

    /// The read-only rule covers a foreign skill whose `SKILL.md` does not parse, and the refusal
    /// sends the reader to the file rather than to a command that cannot reach it.
    ///
    /// Compared against the loaded skills alone, an unparseable file in an `additional_directories`
    /// root is a name nothing has an opinion about and gets shadowed silently, the worst case
    /// to shadow since the original is then reported nowhere; and `meka skill remove` answers
    /// "not found" for a file meka does not own.
    #[tokio::test]
    async fn a_broken_skill_in_a_read_only_root_is_neither_shadowed_nor_misdirected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let native = temp.path().join("native");
        let shared = temp.path().join("shared");
        std::fs::create_dir_all(&native).expect("native");
        write_skill(
            &shared,
            "wrecked",
            "---\ndescription: [unclosed\n---\nTHEIRS\n",
        );
        let skills = SkillCache::new(Some(native.clone()), vec![shared.clone()]);

        let write = SkillWriteTool {
            skills: skills.clone(),
        };
        let result = run(
            &write,
            serde_json::json!({"name": "wrecked", "description": "mine", "body": "MINE"}),
        )
        .await;
        assert!(result.is_error, "{}", result.text_content());
        let text = result.text_content();
        assert!(
            text.contains(&shared.join("wrecked").display().to_string()),
            "the refusal must name where the file is: {text}"
        );
        assert!(
            !text.contains("meka skill remove"),
            "that command cannot reach a read-only root: {text}"
        );
        assert!(
            !native.join("wrecked").exists(),
            "an unparseable foreign skill must not be shadowed either"
        );

        // A broken skill meka *does* own still gets the remedy that works for it.
        write_skill(&native, "ours-wrecked", "no frontmatter\n");
        skills.invalidate().await;
        let result = run(
            &write,
            serde_json::json!({"name": "ours-wrecked", "description": "mine"}),
        )
        .await;
        assert!(result.is_error);
        assert!(
            result
                .text_content()
                .contains("meka skill remove ours-wrecked"),
            "{}",
            result.text_content()
        );
    }

    /// A name that loaded is writable, whatever a shadowed copy of it elsewhere looks like.
    ///
    /// Roots merge first-wins and the skip list records every failure, so meka's own working
    /// `deploy` and a broken `deploy/` in a read-only root would put one name in both halves of
    /// the index, and `reject_unreadable` answering from the skipped half alone would refuse every
    /// write and delete of a skill plainly in the index.
    #[tokio::test]
    async fn a_skill_that_loaded_is_writable_though_a_broken_copy_shadows_it() {
        let temp = tempfile::tempdir().expect("tempdir");
        let native = temp.path().join("native");
        let shared = temp.path().join("shared");
        write_skill(
            &native,
            "deploy",
            "---\nname: deploy\ndescription: mine and working\n---\nMINE\n",
        );
        write_skill(
            &shared,
            "deploy",
            "---\ndescription: [unclosed\n---\nTHEIRS\n",
        );
        let skills = SkillCache::new(Some(native.clone()), vec![shared.clone()]);
        let index = skills.current().await;
        assert!(index.find("deploy").is_some(), "the native copy wins");
        assert_eq!(
            index.skip_reason("deploy"),
            None,
            "and the shadowed broken copy must not also claim the name"
        );

        let write = SkillWriteTool {
            skills: skills.clone(),
        };
        let result = run(
            &write,
            serde_json::json!({"name": "deploy", "description": "refined", "body": "MINE2"}),
        )
        .await;
        assert!(!result.is_error, "{}", result.text_content());
        assert!(
            std::fs::read_to_string(native.join("deploy/SKILL.md"))
                .expect("still there")
                .contains("MINE2"),
            "the write must land in meka's own root"
        );
        assert!(
            std::fs::read_to_string(shared.join("deploy/SKILL.md"))
                .expect("still there")
                .contains("THEIRS"),
            "and must not touch the read-only one"
        );

        skills.invalidate().await;
        let delete = SkillDeleteTool {
            skills: skills.clone(),
        };
        let result = run(&delete, serde_json::json!({"name": "deploy"})).await;
        assert!(!result.is_error, "{}", result.text_content());
        assert!(!native.join("deploy").exists(), "removed from meka's store");
        assert!(shared.join("deploy").exists(), "left alone elsewhere");
    }

    /// A skill whose file is unreadable is reported as unreadable, not as absent: collapsed into
    /// "not found", a model handed a procedure with a typo in its frontmatter improvises its own
    /// version.
    #[tokio::test]
    async fn read_says_a_broken_skill_is_broken_rather_than_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "broken", "no frontmatter at all\nKEEP ME\n");
        write_skill(temp.path(), "fine", "---\ndescription: d\n---\nbody\n");
        let skills = cache_at(&temp);

        let read = SkillReadTool {
            skills: skills.clone(),
        };
        let result = run(&read, serde_json::json!({"name": "broken"})).await;
        assert!(result.is_error);
        let text = result.text_content();
        assert!(
            text.contains("could not be read"),
            "reported as missing: {text}"
        );
        assert!(
            !text.contains("not found"),
            "a file that is right there is not 'not found': {text}"
        );
        // And the reason, which is the part the model can act on by telling the user.
        assert!(text.contains("frontmatter"), "{text}");

        // A name that really is absent still gets the plain answer, with the available list.
        let result = run(&read, serde_json::json!({"name": "absent"})).await;
        let text = result.text_content();
        assert!(text.contains("not found"), "{text}");
        assert!(text.contains("fine"), "{text}");
    }

    /// `skill_write` surfaces the store's refusal of a `metadata` it cannot record in, rather than
    /// writing and then explaining to the model why the rank it asked for did not apply.
    #[tokio::test]
    async fn write_surfaces_the_refusal_of_a_metadata_it_cannot_record_in() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "verbatim",
            "---\nname: verbatim\ndescription: original\nmetadata: none\n---\nBODY\n",
        );
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        let error = write
            .execute(
                serde_json::json!({"name": "verbatim", "description": "refined", "priority": 1}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("must refuse rather than write and explain");
        assert!(error.to_string().contains("not a map"), "{error}");

        // The file is untouched, and an ordinary skill still reports the rank it was given.
        let result = run(
            &write,
            serde_json::json!({"name": "ordinary", "description": "d", "priority": 1}),
        )
        .await;
        assert!(
            result.text_content().contains("(priority 1)"),
            "{}",
            result.text_content()
        );
    }

    async fn run(tool: &dyn Tool, input: serde_json::Value) -> ToolOutput {
        tool.execute(
            input,
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("tool should return Ok")
    }

    #[tokio::test]
    async fn skill_write_creates_a_loadable_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        let result = run(
            &write,
            serde_json::json!({
                "name": "triage",
                "description": "How to triage a build failure",
                "priority": 2,
                "body": "1. Read the log.\n2. Bisect.\n"
            }),
        )
        .await;
        assert!(!result.is_error, "{}", result.text_content());

        // Round-trips through discovery rather than just checking the bytes: what matters is that
        // the file this wrote is one the parser accepts, since a skill that fails to parse is
        // silently skipped and would look identical from the write side.
        let discovered = skills.current().await;
        let skill = discovered
            .skills
            .iter()
            .find(|skill| skill.name == "triage")
            .expect("written skill must be discoverable");
        assert_eq!(skill.description, "How to triage a build failure");
        assert_eq!(skill.priority, 2);
        assert_eq!(skill.author().as_deref(), Some(AGENT_AUTHOR));

        let read = SkillReadTool { skills };
        let body = run(&read, serde_json::json!({"name": "triage"}))
            .await
            .text_content();
        assert!(body.contains("1. Read the log."), "{}", body);
    }

    /// An omitted `body` is "leave it alone", not "make it empty". A call that only re-prioritizes
    /// a skill is one the schema invites, and treating the absent field as an empty string would
    /// delete the whole procedure on exactly that call.
    #[tokio::test]
    async fn skill_write_without_body_keeps_the_existing_one() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        run(
            &write,
            serde_json::json!({
                "name": "keep",
                "description": "first",
                "body": "PRECIOUS PROCEDURE"
            }),
        )
        .await;
        let result = run(
            &write,
            serde_json::json!({"name": "keep", "description": "second", "priority": 1}),
        )
        .await;
        assert!(result.text_content().contains("keeping the existing body"));

        let read = SkillReadTool {
            skills: skills.clone(),
        };
        let body = run(&read, serde_json::json!({"name": "keep"}))
            .await
            .text_content();
        assert!(body.contains("PRECIOUS PROCEDURE"), "{}", body);

        let discovered = skills.current().await;
        let skill = discovered
            .skills
            .iter()
            .find(|s| s.name == "keep")
            .expect("keep");
        assert_eq!(skill.description, "second");
        assert_eq!(skill.priority, 1);
    }

    #[tokio::test]
    async fn skill_write_clears_the_body_on_an_explicit_empty_string() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        run(
            &write,
            serde_json::json!({"name": "clear", "description": "d", "body": "GONE"}),
        )
        .await;
        run(
            &write,
            serde_json::json!({"name": "clear", "description": "d", "body": ""}),
        )
        .await;

        let read = SkillReadTool { skills };
        let body = run(&read, serde_json::json!({"name": "clear"}))
            .await
            .text_content();
        assert!(!body.contains("GONE"), "{}", body);
        // Pinned, not merely "GONE is absent": a skill *is* its body, so an emptied one falls back
        // to a bare heading rather than leaving `skill_read` with only the directory header.
        assert!(body.contains("# clear"), "{}", body);
    }

    /// The name is joined onto the skills root, so this is the guard that keeps a read-permission
    /// tool from writing anywhere on disk.
    #[tokio::test]
    async fn skill_write_rejects_a_traversing_name() {
        let temp = tempfile::tempdir().expect("tempdir");
        let write = SkillWriteTool {
            skills: cache_at(&temp),
        };
        let result = write
            .execute(
                serde_json::json!({"name": "../escape", "description": "d"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(result.is_err(), "a traversing name must not reach the disk");
        assert!(
            !temp
                .path()
                .parent()
                .is_some_and(|p| p.join("escape").exists())
        );
    }

    /// Bundled files are part of a skill, so a delete that left them behind would produce a broken
    /// half-skill that discovery keeps warning about.
    #[tokio::test]
    async fn skill_delete_removes_bundled_files_too() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "bundled", "---\ndescription: x\n---\nbody\n");
        std::fs::write(temp.path().join("bundled/helper.sh"), "#!/bin/sh\n").expect("write helper");

        let delete = SkillDeleteTool {
            skills: cache_at(&temp),
        };
        let result = run(&delete, serde_json::json!({"name": "bundled"})).await;
        assert!(!result.is_error, "{}", result.text_content());
        assert!(!temp.path().join("bundled").exists());
    }

    #[tokio::test]
    async fn skill_delete_reports_a_missing_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        let delete = SkillDeleteTool {
            skills: cache_at(&temp),
        };
        let result = run(&delete, serde_json::json!({"name": "absent"})).await;
        assert!(result.is_error);
        assert!(result.text_content().contains("not found"));
    }

    /// A directory whose `SKILL.md` does not parse is absent from every index, so "not found" is a
    /// lie the user can disprove with `ls`. Both tools refuse it, and say which case it is.
    #[tokio::test]
    async fn both_tools_distinguish_a_broken_skill_from_a_missing_one() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "broken", "no frontmatter at all\nKEEP ME\n");
        let skills = cache_at(&temp);

        let delete = SkillDeleteTool {
            skills: skills.clone(),
        };
        let result = run(&delete, serde_json::json!({"name": "broken"})).await;
        assert!(result.is_error);
        let text = result.text_content();
        assert!(text.contains("not a valid skill"), "{text}");
        assert!(!text.contains("not found"), "{text}");

        let write = SkillWriteTool { skills };
        let result = run(
            &write,
            serde_json::json!({"name": "broken", "description": "d", "body": "new"}),
        )
        .await;
        assert!(result.is_error);
        assert!(result.text_content().contains("not a valid skill"));

        assert!(
            std::fs::read_to_string(temp.path().join("broken/SKILL.md"))
                .expect("read")
                .contains("KEEP ME")
        );
    }

    /// Searching bodies is the whole point: a skill whose description says nothing about the term
    /// is exactly the one the pushed index cannot help with.
    #[tokio::test]
    async fn skill_search_matches_bodies_not_just_descriptions() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "deploy",
            "---\ndescription: Ship it\n---\nRun kubectl rollout status.\n",
        );
        write_skill(
            temp.path(),
            "unrelated",
            "---\ndescription: Something else\n---\nNothing to see.\n",
        );

        let search = SkillSearchTool {
            skills: cache_at(&temp),
        };
        let text = run(&search, serde_json::json!({"pattern": "kubectl"}))
            .await
            .text_content();
        assert!(text.contains("deploy:"), "{}", text);
        assert!(!text.contains("unrelated"), "{}", text);

        let text = run(&search, serde_json::json!({"pattern": "zzz-no-match"}))
            .await
            .text_content();
        assert!(text.contains("No skills matched"), "{}", text);
    }

    /// Priority arrives through three doors (frontmatter, CLI flag, this schema) and they do not
    /// agree by accident: a number outside the range is clamped, but a non-number is refused
    /// outright rather than silently becoming the default.
    #[tokio::test]
    async fn skill_write_clamps_a_wild_priority_and_refuses_a_non_number() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        for (name, given, expected) in [("low", -5, 0u8), ("high", 99, 9)] {
            run(
                &write,
                serde_json::json!({"name": name, "description": "d", "priority": given}),
            )
            .await;
            let found = skills.current().await;
            let skill = found
                .skills
                .iter()
                .find(|s| s.name == name)
                .expect(name)
                .clone();
            assert_eq!(skill.priority, expected, "{name}");
        }

        let result = write
            .execute(
                serde_json::json!({"name": "words", "description": "d", "priority": "high"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(result.is_err(), "a non-number priority must not be guessed");
    }

    /// The tail matters as much as the matches: without it a truncated result reads as the whole
    /// answer, which is the same failure the capped index exists to avoid.
    #[tokio::test]
    async fn skill_search_reports_when_it_stopped_early() {
        let temp = tempfile::tempdir().expect("tempdir");
        let body: String = (0..MAX_SEARCH_MATCHES + 20)
            .map(|index| format!("needle line {index}\n"))
            .collect();
        write_skill(
            temp.path(),
            "haystack",
            &format!("---\ndescription: x\n---\n{body}"),
        );

        let search = SkillSearchTool {
            skills: cache_at(&temp),
        };
        let text = run(&search, serde_json::json!({"pattern": "needle"}))
            .await
            .text_content();
        assert_eq!(
            text.lines().filter(|l| l.contains("needle")).count(),
            MAX_SEARCH_MATCHES
        );
        assert!(text.contains("narrow the pattern"), "{text}");
    }

    #[tokio::test]
    async fn skill_search_rejects_an_invalid_regex() {
        let temp = tempfile::tempdir().expect("tempdir");
        let search = SkillSearchTool {
            skills: cache_at(&temp),
        };
        let result = search
            .execute(
                serde_json::json!({"pattern": "[unclosed"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(result.is_err());
    }

    /// A rootless cache means "nowhere to write to", which is a different failure from an empty
    /// store and has to say so rather than reporting success against a path that does not exist.
    #[tokio::test]
    async fn write_without_a_root_fails_with_a_reason() {
        let write = SkillWriteTool {
            skills: SkillCache::for_root(None),
        };
        let error = write
            .execute(
                serde_json::json!({"name": "x", "description": "d"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a rootless cache has nowhere to write");
        assert!(error.to_string().contains("disabled"), "{}", error);
    }

    /// An omitted priority keeps the one the skill already has, matching `PUT /v1/skills`: reading
    /// the absence as the default would demote a prioritized skill every time the agent refined
    /// its text, and priority decides which entries the index cap drops.
    #[tokio::test]
    async fn skill_write_keeps_an_omitted_priority() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skills = cache_at(&temp);
        let write = SkillWriteTool {
            skills: skills.clone(),
        };

        run(
            &write,
            serde_json::json!({"name": "ranked", "description": "first", "priority": 1}),
        )
        .await;
        let result = run(
            &write,
            serde_json::json!({"name": "ranked", "description": "refined"}),
        )
        .await;
        assert!(
            result.text_content().contains("(priority 1)"),
            "the confirmation must state what landed: {}",
            result.text_content()
        );

        let discovered = skills.current().await;
        let skill = discovered
            .skills
            .iter()
            .find(|skill| skill.name == "ranked")
            .expect("skill");
        assert_eq!(skill.priority, 1, "a refinement must not demote it");
        assert_eq!(skill.description, "refined");
    }

    /// `skill_edit` swaps one exact string, refuses an ambiguous one, replaces every occurrence
    /// when asked, deletes with an empty replacement, anchors on the stored body rather than the
    /// header `skill_read` shows, leaves the frontmatter alone, points at a near-miss name, and
    /// reports the file before and after.
    #[tokio::test]
    async fn skill_edit_changes_the_body_in_place() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("skills");
        std::fs::create_dir_all(&root).expect("root");
        let skills = SkillCache::new(Some(root.clone()), Vec::new());
        let write = SkillWriteTool {
            skills: skills.clone(),
        };
        let edit = SkillEditTool {
            skills: skills.clone(),
        };
        let file = root.join("note").join("SKILL.md");
        let created = run(
            &write,
            serde_json::json!({
                "name": "note",
                "description": "a note",
                "priority": 2,
                "body": "alpha beta alpha"
            }),
        )
        .await;
        assert!(!created.is_error, "{}", created.text_content());

        let ambiguous = run(
            &edit,
            serde_json::json!({"name": "note", "old_string": "alpha", "new_string": "gamma"}),
        )
        .await;
        assert!(ambiguous.is_error, "{}", ambiguous.text_content());
        assert!(
            ambiguous.text_content().contains("matches 2 times"),
            "{}",
            ambiguous.text_content()
        );

        let all = run(
            &edit,
            serde_json::json!({
                "name": "note",
                "old_string": "alpha",
                "new_string": "gamma",
                "replace_all": true
            }),
        )
        .await;
        assert!(!all.is_error, "{}", all.text_content());
        assert!(
            all.text_content()
                .starts_with("Edited skill 'note': 2 occurrence(s)."),
            "{}",
            all.text_content()
        );
        assert!(
            all.text_content().contains("1 | gamma beta gamma"),
            "the snippet numbers the body's own first line as 1: {}",
            all.text_content()
        );
        match all.frontend_metadata.clone() {
            Some(crate::frontend::ToolOutputMetadata::Diff {
                path: Some(path),
                old_text: Some(old),
                new_text: Some(new),
            }) => {
                assert_eq!(path, file);
                assert!(old.ends_with("---\n\nalpha beta alpha\n"), "{old}");
                assert!(new.ends_with("---\n\ngamma beta gamma\n"), "{new}");
                assert_eq!(new, std::fs::read_to_string(&file).expect("read back"));
            }
            other => panic!("an edit reports the file before and after: {other:?}"),
        }
        let current = skills.current().await;
        let skill = current.find("note").expect("still listed");
        assert_eq!(skill.priority, 2, "an edit leaves the frontmatter alone");
        assert_eq!(skill.description, "a note");

        let deleted = run(
            &edit,
            serde_json::json!({"name": "note", "old_string": "beta ", "new_string": ""}),
        )
        .await;
        assert!(!deleted.is_error, "{}", deleted.text_content());
        let current = skills.current().await;
        let body = skills::load_skill_source(current.find("note").expect("listed"))
            .await
            .expect("body");
        // The blank line the renderer puts after the closing fence is part of the stored body.
        assert_eq!(body, "\ngamma gamma\n");

        // The line `skill_read` puts above the body is not in the file, so an anchor that quotes
        // it misses rather than matching something the write would then duplicate.
        let read = SkillReadTool {
            skills: skills.clone(),
        };
        let shown = run(&read, serde_json::json!({"name": "note"}))
            .await
            .text_content();
        let header = shown.lines().next().expect("a header line").to_string();
        assert!(header.starts_with("Base directory"), "{shown}");
        let missing = run(
            &edit,
            serde_json::json!({"name": "note", "old_string": header, "new_string": "x"}),
        )
        .await;
        assert!(
            missing.is_error
                && missing
                    .text_content()
                    .contains("not found in the body of skill 'note'"),
            "{}",
            missing.text_content()
        );

        let unknown = edit
            .execute(
                serde_json::json!({"name": "nots", "old_string": "a", "new_string": "b"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        match unknown {
            Err(MekaError::ToolExecution { message, .. }) => {
                assert!(message.contains("no skill named 'nots'"), "{message}");
                assert!(message.contains("note"), "{message}");
            }
            other => panic!("an unknown name is an error naming the near miss: {other:?}"),
        }
        let not_a_bool = edit
            .execute(
                serde_json::json!({
                    "name": "note",
                    "old_string": "a",
                    "new_string": "b",
                    "replace_all": "yes"
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(not_a_bool.is_err(), "a string replace_all is refused");
        assert_eq!(edit.required_permission(), Permission::Read);
    }

    /// The edit door applies the write rules: a skill from a read-only root is refused without
    /// a shadow copy, a broken file is refused with its parse error rather than "not found", and
    /// a traversing name is refused before anything is joined onto the root.
    #[tokio::test]
    async fn skill_edit_refuses_what_skill_write_refuses() {
        let temp = tempfile::tempdir().expect("tempdir");
        let native = temp.path().join("native");
        let shared = temp.path().join("shared");
        write_skill(
            &shared,
            "borrowed",
            "---\ndescription: theirs\n---\nTHEIR PROCEDURE\n",
        );
        write_skill(&native, "broken", "no frontmatter at all\nKEEP ME\n");
        let edit = SkillEditTool {
            skills: SkillCache::new(Some(native.clone()), vec![shared.clone()]),
        };

        let foreign = run(
            &edit,
            serde_json::json!({"name": "borrowed", "old_string": "THEIR", "new_string": "MY"}),
        )
        .await;
        assert!(foreign.is_error, "{}", foreign.text_content());
        assert!(
            foreign.text_content().contains("does not write to"),
            "{}",
            foreign.text_content()
        );
        assert!(
            std::fs::read_to_string(shared.join("borrowed/SKILL.md"))
                .expect("still there")
                .contains("THEIR PROCEDURE")
        );
        assert!(
            !native.join("borrowed").exists(),
            "a shadowing copy must not be created in meka's own root"
        );

        let broken = run(
            &edit,
            serde_json::json!({"name": "broken", "old_string": "KEEP", "new_string": "DROP"}),
        )
        .await;
        assert!(broken.is_error, "{}", broken.text_content());
        assert!(
            broken.text_content().contains("not a valid skill"),
            "{}",
            broken.text_content()
        );
        assert!(!broken.text_content().contains("not found"));
        assert!(
            std::fs::read_to_string(native.join("broken/SKILL.md"))
                .expect("read")
                .contains("KEEP ME")
        );

        let traversing = edit
            .execute(
                serde_json::json!({"name": "../escape", "old_string": "a", "new_string": "b"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(
            traversing.is_err(),
            "a traversing name must not reach the disk"
        );
    }

    /// A hand-written `SKILL.md` with CRLF line endings: an LF anchor across a line break misses,
    /// and the refusal names the cause, as `file_edit`'s does.
    #[tokio::test]
    async fn skill_edit_names_crlf_when_an_lf_anchor_misses() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "windows",
            "---\r\ndescription: d\r\n---\r\nline one\r\nline two\r\n",
        );
        let edit = SkillEditTool {
            skills: cache_at(&temp),
        };
        let missed = run(
            &edit,
            serde_json::json!({
                "name": "windows",
                "old_string": "line one\nline two",
                "new_string": "x"
            }),
        )
        .await;
        assert!(missed.is_error, "{}", missed.text_content());
        assert!(
            missed.text_content().contains("CRLF line endings"),
            "{}",
            missed.text_content()
        );
        let matched = run(
            &edit,
            serde_json::json!({
                "name": "windows",
                "old_string": "line one\r\nline two",
                "new_string": "one line"
            }),
        )
        .await;
        assert!(!matched.is_error, "{}", matched.text_content());
    }
}
