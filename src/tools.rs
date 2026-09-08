//! Tool registry and built-in tool modules. Owns the [`ToolRegistry`] type that the agent loop
//! consults to resolve tool names to executable handlers, plus the per-tool submodules (file, find,
//! grep, scratchpad, shell, etc.).

pub(crate) mod background;
pub(crate) mod context;
mod conversation;
mod file;
mod find;
mod grep;
pub(crate) mod load_tool;
pub(crate) mod mcp_adapter;
pub(crate) mod mcp_resources;
mod memory;
mod render_image;
mod schedule;
pub(crate) mod scratchpad;
pub(crate) mod shell;
mod skill;
pub(crate) mod subagent;
pub(crate) mod todo;
pub(crate) mod util;
mod web;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::Arc,
};

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
pub(crate) use web::build_web_client;

#[cfg(test)]
use crate::conversation::ContentBlock;
#[cfg(test)]
use crate::conversation::Message;
use crate::{
    conversation::ToolResultContent, error::Result, permission::Permission,
    provider::ToolDefinition,
};

mod gate;
mod registry;

pub(crate) use self::{gate::*, registry::*};

/// Most unused parameters one advisory will name before it starts costing more context than the
/// discovery is worth.
const MAX_ADVISED_PARAMETERS: usize = 5;

/// The universal output-redirection parameter every tool accepts, handled by the agent loop rather
/// than by the tool itself. See [`crate::tools::scratchpad::save_explicit_scratchpad_results`].
pub(crate) const SCRATCHPAD_PARAMETER: &str = "scratchpad";

/// The universal detach parameter, consumed by the agent loop. Where `scratchpad` changes *where* a
/// tool's output goes, this changes *when* it arrives. See [`crate::background`].
pub(crate) const BACKGROUND_PARAMETER: &str = "background";

/// Whether `background` in this call means the *tool's* `background`, not meka's.
///
/// [`offer_background`] refuses to shadow a name the tool already advertises, so a tool whose
/// schema declares `background` at all never received meka's splice and owns the name outright.
/// The consumption side has to ask the same question: an image tool with `background:
/// "transparent"`, or an exec server with a detach flag of its own, would otherwise have its
/// argument eaten before dispatch and the call quietly detached meka's way instead.
fn tool_owns_background(schema: &serde_json::Value) -> bool {
    declared_type(schema, BACKGROUND_PARAMETER).is_some()
}

/// Whether `scratchpad` in this call means the tool's own parameter.
///
/// A weaker test than [`tool_owns_background`], because it has to be: meka's builtins declare
/// `scratchpad` themselves, as a string, so its presence in a schema proves nothing. Only a
/// different declared *type* says the tool means something else by the name.
fn tool_owns_scratchpad(schema: &serde_json::Value) -> bool {
    matches!(declared_type(schema, SCRATCHPAD_PARAMETER), Some(kind) if kind != "string")
}

/// The `type` a schema declares for one property, or `None` when it declares neither.
fn declared_type<'a>(schema: &'a serde_json::Value, name: &str) -> Option<&'a str> {
    let property = schema.get("properties")?.as_object()?.get(name)?;
    // A property declared with no `type` (or a union) is still declared. Report it as the empty
    // string so `tool_owns_background` sees a declaration while `tool_owns_scratchpad`, which is
    // looking for a conflicting type, also treats it as the tool's.
    Some(
        property
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(""),
    )
}

/// Reject a call whose meka-owned parameters are the wrong type, naming what was expected.
///
/// These two are validated where every other argument is left to the tool because meka is the one
/// that consumes them, and both change what a call *does* rather than what it is called with:
/// `background` decides whether the turn waits for the result, `scratchpad` decides whether the
/// output is kept. Reading a wrong type as "absent" makes each of those a silent no-op, so the
/// model asks for a detached call and blocks for twenty minutes, or asks for its output to be saved
/// and finds nothing saved, with no way to discover why. An error costs one round trip and says
/// exactly what to send instead.
///
/// Guessing is the third option and the worst one: reading `"true"` as `true` would leave `"yes"`,
/// `1` and `"1"` still silently false, so the contract becomes "some wrong types work", which is
/// harder to write against than either strict rule.
///
/// A tool's *own* arguments are deliberately not checked here. The tool, or a remote MCP server, is
/// the authority on what it accepts, and refusing a call the server would have honored would take
/// away capability to enforce a schema meka does not own. Those get an advisory instead; see
/// [`schema_disagreement`]. That includes these two names when the tool turns out to own them: see
/// [`tool_owns_background`] and [`tool_owns_scratchpad`].
///
/// `null` counts as absent throughout: models emit it for optional arguments they are not using,
/// and refusing that would be pedantry rather than a bug caught.
pub(crate) fn meka_parameter_error(
    input: &serde_json::Value,
    schema: &serde_json::Value,
) -> Option<String> {
    let object = input.as_object()?;
    let wrong_type = |key: &str, want: &str, value: &serde_json::Value| {
        format!(
            "Error: `{}` must be {}, but this call sent {}. Retry the call with a real {}, or \
             leave `{}` out.",
            key,
            want,
            json_type_name(value),
            want,
            key,
        )
    };

    if let Some(value) = object.get(BACKGROUND_PARAMETER)
        && !tool_owns_background(schema)
        && !value.is_null()
        && !value.is_boolean()
    {
        return Some(wrong_type(
            BACKGROUND_PARAMETER,
            "a boolean (`true` or `false`, unquoted)",
            value,
        ));
    }
    if let Some(value) = object.get(SCRATCHPAD_PARAMETER)
        && !tool_owns_scratchpad(schema)
        && !value.is_null()
        && !value.is_string()
    {
        return Some(wrong_type(
            SCRATCHPAD_PARAMETER,
            "a string naming the entry to save under",
            value,
        ));
    }
    None
}

/// What a JSON value is, for an error message aimed at whoever sent it.
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Split `background` out of a call's arguments, returning what the tool should receive and whether
/// it asked to be detached.
///
/// Removing rather than ignoring is the point: the property is spliced in by
/// [`ToolRegistry::definitions_active_with_loaded`] and is meka's, so forwarding it would hand an
/// MCP server a key it never advertised.
///
/// Unless the tool advertised it first, in which case the argument is passed straight through and
/// the call does not detach. [`offer_background`] declines to shadow a name a tool already uses,
/// and stripping what it declined to splice would be the same collision one step later, with the
/// tool losing an argument it does declare.
///
/// Only a real boolean detaches. A wrong type never reaches here, having been refused by
/// [`meka_parameter_error`]; `null` and absent both mean the call was not asking to detach.
pub(crate) fn take_background_flag(
    input: &serde_json::Value,
    schema: &serde_json::Value,
) -> (serde_json::Value, bool) {
    let Some(object) = input.as_object() else {
        return (input.clone(), false);
    };
    if !object.contains_key(BACKGROUND_PARAMETER) || tool_owns_background(schema) {
        return (input.clone(), false);
    }
    let mut object = object.clone();
    let detach = object
        .remove(BACKGROUND_PARAMETER)
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    (serde_json::Value::Object(object), detach)
}

/// What every door does with a call's arguments before a tool sees them, so no door can forget a
/// step. A call the provider rejected (see [`crate::provider::finalize_tool_arguments`]) is refused
/// with its reason. Meka's own parameters are type-checked, because a wrong type read as absent
/// would be a silent no-op the model has no way to notice. And `background` is taken out, since it
/// is spliced into the schema by the registry and consumed by the dispatcher, so no tool, least of
/// all a remote MCP server, sees a key it never advertised; a tool that declares `background`
/// itself keeps its argument.
///
/// Returns the arguments the tool receives and whether the call asked to detach; the error is the
/// output the model reads instead.
pub(crate) fn admit_arguments(
    name: &str,
    input: &serde_json::Value,
    schema: &serde_json::Value,
) -> std::result::Result<(serde_json::Value, bool), ToolOutput> {
    if let Some(reason) = input
        .get(crate::provider::INVALID_TOOL_ARGS_MARKER)
        .and_then(|value| value.as_str())
    {
        return Err(ToolOutput::text(
            format!("Tool call rejected: {reason}"),
            true,
        ));
    }
    if let Some(complaint) = meka_parameter_error(input, schema) {
        return Err(ToolOutput::text(complaint, true));
    }
    let (input, detach) = take_background_flag(input, schema);
    if detach && !detachable(name) {
        return Err(ToolOutput::text(
            format!(
                "Error: `{name}` cannot run in the background. It parks a request the turn drains \
                 as soon as this batch's results are in, so a detached one would fire against a \
                 later turn. Call it without `background`."
            ),
            true,
        ));
    }
    Ok((input, detach))
}

/// Whether a tool may be detached with `background: true`. Asked where the flag is offered and
/// where it is consumed, so the schema and the dispatch cannot disagree about it.
///
/// `context_compact` is the one exception: it does not do work, it parks a request the tool loop
/// drains once the batch's results are in. Detaching it would race that drain, leaving the request
/// to fire a round later than the agent asked for, or against the next turn entirely, and would
/// persist a `background_tasks` row for an operation that takes microseconds.
pub(crate) fn detachable(name: &str) -> bool {
    name != "context_compact"
}

/// Splice `background` into one tool's schema.
///
/// Done here, on the definitions handed to the provider, rather than declared per-tool the way
/// `scratchpad` is: one insertion point, config-gated in one place, and it reaches MCP tools too,
/// which matters because a slow MCP call is exactly the kind worth detaching. That is a deliberate
/// exception to passing an MCP server's `input_schema` through verbatim (see `crate::mcp`): the
/// property is meka's own, and [`take_background_flag`] strips it before the adapter forwards the
/// arguments.
fn offer_background(parameters: &mut serde_json::Value) {
    let Some(object) = parameters.as_object_mut() else {
        return;
    };
    // A schema with no `properties` describes a tool taking no arguments in the shape every
    // provider expects; creating the map here would change what the tool advertises.
    let Some(properties) = object
        .get_mut("properties")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    // Never shadow a real parameter: a server that already advertises `background` owns the name.
    if properties.contains_key(BACKGROUND_PARAMETER) {
        return;
    }
    properties.insert(
        BACKGROUND_PARAMETER.to_string(),
        serde_json::json!({
            "type": "boolean",
            "default": false,
            "description":
                "Run this call in the background. It returns a task id immediately and its result \
                 is delivered to you when it finishes, so use it for work that takes minutes (long \
                 builds, test suites, large downloads) and not for anything you need in order to \
                 continue this turn. Track it with `task_list` and stop it with `task_cancel`.",
        }),
    );
}

/// An advisory appended to a `tool_result` when the call and the tool's advertised schema disagree,
/// or `None` when they don't.
///
/// Two disagreements are worth reporting, and they are mirror images:
///
/// - **Parameters the schema documents that the call omitted.** Only raised for a deferred tool the
///   model never loaded, because that is the case where it was working from a truncated one-line
///   summary and could not have known. This is the `send_file(as_photo)` failure: the call
///   succeeds, the default is silently wrong, and nothing anywhere says a knob existed.
/// - **Arguments the schema doesn't declare.** Servers almost universally ignore unknown keys, so
///   without this the model concludes the parameter "didn't work" rather than "was never
///   delivered", which is what a server binary older than its own source looks like from the
///   outside.
pub(crate) fn schema_disagreement(
    tool_name: &str,
    input: &serde_json::Value,
    schema: &serde_json::Value,
    report_unused: bool,
) -> Option<String> {
    let properties = schema.get("properties")?.as_object()?;
    // A schema that declares no properties at all describes nothing to disagree with. Servers do
    // ship these for tools that genuinely take arguments, so treating every key as undeclared would
    // be noise rather than a finding.
    if properties.is_empty() {
        return None;
    }
    let passed: std::collections::BTreeSet<&str> = input
        .as_object()
        .map(|object| object.keys().map(String::as_str).collect())
        .unwrap_or_default();

    let mut lines: Vec<String> = Vec::new();

    let undeclared: Vec<&str> = passed
        .iter()
        .copied()
        .filter(|key| !properties.contains_key(*key))
        // `scratchpad` and `background` are meka's own, accepted on every tool and consumed by the
        // agent loop rather than by the tool. Neither appears in the raw definition this function
        // reads, so flagging them would fire on documented features.
        .filter(|key| *key != SCRATCHPAD_PARAMETER && *key != BACKGROUND_PARAMETER)
        .collect();
    // An explicit `additionalProperties: true` means the server documented that it takes more than
    // it lists, so an undeclared key there is intentional rather than a mistake.
    let open = schema
        .get("additionalProperties")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if !undeclared.is_empty() && !open {
        lines.push(format!(
            "Sent but not declared in this tool's schema, so it was most likely ignored: {}.",
            undeclared.join(", "),
        ));
    }

    if report_unused {
        let mut unused: Vec<String> = properties
            .iter()
            .filter(|(key, _)| !passed.contains(key.as_str()))
            .filter_map(|(key, spec)| {
                let description = spec
                    .get("description")
                    .and_then(serde_json::Value::as_str)?;
                let default = spec
                    .get("default")
                    .map(|value| format!(", default {value}"))
                    .unwrap_or_default();
                let kind = spec
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("any");
                Some(format!("{key} ({kind}{default}): {description}"))
            })
            .collect();
        if !unused.is_empty() {
            let dropped = unused.len().saturating_sub(MAX_ADVISED_PARAMETERS);
            unused.truncate(MAX_ADVISED_PARAMETERS);
            let tail = if dropped > 0 {
                format!("\n  (+{dropped} more)")
            } else {
                String::new()
            };
            lines.push(format!(
                "Called without loading its schema, so these documented parameters took their \
                 defaults:\n  {}{}",
                unused.join("\n  "),
                tail,
            ));
        }
    }

    if lines.is_empty() {
        return None;
    }
    // Only point at `load_tool` when the schema is genuinely still hidden; telling a model to load
    // something it already loaded reads as noise and trains it to skim these.
    if report_unused {
        lines.push(format!(
            "Call `load_tool` with \"{tool_name}\" for the full contract.",
        ));
    }
    Some(format!(
        "\n\n{} {}",
        crate::conversation::HARNESS_NOTE,
        lines.join("\n")
    ))
}

/// `" Did you mean `a` or `b`?"` for a tool name that isn't registered, or `""` when nothing is
/// close. The leading space lets it slot into a sentence unconditionally.
///
/// Plain edit distance is a poor fit for this namespace: omitting the `mcp__<server>__` prefix is
/// the likeliest mistake by far and puts the right answer seventeen edits away. So an exact match
/// on the final `__` segment is tried first and wins outright; distance is only the fallback for
/// genuine typos.
///
/// A bare noun that has since become a family prefix (`skill` for `skill_read` / `skill_write`)
/// is matched the same way: the threshold scales with the typed name, so a five-character needle
/// allows one edit while the answer is five away, and a resumed session reaching for the old name
/// is exactly the case a rename most needs to cover.
pub(crate) fn did_you_mean_hint<'a>(
    target: &str,
    candidates: impl Iterator<Item = &'a str>,
) -> String {
    const MAX_SUGGESTIONS: usize = 3;
    let needle = target.to_ascii_lowercase();
    let needle_tail = needle.rsplit("__").next().unwrap_or(needle.as_str());
    let family_prefix = format!("{needle}_");
    let threshold = (needle.chars().count() / 3).clamp(1, 5);

    let mut by_segment: Vec<&str> = Vec::new();
    let mut by_distance: Vec<(usize, &str)> = Vec::new();
    let needle_chars = needle.chars().count();
    for candidate in candidates {
        let lowered = candidate.to_ascii_lowercase();
        if lowered == needle
            || lowered.rsplit("__").next() == Some(needle.as_str())
            || lowered == needle_tail
            || lowered.starts_with(&family_prefix)
        {
            by_segment.push(candidate);
            continue;
        }
        // Two strings whose lengths differ by more than the threshold cannot be within it, so
        // this rejects them without building the matrix: an unbounded argument would otherwise
        // run a matrix per stored name, synchronously on a runtime worker. This is what lets the
        // lookup doors accept any name the column holds rather than capping length.
        //
        // Counted in characters, because `threshold` and `edit_distance` both are; `len()` would
        // measure bytes and silently discard non-ASCII candidates inside the threshold.
        if needle_chars.abs_diff(lowered.chars().count()) > threshold {
            continue;
        }
        let distance = edit_distance(&needle, &lowered);
        if distance <= threshold {
            by_distance.push((distance, candidate));
        }
    }

    let mut suggestions: Vec<&str> = if by_segment.is_empty() {
        by_distance.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
        by_distance.into_iter().map(|(_, name)| name).collect()
    } else {
        // Destructive members last, then alphabetical: a whole family can exceed
        // `MAX_SUGGESTIONS`, and leading a retry with the destructive verb is the one ordering
        // worth ruling out.
        by_segment.sort_unstable_by_key(|name| (is_destructive(name), *name));
        by_segment
    };
    suggestions.truncate(MAX_SUGGESTIONS);

    if suggestions.is_empty() {
        return String::new();
    }
    let rendered: Vec<String> = suggestions.iter().map(|name| format!("`{name}`")).collect();
    format!(" Did you mean {}?", rendered.join(" or "))
}

/// Whether a tool name reads as one that destroys something, by its verb.
///
/// Only used to order suggestions. A suggestion list is read by a model about to retry, so which
/// member of a family it sees first is not cosmetic.
fn is_destructive(name: &str) -> bool {
    name.ends_with("_delete") || name.ends_with("_cancel") || name.ends_with("_remove")
}

/// Counts entries into [`edit_distance`], so a test can assert the *matrix* was skipped rather than
/// that the candidate was visited.
///
/// The distinction is the whole point of the length band in [`did_you_mean_hint`] and it is not
/// observable any other way: skipping costs work, not output, and a counter placed in the caller's
/// iterator increments before the band is consulted.
#[cfg(test)]
pub(crate) static EDIT_DISTANCE_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Levenshtein distance in chars, two-row. Only ever runs on an error path, so the quadratic cost
/// over the registry is not worth optimizing away.
///
/// Shared with `memory_search`, whose last-resort tier is the same idea applied to memory names
/// and descriptions instead of tool names: when full-text matching finds nothing, the query was
/// probably misspelled rather than absent.
pub(crate) fn edit_distance(left: &str, right: &str) -> usize {
    #[cfg(test)]
    EDIT_DISTANCE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current: Vec<usize> = vec![0; right.len() + 1];
    for (i, left_char) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, right_char) in right.iter().enumerate() {
            let substitution = usize::from(left_char != right_char);
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

/// What a file looked like when a tool last read it, so a later `edit_file` can tell "you never
/// read this" from "this moved under you".
///
/// Compared against whatever the same source says at edit time, which is why this is an enum: a
/// file read through an editor's hosted filesystem and a file read off the disk are two different
/// documents that happen to share a path, and checking one against the other produces a false
/// alarm every time the user saves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadStamp {
    /// Read from the filesystem. Metadata rather than a content hash: both fields come off the
    /// `stat` the read already performs, so recording them is free, and any edit that changes a
    /// file changes at least one of them in practice.
    ///
    /// `mtime` is `None` on the platforms and filesystems that don't report one; two `None` stamps
    /// compare equal, so such a filesystem degrades to length-only detection rather than to a false
    /// "changed on disk" on every edit.
    Disk {
        mtime: Option<std::time::SystemTime>,
        len: u64,
    },
    /// Served by the frontend's hosted filesystem (see [`crate::frontend::Frontend`]), which is the
    /// editor's document and not the bytes on disk.
    ///
    /// Fingerprints what was served, because that is the only thing the disk cannot answer for. An
    /// editor serves its buffer for any file it owns, saved or not, so a disk comparison here is
    /// wrong in both directions: it fires when the user saves (the buffer did not change) and stays
    /// silent when the user types (the disk did not change).
    Delegated { fingerprint: u64 },
}

impl ReadStamp {
    pub(crate) fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self::Disk {
            mtime: metadata.modified().ok(),
            len: metadata.len(),
        }
    }

    /// The stamp for a path, or `None` when it can't be stated.
    pub(crate) async fn of_path(path: &std::path::Path) -> Option<Self> {
        tokio::fs::metadata(path)
            .await
            .ok()
            .map(|metadata| Self::from_metadata(&metadata))
    }

    /// The stamp for text the frontend served, or that meka wrote back through it.
    pub(crate) fn of_delegated(text: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        Self::Delegated {
            fingerprint: hasher.finish(),
        }
    }
}

/// Files a tool has read this session, and what they looked like at the time.
///
/// A set of paths answers only "has this been read", which is a weaker question. It could not
/// answer "is that read still valid", so an edit against a file rewritten in between (by a shell
/// command, a concurrent agent, or the user's editor) silently clobbered.
pub(crate) type ReadTracker = Arc<RwLock<HashMap<PathBuf, ReadStamp>>>;

#[derive(Debug, Default)]
pub(crate) struct ToolOutput {
    pub(crate) content: Vec<ToolResultContent>,
    pub(crate) is_error: bool,
    /// When `persist_oversized_results` has to spill to the scratchpad, use this name instead of
    /// the caller-supplied tool name. Set by MCP tool adapters so the persisted blob is namespaced
    /// as `mcp_<server>_<remote_tool>` for easier debugging.
    pub(crate) scratchpad_hint: Option<String>,
    /// Tool-specific structured side-channel for frontends that know how to render it (e.g. ACP's
    /// `diff` content block). Tools that don't produce extra structure leave this as `None`; the
    /// regular `content` text remains the source of truth for the model.
    pub(crate) frontend_metadata: Option<crate::frontend::ToolOutputMetadata>,
    /// The machine-readable half of the result, for callers that compute on it rather than read
    /// it.
    ///
    /// Set from MCP's `structuredContent`. That value is *also* rendered into `content` as a
    /// fenced JSON block so the model can reason over it, but that rendering is a presentation
    /// choice and the format string is free to change. Anything evaluating a result (a
    /// scheduled job's gate predicate is the one caller today) must read this field rather
    /// than parse the markdown back out, or a readability tweak to the block silently changes
    /// what the predicate decides.
    ///
    /// `None` is the common case: most tools produce prose, and plenty of MCP servers send only
    /// text. A caller that needs JSON from one of those falls back to parsing `content` itself,
    /// where there is no fence to confuse it.
    pub(crate) structured: Option<serde_json::Value>,
}

impl ToolOutput {
    pub(crate) fn text(content: String, is_error: bool) -> Self {
        Self {
            content: vec![ToolResultContent::Text { text: content }],
            is_error,
            scratchpad_hint: None,
            frontend_metadata: None,
            structured: None,
        }
    }

    /// What the model reads when a tool fails. One spelling for the dispatcher, which converts an
    /// `Err` from `execute`, and for a tool stating the same refusal through
    /// [`Tool::refusal_at_level`] before it runs.
    pub(crate) fn from_error(error: &crate::error::MekaError) -> Self {
        Self::text(format!("Tool error: {error}"), true)
    }

    /// Attach structured frontend metadata to an existing output, e.g. the pre/post text from a
    /// successful `edit_file`. Chains after any other builder so the call site reads as
    /// `ToolOutput::text(...).with_metadata(diff)`.
    #[must_use]
    pub(crate) fn with_metadata(mut self, metadata: crate::frontend::ToolOutputMetadata) -> Self {
        self.frontend_metadata = Some(metadata);
        self
    }

    /// Append harness-authored text to the model-visible result, extending the trailing text block
    /// rather than adding one. An MCP result can be a mix of text and images, and a bare text block
    /// tacked onto the end of that is easy to read as part of the tool's own output.
    pub(crate) fn append_notice(&mut self, notice: &str) {
        match self.content.last_mut() {
            Some(ToolResultContent::Text { text }) => text.push_str(notice),
            _ => self.content.push(ToolResultContent::Text {
                text: notice.trim_start().to_string(),
            }),
        }
    }
}

/// What a tool call knows about the session it runs in, handed to [`Tool::execute`] by whoever
/// dispatches it. One per call, and the only way a tool learns any of it: the inline dispatch,
/// the background spawn, the checkpoint turn and a gate probe each build one, so a tool cannot
/// tell which door it came through and none of the doors can forget a field.
#[derive(Clone)]
pub(crate) struct ToolContext {
    /// The session the call belongs to, or `None` for a call outside any session.
    pub(crate) session_id: Option<uuid::Uuid>,
    /// The provider's id for this tool use when a model made the call. A tool that streams output
    /// tags its chunks with it, and an MCP call carries it to the server in `_meta`.
    pub(crate) tool_call_id: Option<String>,
    /// The prompt the turn making this call answers, so a worker spawned by the call bills its
    /// work to the same prompt.
    pub(crate) prompt_id: Option<uuid::Uuid>,
    /// Where this call's prompts, progress and live output go.
    pub(crate) frontend: Arc<dyn crate::frontend::Frontend>,
    /// Canceled when this call, or the turn carrying it, is stopped.
    pub(crate) cancellation: CancellationToken,
}

impl ToolContext {
    /// A call outside any session: no ids, a frontend that swallows everything, the given token.
    /// Tests and gate probes use it.
    #[cfg(test)]
    pub(crate) fn detached(cancellation: CancellationToken) -> Self {
        Self {
            session_id: None,
            tool_call_id: None,
            prompt_id: None,
            frontend: Arc::new(crate::frontend::SilentFrontend),
            cancellation,
        }
    }
}

/// A callable tool surfaced to the model. Built-in tools live under `src/tools/`; MCP tools are
/// wrapped at registration time. Implementors must be safe to invoke concurrently; the dispatch
/// loop runs all tool calls in a single assistant message in parallel via `join_all`.
#[async_trait]
pub(crate) trait Tool: Send + Sync {
    /// Schema surfaced to the model (name + description + JSON-schema for parameters). Called once
    /// per registry build, not per call.
    fn definition(&self) -> ToolDefinition;
    /// Lowest permission level that may invoke this tool. The dispatch loop refuses the call, or
    /// submits it for approval, when the current level is below this.
    fn required_permission(&self) -> Permission;
    /// The refusal this call meets at `level` whatever the user answers, or `None` when the level
    /// alone settles nothing.
    ///
    /// An approved call runs *at the level*: the write fence and the shell's confinement read it,
    /// so approval turns a refusal into a question without widening reach. A tool whose `execute`
    /// would refuse the call for a reason the level and the arguments already decide states it
    /// here, and the door returns it in place of the approval prompt, so the user is never asked a
    /// question whose yes cannot matter. The wording is the one `execute` would have used.
    async fn refusal_at_level(
        &self,
        _level: Permission,
        _input: &serde_json::Value,
    ) -> Option<ToolOutput> {
        None
    }
    /// Whether this tool's work happens in a process meka does not confine.
    ///
    /// True only for MCP adapters today: the call is forwarded to a server meka spawned but does
    /// not sandbox, so no boundary meka can express reaches it. The dispatch gate needs this
    /// because `Permission::allows` deliberately treats `Workspace` and `Unrestricted` as equal
    /// (scope is enforced at the write door, and the tools array must stay byte-identical across
    /// level toggles for the prompt cache), which works for built-ins, which have a door; an MCP
    /// adapter has none, so `Workspace.allows(Unrestricted)` alone would let an unannotated tool
    /// write anywhere from inside the confined level.
    fn runs_outside_confinement(&self) -> bool {
        false
    }
    /// Run the tool. Long-running implementations must observe `context.cancellation` (e.g. via
    /// `tokio::select!`) so a user interrupt or turn-level abort unblocks promptly.
    async fn execute(&self, input: serde_json::Value, context: ToolContext) -> Result<ToolOutput>;
}

/// The text blocks of a tool result, concatenated. Images carry nothing a predicate can judge.
fn flatten_tool_text(output: &ToolOutput) -> String {
    output
        .content
        .iter()
        .filter_map(|block| match block {
            crate::conversation::ToolResultContent::Text { text } => Some(text.as_str()),
            crate::conversation::ToolResultContent::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Resolve the summary string shown next to a tool-call indicator and in the approval prompt. Tries
/// the hardcoded built-in map first; falls back to the tool's JSON schema `required[0]` when
/// provided (covers MCP tools, whose schemas are authored upstream and can't be enumerated here).
pub(crate) fn resolve_primary_param(
    name: &str,
    input: &serde_json::Value,
    schema: Option<&serde_json::Value>,
) -> Option<String> {
    if let Some(value) = builtin_primary_param(name, input) {
        return Some(value);
    }
    schema.and_then(|s| schema_primary_param(s, input))
}
/// Whether [`builtin_primary_param`] answers for `name` given an input shaped like `parameters`.
///
/// The probe is built from the tool's own declared properties, not from a fixed list of keys: a
/// rule keyed to a parameter the tool does not declare (a typo, or a parameter renamed long
/// afterwards) satisfies a hand-written probe while returning `None` for every real call, and the
/// tool silently goes back to rendering bare.
///
/// Drives `every_tool_with_arguments_can_show_a_primary_param`, below.
#[cfg(test)]
pub(crate) fn primary_param_answers_for_schema(name: &str, parameters: &serde_json::Value) -> bool {
    let mut probe = serde_json::Map::new();
    if let Some(properties) = parameters.get("properties").and_then(|p| p.as_object()) {
        for (key, property) in properties {
            // Typed, because a rule may read a value rather than only its presence: `task_cancel`
            // branches on `all` being `true`, and a string there would take the wrong arm.
            let value = match property.get("type").and_then(|t| t.as_str()) {
                Some("integer" | "number") => serde_json::json!(1),
                Some("boolean") => serde_json::json!(true),
                Some("array") => serde_json::json!(["x"]),
                Some("object") => serde_json::json!({"x": "y"}),
                _ => serde_json::json!("x"),
            };
            probe.insert(key.clone(), value);
        }
    }
    builtin_primary_param(name, &serde_json::Value::Object(probe)).is_some()
}
/// The built-ins that take no argument of their own, and so need no rule below.
///
/// The complement of [`builtin_primary_param`]'s coverage over [`BUILTIN_TOOL_NAMES`]. Stated
/// rather than derived because most of these are `list` tools that could grow a filter later, and a
/// new property on one of them must be a decision to revisit the entry rather than a silent
/// exemption; `every_tool_with_arguments_can_show_a_primary_param` checks every entry against the
/// tool's real schema and fails either way round.
#[cfg(test)]
pub(crate) const BUILTINS_WITHOUT_ARGUMENTS: &[&str] = &[
    "agent_list",
    "context_check",
    "mcp_resource_updates_list",
    "schedule_list",
    "scratchpad_list",
    "task_list",
];
/// The argument a tool-call indicator shows next to the tool's name.
///
/// One rule per name in [`BUILTIN_TOOL_NAMES`] that takes an argument, the complement of
/// `BUILTINS_WITHOUT_ARGUMENTS`. Covering every one of them is what the map is for:
/// [`resolve_primary_param`]'s other half needs the tool's JSON Schema, and replayed history has
/// none, so a built-in missing from here renders bare in `/history` having rendered fully live.
fn builtin_primary_param(name: &str, input: &serde_json::Value) -> Option<String> {
    // `render_image` accepts either `from_scratchpad` or inline `base64`. Show the scratchpad name
    // when present; for inline base64 the payload is opaque so there's nothing useful to display.
    if name == "render_image" {
        if let Some(from) = input.get("from_scratchpad").and_then(|v| v.as_str()) {
            return Some(from.to_string());
        }
        if input.get("base64").is_some() {
            return Some("<inline base64>".to_string());
        }
        return None;
    }

    // `todo` has no single primary key. Surface what the agent is doing, preferring the status
    // transitions, then the `title` of a list it is building, then the list size, and finally
    // "read" for an argument-less read.
    if name == "todo" {
        if let Some(set) = input.get("set").and_then(|v| v.as_object()) {
            let parts: Vec<String> = set
                .iter()
                .filter_map(|(id, status)| status.as_str().map(|status| format!("#{id} {status}")))
                .collect();
            if !parts.is_empty() {
                return Some(parts.join(", "));
            }
        }
        if let Some(title) = input.get("title").and_then(|v| v.as_str()) {
            let title = title.trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
        if let Some(items) = input.get("items").and_then(|v| v.as_array()) {
            let count = items.len();
            return Some(format!(
                "{} task{}",
                count,
                if count == 1 { "" } else { "s" }
            ));
        }
        return Some("read".to_string());
    }

    // `task_cancel` takes either an id or `all`, and declares neither as required, so there is no
    // `required[0]` for the schema fallback to reach for. Without this the indicator would render
    // the tool name with no argument, which is the one thing a cancellation must be specific about.
    if name == "task_cancel" {
        if input
            .get("all")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return Some("all".to_string());
        }
        return input.get("id").and_then(|v| v.as_str()).map(str::to_string);
    }

    // Sorted like `BUILTIN_TOOL_NAMES`, so the two can be read against each other. Mostly this
    // agrees with the schema's own `required[0]`, which is what the live path would have fallen
    // back to; where it does not, the schema's first required key names the server a call is
    // addressed to rather than the thing it acts on, and the object is what a reader wants
    // (`mcp_resource_read` shows the URI, not which server holds it).
    let key = match name {
        "agent_delete" | "agent_followup" => "id",
        "agent_spawn" => "prompt",
        "context_compact" => "instructions",
        "conversation_read" => "start",
        "conversation_search" => "query",
        "edit_file" | "read_file" | "write_file" => "path",
        "execute_command" => "command",
        "fetch_url" => "url",
        "find_files" => "glob",
        "load_tool" => "name",
        "mcp_prompt_get" => "name",
        "mcp_prompt_list" | "mcp_resource_list" => "server",
        "mcp_resource_read" | "mcp_resource_subscribe" | "mcp_resource_unsubscribe" => "uri",
        "memory_delete" | "memory_read" | "memory_write" => "name",
        "memory_search" => "queries",
        "schedule_cancel" => "id",
        "schedule_create" => "prompt",
        "scratchpad_delete" | "scratchpad_edit" | "scratchpad_read" | "scratchpad_write" => "name",
        "scratchpad_load_file" => "path",
        "scratchpad_merge" => "sources",
        "scratchpad_rename" => "old",
        "scratchpad_save_file" => "name",
        "search_contents" => "pattern",
        "search_web" => "query",
        "skill_delete" | "skill_read" | "skill_write" => "name",
        "skill_search" => "pattern",
        _ => return None,
    };
    // Coerced rather than read as a string: `load_tool` takes a name or a list of them,
    // `memory_search` takes a list of phrasings, and `conversation_read` takes a number, and
    // `as_str` alone would leave all three replaying bare.
    input.get(key).and_then(coerce_display_value)
}
/// Fallback for tools not covered by the built-in map (MCP tools, dynamically-registered tools,
/// etc.). Uses the first entry of `inputSchema.required` as the key into `input` and coerces the
/// value to a short display string. Returns `None` when the schema offers no `required` field, the
/// required key is missing from `input`, or the value type has no sensible string form (e.g. nested
/// objects / binary blobs).
fn schema_primary_param(schema: &serde_json::Value, input: &serde_json::Value) -> Option<String> {
    let required = schema.get("required")?.as_array()?;
    let key = required.iter().find_map(|v| v.as_str())?;
    let value = input.get(key)?;
    coerce_display_value(value)
}
fn coerce_display_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => {
            if s.is_empty() {
                None
            } else {
                Some(s.clone())
            }
        }
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Array(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|v| match v {
                    serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
                    serde_json::Value::Number(n) => Some(n.to_string()),
                    serde_json::Value::Bool(b) => Some(b.to_string()),
                    _ => None,
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(", "))
            }
        }
        _ => None,
    }
}

#[cfg(test)]
impl ToolOutput {
    /// The text blocks of the result joined, with each image standing in as `[Image]`.
    pub(crate) fn text_content(&self) -> String {
        ContentBlock::tool_result_text_content(&self.content)
    }
}

/// A no-op tool for the deferred-loading tests: registered after `build_default` and then
/// deferred, so `load_tool` is exercised against a tool that is genuinely deferred rather than by
/// re-deferring a production tool that ships active.
#[cfg(test)]
pub(crate) struct FixtureDeferredTool {
    pub(crate) name: String,
}

#[cfg(test)]
#[async_trait::async_trait]
impl Tool for FixtureDeferredTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name.clone(),
            format!("test fixture: deferred tool {}", self.name),
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _context: ToolContext,
    ) -> Result<ToolOutput> {
        Ok(ToolOutput::text("ok".to_string(), false))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BUILTINS_WITHOUT_ARGUMENTS, builtin_primary_param, primary_param_answers_for_schema,
        resolve_primary_param, schema_primary_param,
    };
    use crate::store::Store;

    /// The one tool that cannot detach is refused where the flag is consumed, not only left out
    /// of the schema: a model that generalizes `background` to it would get a detached compaction
    /// that fires against a later turn.
    #[test]
    fn context_compact_cannot_be_detached() {
        let schema = serde_json::json!({"type": "object", "properties": {"instructions": {"type": "string"}}});
        let refused = super::admit_arguments(
            "context_compact",
            &serde_json::json!({"background": true}),
            &schema,
        );
        assert!(
            matches!(refused, Err(ref output) if output.is_error && output.text_content().contains("background")),
            "{refused:?}"
        );
        let (_, detach) = super::admit_arguments(
            "read_file",
            &serde_json::json!({"background": true}),
            &schema,
        )
        .expect("any other tool may detach");
        assert!(detach);
    }

    /// A pathological argument costs one pass over itself, not one distance matrix per candidate,
    /// which is what lets the lookup doors accept any name the column holds rather than capping
    /// length.
    ///
    /// Counted, not timed: a wall-clock bound measures the machine as much as the code, and a
    /// flake on a loaded CI runner and a real regression look identical.
    #[test]
    fn a_pathological_name_does_not_run_a_distance_matrix_per_candidate() {
        use std::sync::atomic::Ordering;

        let candidates: Vec<String> = (0..500).map(|index| format!("memory-{index}")).collect();
        let needle = "x".repeat(200_000);

        // Counted at the callee, not in the caller's iterator: every candidate is still visited
        // (the band is inside the loop), so counting visits answers identically whether or not the
        // band is there.
        EDIT_DISTANCE_CALLS.store(0, Ordering::Relaxed);
        let hint = did_you_mean_hint(&needle, candidates.iter().map(String::as_str));
        let built = EDIT_DISTANCE_CALLS.load(Ordering::Relaxed);

        assert!(
            hint.is_empty(),
            "nothing is within an edit of a 200,000-character name: {hint}"
        );
        assert_eq!(
            built,
            0,
            "no 11-character candidate is within an edit-distance threshold of a \
             200,000-character name, so none should have reached the matrix; {built} of {} did, \
             at ~200,000 x 11 cells each",
            candidates.len()
        );

        // The control, so "no matrices" is not passing because the function stopped working: a name
        // one edit away is still suggested, and that path *does* build the matrix.
        EDIT_DISTANCE_CALLS.store(0, Ordering::Relaxed);
        let near = did_you_mean_hint("memory-1", candidates.iter().map(String::as_str));
        assert!(
            near.contains("memory-1"),
            "a near miss must still be suggested: {near}"
        );
        assert!(
            EDIT_DISTANCE_CALLS.load(Ordering::Relaxed) > 0,
            "a needle inside the band must reach the matrix, or the band is refusing everything"
        );
    }

    use super::*;
    use crate::config::BuiltinToolFilter;

    pub(super) fn shared_permission_for_test() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(
            Permission::Unrestricted,
            crate::permission::EnabledPermissions::ALL,
        )
    }

    pub(super) fn todo_list_for_test() -> crate::todo::SharedTodoList {
        crate::todo::SharedTodoList::default()
    }

    /// Every built-in that takes a meaningful argument must be able to show one in the tool-call
    /// indicator, and must still be able to when the tool's schema next changes.
    ///
    /// Three failure modes: no rule at all (replayed history has no schema, so the built-in
    /// renders bare in `/history`); no rule and no `required` (even the live line has nothing to
    /// reach for); and a rule that has gone stale, because the map names a parameter by string and
    /// renaming that parameter silently breaks it, which is why the probe is built from each
    /// tool's declared properties rather than from a fixed list.
    ///
    /// The sweep is over the whole registry rather than `BUILTIN_TOOL_NAMES` so that each name
    /// arrives with its real schema, and it asserts at the end that it saw every name, because a
    /// family missing from the registry would otherwise go unexamined.
    #[tokio::test]
    async fn every_tool_with_arguments_can_show_a_primary_param() {
        // meka's own universal parameters. A tool whose schema is only these takes no argument of
        // its own and has nothing to display.
        const UNIVERSAL: &[&str] = &[SCRATCHPAD_PARAMETER, BACKGROUND_PARAMETER];

        let registry = registry_with_every_builtin().await;
        // The MCP meta-tools are registered deferred, which is what keeps them out of an ordinary
        // turn's tool list; naming them as loaded is how the sweep sees them.
        let loaded: Vec<String> = MCP_META_TOOL_NAMES.iter().map(|n| n.to_string()).collect();

        let mut swept: HashSet<String> = HashSet::new();
        for definition in registry.definitions_active_with_loaded(&loaded) {
            swept.insert(definition.name.clone());
            let own_properties = definition
                .parameters
                .get("properties")
                .and_then(|properties| properties.as_object())
                .map(|properties| {
                    properties
                        .keys()
                        .filter(|key| !UNIVERSAL.contains(&key.as_str()))
                        .count()
                })
                .unwrap_or(0);
            assert_eq!(
                own_properties == 0,
                BUILTINS_WITHOUT_ARGUMENTS.contains(&definition.name.as_str()),
                "{} takes {} argument(s), so `BUILTINS_WITHOUT_ARGUMENTS` {} list it",
                definition.name,
                own_properties,
                if own_properties == 0 {
                    "should"
                } else {
                    "must not"
                },
            );
            if own_properties == 0 {
                continue;
            }
            assert!(
                primary_param_answers_for_schema(&definition.name, &definition.parameters),
                "{} takes arguments, but `builtin_primary_param` returns nothing for an input \
                 built from its own schema: either it has no rule, or its rule names a parameter \
                 the tool no longer declares. Its tool-call indicator renders with no argument, \
                 live and in `/history`.",
                definition.name,
            );
        }

        for name in BUILTIN_TOOL_NAMES {
            assert!(
                swept.contains(*name),
                "{name} was never examined: `registry_with_every_builtin` does not register it, so \
                 nothing above checked whether its indicator can show an argument",
            );
        }
    }

    /// A registry holding every name in [`BUILTIN_TOOL_NAMES`], for tests that must not silently
    /// skip a family.
    ///
    /// `build_default` alone is not enough: `skill_write` / `skill_delete` need `agent_managed`,
    /// the `task_*` and `context_*` families are registered by their own hosts, and the MCP
    /// meta-tools appear only once a server is configured. None of them need to *work* here, only
    /// to answer `definition()`, so the collaborators are the cheapest that construct: an in-memory
    /// store, a zeroed gauge, and a server whose command does not exist (`prepare` validates config
    /// and spawns nothing).
    async fn registry_with_every_builtin() -> ToolRegistry {
        let registry = tool_registry_for_test_with(BuiltinToolFilter::default(), true).await;
        let store = Store::for_test().await;

        registry.enable_background();
        for tool in background::build(
            store.clone(),
            crate::session::ToolSite::for_test(),
            crate::background::BackgroundTasks::default(),
        ) {
            registry.register(tool).expect("register task tool");
        }

        for definition in [
            subagent::agent_spawn_definition(&[]),
            subagent::agent_list_definition(),
            subagent::agent_followup_definition(),
            subagent::agent_delete_definition(),
        ] {
            registry
                .register(Arc::new(SchemaOnlyTool { definition }))
                .expect("register subagent schema");
        }

        let server = crate::config::McpServerConfig {
            name: "fixture-srv".to_string(),
            transport: crate::config::McpTransport::Stdio,
            command: Some("/bin/false".to_string()),
            args: None,
            env: None,
            url: None,
            headers: None,
            headers_helper: None,
            auth: None,
            permission: None,
            allowed_tools: None,
            disabled_tools: None,
            eager_load_tools: None,
            tool_permissions: None,
            trust_read_only_hint: None,
            disabled: None,
            required: None,
        };
        let manager = crate::mcp::McpClientManager::prepare(
            &[server],
            None,
            None,
            crate::mcp::McpClientContext::new(),
        )
        .await
        .expect("prepare with one configured server");
        mcp_resources::register_all(&registry, manager);

        registry
    }

    /// Stands in for a tool whose only interesting part here is its schema.
    ///
    /// The `agent_*` family's real tools carry a `ToolBuilderParams` that reaches most of the
    /// process; their schemas are free-standing functions precisely so a caller can have the
    /// declaration without the machinery, and this is such a caller.
    struct SchemaOnlyTool {
        definition: ToolDefinition,
    }

    #[async_trait]
    impl Tool for SchemaOnlyTool {
        fn definition(&self) -> ToolDefinition {
            self.definition.clone()
        }

        fn required_permission(&self) -> Permission {
            Permission::Read
        }

        /// An error rather than a panic, so a later test that reaches
        /// [`registry_with_every_builtin`] for some other reason and dispatches one of these fails
        /// on its own assertion instead of on this one's stack.
        async fn execute(
            &self,
            _input: serde_json::Value,
            _context: crate::tools::ToolContext,
        ) -> Result<ToolOutput> {
            Err(crate::error::MekaError::ToolExecution {
                tool_name: self.definition.name.clone(),
                message: "registered for its schema only and cannot run".to_string(),
            })
        }
    }

    pub(super) async fn tool_registry_for_test() -> ToolRegistry {
        tool_registry_for_test_with_filter(BuiltinToolFilter::default()).await
    }

    pub(super) async fn tool_registry_for_test_with_filter(
        filter: BuiltinToolFilter,
    ) -> ToolRegistry {
        tool_registry_for_test_with(filter, false).await
    }

    pub(super) async fn tool_registry_for_test_with(
        filter: BuiltinToolFilter,
        skills_managed: bool,
    ) -> ToolRegistry {
        let store = Store::for_test().await;
        let shared_session_id = crate::session::SharedSessionId::default();
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                providers: std::sync::Arc::new(crate::provider::ProviderRegistry::for_test(
                    store.token_store(),
                    &["test-profile"],
                )),
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability,
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe,
                    builtin_filter: filter,
                    write_locks: crate::workspace::WriteLocks::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                skills_agent_managed: skills_managed,
                memories: crate::store::memory::MemoryStore::detached(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: shared_session_id,
                todo_list: todo_list_for_test(),
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    shared_permission_for_test(),
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("default web client config should build cleanly")
    }

    /// The `send_file` schema, as mekabridge advertises it.
    fn send_file_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "conversation": {"type": "string", "description": "Target conversation"},
                "path": {"type": "string", "description": "Path to the file"},
                "caption": {"type": ["string", "null"], "description": "Optional caption"},
                "as_photo": {
                    "type": "boolean",
                    "default": false,
                    "description": "Send as a viewable photo rather than a downloadable document.",
                },
            },
            "required": ["conversation", "path"]
        })
    }

    /// A call that succeeds, takes a silently wrong default, and has to be told about the flag
    /// that would have fixed it.
    #[test]
    fn schema_disagreement_names_the_unused_parameter() {
        let input = serde_json::json!({"conversation": "telegram:1", "path": "/tmp/a.png"});
        let advisory = schema_disagreement(
            "mcp__mekabridge__send_file",
            &input,
            &send_file_schema(),
            true,
        )
        .expect("blind call omitting as_photo must be advised");

        assert!(
            advisory.contains("as_photo (boolean, default false)"),
            "{advisory}"
        );
        assert!(advisory.contains("viewable photo"), "{advisory}");
        assert!(advisory.contains("caption"), "{advisory}");
        assert!(advisory.contains("load_tool"), "{advisory}");
        // Parameters the call did supply are not worth repeating back.
        assert!(!advisory.contains("\n  path ("), "{advisory}");
    }

    /// The same call, once the model has actually seen the schema, is not worth a word.
    #[test]
    fn schema_disagreement_is_silent_for_a_loaded_tool() {
        let input = serde_json::json!({"conversation": "telegram:1", "path": "/tmp/a.png"});
        assert!(
            schema_disagreement(
                "mcp__mekabridge__send_file",
                &input,
                &send_file_schema(),
                false
            )
            .is_none()
        );
    }

    /// What a server binary older than its own source looks like from the caller's side.
    #[test]
    fn schema_disagreement_flags_an_undeclared_argument() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path"}},
        });
        let input = serde_json::json!({"path": "/tmp/a.png", "as_photo": true});
        let advisory = schema_disagreement("send_file", &input, &schema, false).expect("advised");

        assert!(
            advisory.contains("not declared in this tool's schema"),
            "{advisory}"
        );
        assert!(advisory.contains("as_photo"), "{advisory}");
        // Nothing to load: this tool's schema is already in hand.
        assert!(!advisory.contains("load_tool"), "{advisory}");
    }

    /// `scratchpad` works on every tool and is consumed by the agent loop, so an MCP schema that
    /// doesn't mention it is not a disagreement.
    #[test]
    fn schema_disagreement_ignores_the_scratchpad_parameter() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path"}},
        });
        let input = serde_json::json!({"path": "/tmp/a.png", "scratchpad": "out"});
        assert!(schema_disagreement("mcp__x__read", &input, &schema, false).is_none());
    }

    #[test]
    fn schema_disagreement_respects_additional_properties() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path"}},
            "additionalProperties": true,
        });
        let input = serde_json::json!({"path": "/tmp/a.png", "extra": 1});
        assert!(schema_disagreement("open", &input, &schema, false).is_none());
    }

    #[test]
    fn schema_disagreement_ignores_schemas_with_nothing_to_say() {
        let input = serde_json::json!({"anything": 1});
        assert!(schema_disagreement("x", &input, &serde_json::json!({}), true).is_none());
        assert!(
            schema_disagreement(
                "x",
                &input,
                &serde_json::json!({"type": "object", "properties": {}}),
                true
            )
            .is_none()
        );
    }

    #[test]
    fn schema_disagreement_caps_the_list() {
        let mut properties = serde_json::Map::new();
        for index in 0..MAX_ADVISED_PARAMETERS + 3 {
            properties.insert(
                format!("option_{index}"),
                serde_json::json!({"type": "string", "description": "An option"}),
            );
        }
        let schema = serde_json::json!({"type": "object", "properties": properties});
        let advisory =
            schema_disagreement("x", &serde_json::json!({}), &schema, true).expect("some");
        assert!(advisory.contains("(+3 more)"), "{advisory}");
    }

    /// Undocumented parameters are skipped: naming a knob without saying what it does is not enough
    /// to act on, and the schema is one `load_tool` away.
    #[test]
    fn schema_disagreement_skips_undocumented_parameters() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path"},
                "mystery": {"type": "string"},
            },
        });
        assert!(
            schema_disagreement("x", &serde_json::json!({"path": "/a"}), &schema, true).is_none()
        );
    }

    /// A server that already advertises `background` owns the name. Silently redefining it would be
    /// a far worse bug than losing the feature on that one tool.
    #[test]
    fn offer_background_never_shadows_a_real_parameter() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "background": {"type": "string", "description": "Wallpaper to set"}
            }
        });
        offer_background(&mut schema);
        assert_eq!(
            schema["properties"]["background"]["type"],
            serde_json::json!("string")
        );
    }

    #[test]
    fn offer_background_leaves_a_no_argument_schema_alone() {
        let mut schema = serde_json::json!({"type": "object"});
        offer_background(&mut schema);
        assert!(schema.get("properties").is_none());
    }

    /// A schema for a tool that declares neither of meka's parameters, which is the ordinary case
    /// and the one where meka owns both names.
    fn plain_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"command": {"type": "string"}}
        })
    }

    /// The flag is meka's, so it must never reach the tool, least of all a remote MCP server that
    /// never advertised the key.
    #[test]
    fn take_background_flag_removes_it_from_the_arguments() {
        let schema = plain_schema();
        let (input, detach) = take_background_flag(
            &serde_json::json!({"command": "make", "background": true}),
            &schema,
        );
        assert!(detach);
        assert_eq!(input, serde_json::json!({"command": "make"}));

        let (input, detach) = take_background_flag(
            &serde_json::json!({"command": "make", "background": false}),
            &schema,
        );
        assert!(!detach);
        assert_eq!(input, serde_json::json!({"command": "make"}));

        let (input, detach) =
            take_background_flag(&serde_json::json!({"command": "make"}), &schema);
        assert!(!detach);
        assert_eq!(input, serde_json::json!({"command": "make"}));
    }

    #[test]
    fn take_background_flag_treats_a_non_boolean_as_absent() {
        for value in [
            serde_json::json!("yes"),
            serde_json::json!(1),
            serde_json::json!(null),
        ] {
            let (input, detach) =
                take_background_flag(&serde_json::json!({"background": value}), &plain_schema());
            assert!(!detach, "a wrong-typed flag must not detach by accident");
            assert_eq!(input, serde_json::json!({}));
        }
    }

    /// A tool that advertises `background` itself owns the name: `offer_background` declines to
    /// shadow it, so the model is looking at the *tool's* parameter and the argument is the tool's
    /// to receive. An image tool taking `background: "transparent"`, or an exec server with a
    /// detach flag of its own, would otherwise have the argument eaten before dispatch.
    #[test]
    fn a_tool_that_declares_background_keeps_its_own_argument() {
        let owned = serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "background": {"type": "string", "description": "Backdrop color"}
            }
        });

        // Not spliced, so the model never saw meka's version.
        let mut offered = owned.clone();
        offer_background(&mut offered);
        assert_eq!(offered, owned, "meka must not shadow the tool's parameter");

        // Not stripped, not detached, and not refused for being the wrong type for a flag it is
        // not.
        let call = serde_json::json!({"prompt": "a cat", "background": "transparent"});
        let (input, detach) = take_background_flag(&call, &owned);
        assert!(!detach, "the tool's parameter is not a detach request");
        assert_eq!(
            input, call,
            "the tool must receive the argument it declared"
        );
        assert_eq!(meka_parameter_error(&call, &owned), None);
    }

    /// The same question for `scratchpad`, which needs a weaker test: meka's own builtins declare
    /// it, as a string, so presence proves nothing and only a conflicting *type* means the tool
    /// has its own idea of the name.
    #[test]
    fn scratchpad_ownership_turns_on_the_declared_type() {
        let builtin_style = serde_json::json!({
            "type": "object",
            "properties": {"scratchpad": {"type": "string"}}
        });
        let complaint = meka_parameter_error(&serde_json::json!({"scratchpad": 7}), &builtin_style);
        assert!(
            complaint.is_some(),
            "a tool declaring it as a string agrees with meka, so the check still applies",
        );

        let someone_elses = serde_json::json!({
            "type": "object",
            "properties": {"scratchpad": {"type": "integer"}}
        });
        assert_eq!(
            meka_parameter_error(&serde_json::json!({"scratchpad": 7}), &someone_elses),
            None,
            "a tool that means a number by the name is entitled to be sent one",
        );
    }

    /// Some models emit every argument as a string whatever the schema says; GLM does it through
    /// OpenRouter, which is where this came up. Both of meka's own parameters decide what a call
    /// does rather than what it is called with, so the answer is to say so and let the model retry,
    /// not to read the wrong type as absent (a silent no-op) or to guess at it (a contract where
    /// `"true"` works and `"yes"` does not).
    #[test]
    fn a_wrong_typed_meka_parameter_is_refused_by_name() {
        for value in [
            serde_json::json!("true"),
            serde_json::json!("yes"),
            serde_json::json!(1),
        ] {
            let complaint =
                meka_parameter_error(&serde_json::json!({"background": value}), &plain_schema())
                    .expect("a wrong-typed background flag must be refused, not silently ignored");
            assert!(complaint.contains("background"), "{complaint}");
            assert!(complaint.contains("boolean"), "{complaint}");
        }

        let complaint =
            meka_parameter_error(&serde_json::json!({"scratchpad": 7}), &plain_schema())
                .expect("a wrong-typed scratchpad name must be refused");
        assert!(complaint.contains("scratchpad"), "{complaint}");
        assert!(complaint.contains("string"), "{complaint}");
    }

    /// Models emit `null` for optional arguments they are not using, and a tool's own arguments
    /// belong to the tool: refusing either would cost capability without catching a bug.
    #[test]
    fn well_formed_and_absent_parameters_pass() {
        for input in [
            serde_json::json!({"command": "make", "background": true}),
            serde_json::json!({"command": "make", "background": false}),
            serde_json::json!({"command": "make", "background": null}),
            serde_json::json!({"command": "make", "scratchpad": "build-log"}),
            serde_json::json!({"command": "make", "scratchpad": null}),
            serde_json::json!({"command": "make"}),
            // The tool's own arguments, wrong-typed on purpose: not meka's to police.
            serde_json::json!({"command": 12, "timeout": "30"}),
            serde_json::json!("not an object at all"),
        ] {
            assert_eq!(
                meka_parameter_error(&input, &plain_schema()),
                None,
                "{input}"
            );
        }
    }

    /// `background` is spliced into the definitions sent to the provider, not into the raw one this
    /// check reads, so without an exemption it would be reported as an undeclared argument.
    #[test]
    fn schema_disagreement_ignores_the_background_parameter() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command"}},
        });
        let input = serde_json::json!({"command": "make", "background": true});
        assert!(schema_disagreement("execute_command", &input, &schema, false).is_none());
    }

    /// The commonest slip: naming the tool without its `mcp__<server>__` prefix. Edit distance
    /// alone puts the answer seventeen operations away, so the segment match has to carry it.
    #[test]
    fn did_you_mean_matches_on_the_final_segment() {
        let registered = ["mcp__mekabridge__send_file", "read_file"];
        let hint = did_you_mean_hint("send_file", registered.into_iter());
        assert_eq!(hint, " Did you mean `mcp__mekabridge__send_file`?");
    }

    #[test]
    fn did_you_mean_catches_a_typo() {
        let registered = ["read_file", "write_file"];
        let hint = did_you_mean_hint("raed_file", registered.into_iter());
        assert_eq!(hint, " Did you mean `read_file`?");
    }

    /// The `skill` -> `skill_read` rename in miniature, and the reason the prefix rule exists.
    ///
    /// Distance alone cannot find this: the threshold scales with the *typed* name, so a
    /// five-character needle allows one edit while every answer is five away. Without the rule a
    /// resumed session reaching for the old name gets a bare unknown-tool error and no direction.
    #[test]
    fn did_you_mean_points_a_bare_noun_at_its_family() {
        let registered = ["skill_read", "skill_write", "read_file"];
        let hint = did_you_mean_hint("skill", registered.into_iter());
        assert!(hint.contains("`skill_read`"), "{hint}");
        assert!(hint.contains("`skill_write`"), "{hint}");
        assert!(!hint.contains("read_file"), "{hint}");
    }

    /// A whole family can exceed `MAX_SUGGESTIONS`, so which members survive truncation matters.
    /// Alphabetically `skill_delete` leads and `skill_write` is cut, which points a retrying model
    /// at the destructive verb first. The full four-tool set is the case the smaller fixture above
    /// cannot show.
    #[test]
    fn did_you_mean_never_leads_with_the_destructive_family_member() {
        let registered = [
            "skill_delete",
            "skill_read",
            "skill_search",
            "skill_write",
            "read_file",
        ];
        let hint = did_you_mean_hint("skill", registered.into_iter());
        let delete = hint.find("skill_delete");
        let read = hint.find("skill_read").expect("read must be suggested");
        assert!(
            delete.is_none_or(|delete| delete > read),
            "the delete tool must never come first: {hint}"
        );
        assert!(hint.contains("`skill_write`"), "{hint}");
    }

    /// The prefix has to be a *name segment*, not any shared start, or `search_web` would answer
    /// for `search` alongside genuinely-related tools and `scratchpad_read` would answer for
    /// `scratch`.
    #[test]
    fn did_you_mean_prefix_rule_requires_an_underscore_boundary() {
        let registered = ["skillet_read"];
        assert_eq!(did_you_mean_hint("skill", registered.into_iter()), "");
    }

    #[test]
    fn did_you_mean_is_silent_when_nothing_is_close() {
        let registered = ["read_file", "run_shell"];
        assert_eq!(
            did_you_mean_hint("frobnicate_widget", registered.into_iter()),
            ""
        );
    }

    #[test]
    fn did_you_mean_lists_every_server_offering_the_segment() {
        let registered = ["mcp__a__send_file", "mcp__b__send_file"];
        let hint = did_you_mean_hint("send_file", registered.into_iter());
        assert_eq!(
            hint,
            " Did you mean `mcp__a__send_file` or `mcp__b__send_file`?"
        );
    }

    #[tokio::test]
    async fn the_test_registry_holds_every_core_tool() {
        let registry = tool_registry_for_test().await;
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(registry.get("edit_file").is_some());
        assert!(registry.get("find_files").is_some());
        assert!(registry.get("search_contents").is_some());
        assert!(registry.get("execute_command").is_some());
        assert!(registry.get("fetch_url").is_some());
        assert!(registry.get("search_web").is_some());
        assert!(registry.get("todo").is_some());
        assert!(registry.get("scratchpad_write").is_some());
        assert!(registry.get("scratchpad_read").is_some());
        assert!(registry.get("scratchpad_edit").is_some());
        assert!(registry.get("scratchpad_list").is_some());
        assert!(registry.get("scratchpad_delete").is_some());
        assert!(registry.get("skill_read").is_some());
        assert!(registry.get("memory_write").is_some());
        assert!(registry.get("memory_read").is_some());
        assert!(registry.get("memory_search").is_some());
        assert!(registry.get("memory_delete").is_some());
        assert!(registry.get("render_image").is_some());
        assert!(registry.get("load_tool").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    /// A store with no root is an *empty* store, not a disabled one: its tools still register, so
    /// the agent can write the first memory into a directory that doesn't exist yet. Conflating
    /// the two is what made `meka tools list` hide tools a real session would have had.
    #[tokio::test]
    async fn empty_store_still_registers_its_tools() {
        let registry = tool_registry_for_test().await;
        assert!(registry.get("skill_read").is_some());
        assert!(registry.get("memory_write").is_some());
    }

    /// Reading skills is always on; authoring them is opt-in per installation. Two independent
    /// gates, so a default session never sees a tool that rewrites the user's skill store.
    #[tokio::test]
    async fn skill_authoring_is_off_by_default_and_on_with_agent_managed() {
        let registry = tool_registry_for_test().await;
        assert!(registry.get("skill_read").is_some());
        assert!(registry.get("skill_search").is_some());
        assert!(
            registry.get("skill_write").is_none(),
            "skill_write must not register without [skills] agent_managed"
        );
        assert!(registry.get("skill_delete").is_none());

        let registry = tool_registry_for_test_with(BuiltinToolFilter::default(), true).await;
        assert!(registry.get("skill_write").is_some());
        assert!(registry.get("skill_delete").is_some());
    }

    /// The rename's actual failure case, against the real registry rather than a hand-made list:
    /// a resumed session whose history contains a `skill` call must be pointed somewhere useful,
    /// and that only holds if the family really is registered under those names.
    #[tokio::test]
    async fn a_stale_skill_call_is_pointed_at_the_renamed_tool() {
        let registry = tool_registry_for_test().await;
        let registered = registry.registered_tool_names();
        let hint = did_you_mean_hint("skill", registered.iter().map(String::as_str));
        assert!(
            hint.contains("`skill_read`"),
            "the old name must lead somewhere, got: {hint:?}"
        );
    }

    /// With approvals on, a worker's catalog lists the tools above its level: dispatch puts them
    /// to the user rather than refusing them, so the prompt must not say they do not exist.
    #[tokio::test]
    async fn approvals_list_the_tools_above_the_level() {
        let registry = tool_registry_for_test().await;
        let listed = registry.definitions_for_permission(Permission::Read, true);
        assert!(listed.iter().any(|t| t.name == "write_file"));
        let hidden = registry.definitions_for_permission(Permission::Read, false);
        assert!(!hidden.iter().any(|t| t.name == "write_file"));
    }

    #[tokio::test]
    async fn register_duplicate_returns_error() {
        struct DummyTool;
        #[async_trait::async_trait]
        impl Tool for DummyTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition::new(
                    "dup_tool".to_string(),
                    "dummy".to_string(),
                    serde_json::json!({}),
                )
            }

            fn required_permission(&self) -> Permission {
                Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> crate::error::Result<ToolOutput> {
                Ok(ToolOutput::text(String::new(), false))
            }
        }

        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(DummyTool) as Arc<dyn Tool>)
            .expect("first registration succeeds");
        let error = registry
            .register(Arc::new(DummyTool) as Arc<dyn Tool>)
            .expect_err("second registration with same name must fail");
        let message = format!("{error}");
        assert!(
            message.contains("dup_tool"),
            "error message should mention the duplicate name, got: {message}"
        );
    }

    #[tokio::test]
    async fn registry_filter_drops_disabled_tools() {
        let filter = BuiltinToolFilter::from_config(
            None,
            vec!["search_web".to_string(), "fetch_url".to_string()],
            HashMap::new(),
        );
        let registry = tool_registry_for_test_with_filter(filter).await;
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(
            registry.get("search_web").is_none(),
            "search_web should be filtered out"
        );
        assert!(
            registry.get("fetch_url").is_none(),
            "fetch_url should be filtered out"
        );
    }

    #[tokio::test]
    async fn registry_filter_allow_list_keeps_only_listed() {
        let filter = BuiltinToolFilter::from_config(
            Some(vec!["read_file".to_string(), "find_files".to_string()]),
            Vec::new(),
            HashMap::new(),
        );
        let registry = tool_registry_for_test_with_filter(filter).await;
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("find_files").is_some());
        assert!(registry.get("write_file").is_none());
        assert!(registry.get("execute_command").is_none());
        assert!(registry.get("search_web").is_none());
    }

    /// Stub tool that sleeps for a known duration before returning a payload derived from its
    /// input. Observes the cancellation token via `select!` so cancellation tests can assert early
    /// exit. Used to verify the parent and sub-agent dispatch loops actually run their `join_all`
    /// futures in parallel and propagate cancellation correctly.
    struct SleepTool {
        name: String,
        delay: std::time::Duration,
    }

    #[async_trait]
    impl Tool for SleepTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: "test sleep tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "label": { "type": "string" }
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
            context: crate::tools::ToolContext,
        ) -> Result<ToolOutput> {
            let cancellation = context.cancellation.clone();
            let label = input
                .get("label")
                .and_then(|v| v.as_str())
                .unwrap_or("default")
                .to_string();
            tokio::select! {
                _ = tokio::time::sleep(self.delay) => {
                    Ok(ToolOutput::text(format!("done:{label}"), false))
                }
                _ = cancellation.cancelled() => {
                    Ok(ToolOutput::text(format!("canceled:{label}"), true))
                }
            }
        }
    }

    /// Two tools each sleep ~200 ms; the total wall-clock must be much less than the sum, or
    /// dispatch has become sequential.
    #[tokio::test]
    async fn parallel_dispatch_runs_tools_concurrently() {
        let registry = ToolRegistry::new_with_filter(BuiltinToolFilter::default());
        registry
            .register(Arc::new(SleepTool {
                name: "sleep_one".to_string(),
                delay: std::time::Duration::from_millis(200),
            }))
            .expect("registration should succeed");
        registry
            .register(Arc::new(SleepTool {
                name: "sleep_two".to_string(),
                delay: std::time::Duration::from_millis(200),
            }))
            .expect("registration should succeed");

        let tools = [
            ("a", "sleep_one", serde_json::json!({ "label": "first" })),
            ("b", "sleep_two", serde_json::json!({ "label": "second" })),
        ];
        let cancellation = CancellationToken::new();

        let start = std::time::Instant::now();
        let futures = tools.iter().map(|(_, name, input)| {
            let tool = registry.get(name).expect("tool registered above");
            let cancellation = cancellation.clone();
            async move {
                tool.execute(
                    input.clone(),
                    crate::tools::ToolContext::detached(cancellation),
                )
                .await
            }
        });
        let outputs: Vec<_> = futures::future::join_all(futures).await;
        let elapsed = start.elapsed();

        // 500ms gives ~300ms headroom over the parallel ~200ms baseline while still being well
        // below the ~400ms serial-dispatch case. The wide margin absorbs scheduler jitter on slow
        // CI runners without losing the parallel-vs-sequential discrimination.
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "expected parallel execution (<500ms), got {elapsed:?}"
        );
        assert_eq!(outputs.len(), 2);
        let first = outputs[0].as_ref().expect("first should succeed");
        let second = outputs[1].as_ref().expect("second should succeed");
        assert_eq!(first.text_content(), "done:first");
        assert_eq!(second.text_content(), "done:second");
    }

    /// Verifies that the cancellation token threads through `tool.execute(...)` calls when many are
    /// in flight. Canceling mid-batch should cause every running tool to observe the cancellation
    /// and return early.
    #[tokio::test]
    async fn parallel_dispatch_respects_cancellation() {
        let registry = ToolRegistry::new_with_filter(BuiltinToolFilter::default());
        registry
            .register(Arc::new(SleepTool {
                name: "long_one".to_string(),
                delay: std::time::Duration::from_secs(10),
            }))
            .expect("registration should succeed");
        registry
            .register(Arc::new(SleepTool {
                name: "long_two".to_string(),
                delay: std::time::Duration::from_secs(10),
            }))
            .expect("registration should succeed");

        let tools = [
            ("a", "long_one", serde_json::json!({ "label": "first" })),
            ("b", "long_two", serde_json::json!({ "label": "second" })),
        ];
        let cancellation = CancellationToken::new();

        let cancel_handle = {
            let cancellation = cancellation.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                cancellation.cancel();
            })
        };

        let start = std::time::Instant::now();
        let futures = tools.iter().map(|(_, name, input)| {
            let tool = registry.get(name).expect("tool registered above");
            let cancellation = cancellation.clone();
            async move {
                tool.execute(
                    input.clone(),
                    crate::tools::ToolContext::detached(cancellation),
                )
                .await
            }
        });
        let outputs: Vec<_> = futures::future::join_all(futures).await;
        let elapsed = start.elapsed();
        cancel_handle.await.expect("cancel task should not panic");

        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "expected early exit on cancellation (<500ms), got {elapsed:?}"
        );
        for (i, output) in outputs.iter().enumerate() {
            let output = output.as_ref().expect("execute should not error");
            assert!(
                output.is_error,
                "tool {i} should report cancellation as is_error=true"
            );
            assert!(
                output.text_content().starts_with("canceled:"),
                "tool {} should report cancellation, got {:?}",
                i,
                output.content
            );
        }
    }

    #[test]
    fn builtin_primary_param_todo() {
        // set transitions take priority.
        assert_eq!(
            builtin_primary_param(
                "todo",
                &serde_json::json!({ "title": "Build", "set": {"2": "in_progress"} })
            )
            .as_deref(),
            Some("#2 in_progress")
        );
        // title when building a list.
        assert_eq!(
            builtin_primary_param(
                "todo",
                &serde_json::json!({ "title": "Refactor auth", "items": ["a", "b", "c"] })
            )
            .as_deref(),
            Some("Refactor auth")
        );
        // items size as a fallback when there's no title.
        assert_eq!(
            builtin_primary_param("todo", &serde_json::json!({ "items": ["a", "b", "c"] }))
                .as_deref(),
            Some("3 tasks")
        );
        // empty argument-less call reads.
        assert_eq!(
            builtin_primary_param("todo", &serde_json::json!({})).as_deref(),
            Some("read")
        );
    }

    #[test]
    fn builtin_primary_param_skill() {
        let input = serde_json::json!({"name": "setup-postgres"});
        assert_eq!(
            builtin_primary_param("skill_read", &input).as_deref(),
            Some("setup-postgres")
        );
    }

    /// A cancellation has to say what it canceled. `task_cancel` declares neither `id` nor `all`
    /// as required, so without a rule it renders as a bare tool name.
    #[test]
    fn builtin_primary_param_task_cancel() {
        assert_eq!(
            builtin_primary_param("task_cancel", &serde_json::json!({"id": "7f3a1c22"})).as_deref(),
            Some("7f3a1c22")
        );
        assert_eq!(
            builtin_primary_param("task_cancel", &serde_json::json!({"all": true})).as_deref(),
            Some("all")
        );
        // `all: false` alongside an id is the ordinary single cancel, not a bulk one.
        assert_eq!(
            builtin_primary_param(
                "task_cancel",
                &serde_json::json!({"id": "7f3a1c22", "all": false})
            )
            .as_deref(),
            Some("7f3a1c22")
        );
        assert_eq!(
            builtin_primary_param("task_cancel", &serde_json::json!({})),
            None
        );
    }

    /// The four MCP meta-tools that address a server deliberately show the object rather than the
    /// server, which is where the map departs from what `required[0]` would have picked.
    ///
    /// Written down because the departure looks like an oversight from the schema's side: reading
    /// `mcp_resource_read`'s `"required": ["server", "uri"]` alone, `server` is the obvious answer.
    /// It is also the useless one, identical across every call to a given server.
    #[test]
    fn builtin_primary_param_mcp_meta_tools_show_the_object() {
        let addressed = serde_json::json!({"server": "ida", "uri": "file:///tmp/a.i64"});
        for name in [
            "mcp_resource_read",
            "mcp_resource_subscribe",
            "mcp_resource_unsubscribe",
        ] {
            assert_eq!(
                builtin_primary_param(name, &addressed).as_deref(),
                Some("file:///tmp/a.i64"),
                "{name}"
            );
        }
        assert_eq!(
            builtin_primary_param(
                "mcp_prompt_get",
                &serde_json::json!({"server": "ida", "name": "explain"})
            )
            .as_deref(),
            Some("explain")
        );
        // The two list tools take only `server`, and it is optional: listing every server is the
        // documented default, and has nothing specific to show.
        assert_eq!(
            builtin_primary_param("mcp_resource_list", &serde_json::json!({"server": "ida"}))
                .as_deref(),
            Some("ida")
        );
        assert_eq!(
            builtin_primary_param("mcp_prompt_list", &serde_json::json!({})),
            None
        );
    }

    /// `context_compact` declares no `required`, so like `task_cancel` the schema fallback has
    /// nothing to reach for and the call would render bare on every surface, not just replay.
    #[test]
    fn builtin_primary_param_context_compact() {
        assert_eq!(
            builtin_primary_param(
                "context_compact",
                &serde_json::json!({"instructions": "keep the design decisions"})
            )
            .as_deref(),
            Some("keep the design decisions")
        );
        assert_eq!(
            builtin_primary_param(
                "context_compact",
                &serde_json::json!({"keep_recent": false})
            ),
            None
        );
    }

    /// The whole path the indicator actually uses, not just the built-in map.
    #[test]
    fn resolve_primary_param_renders_a_task_cancellation() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"id": {"type": "string"}, "all": {"type": "boolean"}}
        });
        assert_eq!(
            resolve_primary_param(
                "task_cancel",
                &serde_json::json!({"id": "7f3a1c22"}),
                Some(&schema)
            )
            .as_deref(),
            Some("7f3a1c22")
        );
    }

    #[test]
    fn builtin_primary_param_names_each_tool_s_primary_argument() {
        let input = serde_json::json!({"command": "ls", "path": "/tmp"});
        assert_eq!(
            builtin_primary_param("execute_command", &input).as_deref(),
            Some("ls")
        );
        assert_eq!(
            builtin_primary_param("read_file", &input).as_deref(),
            Some("/tmp")
        );
        assert_eq!(builtin_primary_param("unknown_tool", &input), None);
    }

    #[test]
    fn builtin_primary_param_missing() {
        let input = serde_json::json!({"other": "value"});
        assert_eq!(builtin_primary_param("execute_command", &input), None);
    }

    #[test]
    fn builtin_primary_param_render_image_from_scratchpad() {
        let input = serde_json::json!({"from_scratchpad": "frame4"});
        assert_eq!(
            builtin_primary_param("render_image", &input).as_deref(),
            Some("frame4")
        );
    }

    #[test]
    fn builtin_primary_param_render_image_inline_base64() {
        let input = serde_json::json!({"base64": "iVBOR..."});
        assert_eq!(
            builtin_primary_param("render_image", &input).as_deref(),
            Some("<inline base64>")
        );
    }

    #[test]
    fn builtin_primary_param_render_image_from_scratchpad_takes_precedence() {
        let input = serde_json::json!({"from_scratchpad": "frame4", "base64": "iVBOR..."});
        assert_eq!(
            builtin_primary_param("render_image", &input).as_deref(),
            Some("frame4")
        );
    }

    #[test]
    fn builtin_primary_param_render_image_empty() {
        let input = serde_json::json!({});
        assert_eq!(builtin_primary_param("render_image", &input), None);
    }

    #[test]
    fn schema_primary_param_string_value() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        });
        let input = serde_json::json!({"query": "best keyboards 2026"});
        assert_eq!(
            schema_primary_param(&schema, &input).as_deref(),
            Some("best keyboards 2026")
        );
    }

    #[test]
    fn schema_primary_param_array_of_strings() {
        let schema = serde_json::json!({
            "required": ["urls"],
        });
        let input = serde_json::json!({
            "urls": ["https://example.com", "https://other.example"],
        });
        assert_eq!(
            schema_primary_param(&schema, &input).as_deref(),
            Some("https://example.com, https://other.example")
        );
    }

    #[test]
    fn schema_primary_param_number_and_bool() {
        let schema = serde_json::json!({"required": ["count"]});
        let input = serde_json::json!({"count": 42});
        assert_eq!(schema_primary_param(&schema, &input).as_deref(), Some("42"));
        let schema = serde_json::json!({"required": ["enabled"]});
        let input = serde_json::json!({"enabled": true});
        assert_eq!(
            schema_primary_param(&schema, &input).as_deref(),
            Some("true")
        );
    }

    #[test]
    fn schema_primary_param_no_required_field() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
        });
        let input = serde_json::json!({"query": "hello"});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn schema_primary_param_required_key_absent_from_input() {
        let schema = serde_json::json!({"required": ["query"]});
        let input = serde_json::json!({"other_field": "value"});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn schema_primary_param_empty_required_array() {
        let schema = serde_json::json!({"required": []});
        let input = serde_json::json!({"query": "hello"});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn schema_primary_param_nested_object_skipped() {
        let schema = serde_json::json!({"required": ["config"]});
        let input = serde_json::json!({"config": {"nested": 1}});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn resolve_primary_param_builtin_takes_precedence_over_schema() {
        // A tool that happens to share a built-in name: hardcoded map wins so the display stays
        // consistent with what users know.
        let schema = serde_json::json!({"required": ["path"]});
        let input = serde_json::json!({"command": "ls -la", "path": "/ignored"});
        assert_eq!(
            resolve_primary_param("execute_command", &input, Some(&schema)).as_deref(),
            Some("ls -la")
        );
    }

    #[test]
    fn resolve_primary_param_falls_back_to_schema_for_unknown_tool() {
        let schema = serde_json::json!({"required": ["query"]});
        let input = serde_json::json!({"query": "claude code"});
        assert_eq!(
            resolve_primary_param("exa__web_search_exa", &input, Some(&schema)).as_deref(),
            Some("claude code")
        );
    }

    #[test]
    fn resolve_primary_param_no_schema_no_builtin() {
        let input = serde_json::json!({"anything": "here"});
        assert_eq!(
            resolve_primary_param("unknown__mcp_tool", &input, None),
            None
        );
    }
}
