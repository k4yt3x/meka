//! Small shared helpers for tool-input parsing and validation.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::ToolOutput;
use crate::{
    conversation::ToolResultContent,
    error::{MekaError, Result},
    image::{ImageHandling, prepare_image_source},
};

/// Default cap for regex-search-mode hits; shared by `file_read` and `scratchpad_read`.
pub(super) const MAX_SEARCH_MATCHES: usize = 100;

/// Wall-clock ceiling on one filesystem walk (`file_find`, `file_search`). The result caps
/// bound only what is returned, not what is examined, and a walk rooted high in the tree (`/proc`
/// and `/sys` alone are effectively unbounded) would otherwise run until the filesystem is
/// exhausted. Sized well above any plausible repository-scoped search and well below the point
/// where an unattended run looks hung.
const WALK_TIME_BUDGET: Duration = Duration::from_secs(60);

/// Why a walk stopped before it ran out of tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WalkStop {
    /// The enclosing turn was canceled (Ctrl+C, ACP `session/cancel`).
    Canceled,
    /// The walk outran [`WALK_TIME_BUDGET`].
    TimedOut,
}

/// Stop condition for the blocking filesystem walks.
///
/// Both walks run inside `tokio::task::spawn_blocking`, and a blocking task that has already
/// started cannot be aborted from the outside: dropping its `JoinHandle` detaches the thread but
/// does not stop it. The walk itself therefore has to be the thing that gives up, which is what
/// this type is for. Consult it at every step of the traversal.
pub(super) struct WalkBudget {
    budget: Duration,
    deadline: Instant,
    cancellation: CancellationToken,
}

impl WalkBudget {
    pub(super) fn new(cancellation: CancellationToken) -> Self {
        Self::with_budget(cancellation, WALK_TIME_BUDGET)
    }

    pub(super) fn with_budget(cancellation: CancellationToken, budget: Duration) -> Self {
        Self {
            budget,
            deadline: Instant::now() + budget,
            cancellation,
        }
    }

    /// Seconds the walk was allowed to run, for the message that reports a [`WalkStop::TimedOut`].
    pub(super) fn budget_secs(&self) -> u64 {
        self.budget.as_secs()
    }

    /// Called once per directory entry, so it stays to one atomic load plus one clock read.
    pub(super) fn check(&self) -> Option<WalkStop> {
        if self.cancellation.is_cancelled() {
            return Some(WalkStop::Canceled);
        }
        if Instant::now() >= self.deadline {
            return Some(WalkStop::TimedOut);
        }
        None
    }
}

/// Resolve the active session id for a session-scoped tool, erroring if no session is open. Shared
/// by the scratchpad and conversation tool families.
pub(super) fn resolve_session_id(
    session_id: &crate::session::SharedSessionId,
    tool_name: &str,
) -> Result<Uuid> {
    session_id.get().ok_or_else(|| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: "no active session".to_string(),
    })
}

/// The string a call sent in `field`: `None` when the field is absent or `null`, an error naming
/// the field when it holds anything but a string. The one reader under the four parameter helpers
/// below, so a wrong-typed value is refused the same way at every door: `as_str` alone would read
/// `permission: 0` as "not specified" and run a worker at the parent's own level.
fn string_field<'a>(
    input: &'a serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<Option<&'a str>> {
    match input.get(field) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(text)) => Ok(Some(text)),
        Some(_) => Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("'{field}' must be a string"),
        }),
    }
}

/// A required string parameter, refused when it is missing or blank.
pub(super) fn require_str(
    input: &serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<String> {
    // Blank is missing. Every caller names a thing (a path, a pattern, a command, a name), and an
    // empty one is never that thing: `file_edit` with an empty `old_string` and `replace_all`
    // spliced the replacement between every character of the file and reported success.
    string_field(input, field, tool_name)?
        .filter(|value| !crate::text::is_blank(value))
        .map(str::to_string)
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("missing or blank '{field}' parameter"),
        })
}

/// An optional string parameter: `None` when it is absent, `null` or blank. Blank is absent for
/// the reason blank is missing on [`require_str`], and because a model that fills every field it
/// was shown sends `""` where it has nothing to say, which a strict schema spells `null`. The value
/// is handed back as sent; a caller that parses a word trims it.
pub(super) fn optional_str<'a>(
    input: &'a serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<Option<&'a str>> {
    Ok(string_field(input, field, tool_name)?.filter(|value| !crate::text::is_blank(value)))
}

/// [`optional_str`] for a field whose blank value is meaningful: the replacement an edit makes,
/// a body or an entry's content to clear, an expiry to lift.
pub(super) fn optional_str_allowing_empty<'a>(
    input: &'a serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<Option<&'a str>> {
    string_field(input, field, tool_name)
}

/// [`require_str`] for a field whose empty value is meaningful: a file's whole content, the
/// replacement an edit makes.
pub(super) fn require_str_allowing_empty(
    input: &serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<String> {
    string_field(input, field, tool_name)?
        .map(str::to_string)
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("missing '{field}' parameter"),
        })
}

/// An optional boolean parameter: `None` when absent or `null`, refused naming the field when it
/// holds anything but `true` or `false`. `"true"` read as false would have the call do the
/// opposite of what it asked, as `force` asked of a write that is then refused for want of it.
pub(super) fn optional_bool(
    input: &serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<Option<bool>> {
    match input.get(field) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(value)) => Ok(Some(*value)),
        Some(value) => Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("'{field}' must be true or false, got {value}"),
        }),
    }
}

/// An optional whole-number parameter: `None` when absent or `null`, refused naming the field when
/// it holds anything but a non-negative integer as [`crate::text::whole_number`] reads one, `2.0`
/// included. Read as absent, `"3"` or `-1` would hand the call its default in place of the bound
/// it asked for: ten results to a call that asked for three, or a worker the nesting budget its
/// caller had just tried to deny it.
pub(super) fn optional_u64(
    input: &serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<Option<u64>> {
    match input.get(field) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => crate::text::whole_number(value)
            .and_then(|number| u64::try_from(number).ok())
            .map(Some)
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: format!("'{field}' must be a non-negative whole number, got {value}"),
            }),
    }
}

/// [`optional_u64`] for a field that may be negative.
pub(super) fn optional_i64(
    input: &serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<Option<i64>> {
    match input.get(field) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => crate::text::whole_number(value)
            .and_then(|number| i64::try_from(number).ok())
            .map(Some)
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: tool_name.to_string(),
                message: format!("'{field}' must be a whole number, got {value}"),
            }),
    }
}

/// [`optional_u64`] for a count, an offset or a size, saturating where `usize` is narrower.
pub(super) fn optional_count(
    input: &serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<Option<usize>> {
    Ok(optional_u64(input, field, tool_name)?
        .map(|value| usize::try_from(value).unwrap_or(usize::MAX)))
}

/// A field declared as a list of strings, read as the list, as one bare string (a mistake models
/// make constantly, and the recovery is free), or as nothing: absent, `null` and a blank string
/// are `None`. A blank entry is no entry, as a blank string is no value. An entry that is not a
/// string, or a value of another shape, is refused naming the field; dropping entries would turn
/// `[1, 2]` into `[]`, which for a field like `tags` is the documented "clear them" signal.
pub(super) fn string_or_list(
    input: &serde_json::Value,
    field: &str,
    tool_name: &str,
) -> Result<Option<Vec<String>>> {
    match input.get(field) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(single)) => {
            Ok((!crate::text::is_blank(single)).then(|| vec![single.clone()]))
        }
        Some(serde_json::Value::Array(values)) => {
            let mut entries = Vec::with_capacity(values.len());
            for value in values {
                let Some(text) = value.as_str() else {
                    return Err(MekaError::ToolExecution {
                        tool_name: tool_name.to_string(),
                        message: format!("every entry in '{field}' must be a string"),
                    });
                };
                if !crate::text::is_blank(text) {
                    entries.push(text.to_string());
                }
            }
            Ok(Some(entries))
        }
        Some(_) => Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("'{field}' must be a string or a list of strings"),
        }),
    }
}

/// Why an exact-anchor edit did not apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnchorMiss {
    /// The anchor appears nowhere in the text.
    NotFound,
    /// The anchor appears this many times and the caller did not ask for every occurrence.
    Ambiguous(usize),
}

/// Replace `old` with `new` in `text`: its one occurrence, or every occurrence when
/// `replace_all`. Returns the edited text, how many replacements were made, and the byte offset
/// of the first in the original, which `replacen` and `replace` leave stable in the result.
///
/// `file_edit` and `memory_edit` both apply an edit this way, so "an ambiguous anchor is refused
/// unless every occurrence was asked for" is one rule rather than one per tool.
pub(super) fn replace_anchored(
    text: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> std::result::Result<(String, usize, usize), AnchorMiss> {
    let Some(first) = text.find(old) else {
        return Err(AnchorMiss::NotFound);
    };
    let count = text.matches(old).count();
    if !replace_all && count > 1 {
        return Err(AnchorMiss::Ambiguous(count));
    }
    if replace_all {
        Ok((text.replace(old, new), count, first))
    } else {
        Ok((text.replacen(old, new, 1), 1, first))
    }
}

/// Compile an LLM-supplied regex pattern with bounded compile memory so a pathological pattern like
/// `a{10_000_000}` cannot exhaust the host's RAM during compilation. The `regex` crate's NFA/DFA
/// engines already avoid catastrophic backtracking at *match* time; the remaining DoS surface is
/// the one-time cost of building the automaton, which this bounds.
pub(super) fn compile_user_regex(pattern: &str, tool_name: &str) -> Result<regex::Regex> {
    const MAX_PATTERN_BYTES: usize = crate::text::MIB;
    const MAX_DFA_BYTES: usize = crate::text::MIB;

    regex::RegexBuilder::new(pattern)
        .size_limit(MAX_PATTERN_BYTES)
        .dfa_size_limit(MAX_DFA_BYTES)
        .build()
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("invalid or oversized regex '{pattern}': {error}"),
        })
}

/// Admit one read of `path`, as the model spelled it, against the session's read boundary, and
/// hand back the canonical path every later I/O uses: meka's own directories are refused below
/// `unrestricted`, and with reads confined so is anything outside the readable roots.
///
/// Judged twice. First as spelled, with the components that exist resolved, ahead of any step that
/// depends on the path existing: a path outside the boundary is then refused the same way whether
/// or not it exists, where resolving it first would answer "not found" for a missing one and
/// "outside" for one that is there, and the refusal names the spelling the model used rather than
/// where a link leads. Then on the canonical path, the one the I/O opens, so a link swapped in
/// between the two resolutions is judged where it lands; worded with the same spelling for the same
/// reason.
pub(super) async fn admit_read(
    tool_name: &str,
    site: &crate::session::ToolSite,
    path: &str,
) -> Result<PathBuf> {
    let resolved = crate::workspace::resolve_against_cwd(&site.cwd, path);
    let fence = site.read_scope().fence(&site.cwd);
    let refuse = |message: String| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message,
    };
    if let Some(message) = fence.refusal(&resolved) {
        return Err(refuse(message));
    }
    let canonical = canonicalize_for_tool(tool_name, &resolved).await?;
    if let Some(message) = fence.refusal_shown(&resolved, &canonical) {
        return Err(refuse(message));
    }
    Ok(canonical)
}

/// Resolve the path the model provided to a canonical absolute path, with every symlink component
/// resolved, so a symlink cannot be swapped between the permission check and the I/O. Callers use
/// the returned path for every later filesystem operation, never the raw string.
///
/// Errors when the path cannot be resolved rather than falling back to the raw path, which would
/// leave `..` and symlink components unresolved. A `file_write` target may not exist yet, so its
/// callers canonicalize the parent directory and re-join the file name.
pub(super) async fn canonicalize_for_tool(tool_name: &str, path: &Path) -> Result<PathBuf> {
    tokio::fs::canonicalize(path)
        .await
        // Windows' `canonicalize` returns the `\\?\` spelling, and this path is compared and keyed
        // against paths produced elsewhere (`WriteScope::admit`, the read tracker, the per-path
        // write lock) that were stripped; `strip_verbatim` is the identity everywhere else.
        .map(crate::workspace::strip_verbatim)
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("failed to resolve path '{}': {}", path.display(), error),
        })
}

/// Run a user-supplied regex against `content` line-by-line, returning `line_number:line` rows in a
/// `ToolOutput`. Caps results at [`MAX_SEARCH_MATCHES`] and reports the total when truncated.
/// `tool_name` is used only for the regex-compile error path.
pub(super) fn search_lines(content: &str, pattern: &str, tool_name: &str) -> Result<ToolOutput> {
    let re = compile_user_regex(pattern, tool_name)?;

    let mut matches = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        if re.is_match(line) {
            matches.push(format!("{}:{}", line_num + 1, line));
            if matches.len() >= MAX_SEARCH_MATCHES {
                break;
            }
        }
    }

    if matches.is_empty() {
        return Ok(ToolOutput::text(
            "No matches found for the given regex pattern.".to_string(),
            false,
        ));
    }

    let total_matches = if matches.len() >= MAX_SEARCH_MATCHES {
        // Count the whole document rather than resuming after the cap: skipping `matches.len()`
        // lines to account for that many matches double-counts every hit past that index.
        content.lines().filter(|line| re.is_match(line)).count()
    } else {
        matches.len()
    };

    let mut result = matches.join("\n");
    if total_matches > MAX_SEARCH_MATCHES {
        result.push_str(&format!(
            "\n\n... (showing first {MAX_SEARCH_MATCHES} of {total_matches} matches)",
        ));
    }

    Ok(ToolOutput::text(result, false))
}

/// The lines a read tool was asked for, as the model wrote them: counted from 1, `end` inclusive,
/// which is how [`search_lines`] reports a match and how every editor and compiler names a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LineRange {
    pub(super) start: usize,
    pub(super) end: Option<usize>,
}

impl LineRange {
    /// Every line from the first.
    pub(super) const WHOLE: Self = Self {
        start: 1,
        end: None,
    };

    /// The 0-based index of the first line.
    pub(super) fn first_index(self) -> usize {
        self.start.saturating_sub(1)
    }

    /// How many lines the range names, or `None` when it runs to the end.
    pub(super) fn count(self) -> Option<usize> {
        self.end
            .map(|end| end.saturating_sub(self.start).saturating_add(1))
    }
}

/// Parse `start` and `end` from a read tool's input. `None` when the call named neither, so a
/// caller can tell a whole read from a window that happens to begin at line 1. A value that is not
/// a line number counted from 1 is refused rather than clamped, because a `0` is a model counting
/// from zero, and clamping it would silently shift every window it asks for by one.
pub(super) fn line_range(input: &serde_json::Value, tool_name: &str) -> Result<Option<LineRange>> {
    let start = line_number(input, "start", tool_name)?;
    let end = line_number(input, "end", tool_name)?;
    if start.is_none() && end.is_none() {
        return Ok(None);
    }
    let start = start.unwrap_or(1);
    if let Some(end) = end
        && end < start
    {
        return Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("'end' ({end}) is before 'start' ({start})"),
        });
    }
    Ok(Some(LineRange { start, end }))
}

fn line_number(input: &serde_json::Value, key: &str, tool_name: &str) -> Result<Option<usize>> {
    match input.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => {
            match crate::text::whole_number(value).and_then(|number| usize::try_from(number).ok()) {
                Some(number) if number >= 1 => Ok(Some(number)),
                _ => Err(MekaError::ToolExecution {
                    tool_name: tool_name.to_string(),
                    message: format!("'{key}' must be a line number counted from 1, got {value}"),
                }),
            }
        }
    }
}

/// The note a read appends after the lines it returned: the range shown, counted from 1, the
/// total, why it stopped short when it did, and the line to continue from. One wording for every
/// read tool, so a continuation is written the same way whatever produced the lines.
pub(super) fn lines_shown_notice(
    first: usize,
    last: usize,
    total: usize,
    cut: Option<&str>,
) -> String {
    let mut notice = format!("(showing lines {first}-{last} of {total}");
    if let Some(cut) = cut {
        notice.push_str(", ");
        notice.push_str(cut);
    }
    if last < total {
        notice.push_str(&format!("; continue from line {}", last + 1));
    }
    notice.push(')');
    notice
}

/// The reply to a `start` past the last line. Returning nothing would read as "empty", a
/// different fact, and the one the model would act on.
pub(super) fn no_lines_notice(start: usize, subject: &str, total: usize) -> String {
    format!(
        "(no lines: start {start} is past the end of {subject}, which has {total} line{})",
        if total == 1 { "" } else { "s" },
    )
}

pub(super) fn truncate_string(string: &str, max_length: usize) -> &str {
    if string.len() <= max_length {
        string
    } else {
        &string[..string.floor_char_boundary(max_length)]
    }
}

/// Whether the caller is redirecting this tool's output into the scratchpad via the `scratchpad`
/// parameter. Tools that internally cap result counts or output length should lift those caps when
/// this returns true, because the scratchpad is an overflow buffer and truncation defeats its
/// purpose.
pub(super) fn redirects_to_scratchpad(input: &serde_json::Value) -> bool {
    // A plain predicate rather than `optional_str`: a `scratchpad` of the wrong type is refused
    // ahead of dispatch, by `meka_parameter_error`.
    matches!(
        input.get(crate::tools::SCRATCHPAD_PARAMETER),
        Some(serde_json::Value::String(name)) if !crate::text::is_blank(name)
    )
}

/// Build a two-block `ToolOutput` (text marker + multimodal Image) from raw image bytes plus a
/// pre-computed classification. Wraps `prepare_image_payload` so error paths become a text
/// `ToolOutput` with `is_error: true`. Shared by `web_fetch`, `file_read`, and `image_render`.
pub(crate) fn build_image_tool_output(
    marker: &str,
    handling: ImageHandling,
    bytes: &[u8],
) -> ToolOutput {
    let source = match prepare_image_source(handling, bytes) {
        Ok(source) => source,
        Err(message) => {
            return ToolOutput::text(format!("Error: {marker}: {message}"), true);
        }
    };

    ToolOutput {
        content: vec![
            ToolResultContent::Text {
                text: format!("[{marker}]"),
            },
            ToolResultContent::Image { source },
        ],
        is_error: false,
        is_output: false,
        spill_hint: crate::tools::SpillHint::default(),
        frontend_metadata: None,
        structured: None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_string_cuts_only_what_exceeds_the_limit() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 5), "hello");
    }

    #[test]
    fn compile_user_regex_rejects_oversized() {
        // Pattern that compiles to a gigantic automaton; must be rejected by the size limit rather
        // than consume host memory.
        let result = compile_user_regex("a{10000000}", "tool_for_test");
        assert!(result.is_err(), "oversized pattern should be rejected");
    }

    #[test]
    fn compile_user_regex_accepts_normal_pattern() {
        let re = compile_user_regex(r"\d+", "tool_for_test").expect("normal pattern compiles");
        assert!(re.is_match("abc 123"));
    }

    #[test]
    fn compile_user_regex_rejects_invalid_syntax() {
        let result = compile_user_regex("[invalid", "tool_for_test");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn canonicalize_for_tool_resolves_existing() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let file_path = temp_dir.path().join("a.txt");
        std::fs::write(&file_path, "x").expect("write");

        let canonical = canonicalize_for_tool("tool_for_test", &file_path)
            .await
            .expect("canonicalize");
        assert_eq!(canonical, crate::workspace::canonical_for_test(&file_path));
        // And specifically not the raw spelling: every other door in the tree normalizes, so one
        // that did not would hand back a key nothing else could match.
        assert!(
            !canonical.to_string_lossy().starts_with(r"\\?\"),
            "the verbatim prefix must not survive: {}",
            canonical.display()
        );
    }

    #[tokio::test]
    async fn canonicalize_for_tool_errors_on_missing() {
        let result = canonicalize_for_tool(
            "tool_for_test",
            std::path::Path::new("/this/path/definitely/does/not/exist-xyzzy"),
        )
        .await;
        let error = result.expect_err("missing path should error");
        let message = error.to_string();
        assert!(
            message.contains("failed to resolve path"),
            "unexpected error message: {message}",
        );
    }

    #[test]
    fn walk_budget_allows_work_within_budget() {
        let budget = WalkBudget::new(CancellationToken::new());
        assert!(budget.check().is_none());
    }

    #[test]
    fn walk_budget_stops_on_expired_deadline() {
        let budget = WalkBudget::with_budget(CancellationToken::new(), Duration::from_secs(0));
        assert_eq!(budget.check(), Some(WalkStop::TimedOut));
    }

    #[test]
    fn walk_budget_reports_cancellation_before_timeout() {
        // Both conditions hold; cancellation is the more specific answer and maps to
        // `MekaError::Interrupted` rather than a partial result.
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let budget = WalkBudget::with_budget(cancellation, Duration::from_secs(0));
        assert_eq!(budget.check(), Some(WalkStop::Canceled));
    }

    #[test]
    fn walk_budget_reports_its_own_budget() {
        let budget = WalkBudget::with_budget(CancellationToken::new(), Duration::from_secs(5));
        assert_eq!(budget.budget_secs(), 5);
    }

    /// A boolean parameter is `true`, `false` or not given; a string spelling one is refused by
    /// name rather than read as false.
    #[test]
    fn a_boolean_parameter_is_a_boolean_or_nothing() {
        let input = serde_json::json!({"yes": true, "no": false, "spelled": "true", "gone": null});
        assert_eq!(
            optional_bool(&input, "yes", "a_tool").expect("a bool"),
            Some(true)
        );
        assert_eq!(
            optional_bool(&input, "no", "a_tool").expect("a bool"),
            Some(false)
        );
        assert_eq!(
            optional_bool(&input, "gone", "a_tool").expect("not given"),
            None
        );
        assert_eq!(
            optional_bool(&input, "missing", "a_tool").expect("not given"),
            None
        );
        let refused = optional_bool(&input, "spelled", "a_tool").expect_err("a string");
        assert!(
            refused
                .to_string()
                .contains("'spelled' must be true or false"),
            "{refused}"
        );
    }

    /// A number parameter is an integer of its type or not given, `2.0` being the integer JSON
    /// Schema says it is: a quoted number, a fraction, or a negative where only a count makes sense
    /// is refused naming the field rather than read as absent, since absent is the default the call
    /// did not ask for.
    #[test]
    fn a_number_parameter_is_an_integer_or_nothing() {
        let input = serde_json::json!({
            "zero": 0,
            "negative": -1,
            "quoted": "0",
            "fraction": 2.5,
            "integral": 2.0,
            "gone": null,
        });
        assert_eq!(
            optional_u64(&input, "zero", "a_tool").expect("a count"),
            Some(0)
        );
        assert_eq!(
            optional_u64(&input, "integral", "a_tool").expect("2.0 is the integer 2"),
            Some(2)
        );
        assert_eq!(
            optional_i64(&input, "integral", "a_tool").expect("2.0 is the integer 2"),
            Some(2)
        );
        assert_eq!(
            optional_count(&input, "zero", "a_tool").expect("a count"),
            Some(0)
        );
        assert_eq!(
            optional_i64(&input, "negative", "a_tool").expect("signed"),
            Some(-1)
        );
        for key in ["gone", "missing"] {
            assert_eq!(
                optional_u64(&input, key, "a_tool").expect("not given"),
                None
            );
            assert_eq!(
                optional_i64(&input, key, "a_tool").expect("not given"),
                None
            );
        }
        for key in ["negative", "quoted", "fraction"] {
            let refused = optional_u64(&input, key, "a_tool").expect_err("not a count");
            assert!(
                refused.to_string().contains(&format!("'{key}'")),
                "{refused}"
            );
        }
        for key in ["quoted", "fraction"] {
            let refused = optional_i64(&input, key, "a_tool").expect_err("not an integer");
            assert!(
                refused.to_string().contains(&format!("'{key}'")),
                "{refused}"
            );
        }
    }

    /// Every string parameter is read by one rule: absent, `null` and blank are "not given",
    /// which is missing for a required field and omitted for an optional one; a value of another
    /// type is refused naming the field, never read as absent, since for a restriction absent is
    /// the dangerous direction (`permission: 0` would inherit the parent's level). The value comes
    /// back as sent, and only the fields that opt out by name keep a blank.
    #[test]
    fn string_parameters_are_read_by_one_rule() {
        let input = serde_json::json!({
            "number": 0,
            "flag": true,
            "good": " read ",
            "blank": "   ",
            "empty": "",
            "explicit_null": null,
        });
        for key in ["number", "flag"] {
            for error in [
                optional_str(&input, key, "a_tool").expect_err("a non-string is refused"),
                optional_str_allowing_empty(&input, key, "a_tool").expect_err("refused"),
                require_str(&input, key, "a_tool").expect_err("refused"),
                require_str_allowing_empty(&input, key, "a_tool").expect_err("refused"),
            ] {
                assert!(error.to_string().contains(key), "{error}");
                assert!(error.to_string().contains("must be a string"), "{error}");
            }
        }
        assert_eq!(
            optional_str(&input, "good", "a_tool").expect("a string"),
            Some(" read "),
            "the value is handed back as sent"
        );
        assert_eq!(
            require_str(&input, "good", "a_tool").expect("a string"),
            " read "
        );
        for key in ["missing", "blank", "empty", "explicit_null"] {
            assert_eq!(
                optional_str(&input, key, "a_tool").expect("not given"),
                None
            );
            let error = require_str(&input, key, "a_tool").expect_err("missing");
            assert!(error.to_string().contains("missing or blank"), "{error}");
        }
        assert_eq!(
            optional_str_allowing_empty(&input, "empty", "a_tool").expect("a value"),
            Some(""),
            "a field that opts out keeps its blank"
        );
        assert_eq!(
            require_str_allowing_empty(&input, "empty", "a_tool").expect("a value"),
            ""
        );
        assert_eq!(
            optional_str_allowing_empty(&input, "explicit_null", "a_tool").expect("not given"),
            None
        );
    }

    /// A list field takes a list, one bare string, or nothing, and blank is nothing here too, as
    /// a field and as an entry; an entry of another type is refused rather than dropped, and an
    /// empty list stays a list.
    #[test]
    fn a_list_field_takes_a_list_a_string_or_nothing() {
        let input = serde_json::json!({
            "list": ["a", " ", "b"],
            "one": "a",
            "blank": " ",
            "none": [],
            "mixed": ["a", 1],
            "number": 1,
        });
        assert_eq!(
            string_or_list(&input, "list", "a_tool").expect("a list"),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            string_or_list(&input, "one", "a_tool").expect("one"),
            Some(vec!["a".to_string()])
        );
        assert_eq!(
            string_or_list(&input, "none", "a_tool").expect("an empty list"),
            Some(Vec::new())
        );
        for key in ["blank", "missing"] {
            assert_eq!(
                string_or_list(&input, key, "a_tool").expect("nothing"),
                None
            );
        }
        for key in ["mixed", "number"] {
            let error = string_or_list(&input, key, "a_tool").expect_err("refused");
            assert!(error.to_string().contains(key), "{error}");
        }
    }

    #[test]
    fn redirects_to_scratchpad_needs_a_non_empty_scratchpad_name() {
        assert!(redirects_to_scratchpad(
            &serde_json::json!({ "scratchpad": "img" })
        ));
        assert!(!redirects_to_scratchpad(
            &serde_json::json!({ "scratchpad": "" })
        ));
        assert!(!redirects_to_scratchpad(
            &serde_json::json!({ "scratchpad": "  " })
        ));
        assert!(!redirects_to_scratchpad(&serde_json::json!({})));
        assert!(!redirects_to_scratchpad(
            &serde_json::json!({ "from_scratchpad": "img" })
        ));
    }

    /// A blank string names nothing, so it is refused the way a missing field is. Two copies of
    /// this function disagreed on it, and the lenient one let `file_edit` accept an empty
    /// `old_string`.
    #[test]
    fn a_blank_string_is_a_missing_parameter() {
        let input = serde_json::json!({"path": "  ", "content": ""});
        assert!(require_str(&input, "path", "t").is_err());
        assert!(require_str(&input, "absent", "t").is_err());
        assert_eq!(
            require_str_allowing_empty(&input, "content", "t").expect("empty content is content"),
            ""
        );
    }
}
