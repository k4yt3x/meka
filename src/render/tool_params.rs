//! The tool indicator and the argument block under it: the name, the width budget, and the
//! bounded rendering of a call's arguments for the indicator and for an approval prompt.

use super::*;

/// Columns a tool name may occupy before it is elided.
///
/// The name is served first because it is the part that identifies the call. A truncated argument
/// still conveys its gist (`Jane Street first mon...` is recognizably a search); a truncated name
/// frequently conveys nothing, since MCP names share long prefixes. A genuine name is also not the
/// model's to shape: a built-in's is one meka chose, and an MCP name is normalized at registration.
/// This bound exists for the remaining case, a hallucinated name, which is unvalidated at render
/// time and otherwise unbounded. No genuine name approaches it: built-ins stop at 25 columns
/// (`mcp_resource_updates_list`) and `mcp__exa__web_search_exa` is 24.
pub(super) const TOOL_NAME_MAX_WIDTH: usize = 64;

/// Below this many columns for the argument there is nothing worth showing, so the indicator drops
/// the parenthetical instead of printing an ellipsis in backticks.
pub(super) const TOOL_ARGUMENT_FLOOR: usize = 8;

/// Fixed chrome in `[tool NAME(`ARG`)]`.
pub(super) const TOOL_INDICATOR_CHROME: usize = "[tool (``)]".len();

/// Fixed chrome in `[tool NAME]`.
pub(super) const TOOL_HEADER_CHROME: usize = "[tool ]".len();

/// The bare `[tool X]` line, with no argument.
///
/// The name is sanitized like any other model-supplied string. It arrives verbatim off the provider
/// stream, and while the registry is consulted just before the event is emitted, that lookup only
/// fetches the schema: a name matching nothing still reaches here. (An MCP tool's name is
/// separately normalized to `[A-Za-z0-9_-]` when its server is registered.)
pub(super) fn tool_header(name: &str, width: usize) -> String {
    let display_name = sanitize_to_line(name, usize::MAX);
    format!(
        "[tool {}]",
        elide_to_width(
            &display_name,
            TOOL_NAME_MAX_WIDTH.min(width.saturating_sub(TOOL_HEADER_CHROME))
        )
    )
}

/// Compose the "[tool X(`arg`)]" indicator line.
///
/// The agent loop computes `display_summary` (via [`resolve_primary_param`] over the tool's JSON
/// Schema) and passes it pre-resolved, so the frontend layer does not need the schema. See
/// `FrontendEvent::ToolCallStarted` in `crate::frontend`.
///
/// Replayed history has no schemas to resolve against and passes `None`, which is why the fallback
/// here exists: a built-in's primary parameter is known from its name alone, so a replayed
/// `file_read` shows the path it showed live instead of a bare `[tool file_read]`. An MCP tool
/// replayed from history does stay bare, which is the honest answer -- without its schema nothing
/// says which of its arguments is the one worth showing.
///
/// The whole line is budgeted, not each part, so adjacent indicators that had to be cut end at the
/// same column instead of wherever their own name happened to leave them. Within that budget the
/// name is served first and the argument takes what is left; when what is left is not worth
/// printing, the argument goes and the name stays whole.
pub(super) fn tool_indicator_line(
    name: &str,
    input: &serde_json::Value,
    display_summary: Option<&str>,
    width: usize,
) -> String {
    let resolved = display_summary
        .map(str::to_string)
        .or_else(|| resolve_primary_param(name, input, None));
    let Some(value) = resolved
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return tool_header(name, width);
    };

    let available = width.saturating_sub(TOOL_INDICATOR_CHROME);
    // Sanitize before measuring, then truncate: the display width of the raw name is not the width
    // of what gets printed once escapes and format characters are gone.
    let display_name = sanitize_to_line(name, usize::MAX);
    let display_name = elide_to_width(&display_name, TOOL_NAME_MAX_WIDTH.min(available));
    let argument_budget = available.saturating_sub(display_width(&display_name));
    if argument_budget < TOOL_ARGUMENT_FLOOR {
        return tool_header(name, width);
    }
    format!(
        "[tool {}(`{}`)]",
        display_name,
        // Elided from the middle, not the tail: the primary parameter is an identifier -- a path,
        // a URL, a command -- and its end carries the filename or the destination.
        elide_to_width(&sanitize_to_line(value, usize::MAX), argument_budget)
    )
}

/// One level of nesting in a [`ToolParams::Full`] block.
pub(super) const TOOL_PARAM_INDENT: &str = "  ";

/// Columns a value keeps however long its key is, mirroring [`TOOL_NAME_MAX_WIDTH`]'s reasoning in
/// the other direction: here the key is the label and the value is the substance.
pub(super) const TOOL_VALUE_MIN_WIDTH: usize = 16;

/// How much of a call an argument block may show.
///
/// The same renderer serves two audiences with opposite needs. A tool indicator is a notification
/// scrolling past, so it may trade completeness for brevity. An approval prompt is a decision, so
/// it may not: what it hides is what you authorize unseen.
#[derive(Debug, Clone, Copy)]
pub(super) struct BlockLimits {
    /// Source lines shown under one argument's key before the rest is summarized as a count.
    ///
    /// Counted in the value's own lines, so the `... N more lines` marker means what it says.
    /// Capping rendered *rows* and reporting those as lines tells the reader of a 100-line file
    /// that 108 lines were hidden.
    pub(super) lines_per_argument: usize,
    /// Rows one argument may occupy, however many lines or elements are under its key.
    ///
    /// [`lines_per_argument`](Self::lines_per_argument) does not bound this on its own. A
    /// container has no lines to count and fans out one row per element; a wrapped string
    /// turns one line into several. Without a separate bound the two caps composed by addition
    /// and an argument could spend the whole block's budget by itself.
    pub(super) rows_per_argument: usize,
    /// Rows the block may already hold before the remaining arguments are dropped and named.
    ///
    /// Checked before an argument is rendered, never in the middle of one, so a block reaches at
    /// most this plus [`rows_per_argument`](Self::rows_per_argument) plus the line that names what
    /// went. That sum is the real ceiling, and it is the number the docs quote.
    ///
    /// Both audiences need a ceiling: "show everything" with none lets two hundred decoy arguments
    /// push the real one off the top, which is the same outcome dropping was supposed to be worse
    /// than, minus the marker that would have warned anybody.
    pub(super) block_rows: usize,
    /// Wrap a too-wide line onto the next row instead of cutting it.
    pub(super) wrap: bool,
}

impl BlockLimits {
    /// For `[display].tool_params = "full"`. A `file_write` carrying a whole source file has to be
    /// readable as "this happened" without evicting the turn from scrollback; the untruncated text
    /// is what `meka session export` is for.
    ///
    /// Worst case 93 rows: 59 already there, 33 for the argument that crossed the line, one naming
    /// the rest.
    pub(super) fn indicator() -> Self {
        Self {
            lines_per_argument: 30,
            // One over `lines_per_argument`, plus the key line: enough that a string value capped
            // by its own line budget is never cut again by this one, and its `... N
            // more lines` marker survives to be read.
            rows_per_argument: 32,
            block_rows: 60,
            wrap: false,
        }
    }

    /// For the approval prompt.
    ///
    /// Wrapping rather than cutting, since cutting a line hides the tail of the command being
    /// approved. Twenty lines rather than thirty because a prompt blocks reading and wants to be
    /// short.
    ///
    /// The block ceiling is generous but real, and worth 161 rows in the worst case. Dropping an
    /// argument from a decision is bad, and it was tempting to allow none; but leaving the block
    /// unbounded lets two hundred decoy arguments scroll the real tool name and the real payload
    /// off the top, which is the same outcome with no marker to warn anybody. A named drop is
    /// the lesser harm, and it is the signal to deny.
    pub(super) fn approval() -> Self {
        Self {
            lines_per_argument: 20,
            // Three times the line budget, so wrapping has room to be worth having: a single-line
            // `shell_execute` -- the commonest approval there is -- gets all sixty rows to
            // itself.
            rows_per_argument: 60,
            block_rows: 100,
            wrap: true,
        }
    }
}

/// A width and the limits to render within it. They always travel together, so the `push_*` helpers
/// take one value rather than two parameters.
#[derive(Debug, Clone, Copy)]
pub(super) struct BlockContext {
    pub(super) width: usize,
    pub(super) limits: BlockLimits,
}

/// Render a tool call's whole input as an indented block, one line per element.
///
/// Deliberately not JSON. Quoting every key and escaping every newline turns the two tools whose
/// arguments most need reading (`file_edit`, `file_write`) into a single unreadable line, which is
/// the opposite of what asking for full parameters means. So: a value that fits on a line follows
/// its key, a value that does not gets an indented block under a bare `key:`, and nesting is
/// carried by indentation with `-` for array elements. The cost is that the string/number
/// distinction is gone, which is why the exact JSON stays available through `meka session export`.
pub(super) fn render_tool_params(
    input: &serde_json::Value,
    width: usize,
    limits: BlockLimits,
) -> Vec<String> {
    let context = BlockContext { width, limits };
    let serde_json::Value::Object(fields) = input else {
        // A tool whose input is not an object at all. Nothing sensible to key it by, so it renders
        // as a bare value rather than being dropped, which would read as "no arguments".
        let mut lines = Vec::new();
        match input {
            serde_json::Value::Array(items) => {
                for item in items {
                    push_item(&mut lines, 1, item, context);
                }
            }
            other => push_value_body(&mut lines, 1, other, context),
        }
        // There is no key here to hang a per-argument cap on, and the block ceiling below is only
        // reached through the object path, so without this an array input printed every element it
        // had.
        cap_rows(&mut lines, limits.block_rows, TOOL_PARAM_INDENT, width);
        return lines;
    };

    let mut lines = Vec::new();
    let mut omitted: Vec<String> = Vec::new();
    for (key, value) in fields {
        // Whole arguments are dropped at their own boundary rather than the block being cut
        // wherever the last row happens to land. Cutting mid-argument loses that argument's own
        // elision marker, and reports a count that bears no relation to how much was hidden.
        // Dropping by argument lets the block say what is missing by name, which is what a reader
        // needs: `path` disappearing entirely is worse than any amount of `content` being trimmed.
        if !omitted.is_empty() || lines.len() >= limits.block_rows {
            omitted.push(sanitize_to_line(key, width));
            continue;
        }
        let mut param = Vec::new();
        push_param(&mut param, 1, key, value, context);
        // `lines_per_argument` bounds a *string* value, counted in its own lines by
        // `push_value_body`. A container has no lines to count: it fans out one row per element and
        // needs a row bound of its own, or a single array argument outruns the block on its own.
        // The key line is never cut -- an argument whose name you can read, trimmed, beats one that
        // vanished -- so the ceiling applies to what hangs off it.
        cap_rows(
            &mut param,
            limits.rows_per_argument,
            &TOOL_PARAM_INDENT.repeat(2),
            width,
        );
        lines.append(&mut param);
    }
    if !omitted.is_empty() {
        // Budgeted as one string. Cutting only the names left the count and the words around it
        // unmeasured, and `  ... 240 more arguments: ` is twenty-six columns before a single name
        // is added, so at the narrow end this line alone broke the width every other line
        // here keeps.
        lines.push(truncate_to_width(
            &format!(
                "{}... {} more argument{}: {}",
                TOOL_PARAM_INDENT,
                omitted.len(),
                if omitted.len() == 1 { "" } else { "s" },
                omitted.join(", ")
            ),
            width,
        ));
    }
    lines
}

/// Cut `lines` to `max_rows`, saying how many rows went and **keeping the last one**.
///
/// Counted in rows rather than in the value's source lines, and the wording follows: this is the
/// bound that keeps the block on the screen, and after wrapping a source line is not a row.
///
/// Keeping the last row is the same rule [`wrap_to_width`] and [`elide_to_width`] follow, for the
/// same reason: what a reader most needs from a thing too big to show is its beginning and its end.
/// Plain truncation also deleted whatever marker the row below had carried -- an argument that had
/// already reported `... 480 more lines` lost that line to this cut, so the block ended up
/// admitting to two dropped rows and nothing else.
pub(super) fn cap_rows(lines: &mut Vec<String>, max_rows: usize, indent: &str, width: usize) {
    if lines.len() <= max_rows || max_rows < 2 {
        lines.truncate(lines.len().min(max_rows));
        return;
    }
    let Some(last) = lines.last().cloned() else {
        return;
    };
    // The marker and the kept row are two of the `max_rows`, so the head keeps the rest.
    let elided = lines.len() - (max_rows - 1);
    lines.truncate(max_rows - 2);
    lines.push(format!(
        "{}{}",
        indent,
        truncate_to_width(
            &format!(
                "... {} more row{}",
                elided,
                if elided == 1 { "" } else { "s" }
            ),
            width.saturating_sub(display_width(indent))
        )
    ));
    lines.push(last);
}

/// Append one `key: value` pair at `depth`, recursing for containers.
pub(super) fn push_param(
    lines: &mut Vec<String>,
    depth: usize,
    key: &str,
    value: &serde_json::Value,
    context: BlockContext,
) {
    let indent = block_indent(depth, context.width);
    let available = context.width.saturating_sub(display_width(&indent));
    // A key is model-supplied too: an MCP tool's arguments are whatever the model generated, and
    // nothing checks them against the schema before they are rendered. Its budget reserves room for
    // the value, for the reason on `TOOL_VALUE_MIN_WIDTH`.
    // Floored: a key cut to nothing renders as a bare `:` with no sign anything was there, which
    // is worse than a short name. `truncate_to_width` marks whatever it cuts.
    let key = sanitize_to_line(
        key,
        available
            .saturating_sub(TOOL_VALUE_MIN_WIDTH + ": ".len())
            .max(TRUNCATION_MARKER.len() + 1),
    );
    match value {
        serde_json::Value::Object(fields) if !fields.is_empty() => {
            lines.push(format!("{indent}{key}:"));
            for (nested_key, nested) in fields {
                push_param(lines, depth + 1, nested_key, nested, context);
            }
        }
        serde_json::Value::Array(items) if !items.is_empty() => {
            lines.push(format!("{indent}{key}:"));
            for item in items {
                push_item(lines, depth + 1, item, context);
            }
        }
        serde_json::Value::String(text) if is_multi_line(text) => {
            lines.push(format!("{indent}{key}:"));
            push_value_body(lines, depth + 1, value, context);
        }
        _ => {
            let value_budget = available.saturating_sub(display_width(&key) + ": ".len());
            // When wrapping, a value too wide for the key line gets a block of its own rather than
            // being cut on it. Otherwise the commonest approval of all -- a long `shell_execute`
            // pipeline, which is one line and so never reached `push_value_body` -- would have its
            // tail hidden, which is the whole failure this mode exists to avoid.
            // `wrap` first: rendering the value at full width to measure it is a whole
            // sanitization pass over a megabyte-sized argument, and the indicator never uses it.
            if context.limits.wrap && display_width(&scalar_text(value, usize::MAX)) > value_budget
            {
                lines.push(format!("{indent}{key}:"));
                push_value_body(lines, depth + 1, value, context);
            } else {
                lines.push(format!(
                    "{}{}: {}",
                    indent,
                    key,
                    scalar_text(value, value_budget)
                ));
            }
        }
    }
}

/// Whether a string needs a block of its own rather than a spot on the key line.
///
/// Only `\n` counts, matching [`str::lines`], which is what splits the block. A stray `\r` is a
/// cursor movement rather than a line break and is flattened by [`sanitize_to_line`] instead;
/// treating it as a break here would turn a one-line value into a two-line block.
///
/// A trailing newline does not count either. A `file_write` body almost always ends with one, and
/// counting it turned a one-line value into a bare `key:` followed by a single indented line.
pub(super) fn is_multi_line(text: &str) -> bool {
    text.trim_end_matches('\n').contains('\n')
}

/// Append one array element at `depth`, bulleted with `-`.
///
/// An element that is itself an object puts its first field on the bullet line and aligns the rest
/// under it, so a list of records reads as records rather than as a run of bullets.
pub(super) fn push_item(
    lines: &mut Vec<String>,
    depth: usize,
    item: &serde_json::Value,
    context: BlockContext,
) {
    // Bounded like every other indent in the block. An array nests through this function rather
    // than through `push_param`, so leaving it proportional to depth put a bullet at column 40
    // of a 40-column line and pushed everything under it past the edge.
    let indent = block_indent(depth, context.width);
    match item {
        serde_json::Value::Object(fields) if !fields.is_empty() => {
            let nested_indent = block_indent(depth + 1, context.width);
            // The bullet replaces exactly the one indent level that `depth + 1` added, so the field
            // lands where it would have without the bullet and its siblings stay aligned with it.
            // Past the indent ceiling there is no such level: `depth + 1` indents no further, the
            // field was budgeted against that same indent, and hoisting it would widen its line by
            // the bullet. So the bullet takes a row of its own there, as the arms below already do.
            let hoisted = nested_indent.len() > indent.len();
            if !hoisted {
                lines.push(format!("{indent}-"));
            }
            let first = lines.len();
            for (key, value) in fields {
                push_param(lines, depth + 1, key, value, context);
            }
            if hoisted && let Some(line) = lines.get_mut(first) {
                let body = line
                    .strip_prefix(&nested_indent)
                    .unwrap_or(line)
                    .to_string();
                *line = format!("{indent}- {body}");
            }
        }
        serde_json::Value::Array(nested) if !nested.is_empty() => {
            lines.push(format!("{indent}-"));
            for value in nested {
                push_item(lines, depth + 1, value, context);
            }
        }
        // Mirrors `push_param`'s multi-line arm. Without it a bulleted string kept its newlines and
        // put every line after the first at column 0, which is both wrong to read and enough to
        // forge a `[tool ...]` header outside the block.
        serde_json::Value::String(text) if is_multi_line(text) => {
            lines.push(format!("{indent}-"));
            push_value_body(lines, depth + 1, item, context);
        }
        _ => {
            let budget = context
                .width
                .saturating_sub(display_width(&indent) + "- ".len());
            // Same promotion `push_param` makes: under wrapping, a value too wide for its own row
            // gets a block rather than losing its tail. An MCP tool taking `["bash", "-lc", "<long
            // command>"]` is the shape this exists for, and approvals gate MCP tools.
            if context.limits.wrap && display_width(&scalar_text(item, usize::MAX)) > budget {
                lines.push(format!("{indent}-"));
                push_value_body(lines, depth + 1, item, context);
            } else {
                lines.push(format!("{}- {}", indent, scalar_text(item, budget)));
            }
        }
    }
}

/// Append a value with no key of its own: the body of a multi-line string, or a non-object input.
pub(super) fn push_value_body(
    lines: &mut Vec<String>,
    depth: usize,
    value: &serde_json::Value,
    context: BlockContext,
) {
    let indent = block_indent(depth, context.width);
    let budget = context.width.saturating_sub(display_width(&indent));
    let text = match value {
        serde_json::Value::String(text) => text.clone(),
        other => scalar_text(other, budget),
    };
    // Capped in the value's own lines, so the marker below counts what a reader would count.
    let source_lines: Vec<&str> = text.lines().collect();
    let shown = source_lines.len().min(context.limits.lines_per_argument);
    // Rows are the other budget, and it has to be shared out here rather than left to the cap
    // downstream. Handing every line the whole argument's budget let twenty lines claim twenty
    // times it; the cap then cut the excess and, with it, this function's own `... N more
    // lines` marker, so the block reported two dropped rows where a thousand had gone. One row
    // is held back for that marker.
    let rows_per_line = context
        .limits
        .rows_per_argument
        .saturating_sub(1)
        .checked_div(shown.max(1))
        .unwrap_or(1)
        .max(1);
    for line in &source_lines[..shown] {
        // `str::lines` splits on `\n` and strips a trailing `\r`, but a lone `\r` mid-line survives
        // it; `sanitize_to_line` is what stops that returning the cursor over the indent. Flattened
        // first either way, so wrapping only ever breaks text meka is choosing to spread over rows
        // and a `\n` from the model never reaches the terminal.
        let flattened = sanitize_to_line(line, usize::MAX);
        if context.limits.wrap {
            for row in wrap_to_width(&flattened, budget, rows_per_line) {
                lines.push(format!("{indent}{row}"));
            }
        } else {
            lines.push(format!(
                "{}{}",
                indent,
                truncate_to_width(&flattened, budget)
            ));
        }
    }
    let elided = source_lines.len() - shown;
    if elided > 0 {
        lines.push(format!(
            "{}{}",
            indent,
            truncate_to_width(
                &format!(
                    "... {} more line{}",
                    elided,
                    if elided == 1 { "" } else { "s" }
                ),
                budget
            )
        ));
    }
}

/// The indent for a block at `depth`, bounded so it can never consume the whole width.
///
/// Nesting is unbounded up to serde_json's parse limit, so an indent proportional to depth reaches
/// the width and leaves a zero budget, at which point keys render as bare `:` and values escape
/// their cap entirely. Past the ceiling the block stops indenting rather than stops informing.
pub(super) fn block_indent(depth: usize, width: usize) -> String {
    let max_depth = (width / 2) / TOOL_PARAM_INDENT.len();
    TOOL_PARAM_INDENT.repeat(depth.min(max_depth.max(1)))
}

/// Fit one of meka's own stand-in words into the budget it was given.
///
/// `(no printable text)` is nineteen columns, so emitting it whatever the budget overflows the line
/// by the width of the word describing the value.
pub(super) fn marker_text(marker: &str, budget: usize) -> String {
    truncate_to_width(marker, budget)
}

/// One-line rendering of a value that needs no block: a scalar, or an empty container.
///
/// An empty string is marked rather than left blank, because `key:` with nothing after it is
/// indistinguishable from a key whose block failed to render.
pub(super) fn scalar_text(value: &serde_json::Value, budget: usize) -> String {
    match value {
        serde_json::Value::String(text) if text.is_empty() => marker_text("(empty)", budget),
        // Whitespace-only is marked as such rather than as empty. A tab passed as a delimiter is an
        // ordinary argument, and reporting it as `(empty)` is not vague, it is wrong: it says the
        // model sent `""` when it did not. Decided before truncation, since a long run of spaces
        // would otherwise come back as a cut-off blank rather than as nothing at all.
        serde_json::Value::String(text) if text.trim().is_empty() => {
            marker_text("(whitespace)", budget)
        }
        serde_json::Value::String(text) => {
            // Trailing whitespace is trimmed because flattening manufactures it: a value ending in
            // a newline would otherwise leave the key line with an invisible tail.
            let line = elide_to_width(&sanitize_to_line(text, usize::MAX), budget)
                .trim_end()
                .to_string();
            // A value made only of characters meka refuses to display (bidi controls, soft hyphens,
            // zero-width joiners) is not empty and is not whitespace, and leaving it blank would
            // read as a rendering fault rather than as the deliberate omission it is.
            if line.is_empty() {
                marker_text("(no printable text)", budget)
            } else {
                line
            }
        }
        serde_json::Value::Null => marker_text("null", budget),
        serde_json::Value::Object(fields) if fields.is_empty() => marker_text("(empty)", budget),
        serde_json::Value::Array(items) if items.is_empty() => marker_text("(empty)", budget),
        // A number or a bool, which cannot carry an escape but can still be long: serde will print
        // every digit of a 100-digit integer.
        other => truncate_to_width(&other.to_string(), budget),
    }
}

/// The argument block for an approval prompt: every argument, wrapped rather than cut.
///
/// Separate entry point from the indicator's so the two sets of limits are named at their call
/// sites rather than passed in from the REPL, which has no business knowing them.
pub(crate) fn render_approval_params(input: &serde_json::Value, width: usize) -> Vec<String> {
    render_tool_params(input, width, BlockLimits::approval())
}

/// Split the indicator into its header line and its argument block, per `params`.
///
/// Separate from the printing so the mapping from setting to output is testable; the two are
/// colored differently, which is why this is a pair rather than one list of lines.
pub(super) fn tool_indicator_parts(
    name: &str,
    input: &serde_json::Value,
    display_summary: Option<&str>,
    params: ToolParams,
    width: usize,
) -> (String, Vec<String>) {
    match params {
        ToolParams::Off => (tool_header(name, width), Vec::new()),
        ToolParams::Summary => (
            tool_indicator_line(name, input, display_summary, width),
            Vec::new(),
        ),
        // No `(arg)` on the header: the primary parameter is in the block two lines down, and
        // showing it twice is the noise this layout exists to avoid.
        ToolParams::Full => (
            tool_header(name, width),
            render_tool_params(input, width, BlockLimits::indicator()),
        ),
    }
}

/// Render the tool indicator on stderr, at the detail `params` asks for.
pub(crate) fn render_tool_indicator(
    name: &str,
    input: &serde_json::Value,
    display_summary: Option<&str>,
    params: ToolParams,
) {
    let (header, block) =
        tool_indicator_parts(name, input, display_summary, params, output_width());
    write_stderr_line(header.with(Color::Cyan));
    for line in block {
        // A different hue from the header rather than a dimmer shade of it. The normal and bright
        // slots of one color (4 and 12, 6 and 14) are the same value in a good many terminal
        // themes, so a header/argument split built on brightness renders as no split at all. Gray
        // would separate them but is the color of a thinking block, which is the neighbor these
        // most need to be told apart from.
        write_stderr_line(line.with(Color::Blue));
    }
}
