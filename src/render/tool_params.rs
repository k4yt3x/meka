//! A call's arguments as an indented block, every line of every argument wrapped to a width, for
//! the approval prompt and the screen's expanded call.

use super::*;

/// Below this many columns for the argument there is nothing worth showing, so the screen's call
/// row drops the summary instead of printing an ellipsis.
pub(super) const TOOL_ARGUMENT_FLOOR: usize = 8;

/// One level of nesting in an argument block.
pub(super) const TOOL_PARAM_INDENT: &str = "  ";

/// Columns a value keeps however long its key is: the key is the label and the value is the
/// substance.
pub(super) const TOOL_VALUE_MIN_WIDTH: usize = 16;

/// Append one `key: value` pair at `depth`, recursing for containers.
pub(super) fn push_param(
    lines: &mut Vec<String>,
    depth: usize,
    key: &str,
    value: &serde_json::Value,
    width: usize,
) {
    let indent = block_indent(depth, width);
    let available = width.saturating_sub(display_width(&indent));
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
                push_param(lines, depth + 1, nested_key, nested, width);
            }
        }
        serde_json::Value::Array(items) if !items.is_empty() => {
            lines.push(format!("{indent}{key}:"));
            for item in items {
                push_item(lines, depth + 1, item, width);
            }
        }
        serde_json::Value::String(text) if is_multi_line(text) => {
            lines.push(format!("{indent}{key}:"));
            push_value_body(lines, depth + 1, value, width);
        }
        _ => {
            let value_budget = available.saturating_sub(display_width(&key) + ": ".len());
            // A value too wide for the key line gets a block of its own rather than being cut on
            // it: the commonest approval of all, a long `shell_execute` pipeline, is one line and
            // would otherwise have its tail hidden.
            if display_width(&scalar_text(value, usize::MAX)) > value_budget {
                lines.push(format!("{indent}{key}:"));
                push_value_body(lines, depth + 1, value, width);
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
    width: usize,
) {
    // Bounded like every other indent in the block. An array nests through this function rather
    // than through `push_param`, so leaving it proportional to depth put a bullet at column 40
    // of a 40-column line and pushed everything under it past the edge.
    let indent = block_indent(depth, width);
    match item {
        serde_json::Value::Object(fields) if !fields.is_empty() => {
            let nested_indent = block_indent(depth + 1, width);
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
                push_param(lines, depth + 1, key, value, width);
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
                push_item(lines, depth + 1, value, width);
            }
        }
        // Mirrors `push_param`'s multi-line arm. Without it a bulleted string kept its newlines and
        // put every line after the first at column 0, which is both wrong to read and enough to
        // forge a row of meka's own outside the block.
        serde_json::Value::String(text) if is_multi_line(text) => {
            lines.push(format!("{indent}-"));
            push_value_body(lines, depth + 1, item, width);
        }
        _ => {
            let budget = width.saturating_sub(display_width(&indent) + "- ".len());
            // Same promotion `push_param` makes: a value too wide for its own row gets a block
            // rather than losing its tail. An MCP tool taking `["bash", "-lc", "<long
            // command>"]` is the shape this exists for, and approvals gate MCP tools.
            if display_width(&scalar_text(item, usize::MAX)) > budget {
                lines.push(format!("{indent}-"));
                push_value_body(lines, depth + 1, item, width);
            } else {
                lines.push(format!("{}- {}", indent, scalar_text(item, budget)));
            }
        }
    }
}

/// A value as an indented block of its own, every line wrapped to the width.
pub(super) fn push_value_body(
    lines: &mut Vec<String>,
    depth: usize,
    value: &serde_json::Value,
    width: usize,
) {
    let indent = block_indent(depth, width);
    let budget = width.saturating_sub(display_width(&indent));
    let text = match value {
        serde_json::Value::String(text) => text.clone(),
        other => scalar_text(other, budget),
    };
    for line in text.lines() {
        // `str::lines` splits on `\n` and strips a trailing `\r`, but a lone `\r` mid-line survives
        // it; `sanitize_to_line` is what stops that returning the cursor over the indent. Flattened
        // first, so wrapping only ever breaks text meka is choosing to spread over rows and a `\n`
        // from the model never reaches the terminal.
        let flattened = sanitize_to_line(line, usize::MAX);
        for row in wrap_to_width(&flattened, budget, usize::MAX) {
            lines.push(format!("{indent}{row}"));
        }
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

/// A call's arguments as an indented block: one row per scalar argument, a bare `key:` over an
/// indented block for a multi-line or over-wide value, and nested objects and arrays by
/// indentation. Every row stays within `width` and nothing is cut, because the surfaces that draw
/// one, the approval prompt and the screen's expanded call, are where a decision needs the whole of
/// what is being written.
pub(crate) fn render_whole_params(input: &serde_json::Value, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    match input {
        serde_json::Value::Object(fields) => {
            for (key, value) in fields {
                push_param(&mut lines, 1, key, value, width);
            }
        }
        // A tool whose input is not an object at all. Nothing sensible to key it by, so it renders
        // as a bare value rather than being dropped, which would read as "no arguments".
        serde_json::Value::Array(items) => {
            for item in items {
                push_item(&mut lines, 1, item, width);
            }
        }
        other => push_value_body(&mut lines, 1, other, width),
    }
    lines
}
