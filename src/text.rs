//! Text helpers with no terminal behind them: widths, truncation, sanitising, columns and the
//! id-prefix rules. Everything here is pure, so any layer may use it; what knows the terminal's
//! width or writes to a stream lives in `render`.

use std::sync::LazyLock;

use chrono::{DateTime, Local, Utc};
use regex::Regex;

/// Strip control + format characters that could hijack the terminal or be
/// used as homograph-style attacks on users reviewing tool output:
///
/// - Unicode category **Cc** (C0/C1 controls) except `\n` and `\t`.
/// - Unicode category **Cf** (formatters: RTL/LTR overrides, zero-width joiners, byte-order marks,
///   language tags).
/// - Unpaired surrogate code units (already impossible in a valid `&str`, noted for completeness).
///
/// Emoji, CJK, combining marks, and all other printable Unicode pass through unchanged.
pub(crate) fn sanitize_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if is_safe_char(ch) {
            out.push(ch);
        }
    }
    out
}
fn is_safe_char(ch: char) -> bool {
    // Whitelist the two whitespace controls we care about. `\r` is deliberately not among them: it
    // returns the cursor to column zero without advancing a line, which is enough to overwrite a
    // line meka has already printed using no escape sequence at all. A server progress message of
    // `"\r[approval] Shell\n  command: ls\nAllow? (Y/n) "` would otherwise repaint a convincing
    // approval block at column zero, and everything that renders server text -- the elicitation
    // banner, the form labels, the progress line -- trusts this function to have made it
    // terminal-safe.
    if ch == '\n' || ch == '\t' {
        return true;
    }
    let code = ch as u32;

    // C0 controls (U+0000–U+001F) and DEL (U+007F).
    if code < 0x20 || code == 0x7F {
        return false;
    }

    // C1 controls (U+0080–U+009F).
    if (0x80..=0x9F).contains(&code) {
        return false;
    }

    // Cf category (Format): covers RTL/LTR overrides, ZWJ/ZWNJ, BOM, interlinear annotations, and
    // language tags (E0000–E007F).
    if is_format_char(code) {
        return false;
    }

    true
}
/// Returns true for the bidirectional formatting characters that can reorder a line on screen.
///
/// The embedding, override and isolate initiators plus their terminators -- the "trojan source"
/// set. These are the ones that let text render in an order the bytes do not have, which is what
/// makes a path or a command read as something other than what it is.
///
/// Deliberately narrower than [`is_format_char`]. The rest of `Cf` carries meaning that a model may
/// legitimately emit: ZWJ joins emoji into families and professions, ZWNJ is required to spell
/// ordinary Persian and Arabic words, and both drive Indic conjunct forms. Filtering the whole
/// category off assistant output corrupts that text for no security gain, since none of it can
/// reorder anything. `LRM`/`RLM` are likewise left in: they mark direction rather than override it,
/// and are ordinary content in Hebrew and Arabic.
pub(crate) fn is_bidi_control(code: u32) -> bool {
    matches!(
        code,
        0x202A..=0x202E   // LRE, RLE, PDF, LRO, RLO
        | 0x2066..=0x2069 // LRI, RLI, FSI, PDI
    )
}
/// Returns true for Unicode General Category `Cf` (Format).
///
/// Enumerated from Unicode 15.1: only the ranges that exist; the bulk of the BMP has no Cf
/// characters so this stays a short list.
pub(crate) fn is_format_char(code: u32) -> bool {
    matches!(
        code,
        0x00AD                   // SOFT HYPHEN
        | 0x0600..=0x0605        // Arabic number signs
        | 0x061C                 // ARABIC LETTER MARK
        | 0x06DD                 // ARABIC END OF AYAH
        | 0x070F                 // SYRIAC ABBREVIATION MARK
        | 0x0890..=0x0891        // Arabic POUND/PIASTRE
        | 0x08E2                 // ARABIC DISPUTED END OF AYAH
        | 0x180E                 // MONGOLIAN VOWEL SEPARATOR
        | 0x200B..=0x200F        // ZWSP, ZWNJ, ZWJ, LRM, RLM
        | 0x202A..=0x202E        // LRE, RLE, PDF, LRO, RLO
        | 0x2060..=0x2064        // WJ + invisible operators
        | 0x2066..=0x2069        // LRI, RLI, FSI, PDI
        | 0xFEFF                 // BOM / ZWNBSP
        | 0xFFF9..=0xFFFB        // Interlinear annotation anchors
        | 0x110BD                // KAITHI NUMBER SIGN
        | 0x110CD                // KAITHI NUMBER SIGN ABOVE
        | 0x13430..=0x13438      // Egyptian hieroglyph format controls
        | 0x1BCA0..=0x1BCA3      // Shorthand format controls
        | 0x1D173..=0x1D17A      // Musical symbol format controls
        | 0xE0001                // LANGUAGE TAG
        | 0xE0020..=0xE007F      // TAG characters
    )
}

pub(crate) fn display_width(string: &str) -> usize {
    // The larger of two measures, because a terminal may follow either and a budget must never be
    // built on the smaller one. `unicode_width` merges an emoji and its skin-tone modifier into one
    // two-column cluster; VTE -- gnome-terminal, Console, Tilix, Terminator -- paints them as two
    // glyphs across four columns, so every skin-toned emoji in an argument was a two-times
    // under-count. Summing per character catches that, and the whole-string measure catches
    // sequences a sum would under-count instead. Taking the maximum shows less than might have fit,
    // which is the direction to be wrong in.
    unicode_width::UnicodeWidthStr::width(string).max(string.chars().map(char_width).sum())
}
/// Columns one character occupies, counting anything `unicode_width` will not score as zero.
///
/// `None` comes back for the control characters, which [`sanitize_to_line`] has already removed by
/// the time any budget is computed.
pub(crate) fn char_width(character: char) -> usize {
    unicode_width::UnicodeWidthChar::width(character).unwrap_or(0)
}
/// Columns a tab advances to. Four rather than eight because these lines already carry a block
/// indent, and eight pushes nested code past the width budget for no extra clarity.
pub(crate) const TAB_WIDTH: usize = 4;
/// Make a model-supplied string safe to place on one line of meka's own UI.
///
/// [`sanitize_for_display`] drops escapes and control characters but deliberately keeps `\n`, `\r`
/// and `\t`, which is right for text meant to span lines and wrong everywhere a string is being
/// slotted into a line meka composed. A kept `\n` walks out of an indented block and lands
/// attacker-chosen text at column 0; a kept `\r` returns the cursor and overwrites the label that
/// was supposed to introduce the value. Both forge meka's chrome without needing an escape
/// sequence, so every such site flattens them to spaces and caps the result.
///
/// A tab is expanded rather than flattened. It cannot move the cursor left or up, so it forges
/// nothing, and collapsing it to one space destroys the indentation of every tab-indented file the
/// block exists to let you read. Expanding also makes the width cap honest, since a tab otherwise
/// hides several columns behind a single character.
///
/// The cap is in terminal columns, not characters: a line of CJK or emoji is twice as wide as its
/// character count suggests, and a cap that misses that lets a "capped" line wrap into rows.
pub(crate) fn sanitize_to_line(text: &str, max_columns: usize) -> String {
    // Every character meka cannot measure is dropped, which `char::is_control` does not cover.
    // The rule is one line below -- a character worth zero columns does not survive -- and
    // it is deliberately wider than the classes that motivated it:
    //
    // `unicode_width` scores U+00AD SOFT HYPHEN and U+3164 HANGUL FILLER as zero columns
    // while a terminal following `wcwidth` draws one and two. A run of either passes any
    // column budget unmeasured, which is how a model pushes its own text onto a row meka believes
    // is empty -- and the filler draws blank, so the overrun is invisible padding.
    //
    // A variation selector (U+FE00-FE0F) changes the width of the character *before* it, so a
    // budget measured before it is applied is wrong afterwards.
    //
    // This class also holds the bidi overrides, where the argument the user reads is not the
    // argument that runs.
    //
    // Dropping by measured width rather than by category costs the combining marks: a
    // decomposed `e` + U+0301 renders as `e`. Precomposed text, which is what NFC and almost
    // every source of these strings produces, is untouched. That is the same trade the ZWJ case
    // already makes, and it buys the property every budget here rests on -- that each surviving
    // character advances the count by at least one, so a cut is always reached.
    let flattened: String = sanitize_for_display(text)
        .chars()
        .flat_map(|character| match character {
            '\t' => std::iter::repeat_n(' ', TAB_WIDTH),
            character if character.is_whitespace() => std::iter::repeat_n(' ', 1),
            character => std::iter::repeat_n(character, 1),
        })
        // Applied after the whitespace above becomes spaces, so a newline still separates the words
        // it separated rather than being dropped as the zero-width character it measures as.
        .filter(|character| char_width(*character) > 0)
        .collect();
    truncate_to_width(&flattened, max_columns)
}
/// Marks a cut made by [`truncate_to_width`].
pub(crate) const TRUNCATION_MARKER: &str = "...";
/// Cut `text` to `max_columns` terminal columns, marking the cut.
///
/// Measured with [`display_width`] rather than `chars().count()`, because a "200 character"
/// argument of full-width characters occupies 400 columns and wraps into rows the cap exists to
/// prevent.
///
/// The marker is inside the budget, not added on top of it. Callers compose a line out of several
/// truncated parts against one total width, so a function that can return `max_columns + 3` makes
/// that total unenforceable. Below the marker's own width there is no room to say a cut happened,
/// so the text is simply cut.
pub(crate) fn truncate_to_width(text: &str, max_columns: usize) -> String {
    if display_width(text) <= max_columns {
        return text.to_string();
    }
    // A cut always says so, even when saying so is all there is room for. Emitting the text alone
    // when the budget cannot fit a marker produced a string that reads as complete: at 37 columns
    // `mcp__exa__web_search_exa` came out as `mc`, which is not a shortened name, it is a different
    // name.
    let marker = &TRUNCATION_MARKER[..TRUNCATION_MARKER.len().min(max_columns)];
    let budget = max_columns - display_width(marker);
    let mut kept = take_columns(text, budget);
    kept.push_str(marker);
    kept
}
/// The longest prefix of `text` that fits in `max_columns`.
///
/// Measured by re-measuring the whole prefix rather than by summing per-character widths, because
/// the two are not the same number and the callers gate on the former. `unicode_width` scores
/// `"1\u{fe0f}"` as two columns as a string and one as a sum, so a per-character fill packed twice
/// what the gate believed fit and every budget in the file came out at double. Re-measuring is
/// quadratic in the budget, which is bounded and small; being wrong is not.
pub(crate) fn take_columns(text: &str, max_columns: usize) -> String {
    // Only the open cluster is re-measured. Re-measuring the whole prefix per character was
    // quadratic in the prefix, and zero-width characters lengthen the prefix without spending the
    // budget, so a line padded with them spun the renderer for minutes. Everything before the
    // cluster a new character can still join has settled: a base character closes the cluster
    // before it, and a cluster longer than the window is cut, because nothing renders a hundred
    // combining marks on one base anyway.
    const CLUSTER_WINDOW: usize = 32;
    let mut kept = String::new();
    let mut settled_columns = 0usize;
    let mut cluster_start = 0usize;
    let mut cluster_characters = 0usize;
    for character in text.chars() {
        let opens_cluster = char_width(character) > 0
            || breaks_cluster(character)
            || cluster_characters >= CLUSTER_WINDOW;
        if opens_cluster && cluster_characters > 0 {
            settled_columns += display_width(&kept[cluster_start..]);
            cluster_start = kept.len();
            cluster_characters = 0;
        }
        kept.push(character);
        cluster_characters += 1;
        if settled_columns + display_width(&kept[cluster_start..]) > max_columns {
            kept.pop();
            break;
        }
    }
    kept
}
/// A zero-width character that starts its own cluster rather than attaching to the one before it:
/// the format characters that separate rather than join. A variation selector, a joiner or a
/// combining mark can change the width of what it follows; these cannot.
pub(crate) fn breaks_cluster(character: char) -> bool {
    matches!(
        character,
        '\u{200B}' | '\u{2060}' | '\u{FEFF}' | '\u{180E}' | '\u{200E}' | '\u{200F}'
    ) || ('\u{202A}'..='\u{202E}').contains(&character)
        || ('\u{2066}'..='\u{2069}').contains(&character)
}
/// Cut `text` to `max_columns`, keeping both ends.
///
/// For an *identifier*, where both ends carry meaning and the middle is filler. A tool name is
/// back-loaded: `mcp__exa__web_search_exa` and `mcp__exa__web_fetch_exa` agree for fifteen
/// characters and differ only at the end, so a tail cut throws away exactly what says which tool
/// ran. A path behaves the same way, and it is the commoner case:
/// `/home/you/projects/meka/docs/book/src/configuration/config-file.md` cut from the tail keeps
/// six directories and loses the filename, which is the part you were reading it for.
///
/// Use [`truncate_to_width`] for a *line of content* instead -- a line of source, a wrapped body --
/// where the text runs left to right and a hole in the middle would misrepresent it.
pub(crate) fn elide_to_width(text: &str, max_columns: usize) -> String {
    if display_width(text) <= max_columns {
        return text.to_string();
    }
    let marker_width = display_width(TRUNCATION_MARKER);
    // Too narrow to show both ends and say so; a tail cut at least stays readable.
    if max_columns <= marker_width + 2 {
        return truncate_to_width(text, max_columns);
    }
    let available = max_columns - marker_width;
    // The tail gets the larger half when the split is odd: it carries the operation in a tool name
    // and the filename in a path.
    let head_width = available / 2;
    let tail_width = available - head_width;
    format!(
        "{}{}{}",
        take_columns(text, head_width),
        TRUNCATION_MARKER,
        tail_columns(text, tail_width)
    )
}
/// The longest suffix of `text` that fits in `max_columns`, the mirror of [`take_columns`].
///
/// Measures the real suffix rather than reversing the string and taking a prefix, because width is
/// **not** order-independent and the reversed measurement is not the one that gets printed:
/// `display_width("\u{1F44D}\u{1F3FB}")` is 2 and `display_width("\u{1F3FB}\u{1F44D}")` is 4, so a
/// tail of skin-toned emoji measured backwards came back a third under its budget and the composed
/// line ran 100 columns wide where 80 was asked for.
///
/// [`display_width`] taking the larger of two measures also closes that case, since a per-character
/// sum does not care about order. This does not lean on it: measuring what is printed is correct
/// whatever the measure does next.
pub(crate) fn tail_columns(text: &str, max_columns: usize) -> String {
    let mut kept = "";
    for (index, _) in text.char_indices().rev() {
        let candidate = &text[index..];
        if display_width(candidate) > max_columns {
            break;
        }
        kept = candidate;
    }
    kept.to_string()
}
/// Break `text` into at most `max_rows` rows of at most `max_columns`, preferring a space.
///
/// For the approval prompt, where cutting a line hides the tail of what is being authorized. The
/// caller prefixes every row with the block indent, so no row begins at column zero even though the
/// value now spans several.
///
/// **When it does not fit, the last two rows are a count and the END of the text**, not wherever
/// the budget ran out. This is [`elide_to_width`]'s reasoning one dimension up. Wrapping was chosen
/// over cutting so the tail of a command could not be hidden from the line being approved, and a
/// wrap that shows the first `max_rows` rows and stops hides exactly that: a 90 KB
/// `execute_command` filled every row it was given and left `; rm -rf /important` off the end of
/// the last one. The notification surface, which elides from the middle, showed that tail; the
/// decision surface did not.
pub(crate) fn wrap_to_width(text: &str, max_columns: usize, max_rows: usize) -> Vec<String> {
    // A zero budget can show nothing. Returning the text would be worse than showing none of it:
    // the caller has already spent the width on indent, and an unbounded row of model output is the
    // one thing the budget exists to prevent.
    if max_columns == 0 || max_rows == 0 {
        return Vec::new();
    }
    if display_width(text) <= max_columns {
        return vec![text.to_string()];
    }
    // Continuation rows keep the line's own leading whitespace, so wrapped code still reads at the
    // depth it was written at instead of appearing to dedent.
    let hanging = &text[..text.len() - text.trim_start_matches(' ').len()];
    let hanging = take_columns(hanging, max_columns / 2);
    // Two rows held back for the count and the end. Below three rows there is no room for that
    // shape, so one row is held back and cut where it lands.
    let keeps_the_end = max_rows >= 3;
    let head_rows = if keeps_the_end {
        max_rows - 2
    } else {
        max_rows - 1
    };
    let mut rows: Vec<String> = Vec::new();
    let mut rest = text;
    while rows.len() < head_rows {
        let prefix = if rows.is_empty() {
            ""
        } else {
            hanging.as_str()
        };
        let budget = max_columns.saturating_sub(display_width(prefix));
        if budget == 0 || display_width(rest) <= budget {
            break;
        }
        let head = take_columns(rest, budget);
        if head.is_empty() {
            // One character is wider than the whole budget, so no row can hold it. Taking it anyway
            // was the way out of the loop and it overflowed the width by that character; falling
            // through to the truncation below emits a marker, which fits any budget at all.
            break;
        }
        // Break at the last space that fits, but never inside a leading run of them: breaking there
        // emits a row that is empty once trimmed and silently drops the line's indentation.
        let split = match head.rfind(' ') {
            Some(index) if !head[..index].trim().is_empty() => index,
            _ => head.len(),
        };
        rows.push(format!("{}{}", prefix, rest[..split].trim_end()));
        rest = rest[split..].trim_start_matches(' ');
        if rest.is_empty() {
            return rows;
        }
    }
    let prefix = if rows.is_empty() {
        ""
    } else {
        hanging.as_str()
    };
    let budget = max_columns.saturating_sub(display_width(prefix));
    if keeps_the_end && budget > 0 && display_width(rest) > budget {
        let tail = tail_columns(rest, budget);
        let dropped = rest.chars().count() - tail.chars().count();
        rows.push(format!(
            "{}{}",
            prefix,
            truncate_to_width(&format!("... {dropped} more characters ..."), budget)
        ));
        rows.push(format!("{prefix}{tail}"));
        return rows;
    }
    rows.push(format!("{}{}", prefix, truncate_to_width(rest, budget)));
    rows
}
/// Match ANSI CSI (Control Sequence Introducer) escapes: `ESC [` followed by parameter bytes
/// (`0x30-0x3F`), optional intermediate bytes (`0x20-0x2F`), and a final byte (`0x40-0x7E`). This
/// covers the sequences an attacker would use to clear the screen, move the cursor, or alter
/// colors.
#[allow(
    clippy::expect_used,
    reason = "the pattern is a literal, so a failure is a typo caught on first build"
)]
pub(crate) static CSI_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]").expect("static CSI pattern")
});
/// Strip ANSI CSI escapes and C0 control characters (except `\n`, `\r`, `\t`) from a string
/// destined for the user's terminal. Intended for text that originates in untrusted sources (LLM
/// tool arguments, command output echoed into indicators/prompts, etc.) so a hostile or broken
/// string cannot forge UI chrome or corrupt terminal state.
///
/// The sanitized form is for **display only**. The conversation copy sent back to the LLM keeps
/// full fidelity.
pub(crate) fn sanitize_for_display(text: &str) -> String {
    let stripped = CSI_PATTERN.replace_all(text, "");
    stripped
        .chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
        .collect()
}
/// Same as [`sanitize_for_display`], but also drops `\r`. For multi-line prose that will be
/// rendered as markdown: streamed assistant text.
///
/// `\r` is excluded because it is the forgery primitive that needs no escape sequence at all. It
/// returns the cursor to column zero without advancing a line, so a model that has read attacker
/// text can overwrite a line meka already printed -- including the tail of an approval prompt --
/// using nothing but ordinary characters. `\n` and `\t` stay: they are structural in markdown and
/// can only move the cursor forward.
///
/// Applying this per delta is sound even though a CSI sequence can straddle a chunk boundary,
/// because the `is_control` filter removes every `\x1b` regardless of what follows it. With no
/// `ESC` reaching the terminal no escape sequence can form, whatever the chunking. The regex is
/// there to remove a *whole* sequence cleanly rather than leaving `[2J` visible in the prose.
///
/// Bidi controls go too, because a bidi override reorders a rendered line without changing a byte
/// of it: an assistant that has read attacker text could make a path or a command read as something
/// else entirely. `char` boundaries are safe per delta because these are single scalars.
///
/// Only the bidi set, unlike [`sanitize_to_line`] and `mcp::sanitize::sanitize_text`, which drop
/// the whole `Cf` category. Those two render *server*-controlled strings into one row of meka's own
/// chrome, where nothing in `Cf` has a legitimate use. This is prose the model wrote for the user,
/// and most of `Cf` is ordinary content there: ZWJ builds emoji families and profession sequences,
/// ZWNJ spells ordinary Persian and Arabic words, and both drive Indic conjuncts. Stripping the
/// category here mangled all of it, and bought nothing, since none of those can reorder a line.
pub(crate) fn sanitize_stream_text(text: &str) -> String {
    let stripped = CSI_PATTERN.replace_all(text, "");
    stripped
        .chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .filter(|c| !crate::text::is_bidi_control(*c as u32))
        .collect()
}
/// Format `rows` into a left-aligned, space-padded column layout, the shared renderer for meka's
/// CLI list tables (`skill list`, `mcp list`, `list`, `scratchpad_list`).
///
/// Each column is widened to its longest cell, the matching header included. Columns are separated
/// by two spaces; the final column is left unpadded so a long trailing value (a path, a URL, a
/// preview) doesn't drag a run of trailing whitespace. The returned string has one trailing newline
/// per line and no extra blank line; the caller picks the stream (`print!` for stdout list
/// commands, or embed it in a tool result).
///
/// (Distinct from the private `format_table`, which lays out *markdown* pipe tables for the
/// streaming renderer.)
///
/// Width is measured in terminal columns, the same unit [`sanitize_to_line`] truncates in and the
/// same unit every caller reserves its budget in. Measuring in `char`s instead let a cell whose
/// characters are two columns wide -- a CJK provider name, an MCP tool name a server chose -- pad
/// to less than it renders, shifting every column after it on that row and pushing the row past
/// the budget its cells individually respected.
pub(crate) fn format_columns(headers: &[&str], rows: &[Vec<String>]) -> String {
    if headers.is_empty() {
        return String::new();
    }

    let mut widths: Vec<usize> = headers.iter().map(|header| display_width(header)).collect();
    for row in rows {
        for (index, cell) in row.iter().take(widths.len()).enumerate() {
            widths[index] = widths[index].max(display_width(cell));
        }
    }

    let mut out = format_columns_row(headers, &widths);
    for row in rows {
        let cells: Vec<&str> = row.iter().map(String::as_str).collect();
        out.push_str(&format_columns_row(&cells, &widths));
    }
    out
}
/// How much of an id a row shows before anything forces it wider: a UUID's first segment.
pub(crate) const ID_PREFIX: usize = 8;
/// Whether `prefix` could be a prefix of an id meka printed.
///
/// Every resolver asks this before matching, because `id.starts_with("")` is true of every id: an
/// unset shell variable in `meka schedule cancel "$JOB"` otherwise reads as "the only job", and
/// resolves cleanly right up until the store holds two. The stores are the doors this covers --
/// sessions, scheduled jobs and background tasks all key on a UUID string.
///
/// Rejecting rather than matching everything is also what makes an ambiguity report honest: an
/// empty prefix names nothing the caller could retype a longer version of.
pub(crate) fn is_usable_id_prefix(prefix: &str) -> bool {
    !prefix.is_empty()
        && prefix
            .chars()
            .all(|character| character.is_ascii_hexdigit() || character == '-')
}
/// An id prefix as the resolvers compare it.
///
/// Every id meka stores and prints is a lowercase UUID, but the resolvers disagreed about case:
/// sessions go through SQL `LIKE`, which is ASCII-case-insensitive, while jobs and tasks use
/// `str::starts_with`. So a pasted `4D71EECA` resolved a session and was reported as no such job --
/// a clean answer from one command and a false miss from its sibling. Folded once, here.
pub(crate) fn id_prefix_for_matching(prefix: &str) -> String {
    prefix.to_ascii_lowercase()
}
/// Shortest prefix at which every one of `ids` is distinct, never below [`ID_PREFIX`] unless an
/// id is itself shorter than that.
///
/// A prefix is what the reader retypes into `schedule show`, `schedule cancel` or `--session`, all
/// of which refuse an ambiguous one. Printing a prefix that cannot be used is the failure worth
/// avoiding, and a full UUID in both id columns spends 76 of the 120 available to say what eight
/// characters usually say.
///
/// Uniqueness is over the rows being rendered. For `meka schedule list` that is every job there is;
/// a filtered listing can still print a prefix that a wider set makes ambiguous, which those
/// commands report rather than act on.
///
/// Never ends on a UUID's hyphen, since `4d71eeca-` reads as a truncation of nothing.
/// [`unique_prefix_len`] where the ids on screen are a subset of the ids that must be resolved.
///
/// A listing filters: `meka session list` hides sub-agent sessions and honors `-n`, and
/// `meka schedule list --session` narrows to one conversation. The resolvers do not filter -- they
/// scan the whole store. Sizing the column to the rows alone therefore printed a prefix that the
/// `show` beside it refused as ambiguous, which is the one thing the id rule promises cannot
/// happen. `universe` is what the resolver will search, so the width is the width that resolves.
pub(crate) fn unique_prefix_len_within<'a>(
    shown: impl Iterator<Item = &'a str> + Clone,
    universe: impl Iterator<Item = &'a str> + Clone,
) -> usize {
    let universe: std::collections::HashSet<&str> = universe.collect();
    let longest = shown.clone().map(str::len).max().unwrap_or(ID_PREFIX);
    for length in ID_PREFIX..longest {
        if shown
            .clone()
            .any(|id| id.as_bytes().get(length - 1) == Some(&b'-'))
        {
            continue;
        }
        let resolves = shown.clone().all(|id| {
            let prefix = id.get(..length).unwrap_or(id);
            universe
                .iter()
                .filter(|other| other.starts_with(prefix))
                .count()
                == 1
        });
        if resolves {
            return length;
        }
    }
    longest
}
pub(crate) fn unique_prefix_len<'a>(ids: impl Iterator<Item = &'a str> + Clone) -> usize {
    // The rows are their own universe, and it is deduplicated by `unique_prefix_len_within`:
    // repeating an id is ordinary -- a session with several scheduled jobs fills the whole Session
    // column with itself -- and counting rows made that unsatisfiable, so the column widened to a
    // full UUID to distinguish an id from itself.
    unique_prefix_len_within(ids.clone(), ids)
}
pub(crate) fn format_columns_row(cells: &[&str], widths: &[usize]) -> String {
    let mut line = String::new();
    let last = cells.len().saturating_sub(1);
    for (index, cell) in cells.iter().enumerate() {
        if index == last {
            // Final column: never padded; nothing follows it.
            line.push_str(cell);
        } else {
            // Padded by hand rather than with `{:<w$}`, which counts `char`s: the widths above are
            // terminal columns, and the two disagree on exactly the cells that motivated them.
            let width = widths.get(index).copied().unwrap_or(0);
            line.push_str(cell);
            for _ in 0..width.saturating_sub(display_width(cell)) {
                line.push(' ');
            }
            line.push_str("  ");
        }
    }
    line.push('\n');
    line
}
/// `label:` lines with every value starting in one column, for the `show` and `get` commands.
///
/// The column is the widest `label:` plus two, measured over the fields that carry a value. A field
/// whose value is empty is a heading over the indented block its caller appends next (`prompt:`,
/// `result:`) and is written bare, so no row ends in a run of spaces; it takes no part in the width
/// either, because nothing lines up against it.
pub(crate) fn format_fields(fields: &[(&str, String)]) -> String {
    let width = fields
        .iter()
        .filter(|(_, value)| !value.is_empty())
        .map(|(label, _)| display_width(label) + 1)
        .max()
        .unwrap_or(0);
    let mut out = String::new();
    for (label, value) in fields {
        out.push_str(label);
        out.push(':');
        if !value.is_empty() {
            for _ in 0..(width + 2).saturating_sub(display_width(label) + 1) {
                out.push(' ');
            }
            out.push_str(value);
        }
        out.push('\n');
    }
    out
}
pub(crate) fn format_token_count(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", (n as f64) / 1_000.0)
    } else {
        format!("{:.1}M", (n as f64) / 1_000_000.0)
    }
}
/// Format a non-negative duration in seconds compactly, e.g. `2d 3h`, `4h 12m`, `45m`, `30s`. Used
/// by `meka account whoami` for the token time-to-expiry.
pub(crate) fn format_duration_short(seconds: i64) -> String {
    let seconds = seconds.max(0);
    let minutes = seconds / 60;
    if minutes >= 24 * 60 {
        format!("{}d {}h", minutes / (24 * 60), (minutes % (24 * 60)) / 60)
    } else if minutes >= 60 {
        format!("{}h {}m", minutes / 60, minutes % 60)
    } else if minutes >= 1 {
        format!("{minutes}m")
    } else {
        format!("{seconds}s")
    }
}

/// Binary units, because every size meka caps or reports is a memory or body size: a provider's
/// request limit, a decode allocation, a read ceiling, all stated in MiB by the systems that impose
/// them.
pub(crate) const KIB: usize = 1 << 10;
pub(crate) const MIB: usize = 1 << 20;

/// A byte count in the largest binary unit that keeps it above one: `2.5 MiB`, `640.0 KiB`, `16 B`.
///
/// One decimal rather than integer division, which truncates: every body from 2.0 to just under 3.0
/// MiB once reported as "2 MiB", and the figure is what a user quotes when asking why a request was
/// refused. Every size meka prints comes through here, so a ceiling and the body measured against
/// it are stated in the same unit to the same precision.
pub(crate) fn format_size(bytes: usize) -> String {
    if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// How much of an instant a surface shows.
///
/// Every precision renders in the reader's local time with the numeric UTC offset. Local, because a
/// clock with no zone reads as the reader's own and every instant meka stores is UTC; the offset,
/// because the same store is read from more than one machine and a rendering pasted into a report
/// has to say which clock it was on. chrono prints an offset for `%Z` too, never a zone name, so
/// the offset is spelled as the offset it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Precision {
    /// `2026-09-02 22:57:28 +02:00`: a record of when something happened.
    Seconds,
    /// `2026-09-02 22:57 +02:00`: a listing column, or an appointment.
    Minutes,
    /// `Wed 2026-09-02 22:57 +02:00`: an appointment the reader checks against a calendar.
    WeekdayMinutes,
}

impl Precision {
    const fn pattern(self) -> &'static str {
        match self {
            Self::Seconds => "%Y-%m-%d %H:%M:%S %:z",
            Self::Minutes => "%Y-%m-%d %H:%M %:z",
            Self::WeekdayMinutes => "%a %Y-%m-%d %H:%M %:z",
        }
    }

    /// Columns a rendering occupies, for a table that budgets a column for one. Fixed per
    /// precision: the numeric offset is what keeps it so, where a zone name would not be.
    pub(crate) const fn width(self) -> usize {
        match self {
            Self::Seconds => "2026-09-02 22:57:28 +02:00".len(),
            Self::Minutes => "2026-09-02 22:57 +02:00".len(),
            Self::WeekdayMinutes => "Wed 2026-09-02 22:57 +02:00".len(),
        }
    }
}

/// An instant as every terminal surface shows one. The wire (HTTP views, exports, the store) keeps
/// RFC 3339; this is for a human reading a screen.
pub(crate) fn format_timestamp(at: DateTime<Utc>, precision: Precision) -> String {
    at.with_timezone(&Local)
        .format(precision.pattern())
        .to_string()
}

/// The one sentence for a name that matches nothing: `no <noun> named '<given>' (configured: a,
/// b)`, or `(none configured)` when there is nothing to list. Every door that refuses by name
/// renders it here, so a profile and an account are refused in the same words on every host.
pub(crate) fn unknown_name(
    noun: &str,
    given: &str,
    known: impl IntoIterator<Item = impl AsRef<str>>,
) -> String {
    let known: Vec<String> = known
        .into_iter()
        .map(|name| name.as_ref().to_string())
        .collect();
    if known.is_empty() {
        format!("no {noun} named '{given}' (none configured)")
    } else {
        format!(
            "no {noun} named '{given}' (configured: {})",
            known.join(", ")
        )
    }
}

/// The `Authorization` value for a bearer token (RFC 6750): one spelling for every backend and
/// every stored credential that sends one, whether the token is an API key or an OAuth access
/// token.
pub(crate) fn bearer(token: &str) -> String {
    format!("Bearer {token}")
}

#[cfg(test)]
mod tests {
    use super::{
        Precision, elide_to_width, format_fields, format_size, format_timestamp,
        id_prefix_for_matching, take_columns, unknown_name, wrap_to_width,
    };

    /// Every value starts two columns past the widest `label:`, a heading is written bare and does
    /// not widen the column, and a label is measured in terminal columns rather than bytes.
    #[test]
    fn fields_line_up_on_the_widest_label_and_a_heading_stays_bare() {
        let rendered = format_fields(&[
            ("id", "7f3a".to_string()),
            ("permission", "read".to_string()),
            ("a heading that is longer", String::new()),
            ("日本", "wide".to_string()),
        ]);
        assert_eq!(
            rendered,
            "id:          7f3a\npermission:  read\na heading that is longer:\n日本:        wide\n"
        );
        assert_eq!(format_fields(&[]), "");
    }

    /// The unit ladder and the one decimal. `3_145_727` is the case integer division got wrong:
    /// two and a half megabytes is not "2 MiB", and a body just under three is not "2" either.
    #[test]
    fn a_size_is_stated_in_binary_units_to_one_decimal() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1_024), "1.0 KiB");
        assert_eq!(format_size(655_360), "640.0 KiB");
        assert_eq!(format_size(1_048_576), "1.0 MiB");
        assert_eq!(format_size(2_621_440), "2.5 MiB");
        assert_eq!(format_size(3_145_727), "3.0 MiB");
        assert_eq!(format_size(16 * super::MIB), "16.0 MiB");
    }

    /// Every precision renders the reader's local clock, ends in the offset that clock has from
    /// UTC, occupies exactly the width it declares, and reads back to the instant it was given.
    /// Parsed back rather than compared to a literal, so the assertion holds in every zone the test
    /// runs in.
    #[test]
    fn a_timestamp_renders_in_local_time_with_its_offset_at_every_precision() {
        let instant = chrono::DateTime::parse_from_rfc3339("2026-09-02T22:57:28Z")
            .expect("a valid instant")
            .with_timezone(&chrono::Utc);
        let to_the_minute = instant - chrono::TimeDelta::seconds(28);
        let local_offset =
            chrono::TimeZone::offset_from_utc_datetime(&chrono::Local, &instant.naive_utc());
        for (precision, pattern, expected) in [
            (Precision::Seconds, "%Y-%m-%d %H:%M:%S %:z", instant),
            (Precision::Minutes, "%Y-%m-%d %H:%M %:z", to_the_minute),
            (
                Precision::WeekdayMinutes,
                "%a %Y-%m-%d %H:%M %:z",
                to_the_minute,
            ),
        ] {
            let rendered = format_timestamp(instant, precision);
            assert_eq!(rendered.len(), precision.width(), "{rendered}");
            assert!(
                rendered.ends_with(&local_offset.to_string()),
                "{rendered} must carry the local offset {local_offset}"
            );
            let parsed = chrono::DateTime::parse_from_str(&rendered, pattern)
                .unwrap_or_else(|error| panic!("{rendered} does not read back: {error}"));
            assert_eq!(parsed.with_timezone(&chrono::Utc), expected, "{rendered}");
        }
    }

    /// Both shapes of the template: the configured list, and the empty case that says so rather
    /// than trailing a colon into nothing.
    #[test]
    fn an_unknown_name_is_refused_beside_the_names_that_exist() {
        assert_eq!(
            unknown_name("profile", "ghost", ["work", "personal"]),
            "no profile named 'ghost' (configured: work, personal)"
        );
        assert_eq!(
            unknown_name("account", "ghost", Vec::<String>::new()),
            "no account named 'ghost' (none configured)"
        );
    }

    /// The longest prefix that fits, whether the cut lands between characters or in front of a
    /// wide one that would overflow, and the whole text when it fits: both ways out of the loop.
    #[test]
    fn take_columns_keeps_the_longest_prefix_that_fits() {
        assert_eq!(take_columns("abcdef", 3), "abc");
        assert_eq!(take_columns("漢字漢", 3), "漢");
        assert_eq!(take_columns("abc", 3), "abc");
        assert_eq!(take_columns("abc", 0), "");
    }

    /// Both ends survive with the marker between them and the tail takes the odd column; a text
    /// that fits comes back as it is, and a budget too narrow for two ends falls back to a tail
    /// cut.
    #[test]
    fn an_elision_keeps_both_ends_and_gives_the_tail_the_odd_column() {
        assert_eq!(elide_to_width("abcdefghij", 20), "abcdefghij");
        assert_eq!(elide_to_width("abcdefghijklmnop", 10), "abc...mnop");
        assert_eq!(elide_to_width("abcdefghij", 5), "ab...");
    }

    /// A zero budget in either dimension shows nothing: the caller has spent the width on indent,
    /// and returning the text anyway is the unbounded row the budget exists to prevent.
    #[test]
    fn a_zero_budget_in_either_dimension_wraps_to_no_rows() {
        assert_eq!(wrap_to_width("abc", 0, 3), Vec::<String>::new());
        assert_eq!(wrap_to_width("abc", 10, 0), Vec::<String>::new());
    }

    /// A remainder that fits its row is the last row with no count in front of it, and a remainder
    /// exactly as wide as the row fits.
    #[test]
    fn a_remainder_that_fits_is_the_last_row_without_a_count() {
        assert_eq!(wrap_to_width("aaaaa bbbbb", 10, 3), ["aaaaa", "bbbbb"]);
        assert_eq!(wrap_to_width("aaaaa bbbbbbbbbb", 10, 3), [
            "aaaaa",
            "bbbbbbbbbb"
        ]);
    }

    /// When the text does not fit, the last two rows are a count of what was skipped and the end
    /// of the text, so the tail of a command being approved is never the part that is hidden.
    #[test]
    fn an_overflowing_wrap_ends_with_a_count_and_the_end_of_the_text() {
        let text = format!("aaaa {}", "b".repeat(60));
        let tail = "b".repeat(30);
        assert_eq!(wrap_to_width(&text, 30, 3), [
            "aaaa",
            "... 30 more characters ...",
            tail.as_str()
        ]);
    }

    /// Below three rows there is no room for a count and an end, so the last row is cut where it
    /// lands; the row budget is still the row budget.
    #[test]
    fn below_three_rows_the_last_row_is_cut_and_the_row_budget_still_holds() {
        let text = "aaaaa bbbbb ccccc ddddd eeeee";
        assert_eq!(wrap_to_width(text, 10, 1), ["aaaaa b..."]);
        assert_eq!(wrap_to_width(text, 10, 2), ["aaaaa", "bbbbb c..."]);
    }

    /// The resolvers compare in lowercase, so a pasted uppercase prefix folds to what the store
    /// holds, and a prefix shorter than the printed [`super::ID_PREFIX`] folds rather than
    /// vanishes.
    #[test]
    fn an_id_prefix_is_folded_to_lowercase_for_matching() {
        assert_eq!(id_prefix_for_matching("4D71EECA"), "4d71eeca");
        assert_eq!(id_prefix_for_matching("4D"), "4d");
        assert_eq!(id_prefix_for_matching("4d71eeca-ab"), "4d71eeca-ab");
    }
}
