//! The dependency direction of the tree, as a test.
//!
//! Every top-level module has a rank in [`RANKS`], and a `crate::<module>` path in production code
//! is an edge that may only point at a module of strictly greater rank: further down the tree.
//! Siblings may not name each other, so the tree is a DAG by construction rather than by a list of
//! forbidden pairs, and a module added without a rank fails the test until it is placed.
//!
//! Test code is stripped before edges are read: a `#[cfg(test)]` module or item may reach anywhere,
//! because it ships nowhere. Doc comments are ignored for the same reason.
//!
//! Edges that exist today and are scheduled to go are listed in [`TOLERATED`], so the test fails in
//! both directions: a new upward or sideways edge fails, and removing a tolerated one without
//! deleting its entry fails too, which keeps the list an honest ledger of what is left.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "tests panic on failure by design, and indexing a JSON document is the readable form"
)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

/// Every top-level module and its rank. Lower is higher in the tree; an edge must point from a
/// lower number to a strictly greater one.
const RANKS: &[(&str, u32)] = &[
    ("main", 0),
    // The hosts, then the clap handlers: the REPL's slash commands are the interactive twins of
    // the subcommands and run their handlers, so `host` sits above `cli`.
    ("host", 1),
    ("cli", 2),
    // Presentation, reached only from the layers that own a terminal. The relay writes through
    // the console, which draws with the renderer.
    ("relay", 2),
    ("console", 3),
    ("render", 4),
    ("agent", 5),
    // The JSON shapes both hosts print, converted from the domain types below them.
    ("view", 5),
    ("tools", 6),
    // The system prompt reads the scheduled and background state it describes; a session's
    // materials hold the background handles.
    ("prompt", 7),
    ("session", 8),
    // The scheduler runs jobs; `schedule`, below the store, is what a job is.
    ("scheduler", 9),
    ("background", 9),
    ("mcp", 10),
    ("provider", 11),
    ("frontend", 12),
    ("skills", 12),
    ("instructions", 12),
    ("oauth", 12),
    ("sandbox", 12),
    ("workspace", 13),
    ("tokens", 13),
    ("store", 14),
    ("schedule", 15),
    // The model: vocabulary every layer above may hold.
    ("conversation", 16),
    ("stats", 16),
    ("config", 16),
    ("memory", 16),
    ("entry", 17),
    ("permission", 17),
    ("todo", 17),
    ("image", 17),
    // Leaves.
    ("fs", 18),
    ("error", 19),
    ("sync", 19),
    ("text", 20),
    ("streams", 19),
    ("paths", 19),
];

/// Edges that exist today and are scheduled to go. Each is one file's use of a module at or above
/// its own rank.
const TOLERATED: &[(&str, &str)] = &[];

/// Edges that point up by design and stay: the sub-agent tool builds an agent, so it is the one
/// place below `agent` that must name it.
const BY_DESIGN: &[(&str, &str)] = &[("src/tools/subagent.rs", "agent")];

fn source_files(directory: &Path, into: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(directory).expect("read a source directory");
    for entry in entries {
        let path = entry.expect("read a directory entry").path();
        if path.is_dir() {
            source_files(&path, into);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            into.push(path);
        }
    }
}

/// The source with every `#[cfg(test)]` item removed: a module, a function, an `impl` block, a
/// `use`, whatever the attribute is on. The item runs to its closing brace, or to the `;` when it
/// has no body.
fn strip_test_code(source: &str) -> String {
    let mut kept = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(at) = rest.find("#[cfg(test)]") {
        kept.push_str(&rest[..at]);
        let after_attribute = &rest[at + "#[cfg(test)]".len()..];
        rest = skip_item(after_attribute);
    }
    kept.push_str(rest);
    kept
}

/// What follows one item: the text after its closing brace, or after its `;` when it has no body.
/// Further attributes on the same item are part of it.
fn skip_item(text: &str) -> &str {
    let mut rest = text;
    loop {
        let trimmed = rest.trim_start();
        if trimmed.starts_with("#[") {
            let close = trimmed.find(']').expect("an attribute closes");
            rest = &trimmed[close + 1..];
            continue;
        }
        rest = trimmed;
        break;
    }
    let body_open = rest.find('{');
    let terminator = rest.find(';');
    match (body_open, terminator) {
        (Some(open), Some(end)) if end < open => &rest[end + 1..],
        (None, Some(end)) => &rest[end + 1..],
        (Some(open), _) => {
            let (_, after) = split_group(&rest[open + 1..]);
            after
        }
        (None, None) => "",
    }
}

/// The `crate::<module>` edges in one file, ignoring comment lines so a doc link is not an edge.
///
/// A grouped import, `use crate::{agent::Agent, store::Store}`, names one module per top-level
/// entry, so the group is opened and each entry's first segment counts. Without that a file could
/// reach up the tree through a group and never be seen doing it.
fn edges_of(source: &str) -> BTreeSet<String> {
    let code = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut edges = BTreeSet::new();
    let mut rest = code.as_str();
    while let Some(at) = rest.find("crate::") {
        rest = &rest[at + "crate::".len()..];
        if let Some(group) = rest.strip_prefix('{') {
            let (inner, after) = split_group(group);
            for entry in top_level_entries(inner) {
                if let Some(module) = leading_module(entry) {
                    edges.insert(module);
                }
            }
            rest = after;
        } else if let Some(module) = leading_module(rest) {
            edges.insert(module);
        }
    }
    edges
}

/// The text inside a brace group that has just been opened, and what follows its close.
///
/// A brace inside a string, a char literal or a comment does not count: a test that scans source
/// text may hold `"\n    }\n"`, and a comment may mention `{` alone.
fn split_group(after_open: &str) -> (&str, &str) {
    let characters: Vec<(usize, char)> = after_open.char_indices().collect();
    let mut depth = 1;
    let mut index = 0;
    while index < characters.len() {
        let (offset, character) = characters[index];
        let next = characters.get(index + 1).map(|(_, next)| *next);
        match character {
            '/' if next == Some('/') => {
                while index < characters.len() && characters[index].1 != '\n' {
                    index += 1;
                }
            }
            '/' if next == Some('*') => {
                index += 2;
                while index + 1 < characters.len()
                    && !(characters[index].1 == '*' && characters[index + 1].1 == '/')
                {
                    index += 1;
                }
                index += 2;
            }
            '"' => index = skip_string(&characters, index),
            'r' if matches!(next, Some('"') | Some('#'))
                && raw_string_starts(&characters, index) =>
            {
                index = skip_raw_string(&characters, index);
            }
            '\'' => index = skip_char_literal(&characters, index),
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return (&after_open[..offset], &after_open[offset + 1..]);
                }
            }
            _ => {}
        }
        index += 1;
    }
    (after_open, "")
}

/// The index of the closing quote of the string that opens at `start`, honoring escapes.
fn skip_string(characters: &[(usize, char)], start: usize) -> usize {
    let mut index = start + 1;
    while index < characters.len() {
        match characters[index].1 {
            '\\' => index += 2,
            '"' => return index,
            _ => index += 1,
        }
    }
    characters.len()
}

/// Whether `r` at `start` opens a raw string (`r"` or `r#"`), not an identifier.
fn raw_string_starts(characters: &[(usize, char)], start: usize) -> bool {
    let preceded_by_identifier =
        start > 0 && (characters[start - 1].1.is_alphanumeric() || characters[start - 1].1 == '_');
    if preceded_by_identifier {
        return false;
    }
    let mut index = start + 1;
    while index < characters.len() && characters[index].1 == '#' {
        index += 1;
    }
    index < characters.len() && characters[index].1 == '"'
}

/// The index of the last character of the raw string that opens at `start`.
fn skip_raw_string(characters: &[(usize, char)], start: usize) -> usize {
    let mut hashes = 0;
    let mut index = start + 1;
    while characters[index].1 == '#' {
        hashes += 1;
        index += 1;
    }
    index += 1;
    while index < characters.len() {
        if characters[index].1 == '"'
            && (1..=hashes)
                .all(|offset| characters.get(index + offset).map(|(_, c)| *c) == Some('#'))
        {
            return index + hashes;
        }
        index += 1;
    }
    characters.len()
}

/// The index of the closing quote of a char literal at `start`, or `start` itself for a lifetime.
fn skip_char_literal(characters: &[(usize, char)], start: usize) -> usize {
    match (
        characters.get(start + 1).map(|(_, c)| *c),
        characters.get(start + 2).map(|(_, c)| *c),
    ) {
        (Some('\\'), _) => {
            let mut index = start + 2;
            while index < characters.len() && characters[index].1 != '\'' {
                index += 1;
            }
            index
        }
        (Some(_), Some('\'')) => start + 2,
        _ => start,
    }
}

/// The comma-separated entries of a group, ignoring commas inside nested groups.
fn top_level_entries(inner: &str) -> Vec<&str> {
    let mut entries = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (index, character) in inner.char_indices() {
        match character {
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 0 => {
                entries.push(inner[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    entries.push(inner[start..].trim());
    entries
}

/// The module an entry or path starts with: its first identifier, if it is one.
fn leading_module(text: &str) -> Option<String> {
    let module: String = text
        .trim_start()
        .chars()
        .take_while(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || *character == '_'
        })
        .collect();
    (!module.is_empty()).then_some(module)
}

/// `src/store.rs` and `src/store/sessions.rs` both belong to `store`; `src/main.rs` is `main`.
fn top_module(relative: &str) -> &str {
    let inside = relative.strip_prefix("src/").expect("a path under src/");
    let first = inside.split('/').next().expect("a non-empty path");
    first.strip_suffix(".rs").unwrap_or(first)
}

#[test]
fn no_module_reaches_a_layer_above_or_beside_it() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    source_files(&root.join("src"), &mut files);
    files.sort();

    let ranks: BTreeMap<&str, u32> = RANKS.iter().copied().collect();
    let allowed: BTreeSet<(&str, &str)> = TOLERATED.iter().chain(BY_DESIGN).copied().collect();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    let mut violations = Vec::new();

    for path in &files {
        let relative = path
            .strip_prefix(root)
            .expect("a path under the manifest directory")
            .to_string_lossy()
            .replace('\\', "/");
        let source = fs::read_to_string(path).expect("read a source file");
        let module = top_module(&relative);
        let Some(&rank) = ranks.get(module) else {
            violations.push(format!(
                "{relative}: module `{module}` has no rank; place it"
            ));
            continue;
        };
        for edge in edges_of(&strip_test_code(&source)) {
            if edge == module {
                continue;
            }
            let Some(&target_rank) = ranks.get(edge.as_str()) else {
                // `crate::sync_helper` style names that are not modules: a path into a
                // re-exported item at the crate root. The root re-exports nothing today, so any
                // unknown name is a module that was added without a rank.
                violations.push(format!("{relative} names crate::{edge}, which has no rank"));
                continue;
            };
            if target_rank > rank {
                continue;
            }
            if allowed.contains(&(relative.as_str(), edge.as_str())) {
                seen.insert((relative.clone(), edge));
                continue;
            }
            let direction = if target_rank == rank {
                "beside"
            } else {
                "above"
            };
            violations.push(format!(
                "{relative} names crate::{edge}, which is {direction} it"
            ));
        }
    }

    for (file, module) in TOLERATED.iter().chain(BY_DESIGN) {
        if !seen.contains(&((*file).to_string(), (*module).to_string())) {
            violations.push(format!(
                "{file} does not name crate::{module}; remove it from the tolerated list"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "layering violations:\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn code_is_not_an_edge() {
    let source = "use crate::store::Store;\n#[cfg(test)]\nmod tests {\n    use crate::host::Sessions;\n    let close = \"\\n    }\\n\"; let open = '{'; // a brace: }\n    let raw = r#\"}\"#;\n}\n#[cfg(test)]\nuse crate::cli::Cli;\n#[cfg(test)]\n#[allow(dead_code)]\nfn helper() { crate::render::x(); }\nfn kept() { crate::agent::y(); }\n";
    let edges = edges_of(&strip_test_code(source));
    let expected: BTreeSet<String> = ["store", "agent"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert_eq!(edges, expected);
}
