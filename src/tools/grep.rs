//! `file_search` tool: ripgrep-style content search powered by the `grep-*` crates, with glob
//! filtering.
//!
//! It does **not** honor `.gitignore`, despite the name suggesting ripgrep's behavior: the walk
//! here is a hand-rolled `read_dir` traversal whose only exclusions are dotfiles, `target` and
//! `node_modules`, and the `ignore` crate is not a dependency. Only the matcher comes from the
//! `grep-*` family.

use std::io::Read;

use async_trait::async_trait;
use grep_searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkMatch};

use super::{
    Tool, ToolOutput,
    util::{WalkBudget, WalkStop, optional_str, redirects_to_scratchpad, require_str},
};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
};

/// Inline match cap when the agent isn't redirecting to the scratchpad, and the ceiling an explicit
/// `limit` is clamped to. Single source of truth for the description and the runtime cap.
const MAX_INLINE_MATCHES: usize = 100;

/// The most the searcher holds of one file at a time, which makes it the longest line a search
/// reads. A file with no line terminator is one line, and unbounded the searcher grows its buffer
/// to the whole of it: a disk image or a multi-gigabyte export on one line would take the process
/// down, and under `serve` every session with it. The ceiling is the one a whole `file_read` is
/// held to, since a line past that is one no reader here returns either.
const SEARCH_HEAP_BYTES: usize = super::file::MAX_READ_FILE_BYTES;

pub(super) struct SearchContentsTool {
    pub(crate) site: crate::session::ToolSite,
}

#[async_trait]
impl Tool for SearchContentsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "file_search".to_string(),
            description: format!(
                "Search file contents with a case-sensitive Rust regex. Directory walks skip dot-prefixed entries, `target`, `node_modules`, and symlinked directories; name an excluded path explicitly to search it. The `glob` filter matches filenames, not relative paths. Searches do not apply gitignore rules. Inline results are capped at {MAX_INLINE_MATCHES}; `scratchpad` without `limit` removes the match cap, not the search time budget."
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in. Omit to search the workspace: the working directory and the additional workspace directories the environment context lists. Set it to search a specific file or subtree."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob matched against each filename, not its relative path."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_INLINE_MATCHES,
                        "default": MAX_INLINE_MATCHES,
                        "description": format!(
                            "Maximum matches to return, at most {MAX_INLINE_MATCHES}. Default: \
                             {MAX_INLINE_MATCHES}, or unbounded when `scratchpad` is set and no \
                             `limit` is passed."
                        )
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
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
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        let pattern = require_str(&input, "pattern", "file_search")?;
        // An explicit `path` searches exactly that tree, resolved against the per-session cwd. With
        // no `path`, sweep every workspace root: in a multi-root ACP workspace, searching only
        // `cwd` silently misses whole folders the user can see in their editor. Carried as
        // `PathBuf` end to end: a round trip through `to_string_lossy` would replace every
        // non-UTF-8 byte with U+FFFD and report the user's own cwd as missing.
        let search_paths: Vec<std::path::PathBuf> =
            match optional_str(&input, "path", "file_search")? {
                Some(raw) => vec![crate::workspace::resolve_against_cwd(&self.site.cwd, raw)],
                None => crate::workspace::search_roots(&self.site.cwd, &self.site.roots),
            };
        let file_glob = optional_str(&input, "glob", "file_search")?.map(str::to_string);
        // Cap precedence, as `file_find` has it: an explicit `limit` wins, clamped to the inline
        // cap the way `memory_search` clamps its own; with none, `scratchpad` lifts the cap so the
        // agent can collect an unbounded result set.
        let max_results = match super::util::optional_count(&input, "limit", "file_search")? {
            Some(limit) => limit.clamp(1, MAX_INLINE_MATCHES),
            None if redirects_to_scratchpad(&input) => usize::MAX,
            None => MAX_INLINE_MATCHES,
        };

        // One budget for the whole call, not one per root: `WalkBudget::new` stamps its deadline at
        // construction, so a per-root budget would silently multiply the ceiling by the root count.
        let budget = WalkBudget::new(cancellation.clone());
        let fence = self.site.read_scope().fence(&self.site.cwd);
        let search = tokio::task::spawn_blocking(move || {
            search_with_grep(
                &pattern,
                &search_paths,
                file_glob.as_deref(),
                max_results,
                &budget,
                &fence,
            )
        });

        // Race the search against the token: a started `spawn_blocking` task cannot be aborted, so
        // awaiting it unconditionally would let a walk rooted high in the tree hold the turn open.
        // The search checks the same token per directory entry and stops on its own.
        let result = tokio::select! {
            joined = search => joined.map_err(|error| MekaError::ToolExecution {
                tool_name: "file_search".to_string(),
                message: format!("task join error: {error}"),
            })??,
            _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
        };

        Ok(ToolOutput::text(result, false))
    }
}

/// `search_paths` holds one workspace root per entry (or the single tree the caller named via
/// `path`), walked in order under a single shared `budget`. Results, the truncation cap, and the
/// timeout note all span the whole set, so the output describes the search rather than its last
/// leg.
fn search_with_grep(
    pattern: &str,
    search_paths: &[std::path::PathBuf],
    file_glob: Option<&str>,
    max_results: usize,
    budget: &WalkBudget,
    // Meka's own directories at this permission, which a named root may not be inside and the
    // walk steps around; see `crate::workspace::private_read_refusal`.
    fence: &crate::workspace::ReadFence,
) -> Result<String> {
    use grep_regex::RegexMatcherBuilder;

    // Cap the compiled-regex automaton and DFA cache sizes so an LLM-supplied pattern like
    // `a{10_000_000}` can't exhaust host memory during compile.
    const PATTERN_SIZE_BYTES: usize = crate::text::MIB;
    const DFA_SIZE_BYTES: usize = crate::text::MIB;

    let matcher = RegexMatcherBuilder::new()
        .size_limit(PATTERN_SIZE_BYTES)
        .dfa_size_limit(DFA_SIZE_BYTES)
        .build(pattern)
        .map_err(|error| MekaError::ToolExecution {
            tool_name: "file_search".to_string(),
            message: format!("invalid or oversized regex '{pattern}': {error}"),
        })?;

    let mut results = Vec::new();
    let mut timed_out = false;
    // A root that does not exist is skipped rather than fatal, because one stale entry in a
    // multi-root workspace should not sink a search the other roots can answer; with a single
    // explicit `path` the error below still fires.
    let mut searched_any = false;
    // Compiled lazily and at most once. Deliberately inside the directory branch rather than
    // hoisted above the loop: hoisting would report an invalid `glob` for a path that doesn't
    // exist, or for a single file where the glob is irrelevant, changing single-root behavior.
    let mut glob_pattern: Option<glob::Pattern> = None;
    // Roots left unsearched because the match cap filled up first. cwd is always root #1, so a
    // busy cwd would otherwise starve every other root and report only "truncated", which reads as
    // "the other folders had nothing".
    let mut unsearched_roots = 0usize;
    let mut skipped = Skipped::default();

    for search_path in search_paths {
        // Checked per root as well as inside `walk_directory`: a root that is a plain file, or that
        // doesn't exist, never reaches the walk, so a long list of them would advance without
        // consulting the budget once and ignore both the deadline and a `session/cancel`.
        match budget.check() {
            Some(WalkStop::Canceled) => return Err(MekaError::Interrupted),
            Some(WalkStop::TimedOut) => {
                timed_out = true;
                break;
            }
            None => {}
        }

        let path = search_path.as_path();

        // Stop before the walk, not after, so the remaining roots are counted rather than silently
        // dropped. Checked after `exists`, not before: counting a stale root here would advertise
        // "pass `path` to search one of them directly" about a directory that is gone, and the
        // model spends a round trip discovering that.
        if results.len() > max_results {
            if path.exists() {
                unsearched_roots += 1;
            }
            continue;
        }

        // A root the caller named is refused outright rather than stepped around: pointing `path`
        // at the store, or outside the readable roots, is the question, and a silent empty answer
        // would read as "nothing there".
        if let Some(message) = fence.refusal(path) {
            return Err(MekaError::ToolExecution {
                tool_name: "file_search".to_string(),
                message,
            });
        }

        if path.is_file() {
            searched_any = true;
            if search_file(
                &matcher,
                path,
                &mut results,
                max_results,
                budget,
                &mut skipped,
            )? {
                timed_out = true;
                break;
            }
        } else if path.is_dir() {
            searched_any = true;
            if glob_pattern.is_none()
                && let Some(g) = file_glob
            {
                glob_pattern =
                    Some(
                        glob::Pattern::new(g).map_err(|error| MekaError::ToolExecution {
                            tool_name: "file_search".to_string(),
                            message: format!("invalid glob pattern '{g}': {error}"),
                        })?,
                    );
            }
            let scope = SearchScope {
                matcher: &matcher,
                glob_pattern: &glob_pattern,
                max_results,
                budget,
                fence,
            };
            if walk_directory(path, &scope, &mut results, &mut skipped)? {
                timed_out = true;
                break;
            }
        } else {
            continue;
        }
    }

    // Only claim the path is missing when we actually looked. A budget that expired before the
    // first root was examined leaves `searched_any` false while saying nothing about whether the
    // path exists, and "does not exist" is a definitive answer the model will act on.
    if !searched_any && !timed_out {
        return Err(MekaError::ToolExecution {
            tool_name: "file_search".to_string(),
            message: format!(
                "path '{}' does not exist",
                search_paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join("', '")
            ),
        });
    }

    // The walk collects one match past the cap so "more exist" is distinguishable from "exactly
    // this many exist" without having to search the rest of the tree to find out.
    let truncated = results.len() > max_results;
    if truncated {
        results.truncate(max_results);
    }
    let Skipped {
        unreadable,
        binary,
        oversized,
    } = skipped;

    // A search that was cut short must say so even when it found nothing: a bare "No matches
    // found." on an unfinished search reads as a definitive answer.
    let mut notes: Vec<String> = Vec::new();
    if truncated {
        notes.push(format!("truncated, showing first {max_results} matches"));
    }
    if unsearched_roots > 0 {
        notes.push(format!(
            "{unsearched_roots} workspace root(s) were not searched because the match cap filled first: pass \
             `path` to search one of them directly, or `scratchpad` to lift the cap",
        ));
    }
    if timed_out {
        notes.push(format!(
            "search was still running after {}s and was stopped, so these results are \
             incomplete: narrow `path` to a smaller subtree or add a tighter `glob`",
            budget.budget_secs(),
        ));
    }
    if unreadable > 0 {
        notes.push(format!(
            "{unreadable} file(s) or director(ies) could not be read and were skipped, so a match \
             inside them would not appear here",
        ));
    }
    if binary > 0 {
        notes.push(format!(
            "{binary} file(s) hold binary data and were skipped"
        ));
    }
    if oversized > 0 {
        notes.push(format!(
            "{oversized} file(s) have a line longer than {} and were skipped",
            crate::text::format_size(SEARCH_HEAP_BYTES)
        ));
    }

    let body = if results.is_empty() {
        "No matches found.".to_string()
    } else {
        results.join("\n")
    };
    if notes.is_empty() {
        Ok(body)
    } else {
        Ok(format!("{}\n\n... ({})", body, notes.join("; ")))
    }
}

/// What a search left out, by why: each is a place a match could be, so each is said.
#[derive(Default)]
struct Skipped {
    /// A directory that could not be opened, or a file that could not be opened or read or that
    /// matched on a line that is not UTF-8.
    unreadable: usize,
    /// A file that holds a NUL byte.
    binary: usize,
    /// A file with a line longer than [`SEARCH_HEAP_BYTES`].
    oversized: usize,
}

/// Search one file, stopping once `results` holds one entry more than `max_results`: a single file
/// can hold millions of matching lines, so the cap has to bound collection here too, not only at
/// the end of the walk. Returns whether the time budget ran out inside the file; errors with
/// [`MekaError::Interrupted`] when the turn was canceled.
///
/// A file that could not be searched whole is counted in `skipped` by why, since it may hold the
/// match and a `debug!` alone would let "No matches found." read as definitive. A binary file and
/// one with a line over the limit give up what they matched before that was known, so "skipped"
/// is true of them.
fn search_file(
    matcher: &grep_regex::RegexMatcher,
    path: &std::path::Path,
    results: &mut Vec<String>,
    max_results: usize,
    budget: &WalkBudget,
    skipped: &mut Skipped,
) -> Result<bool> {
    let mut searcher = SearcherBuilder::new()
        // A NUL is what tells a binary file from text, as ripgrep reads it: the lines of an image
        // or a database are noise to the model, and the longest of them is unbounded.
        .binary_detection(BinaryDetection::quit(b'\x00'))
        .heap_limit(Some(SEARCH_HEAP_BYTES))
        .build();
    let found_before = results.len();
    let mut sink = Matches {
        path,
        results,
        max_results,
        is_binary: false,
        is_undecodable: false,
    };
    let searched = std::fs::File::open(path)
        .and_then(|file| searcher.search_reader(matcher, Budgeted { file, budget }, &mut sink));
    let (is_binary, is_undecodable) = (sink.is_binary, sink.is_undecodable);
    match searched {
        Ok(()) if is_binary => {
            results.truncate(found_before);
            skipped.binary += 1;
        }
        Ok(()) if is_undecodable => skipped.unreadable += 1,
        Ok(()) => {}
        Err(error) => match Halted::in_error(&error) {
            Some(WalkStop::Canceled) => return Err(MekaError::Interrupted),
            Some(WalkStop::TimedOut) => return Ok(true),
            // The standard library never makes an error of this kind, so one that is not the
            // reader's own is the searcher's, and the one it makes over a file it could open is
            // its refusal to grow the buffer past the limit.
            None if error.kind() == std::io::ErrorKind::Other => {
                results.truncate(found_before);
                skipped.oversized += 1;
            }
            None => {
                let path = path.display();
                tracing::debug!("failed to search {path}: {error}");
                skipped.unreadable += 1;
            }
        },
    }
    Ok(false)
}

/// Where one file's matching lines go, as `path:line:text` rows.
struct Matches<'a> {
    path: &'a std::path::Path,
    results: &'a mut Vec<String>,
    max_results: usize,
    /// Whether the searcher met a NUL and quit.
    is_binary: bool,
    /// Whether a matching line was not UTF-8, which ends the file's search.
    is_undecodable: bool,
}

impl Sink for Matches<'_> {
    type Error = std::io::Error;

    fn matched(&mut self, _searcher: &Searcher, matched: &SinkMatch<'_>) -> std::io::Result<bool> {
        let (Ok(line), Some(line_number)) =
            (std::str::from_utf8(matched.bytes()), matched.line_number())
        else {
            self.is_undecodable = true;
            return Ok(false);
        };
        self.results.push(format!(
            "{}:{}:{}",
            self.path.display(),
            line_number,
            line.trim_end()
        ));
        Ok(self.results.len() <= self.max_results)
    }

    fn binary_data(&mut self, _searcher: &Searcher, _offset: u64) -> std::io::Result<bool> {
        self.is_binary = true;
        Ok(false)
    }
}

/// A file as the searcher reads it, asking the budget at every read. The walk asks between
/// files, and the searcher calls its sink only for a line that matches, so the reads are the one
/// place inside a file that every search passes however little it finds: without this a single
/// huge file outlasts the deadline and a canceled call.
struct Budgeted<'a> {
    file: std::fs::File,
    budget: &'a WalkBudget,
}

impl Read for Budgeted<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        match self.budget.check() {
            Some(stop) => Err(std::io::Error::other(Halted(stop))),
            None => self.file.read(buffer),
        }
    }
}

/// The error a [`Budgeted`] read ends a file's search with.
#[derive(Debug)]
struct Halted(WalkStop);

impl Halted {
    /// The stop `error` carries, when it is a [`Budgeted`] read's.
    fn in_error(error: &std::io::Error) -> Option<WalkStop> {
        error
            .get_ref()
            .and_then(|inner| inner.downcast_ref::<Self>())
            .map(|halted| halted.0)
    }
}

impl std::fmt::Display for Halted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("the search was stopped")
    }
}

impl std::error::Error for Halted {}

/// What one search is judged against, the same for every root and every directory under it.
struct SearchScope<'a> {
    matcher: &'a grep_regex::RegexMatcher,
    glob_pattern: &'a Option<glob::Pattern>,
    max_results: usize,
    budget: &'a WalkBudget,
    /// Meka's own directories at this permission, which the walk steps around; see
    /// `crate::workspace::private_read_refusal`.
    fence: &'a crate::workspace::ReadFence,
}

/// Walk `directory`, searching every file that passes the scope's glob. Returns whether the walk
/// was stopped by the time budget; errors with [`MekaError::Interrupted`] when the turn was
/// canceled.
///
/// `skipped` counts the directories the walk could not open and the files it could not search,
/// so the caller can say so: a silent skip turns a tree with an unreadable subdirectory into a
/// confident "No matches found.".
fn walk_directory(
    directory: &std::path::Path,
    scope: &SearchScope<'_>,
    results: &mut Vec<String>,
    skipped: &mut Skipped,
) -> Result<bool> {
    let SearchScope {
        matcher,
        glob_pattern,
        max_results,
        budget,
        fence,
    } = *scope;
    // Iterative traversal via an explicit work-stack: a recursive walk would overflow the call
    // stack on a pathologically deep directory tree.
    let mut pending: Vec<std::path::PathBuf> = vec![directory.to_path_buf()];

    while let Some(dir) = pending.pop() {
        // Stepped around, like the dot-directories below: a root above Meka's directories is a
        // legitimate workspace, and what lies inside them is not this walk's to read. Resolved
        // per directory, since the root itself need not be canonical.
        if fence.hides(&dir) {
            continue;
        }
        // Checked here as well as per entry: a run of directories that all fail `read_dir` (a tree
        // the user has no permission for) never reaches the inner loop, and would otherwise grind
        // through the whole work-stack without consulting the budget once.
        match budget.check() {
            Some(WalkStop::Canceled) => return Err(MekaError::Interrupted),
            Some(WalkStop::TimedOut) => return Ok(true),
            None => {}
        }

        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) => {
                let directory = dir.display();
                tracing::debug!("failed to read directory '{directory}': {error}");
                skipped.unreadable += 1;
                continue;
            }
        };

        for entry in entries {
            match budget.check() {
                Some(WalkStop::Canceled) => return Err(MekaError::Interrupted),
                Some(WalkStop::TimedOut) => return Ok(true),
                None => {}
            }

            let Ok(entry) = entry else { continue };
            let path = entry.path();

            // `to_string_lossy`, not `to_str().unwrap_or("")`: a `.cache\xff` would yield `""`
            // and walk straight past this skip. Lossy conversion never alters ASCII bytes, so the
            // leading dot and both literals below survive it intact.
            let file_name = path.file_name().unwrap_or_default().to_string_lossy();
            if file_name.starts_with('.') || file_name == "target" || file_name == "node_modules" {
                continue;
            }

            // `entry.file_type()` does not follow symlinks: a symlinked directory reports as a
            // symlink, not a dir, so it is never descended into. That removes any symlink-cycle
            // risk while still letting symlinked *files* be searched via the path-based `is_file()`
            // check below.
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                pending.push(path);
            } else if path.is_file() {
                if let Some(pattern) = glob_pattern
                    && !pattern.matches(&file_name)
                {
                    continue;
                }
                // A symlink is the one way a file under a permitted directory resolves into
                // Meka's own; a regular file cannot, since the walk never descends a symlinked
                // directory and every directory it does enter was judged above.
                if file_type.is_symlink() && fence.hides(&path) {
                    continue;
                }
                if search_file(matcher, &path, results, max_results, budget, skipped)? {
                    return Ok(true);
                }
                // Stop walking once the cap is exceeded. Reading every remaining file on the
                // machine to fill a result set that is already being truncated is pure waste.
                if results.len() > max_results {
                    return Ok(false);
                }
            }
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;

    /// `file_search` below `unrestricted` refuses a `path` inside Meka's own directories, and a
    /// walk from a root above them neither enters them nor follows a symlink into them. `meka.db`
    /// holds a NUL, so a search would skip it as binary, but not the text beside it: a pattern
    /// for a key prefix would pull a value out of `config.toml`.
    #[tokio::test]
    async fn search_contents_steps_around_meka_s_own_directories_below_unrestricted() {
        use crate::permission::{EnabledPermissions, Permission, SharedPermission};

        let home = tempfile::tempdir().expect("tempdir");
        let config_dir = home.path().join("meka-config");
        std::fs::create_dir(&config_dir).expect("config dir");
        std::fs::write(
            config_dir.join("config.toml"),
            "token = \"sk-secret-value\"\n",
        )
        .expect("write config.toml");
        std::fs::write(home.path().join("notes.txt"), "sk-public-value\n").expect("write notes");
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            config_dir.join("config.toml"),
            home.path().join("link.toml"),
        )
        .expect("symlink into the store");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set -> search -> clear cycle.
        let _env = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", &config_dir) };

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test())
                .with_permission(SharedPermission::new(
                    Permission::Read,
                    EnabledPermissions::ALL,
                )),
        };
        let named = tool
            .execute(
                serde_json::json!({
                    "pattern": "sk-",
                    "path": config_dir.to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await;
        let walked = tool
            .execute(
                serde_json::json!({
                    "pattern": "sk-",
                    "path": home.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = named.expect_err("a root inside the store is refused");
        assert!(error.to_string().contains("Meka's own"), "{error}");
        let walked = walked.expect("a root above it is searched").text_content();
        assert!(walked.contains("sk-public-value"), "{walked}");
        assert!(
            !walked.contains("sk-secret-value"),
            "neither the directory nor the symlink into it is read: {walked}"
        );
    }

    /// A file the search cannot open is counted with the directories it cannot read, so the
    /// result says the answer may be incomplete instead of a definitive "No matches found.".
    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_file_is_reported_not_skipped() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let hidden = temp_dir.path().join("hidden.txt");
        std::fs::write(&hidden, "needle\n").expect("write");
        std::fs::set_permissions(&hidden, std::fs::Permissions::from_mode(0o000)).expect("chmod");
        if std::fs::read(&hidden).is_ok() {
            // Running as root, where the mode is not a boundary; there is nothing to observe.
            return;
        }

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "needle",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("the search completes");
        let text = result.text_content();
        assert!(text.contains("No matches found."), "{text}");
        assert!(
            text.contains("1 file(s) or director(ies) could not be read"),
            "the unreadable file must be disclosed: {text}"
        );
    }

    #[tokio::test]
    async fn search_contents_returns_every_matching_line() {
        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(
            temp_dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again\n",
        )
        .expect("failed");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "hello",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(result.text_content().contains("hello world"));
        assert!(result.text_content().contains("hello again"));
    }

    /// The counterpart to `file_find`' nested-root test, pinning why the two tools use different
    /// root sets. `file_search` descends, so `search_roots` pruning `cwd` in favor of an
    /// ancestor genuinely loses nothing here. If that ever stops holding, this fails rather than
    /// the tool quietly reporting a file in `cwd` as absent.
    #[tokio::test]
    async fn search_contents_reaches_cwd_through_an_ancestor_root() {
        let top = tempfile::tempdir().expect("tempdir");
        let nested = top.path().join("main");
        std::fs::create_dir(&nested).expect("mkdir");
        std::fs::write(nested.join("README.md"), "needle here\n").expect("write");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::SharedCwd::new(nested.clone()))
                .with_roots(crate::workspace::SharedRoots::new(vec![
                    top.path().to_path_buf(),
                ])),
        };
        let result = tool
            .execute(
                serde_json::json!({ "pattern": "needle" }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        let text = result.text_content();
        assert!(
            text.contains("needle here"),
            "a descending walk from the ancestor must still reach cwd; got: {text}",
        );
        assert_eq!(
            text.matches("README.md").count(),
            1,
            "and must not report it twice; got: {text}",
        );
    }

    #[tokio::test]
    async fn search_contents_deeply_nested_tree() {
        // Exercises the iterative work-stack traversal: a file buried many directory levels deep
        // must still be found. A recursive walk would recurse once per level; the iterative version
        // uses a heap stack.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let mut deep = temp_dir.path().to_path_buf();
        for _ in 0..300 {
            deep.push("d");
        }
        std::fs::create_dir_all(&deep).expect("create nested tree");
        std::fs::write(deep.join("buried.txt"), "needle here\n").expect("write");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "needle",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(result.text_content().contains("needle here"));
    }

    #[tokio::test]
    async fn search_contents_inline_capped_at_100() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        // One file with 150 matching lines.
        let content = (0..150).map(|_| "match\n").collect::<String>();
        std::fs::write(temp_dir.path().join("many.txt"), content).expect("write");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        assert!(
            result
                .text_content()
                .contains("truncated, showing first 100")
        );
    }

    /// `limit` lowers the inline cap and never raises it: the cap is the schema's declared
    /// maximum, and a value past it is clamped rather than refused, as `memory_search` does.
    #[tokio::test]
    async fn a_limit_lowers_the_inline_cap_and_is_clamped_to_it() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let content = (0..150).map(|_| "match\n").collect::<String>();
        std::fs::write(temp_dir.path().join("many.txt"), content).expect("write");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        for (limit, shown) in [(10, 10), (1_000, MAX_INLINE_MATCHES)] {
            let result = tool
                .execute(
                    serde_json::json!({
                        "pattern": "match",
                        "path": temp_dir.path().to_str().expect("path"),
                        "limit": limit,
                    }),
                    crate::tools::ToolContext::for_test(CancellationToken::new()),
                )
                .await
                .expect("should succeed");
            let text = result.text_content();
            assert!(
                text.contains(&format!("truncated, showing first {shown} matches")),
                "limit {limit}: got {text}"
            );
            let match_lines = text.lines().filter(|line| line.contains(":match")).count();
            assert_eq!(match_lines, shown, "limit {limit}: got {text}");
        }
    }

    /// An explicit `limit` beats the unbounded default `scratchpad` would otherwise apply, the
    /// precedence `file_find` documents for its own `limit`.
    #[tokio::test]
    async fn an_explicit_limit_beats_the_scratchpad_lift() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let content = (0..150).map(|_| "match\n").collect::<String>();
        std::fs::write(temp_dir.path().join("many.txt"), content).expect("write");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path"),
                    "scratchpad": "matches",
                    "limit": 10,
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("should succeed");
        let text = result.text_content();
        assert!(
            text.contains("truncated, showing first 10 matches"),
            "got: {text}"
        );
    }

    #[tokio::test]
    async fn search_contents_invalid_glob_errors() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp_dir.path().join("a.txt"), "match").expect("write");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        let error = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path"),
                    "glob": "[unclosed",
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect_err("invalid glob must be rejected, not silently scan everything");
        let message = format!("{error}");
        assert!(
            message.contains("invalid glob pattern"),
            "unexpected error: {message}"
        );
    }

    #[tokio::test]
    async fn search_contents_scratchpad_lifts_cap() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let content = (0..150).map(|_| "match\n").collect::<String>();
        std::fs::write(temp_dir.path().join("many.txt"), content).expect("write");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path"),
                    "scratchpad": "matches"
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        let text = result.text_content();
        assert!(
            !text.contains("truncated"),
            "expected no truncation marker when scratchpad set"
        );
        let match_lines = text.lines().filter(|l| l.contains("match")).count();
        assert!(
            match_lines >= 150,
            "expected >= 150 match lines, got {match_lines}"
        );
    }

    #[tokio::test]
    async fn search_contents_canceled_search_is_interrupted() {
        // An ignored cancellation token leaves a search rooted high in the tree running to
        // completion no matter what the user does.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..50 {
            std::fs::write(temp_dir.path().join(format!("f{i}.txt")), "match\n").expect("write");
        }

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = tool
            .execute(
                serde_json::json!({
                    "pattern": "match",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(cancellation),
            )
            .await
            .expect_err("a canceled turn must not run the search to completion");
        assert!(matches!(error, MekaError::Interrupted), "got: {error}");
    }

    /// A subdirectory the walk cannot open is not "no matches here", it is a part of the tree
    /// nobody looked at. Folding the two together turns a permissions error into a confident
    /// negative the model then answers from, which is the failure every other disclosure in this
    /// tool exists to prevent.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_directory_that_cannot_be_read_is_disclosed_not_counted_as_no_match() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let sealed = temp_dir.path().join("sealed");
        std::fs::create_dir(&sealed).expect("mkdir");
        std::fs::write(sealed.join("hit.txt"), "needle\n").expect("write");
        std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o000)).expect("seal");
        if std::fs::read_dir(&sealed).is_ok() {
            // Running as root, where the mode is advisory. Nothing to assert.
            return;
        }

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::cwd_for_test())
                .with_roots(crate::workspace::roots_for_test()),
        };
        let result = tool
            .execute(
                serde_json::json!({
                    "pattern": "needle",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        // Restore the mode so the temp dir can be torn down.
        std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o700)).expect("unseal");

        let text = result.text_content();
        assert!(
            text.contains("could not be read"),
            "the unreadable directory must be named as unsearched, got: {text}"
        );
    }

    /// cwd is always root #1, so a busy cwd filling the cap would otherwise starve every other
    /// root while the output said only "truncated", which reads as "the other folders had
    /// nothing".
    #[test]
    fn search_discloses_roots_left_unsearched_when_the_cap_fills() {
        let busy = tempfile::tempdir().expect("tempdir");
        let other = tempfile::tempdir().expect("tempdir");
        // More matches than the cap, all in the first root.
        let body = (0..MAX_INLINE_MATCHES + 20)
            .map(|_| "needle\n")
            .collect::<String>();
        std::fs::write(busy.path().join("busy.txt"), body).expect("write");
        std::fs::write(other.path().join("other.txt"), "needle\n").expect("write");

        let budget = WalkBudget::new(CancellationToken::new());
        let output = search_with_grep(
            "needle",
            &[busy.path().to_path_buf(), other.path().to_path_buf()],
            None,
            MAX_INLINE_MATCHES,
            &budget,
            &crate::workspace::ReadFence::at(crate::permission::Permission::Unrestricted),
        )
        .expect("search should return, not error");

        assert!(output.contains("truncated"), "got: {output}");
        assert!(
            output.contains("1 workspace root(s) were not searched"),
            "an unsearched root must be disclosed, not implied by absence; got: {output}"
        );
        // The note is prose the model reads and acts on, so pin it as prose: a `\`-continued string
        // literal that loses its leading-whitespace escape silently ships a run of spaces
        // mid-sentence, which no substring assertion would notice.
        assert!(
            !output.contains("  "),
            "the disclosure must not contain runs of whitespace; got: {output}"
        );
    }

    /// A root that no longer exists must not be counted into the cap-filled disclosure. It was not
    /// skipped because the cap filled, and telling the model to `path`-search it directly buys a
    /// round trip that can only answer "does not exist".
    #[test]
    fn search_does_not_blame_the_cap_for_a_stale_root() {
        let busy = tempfile::tempdir().expect("tempdir");
        let body = (0..MAX_INLINE_MATCHES + 20)
            .map(|_| "needle\n")
            .collect::<String>();
        std::fs::write(busy.path().join("busy.txt"), body).expect("write");

        let budget = WalkBudget::new(CancellationToken::new());
        let output = search_with_grep(
            "needle",
            &[
                busy.path().to_path_buf(),
                std::path::PathBuf::from("/nonexistent-workspace-root"),
            ],
            None,
            MAX_INLINE_MATCHES,
            &budget,
            &crate::workspace::ReadFence::at(crate::permission::Permission::Unrestricted),
        )
        .expect("search should return, not error");

        assert!(output.contains("truncated"), "got: {output}");
        assert!(
            !output.contains("not searched"),
            "a stale root is not a root the cap starved; got: {output}"
        );
    }

    /// A budget that expired before any root was examined says nothing about whether the path
    /// exists, so it must not report "does not exist", a definitive answer the model acts on.
    #[test]
    fn expired_budget_reports_timeout_not_missing_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("a.txt"), "needle\n").expect("write");

        let budget =
            WalkBudget::with_budget(CancellationToken::new(), std::time::Duration::from_secs(0));
        let output = search_with_grep(
            "needle",
            &[temp.path().to_path_buf()],
            None,
            MAX_INLINE_MATCHES,
            &budget,
            &crate::workspace::ReadFence::at(crate::permission::Permission::Unrestricted),
        )
        .expect("an expired budget must not be reported as a missing path");
        assert!(output.contains("still running"), "got: {output}");
    }

    #[test]
    fn search_with_grep_discloses_timeout_with_no_matches() {
        // The dangerous shape: the budget expires before anything is found, and reporting a bare
        // "No matches found." would present an unfinished search as a definitive answer.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp_dir.path().join("a.txt"), "needle\n").expect("write");

        let budget =
            WalkBudget::with_budget(CancellationToken::new(), std::time::Duration::from_secs(0));
        let output = search_with_grep(
            "needle",
            &[temp_dir.path().to_path_buf()],
            None,
            MAX_INLINE_MATCHES,
            &budget,
            &crate::workspace::ReadFence::at(crate::permission::Permission::Unrestricted),
        )
        .expect("search should return, not error");

        assert!(output.contains("No matches found."), "got: {output}");
        assert!(output.contains("incomplete"), "got: {output}");
    }

    #[test]
    fn search_file_stops_collecting_past_the_cap() {
        // A single file can hold millions of matching lines; the cap has to bound collection here
        // and not only at the end of the walk.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let content = (0..5_000).map(|_| "match\n").collect::<String>();
        let file_path = temp_dir.path().join("many.txt");
        std::fs::write(&file_path, content).expect("write");

        let matcher = grep_regex::RegexMatcherBuilder::new()
            .build("match")
            .expect("matcher");
        let mut results = Vec::new();
        let budget = WalkBudget::new(CancellationToken::new());
        search_file(
            &matcher,
            &file_path,
            &mut results,
            10,
            &budget,
            &mut Skipped::default(),
        )
        .expect("search");

        assert_eq!(
            results.len(),
            11,
            "expected the cap plus the one entry that proves more exist"
        );
    }

    /// A file with a NUL in it is binary: its lines are noise and the longest of them has no
    /// bound. It gives nothing to the results, including what matched ahead of the NUL, and the
    /// answer says a file was left out and why, so "No matches found." is not read as the whole
    /// truth.
    #[tokio::test]
    async fn a_binary_file_is_skipped_and_said() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp_dir.path().join("text.txt"), "needle in text\n").expect("write");
        std::fs::write(temp_dir.path().join("image.bin"), b"needle\0needle\n").expect("write");
        // Far enough in that the searcher has handed over the first line before it meets the NUL.
        let mut late = b"needle on the first line\n".to_vec();
        late.extend(std::iter::repeat_n(b"filler line\n".as_slice(), 20_000).flatten());
        late.push(0);
        std::fs::write(temp_dir.path().join("late.bin"), late).expect("write");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test(),
        };
        let text = tool
            .execute(
                serde_json::json!({
                    "pattern": "needle",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("the search completes")
            .text_content();
        assert!(text.contains("needle in text"), "{text}");
        assert!(
            !text.contains("image.bin") && !text.contains("late.bin"),
            "a binary file gives no matches: {text}"
        );
        assert!(text.contains("2 file(s) hold binary data"), "{text}");

        let single = tool
            .execute(
                serde_json::json!({
                    "pattern": "needle",
                    "path": temp_dir.path().join("image.bin").to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("the search completes")
            .text_content();
        assert!(
            single.contains("No matches found.") && single.contains("1 file(s) hold binary data"),
            "a binary file named as `path` says why it gave nothing: {single}"
        );
    }

    /// The searcher holds a line whole, and a run with no line terminator is one line. Past the
    /// limit the file is refused and said, where reading it would take as much memory as the run
    /// is long; what matched ahead of the run goes with it, so the file is skipped whole.
    #[tokio::test]
    async fn a_line_longer_than_the_heap_limit_is_refused_and_said() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let mut content = b"needle on the first line\n".to_vec();
        content.resize(SEARCH_HEAP_BYTES + 1024, b'x');
        std::fs::write(temp_dir.path().join("export.json"), content).expect("write");

        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test(),
        };
        let text = tool
            .execute(
                serde_json::json!({
                    "pattern": "needle",
                    "path": temp_dir.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("the search completes")
            .text_content();
        assert!(text.contains("No matches found."), "{text:.300}");
        assert!(
            text.contains("1 file(s) have a line longer than"),
            "{text:.300}"
        );
    }

    /// The budget is asked inside a file as well as between files: one large file with nothing
    /// to match is otherwise searched to its end however long that takes.
    #[test]
    fn the_budget_ends_a_search_inside_one_file() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let mut content = "nothing here\n".repeat(100_000);
        content.push_str("needle at the end\n");
        let file_path = temp_dir.path().join("large.log");
        std::fs::write(&file_path, content).expect("write");

        let matcher = grep_regex::RegexMatcherBuilder::new()
            .build("needle")
            .expect("matcher");
        let budget =
            WalkBudget::with_budget(CancellationToken::new(), std::time::Duration::from_secs(0));
        let mut results = Vec::new();
        let timed_out = search_file(
            &matcher,
            &file_path,
            &mut results,
            MAX_INLINE_MATCHES,
            &budget,
            &mut Skipped::default(),
        )
        .expect("a spent budget is a stop, not an error");
        assert!(timed_out, "the spent budget ends the file's search");
        assert!(
            results.is_empty(),
            "the search stopped before the end of the file: {results:?}"
        );

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let canceled = search_file(
            &matcher,
            &file_path,
            &mut results,
            MAX_INLINE_MATCHES,
            &WalkBudget::new(cancellation),
            &mut Skipped::default(),
        );
        assert!(
            matches!(canceled, Err(MekaError::Interrupted)),
            "a stop ends the file's search as the turn's interruption: {canceled:?}"
        );
    }

    /// The budget is asked at every read, not once per file: a stop that arrives while a file is
    /// being read ends its search there. The file is a FIFO the test feeds, so the stop is known
    /// to land after the search has read from it and before the file ends.
    #[cfg(unix)]
    #[test]
    fn a_stop_that_arrives_mid_file_ends_its_search() {
        use std::io::Write as _;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let fifo = temp_dir.path().join("stream.log");
        let named = std::ffi::CString::new(fifo.to_str().expect("path")).expect("no NUL");
        // SAFETY: `mkfifo(3)` reads a NUL-terminated path, which `named` is.
        assert_eq!(unsafe { libc::mkfifo(named.as_ptr(), 0o600) }, 0, "mkfifo");

        let cancellation = CancellationToken::new();
        let writer = std::thread::spawn({
            let fifo = fifo.clone();
            let cancellation = cancellation.clone();
            move || {
                let mut pipe = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&fifo)
                    .expect("open the FIFO");
                // More than a pipe holds, so this returns only once the search has read from it.
                pipe.write_all("nothing here\n".repeat(100_000).as_bytes())
                    .expect("the search reads what is written");
                cancellation.cancel();
                // The search may be waiting in a read it began before the stop; this ends that
                // read, so the one after it meets the stop. A search that has stopped already
                // has closed its end.
                if let Err(error) = pipe.write_all(b"needle\n") {
                    assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe, "{error}");
                }
            }
        });

        let matcher = grep_regex::RegexMatcherBuilder::new()
            .build("needle")
            .expect("matcher");
        let stopped = search_file(
            &matcher,
            &fifo,
            &mut Vec::new(),
            MAX_INLINE_MATCHES,
            &WalkBudget::new(cancellation),
            &mut Skipped::default(),
        );
        writer.join().expect("the writer");
        assert!(
            matches!(stopped, Err(MekaError::Interrupted)),
            "a read made after the stop ends the search: {stopped:?}"
        );
    }

    /// With reads confined, a search rooted outside the roots is refused by name, and one inside
    /// them finds nothing a symlink reaches outside.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_confined_search_refuses_a_base_outside_the_roots_and_skips_a_link_out() {
        let temp = tempfile::tempdir().expect("tempdir");
        let work = crate::workspace::canonical_for_test(temp.path()).join("work");
        let elsewhere = crate::workspace::canonical_for_test(temp.path()).join("elsewhere");
        std::fs::create_dir_all(&work).expect("mkdir");
        std::fs::create_dir_all(&elsewhere).expect("mkdir");
        std::fs::write(work.join("inside.txt"), "needle here").expect("write");
        std::fs::write(elsewhere.join("secret.txt"), "needle there").expect("write");
        std::os::unix::fs::symlink(elsewhere.join("secret.txt"), work.join("link.txt"))
            .expect("link");
        let tool = SearchContentsTool {
            site: crate::session::ToolSite::for_test()
                .with_cwd(crate::workspace::SharedCwd::new(work.clone()))
                .with_roots(crate::workspace::roots_for_test())
                .with_permission(crate::permission::SharedPermission::new(
                    crate::permission::Permission::Read,
                    crate::permission::EnabledPermissions::ALL,
                ))
                .with_read_boundary(crate::workspace::ReadBoundary {
                    confined: true,
                    base: Vec::new(),
                }),
        };
        let refused = tool
            .execute(
                serde_json::json!({"pattern": "needle", "path": elsewhere.to_str().expect("path")}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect_err("a base outside the roots");
        assert!(
            refused
                .to_string()
                .contains("outside the readable directories"),
            "{refused}"
        );
        let found = tool
            .execute(
                serde_json::json!({"pattern": "needle"}),
                crate::tools::ToolContext::for_test(CancellationToken::new()),
            )
            .await
            .expect("a search inside the roots")
            .text_content();
        assert!(found.contains("inside.txt"), "{found}");
        assert!(
            !found.contains("link.txt") && !found.contains("there"),
            "{found}"
        );
    }
}
