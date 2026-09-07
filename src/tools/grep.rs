//! `search_contents` tool: ripgrep-style content search powered by the `grep-*` crates, with glob
//! filtering.
//!
//! It does **not** honor `.gitignore`, despite the name suggesting ripgrep's behavior: the walk
//! here is a hand-rolled `read_dir` traversal whose only exclusions are dotfiles, `target` and
//! `node_modules`, and the `ignore` crate is not a dependency. Only the matcher comes from the
//! `grep-*` family.

use async_trait::async_trait;

use super::{
    Tool, ToolOutput,
    util::{WalkBudget, WalkStop, redirects_to_scratchpad, require_str},
};
use crate::{
    error::{MekaError, Result},
    permission::Permission,
    provider::ToolDefinition,
};

/// Inline match cap when the agent isn't redirecting to the scratchpad, and the ceiling an explicit
/// `limit` is clamped to. Single source of truth for the description and the runtime cap.
const MAX_INLINE_MATCHES: usize = 100;

pub(super) struct SearchContentsTool {
    pub(crate) site: crate::session::ToolSite,
}

#[async_trait]
impl Tool for SearchContentsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_contents".to_string(),
            description: format!(
                "Search file contents using a regex pattern (powered by ripgrep). \
                 Avoid overly broad searches: scanning a large tree is slow \
                 and will hit many directories the user has no read permission \
                 for, producing noisy errors. Start with the smallest `path` \
                 and a tight `glob` filter that plausibly contains the match; \
                 if that returns nothing, widen the `path` by one level or \
                 loosen the `glob`, and repeat. Only fall back to a tree-wide \
                 scan if targeted attempts have all failed. Inline results are \
                 capped at {MAX_INLINE_MATCHES} matches; pass `limit` for fewer, or the \
                 `scratchpad` parameter to collect an unbounded result set. Multiple independent \
                 search_contents calls in one assistant message run in parallel.",
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
                        "description": "File or directory to search in. Omit to search every workspace root (the working directory plus any additional roots listed in the environment context). Set it to narrow to the smallest subtree that can answer the question."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.rs'). Strongly recommended when searching directories to avoid scanning unrelated files."
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
        let pattern = require_str(&input, "pattern", "search_contents")?;
        // An explicit `path` searches exactly that tree, resolved against the per-session cwd. With
        // no `path`, sweep every workspace root: in a multi-root ACP workspace, searching only
        // `cwd` silently misses whole folders the user can see in their editor. Carried as
        // `PathBuf` end to end. Rendering each root through `to_string_lossy` and rebuilding it
        // with `Path::new` replaced every non-UTF-8 byte with U+FFFD, so a working directory whose
        // name is not valid UTF-8 -- `mkdir $'proj\xff'` -- named a directory that does not exist,
        // and the tool reported the user's own cwd as missing under a spelling they never typed.
        let search_paths: Vec<std::path::PathBuf> = match input["path"].as_str() {
            Some(raw) => vec![crate::workspace::resolve_against_cwd(&self.site.cwd, raw)],
            None => crate::workspace::search_roots(&self.site.cwd, &self.site.roots),
        };
        let file_glob = input["glob"].as_str().map(|s| s.to_string());
        // Cap precedence, as `find_files` has it: an explicit `limit` wins, clamped to the inline
        // cap the way `memory_search` clamps its own; with none, `scratchpad` lifts the cap so the
        // agent can collect an unbounded result set.
        let max_results = match input.get("limit").and_then(serde_json::Value::as_u64) {
            Some(limit) => usize::try_from(limit)
                .unwrap_or(usize::MAX)
                .clamp(1, MAX_INLINE_MATCHES),
            None if redirects_to_scratchpad(&input) => usize::MAX,
            None => MAX_INLINE_MATCHES,
        };

        // One budget for the whole call, not one per root: `WalkBudget::new` stamps its deadline at
        // construction, so a per-root budget would silently multiply the ceiling by the root count.
        let budget = WalkBudget::new(cancellation.clone());
        let private = crate::workspace::private_directories_hidden_at(self.site.permission.get());
        let search = tokio::task::spawn_blocking(move || {
            search_with_grep(
                &pattern,
                &search_paths,
                file_glob.as_deref(),
                max_results,
                &budget,
                &private,
            )
        });

        // Race the search against the token: a started `spawn_blocking` task cannot be aborted, so
        // awaiting it unconditionally would let a walk rooted high in the tree hold the turn open.
        // The search checks the same token per directory entry and stops on its own.
        let result = tokio::select! {
            joined = search => joined.map_err(|error| MekaError::ToolExecution {
                tool_name: "search_contents".to_string(),
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
    // meka's own directories at this permission, which a named root may not be inside and the
    // walk steps around; see `crate::workspace::private_read_refusal`.
    private: &[std::path::PathBuf],
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
            tool_name: "search_contents".to_string(),
            message: format!("invalid or oversized regex '{pattern}': {error}"),
        })?;

    let mut results = Vec::new();
    let mut timed_out = false;
    // A root that doesn't exist is skipped rather than fatal, because one stale entry in a
    // multi-root workspace shouldn't sink a search the other roots can answer. With a single
    // explicit `path` this reduces to today's behavior exactly: nothing existed, so the error
    // below fires with the same message.
    let mut searched_any = false;
    // Compiled lazily and at most once. Deliberately inside the directory branch rather than
    // hoisted above the loop: hoisting would report an invalid `glob` for a path that doesn't
    // exist, or for a single file where the glob is irrelevant, changing single-root behavior.
    let mut glob_pattern: Option<glob::Pattern> = None;
    // Roots left unsearched because the match cap filled up first. cwd is always root #1, so a
    // busy cwd would otherwise starve every other root and report only "truncated", which reads as
    // "the other folders had nothing" -- the exact failure multi-root support exists to prevent.
    let mut unsearched_roots = 0usize;
    let mut unreadable = 0usize;

    for search_path in search_paths {
        // Checked per root as well as inside `walk_directory`: a root that is a plain file, or that
        // doesn't exist, never reaches the walk, so a long list of them would advance without
        // consulting the budget once and ignore both the deadline and a `session/cancel`.
        match budget.check() {
            Some(WalkStop::Cancelled) => return Err(MekaError::Interrupted),
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
        // at the store is the question, and a silent empty answer would read as "nothing there".
        if crate::workspace::resolves_into_private(path, private) {
            return Err(MekaError::ToolExecution {
                tool_name: "search_contents".to_string(),
                message: format!(
                    "'{}' is inside meka's own directories, which only `unrestricted` reads.",
                    path.display()
                ),
            });
        }

        if path.is_file() {
            searched_any = true;
            search_file(&matcher, path, &mut results, max_results, &mut unreadable)?;
        } else if path.is_dir() {
            searched_any = true;
            if glob_pattern.is_none()
                && let Some(g) = file_glob
            {
                glob_pattern =
                    Some(
                        glob::Pattern::new(g).map_err(|error| MekaError::ToolExecution {
                            tool_name: "search_contents".to_string(),
                            message: format!("invalid glob pattern '{g}': {error}"),
                        })?,
                    );
            }
            let scope = SearchScope {
                matcher: &matcher,
                glob_pattern: &glob_pattern,
                max_results,
                budget,
                private,
            };
            if walk_directory(path, &scope, &mut results, &mut unreadable)? {
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
            tool_name: "search_contents".to_string(),
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

/// Search one file, stopping once `results` holds one entry more than `max_results`: a single file
/// can hold millions of matching lines, so the cap has to bound collection here too, not only at
/// the end of the walk.
fn search_file(
    matcher: &grep_regex::RegexMatcher,
    path: &std::path::Path,
    results: &mut Vec<String>,
    max_results: usize,
    // Counted like an unreadable directory: a file that cannot be opened or searched may hold the
    // match, and dropping it at `debug!` let "No matches found." read as definitive.
    unreadable: &mut usize,
) -> Result<()> {
    use grep_searcher::{Searcher, sinks::UTF8};

    let mut searcher = Searcher::new();
    if let Err(error) = searcher.search_path(
        matcher,
        path,
        UTF8(|line_number, line| {
            results.push(format!(
                "{}:{}:{}",
                path.display(),
                line_number,
                line.trim_end()
            ));
            Ok(results.len() <= max_results)
        }),
    ) {
        let path = path.display();
        tracing::debug!("failed to search {path}: {error}");
        *unreadable += 1;
    }

    Ok(())
}

/// What one search is judged against, the same for every root and every directory under it.
struct SearchScope<'a> {
    matcher: &'a grep_regex::RegexMatcher,
    glob_pattern: &'a Option<glob::Pattern>,
    max_results: usize,
    budget: &'a WalkBudget,
    /// meka's own directories at this permission, which the walk steps around; see
    /// `crate::workspace::private_read_refusal`.
    private: &'a [std::path::PathBuf],
}

/// Walk `directory`, searching every file that passes the scope's glob. Returns whether the walk
/// was stopped by the time budget; errors with [`MekaError::Interrupted`] when the turn was
/// canceled.
///
/// `unreadable` counts the directories the walk could not open and the files it could not search,
/// so the caller can say so. A silent skip turns `search_contents` over a tree with an unreadable
/// subdirectory into a confident "No matches found.", which is the definitive-sounding wrong answer
/// the truncation and timeout notices already exist to prevent. `find_files` has reported this all
/// along.
fn walk_directory(
    directory: &std::path::Path,
    scope: &SearchScope<'_>,
    results: &mut Vec<String>,
    unreadable: &mut usize,
) -> Result<bool> {
    let SearchScope {
        matcher,
        glob_pattern,
        max_results,
        budget,
        private,
    } = *scope;
    // Iterative traversal via an explicit work-stack: a recursive walk would overflow the call
    // stack on a pathologically deep directory tree.
    let mut pending: Vec<std::path::PathBuf> = vec![directory.to_path_buf()];

    while let Some(dir) = pending.pop() {
        // Stepped around, like the dot-directories below: a root above meka's directories is a
        // legitimate workspace, and what lies inside them is not this walk's to read. Resolved
        // per directory, since the root itself need not be canonical.
        if crate::workspace::resolves_into_private(&dir, private) {
            continue;
        }
        // Checked here as well as per entry: a run of directories that all fail `read_dir` (a tree
        // the user has no permission for) never reaches the inner loop, and would otherwise grind
        // through the whole work-stack without consulting the budget once.
        match budget.check() {
            Some(WalkStop::Cancelled) => return Err(MekaError::Interrupted),
            Some(WalkStop::TimedOut) => return Ok(true),
            None => {}
        }

        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) => {
                let directory = dir.display();
                tracing::debug!("search_contents: cannot read '{directory}': {error}");
                *unreadable += 1;
                continue;
            }
        };

        for entry in entries {
            match budget.check() {
                Some(WalkStop::Cancelled) => return Err(MekaError::Interrupted),
                Some(WalkStop::TimedOut) => return Ok(true),
                None => {}
            }

            let Ok(entry) = entry else { continue };
            let path = entry.path();

            // `to_string_lossy`, not `to_str().unwrap_or("")`. A directory whose name is not
            // valid UTF-8 made `to_str` yield `None` and the fallback yield `""`, which does not
            // start with `.` -- so `.cache\xff` was the one shape that walked straight past the
            // skip this exists for. Lossy conversion never alters ASCII bytes, so the leading dot
            // and both literals below survive it intact.
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
                // meka's own; a regular file cannot, since the walk never descends a symlinked
                // directory and every directory it does enter was judged above.
                if file_type.is_symlink() && crate::workspace::resolves_into_private(&path, private)
                {
                    continue;
                }
                search_file(matcher, &path, results, max_results, unreadable)?;
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

    /// `search_contents` below `unrestricted` refuses a `path` inside meka's own directories, and a
    /// walk from a root above them neither enters them nor follows a symlink into them. The
    /// database is searched as bytes, so a pattern for a key prefix pulled runs out of `meka.db`.
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        let walked = tool
            .execute(
                serde_json::json!({
                    "pattern": "sk-",
                    "path": home.path().to_str().expect("path")
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = named.expect_err("a root inside the store is refused");
        assert!(error.to_string().contains("meka's own"), "{error}");
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(result.text_content().contains("hello world"));
        assert!(result.text_content().contains("hello again"));
    }

    /// The counterpart to `find_files`' nested-root test, pinning why the two tools use different
    /// root sets. `search_contents` descends, so `search_roots` pruning `cwd` in favor of an
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
                    crate::tools::ToolContext::detached(CancellationToken::new()),
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
    /// precedence `find_files` documents for its own `limit`.
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
                crate::tools::ToolContext::detached(cancellation),
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
                crate::tools::ToolContext::detached(CancellationToken::new()),
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
    /// root while the output said only "truncated" -- which reads as "the other folders had
    /// nothing", the exact failure multi-root support exists to prevent.
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
            &[],
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
            &[],
        )
        .expect("search should return, not error");

        assert!(output.contains("truncated"), "got: {output}");
        assert!(
            !output.contains("not searched"),
            "a stale root is not a root the cap starved; got: {output}"
        );
    }

    /// A budget that expired before any root was examined says nothing about whether the path
    /// exists, so it must not report "does not exist" -- a definitive answer the model acts on.
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
            &[],
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
            &[],
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
        search_file(&matcher, &file_path, &mut results, 10, &mut 0).expect("search");

        assert_eq!(
            results.len(),
            11,
            "expected the cap plus the one entry that proves more exist"
        );
    }
}
