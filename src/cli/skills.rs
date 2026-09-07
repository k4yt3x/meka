//! `meka skill`: list, get, show, add, remove. Parseable data goes to stdout and lifecycle and
//! diagnostics go through `tracing`, as in [`crate::cli::mcp`].

use std::{collections::BTreeMap, path::Path};

use crate::{
    config::ResolvedConfig,
    error::{MekaError, Result},
    skills,
};

const DESCRIPTION_TRUNCATE: usize = 40;

/// Attribution is free text from a file meka may not have written, and
/// [`crate::text::format_columns`] pads every column to its widest cell, so one long author would
/// indent every later column on every row.
const AUTHOR_TRUNCATE: usize = 20;

/// Argument bag for [`run_add`]. Borrowed so callers don't have to clone every field out of the
/// clap-derived `cli::SkillAction::Add` variant.
pub(crate) struct AddArgs<'a> {
    pub(crate) name: &'a str,
    pub(crate) description: Option<&'a str>,
    pub(crate) priority: Option<u8>,
    /// Raw `key=value` strings from `--metadata`, parsed by [`parse_metadata`].
    pub(crate) metadata: &'a [String],
    pub(crate) from_file: Option<&'a Path>,
    pub(crate) force: bool,
    pub(crate) edit: bool,
}

/// `/skill` in the REPL, which has no format flag: the plain table.
pub(crate) fn run_list(roots: &[std::path::PathBuf], paths: bool) -> Result<()> {
    list(roots, paths, crate::cli::OutputFormat::Plain)
}

/// `meka skill list`: a column table of every installed skill, or under `--format json` the
/// palette as `{"skills": [...]}`. The empty table is a status note on stderr, not data, so a
/// caller piping it gets a clean empty stdout.
pub(crate) fn list(
    roots: &[std::path::PathBuf],
    paths: bool,
    format: crate::cli::OutputFormat,
) -> Result<()> {
    let skills = skills::discover_skills_in_roots(roots);
    let native_root = crate::paths::skills_dir();
    if format == crate::cli::OutputFormat::Json {
        let views: Vec<crate::view::InstalledSkillView> = skills
            .skills
            .iter()
            .map(|skill| crate::view::InstalledSkillView::new(skill, native_root.as_deref()))
            .collect();
        crate::cli::write_json_listing("skills", &views)?;
        return Ok(());
    }
    print_list(&skills.skills, native_root.as_deref(), paths)?;
    Ok(())
}

/// Render the table.
///
/// The column set is *fixed*, not adaptive: this goes to stdout so it can be piped, and a column
/// that appeared only when the store happened to contain a foreign skill would silently change the
/// field offsets a script reads. `External` is therefore always present and always a boolean, even
/// in the common case where every row says `false`.
///
/// `native_root` is meka's own store, the only directory anything writes to. A skill from anywhere
/// else came from `[skills] extra_paths`, which is what `External` reports; `paths` answers the
/// question that raises, by naming where each one actually is.
///
/// Nothing else about a skill belongs here. `license`, `compatibility`, `allowed-tools`, `version`
/// and arbitrary `metadata` keys are per-skill detail, which is what `meka skill get` is for; a
/// column apiece would be mostly empty and would push `Description` off the screen.
fn print_list(skills: &[skills::Skill], native_root: Option<&Path>, paths: bool) -> Result<()> {
    if skills.is_empty() {
        crate::streams::write_stderr_line("No skills.");
        return Ok(());
    }
    crate::render::write_stdout(render_list(skills, native_root, paths))?;
    Ok(())
}

/// The table as text. Split from [`print_list`] only so a test can assert on the bytes a user
/// sees without capturing stdout; nothing else may call it, or the two would drift and the test
/// would be checking a copy of the layout rather than the layout.
fn render_list(skills: &[skills::Skill], native_root: Option<&Path>, paths: bool) -> String {
    let mut headers = vec!["Name", "Author", "Pri", "External"];
    if paths {
        headers.push("Path");
    }
    // Last, and the only unpadded column, so it is the one that may run long.
    headers.push("Description");

    let rows: Vec<Vec<String>> = skills
        .iter()
        .map(|skill| {
            let external = native_root.is_none_or(|native| skill.root != native);
            let mut row = vec![
                skill.name.clone(),
                truncate(
                    &display_metadata(skill.author().as_deref()),
                    AUTHOR_TRUNCATE,
                ),
                skill.priority.to_string(),
                external.to_string(),
            ];
            if paths {
                // Sanitized like the author cell beside it. A path is not meka's text either: its
                // last component is a directory name someone else chose, and a newline in one
                // splits this row in two, which for a table this file advertises as pipeable is a
                // fabricated record rather than a cosmetic smudge.
                row.push(display_path(&skill.source_dir));
            }
            row.push(truncate(
                &crate::memory::render_description_for_model(&skill.description),
                DESCRIPTION_TRUNCATE,
            ));
            row
        })
        .collect();

    crate::text::format_columns(&headers, &rows)
}

/// `meka skill get <name>`: the frontmatter as `key: value` lines.
///
/// Every key the file carries, not only the ones meka models: the unmodeled ones are preserved
/// across a rewrite, and this is the command that shows they were.
pub(crate) fn run_get(
    name: &str,
    roots: &[std::path::PathBuf],
    format: crate::cli::OutputFormat,
) -> Result<()> {
    let skill = skills::require_skill(name, roots).map_err(MekaError::Config)?;
    if format == crate::cli::OutputFormat::Json {
        crate::cli::write_json(&crate::view::InstalledSkillDetail::new(&skill, None))?;
        return Ok(());
    }
    let body_bytes = std::fs::metadata(&skill.body_path)
        .map(|m| m.len())
        .unwrap_or(0);
    crate::render::write_stdout_line(format!("name: {}", skill.name))?;
    // Sanitized like every other cell: a directory name is chosen by whoever put the skill on disk
    // and can carry a newline or an escape, and this line goes straight to a terminal.
    crate::render::write_stdout_line(format!("source_dir: {}", display_path(&skill.source_dir)))?;
    crate::render::write_stdout_line(format!("body_path: {}", display_path(&skill.body_path)))?;
    crate::render::write_stdout_line(format!(
        "description: {}",
        crate::memory::render_description_for_model(&skill.description)
    ))?;
    crate::render::write_stdout_line(format!("priority: {}", skill.priority))?;
    crate::render::write_stdout_line(format!("license: {}", optional(skill.license.as_deref())))?;
    crate::render::write_stdout_line(format!(
        "compatibility: {}",
        optional(skill.compatibility.as_deref())
    ))?;
    crate::render::write_stdout_line(format!(
        "allowed-tools: {}",
        optional(skill.allowed_tools.as_deref())
    ))?;
    // Both halves sanitized: a key is arbitrary YAML text from a file meka may not have written,
    // and an escape or a newline in one reaches the terminal or fakes an extra output line.
    match skill.metadata_map() {
        Some(map) => {
            for (key, value) in map {
                crate::render::write_stdout_line(format!(
                    "metadata.{}: {}",
                    crate::entry::sanitize_stored_description(&skills::yaml_value_to_string(key)),
                    crate::entry::sanitize_stored_description(&skills::yaml_value_to_string(value))
                ))?;
            }
        }
        // Whatever the file put there instead. Shown rather than skipped: meka keeps it verbatim
        // across a rewrite, so a command that hid it would hide the thing being preserved.
        None => {
            if let Some(value) = skill.metadata.as_ref() {
                crate::render::write_stdout_line(format!(
                    "metadata: {}",
                    crate::entry::sanitize_stored_description(&skills::yaml_value_to_string(value))
                ))?;
            }
        }
    }
    // Namespaced, like the `metadata.` lines above, rather than bare `key: value`. A frontmatter
    // key is chosen by whoever wrote the file, so a bare replay collided with the modeled lines: a
    // skill carrying a top-level `priority: 3` printed `priority` twice with two values, and a
    // hostile one could add a second `source_dir:` or `body:` line contradicting the real one. This
    // is stdout, which the project treats as parseable data.
    for (key, value) in &skill.extra {
        crate::render::write_stdout_line(format!(
            "extra.{}: {}",
            crate::entry::sanitize_stored_description(&skills::yaml_value_to_string(key)),
            crate::entry::sanitize_stored_description(&skills::yaml_value_to_string(value))
        ))?;
    }
    crate::render::write_stdout_line(format!("body: {body_bytes} bytes"))?;
    Ok(())
}

/// `meka skill show <name>`: print the body as the agent receives it, i.e. the base-directory
/// header followed by the body verbatim.
pub(crate) async fn run_show(
    name: &str,
    roots: &[std::path::PathBuf],
    format: crate::cli::OutputFormat,
) -> Result<()> {
    let skill = skills::require_skill(name, roots).map_err(MekaError::Config)?;
    let body = skills::load_skill_body(&skill)
        .await
        .map_err(|error| MekaError::Config(format!("failed to load skill body: {error}")))?;
    if format == crate::cli::OutputFormat::Json {
        crate::cli::write_json(&crate::view::InstalledSkillDetail::new(&skill, Some(body)))?;
        return Ok(());
    }
    crate::render::write_stdout(&body)?;
    Ok(())
}

/// `meka skill add <name> [flags]`: scaffold a new skill directory. Everything `meka skill add` can
/// refuse, run before anything is destroyed.
///
/// A *function*, not a comment marking a region: a refusal placed below the delete leaves `--force`
/// having deleted a skill it then declines to replace. A new refusal added to `run_add` has nowhere
/// to go but here, and everything here returns before the caller touches the filesystem.
///
/// Returns the directory to write and the bytes to put in it.
fn prepare_add(
    args: &AddArgs<'_>,
    roots: &[std::path::PathBuf],
) -> Result<(std::path::PathBuf, String)> {
    skills::validate_skill_name(args.name).map_err(MekaError::Config)?;

    // Resolved once and joined from, so there is one answer for a missing config directory.
    let native_root = crate::paths::skills_dir()
        .ok_or_else(|| MekaError::Config("failed to determine the config directory".to_string()))?;
    let dir = native_root.join(args.name);

    // The same refusal the agent tools and `PUT /v1/skills` apply, from the same function: a copy
    // that compared against the *loaded* skills would let a broken `SKILL.md` in a read-only root
    // be shadowed without a word.
    if let Some(refusal) = skills::refuse_foreign_write(
        &skills::discover_skills_in_roots(roots),
        args.name,
        &native_root,
    ) {
        return Err(MekaError::Config(refusal.where_it_lives()));
    }

    if dir.exists() && !args.force {
        return Err(MekaError::Config(format!(
            "skill '{}' already exists at {}; pass `--force` to overwrite",
            args.name,
            dir.display()
        )));
    }

    // On a case-insensitive filesystem `Deploy` and `deploy` are one directory, so creating the
    // second silently edits the first; on a case-sensitive one they are two skills the model cannot
    // tell apart in its index. `check_case_collision` ignores an exact match, so this does not fire
    // on the `--force` replace of the same name.
    if let Ok(entries) = std::fs::read_dir(&native_root) {
        let names: Vec<String> = entries
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
            .collect();
        crate::entry::check_case_collision(args.name, names.iter().map(String::as_str), "skill")
            .map_err(MekaError::Config)?;
    }

    // `remove_dir_all` does not follow a link, so a symlinked entry would lose the link and keep
    // whatever it pointed at. `delete_skill` and `write_skill` refuse this too, so `--force` cannot
    // quietly destroy an artifact the user planted deliberately.
    crate::fs::reject_symlinked_path(&dir, "skill").map_err(MekaError::Config)?;

    let body = build_skill_body(args)?;
    let parsed =
        skills::parse_skill_definition(args.name, &native_root, &dir, &dir.join("SKILL.md"), &body)
            .map_err(|error| {
                MekaError::Config(format!(
                    "refusing to write a skill that would not parse back: {error}"
                ))
            })?;

    // The rule `write_skill` applies, which this path does not go through: it renders its own bytes
    // and writes them directly, so it inherits none of that function's guards. Without the cap this
    // door would author a skill the reference validator rejects the moment it exists, and that
    // `skill_write` and `PUT /v1/skills` would have refused. Measured raw, like the validator does.
    if parsed.conformance.description_chars > skills::MAX_DESCRIPTION_CHARS {
        return Err(MekaError::Config(format!(
            "description is {} characters; the Agent Skills spec allows at most {}",
            parsed.conformance.description_chars,
            skills::MAX_DESCRIPTION_CHARS
        )));
    }
    // `name` is required by the spec, and every *other* write door renders `name:` from the
    // directory. This one copies bytes, and meka reads identity from the directory, so nothing
    // else would miss the key. A mismatched `name` is `parse_skill_definition`'s to refuse, above.
    //
    // Deliberately narrow: a key the spec does not define, such as a Claude Code `when_to_use`, is
    // preserved on purpose, and a missing required field is not that.
    if !parsed.conformance.declares_name {
        return Err(MekaError::Config(format!(
            "{} declares no `name`, which the Agent Skills spec requires; add `name: {}` to its \
             frontmatter",
            args.from_file
                .map(display_path)
                .unwrap_or_else(|| "the file".to_string()),
            args.name
        )));
    }

    Ok((dir, body))
}

/// `meka skill add <name> [flags]`: scaffold a new skill directory.
pub(crate) async fn run_add(args: AddArgs<'_>, roots: &[std::path::PathBuf]) -> Result<()> {
    let (dir, body) = prepare_add(&args, roots)?;

    // ---- nothing above this line has changed the filesystem ----

    // The store lock `write_skill` takes, which this path does not go through: it renders its own
    // bytes and writes them directly, so it inherited none of that function's guards.
    let native_root = roots.first().cloned().unwrap_or_else(|| dir.clone());
    let _store_lock = crate::fs::lock_store(&native_root)
        .map_err(|error| MekaError::Config(format!("failed to lock the skill store: {error}")))?;

    let skill_md = dir.join("SKILL.md");
    // The replacement lands *first*, and the old bundled files are cleared afterwards.
    //
    // Not `remove_dir_all` and then write, which is a destructive step with no guarantee anything
    // follows it. A skill directory holding a read-only subdirectory (an ordinary shape for
    // vendored data) has its `SKILL.md` unlinked and then the removal fails partway, leaving
    // nothing written, nothing to roll back to, and an error reading "failed to remove" as though
    // nothing had happened.
    //
    // `write_file_atomic` creates the parents, so no separate `create_dir_all` is needed; and
    // because it publishes by rename, the directory never holds a partial `SKILL.md`.
    crate::fs::write_file_atomic(&skill_md, &body).map_err(|error| {
        MekaError::Config(format!("failed to write {}: {}", skill_md.display(), error))
    })?;

    // `--force` on a name that exists replaces the *skill*, bundled files included, which is what
    // the docs say. Failing to clear one is a warning rather than a lost skill: the procedure the
    // user asked for is already on disk, and a leftover script beside it is a smaller problem than
    // no skill at all.
    let mut kept: Vec<String> = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(&dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry.file_name() == "SKILL.md" {
                continue;
            }
            let path = entry.path();
            let removed = match entry.file_type().await {
                // Not `remove_dir_all` on a symlinked directory: that would follow nothing but
                // would still take the link, where `remove_file` is what a link wants.
                Ok(file_type) if file_type.is_dir() => tokio::fs::remove_dir_all(&path).await,
                _ => tokio::fs::remove_file(&path).await,
            };
            if let Err(error) = removed {
                tracing::warn!("failed to remove {path}: {error}", path = path.display());
                kept.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
    }
    if !kept.is_empty() {
        // Printed, not logged: the user asked for a replacement and did not entirely get one, and
        // the exit code says success.
        kept.sort();
        crate::render::render_hint(&format!(
            "the skill was written; failed to remove {} from the previous version",
            kept.join(", ")
        ));
    }

    tracing::info!("created skill '{name}'", name = args.name);
    // Held rather than propagated: the path is the answer, but `--edit` is the rest of the command,
    // and a stdout that went away must not decide whether the editor opens.
    let written = crate::render::write_stdout_line(skill_md.display());

    // Released before the editor runs. The lock exists to protect the *write*, which is finished;
    // holding it across an interactive session blocked every other meka skill write in every other
    // process for as long as the editor stayed open, and made a concurrent `PUT /v1/skills` wait
    // 16 seconds. `lock_store`'s own justification for blocking rather than failing is that "the
    // contended window is one small file write", which is only true if this drop happens.
    drop(_store_lock);

    if args.edit {
        // Through the shared builder, which splits `$EDITOR` on whitespace: `code --wait` is an
        // ordinary setting, and passing the whole string as a program name looks for a binary
        // literally called that.
        if let Some(mut command) = crate::entry::editor_command(&skill_md) {
            let status = command.status().map_err(|error| {
                MekaError::Config(format!("failed to launch your editor: {error}"))
            })?;
            if !status.success() {
                tracing::warn!("your editor exited with {status}");
            }
        } else {
            tracing::warn!("ignoring `--edit`: neither `$VISUAL` nor `$EDITOR` is set");
        }
    }

    written?;
    Ok(())
}

fn build_skill_body(args: &AddArgs<'_>) -> Result<String> {
    if let Some(path) = args.from_file {
        // `--from-file` copies the file byte for byte, so every field that would go into rendered
        // frontmatter is refused rather than accepted and dropped. Silently ignoring `--priority`
        // here would leave the skill at the default with no indication the flag did nothing.
        if args.description.is_some() {
            return Err(MekaError::Config(
                "`--from-file` cannot be combined with `--description`".to_string(),
            ));
        }
        for (flag, given) in [
            ("--priority", args.priority.is_some()),
            ("--metadata", !args.metadata.is_empty()),
        ] {
            if given {
                return Err(MekaError::Config(format!(
                    "`--from-file` cannot be combined with `{flag}`; set the key in the file"
                )));
            }
        }
        let content = std::fs::read_to_string(path).map_err(|error| {
            MekaError::Config(format!("failed to read {}: {}", path.display(), error))
        })?;
        Ok(content)
    } else {
        let description = args.description.ok_or_else(|| {
            MekaError::Config("`--description` is required without `--from-file`".to_string())
        })?;
        let priority = args.priority.unwrap_or(crate::entry::DEFAULT_PRIORITY);
        if priority > crate::entry::MAX_PRIORITY {
            return Err(MekaError::Config(format!(
                "`--priority` must be between {} and {}",
                crate::entry::MIN_PRIORITY,
                crate::entry::MAX_PRIORITY
            )));
        }
        Ok(skills::render_template(
            args.name,
            description,
            priority,
            parse_metadata(args.metadata)?,
        ))
    }
}

/// `meka skill remove <name>`: delete the skill directory. No prompt; matches `meka mcp remove`'s
/// convention.
pub(crate) async fn run_remove(name: &str, roots: &[std::path::PathBuf]) -> Result<()> {
    // Lookup rules, not write rules. A name the spec refuses is skipped rather than listed, but
    // the startup warning names it, and a remove that refused it would leave `rm -rf` as the only
    // way out.
    skills::validate_addressable_name(name).map_err(MekaError::Config)?;

    let native_root = crate::paths::skills_dir()
        .ok_or_else(|| MekaError::Config("failed to determine the config directory".to_string()))?;
    // The same refusal every other delete door gives, from the same function. Without it this is
    // the one command that answers "not found" for a skill `meka skill list` just showed.
    if let Some(refusal) =
        skills::refuse_foreign_delete(&skills::discover_skills_in_roots(roots), name, &native_root)
    {
        return Err(MekaError::Config(refusal.where_it_lives()));
    }
    let dir = native_root.join(name);
    if !dir.exists() {
        return Err(MekaError::Config(format!(
            "no skill named '{}' in {}",
            name,
            dir.display()
        )));
    }
    // The refusal `delete_skill` gives: `remove_dir_all` does not follow the link, so removing it
    // loses the link and keeps whatever it pointed at, and reporting that as a deleted skill is a
    // lie about what happened.
    crate::fs::reject_symlinked_path(&dir, "skill").map_err(MekaError::Config)?;
    // The store lock, taken explicitly because this door does not go through `delete_skill`, the
    // way `run_add` takes it for not going through `write_skill`. Without it this could
    // `remove_dir_all` a skill directory while `skill_write` or `PUT /v1/skills` is composing and
    // renaming `SKILL.md` inside it.
    let root = native_root.clone();
    let target = dir.clone();
    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        let _store_lock = crate::fs::lock_store(&root)?;
        std::fs::remove_dir_all(&target)
    })
    .await
    .map_err(|error| MekaError::Config(format!("failed to run the removal: {error}")))?
    .map_err(|error| MekaError::Config(format!("failed to remove {}: {}", dir.display(), error)))?;
    tracing::info!("removed skill '{name}'");
    Ok(())
}

/// Parse repeated `--metadata key=value` arguments into the frontmatter's `metadata` map.
///
/// One flag rather than one per key, because the spec's `metadata` is an open map: a flag per key
/// would need a new flag every time the ecosystem settles on another convention, and could never
/// express the client-specific keys the field exists for.
///
/// Splits on the *first* `=`, so a value may contain more of them.
fn parse_metadata(pairs: &[String]) -> Result<BTreeMap<String, String>> {
    let mut metadata = BTreeMap::new();
    for pair in pairs {
        let Some((key, value)) = pair.split_once('=') else {
            return Err(MekaError::Config(format!(
                "`--metadata` takes KEY=VALUE, got '{pair}'"
            )));
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(MekaError::Config(format!(
                "`--metadata` entry '{pair}' has an empty key"
            )));
        }
        metadata.insert(key.to_string(), value.to_string());
    }
    Ok(metadata)
}

/// A metadata value for a table cell: sanitized, since these come from a file meka may not have
/// written and land straight in a terminal, with `-` for absent so the column stays aligned.
fn display_metadata(value: Option<&str>) -> String {
    value.map_or_else(
        || "-".to_string(),
        crate::entry::sanitize_stored_description,
    )
}

/// A filesystem path for display: the same sanitization every other cell gets.
///
/// A path is not meka's own text. Its leaf is a directory name chosen by whoever installed the
/// skill, so it can carry a newline, a terminal escape or a bidi override, and both callers write
/// it straight to a terminal, one of them into a column table.
fn display_path(path: &Path) -> String {
    crate::entry::sanitize_stored_description(&path.display().to_string())
}

/// A `key: value` line's value, naming the absent case rather than printing an empty string.
fn optional(value: Option<&str>) -> String {
    value.map_or_else(
        || "(unset)".to_string(),
        crate::entry::sanitize_stored_description,
    )
}

fn truncate(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

pub(crate) async fn run_skill_subcommand(
    action: &crate::cli::SkillAction,
    cli_args: &crate::cli::Cli,
) -> anyhow::Result<()> {
    // Resolved rather than defaulted: `[skills] extra_paths` decides which roots these read, and a
    // handler that ignored it would report a store the running agent does not have.
    let config = ResolvedConfig::resolve(cli_args.overrides());
    config.require_readable_config()?;
    let roots = config.skill_roots();
    match action {
        crate::cli::SkillAction::List { paths, format } => {
            crate::cli::skills::list(&roots, *paths, *format)?
        }
        crate::cli::SkillAction::Get { name, format } => {
            crate::cli::skills::run_get(name, &roots, *format)?
        }
        crate::cli::SkillAction::Show { name, format } => {
            crate::cli::skills::run_show(name, &roots, *format).await?
        }
        crate::cli::SkillAction::Add {
            name,
            description,
            priority,
            metadata,
            from_file,
            force,
            edit,
        } => {
            crate::cli::skills::run_add(
                crate::cli::skills::AddArgs {
                    name,
                    description: description.as_deref(),
                    priority: *priority,
                    metadata,
                    from_file: from_file.as_deref(),
                    force: *force,
                    edit: *edit,
                },
                &roots,
            )
            .await?
        }
        crate::cli::SkillAction::Remove { name } => {
            crate::cli::skills::run_remove(name, &roots).await?
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Holds the env-lock and clears `MEKA_CONFIG_DIR` when dropped, so the var never outlives the
    /// tempdir it points at. The `config.rs` users of the same lock unset it symmetrically; leaving
    /// it set here would mean two modules sharing one lock while disagreeing on what state they
    /// hand back to each other.
    struct ConfigDirGuard(
        #[allow(dead_code, reason = "held for its lock, never read")]
        tokio::sync::MutexGuard<'static, ()>,
    );

    impl Drop for ConfigDirGuard {
        fn drop(&mut self) {
            // SAFETY: still under the lock; the guard field is dropped after this.
            unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        }
    }

    /// Acquire the env-lock and point `MEKA_CONFIG_DIR` at `temp`. The returned guard must be held
    /// by the caller for the lifetime of the test; dropping it clears the var and releases the lock
    /// so the next test can run.
    ///
    /// Uses the crate-wide [`crate::config::CONFIG_DIR_ENV_LOCK`] rather than a lock private to
    /// this module: the var is process-global, so a per-module lock leaves these tests racing the
    /// ones in `config.rs`, and losing that race points `run_add` at the developer's real config
    /// directory.
    async fn isolate_config_dir(temp: &tempfile::TempDir) -> ConfigDirGuard {
        let guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        // SAFETY: the mutex makes this access exclusive across tests in this process; no other code
        // reads the var while the lock is held.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", temp.path()) };
        ConfigDirGuard(guard)
    }

    fn add_args<'a>(name: &'a str, description: &'a str) -> AddArgs<'a> {
        AddArgs {
            name,
            description: Some(description),
            priority: None,
            metadata: &[],
            from_file: None,
            force: false,
            edit: false,
        }
    }

    #[tokio::test]
    async fn run_add_then_discover_round_trip() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(
            add_args("demo", "demo desc"),
            &crate::paths::skill_roots(&[]),
        )
        .await
        .expect("add");

        let skills = skills::discover_skills_in_roots(&crate::paths::skill_roots(&[]));
        assert_eq!(skills.skills.len(), 1);
        let skill = &skills.skills[0];
        assert_eq!(skill.name, "demo");
        assert_eq!(skill.description, "demo desc");
    }

    /// `meka skill add` refuses an over-length description, like every other write door.
    ///
    /// This path renders its own bytes and writes them directly rather than going through
    /// `write_skill`, so it inherits none of that function's guards. Without the cap meka's own
    /// create command would author a skill the reference validator rejects on sight, and that
    /// `skill_write` and `PUT /v1/skills` would have refused outright.
    #[tokio::test]
    async fn add_refuses_a_description_the_spec_rejects() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;
        let roots = crate::paths::skill_roots(&[]);

        let boundary = "d".repeat(skills::MAX_DESCRIPTION_CHARS);
        run_add(add_args("atlimit", &boundary), &roots)
            .await
            .expect("the limit itself is allowed");

        let over = "d".repeat(skills::MAX_DESCRIPTION_CHARS + 1);
        let error = run_add(add_args("overlimit", &over), &roots)
            .await
            .expect_err("one character over must be refused");
        assert!(format!("{error}").contains("1025 characters"), "{error}");
        assert!(
            !temp.path().join("skills/overlimit").exists(),
            "a refused add must leave nothing behind"
        );

        // And through `--from-file`, which builds its bytes from a template rather than a flag.
        let template = temp.path().join("template.md");
        std::fs::write(
            &template,
            format!("---\nname: fromfile\ndescription: {over}\n---\n\nbody\n"),
        )
        .expect("write template");
        let mut args = add_args("fromfile", "");
        args.description = None;
        args.from_file = Some(&template);
        let error = run_add(args, &roots)
            .await
            .expect_err("--from-file must be held to the same rule");
        assert!(format!("{error}").contains("1025 characters"), "{error}");
    }

    /// `--from-file` may not install a skill whose `name` disagrees with its directory.
    ///
    /// Refused by `parse_skill_definition`, which every write door goes through, rather than by a
    /// rule of this command's own, which would be unreachable behind it. What this pins is that
    /// `--from-file` (the one door that copies bytes instead of rendering `name:` from the
    /// directory) is still held to it.
    #[tokio::test]
    async fn add_from_file_refuses_a_name_that_disagrees_with_the_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;
        let roots = crate::paths::skill_roots(&[]);

        let template = temp.path().join("template.md");
        std::fs::write(
            &template,
            "---\nname: something-else\ndescription: A template.\n---\n\nbody\n",
        )
        .expect("write template");
        let mut args = add_args("fromfile", "");
        args.description = None;
        args.from_file = Some(&template);
        let error = run_add(args, &roots).await.expect_err("must be refused");
        assert!(format!("{error}").contains("something-else"), "{error}");
        assert!(
            !temp.path().join("skills/fromfile").exists(),
            "a refused add must leave nothing behind"
        );

        // A template whose name agrees is fine, including one carrying a key the spec does not
        // define: preserving `when_to_use` is the point of the design, not a fault to refuse.
        for (name, body) in [
            ("agrees", "---\nname: agrees\ndescription: d\n---\n\nb\n"),
            (
                "extras",
                "---\nname: extras\ndescription: d\nwhen_to_use: whenever\n---\n\nb\n",
            ),
        ] {
            let path = temp.path().join(format!("{name}.md"));
            std::fs::write(&path, body).expect("write");
            let mut args = add_args(name, "");
            args.description = None;
            args.from_file = Some(&path);
            run_add(args, &roots)
                .await
                .unwrap_or_else(|error| panic!("'{name}' should be allowed: {error}"));
        }
    }

    /// `--from-file` may not install a skill that declares no `name` at all.
    ///
    /// The spec makes `name` required, and every other write door satisfies that by construction:
    /// they render `name:` from the directory. This one copies bytes, and meka reads identity from
    /// the directory, so a missing key cost nothing *here* and produced a file `skills-ref
    /// validate` rejects for a missing required field. meka's own store is the one place its
    /// conformance claim has to hold.
    #[tokio::test]
    async fn add_from_file_refuses_a_file_that_declares_no_name() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;
        let roots = crate::paths::skill_roots(&[]);

        let template = temp.path().join("template.md");
        std::fs::write(
            &template,
            "---\ndescription: A procedure with no declared name.\nwhen_to_use: whenever\n---\n\nb\n",
        )
        .expect("write template");
        let mut args = add_args("nameless", "");
        args.description = None;
        args.from_file = Some(&template);
        let error = run_add(args, &roots).await.expect_err("must be refused");
        let message = format!("{error}");
        assert!(message.contains("declares no `name`"), "{message}");
        assert!(
            message.contains("name: nameless"),
            "the refusal must say what to add: {message}"
        );
        assert!(
            !temp.path().join("skills/nameless").exists(),
            "a refused add must leave nothing behind"
        );
    }

    #[tokio::test]
    async fn run_add_rejects_existing_without_force() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("dup", "first"), &crate::paths::skill_roots(&[]))
            .await
            .expect("first add");
        let err = run_add(add_args("dup", "second"), &crate::paths::skill_roots(&[]))
            .await
            .expect_err("second add should fail");
        assert!(format!("{err}").contains("already exists"));
    }

    #[tokio::test]
    async fn run_add_force_overwrites() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("over", "old"), &crate::paths::skill_roots(&[]))
            .await
            .expect("first add");
        let mut args = add_args("over", "new");
        args.force = true;
        run_add(args, &crate::paths::skill_roots(&[]))
            .await
            .expect("force overwrite");

        let skills = skills::discover_skills_in_roots(&crate::paths::skill_roots(&[]));
        assert_eq!(skills.skills.len(), 1);
        assert_eq!(skills.skills[0].description, "new");
    }

    #[tokio::test]
    async fn run_add_with_metadata_round_trips() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let pairs = [
            "version=1.0".to_string(),
            "author=John Doe <john.doe@example.com>".to_string(),
        ];
        let mut args = add_args("sourced", "a sourced skill");
        args.metadata = &pairs;
        run_add(args, &crate::paths::skill_roots(&[]))
            .await
            .expect("add");

        let skill =
            skills::require_skill("sourced", &crate::paths::skill_roots(&[])).expect("found");
        assert_eq!(skill.version().as_deref(), Some("1.0"));
        assert_eq!(
            skill.author().as_deref(),
            Some("John Doe <john.doe@example.com>")
        );
    }

    #[tokio::test]
    async fn run_add_from_file_copies_verbatim() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let template = temp.path().join("template.md");
        let body = "---\nname: tpl\ndescription: tpl desc\n---\n# Templated\n\nbody.\n";
        std::fs::write(&template, body).expect("write template");

        let args = AddArgs {
            name: "tpl",
            description: None,
            priority: None,
            metadata: &[],
            from_file: Some(&template),
            force: false,
            edit: false,
        };
        run_add(args, &crate::paths::skill_roots(&[]))
            .await
            .expect("from-file add");

        let skills = skills::discover_skills_in_roots(&crate::paths::skill_roots(&[]));
        assert_eq!(skills.skills.len(), 1);
        assert_eq!(skills.skills[0].description, "tpl desc");
    }

    #[tokio::test]
    async fn run_add_from_file_rejects_with_description_flag() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let template = temp.path().join("template.md");
        std::fs::write(&template, "---\ndescription: x\n---\n").expect("write");

        let args = AddArgs {
            name: "tpl",
            description: Some("collides"),
            priority: None,
            metadata: &[],
            from_file: Some(&template),
            force: false,
            edit: false,
        };
        let err = run_add(args, &crate::paths::skill_roots(&[]))
            .await
            .expect_err("should reject");
        assert!(format!("{err}").contains("cannot be combined"));
    }

    #[tokio::test]
    async fn run_add_requires_description_without_from_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let args = AddArgs {
            name: "needs",
            description: None,
            priority: None,
            metadata: &[],
            from_file: None,
            force: false,
            edit: false,
        };
        let err = run_add(args, &crate::paths::skill_roots(&[]))
            .await
            .expect_err("should reject");
        assert!(format!("{err}").contains("is required"));
    }

    /// A refused `--force` must leave the existing skill alone.
    ///
    /// `--force` deletes the whole directory, bundled files included. Validating after that meant a
    /// rejected write destroyed a skill that was fine a moment earlier and wrote nothing in its
    /// place: the user was left with an empty directory and an error.
    #[tokio::test]
    async fn a_refused_force_add_leaves_the_existing_skill_intact() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(
            add_args("precious", "a precious procedure"),
            &crate::paths::skill_roots(&[]),
        )
        .await
        .expect("seed");
        let dir = crate::paths::skills_dir()
            .expect("skills dir")
            .join("precious");
        std::fs::create_dir_all(dir.join("scripts")).expect("mkdir");
        std::fs::write(dir.join("scripts/helper.sh"), "#!/bin/sh\n").expect("bundled file");

        // A `--from-file` whose contents are not a skill: the guard must fire.
        let template = temp.path().join("broken.md");
        std::fs::write(&template, "not a skill\n").expect("write");
        let args = AddArgs {
            name: "precious",
            description: None,
            priority: None,
            metadata: &[],
            from_file: Some(&template),
            force: true,
            edit: false,
        };
        let error = run_add(args, &crate::paths::skill_roots(&[]))
            .await
            .expect_err("must refuse");
        assert!(
            format!("{error}").contains("would not parse back"),
            "{error}"
        );

        // Everything the skill had is still there.
        let skill = skills::require_skill("precious", &crate::paths::skill_roots(&[]))
            .expect("the skill must still exist");
        assert_eq!(skill.description, "a precious procedure");
        assert!(
            dir.join("scripts/helper.sh").exists(),
            "a bundled file was destroyed by a write that never happened"
        );
    }

    /// The whole point of the check/mutate split: every refusal has to fire before anything is
    /// destroyed. The case-collision check was the second guard found sitting below the delete, so
    /// this covers the ordering itself rather than one guard.
    #[tokio::test]
    async fn every_force_refusal_leaves_the_existing_skill_intact() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;
        let roots = crate::paths::skill_roots(&[]);

        run_add(
            add_args("deploy", "a precious deployment procedure"),
            &roots,
        )
        .await
        .expect("seed");
        let dir = crate::paths::skills_dir()
            .expect("skills dir")
            .join("deploy");
        std::fs::create_dir_all(dir.join("scripts")).expect("mkdir");
        std::fs::write(dir.join("scripts/helper.sh"), "#!/bin/sh\n").expect("bundled file");

        // A directory differing only by case: `check_case_collision` refuses, and must do so before
        // the delete runs.
        //
        // Only representable on a case-sensitive filesystem. Windows and a default macOS volume
        // fold the two names onto one directory, so `create_dir_all` would silently reopen the
        // skill's own directory and the `SKILL.md` written below would overwrite the very file this
        // test asserts survives, a fixture that destroys its own subject and then fails on the
        // missing refusal. Probed rather than `cfg!`-gated, because the answer is a property of the
        // volume rather than of the operating system.
        let case_sensitive = {
            let probe = temp.path().join("Case-Sensitivity-Probe");
            std::fs::create_dir_all(&probe).expect("probe dir");
            !temp.path().join("case-sensitivity-probe").exists()
        };
        if case_sensitive {
            let collides = dir.with_file_name("Deploy");
            std::fs::create_dir_all(&collides).expect("mkdir");
            std::fs::write(
                collides.join("SKILL.md"),
                "---\nname: Deploy\ndescription: other\n---\nother\n",
            )
            .expect("seed");

            let mut args = add_args("deploy", "an updated procedure");
            args.force = true;
            let error = run_add(args, &roots).await.expect_err("must refuse");
            assert!(format!("{error}").contains("only by case"), "{error}");
        }

        let skill =
            skills::require_skill("deploy", &roots).expect("the skill must survive its refusal");
        assert_eq!(skill.description, "a precious deployment procedure");
        assert!(
            dir.join("scripts/helper.sh").exists(),
            "a bundled file was destroyed by a write that never happened"
        );
    }

    /// `remove_dir_all` does not follow a link, so removing a symlinked entry loses the link and
    /// keeps the target. The store functions refuse it; the CLI was the door that did not.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_cli_refuses_to_remove_or_replace_a_symlinked_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;
        let roots = crate::paths::skill_roots(&[]);

        let outside = temp.path().join("outside");
        std::fs::create_dir_all(&outside).expect("outside");
        let root = crate::paths::skills_dir().expect("root");
        std::fs::create_dir_all(&root).expect("root");
        std::os::unix::fs::symlink(&outside, root.join("linked")).expect("symlink");

        let error = run_remove("linked", &roots).await.expect_err("must refuse");
        assert!(format!("{error}").contains("symlink"), "{error}");
        assert!(root.join("linked").exists(), "the link was removed");

        let mut args = add_args("linked", "replacing a link");
        args.force = true;
        let error = run_add(args, &roots).await.expect_err("must refuse");
        assert!(format!("{error}").contains("symlink"), "{error}");
        assert!(outside.is_dir(), "the link target must survive");
    }

    /// `meka skill remove` is a write door and refuses a read-only root like the others, rather
    /// than reporting "not found" for a skill `meka skill list` just showed.
    #[tokio::test]
    async fn removing_a_skill_from_a_read_only_root_is_refused_with_its_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let shared = temp.path().join("shared");
        std::fs::create_dir_all(shared.join("borrowed")).expect("mkdir");
        std::fs::write(
            shared.join("borrowed/SKILL.md"),
            "---\nname: borrowed\ndescription: theirs\n---\nTHEIRS\n",
        )
        .expect("seed");
        let roots = crate::paths::skill_roots(std::slice::from_ref(&shared));

        let error = run_remove("borrowed", &roots)
            .await
            .expect_err("must refuse a foreign skill");
        let message = format!("{error}");
        assert!(message.contains("does not write to"), "{message}");
        assert!(message.contains(&shared.display().to_string()), "{message}");
        assert!(
            shared.join("borrowed/SKILL.md").exists(),
            "the foreign file must survive"
        );
    }

    /// `meka skill add` must refuse a name that already resolves to a read-only root, and must do
    /// so whether or not the file there parses.
    ///
    /// Both halves were untested, and the second was wrong: the check read the *loaded* skills, so
    /// a broken `SKILL.md` in an `extra_paths` root was a name nothing had an opinion about. The
    /// add went through, meka's own store won precedence forever after, and nothing said a word,
    /// which is the worst case to shadow silently, because the original is not reported anywhere
    /// either.
    #[tokio::test]
    async fn adding_over_a_read_only_root_is_refused_even_when_that_file_is_broken() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let shared = temp.path().join("shared");
        for (name, body) in [
            (
                "borrowed",
                "---\nname: borrowed\ndescription: theirs\n---\nTHEIRS\n",
            ),
            (
                "wrecked",
                "---\nname: wrecked\ndescription: [unclosed\n---\nTHEIRS\n",
            ),
        ] {
            std::fs::create_dir_all(shared.join(name)).expect("mkdir");
            std::fs::write(shared.join(name).join("SKILL.md"), body).expect("seed");
        }
        let roots = crate::paths::skill_roots(std::slice::from_ref(&shared));

        for name in ["borrowed", "wrecked"] {
            let error = run_add(add_args(name, "mine"), &roots)
                .await
                .expect_err("must refuse to shadow a foreign skill");
            let message = format!("{error}");
            assert!(message.contains("does not write to"), "{name}: {message}");
            assert!(
                message.contains(&shared.join(name).display().to_string()),
                "the refusal must name where the file actually is: {message}"
            );
            assert!(
                !temp.path().join("skills").join(name).exists(),
                "{name}: nothing may be created in meka's own store"
            );
        }
    }

    /// The same rule on the delete door, for a foreign file that does not parse.
    ///
    /// `run_remove` compared against the loaded skills too, so this answered "not found at
    /// <meka's own store>", pointing at a directory that was never the one holding the file.
    #[tokio::test]
    async fn removing_a_broken_skill_in_a_read_only_root_names_where_it_lives() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let shared = temp.path().join("shared");
        std::fs::create_dir_all(shared.join("wrecked")).expect("mkdir");
        std::fs::write(
            shared.join("wrecked/SKILL.md"),
            "---\nname: wrecked\ndescription: [unclosed\n---\nTHEIRS\n",
        )
        .expect("seed");
        let roots = crate::paths::skill_roots(std::slice::from_ref(&shared));

        let message = format!(
            "{}",
            run_remove("wrecked", &roots)
                .await
                .expect_err("must refuse a foreign skill")
        );
        assert!(message.contains("remove it where it lives"), "{message}");
        assert!(
            message.contains(&shared.join("wrecked").display().to_string()),
            "{message}"
        );
        assert!(shared.join("wrecked/SKILL.md").exists());
    }

    /// `meka skill get` and `meka skill show` must not deny a skill the startup warning names.
    ///
    /// Both went through one lookup that answered "no skill named 'x'" for a file that is sitting
    /// in the store and merely will not parse, which is the reading a user reaches for straight
    /// after seeing that file named on stderr.
    #[tokio::test]
    async fn showing_a_broken_skill_says_it_is_broken_rather_than_absent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let root = temp.path().join("skills");
        std::fs::create_dir_all(root.join("wrecked")).expect("mkdir");
        std::fs::write(
            root.join("wrecked/SKILL.md"),
            "---\nname: wrecked\ndescription: [unclosed\n---\nBODY\n",
        )
        .expect("seed");
        let roots = crate::paths::skill_roots(&[]);

        let broken = format!(
            "{}",
            run_get("wrecked", &roots, crate::cli::OutputFormat::Plain).expect_err("must fail")
        );
        assert!(
            broken.contains("failed to load"),
            "a present-but-unparseable file must not read as absent: {broken}"
        );
        assert!(broken.contains("invalid frontmatter"), "{broken}");

        let absent = format!(
            "{}",
            run_show("no-such-thing", &roots, crate::cli::OutputFormat::Plain)
                .await
                .expect_err("must fail")
        );
        assert_eq!(
            absent, "configuration error: no skill named 'no-such-thing'",
            "a name nobody wrote still reads as absent"
        );
    }

    #[tokio::test]
    async fn run_remove_deletes_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("gone", "x"), &crate::paths::skill_roots(&[]))
            .await
            .expect("add");
        assert_eq!(
            skills::discover_skills_in_roots(&crate::paths::skill_roots(&[]))
                .skills
                .len(),
            1
        );

        run_remove("gone", &crate::paths::skill_roots(&[]))
            .await
            .expect("remove");
        assert!(
            skills::discover_skills_in_roots(&crate::paths::skill_roots(&[]))
                .skills
                .is_empty()
        );
    }

    #[tokio::test]
    async fn run_remove_errors_on_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let err = run_remove("ghost", &crate::paths::skill_roots(&[]))
            .await
            .expect_err("should error");
        assert!(format!("{err}").contains("no skill named"));
    }

    #[tokio::test]
    async fn run_get_errors_on_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let err = run_get(
            "ghost",
            &crate::paths::skill_roots(&[]),
            crate::cli::OutputFormat::Plain,
        )
        .expect_err("should error");
        assert!(format!("{err}").contains("no skill named"));
    }

    #[tokio::test]
    async fn run_show_renders_header_and_verbatim_body() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("subst", "desc"), &crate::paths::skill_roots(&[]))
            .await
            .expect("add");

        let dir = crate::paths::skills_dir()
            .expect("skills dir")
            .join("subst");
        let body = "---\ndescription: x\n---\nRun scripts/helper.sh\nLiteral ${MEKA_SKILL_DIR}\n";
        std::fs::write(dir.join("SKILL.md"), body).expect("rewrite");

        // run_show prints to stdout; exercise the loader directly, since capturing stdout in tests
        // is brittle.
        let skill = skills::require_skill("subst", &crate::paths::skill_roots(&[])).expect("found");
        let rendered = skills::load_skill_body(&skill).await.expect("load");
        // The header names the base directory; the body itself is untouched.
        assert!(rendered.contains(&dir.display().to_string()));
        assert!(rendered.contains("Run scripts/helper.sh"));
        assert!(rendered.contains("Literal ${MEKA_SKILL_DIR}"));
    }

    /// The listing goes to stdout so it can be piped, so its columns are fixed: `External` is
    /// present and boolean whether or not any skill is external, and `--paths` is the only thing
    /// that changes the shape.
    #[test]
    fn the_listing_reports_external_skills_without_changing_shape() {
        let native = std::path::PathBuf::from("/config/skills");
        let shared = std::path::PathBuf::from("/elsewhere/skills");
        let mine = sample_skill("mine", &native, "Jane Doe");
        let theirs = sample_skill("theirs", &shared, "someone else");

        // Native only: the column is still there, and every row says so.
        let narrow = render(std::slice::from_ref(&mine), Some(&native), false);
        assert!(narrow.contains("External"), "{narrow}");
        assert!(narrow.contains("false"), "{narrow}");
        assert!(
            !narrow.contains("Version") && !narrow.contains("Path"),
            "{narrow}"
        );

        // A skill from a read-only root is flagged, without the header changing.
        let mixed = render(&[mine, theirs.clone()], Some(&native), false);
        // Fields, not bytes: `format_columns` pads to the widest cell, so adding a row legitimately
        // changes the spacing. What must not change is which columns exist, or in what order.
        assert_eq!(
            fields(&mixed),
            fields(&narrow),
            "the columns must not depend on what the store happens to hold"
        );
        assert!(
            mixed
                .lines()
                .any(|line| line.starts_with("theirs") && line.contains("true")),
            "{mixed}"
        );

        // `--paths` answers the question `External` raises, and `Description` stays last either
        // way.
        let with_paths = render(&[theirs], Some(&native), true);
        assert_eq!(
            fields(&with_paths),
            vec!["Name", "Author", "Pri", "External", "Path", "Description"],
            "--paths adds exactly one column, and Description stays last"
        );
        // Joined rather than spelled, because the rendered path uses the host separator and the
        // literal only matched on Unix.
        let expected = std::path::Path::new("/elsewhere/skills").join("theirs");
        assert!(
            with_paths.contains(&expected.display().to_string()),
            "{with_paths}"
        );
    }

    /// `format_columns` pads to the widest cell, so an untruncated author indents every other
    /// column on every row. This is the layout bug, not merely a long cell.
    #[test]
    fn a_long_author_cannot_widen_the_whole_table() {
        let native = std::path::PathBuf::from("/config/skills");
        let long = "Anthropic (claude-security plugin), ported to meka";
        let rendered = render(
            &[sample_skill("security-review", &native, long)],
            Some(&native),
            false,
        );

        assert!(
            !rendered.contains(long),
            "the author must be cut: {rendered}"
        );
        for line in rendered.lines() {
            assert!(
                line.chars().count() < 80,
                "a {}-char author widened the table to {}: {line}",
                long.chars().count(),
                line.chars().count()
            );
        }
    }

    fn sample_skill(name: &str, root: &Path, author: &str) -> skills::Skill {
        let mut map = serde_norway::Mapping::new();
        map.insert("author".into(), author.into());
        map.insert("version".into(), "1.0".into());
        let metadata = Some(serde_norway::Value::Mapping(map));
        skills::Skill {
            name: name.to_string(),
            source_dir: root.join(name),
            description: "What this skill is for.".to_string(),
            license: None,
            compatibility: None,
            allowed_tools: None,
            priority: crate::entry::DEFAULT_PRIORITY,
            metadata,

            extra: serde_norway::Mapping::new(),
            conformance: skills::Conformance {
                declares_name: true,
                ..Default::default()
            },
            body_path: root.join(name).join("SKILL.md"),
            root: root.to_path_buf(),
        }
    }

    fn render(skills: &[skills::Skill], native_root: Option<&Path>, paths: bool) -> String {
        render_list(skills, native_root, paths)
    }

    /// The header's column names, with the padding that varies by content removed.
    fn fields(table: &str) -> Vec<&str> {
        table
            .lines()
            .next()
            .expect("header")
            .split_whitespace()
            .collect()
    }

    /// The `--paths` column is a path, and a path's leaf is a directory name someone else chose.
    ///
    /// Every other cell went through the sanitizer and this one did not, so a directory whose name
    /// carried a newline split its row in two, in a table this module's own docs advertise as
    /// pipeable, which makes the extra line a fabricated record rather than a smudge.
    #[test]
    fn the_paths_column_cannot_fabricate_a_row() {
        let root = Path::new("/tmp/skills");
        let hostile = root.join("ok\nINJECTED  x  9  true");
        let mut skill = sample_skill("ok", root, "someone");
        skill.source_dir = hostile;

        let table = render(std::slice::from_ref(&skill), Some(root), true);
        assert_eq!(
            table.lines().count(),
            2,
            "one skill and one header is two lines: {table:?}"
        );
        assert!(!table.contains('\r'), "{table:?}");

        // The column is still there and still names the directory, minus what it cannot carry.
        assert!(table.contains("INJECTED"), "{table:?}");
    }

    #[test]
    fn truncate_short() {
        assert_eq!(truncate("hello", 40), "hello");
    }

    #[test]
    fn truncate_long() {
        let long = "a".repeat(80);
        let truncated = truncate(&long, 40);
        assert_eq!(truncated.chars().count(), 40);
        assert!(truncated.ends_with('…'));
    }
}
