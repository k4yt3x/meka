//! `meka project`: the work contexts a session opens, as `config.toml` defines them under
//! `[projects.<name>]`.
//!
//! Mirrors `meka profile`: the file is edited through `toml_edit` under the config lock so comments
//! and order survive, `rename` moves every row that names the project, and `remove` refuses while
//! any row still does.

use super::account::{
    open_document, place_table_after, rename_table_entry, reparse_after_edit, table_names,
};
use crate::{cli::ProjectAction, config, store::Store};

/// The sections a new `[projects.<name>]` is placed after, nearest first: the reference page puts
/// projects after `[mcp]`, the last of the multi-entry blocks.
const PLACED_AFTER: &[&str] = &["mcp", "profiles", "accounts"];

const PROJECT_COLUMNS: [crate::text::Column; 6] = [
    crate::text::Column::capped("Name", crate::text::NAME_WIDTH),
    crate::text::Column::content("Memories"),
    crate::text::Column::content("Sessions"),
    crate::text::Column::capped("Servers", crate::text::NAME_WIDTH),
    crate::text::Column::capped("Skills", crate::text::NAME_WIDTH),
    crate::text::Column::content("Configured"),
];

/// The four lists `meka project set` writes, in the order [`config::ProjectConfig`] declares them.
pub(super) const SETTABLE_PROJECT_KEYS: &[&str] = &[
    "allowed_servers",
    "disabled_servers",
    "allowed_skills",
    "disabled_skills",
];

/// The lists `project add` was given; each is written only when it names something.
#[derive(Default)]
pub(crate) struct ProjectLists {
    pub(crate) allowed_servers: Vec<String>,
    pub(crate) disabled_servers: Vec<String>,
    pub(crate) allowed_skills: Vec<String>,
    pub(crate) disabled_skills: Vec<String>,
}

pub(crate) async fn run(
    action: &ProjectAction,
    store: &Store,
    inline_max_bytes: usize,
) -> anyhow::Result<()> {
    match action {
        ProjectAction::Add {
            name,
            allowed_server,
            disabled_server,
            allowed_skill,
            disabled_skill,
        } => run_add(name, &ProjectLists {
            allowed_servers: allowed_server.clone(),
            disabled_servers: disabled_server.clone(),
            allowed_skills: allowed_skill.clone(),
            disabled_skills: disabled_skill.clone(),
        }),
        ProjectAction::List { format } => run_list(store, inline_max_bytes, *format).await,
        ProjectAction::Set {
            name,
            key,
            value,
            unset,
        } => run_set(name, key, value.as_deref(), *unset),
        ProjectAction::Rename { name, new_name } => {
            run_rename(store, inline_max_bytes, name, new_name).await
        }
        ProjectAction::Remove { name } => run_remove(store, inline_max_bytes, name).await,
    }
}

/// Refuse a document whose projects no longer pass the file's own checks: a server no
/// `[[mcp.servers]]` entry defines, or a name in both lists of a pair. Asked of the document this
/// command is about to write, so a file that fails to start is never left behind.
fn refuse_a_project_the_file_would_refuse(before: &str, after: &str) -> anyhow::Result<()> {
    let parsed = reparse_after_edit(before, after)?;
    let servers: Vec<String> = parsed
        .mcp
        .as_ref()
        .and_then(|mcp| mcp.servers.as_ref())
        .map(|servers| servers.iter().map(|server| server.name.clone()).collect())
        .unwrap_or_default();
    config::validate_project_views(&parsed.projects, servers.iter().map(String::as_str))
        .map_err(|message| anyhow::anyhow!(message))
}

/// Refuse a name that is not a project name, or that another project already has, under case
/// folding: a project is compared exactly where it is recorded, so two names a listing cannot tell
/// apart are refused at the door, as `validate_project_names` refuses them in the file.
fn refuse_a_taken_name(
    document: &toml_edit::DocumentMut,
    name: &str,
    except: Option<&str>,
) -> anyhow::Result<()> {
    crate::entry::validate_entry_name(name, "project")
        .map_err(|message| anyhow::anyhow!(message))?;
    // `except` is a rename's own name, so a rename may change nothing but its case.
    if let Some(taken) = table_names(document, "projects")
        .into_iter()
        .find(|existing| Some(existing.as_str()) != except && existing.eq_ignore_ascii_case(name))
    {
        if taken == name {
            anyhow::bail!("a project named '{name}' already exists");
        }
        anyhow::bail!("a project named '{taken}' already exists, and names differ only by case");
    }
    Ok(())
}

fn run_add(name: &str, lists: &ProjectLists) -> anyhow::Result<()> {
    // `_lock` is held to the end, so the guard and the write are one critical section.
    let (_lock, path, mut document) = open_document()?;
    refuse_a_taken_name(&document, name, None)?;
    let before = document.to_string();
    add_project_document(&mut document, name, lists)?;
    let after = document.to_string();
    refuse_a_project_the_file_would_refuse(&before, &after)?;
    crate::fs::write_file_atomic(&path, &after)?;
    tracing::info!("added project '{name}'");
    Ok(())
}

/// Insert `[projects.<name>]` into `document`, carrying whichever of `lists` name something and
/// placed after the last multi-entry block. Pure mutation so it can be unit-tested without
/// touching the filesystem.
fn add_project_document(
    document: &mut toml_edit::DocumentMut,
    name: &str,
    lists: &ProjectLists,
) -> anyhow::Result<()> {
    let placed = document.get("projects").is_none();
    let mut table = toml_edit::Table::new();
    for (key, names) in [
        ("allowed_servers", &lists.allowed_servers),
        ("disabled_servers", &lists.disabled_servers),
        ("allowed_skills", &lists.allowed_skills),
        ("disabled_skills", &lists.disabled_skills),
    ] {
        if !names.is_empty() {
            table.insert(key, toml_edit::value(names_array(names)));
        }
    }
    super::account::ensure_section_table(document, "projects")?
        .insert(name, toml_edit::Item::Table(table));
    if placed {
        place_table_after(document, "projects", name, PLACED_AFTER);
    }
    Ok(())
}

fn names_array(names: &[String]) -> toml_edit::Array {
    names.iter().map(String::as_str).collect()
}

/// The names a list value on the command line carries: comma-separated, blanks dropped.
fn list_names(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}

/// `meka project set <name> <key> <value>`, or `--unset` to drop the list.
fn run_set(name: &str, key: &str, value: Option<&str>, unset: bool) -> anyhow::Result<()> {
    if !SETTABLE_PROJECT_KEYS.contains(&key) {
        anyhow::bail!(
            "'{key}' is not a project setting (settable: {})",
            SETTABLE_PROJECT_KEYS.join(", ")
        );
    }
    // Clap's `conflicts_with` rules out "both"; this is the other half, which it cannot express.
    let names = match (value, unset) {
        (Some(value), _) => {
            let names = list_names(value);
            if names.is_empty() {
                anyhow::bail!("`{key}` needs at least one name, or `--unset` to remove the list");
            }
            Some(names)
        }
        (None, true) => None,
        (None, false) => anyhow::bail!(
            "`meka project set {name} {key}` needs a value, or `--unset` to remove the list"
        ),
    };
    // `_lock` is held to the end of the function, so the read, the check and the write are one
    // critical section.
    let (_lock, path, mut document) = open_document()?;
    let before = document.to_string();
    if !set_project_field(&mut document, name, key, names.as_deref()) {
        anyhow::bail!(crate::text::unknown_name(
            "project",
            name,
            table_names(&document, "projects")
        ));
    }
    let after = document.to_string();
    refuse_a_project_the_file_would_refuse(&before, &after)?;
    crate::fs::write_file_atomic(&path, &after)?;
    match names {
        Some(names) => tracing::info!("set {key} = {} on project '{name}'", names.join(", ")),
        None => tracing::info!("cleared {key} on project '{name}'"),
    }
    Ok(())
}

/// Write or remove one list on `[projects.<name>]`, answering whether the project was there. A
/// field-level edit, so `toml_edit` keeps the other keys and any comment beside them.
fn set_project_field(
    document: &mut toml_edit::DocumentMut,
    name: &str,
    key: &str,
    names: Option<&[String]>,
) -> bool {
    let Some(project) = document
        .get_mut("projects")
        .and_then(|item| item.as_table_like_mut())
        .and_then(|projects| projects.get_mut(name))
        .and_then(|item| item.as_table_like_mut())
    else {
        return false;
    };
    match names {
        Some(names) => {
            project.insert(key, toml_edit::value(names_array(names)));
        }
        None => {
            project.remove(key);
        }
    }
    true
}

/// One side of a project's view as a listing cell: `all`, the names allowed, or `all but` the
/// names disabled.
fn describe_view(allowed: Option<&Vec<String>>, disabled: Option<&Vec<String>>) -> String {
    let disabled: &[String] = disabled.map_or(&[], Vec::as_slice);
    match allowed {
        Some(allowed) => allowed
            .iter()
            .filter(|name| !disabled.contains(name))
            .cloned()
            .collect::<Vec<_>>()
            .join(", "),
        None if disabled.is_empty() => "all".to_string(),
        None => format!("all but {}", disabled.join(", ")),
    }
}

async fn run_list(
    store: &Store,
    inline_max_bytes: usize,
    format: crate::cli::OutputFormat,
) -> anyhow::Result<()> {
    let config_file = config::load_config_file_or_err()?;
    let memories = store.memory_store(true, inline_max_bytes);
    let views = project_views(store, &memories, &config_file.projects).await?;
    match format {
        crate::cli::OutputFormat::Json => crate::cli::write_json_listing("projects", &views)?,
        crate::cli::OutputFormat::Plain if views.is_empty() => {
            crate::streams::write_stderr_line("No projects.");
        }
        crate::cli::OutputFormat::Plain => {
            let rows: Vec<Vec<String>> = views
                .iter()
                .map(|view| {
                    vec![
                        view.name.clone(),
                        view.memories.to_string(),
                        view.sessions.to_string(),
                        describe_view(
                            view.allowed_servers.as_ref(),
                            view.disabled_servers.as_ref(),
                        ),
                        describe_view(view.allowed_skills.as_ref(), view.disabled_skills.as_ref()),
                        if view.configured { "yes" } else { "no" }.to_string(),
                    ]
                })
                .collect();
            crate::render::write_stdout(crate::text::format_table(&PROJECT_COLUMNS, &rows))?;
        }
    }
    Ok(())
}

/// Every project the file defines or a row names, with its counts and its view: the one list
/// behind `meka project list` and `GET /v1/projects`, so a project that left the file while rows
/// still name it is shown under either and the two cannot drift.
pub(crate) async fn project_views(
    store: &Store,
    memories: &crate::store::memory::MemoryStore,
    projects: &std::collections::BTreeMap<String, config::ProjectConfig>,
) -> anyhow::Result<Vec<crate::view::ProjectView>> {
    let mut names: std::collections::BTreeSet<String> = projects.keys().cloned().collect();
    names.extend(memories.project_names().await?);
    names.extend(store.session_project_names().await?);
    let mut views = Vec::with_capacity(names.len());
    for name in names {
        let config = projects.get(&name);
        views.push(crate::view::ProjectView {
            memories: memories.count_memories_on_project(&name).await?,
            sessions: store.count_sessions_on_project(&name).await?,
            configured: config.is_some(),
            allowed_servers: config.and_then(|project| project.allowed_servers.clone()),
            disabled_servers: config.and_then(|project| project.disabled_servers.clone()),
            allowed_skills: config.and_then(|project| project.allowed_skills.clone()),
            disabled_skills: config.and_then(|project| project.disabled_skills.clone()),
            name,
        });
    }
    Ok(views)
}

async fn run_rename(
    store: &Store,
    inline_max_bytes: usize,
    name: &str,
    new_name: &str,
) -> anyhow::Result<()> {
    {
        let (_lock, _path, document) = open_document()?;
        refuse_an_unrenameable_project(&document, name, new_name)?;
    }
    // Rows can name the new name though the file does not, after a hand edit; renaming onto them
    // would merge two tiers, and a memory name both hold would fail the store's constraint midway.
    // Counted outside the lock, since the store must not be awaited under it.
    let memories = store
        .memory_store(true, inline_max_bytes)
        .count_memories_on_project(new_name)
        .await?;
    let sessions = store.count_sessions_on_project(new_name).await?;
    if memories + sessions > 0 {
        anyhow::bail!(
            "{memories} memor{} and {sessions} session{} already name project '{new_name}'; choose \
             another name or move them first",
            if memories == 1 { "y" } else { "ies" },
            if sessions == 1 { "" } else { "s" }
        );
    }
    // The rows move first, and move back if the config write then fails, so a session never names
    // a project the file does not have for longer than this function runs; the order `profile
    // rename` gives.
    let (sessions, memories) = store.rename_project(name, new_name).await?;
    if let Err(error) = rename_project_under_lock(name, new_name) {
        if let Err(undo) = store.rename_project(new_name, name).await {
            tracing::warn!(
                "failed to move {sessions} session(s) and {memories} memor(ies) back to '{name}': \
                 {undo}; they are now on '{new_name}'"
            );
        }
        return Err(error);
    }
    tracing::info!(
        "renamed project '{name}' to '{new_name}'; {sessions} session(s) and {memories} \
         memor(ies) follow"
    );
    Ok(())
}

/// Refuse a rename that names no project or takes a name in use. Asked twice by [`run_rename`], of
/// the file as it stands each time, because a `project add` taking the new name can land between
/// the probe and the write.
fn refuse_an_unrenameable_project(
    document: &toml_edit::DocumentMut,
    name: &str,
    new_name: &str,
) -> anyhow::Result<()> {
    let projects = table_names(document, "projects");
    if !projects.iter().any(|project| project == name) {
        anyhow::bail!(crate::text::unknown_name("project", name, &projects));
    }
    refuse_a_taken_name(document, new_name, Some(name))
}

/// Rename `[projects.<name>]` as one critical section under the config lock.
fn rename_project_under_lock(name: &str, new_name: &str) -> anyhow::Result<()> {
    let (_lock, path, mut document) = open_document()?;
    refuse_an_unrenameable_project(&document, name, new_name)?;
    rename_table_entry(&mut document, "projects", name, new_name)?;
    crate::fs::write_file_atomic(&path, &document.to_string())?;
    Ok(())
}

async fn run_remove(store: &Store, inline_max_bytes: usize, name: &str) -> anyhow::Result<()> {
    // Counted before the lock is taken, since the counts are the refusal and need the store, which
    // must not be awaited under the config lock.
    let memories = store
        .memory_store(true, inline_max_bytes)
        .count_memories_on_project(name)
        .await?;
    let sessions = store.count_sessions_on_project(name).await?;
    if memories + sessions > 0 {
        anyhow::bail!(
            "project '{name}' has {memories} memor{} and {sessions} session{}; move or remove them \
             first",
            if memories == 1 { "y" } else { "ies" },
            if sessions == 1 { "" } else { "s" }
        );
    }
    let (_lock, path, mut document) = open_document()?;
    let removed = document
        .get_mut("projects")
        .and_then(|item| item.as_table_like_mut())
        .and_then(|projects| projects.remove(name))
        .is_some();
    if !removed {
        anyhow::bail!(crate::text::unknown_name(
            "project",
            name,
            table_names(&document, "projects")
        ));
    }
    crate::fs::write_file_atomic(&path, &document.to_string())?;
    tracing::info!("removed project '{name}'");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first project lands after the last multi-entry block, where the reference page puts the
    /// section, rather than at the end of the file; a later one joins it there.
    #[test]
    fn a_project_is_placed_after_the_mcp_block() {
        let mut document: toml_edit::DocumentMut = "[accounts.a]\nbackend = \"openai-chat-completions\"\n\n[profiles.p]\naccount = \"a\"\nmodel = \"m\"\n\n[mcp]\ndefault_permission = \"read\"\n\n[[mcp.servers]]\nname = \"fs\"\ntransport = \"stdio\"\ncommand = \"fs\"\n\n[permissions]\ndefault = \"read\"\n\n[serve]\nbind = \"127.0.0.1:1\"\n"
            .parse()
            .expect("parse");
        add_project_document(&mut document, "webdev", &ProjectLists::default()).expect("add");
        add_project_document(&mut document, "movie", &ProjectLists::default()).expect("add");
        let rendered = document.to_string();
        let position = |needle: &str| rendered.find(needle).unwrap_or_else(|| panic!("{needle}"));
        assert!(
            position("[[mcp.servers]]") < position("[projects.webdev]"),
            "{rendered}"
        );
        assert!(
            position("[projects.webdev]") < position("[projects.movie]"),
            "{rendered}"
        );
        assert!(
            position("[projects.movie]") < position("[permissions]"),
            "{rendered}"
        );
        let parsed: config::ConfigFile = toml::from_str(&rendered).expect("the file still parses");
        assert_eq!(parsed.projects.len(), 2);
    }

    /// Without an `[mcp]` block the project follows the profiles, and a file with no block to
    /// follow takes the table where `toml_edit` puts it.
    #[test]
    fn a_project_follows_the_profiles_when_there_is_no_mcp_block() {
        let mut document: toml_edit::DocumentMut =
            "[profiles.p]\naccount = \"a\"\nmodel = \"m\"\n\n[permissions]\ndefault = \"read\"\n"
                .parse()
                .expect("parse");
        add_project_document(&mut document, "webdev", &ProjectLists::default()).expect("add");
        let rendered = document.to_string();
        let position = |needle: &str| rendered.find(needle).unwrap_or_else(|| panic!("{needle}"));
        assert!(
            position("[profiles.p]") < position("[projects.webdev]"),
            "{rendered}"
        );
        assert!(
            position("[projects.webdev]") < position("[permissions]"),
            "{rendered}"
        );
        let mut bare: toml_edit::DocumentMut = "[permissions]\ndefault = \"read\"\n"
            .parse()
            .expect("parse");
        add_project_document(&mut bare, "webdev", &ProjectLists::default()).expect("add");
        assert!(bare.to_string().contains("[projects.webdev]"));
    }

    /// The lists `add` is given are written as arrays, `set` replaces or removes one, and a
    /// document whose project names a server the file does not configure is refused before it is
    /// written.
    #[test]
    fn the_lists_are_written_set_and_checked_against_the_file() {
        let mut document: toml_edit::DocumentMut =
            "[mcp]\n[[mcp.servers]]\nname = \"fs\"\ntransport = \"stdio\"\ncommand = \"fs\"\n"
                .parse()
                .expect("parse");
        let before = document.to_string();
        add_project_document(&mut document, "webdev", &ProjectLists {
            allowed_servers: vec!["fs".to_string()],
            disabled_skills: vec!["release".to_string()],
            ..ProjectLists::default()
        })
        .expect("add");
        let after = document.to_string();
        assert!(after.contains("allowed_servers = [\"fs\"]"), "{after}");
        assert!(after.contains("disabled_skills = [\"release\"]"), "{after}");
        assert!(
            !after.contains("allowed_skills"),
            "an empty list is not written: {after}"
        );
        refuse_a_project_the_file_would_refuse(&before, &after).expect("fs is configured");

        assert!(set_project_field(
            &mut document,
            "webdev",
            "allowed_servers",
            Some(&["ghost".to_string()])
        ));
        let broken = document.to_string();
        let error = refuse_a_project_the_file_would_refuse(&after, &broken).expect_err("ghost");
        assert!(error.to_string().contains("names 'ghost'"), "{error}");
        assert!(set_project_field(
            &mut document,
            "webdev",
            "allowed_servers",
            None
        ));
        assert!(!document.to_string().contains("allowed_servers"));
        assert!(!set_project_field(
            &mut document,
            "nope",
            "allowed_servers",
            None
        ));
        assert_eq!(list_names(" fs, exa ,,"), ["fs", "exa"]);
        assert_eq!(
            describe_view(
                Some(&vec!["a".to_string(), "b".to_string()]),
                Some(&vec!["b".to_string()])
            ),
            "a"
        );
        assert_eq!(
            describe_view(None, Some(&vec!["c".to_string()])),
            "all but c"
        );
        assert_eq!(describe_view(None, None), "all");
    }

    /// A name the entry rule refuses, an existing name and a case-only variant are refused before
    /// anything is written.
    #[test]
    fn add_refuses_a_bad_a_taken_and_a_case_colliding_name() {
        let document: toml_edit::DocumentMut = "[projects.Webdev]\n".parse().expect("parse");
        assert!(refuse_a_taken_name(&document, "web dev", None).is_err());
        assert!(
            refuse_a_taken_name(&document, "Webdev", None)
                .expect_err("taken")
                .to_string()
                .contains("already exists")
        );
        assert!(
            refuse_a_taken_name(&document, "webdev", None)
                .expect_err("case")
                .to_string()
                .contains("differ only by case")
        );
        refuse_a_taken_name(&document, "movie", None).expect("free");
    }
}
