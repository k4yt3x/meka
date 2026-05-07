//! Handlers for the `agsh skill <subcommand>` CLI: list, get, show, add,
//! remove. Mirrors the structure of [`crate::mcp::cli`]: each handler
//! returns `Result<()>`, prints parseable data to stdout (the user
//! requested it; pipes / scripts read from there) and lifecycle /
//! diagnostic messages via `tracing` per the project's logging
//! guidelines.

use std::path::Path;

use crate::error::{AgshError, Result};
use crate::skills;

const DESCRIPTION_TRUNCATE: usize = 40;
const WHEN_TO_USE_TRUNCATE: usize = 40;

/// Argument bag for [`run_add`]. Borrowed so callers don't have to
/// clone every field out of the clap-derived `cli::SkillAction::Add`
/// variant.
pub struct AddArgs<'a> {
    pub name: &'a str,
    pub description: Option<&'a str>,
    pub when_to_use: Option<&'a str>,
    pub allowed_tools: &'a [String],
    pub version: Option<&'a str>,
    pub user_invocable: Option<bool>,
    pub from_file: Option<&'a Path>,
    pub force: bool,
    pub edit: bool,
}

/// `agsh skill list` — print a tab-separated table of every installed
/// skill. Empty case prints `(no skills installed)` so scripts grepping
/// the output don't get a confusing zero-byte result.
pub async fn run_list() -> Result<()> {
    let skills = skills::discover_skills();
    print_list(&skills);
    Ok(())
}

fn print_list(skills: &[skills::Skill]) {
    if skills.is_empty() {
        println!("(no skills installed)");
        return;
    }
    println!("name\tdescription\twhen_to_use\tinvocable\tpath");
    for skill in skills {
        println!(
            "{}\t{}\t{}\t{}\t{}",
            skill.name,
            truncate(&skill.description, DESCRIPTION_TRUNCATE),
            truncate(&skill.when_to_use, WHEN_TO_USE_TRUNCATE),
            skill.user_invocable,
            skill.source_dir.display(),
        );
    }
}

/// `agsh skill get <name>` — dump frontmatter as `key: value` lines.
pub async fn run_get(name: &str) -> Result<()> {
    let skill = require_skill(name)?;
    let body_bytes = std::fs::metadata(&skill.body_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("name: {}", skill.name);
    println!("source_dir: {}", skill.source_dir.display());
    println!("body_path: {}", skill.body_path.display());
    println!("description: {}", skill.description);
    println!("when_to_use: {}", skill.when_to_use);
    println!("allowed_tools: {}", skill.allowed_tools.join(", "));
    println!("version: {}", skill.version.as_deref().unwrap_or("(unset)"));
    println!("user_invocable: {}", skill.user_invocable);
    println!("body: {} bytes", body_bytes);
    Ok(())
}

/// `agsh skill show <name>` — print the rendered body with
/// `${AGSH_SKILL_DIR}` substituted. `${AGSH_SESSION_ID}` stays literal
/// because there's no active session in the CLI context.
pub async fn run_show(name: &str) -> Result<()> {
    let skill = require_skill(name)?;
    let body = skills::load_skill_body(&skill, None)
        .map_err(|error| AgshError::Config(format!("failed to load skill body: {}", error)))?;
    print!("{}", body);
    Ok(())
}

/// `agsh skill add <name> [flags]` — scaffold a new skill directory.
pub async fn run_add(args: AddArgs<'_>) -> Result<()> {
    skills::validate_skill_name(args.name).map_err(AgshError::Config)?;

    let dir = skills::skill_dir_for(args.name)
        .ok_or_else(|| AgshError::Config("could not resolve agsh config directory".to_string()))?;

    if dir.exists() {
        if !args.force {
            return Err(AgshError::Config(format!(
                "skill '{}' already exists at {}; pass --force to overwrite",
                args.name,
                dir.display()
            )));
        }
        tokio::fs::remove_dir_all(&dir).await.map_err(|error| {
            AgshError::Config(format!("failed to remove {}: {}", dir.display(), error))
        })?;
    }

    let body = build_skill_body(&args)?;

    tokio::fs::create_dir_all(&dir).await.map_err(|error| {
        AgshError::Config(format!(
            "failed to create skill dir {}: {}",
            dir.display(),
            error
        ))
    })?;

    let skill_md = dir.join("SKILL.md");
    tokio::fs::write(&skill_md, &body).await.map_err(|error| {
        AgshError::Config(format!("failed to write {}: {}", skill_md.display(), error))
    })?;

    tracing::info!("created skill '{}' at {}", args.name, skill_md.display());
    println!("{}", skill_md.display());

    if args.edit {
        if let Some(editor) = std::env::var_os("EDITOR")
            && !editor.is_empty()
        {
            let status = std::process::Command::new(&editor)
                .arg(&skill_md)
                .status()
                .map_err(|error| {
                    AgshError::Config(format!("failed to launch $EDITOR: {}", error))
                })?;
            if !status.success() {
                tracing::warn!("$EDITOR exited with non-zero status: {:?}", status.code());
            }
        } else {
            tracing::info!("--edit was requested but $EDITOR is unset; skipping");
        }
    }

    Ok(())
}

fn build_skill_body(args: &AddArgs<'_>) -> Result<String> {
    if let Some(path) = args.from_file {
        if args.description.is_some() || args.when_to_use.is_some() {
            return Err(AgshError::Config(
                "--from-file is mutually exclusive with --description / --when-to-use".to_string(),
            ));
        }
        let content = std::fs::read_to_string(path).map_err(|error| {
            AgshError::Config(format!("failed to read {}: {}", path.display(), error))
        })?;
        Ok(content)
    } else {
        let description = args.description.ok_or_else(|| {
            AgshError::Config(
                "--description is required (or pass --from-file to copy a template)".to_string(),
            )
        })?;
        let when_to_use = args.when_to_use.ok_or_else(|| {
            AgshError::Config(
                "--when-to-use is required (or pass --from-file to copy a template)".to_string(),
            )
        })?;
        Ok(skills::render_template(
            args.name,
            description,
            when_to_use,
            args.allowed_tools,
            args.version,
            args.user_invocable.unwrap_or(true),
        ))
    }
}

/// `agsh skill remove <name>` — delete the skill directory. No prompt;
/// matches `agsh mcp remove`'s convention.
pub async fn run_remove(name: &str) -> Result<()> {
    skills::validate_skill_name(name).map_err(AgshError::Config)?;
    let dir = skills::skill_dir_for(name)
        .ok_or_else(|| AgshError::Config("could not resolve agsh config directory".to_string()))?;
    if !dir.exists() {
        return Err(AgshError::Config(format!(
            "skill '{}' not found at {}",
            name,
            dir.display()
        )));
    }
    tokio::fs::remove_dir_all(&dir).await.map_err(|error| {
        AgshError::Config(format!("failed to remove {}: {}", dir.display(), error))
    })?;
    tracing::info!("removed skill '{}'", name);
    Ok(())
}

pub(crate) fn require_skill(name: &str) -> Result<skills::Skill> {
    let skills = skills::discover_skills();
    skills
        .into_iter()
        .find(|skill| skill.name == name)
        .ok_or_else(|| AgshError::Config(format!("no skill named '{}'", name)))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that mutate the global `AGSH_CONFIG_DIR` env var.
    /// Without this, tokio's parallel test runner causes one test's
    /// tempdir to be observed by another test's `discover_skills()`.
    /// `tokio::sync::Mutex` (rather than `std::sync::Mutex`) so the
    /// guard is awaitable — tests must hold it across `.await` calls.
    static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Acquire the env-lock and point `AGSH_CONFIG_DIR` at `temp`. The
    /// returned guard must be held by the caller for the lifetime of
    /// the test; dropping it releases the lock so the next test can run.
    async fn isolate_config_dir(temp: &tempfile::TempDir) -> tokio::sync::MutexGuard<'static, ()> {
        let guard = ENV_LOCK.lock().await;
        // SAFETY: the mutex makes this access exclusive across tests in
        // this process; no other code reads the var while the lock is
        // held. Matches the env-var override at `src/config.rs:462-467`.
        unsafe { std::env::set_var("AGSH_CONFIG_DIR", temp.path()) };
        guard
    }

    fn add_args<'a>(
        name: &'a str,
        description: &'a str,
        when_to_use: &'a str,
        allowed_tools: &'a [String],
    ) -> AddArgs<'a> {
        AddArgs {
            name,
            description: Some(description),
            when_to_use: Some(when_to_use),
            allowed_tools,
            version: None,
            user_invocable: None,
            from_file: None,
            force: false,
            edit: false,
        }
    }

    #[tokio::test]
    async fn test_run_add_then_discover_round_trip() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let tools = vec!["execute_command".to_string()];
        run_add(add_args("demo", "demo desc", "for testing", &tools))
            .await
            .expect("add");

        let skills = skills::discover_skills();
        assert_eq!(skills.len(), 1);
        let skill = &skills[0];
        assert_eq!(skill.name, "demo");
        assert_eq!(skill.description, "demo desc");
        assert_eq!(skill.when_to_use, "for testing");
        assert_eq!(skill.allowed_tools, vec!["execute_command".to_string()]);
        assert!(skill.user_invocable);
    }

    #[tokio::test]
    async fn test_run_add_rejects_existing_without_force() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("dup", "first", "first use", &[]))
            .await
            .expect("first add");
        let err = run_add(add_args("dup", "second", "second use", &[]))
            .await
            .expect_err("second add should fail");
        assert!(format!("{}", err).contains("already exists"));
    }

    #[tokio::test]
    async fn test_run_add_force_overwrites() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("over", "old", "old use", &[]))
            .await
            .expect("first add");
        let mut args = add_args("over", "new", "new use", &[]);
        args.force = true;
        run_add(args).await.expect("force overwrite");

        let skills = skills::discover_skills();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "new");
    }

    #[tokio::test]
    async fn test_run_add_from_file_copies_verbatim() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let template = temp.path().join("template.md");
        let body = "---\ndescription: tpl desc\nwhen_to_use: tpl use\n---\n# Templated\n\nbody.\n";
        std::fs::write(&template, body).expect("write template");

        let args = AddArgs {
            name: "tpl",
            description: None,
            when_to_use: None,
            allowed_tools: &[],
            version: None,
            user_invocable: None,
            from_file: Some(&template),
            force: false,
            edit: false,
        };
        run_add(args).await.expect("from-file add");

        let skills = skills::discover_skills();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "tpl desc");
    }

    #[tokio::test]
    async fn test_run_add_from_file_rejects_with_description_flag() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let template = temp.path().join("template.md");
        std::fs::write(&template, "---\ndescription: x\nwhen_to_use: y\n---\n").expect("write");

        let args = AddArgs {
            name: "tpl",
            description: Some("collides"),
            when_to_use: None,
            allowed_tools: &[],
            version: None,
            user_invocable: None,
            from_file: Some(&template),
            force: false,
            edit: false,
        };
        let err = run_add(args).await.expect_err("should reject");
        assert!(format!("{}", err).contains("mutually exclusive"));
    }

    #[tokio::test]
    async fn test_run_add_requires_description_without_from_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let args = AddArgs {
            name: "needs",
            description: None,
            when_to_use: Some("only when"),
            allowed_tools: &[],
            version: None,
            user_invocable: None,
            from_file: None,
            force: false,
            edit: false,
        };
        let err = run_add(args).await.expect_err("should reject");
        assert!(format!("{}", err).contains("--description is required"));
    }

    #[tokio::test]
    async fn test_run_remove_deletes_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        run_add(add_args("gone", "x", "y", &[])).await.expect("add");
        assert_eq!(skills::discover_skills().len(), 1);

        run_remove("gone").await.expect("remove");
        assert!(skills::discover_skills().is_empty());
    }

    #[tokio::test]
    async fn test_run_remove_errors_on_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let err = run_remove("ghost").await.expect_err("should error");
        assert!(format!("{}", err).contains("not found"));
    }

    #[tokio::test]
    async fn test_run_get_errors_on_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let err = run_get("ghost").await.expect_err("should error");
        assert!(format!("{}", err).contains("no skill named"));
    }

    #[tokio::test]
    async fn test_run_show_substitutes_skill_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env = isolate_config_dir(&temp).await;

        let args = AddArgs {
            name: "subst",
            description: Some("desc"),
            when_to_use: Some("use"),
            allowed_tools: &[],
            version: None,
            user_invocable: None,
            from_file: None,
            force: false,
            edit: false,
        };
        run_add(args).await.expect("add");

        // Inject a marker into the skill body that references the dir.
        let dir = skills::skill_dir_for("subst").expect("dir resolves");
        let body = "---\ndescription: x\nwhen_to_use: y\n---\nDir is ${AGSH_SKILL_DIR}\n";
        std::fs::write(dir.join("SKILL.md"), body).expect("rewrite");

        // run_show prints to stdout; we exercise the loader directly to
        // assert the substitution since capturing stdout in tests is
        // brittle.
        let skill = require_skill("subst").expect("found");
        let rendered = skills::load_skill_body(&skill, None).expect("load");
        assert!(rendered.contains(&dir.display().to_string()));
        assert!(!rendered.contains("${AGSH_SKILL_DIR}"));
    }

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 40), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let long = "a".repeat(80);
        let truncated = truncate(&long, 40);
        assert_eq!(truncated.chars().count(), 40);
        assert!(truncated.ends_with('…'));
    }
}
