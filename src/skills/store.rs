//! Writing skills to disk and removing them: the file layout, the template a new skill starts from,
//! and the refusal to touch a skill meka did not put there.
//!
//! Every refusal here names the skill and logs the path. These functions serve five doors, and two
//! of them answer someone who is not the operator: `PUT`/`DELETE /v1/skills/{name}` puts the string
//! in a 409 or 422 body, and `skill_write`/`skill_delete` hand it to the model. An absolute path
//! out of the operator's `config.toml` is a fact about someone else's host, so the sentence carries
//! the name and the `warn!` beside it carries the path.
//!
//! [`ForeignSkill`] is the exception: `meka skill` answers someone at this installation with
//! several roots configured, for whom which root is the whole answer, so that one refusal carries
//! the location as data and each door renders the audience it has.

use super::*;

/// Refuse to create or overwrite `name` because it belongs to a read-only root, or `None`.
///
/// `[skills] extra_paths` roots are scanned but never written to, and [`SkillCache::root`] only
/// ever names meka's own, so a write to a name that already resolves elsewhere would not update
/// that skill: it would put a second one in meka's store that shadows it.
///
/// One function for all five write doors (`skill_write`, `meka skill add`, `PUT /v1/skills`, and
/// through [`refuse_foreign_delete`] the two delete doors), because copies of a rule drift: checked
/// against loaded skills, every site misses a shadowed file that does not parse.
///
/// The path is carried rather than written into the sentence, so each door renders the audience it
/// has: see [`ForeignSkill`].
pub(crate) fn refuse_foreign_write(
    index: &SkillIndex,
    name: &str,
    native_root: &Path,
) -> Option<ForeignSkill> {
    Some(ForeignSkill {
        name: name.to_string(),
        source_dir: foreign_location(index, name, native_root)?,
        remedy: "edit it where it lives",
    })
}
/// The same rule for a delete, with the remedy a delete has.
pub(crate) fn refuse_foreign_delete(
    index: &SkillIndex,
    name: &str,
    native_root: &Path,
) -> Option<ForeignSkill> {
    Some(ForeignSkill {
        name: name.to_string(),
        source_dir: foreign_location(index, name, native_root)?,
        remedy: "remove it where it lives",
    })
}
/// A skill a write or delete door may not touch, because the name resolves under a root `[skills]
/// extra_paths` names.
///
/// Two renderings of one refusal, because the doors have different audiences. `meka skill` and the
/// `skill_*` tools answer someone at this installation, for whom the file's location is the
/// actionable part; they take [`Self::where_it_lives`]. `PUT` and `DELETE /v1/skills/{name}` answer
/// whoever holds a token, for whom an absolute path out of the operator's `config.toml` is a fact
/// about someone else's machine; they use `Display` and log the location themselves.
///
/// A type rather than two functions because the rule is one: [`foreign_location`] decides, and
/// nothing else may re-derive it.
pub(crate) struct ForeignSkill {
    name: String,
    pub(crate) source_dir: PathBuf,
    /// What to do about it, which differs between writing and deleting.
    remedy: &'static str,
}

impl std::fmt::Display for ForeignSkill {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "skill '{}' is in a `[skills] extra_paths` root, which meka does not write to; {}",
            self.name, self.remedy
        )
    }
}

impl ForeignSkill {
    /// The same refusal with the file's location, for a door whose reader is the person who wrote
    /// the configuration.
    pub(crate) fn where_it_lives(&self) -> String {
        format!("{self}: {}", self.source_dir.display())
    }
}
/// The refusal for a skill whose directory or file is a symlink.
///
/// [`crate::fs::reject_symlinked_path`] answers with the path, which is right for its other
/// callers and wrong for this store's; its own `warn!` is what keeps the path visible. One sentence
/// for the two levels, because which link it was is a fact about the store rather than the skill.
fn symlinked(name: &str) -> String {
    format!("skill '{name}' is a symlink, which meka refuses to follow; remove the link")
}
/// The directory `name` occupies when that directory is not meka's own to write to.
pub(super) fn foreign_location(
    index: &SkillIndex,
    name: &str,
    native_root: &Path,
) -> Option<PathBuf> {
    let (root, source_dir) = index.location(name)?;
    (root != native_root).then_some(source_dir)
}
/// Write one skill's `SKILL.md`, creating its directory if needed, and return the skill as written.
///
/// The written skill, not the requested one: a caller reports what it did, and the only honest
/// source is the bytes that reached disk, which this function already parses for the guard below.
///
/// The name is joined onto `root` here, so [`validate_skill_name`] has to run before any of it.
/// Callers validate too; this is the backstop that makes the join safe regardless.
///
/// `body: None` preserves whatever the existing file said, as `memory_write` does: a call that
/// changes only the description or the priority is one the schema invites, and rendering an absent
/// body as empty would silently delete everything the skill documented. `Some("")` empties it,
/// which renders as a bare `# <name>` heading, because a file whose body is zero bytes gives
/// `skill_read` nothing to return but the base-directory header.
///
/// Rebuilds the file from the [`Skill`] the existing one parsed to, changing only what was asked
/// for, so every frontmatter key survives a rewrite, including `metadata` entries meka has no
/// meaning for. `author` is only stamped on a skill that does not already claim one, since
/// overwriting a human's attribution loses information nothing else records.
///
/// Refuses outright when the file exists but does not parse. Such a file is in no index and no
/// listing, so neither the caller nor the model can know what is about to be overwritten.
pub(crate) fn write_skill(
    root: &Path,
    name: &str,
    description: &str,
    priority: u8,
    author: Option<&str>,
    body: Option<&str>,
) -> Result<Skill, String> {
    validate_skill_name(name)?;
    // An empty description parses back as a missing required field, so without this a write
    // succeeds and produces a skill that can never be loaded again; the length ceiling is the
    // spec's, refused here and only warned about on read.
    if let Some(problem) = description_problem(description) {
        return Err(problem);
    }

    // Held across everything below, and across processes: the collision and symlink checks read
    // the store and the write acts on what they read, so two writers that both checked before
    // either locked would both pass (`Foo/` beside `foo/`, or a link planted after the check), and
    // the read-modify-write of `meka skill add` and a turn's `skill_write` would lose one edit.
    let _store_lock = crate::fs::lock_store(root).map_err(|error| {
        tracing::warn!(
            "failed to lock the skill store at {root}: {error}",
            root = root.display()
        );
        format!("failed to lock the skill store: {error}")
    })?;

    // Read the directory rather than the discovered index: this must see what is on disk right now,
    // including a skill written since the index was built.
    if let Ok(entries) = std::fs::read_dir(root) {
        let names: Vec<String> = entries
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
            .collect();
        crate::entry::check_case_collision(name, names.iter().map(String::as_str), "skill")?;
    }

    let dir = root.join(name);
    // Whichever spelling is already there, so an edit changes the skill rather than creating a
    // second file beside it: a skill stored as `skill.md` would otherwise read as absent and be
    // clobbered.
    let skill_file = skill_file_in(&dir);
    // Both levels: a skill is a directory, so either the directory or the file inside it can be
    // the redirect. See [`crate::fs::reject_symlinked_path`], which warns with the path it refused;
    // what comes back here says which skill, for the reason the module header gives.
    crate::fs::reject_symlinked_path(&dir, "skill").map_err(|_| symlinked(name))?;
    crate::fs::reject_symlinked_path(&skill_file, "skill").map_err(|_| symlinked(name))?;

    // A read error is not "there is no file here": a `SKILL.md` in Latin-1 or at mode 000 would
    // otherwise be replaced by a stub, because `body: None` then means "there was no body".
    let existing = match std::fs::read_to_string(&skill_file) {
        Ok(content) => Some(content),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            tracing::warn!(
                "refusing to write skill '{name}': failed to read {path}: {error}",
                path = skill_file.display()
            );
            return Err(format!(
                "skill '{name}' exists but failed to read ({error}); refusing to overwrite \
                 contents meka cannot see, so repair the file or remove the skill first"
            ));
        }
    };
    let existing_skill = match existing.as_deref() {
        Some(content) => match parse_skill_definition(name, root, &dir, &skill_file, content) {
            Ok(skill) => Some(skill),
            Err(reason) => {
                tracing::warn!(
                    "refusing to write skill '{name}': {path} is not a valid skill: {reason}",
                    path = skill_file.display()
                );
                return Err(format!(
                    "skill '{name}' exists but is not a valid skill ({reason}); refusing to \
                     overwrite it"
                ));
            }
        },
        None => None,
    };
    // Refused rather than worked around: the spec says `metadata` is a map, and a file where it is
    // not leaves meka nowhere spec-legal to put `meka-priority` or `author`. Reading such a skill
    // still works; only rewriting it is refused.
    if let Some(existing) = existing_skill.as_ref()
        && existing
            .metadata
            .as_ref()
            .is_some_and(|value| !value.is_mapping())
    {
        tracing::warn!(
            "refusing to rewrite skill '{name}': {path} has a `metadata` that is not a map",
            path = skill_file.display()
        );
        return Err(format!(
            "skill '{name}' has a `metadata` that is not a map; make it the map of string to \
             string the Agent Skills spec describes"
        ));
    }

    let body = match body {
        Some(body) => body.to_string(),
        None => existing
            .as_deref()
            .and_then(|content| split_frontmatter(content).map(|(_, body)| body.to_string()))
            .unwrap_or_default(),
    };

    // Start from what the file said and change only what was asked for. Rebuilding from a fixed
    // list of fields loses every frontmatter key meka does not model.
    let mut merged = match existing_skill {
        Some(existing) => existing,
        None => Skill {
            name: name.to_string(),
            source_dir: dir.clone(),
            description: String::new(),
            license: None,
            compatibility: None,
            allowed_tools: None,
            priority,
            metadata: None,
            extra: serde_norway::Mapping::new(),
            conformance: Conformance::default(),
            body_path: skill_file.clone(),
            root: root.to_path_buf(),
        },
    };
    merged.description = description.to_string();
    merged.priority = priority;
    // The struct is about to be rendered and parsed back, and the parse-back is what every caller
    // reads. Leaving stale conformance on it would be a `Skill` that contradicts its own file.
    merged.conformance = Conformance::default();
    // Only when the skill does not already claim one, in either spelling: [`Skill::author`] rather
    // than `metadata` alone, because a hand-written file keeps its claim at the top level, and
    // reading only the nested spelling would let an agent rewriting that skill sign it.
    let claims_an_author = merged.author().is_some();
    if let Some(author) = author
        && !claims_an_author
        && let serde_norway::Value::Mapping(map) = merged
            .metadata
            .get_or_insert_with(|| serde_norway::Value::Mapping(serde_norway::Mapping::new()))
    {
        map.insert(serde_norway::Value::from(META_AUTHOR), author.into());
    }

    let rendered = render_skill_file(&merged, &body);

    // Parse the bytes about to be written, exactly as discovery will: a file the renderer got
    // wrong would otherwise report success and then be skipped by discovery forever, refused by
    // this function's own clobber guard, so the agent could not even repair it.
    let written = parse_skill_definition(name, root, &dir, &skill_file, &rendered)
        .map_err(|error| format!("refusing to write a skill that would not parse back: {error}"))?;

    // Atomic, like every other store write: `fs::write` truncates in place, so an interrupted write
    // leaves a half-file that discovery rejects and the guard above then refuses to overwrite.
    crate::fs::write_file_atomic(&skill_file, &rendered).map_err(|error| {
        tracing::warn!(
            "failed to write skill '{name}' to {path}: {error}",
            path = skill_file.display()
        );
        format!("failed to write skill '{name}': {error}")
    })?;
    Ok(written)
}
/// Delete one skill's whole directory, returning the path removed.
///
/// The directory, not just `SKILL.md`: a skill's bundled scripts and data files are part of it, and
/// leaving them behind would turn a delete into a broken half-skill that discovery keeps warning
/// about. Matches `meka skill remove`.
pub(crate) fn delete_skill(root: &Path, name: &str) -> Result<PathBuf, String> {
    // Lookup rules, not write rules. A name the spec refuses does not load, but it is still named
    // by the startup warning and by the skipped list, so the door that removes it must accept what
    // the user was just told to remove. See `validate_addressable_name`.
    validate_addressable_name(name)?;
    let dir = root.join(name);
    // Held across the check and the removal, as `write_skill` holds it across its checks and the
    // write: a link planted between the two by a sibling process would be checked under one
    // filesystem and removed under another.
    let _store_lock = crate::fs::lock_store(root).map_err(|error| {
        tracing::warn!(
            "failed to lock the skill store at {root}: {error}",
            root = root.display()
        );
        format!("failed to lock the skill store: {error}")
    })?;
    // `remove_dir_all` does not follow the link, so a symlinked entry would lose the link and keep
    // whatever it pointed at, which is not the deleted skill it would be reported as.
    crate::fs::reject_symlinked_path(&dir, "skill").map_err(|_| symlinked(name))?;
    if !dir.is_dir() {
        return Err(format!("skill '{name}' not found"));
    }
    std::fs::remove_dir_all(&dir).map_err(|error| {
        tracing::warn!(
            "failed to remove skill '{name}' at {path}: {error}",
            path = dir.display()
        );
        format!("failed to remove skill '{name}': {error}")
    })?;
    Ok(dir)
}
/// Render a complete `SKILL.md` in the shape the Agent Skills spec defines. Shared by
/// [`write_skill`] and [`render_template`] so the frontmatter key order has one owner.
///
/// Takes the whole [`Skill`] rather than a field list, which is what lets a rewrite preserve keys
/// meka does not model: [`Skill::metadata`] and [`Skill::extra`] are both emitted back out.
///
/// The frontmatter is built as a YAML mapping and handed to the serializer rather than written line
/// by line: hand-rolled quoting gets a newline in a `license` or in a metadata key wrong, and each
/// produces a file that either loses content silently or can never be written again.
///
/// Every optional key is omitted when unset, `metadata` is omitted entirely when empty, and the
/// rank is omitted at its default, so a minimal skill renders as exactly the spec's minimal
/// example: `name` and `description` and nothing else.
pub(super) fn render_skill_file(skill: &Skill, body: &str) -> String {
    use serde_norway::{Mapping, Value};

    let mut front = Mapping::new();
    front.insert("name".into(), skill.name.as_str().into());
    // Normalized, not merely quoted: the description is a one-line label everywhere it is rendered,
    // and `entry::normalize_description` is what guarantees that regardless of the file it came
    // from.
    front.insert(
        "description".into(),
        crate::entry::normalize_description(&skill.description).into(),
    );
    for (key, value) in [
        ("license", skill.license.as_deref()),
        ("compatibility", skill.compatibility.as_deref()),
        ("allowed-tools", skill.allowed_tools.as_deref()),
    ] {
        if let Some(value) = value {
            front.insert(key.into(), value.into());
        }
    }

    // The file's own `metadata`, with the rank put back. Re-inserted here rather than kept in the
    // map so `Skill::priority` is its single owner between parse and render, and appended rather
    // than sorted in so the rest of the map keeps the order its author wrote.
    //
    // Always a map or nothing: [`write_skill`] refuses a file whose `metadata` is anything else,
    // rather than growing a second arm here that writes the value back and quietly drops the rank.
    let mut metadata = match skill.metadata.clone() {
        Some(Value::Mapping(map)) => map,
        _ => Mapping::new(),
    };
    if skill.priority != crate::entry::DEFAULT_PRIORITY {
        metadata.insert(META_PRIORITY.into(), skill.priority.to_string().into());
    }
    if !metadata.is_empty() {
        front.insert("metadata".into(), Value::Mapping(metadata));
    }
    // Last, so a key the spec defines never sorts below one it does not, and so a file meka wrote
    // reads top-down as the spec's own field order. Safe against clobbering the fields above:
    // `metadata` is a named field, so `flatten` cannot route one here.
    for (key, value) in &skill.extra {
        front.insert(key.clone(), value.clone());
    }

    let mut out = String::from("---\n");
    // A serializer failure here would mean a YAML value that cannot be represented as YAML, which
    // this map cannot hold. The empty string it degrades to is caught by `write_skill`'s
    // parse-back guard rather than reaching disk.
    out.push_str(&serde_norway::to_string(&Value::Mapping(front)).unwrap_or_default());
    out.push_str("---\n\n");
    if body.trim().is_empty() {
        out.push_str(&format!("# {}\n", skill.name));
    } else {
        out.push_str(body.trim_start_matches('\n'));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}
/// Render the default `SKILL.md` template for a new skill. Optional fields are emitted only when
/// set, so the resulting file stays as minimal as the user's input.
pub(crate) fn render_template(
    name: &str,
    description: &str,
    priority: u8,
    metadata: BTreeMap<String, String>,
) -> String {
    // `--metadata key=value` can only produce strings, so the conversion is total and one-way; the
    // richer [`Skill::metadata`] type exists for values that arrive from a file. Absent rather than
    // an empty map when nothing was given, so a minimal skill renders as the spec's minimal
    // example.
    let metadata = (!metadata.is_empty()).then(|| {
        serde_norway::Value::Mapping(
            metadata
                .into_iter()
                .map(|(key, value)| (key.as_str().into(), value.as_str().into()))
                .collect(),
        )
    });
    let skill = Skill {
        name: name.to_string(),
        source_dir: PathBuf::new(),
        description: description.to_string(),
        license: None,
        compatibility: None,
        allowed_tools: None,
        priority,
        metadata,

        extra: serde_norway::Mapping::new(),
        conformance: Conformance {
            declares_name: true,
            ..Default::default()
        },
        body_path: PathBuf::new(),
        root: PathBuf::new(),
    };
    render_skill_file(
        &skill,
        &format!(
            "# {name}\n\nSkill body. Reference files bundled in this skill's directory by relative \
             path\n(e.g. `scripts/helper.sh`); they resolve against the directory this file is \
             in.\n"
        ),
    )
}
