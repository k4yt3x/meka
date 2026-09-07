//! Writing skills to disk and removing them: the file layout, the template a new skill starts from,
//! and the refusal to touch a skill meka did not put there.
//!
//! **Every refusal here names the skill and logs the path.** These functions serve five doors, and
//! two of them answer someone who is not the operator: `PUT`/`DELETE /v1/skills/{name}` puts the
//! string in a 409 or 422 body, and `skill_write`/`skill_delete` hand it to the model. An absolute
//! path out of the operator's `config.toml` is no use to either and is a fact about the host they
//! are running against, so the sentence carries the name and the reason class and the `warn!`
//! beside it carries the path. The operator reads that at the default log level; nothing is lost,
//! only moved.
//!
//! [`ForeignSkill`] is the exception, and it proves the rule rather than bending it: `meka skill`
//! answers someone standing at this installation with several roots configured, for whom *which*
//! root is the whole answer, so that one refusal carries the location as data and each door renders
//! the audience it has.

use super::*;

/// Refuse to create or overwrite `name` because it belongs to a read-only root, or `None`.
///
/// `[skills] extra_paths` roots are scanned but never written to, and [`SkillCache::root`] only
/// ever names meka's own. So a write to a name that already resolves elsewhere does not update that
/// skill: it puts a second one in meka's store which *shadows* it. The caller believes it refined a
/// procedure; it forked one, and the original keeps being the file every other client reads.
///
/// One function for all five write doors -- `skill_write`, `meka skill add`, `PUT /v1/skills`, and
/// through [`refuse_foreign_delete`] the two delete doors -- because they were five copies of one
/// rule with five message strings, and copies of a rule drift. That is not hypothetical: the check
/// was written against loaded skills at every site, so every site had the same blind spot for a
/// shadowed file that does not parse.
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
        remedy: "writing here would create a second copy that shadows the original rather than \
                 changing it. Use a different name, or edit that file where it lives.",
    })
}
/// The same rule for a delete, which has a different remedy: there is no "use another name" for
/// removing something, only removing it where it lives.
pub(crate) fn refuse_foreign_delete(
    index: &SkillIndex,
    name: &str,
    native_root: &Path,
) -> Option<ForeignSkill> {
    Some(ForeignSkill {
        name: name.to_string(),
        source_dir: foreign_location(index, name, native_root)?,
        remedy: "meka does not delete files there. Remove it where it lives.",
    })
}
/// A skill a write or delete door may not touch, because the name resolves under a root `[skills]
/// extra_paths` names.
///
/// **Two renderings of one refusal, because the doors have different audiences.** `meka skill` and
/// the `skill_*` tools answer someone sitting at this installation, for whom the file's location is
/// the actionable part and who may have several roots configured; they take
/// [`Self::where_it_lives`].
/// `PUT` and `DELETE /v1/skills/{name}` answer whoever holds a token, for whom an absolute path out
/// of the operator's `config.toml` is a fact about someone else's machine; they use `Display` and
/// log the location themselves, since the door that drops a detail is the one that knows it did.
///
/// A type rather than two functions because the *rule* is one: [`foreign_location`] decides, and
/// nothing else may re-derive it.
pub(crate) struct ForeignSkill {
    name: String,
    pub(crate) source_dir: PathBuf,
    /// What to do about it, which differs between writing and deleting: there is no "use another
    /// name" for removing something.
    remedy: &'static str,
}

impl std::fmt::Display for ForeignSkill {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "skill '{}' comes from a root [skills] extra_paths names, which meka reads but does \
             not write to; {}",
            self.name, self.remedy
        )
    }
}

impl ForeignSkill {
    /// The same refusal with the file's location, for a door whose reader is the person who wrote
    /// the configuration.
    pub(crate) fn where_it_lives(&self) -> String {
        format!("{self} It lives at {}.", self.source_dir.display())
    }
}
/// The refusal for a skill whose directory or file is a symlink.
///
/// [`crate::fs::reject_symlinked_path`] answers with the path, which is right for its other
/// callers and wrong for this store's; its own `warn!` is what keeps the path visible. One sentence
/// for the two levels, because "which of the two links it was" is a fact about the store rather
/// than about the skill, and both have the same remedy.
fn symlinked(name: &str) -> String {
    format!(
        "skill '{name}' is a symlink; refusing to write through it, because it could leave the \
         store meka owns. Remove the link, or use a different name."
    )
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
/// The *written* skill, not the requested one. A caller reports what it did, and the only honest
/// source is the bytes that reached disk, which this function already parses for the guard below.
/// It cannot always record what it was asked to (see [`render_skill_file`] on a `metadata` it may
/// not replace), so a caller echoing its own arguments would eventually report "priority 2" onto a
/// file that says 5.
///
/// The agent-facing counterpart to `meka skill add`, and the reason it is a store function rather
/// than living in the tool: the name is joined onto `root` here, so [`validate_skill_name`] has to
/// run before any of it. Callers validate too; this is the backstop that makes the join safe
/// regardless.
///
/// `body: None` preserves whatever the existing file said. That asymmetry is deliberate and mirrors
/// `memory_write`: a call that changes only the description or the priority is one the schema
/// invites, and rendering an absent body as empty would silently delete everything the skill
/// documented on exactly that call.
///
/// `Some("")` empties it, which renders as a bare `# <name>` heading rather than nothing at all:
/// unlike a memory, a skill *is* its body, and a file whose body is zero bytes gives `skill_read`
/// nothing to return but the base-directory header.
///
/// Rebuilds the file from the [`Skill`] the existing one parsed to, changing only what was asked
/// for, so every frontmatter key survives a rewrite: `license`, `compatibility`, `allowed-tools`
/// and every `metadata` entry, including ones meka has no meaning for. `author` is therefore only
/// stamped on a skill that does not already claim one, since overwriting a human's attribution
/// because an agent edited their file loses information nothing else records.
///
/// Refuses outright when the file exists but does not parse. Such a file is invisible everywhere
/// else in meka (discovery skips it with a warning, so it is in no index and no listing), which
/// means neither the caller nor the model can know what is about to be overwritten. Clobbering it
/// destroys content whose only copy is that file, and the caller can always pick another name.
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

    // Held across everything below, and across processes. The collision and symlink checks read
    // the store and the write acts on what they read, so two writers that both checked before
    // either locked both passed: `Foo/` beside `foo/`, or a link planted after the check and
    // followed by the write. Nothing serialized the read-modify-write either, so `meka skill add`
    // in a shell and a turn's `skill_write` each kept their own view and the loser's edit vanished.
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
    // second file beside it. Hardcoding `SKILL.md` meant a skill stored as `skill.md` was read as
    // absent: the clobber guard never fired, the body defaulted to empty, and the rewrite reported
    // that it had kept a body it had just replaced with a bare heading.
    let skill_file = skill_file_in(&dir);
    // Both levels: a skill is a directory, so either the directory or the file inside it can be
    // the redirect. See [`crate::fs::reject_symlinked_path`], which warns with the path it refused;
    // what comes back here says which skill, for the reason the module header gives.
    crate::fs::reject_symlinked_path(&dir, "skill").map_err(|_| symlinked(name))?;
    crate::fs::reject_symlinked_path(&skill_file, "skill").map_err(|_| symlinked(name))?;

    // `read_to_string(...).ok()` collapsed every read *error* into "there is no file here", so the
    // clobber guard below -- which only ever saw files that decoded -- never fired for one that did
    // not. A `SKILL.md` in Latin-1 (an ordinary editor artifact) or at mode 000 was replaced by a
    // five-line stub and the write reported success, because `body: None` then means "there was no
    // body" rather than "the body could not be read". Distinguished here so both cases refuse.
    let existing = match std::fs::read_to_string(&skill_file) {
        Ok(content) => Some(content),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            tracing::warn!(
                "refusing to write skill '{name}': {path} could not be read: {error}",
                path = skill_file.display()
            );
            return Err(format!(
                "skill '{name}' exists but could not be read ({error}), so a write would replace \
                 contents meka cannot see. Repair the file directly, or remove the skill first if \
                 you mean to start over."
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
                     overwrite it. Fix or remove that file, or use a different name."
                ));
            }
        },
        None => None,
    };
    // Refused rather than worked around. The spec says `metadata` is an object; a file where it is
    // a string or a list is a typo, not a shape another client produces. Carrying on regardless
    // meant meka had nowhere spec-legal to put `meka-priority` or `author`, and rather than say so
    // it grew a branch in the renderer, another in the author stamp, a gate in `take_priority`, and
    // a line in `skill_write`'s confirmation explaining to the *model* why the rank it asked for
    // did not apply -- four places quietly doing something other than what was asked, for an input
    // nobody writes. One refusal naming the fix costs the user one edit and costs the code nothing.
    //
    // Reading such a skill still works: discovery warns, the value round-trips verbatim, and
    // `meka skill get` shows it. Only rewriting it is refused.
    if let Some(existing) = existing_skill.as_ref()
        && existing
            .metadata
            .as_ref()
            .is_some_and(|value| !value.is_mapping())
    {
        tracing::warn!(
            "refusing to rewrite skill '{name}': {path} has a 'metadata' that is not a map",
            path = skill_file.display()
        );
        return Err(format!(
            "skill '{name}' has a 'metadata' that is not a map, so meka cannot record anything in \
             it; refusing to rewrite the skill. The Agent Skills spec defines 'metadata' as a map \
             of string to string; fix that file, or use a different name."
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
    // Only when the skill does not already claim one, in *either* spelling: overwriting a human's
    // attribution because an agent edited their file loses information nothing else records.
    //
    // [`Skill::author`] rather than a look in `metadata` alone, because a hand-written file keeps
    // its claim at the top level and that is still the file saying who wrote it. Reading only the
    // nested spelling let an agent rewriting a hand-written skill sign it.
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

    // Parse the bytes we are about to write, exactly as discovery will. Without this a description
    // the renderer could not represent produces a file that writes fine, reports success, and is
    // then skipped by discovery forever: absent from the index, unreachable by `skill_read`, and
    // now refused by this function's own clobber guard, so the agent cannot even repair it. The
    // check also makes any future change to the renderer fail here rather than silently.
    let written = parse_skill_definition(name, root, &dir, &skill_file, &rendered)
        .map_err(|error| format!("refusing to write a skill that would not parse back: {error}"))?;

    // Atomic, like every other store write. `fs::write` truncates in place, so an interrupted write
    // leaves a half-file that discovery rejects and the guard above then refuses to overwrite.
    // That was survivable when only `meka skill add` wrote skills; an agent that may write on
    // any turn makes it worth the rename.
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
    // whatever it pointed at. Reporting that as a deleted skill is a lie about what happened, and
    // the user planted the link for a reason.
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
/// by line. Hand-rolled quoting was getting this wrong in ways that only showed up on hostile input
/// -- a newline in a `license`, a metadata *key* containing one -- and each of those produced a
/// file that either lost content silently or could never be written again. The serializer's job is
/// to know when a value needs quoting, folding or an explicit key, so it is allowed to do it.
///
/// Every optional key is omitted when unset, `metadata` is omitted entirely when empty, and the
/// rank is omitted at its default, so a minimal skill renders as exactly the spec's minimal
/// example: `name` and `description` and nothing else.
pub(super) fn render_skill_file(skill: &Skill, body: &str) -> String {
    use serde_norway::{Mapping, Value};

    let mut front = Mapping::new();
    front.insert("name".into(), skill.name.as_str().into());
    // Normalized, not merely quoted: the description is a one-line label everywhere it is rendered,
    // and `store::normalize_description` is what guarantees that regardless of the file it came
    // from. See that function for why it is load-bearing rather than cosmetic.
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
