//! Agent memory: durable notes the agent writes for itself, surviving compaction and outliving any
//! one session.
//!
//! Memories are rows in the `memories` table of the store (`MEKA_DATA_DIR`), one per note,
//! carrying a required one-line `description` and an optional body, priority, tag set, load
//! policy, references, source, verification stamp and expiry. The store
//! is scoped to the meka *instance*, not to a session or a directory: meka has no Project concept,
//! and the motivating deployment is a single always-on session reachable over chat, where the agent
//! is closer to a person than to a checkout.
//!
//! This module owns the vocabulary: what a [`Memory`] is, what names and tags are legal, how an
//! age is rendered, and how a memory is written out for `meka memory export`. Storage and retrieval
//! are [`crate::store::memory`], which is the source of truth.
//!
//! One row is the whole of a memory: there is no second copy to keep in step, and a transaction is
//! what makes a write atomic rather than a lock. `meka memory export` is the answer for anyone who
//! wants `$EDITOR`, `grep` or a git-able directory over the same content.
//!
//! [`crate::skills`] is deliberately not built this way. A `SKILL.md` is a shared spec format other
//! clients read, and a skill directory carries bundled scripts and assets, so files are right
//! there.
//!
//! Why this survives compaction: the index rides [`crate::prompt::WorldSnapshot`], which
//! `Agent::last_rendered_world` re-states in full at session start, after every compaction, and
//! whenever the previous render scrolls out of the context window.

use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};

// Re-exported rather than referenced through `crate::entry` at each use site: priority is part
// of the memory store's public vocabulary (`meka memory add --priority`, the `memory_write`
// schema), and its default lives there only so `skills` can share the same scale.
pub(crate) use crate::entry::{DEFAULT_PRIORITY, normalize_description};
use crate::entry::{validate_entry_name, yaml_scalar};

/// A single durable note, as one row of the `memories` table.
///
/// `description` is what the agent sees every turn; `body` is fetched on demand through the
/// `memory_read` tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Memory {
    pub(crate) name: String,
    pub(crate) description: String,
    /// How important the note is: where it sorts in the index and how heavily search weighs it.
    /// Not whether it is in context; that is [`Self::load`].
    pub(crate) priority: u8,
    /// How the memory reaches the model's context.
    pub(crate) load: Load,
    /// What the memory depends on: other memories by name, paths, URLs, external ids, each one
    /// line and validated by [`validate_reference`]. A name that matches no memory is allowed, as
    /// a pointer to something worth writing later; the backlink query is what makes a changed
    /// fact findable everywhere it is cited.
    pub(crate) references: Vec<String>,
    /// Where the fact came from, one line; empty when not recorded.
    pub(crate) source: String,
    /// When the fact was last verified to hold. Stamped at creation and moved only by an
    /// explicit verification, never as a side effect of a write: a stamp that moved on any edit
    /// would answer the same question [`Self::updated_at`] does, and the question this one is
    /// for is whether anyone has checked the note since it was written.
    pub(crate) verified_at: SystemTime,
    /// When the memory stops applying, if a date was given. Past it the memory leaves the index
    /// and the inline band and is reached by search alone; nothing deletes it.
    pub(crate) expires_at: Option<SystemTime>,
    /// Free-form labels, validated to [`validate_tag`]'s character class.
    ///
    /// What makes a store of thousands navigable rather than merely searchable: the index can only
    /// show a couple of hundred entries, and "4,910 more memories not shown" is not a usable
    /// signal, where "most common tags infra, people, decisions" is a query the model can act on.
    pub(crate) tags: Vec<String>,
    /// When the memory was *created*: stamped once, at create, and carried across every later
    /// write by the upsert itself (see [`crate::store::memory::MemoryStore::write`]).
    ///
    /// Distinct from [`Self::updated_at`] because the two answer different questions and only one
    /// of them is the one being asked. With one timestamp, a `memory_write` that merely rewords a
    /// description makes a years-old note render as "today", sort to the top of its priority band,
    /// and arrive through `memory_read` under the caption "Saved today. This is what you recorded
    /// then".
    pub(crate) created_at: SystemTime,
    /// When the row was last written. Reported by `meka memory get` and the HTTP API; it takes no
    /// part in ordering, ranking or the rendered age.
    pub(crate) updated_at: SystemTime,
    /// How many times `memory_read` has opened this memory. Feeds the usage weight in
    /// [`crate::store::memory::Ranking`], which is the counterweight to a priority the agent
    /// guessed once and never revised.
    pub(crate) read_count: u32,
    /// The body text, present when the query that produced this loaded it.
    ///
    /// [`crate::store::memory::MemoryStore::get`] always loads it;
    /// [`crate::store::memory::MemoryStore::index`] loads it only for the band the `[Memory]`
    /// section renders in full ([`Load::Inline`]), because carrying every body would put the
    /// whole store in resident memory for the sake of a handful of entries.
    pub(crate) body: Option<String>,
}

impl Memory {
    /// Whether the memory's expiry has passed as of `now`.
    pub(crate) fn is_expired(&self, now: SystemTime) -> bool {
        self.expires_at.is_some_and(|expires| expires <= now)
    }
}

/// How a memory reaches the model's context.
///
/// Separate from priority because the two answer different questions: a reference note can be
/// worth keeping without earning a line in every turn's index, and a rule can need to be in front
/// of the model without being its most important note. One spelling per value, on the
/// `memory_write` schema, `meka memory add --load`, the HTTP views and the export frontmatter.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum Load {
    /// The body is rendered into the per-turn context within the inline band's budget. For a
    /// standing rule the body *is* the rule, and leaving it behind a tool call the model may never
    /// make is a rule it is held to and cannot read.
    Inline,
    /// The description is listed in the per-turn index within its budget.
    #[default]
    Index,
    /// Reached by `memory_search` alone: neither listed nor inlined.
    Search,
}

impl Load {
    /// Every policy, in the order the names sort.
    pub(crate) const ALL: [Load; 3] = [Self::Index, Self::Inline, Self::Search];

    /// The one spelling of this policy.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::Index => "index",
            Self::Search => "search",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    pub(crate) fn supported() -> String {
        Self::ALL
            .iter()
            .map(|load| load.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl std::fmt::Display for Load {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for Load {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|load| load.name() == raw)
            .ok_or_else(|| {
                format!(
                    "unknown load '{raw}'; expected one of {}",
                    Self::supported()
                )
            })
    }
}

impl TryFrom<String> for Load {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<Load> for String {
    fn from(load: Load) -> Self {
        load.name().to_string()
    }
}

/// Most references one memory may carry. Past a score of dependencies the note is an index of
/// its own and belongs in a body.
pub(crate) const MAX_REFERENCES: usize = 20;
/// Longest a single reference may be, sized for a URL with a query string.
pub(crate) const MAX_REFERENCE_CHARS: usize = 512;
/// Longest a source line may be: the length past which the index elides a description, since a
/// source is one line the reader takes in at a glance and a read shows it whole.
pub(crate) const MAX_SOURCE_CHARS: usize = 500;

/// Validate one reference: non-empty, one line, no control characters, within
/// [`MAX_REFERENCE_CHARS`].
///
/// Nothing stricter, because a reference is any of four kinds of pointer (a memory name, a path,
/// a URL, an external id) and a grammar for one refuses the others. One line is load-bearing: the
/// export writes each as a YAML scalar, and the backlink query matches the stored string whole.
pub(crate) fn validate_reference(reference: &str) -> Result<(), String> {
    if reference.is_empty() {
        return Err("a reference cannot be empty".to_string());
    }
    if reference.chars().count() > MAX_REFERENCE_CHARS {
        return Err(format!(
            "reference '{}' exceeds {MAX_REFERENCE_CHARS} characters",
            crate::text::sanitize_text(reference)
        ));
    }
    if reference.chars().any(char::is_control) {
        return Err(format!(
            "reference '{}' contains a control character",
            crate::text::sanitize_text(reference)
        ));
    }
    Ok(())
}

/// Trim, deduplicate and validate a reference list on its way to the store.
///
/// Deduplicated without regard to case, keeping the first spelling, because the backlink query
/// matches without regard to case: two spellings of one name would otherwise count as two
/// dependencies on one memory. Order is kept as given, since a list of dependencies has no
/// natural order and the agent's is as good as any.
pub(crate) fn normalize_references(references: &[String]) -> Result<Vec<String>, String> {
    let mut kept: Vec<String> = Vec::with_capacity(references.len());
    for reference in references {
        let reference = reference.trim();
        validate_reference(reference)?;
        if !kept
            .iter()
            .any(|known| known.eq_ignore_ascii_case(reference))
        {
            kept.push(reference.to_string());
        }
    }
    if kept.len() > MAX_REFERENCES {
        return Err(format!(
            "{} references given; at most {MAX_REFERENCES} are allowed",
            kept.len()
        ));
    }
    Ok(kept)
}

/// Collapse a source to the one line it is, and bound it.
pub(crate) fn normalize_source(source: &str) -> Result<String, String> {
    let source = normalize_description(source);
    if source.chars().count() > MAX_SOURCE_CHARS {
        return Err(format!("source exceeds {MAX_SOURCE_CHARS} characters"));
    }
    Ok(source)
}

/// Parse the instant a memory stops applying: an RFC 3339 timestamp, or a `YYYY-MM-DD` date
/// meaning the whole of that day in local time.
///
/// A date rather than a timestamp is what a model writes when it means "until the freeze ends on
/// the 25th", and the day is taken whole, so the memory expires when the 26th begins. Local time
/// because that is the clock every `Date:` line in the context states, so the model's "today" and
/// this function's agree.
pub(crate) fn parse_expiry(raw: &str) -> Result<SystemTime, String> {
    let raw = raw.trim();
    if let Some(parsed) = parse_stamp(raw) {
        return Ok(parsed);
    }
    let date = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d")
        .map_err(|_| format!("'{raw}' is not an RFC 3339 timestamp or a YYYY-MM-DD date"))?;
    let next_day = date
        .succ_opt()
        .ok_or_else(|| format!("'{raw}' is past the last representable date"))?;
    let midnight = next_day
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| format!("'{raw}' has no midnight"))?;
    // `earliest` rather than `single`: a midnight that a clock change skips or repeats is still
    // a moment the day begins, and refusing the date for it would be refusing one day a year.
    let local = midnight
        .and_local_timezone(chrono::Local)
        .earliest()
        .ok_or_else(|| format!("'{raw}' has no midnight in this time zone"))?;
    Ok(SystemTime::from(local.with_timezone(&chrono::Utc)))
}

/// Parse the window an `unverified` filter takes, `"90d"` or `"6months"`: the grammar every other
/// duration meka reads uses, and one parser for the tool, the CLI and the HTTP route.
pub(crate) fn parse_unverified_window(raw: &str) -> Result<std::time::Duration, String> {
    humantime_serde::re::humantime::parse_duration(raw.trim())
        .map_err(|error| format!("'{raw}' is not a duration like '90d': {error}"))
}

/// How far off an expiry is, e.g. "expires in 3 days", "expires today", "expired 2 days ago".
pub(crate) fn render_expiry(expires_at: SystemTime, now: SystemTime) -> String {
    match now.duration_since(expires_at) {
        Ok(elapsed) => match elapsed.as_secs() / 86_400 {
            0 => "expired today".to_string(),
            1 => "expired yesterday".to_string(),
            days => format!("expired {days} days ago"),
        },
        Err(ahead) => match ahead.duration().as_secs() / 86_400 {
            0 => "expires today".to_string(),
            1 => "expires tomorrow".to_string(),
            days => format!("expires in {days} days"),
        },
    }
}

/// When the note was recorded and when it was last verified, as one sentence for the agent and
/// the operator: "Recorded 47 days ago, verified yesterday." or "Recorded 47 days ago, not
/// verified since." A note recorded today has nothing to verify yet and says only that.
pub(crate) fn render_record_caption(memory: &Memory, now: SystemTime) -> String {
    let recorded = render_age(memory.created_at, now);
    if memory.verified_at != memory.created_at {
        return format!(
            "Recorded {recorded}, verified {}.",
            render_age(memory.verified_at, now)
        );
    }
    if recorded == "today" {
        return "Recorded today.".to_string();
    }
    format!("Recorded {recorded}, not verified since.")
}

/// Most tags one memory may carry. A tag set is a handful of labels; past this it is a body.
pub(crate) const MAX_TAGS: usize = 10;
/// Longest a single tag may be. Bounded so the histogram in the `[Memory]` index stays readable.
pub(crate) const MAX_TAG_LEN: usize = 32;

/// Validate one tag: lowercase alphanumerics and hyphens, starting with an alphanumeric.
///
/// Strict deliberately, and load-bearing rather than cosmetic in two places. Tags are stored
/// space-joined in one column, so a tag containing a space would come back as two; and
/// [`render_memory`] emits the list as a YAML flow sequence through [`yaml_scalar`], which is safe
/// only for values already normalized to one line.
pub(crate) fn validate_tag(tag: &str) -> Result<(), String> {
    if tag.is_empty() {
        return Err("a tag cannot be empty".to_string());
    }
    if tag.chars().count() > MAX_TAG_LEN {
        return Err(format!("tag '{tag}' exceeds {MAX_TAG_LEN} characters"));
    }
    let mut chars = tag.chars();
    #[allow(
        clippy::expect_used,
        reason = "the empty tag was refused above, so the first char is always `Some`"
    )]
    let first = chars.next().expect("non-empty checked above");
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return Err(format!(
            "tag '{tag}' must start with a lowercase letter or digit"
        ));
    }
    for character in chars {
        if !(character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-') {
            return Err(format!(
                "tag '{tag}' contains invalid character '{character}'; only [a-z0-9-] are allowed"
            ));
        }
    }
    Ok(())
}

/// Validate a whole tag list for a write, rejecting the set rather than silently dropping members.
pub(crate) fn validate_tags(tags: &[String]) -> Result<(), String> {
    if tags.len() > MAX_TAGS {
        return Err(format!(
            "{} tags given; at most {} are allowed",
            tags.len(),
            MAX_TAGS
        ));
    }
    for tag in tags {
        validate_tag(tag)?;
    }
    Ok(())
}

/// Lowercase, sort, deduplicate and validate a tag list on its way to the store.
///
/// Lowercased *before* validating, so the doors agree: search normalizes case anyway, and refusing
/// `Infra` here would mean a label meka itself renders is one meka itself will not take back.
/// Sorted before the dedup because `dedup` only removes *consecutive* duplicates, so the other
/// order leaves `[a, b, a]` intact and the row ends up declaring `a` twice.
pub(crate) fn normalize_tags(tags: &[String]) -> Result<Vec<String>, String> {
    let mut tags: Vec<String> = tags.iter().map(|tag| tag.trim().to_lowercase()).collect();
    tags.sort();
    tags.dedup();
    validate_tags(&tags)?;
    Ok(tags)
}

/// Parse an RFC 3339 timestamp column, or `None` if it is not one.
pub(crate) fn parse_stamp(raw: &str) -> Option<SystemTime> {
    chrono::DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|parsed| SystemTime::from(parsed.with_timezone(&chrono::Utc)))
}

/// Render a [`SystemTime`] as the RFC 3339 string the timestamp columns carry.
pub(crate) fn render_stamp(time: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(time).to_rfc3339()
}

/// The file name one exported memory lands under. The sole owner of the `<name>.md` layout
/// convention, so changing it is a one-line edit rather than a grep.
fn memory_file_name(name: &str) -> String {
    format!("{name}.md")
}

/// Resolve one exported memory's path inside `root`. Performs no I/O and does not validate the
/// name; callers pair it with [`validate_memory_name`].
pub(crate) fn memory_file_in(root: &Path, name: &str) -> PathBuf {
    root.join(memory_file_name(name))
}

/// Validate that `name` is a safe prompt-embeddable memory identifier. See [`validate_entry_name`]
/// for the rules.
///
/// Load-bearing for two reasons: a name is what `meka memory export` turns into a file name, and
/// it is text the model reads in every turn's index.
pub(crate) fn validate_memory_name(name: &str) -> Result<(), String> {
    validate_entry_name(name, "memory")
}

/// Bound a name being *looked up*, without demanding it be one this store would write.
///
/// The lookup half of [`validate_memory_name`]; see [`crate::entry::validate_lookup_name`] for why
/// a store that applies its write rule to reads and deletes cannot get rid of a row it should never
/// have accepted.
pub(crate) fn validate_memory_lookup(name: &str) -> Result<(), String> {
    crate::entry::validate_lookup_name(name, "memory")
}

/// Clamp a caller-supplied `priority` for a memory. Thin wrapper over
/// [`crate::entry::parse_priority`] that supplies this store's noun, mirroring
/// [`validate_memory_name`].
pub(crate) fn parse_priority(raw: Option<i64>, name: &str) -> u8 {
    crate::entry::parse_priority(raw, "memory", name)
}

/// The header an exported memory file carries.
///
/// A struct rather than four positional arguments, as `render_skill_file` takes:
/// `render_memory("x", 5, None, &[], body)` is unreadable at the call site and silently accepts a
/// swapped pair.
#[derive(Debug, Clone)]
pub(crate) struct MemoryFrontmatter {
    pub(crate) description: String,
    pub(crate) priority: u8,
    pub(crate) load: Load,
    /// The `created:` value as it will appear in the file, RFC 3339.
    pub(crate) created: Option<String>,
    /// The `verified:` value, RFC 3339, given only when it differs from `created`: equal stamps
    /// say nothing the file does not already, and leaving them out keeps the common header short.
    pub(crate) verified: Option<String>,
    /// The `expires:` value, RFC 3339, when an expiry was set.
    pub(crate) expires: Option<String>,
    /// One line, emitted only when non-empty.
    pub(crate) source: String,
    pub(crate) references: Vec<String>,
    pub(crate) tags: Vec<String>,
    /// How many times the memory has been read, emitted only when non-zero.
    ///
    /// Not content but usage: what the agent has *done* with the note. It rides along in an export
    /// because it is the one value the rest of the file cannot reconstruct: descriptions, bodies
    /// and dates are all there, but a store restored with every counter at zero has silently lost
    /// each memory's accumulated ranking weight. A reader that does not model the key ignores it,
    /// as it ignores any other it does not model.
    pub(crate) read_count: u32,
}

/// Render one memory as a Markdown file: frontmatter followed by the body.
///
/// The export format, and the only place meka writes YAML for this store. Frontmatter rather than
/// JSON because the point of an export is to be read and edited by a person, and to be greppable
/// in a directory the way the rest of a notes tree is.
///
/// `priority` is emitted only when it differs from [`DEFAULT_PRIORITY`], `load` only when it is
/// not the default, `created` only when known, `verified` only when it differs from `created`,
/// `expires`, `source`, `references` and `tags` only when set, so the common case stays a two-line
/// header. The same text is what a write or a delete draws as a diff, so each field is its own
/// line and a change to one shows as that line.
pub(crate) fn render_memory(frontmatter: &MemoryFrontmatter, body: &str) -> String {
    let mut out = String::new();
    out.push_str("---\n");
    // Only the characters YAML genuinely cannot carry are dropped. A C0 or C1 control inside a
    // double-quoted scalar is outside YAML's `c-printable` production, so an export holding one is
    // a file no parser will read. `sanitize_stored_description` would go further than that
    // argument: it also strips the whole `Cf` category, so a Persian description would come back
    // a different word after a backup, a read that sanitizes written to a persistent store.
    out.push_str(&format!(
        "description: {}\n",
        yaml_scalar(&normalize_description(&yaml_printable(
            &frontmatter.description
        )))
    ));
    if frontmatter.priority != DEFAULT_PRIORITY {
        out.push_str(&format!("priority: {}\n", frontmatter.priority));
    }
    if frontmatter.load != Load::default() {
        out.push_str(&format!("load: {}\n", frontmatter.load));
    }
    if let Some(created) = &frontmatter.created {
        // Quoted by `yaml_scalar` on the strength of the colons in the time, which is what keeps
        // the offset from parsing as a nested mapping.
        out.push_str(&format!("created: {}\n", yaml_scalar(created)));
    }
    if let Some(verified) = &frontmatter.verified {
        out.push_str(&format!("verified: {}\n", yaml_scalar(verified)));
    }
    if let Some(expires) = &frontmatter.expires {
        out.push_str(&format!("expires: {}\n", yaml_scalar(expires)));
    }
    if !frontmatter.source.is_empty() {
        out.push_str(&format!(
            "source: {}\n",
            yaml_scalar(&normalize_description(&yaml_printable(&frontmatter.source)))
        ));
    }
    if frontmatter.read_count > 0 {
        out.push_str(&format!("read_count: {}\n", frontmatter.read_count));
    }
    if !frontmatter.references.is_empty() {
        // Quoted one by one like the tags below: a reference has passed `validate_reference`,
        // so it is one line, but a URL holds colons and a path may open with a YAML indicator.
        let quoted: Vec<String> = frontmatter
            .references
            .iter()
            .map(|reference| yaml_scalar(&yaml_printable(reference)))
            .collect();
        out.push_str(&format!("references: [{}]\n", quoted.join(", ")));
    }
    if !frontmatter.tags.is_empty() {
        // Each element through `yaml_scalar` rather than a bare flow sequence. Every tag reaching
        // here has passed `validate_tag`, so the bare form would be safe today; quoting costs
        // nothing and means a later loosening of that character class cannot quietly produce a file
        // a reader takes as something else.
        let quoted: Vec<String> = frontmatter
            .tags
            .iter()
            .map(|tag| yaml_scalar(tag))
            .collect();
        out.push_str(&format!("tags: [{}]\n", quoted.join(", ")));
    }
    // The body verbatim between one separator newline and one terminator newline, both added
    // unconditionally. Trimming leading newlines and appending a terminator only when one is
    // missing makes the framing ambiguous, so a reader cannot tell padding from content: a body
    // of `b` comes back `\nb`, and one ending `\r` comes back ending `\r\n`. Adding exactly one
    // of each, always, is what makes the round trip exact for every body including an empty one.
    out.push_str("---\n\n");
    out.push_str(body);
    out.push('\n');
    out
}

/// Drop the characters YAML cannot represent in a scalar, and nothing else.
///
/// Deliberately narrower than [`render_for_model`]: this is the *export* path, so the only
/// justification for touching the text is that the file would otherwise be unparseable. Format
/// characters (ZWJ, ZWNJ, bidi marks) are `c-printable` and stay.
fn yaml_printable(text: &str) -> String {
    text.chars()
        .filter(|character| {
            let code = *character as u32;
            *character == '\n'
                || *character == '\t'
                || (code >= 0x20 && code != 0x7F && !(0x80..=0x9F).contains(&code))
        })
        .collect()
}

/// Whether a description will still say something by the time the model reads it.
///
/// `trim().is_empty()` is a question about whitespace. Format characters (zero-width spaces,
/// joiners, bidi controls) are not whitespace, so three of them pass as a description and then
/// render as nothing: a blank cell in `meka memory list`, a blank line in `memory_search`, and
/// `- **name** (p5, today): ` in the index the model reads every turn.
/// [`render_description_for_model`] strips exactly that class at the render boundary, so asking
/// it is asking the question the store actually has.
///
/// Distinct from [`description_survives_export`], which asks whether YAML can carry the text. A
/// description can fail either check independently.
pub(crate) fn description_says_something(description: &str) -> bool {
    !render_description_for_model(description).trim().is_empty()
}

/// Whether a description would still say something once written to an export file.
///
/// [`render_memory`] drops what YAML cannot carry, and a description made only of such characters
/// becomes `description: ""`, which reads back as no description at all, losing the memory through
/// the one path that is supposed to preserve it. `meka memory export` asks this before it writes
/// anything, and refuses the whole run rather than write a file that would come back empty.
///
/// Distinct from [`description_says_something`], which asks whether the *model* would see anything.
/// A description can fail either check independently: YAML carries a zero-width space fine, and the
/// render boundary strips it.
pub(crate) fn description_survives_export(description: &str) -> bool {
    !normalize_description(&yaml_printable(description))
        .trim()
        .is_empty()
}

/// Render one [`Memory`] as an export file, body included.
pub(crate) fn export_memory(memory: &Memory) -> String {
    render_memory(
        &MemoryFrontmatter {
            description: memory.description.clone(),
            priority: memory.priority,
            load: memory.load,
            created: Some(render_stamp(memory.created_at)),
            verified: (memory.verified_at != memory.created_at)
                .then(|| render_stamp(memory.verified_at)),
            expires: memory.expires_at.map(render_stamp),
            source: memory.source.clone(),
            references: memory.references.clone(),
            tags: memory.tags.clone(),
            read_count: memory.read_count,
        },
        memory.body.as_deref().unwrap_or_default(),
    )
}

/// Make stored memory text safe to render into a model's context or a terminal.
///
/// The store returns bytes (see `crate::store::memory::row_to_memory`), because `meka memory edit`
/// round-trips a body through `$EDITOR` and a read that stripped characters would destroy them
/// permanently. Neutralizing therefore happens here, at each boundary where the text is *displayed*
/// rather than carried: the `[Memory]` index and its standing band, `memory_read`, both search
/// renderers, a sub-agent's index, and the `meka memory` listing.
///
/// The two doors that deliberately do **not** call this are `meka memory export` and
/// `meka memory edit`, which exist to hand back exactly what is stored.
pub(crate) fn render_for_model(text: &str) -> String {
    crate::text::sanitize_text(text)
}

/// The same for a description, which is additionally contracted to be one line.
///
/// A memory whose description carries a newline would otherwise open what looks like a new section
/// in the per-turn index the model reads every turn.
pub(crate) fn render_description_for_model(description: &str) -> String {
    crate::entry::sanitize_stored_description(description)
}

/// Human-readable age, e.g. "today", "yesterday", "47 days ago".
///
/// Deliberately not an ISO timestamp: models are poor at date arithmetic, and a rendered age
/// prompts staleness reasoning in a way a raw date does not. A memory is a point-in-time
/// observation, and the agent needs to weigh an old one accordingly.
///
/// Callers pass [`Memory::created_at`] or [`Memory::verified_at`], never [`Memory::updated_at`].
/// The number is only worth rendering if it answers "how old is what this says" or "when was it
/// last checked"; an edit date dressed up as either is worse than no date, because it reads as a
/// fact the model can rely on.
pub(crate) fn render_age(created: SystemTime, now: SystemTime) -> String {
    // A stamp in the future is its own answer, not "today". `duration_since` fails for one, and
    // folding that to zero tells the model a note dated next year was written this morning, while
    // the same row sorts to the top of its priority band, so the memory most likely to be wrong is
    // also the most prominent. Reachable through clock skew between two machines sharing a data
    // directory, or a hand-written date. Saying so is what lets the model discount it.
    let Ok(elapsed) = now.duration_since(created) else {
        return "at a future date".to_string();
    };
    match elapsed.as_secs() / 86_400 {
        0 => "today".to_string(),
        1 => "yesterday".to_string(),
        n => format!("{n} days ago"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_memory_name_rejects_traversal_and_separators() {
        for bad in [
            "../escape",
            "a/b",
            "a\\b",
            "/abs",
            ".hidden",
            "",
            "with space",
        ] {
            assert!(
                validate_memory_name(bad).is_err(),
                "'{bad}' must be rejected"
            );
        }
        for good in ["alice-timezone", "deploy_host", "note42", "A-B_c9"] {
            assert!(
                validate_memory_name(good).is_ok(),
                "'{good}' must be accepted"
            );
        }
    }

    #[test]
    fn priority_defaults_and_clamps() {
        assert_eq!(parse_priority(None, "n"), DEFAULT_PRIORITY);
        assert_eq!(parse_priority(Some(0), "n"), 0);
        assert_eq!(parse_priority(Some(9), "n"), 9);
        assert_eq!(parse_priority(Some(-5), "n"), crate::entry::MIN_PRIORITY);
        assert_eq!(parse_priority(Some(99), "n"), crate::entry::MAX_PRIORITY);
    }

    /// A tag is stored space-joined in one column and rendered into a YAML flow sequence, so
    /// anything either of those would read as structure has to be refused at the door.
    #[test]
    fn tag_validation_refuses_anything_yaml_would_interpret() {
        for bad in [
            "",
            "Infra",
            "-leading",
            "has space",
            "with:colon",
            "under_score",
            "quote\"mark",
            "comma,tag",
            "bracket]",
            &"x".repeat(MAX_TAG_LEN + 1),
        ] {
            assert!(validate_tag(bad).is_err(), "'{bad}' must be rejected");
        }
        for good in ["infra", "deploy-host", "k8s", "0day"] {
            assert!(validate_tag(good).is_ok(), "'{good}' must be accepted");
        }
        assert!(validate_tags(&vec!["a".to_string(); MAX_TAGS]).is_ok());
        assert!(validate_tags(&vec!["a".to_string(); MAX_TAGS + 1]).is_err());
    }

    /// The normalization the write door applies. Sorting before the dedup is the load-bearing part:
    /// `dedup` only removes *consecutive* duplicates, so `[a, b, a]` would otherwise survive whole
    /// and the row would declare `a` twice.
    #[test]
    fn normalize_tags_lowercases_sorts_and_deduplicates() {
        assert_eq!(
            normalize_tags(&[
                "Infra".to_string(),
                "deploy".to_string(),
                " infra ".to_string()
            ])
            .expect("valid"),
            vec!["deploy".to_string(), "infra".to_string()]
        );
        assert!(normalize_tags(&["has space".to_string()]).is_err());
    }

    /// An export has to parse back as YAML frontmatter, or `meka memory export` produces files
    /// that only look like the format they claim.
    #[test]
    fn export_round_trips_through_frontmatter() {
        let created = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let memory = Memory {
            name: "note".to_string(),
            description: "a description: with a colon".to_string(),
            priority: 2,
            load: Load::Search,
            references: vec![
                "alice-timezone".to_string(),
                "https://example.com/a?b=c".to_string(),
                "[bracketed]".to_string(),
            ],
            source: "alice, in chat: on the 3rd".to_string(),
            verified_at: created + std::time::Duration::from_secs(86_400),
            expires_at: Some(created + std::time::Duration::from_secs(30 * 86_400)),
            tags: vec!["deploy".to_string(), "infra".to_string()],
            created_at: created,
            updated_at: SystemTime::now(),
            read_count: 3,
            body: Some("Body text.\n".to_string()),
        };
        let rendered = export_memory(&memory);
        let (frontmatter, body) =
            crate::entry::split_frontmatter(&rendered).expect("must have frontmatter");
        let parsed: yaml_serde::Value =
            yaml_serde::from_str(frontmatter).expect("frontmatter must parse");
        assert_eq!(
            parsed["description"].as_str(),
            Some("a description: with a colon")
        );
        assert_eq!(parsed["priority"].as_i64(), Some(2));
        assert_eq!(
            parsed["created"].as_str(),
            Some("2023-11-14T22:13:20+00:00")
        );
        assert_eq!(
            parsed["tags"]
                .as_sequence()
                .map(|tags| tags.len())
                .unwrap_or(0),
            2
        );
        // The one value the file cannot otherwise carry: what the agent has done with the note.
        assert_eq!(parsed["read_count"].as_i64(), Some(3));
        assert_eq!(parsed["load"].as_str(), Some("search"));
        assert_eq!(
            parsed["verified"].as_str(),
            Some("2023-11-15T22:13:20+00:00")
        );
        assert_eq!(
            parsed["expires"].as_str(),
            Some("2023-12-14T22:13:20+00:00")
        );
        assert_eq!(
            parsed["source"].as_str(),
            Some("alice, in chat: on the 3rd")
        );
        let references: Vec<&str> = parsed["references"]
            .as_sequence()
            .expect("a sequence")
            .iter()
            .filter_map(|value| value.as_str())
            .collect();
        assert_eq!(references, [
            "alice-timezone",
            "https://example.com/a?b=c",
            "[bracketed]"
        ]);
        assert_eq!(body.trim(), "Body text.");
    }

    /// A default priority and an empty tag set are omitted, so an export stays readable.
    #[test]
    fn export_omits_default_priority_and_empty_tags() {
        let memory = Memory {
            name: "note".to_string(),
            description: "plain".to_string(),
            priority: DEFAULT_PRIORITY,
            load: Load::default(),
            references: Vec::new(),
            source: String::new(),
            verified_at: SystemTime::UNIX_EPOCH,
            expires_at: None,
            tags: Vec::new(),
            created_at: SystemTime::UNIX_EPOCH,
            updated_at: SystemTime::UNIX_EPOCH,
            read_count: 0,
            body: None,
        };
        let rendered = export_memory(&memory);
        assert!(!rendered.contains("priority:"), "{rendered}");
        assert!(!rendered.contains("load:"), "{rendered}");
        assert!(!rendered.contains("verified:"), "{rendered}");
        assert!(!rendered.contains("expires:"), "{rendered}");
        assert!(!rendered.contains("source:"), "{rendered}");
        assert!(!rendered.contains("references:"), "{rendered}");
        assert!(!rendered.contains("tags:"), "{rendered}");
        assert!(!rendered.contains("read_count:"), "{rendered}");
        assert!(rendered.contains("created:"), "{rendered}");
    }

    /// One spelling per policy, and the default is the listed tier.
    #[test]
    fn a_load_has_one_spelling_and_defaults_to_index() {
        for load in Load::ALL {
            assert_eq!(load.name().parse::<Load>(), Ok(load));
            assert_eq!(String::from(load).parse::<Load>(), Ok(load));
        }
        assert_eq!(Load::default(), Load::Index);
        assert!("Inline".parse::<Load>().is_err());
        // The refusal names every spelling that would have been accepted.
        let refusal = "always".parse::<Load>().expect_err("unknown");
        assert!(
            refusal.contains("'always'") && refusal.contains("index, inline, search"),
            "{refusal}"
        );
    }

    /// A reference is any one-line pointer, deduplicated without regard to case, bounded in
    /// count and length, and refused when it could break the line it is written on.
    #[test]
    fn references_are_trimmed_deduplicated_and_bounded() {
        let kept = normalize_references(&[
            " alice-timezone ".to_string(),
            "Alice-Timezone".to_string(),
            "/srv/deploy/run.sh".to_string(),
            "https://example.com/a?b=c#d".to_string(),
            "JIRA-123".to_string(),
        ])
        .expect("valid");
        assert_eq!(kept, [
            "alice-timezone",
            "/srv/deploy/run.sh",
            "https://example.com/a?b=c#d",
            "JIRA-123"
        ]);
        assert!(normalize_references(&["".to_string()]).is_err());
        assert!(normalize_references(&["two\nlines".to_string()]).is_err());
        // The bounds are inclusive: the longest reference and the fullest list pass, one more
        // of either is refused.
        assert!(normalize_references(&["x".repeat(MAX_REFERENCE_CHARS)]).is_ok());
        assert!(normalize_references(&["x".repeat(MAX_REFERENCE_CHARS + 1)]).is_err());
        let full: Vec<String> = (0..MAX_REFERENCES)
            .map(|index| format!("reference-{index}"))
            .collect();
        assert_eq!(
            normalize_references(&full).expect("the fullest list").len(),
            MAX_REFERENCES
        );
        let many: Vec<String> = (0..=MAX_REFERENCES)
            .map(|index| format!("reference-{index}"))
            .collect();
        assert!(normalize_references(&many).is_err());
    }

    /// A source is one line, bounded inclusively, and an empty one is the explicit clear.
    #[test]
    fn a_source_is_one_bounded_line() {
        assert_eq!(
            normalize_source("  alice,\n  in chat ").expect("one line"),
            "alice, in chat"
        );
        assert_eq!(normalize_source("").expect("a clear"), "");
        assert!(normalize_source(&"s".repeat(MAX_SOURCE_CHARS)).is_ok());
        assert!(normalize_source(&"s".repeat(MAX_SOURCE_CHARS + 1)).is_err());
    }

    /// An expiry is a timestamp, or a date meaning the whole of that day: the memory holds
    /// through the day named and expires when the next one begins.
    #[test]
    fn an_expiry_date_is_the_whole_day_and_a_timestamp_is_itself() {
        let stamp = parse_expiry("2030-01-31T10:00:00+00:00").expect("a timestamp");
        assert_eq!(render_stamp(stamp), "2030-01-31T10:00:00+00:00");

        let day = parse_expiry("2030-01-31").expect("a date");
        let local: chrono::DateTime<chrono::Local> =
            chrono::DateTime::<chrono::Utc>::from(day).into();
        assert_eq!(
            local.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2030-02-01 00:00:00"
        );

        assert!(parse_expiry("next week").is_err());
        assert!(parse_expiry("2030-13-01").is_err());

        let now = SystemTime::now();
        let day = std::time::Duration::from_secs(86_400);
        assert_eq!(render_expiry(now + day * 3, now), "expires in 3 days");
        assert_eq!(render_expiry(now + day, now), "expires tomorrow");
        assert_eq!(render_expiry(now + day / 2, now), "expires today");
        assert_eq!(render_expiry(now - day / 2, now), "expired today");
        assert_eq!(render_expiry(now - day, now), "expired yesterday");
        assert_eq!(render_expiry(now - day * 2, now), "expired 2 days ago");
    }

    /// The caption states the verification only when one has been made since the note was
    /// recorded; equal stamps mean nobody has checked it, which is worth saying once the note
    /// has an age.
    #[test]
    fn the_record_caption_names_a_verification_only_when_there_was_one() {
        let now = SystemTime::now();
        let day = std::time::Duration::from_secs(86_400);
        let mut memory = Memory {
            name: "note".to_string(),
            description: "plain".to_string(),
            priority: DEFAULT_PRIORITY,
            load: Load::default(),
            references: Vec::new(),
            source: String::new(),
            verified_at: now - day * 47,
            expires_at: None,
            tags: Vec::new(),
            created_at: now - day * 47,
            updated_at: now,
            read_count: 0,
            body: None,
        };
        assert_eq!(
            render_record_caption(&memory, now),
            "Recorded 47 days ago, not verified since."
        );
        memory.verified_at = now - day;
        assert_eq!(
            render_record_caption(&memory, now),
            "Recorded 47 days ago, verified yesterday."
        );
        memory.created_at = now;
        memory.verified_at = now;
        assert_eq!(render_record_caption(&memory, now), "Recorded today.");
        assert!(!memory.is_expired(now));
        memory.expires_at = Some(now - day);
        assert!(memory.is_expired(now));
    }

    #[test]
    fn an_age_renders_as_today_yesterday_days_ago_or_a_future_date() {
        let now = SystemTime::now();
        assert_eq!(render_age(now, now), "today");
        assert_eq!(
            render_age(now - std::time::Duration::from_secs(86_400), now),
            "yesterday"
        );
        assert_eq!(
            render_age(now - std::time::Duration::from_secs(47 * 86_400), now),
            "47 days ago"
        );
        // A stamp in the future must not panic, and must not read as "today" either: such a row
        // sorts to the top of its priority band, so the memory most likely to be wrong would also
        // be the most prominent, with nothing saying why.
        assert_eq!(
            render_age(now + std::time::Duration::from_secs(86_400), now),
            "at a future date"
        );
    }
}
