//! Skill discovery and loading. Walks `~/.config/agsh/skills/<name>/SKILL.md`,
//! parses the YAML frontmatter (description, when_to_use, allowed_tools,
//! version, user_invocable), and exposes the resulting [`Skill`] structs to
//! the agent for system-prompt injection and `skill` tool dispatch.

pub mod cli;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use serde::Deserialize;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub source_dir: PathBuf,
    pub description: String,
    pub when_to_use: String,
    pub allowed_tools: Vec<String>,
    pub version: Option<String>,
    /// User-side invocability gate consulted by the REPL `/skill <name>`
    /// command. Defaults to `true` when the frontmatter omits the field.
    /// The agent-side `SkillTool` (in `src/tools/skill.rs`) ignores this
    /// — gating model invocation is a separate concern.
    pub user_invocable: bool,
    pub body_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Frontmatter {
    description: Option<String>,
    when_to_use: Option<String>,
    #[serde(default, deserialize_with = "deserialize_allowed_tools")]
    allowed_tools: Vec<String>,
    version: Option<String>,
    #[serde(default = "default_user_invocable")]
    user_invocable: bool,
}

fn default_user_invocable() -> bool {
    true
}

/// Accept either an array `[a, b]` or a CSV string `"a, b"` for `allowed_tools`.
fn deserialize_allowed_tools<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        String(String),
        Vec(Vec<String>),
    }

    match Option::<StringOrVec>::deserialize(deserializer)? {
        None => Ok(Vec::new()),
        Some(StringOrVec::Vec(vec)) => Ok(vec),
        Some(StringOrVec::String(s)) => Ok(s
            .split(',')
            .map(|token| token.trim().to_string())
            .filter(|token| !token.is_empty())
            .collect()),
    }
}

pub fn skills_dir() -> Option<PathBuf> {
    crate::config::agsh_config_dir().map(|dir| dir.join("skills"))
}

/// Discover all valid skills in the user's skills directory. Returns an empty
/// vec if the directory is missing or contains no valid skills.
pub fn discover_skills() -> Vec<Skill> {
    let Some(root) = skills_dir() else {
        return Vec::new();
    };
    discover_skills_in(&root)
}

/// Walk a specific skills root and parse every `SKILL.md`. Emits
/// `tracing::warn!` for each malformed entry; that warning behavior is the
/// signal the [`SkillCache`] relies on to surface broken-skill notices at
/// startup and only re-fire when the on-disk snapshot changes.
fn discover_skills_in(root: &Path) -> Vec<Skill> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(error) => {
            tracing::warn!("failed to read skills dir {}: {}", root.display(), error);
            return Vec::new();
        }
    };

    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        // Skip any dot-prefixed entry: VCS metadata (`.git`), editor/IDE
        // state (`.vscode`, `.idea`), filesystem artifacts (`.DS_Store`),
        // etc. None are real skills, and silently skipping them avoids
        // spurious "missing SKILL.md" warnings.
        if name.starts_with('.') {
            continue;
        }
        let name = name.to_string();

        let skill_file = path.join("SKILL.md");
        match load_skill_definition(&name, &path, &skill_file) {
            Ok(skill) => skills.push(skill),
            Err(error) => {
                tracing::warn!("skipping skill '{}': {}", name, error);
            }
        }
    }

    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Snapshot the disk state of a skills root: `subdir/SKILL.md → mtime` for
/// every non-dot subdirectory. Used by [`SkillCache`] to decide whether to
/// re-run discovery on the next turn.
///
/// Returns `None` when `read_dir` fails with anything other than `NotFound`
/// — that signals the caller to serve the cached (stale) state rather than
/// wiping it on a transient filesystem hiccup. A `NotFound` error maps to
/// `Some(empty)` so a deleted skills dir properly clears the cache.
fn disk_snapshot(root: &Path) -> Option<BTreeMap<PathBuf, SystemTime>> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Some(BTreeMap::new());
        }
        Err(_) => return None,
    };

    let mut map = BTreeMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let skill_file = path.join("SKILL.md");
        // Stat failure (file missing, perm denied) maps to UNIX_EPOCH so a
        // later stat-success transition forces a snapshot diff and reload.
        let mtime = std::fs::metadata(&skill_file)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        map.insert(skill_file, mtime);
    }
    Some(map)
}

/// Shared, atomically-swappable view of the skill list. Construction runs
/// an initial [`discover_skills_in`] pass so broken-skill warnings surface
/// during agent startup (above the first REPL prompt) instead of during
/// the first turn. Subsequent reads via [`SkillCache::current`] perform a
/// cheap mtime-snapshot check and only re-discover when the on-disk state
/// actually changed; identical broken-skill warnings naturally dedup
/// across turns because the inner walk is skipped when the snapshot is
/// stable.
pub struct SkillCache {
    /// Resolved skills root. `None` when [`skills_dir`] returns `None` or
    /// when constructed via `SkillCache::for_root(None)` for test
    /// scaffolding / subcommands that don't read skills.
    root: Option<PathBuf>,
    state: Mutex<CacheState>,
}

struct CacheState {
    skills: Arc<Vec<Skill>>,
    snapshot: BTreeMap<PathBuf, SystemTime>,
}

impl SkillCache {
    /// Production constructor. Resolves [`skills_dir`] and seeds the cache.
    pub fn discover() -> Arc<Self> {
        Self::for_root(skills_dir())
    }

    /// Construct a cache backed by a specific root. `None` produces a
    /// permanently-empty cache — useful for tests and for subcommands
    /// (`agsh tools list`) that don't read skill metadata.
    pub fn for_root(root: Option<PathBuf>) -> Arc<Self> {
        let (skills, snapshot) = match root.as_deref() {
            Some(root) => (
                discover_skills_in(root),
                disk_snapshot(root).unwrap_or_default(),
            ),
            None => (Vec::new(), BTreeMap::new()),
        };
        Arc::new(Self {
            root,
            state: Mutex::new(CacheState {
                skills: Arc::new(skills),
                snapshot,
            }),
        })
    }

    /// Return the current skill list, re-discovering first if the on-disk
    /// snapshot has changed since the last call. Cheap when nothing
    /// changed: one `read_dir` + N `metadata()` calls and a `BTreeMap`
    /// comparison, then an `Arc::clone` of the cached vec.
    pub async fn current(&self) -> Arc<Vec<Skill>> {
        let Some(root) = self.root.as_deref() else {
            return self.state.lock().await.skills.clone();
        };
        // Transient errors (e.g. EACCES on the dir) yield `None` — serve
        // stale state rather than wipe the cache.
        let Some(now) = disk_snapshot(root) else {
            return self.state.lock().await.skills.clone();
        };
        let mut state = self.state.lock().await;
        if state.snapshot != now {
            state.skills = Arc::new(discover_skills_in(root));
            state.snapshot = now;
        }
        state.skills.clone()
    }
}

fn load_skill_definition(
    name: &str,
    source_dir: &Path,
    skill_file: &Path,
) -> Result<Skill, String> {
    let content = std::fs::read_to_string(skill_file)
        .map_err(|error| format!("failed to read {}: {}", skill_file.display(), error))?;

    let (frontmatter_str, _body) =
        split_frontmatter(&content).ok_or_else(|| "missing YAML frontmatter".to_string())?;

    let frontmatter: Frontmatter = serde_yaml::from_str(frontmatter_str)
        .map_err(|error| format!("invalid frontmatter: {}", error))?;

    let description = frontmatter
        .description
        .filter(|description| !description.trim().is_empty())
        .ok_or_else(|| "missing required field 'description'".to_string())?;

    let when_to_use = frontmatter
        .when_to_use
        .filter(|when_to_use| !when_to_use.trim().is_empty())
        .ok_or_else(|| "missing required field 'when_to_use'".to_string())?;

    Ok(Skill {
        name: name.to_string(),
        source_dir: source_dir.to_path_buf(),
        description,
        when_to_use,
        allowed_tools: frontmatter.allowed_tools,
        version: frontmatter.version,
        user_invocable: frontmatter.user_invocable,
        body_path: skill_file.to_path_buf(),
    })
}

/// Split a file into (frontmatter, body) if it starts with a `---` fence.
/// Returns None when no valid frontmatter block is present.
fn split_frontmatter(content: &str) -> Option<(&str, &str)> {
    let rest = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"))?;

    for (marker, offset) in [("\n---\n", 5), ("\n---\r\n", 6)] {
        if let Some(end) = rest.find(marker) {
            let frontmatter = &rest[..end];
            let body = &rest[end + offset..];
            return Some((frontmatter, body));
        }
    }
    None
}

/// Load the body (post-frontmatter) of a skill and perform variable substitution.
pub fn load_skill_body(skill: &Skill, session_id: Option<&str>) -> Result<String, String> {
    let content = std::fs::read_to_string(&skill.body_path)
        .map_err(|error| format!("failed to read {}: {}", skill.body_path.display(), error))?;

    let body = split_frontmatter(&content)
        .map(|(_, body)| body.to_string())
        .unwrap_or(content);

    Ok(substitute_variables(&body, skill, session_id))
}

fn substitute_variables(text: &str, skill: &Skill, session_id: Option<&str>) -> String {
    let mut result = text.replace("${AGSH_SKILL_DIR}", &skill.source_dir.display().to_string());
    if let Some(id) = session_id {
        result = result.replace("${AGSH_SESSION_ID}", id);
    }
    result
}

/// Maximum length of a skill name. Kept small so the system-prompt
/// `## Skills` listing stays readable and per-line bounded.
pub const MAX_SKILL_NAME_LEN: usize = 64;

/// Validate that `name` is a safe filesystem-and-prompt-embeddable skill
/// identifier. Accepts `[A-Za-z0-9][A-Za-z0-9_-]*`, max
/// [`MAX_SKILL_NAME_LEN`] characters. Rejects anything that could escape
/// the skills directory (path components, hidden files) or break parsing
/// of the slash-command grammar (whitespace, `:`).
pub fn validate_skill_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("skill name cannot be empty".to_string());
    }
    if name.len() > MAX_SKILL_NAME_LEN {
        return Err(format!(
            "skill name '{}' exceeds {} characters",
            name, MAX_SKILL_NAME_LEN
        ));
    }
    let mut chars = name.chars();
    let first = chars.next().expect("non-empty checked above");
    if !first.is_ascii_alphanumeric() {
        return Err(format!(
            "skill name '{}' must start with a letter or digit",
            name
        ));
    }
    for ch in chars {
        if !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-') {
            return Err(format!(
                "skill name '{}' contains invalid character '{}'; only [A-Za-z0-9_-] are allowed",
                name, ch
            ));
        }
    }
    Ok(())
}

/// Resolve `~/.config/agsh/skills/<name>` for a given skill name.
/// Returns `None` if the agsh config directory cannot be determined.
/// Performs no I/O and does not validate the name — callers are expected
/// to call [`validate_skill_name`] first.
pub fn skill_dir_for(name: &str) -> Option<PathBuf> {
    skills_dir().map(|root| root.join(name))
}

/// Render the default `SKILL.md` template for a new skill. Optional
/// fields are emitted only when set, so the resulting file stays as
/// minimal as the user's input — `--user-invocable true` (the default)
/// is *not* written to keep the frontmatter lean.
pub fn render_template(
    name: &str,
    description: &str,
    when_to_use: &str,
    allowed_tools: &[String],
    version: Option<&str>,
    user_invocable: bool,
) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    out.push_str("---\n");
    let _ = writeln!(out, "description: {}", yaml_scalar(description));
    let _ = writeln!(out, "when_to_use: {}", yaml_scalar(when_to_use));
    if !allowed_tools.is_empty() {
        let joined = allowed_tools
            .iter()
            .map(|t| yaml_scalar(t))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "allowed_tools: [{}]", joined);
    }
    if let Some(v) = version {
        let _ = writeln!(out, "version: {}", yaml_scalar(v));
    }
    if !user_invocable {
        out.push_str("user_invocable: false\n");
    }
    out.push_str("---\n\n");
    let _ = writeln!(out, "# {}", name);
    out.push('\n');
    out.push_str(
        "Skill body. Use `${AGSH_SKILL_DIR}` to reference files bundled in this skill's\n\
         directory, or `${AGSH_SESSION_ID}` for the active session UUID.\n",
    );
    out
}

/// YAML-quote a scalar when it contains characters that would otherwise
/// require structural interpretation. Plain ASCII text without leading
/// punctuation, colons, or hash marks passes through unquoted.
fn yaml_scalar(text: &str) -> String {
    let needs_quotes = text.is_empty()
        || text.starts_with([
            '-', '?', ':', '!', '&', '*', '#', '|', '>', '%', '@', '`', '"', '\'',
        ])
        || text.contains(':')
        || text.contains('#')
        || text.contains('\n');
    if needs_quotes {
        let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{}\"", escaped)
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_skill(root: &Path, name: &str, skill_md: &str) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("create skill dir");
        std::fs::write(dir.join("SKILL.md"), skill_md).expect("write SKILL.md");
    }

    #[test]
    fn test_split_frontmatter_simple() {
        let content = "---\ndescription: hi\n---\nbody here\n";
        let (fm, body) = split_frontmatter(content).expect("should split");
        assert_eq!(fm, "description: hi");
        assert_eq!(body, "body here\n");
    }

    #[test]
    fn test_split_frontmatter_crlf() {
        let content = "---\r\ndescription: hi\r\n---\r\nbody\r\n";
        let split = split_frontmatter(content);
        assert!(split.is_some());
    }

    #[test]
    fn test_split_frontmatter_no_fence() {
        let content = "no frontmatter here\n";
        assert!(split_frontmatter(content).is_none());
    }

    #[test]
    fn test_load_valid_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "test-skill",
            "---\ndescription: A test skill\nwhen_to_use: For testing\n---\nBody content\n",
        );

        let skill_path = temp.path().join("test-skill");
        let skill_file = skill_path.join("SKILL.md");
        let skill =
            load_skill_definition("test-skill", &skill_path, &skill_file).expect("should load");

        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.description, "A test skill");
        assert_eq!(skill.when_to_use, "For testing");
        assert!(skill.user_invocable);
        assert!(skill.allowed_tools.is_empty());
    }

    #[test]
    fn test_load_skill_with_all_fields() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "full-skill",
            "---\n\
             description: Complete skill\n\
             when_to_use: All the time\n\
             allowed_tools: [read_file, execute_command]\n\
             version: \"1.2\"\n\
             user_invocable: false\n\
             ---\nBody\n",
        );

        let skill_path = temp.path().join("full-skill");
        let skill = load_skill_definition("full-skill", &skill_path, &skill_path.join("SKILL.md"))
            .expect("should load");

        assert_eq!(skill.allowed_tools, vec!["read_file", "execute_command"]);
        assert_eq!(skill.version.as_deref(), Some("1.2"));
        assert!(!skill.user_invocable);
    }

    #[test]
    fn test_allowed_tools_as_csv_string() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "csv-skill",
            "---\n\
             description: X\n\
             when_to_use: Y\n\
             allowed_tools: \"read_file, execute_command\"\n\
             ---\nBody\n",
        );

        let skill_path = temp.path().join("csv-skill");
        let skill = load_skill_definition("csv-skill", &skill_path, &skill_path.join("SKILL.md"))
            .expect("should load");

        assert_eq!(skill.allowed_tools, vec!["read_file", "execute_command"]);
    }

    #[test]
    fn test_missing_description_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "bad-skill",
            "---\nwhen_to_use: something\n---\nBody\n",
        );

        let skill_path = temp.path().join("bad-skill");
        let result = load_skill_definition("bad-skill", &skill_path, &skill_path.join("SKILL.md"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("description"));
    }

    #[test]
    fn test_missing_when_to_use_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "bad-skill",
            "---\ndescription: something\n---\nBody\n",
        );

        let skill_path = temp.path().join("bad-skill");
        let result = load_skill_definition("bad-skill", &skill_path, &skill_path.join("SKILL.md"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("when_to_use"));
    }

    #[test]
    fn test_no_frontmatter_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "no-fm", "Just body, no frontmatter\n");

        let skill_path = temp.path().join("no-fm");
        let result = load_skill_definition("no-fm", &skill_path, &skill_path.join("SKILL.md"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("frontmatter"));
    }

    #[test]
    fn test_malformed_yaml_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "bad-yaml",
            "---\ndescription: [unclosed\n---\nBody\n",
        );

        let skill_path = temp.path().join("bad-yaml");
        let result = load_skill_definition("bad-yaml", &skill_path, &skill_path.join("SKILL.md"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_skill_body_with_substitution() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "var-skill",
            "---\n\
             description: X\n\
             when_to_use: Y\n\
             ---\n\
             Path: ${AGSH_SKILL_DIR}\nSession: ${AGSH_SESSION_ID}\n",
        );

        let skill_path = temp.path().join("var-skill");
        let skill = load_skill_definition("var-skill", &skill_path, &skill_path.join("SKILL.md"))
            .expect("load");

        let body = load_skill_body(&skill, Some("abc-123")).expect("body");
        assert!(body.contains(&skill_path.display().to_string()));
        assert!(body.contains("Session: abc-123"));
    }

    #[test]
    fn test_load_skill_body_without_session_id() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "var-skill",
            "---\n\
             description: X\n\
             when_to_use: Y\n\
             ---\n\
             Session: ${AGSH_SESSION_ID}\n",
        );

        let skill_path = temp.path().join("var-skill");
        let skill = load_skill_definition("var-skill", &skill_path, &skill_path.join("SKILL.md"))
            .expect("load");

        let body = load_skill_body(&skill, None).expect("body");
        assert!(body.contains("Session: ${AGSH_SESSION_ID}"));
    }

    fn valid_frontmatter(description: &str) -> String {
        format!(
            "---\ndescription: {}\nwhen_to_use: when needed\n---\nBody\n",
            description
        )
    }

    /// Bump the mtime of a file far enough in the future to defeat 1-second
    /// filesystem resolution. Uses `File::set_modified` (stable since Rust
    /// 1.75) so no extra dep is required.
    fn bump_mtime(path: &Path) {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open for mtime bump");
        let future = SystemTime::now() + std::time::Duration::from_secs(10);
        file.set_modified(future).expect("set_modified");
    }

    #[tokio::test]
    async fn test_skill_cache_picks_up_new_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        assert!(cache.current().await.is_empty());

        write_skill(temp.path(), "foo", &valid_frontmatter("first"));

        let skills = cache.current().await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "foo");
    }

    #[tokio::test]
    async fn test_skill_cache_detects_modified_frontmatter() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "foo", &valid_frontmatter("old"));

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let skills = cache.current().await;
        assert_eq!(skills[0].description, "old");

        let skill_md = temp.path().join("foo").join("SKILL.md");
        std::fs::write(&skill_md, valid_frontmatter("new")).expect("rewrite");
        bump_mtime(&skill_md);

        let skills = cache.current().await;
        assert_eq!(skills[0].description, "new");
    }

    #[tokio::test]
    async fn test_skill_cache_drops_removed_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "foo", &valid_frontmatter("first"));

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        assert_eq!(cache.current().await.len(), 1);

        std::fs::remove_dir_all(temp.path().join("foo")).expect("rm skill");
        let skills = cache.current().await;
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn test_skill_cache_stable_when_unchanged() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "foo", &valid_frontmatter("first"));

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let first = cache.current().await;
        let second = cache.current().await;

        // Same Arc pointer ⇒ no rediscovery happened — proves the cache
        // really did skip the inner walk on the stable-snapshot path.
        assert!(
            Arc::ptr_eq(&first, &second),
            "expected cache to skip rediscovery when nothing changed"
        );
    }

    #[tokio::test]
    async fn test_skill_cache_with_no_root_is_empty() {
        let cache = SkillCache::for_root(None);
        assert!(cache.current().await.is_empty());
    }
}
