//! Skill discovery and loading. Walks `~/.config/agsh/skills/<name>/SKILL.md`,
//! parses the YAML frontmatter (description, when_to_use, allowed_tools,
//! version, user_invocable), and exposes the resulting [`Skill`] structs to
//! the agent for system-prompt injection and `skill` tool dispatch.

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub source_dir: PathBuf,
    pub description: String,
    pub when_to_use: String,
    pub allowed_tools: Vec<String>,
    // Parsed but not yet consumed — planned for future features
    // (version reporting, user-invocable gating via `/skills` command).
    #[allow(dead_code)]
    pub version: Option<String>,
    #[allow(dead_code)]
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

    let entries = match std::fs::read_dir(&root) {
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
}
