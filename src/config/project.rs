//! Projects: [`ProjectConfig`] as `config.toml` writes a `[projects.<name>]` table, the view of
//! servers and skills it gives its sessions, the summary a listing shows, and the name rule every
//! project passes.

use std::collections::{BTreeMap, HashSet};

use serde::Deserialize;

/// One named project from `[projects.<name>]`: a work context a session opens by name.
///
/// A project is a memory tier. A session on a project sees the global memories and the project's
/// own, and a memory it writes files into the project unless the model asks for global. Nothing
/// selects a project but an explicit act (`--project`, `/project`, the session's `project` field
/// over HTTP and ACP), so a project is an idea the operator names, bound to no directory.
///
/// Defined in config rather than in the store because a project that carries a setting (which
/// servers and skills its sessions see) must be written where settings are written, and one
/// placement serves a project with memory alone as well. An empty table is a complete project.
///
/// The four lists are the view side of the house pair, as `[mcp.servers]`'s `allowed_tools` and
/// `disabled_tools` are: an `allowed_` list is the whole view named in full, a `disabled_` list
/// subtracts from it, and unset is everything installed. Nothing on a server's table or in a
/// skill's frontmatter names a project, so a skill file stays portable and a server stays one
/// entry.
#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProjectConfig {
    /// The MCP servers a session on this project sees, named in full; unset, every configured
    /// server.
    pub(crate) allowed_servers: Option<Vec<String>>,
    /// The MCP servers a session on this project does not see, taken out of the view above.
    pub(crate) disabled_servers: Option<Vec<String>>,
    /// The skills a session on this project sees, named in full; unset, every installed skill.
    pub(crate) allowed_skills: Option<Vec<String>>,
    /// The skills a session on this project does not see, taken out of the view above.
    pub(crate) disabled_skills: Option<Vec<String>>,
}

/// What a session on a project sees of the installed servers and skills: the one predicate every
/// door that lists, loads or runs one asks, so a server a project hides is hidden from the tool
/// catalog, `tool_load`, the MCP meta-tools and a sub-agent's scope alike. A session on no
/// project sees everything; one on a project the file does not define sees no server and no
/// skill.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ProjectAccess {
    /// The project the view is of, for a refusal that names it; `None` for a session on none.
    project: Option<String>,
    allowed_servers: Option<HashSet<String>>,
    disabled_servers: HashSet<String>,
    allowed_skills: Option<HashSet<String>>,
    disabled_skills: HashSet<String>,
}

impl ProjectAccess {
    /// The view of the project `project` names among `projects`; everything when it names none,
    /// by `None` or by the empty string a row records for none.
    pub(crate) fn of(projects: &BTreeMap<String, ProjectConfig>, project: Option<&str>) -> Self {
        let Some(name) = project.filter(|name| !name.is_empty()) else {
            return Self::default();
        };
        let Some(config) = projects.get(name) else {
            // Every door that puts a name in a session's cell refuses one the file lacks, so this
            // is reached only by a door that forgot, and a view narrowed to nothing is noticed
            // where one widened to everything would not be.
            return Self {
                project: Some(name.to_string()),
                allowed_servers: Some(HashSet::new()),
                disabled_servers: HashSet::new(),
                allowed_skills: Some(HashSet::new()),
                disabled_skills: HashSet::new(),
            };
        };
        let set = |names: &Option<Vec<String>>| -> Option<HashSet<String>> {
            names.as_ref().map(|names| names.iter().cloned().collect())
        };
        Self {
            project: Some(name.to_string()),
            allowed_servers: set(&config.allowed_servers),
            disabled_servers: set(&config.disabled_servers).unwrap_or_default(),
            allowed_skills: set(&config.allowed_skills),
            disabled_skills: set(&config.disabled_skills).unwrap_or_default(),
        }
    }

    /// The project the view is of, or `None` for a session on none.
    pub(crate) fn project(&self) -> Option<&str> {
        self.project.as_deref()
    }

    /// Whether a session on the project sees the skill `name`.
    pub(crate) fn sees_skill(&self, name: &str) -> bool {
        !self.disabled_skills.contains(name)
            && self
                .allowed_skills
                .as_ref()
                .is_none_or(|skills| skills.contains(name))
    }

    /// Whether the view narrows the skills at all, so a reader can hand back the index it has.
    pub(crate) fn sees_every_skill(&self) -> bool {
        self.allowed_skills.is_none() && self.disabled_skills.is_empty()
    }

    /// The servers named in full, or `None` for every configured one.
    pub(crate) fn allowed_servers(&self) -> Option<&HashSet<String>> {
        self.allowed_servers.as_ref()
    }

    /// The servers taken out of the view.
    pub(crate) fn disabled_servers(&self) -> &HashSet<String> {
        &self.disabled_servers
    }

    /// Every skill name the view mentions, for the check that each is installed.
    pub(crate) fn skill_names(&self) -> impl Iterator<Item = &String> {
        self.allowed_skills
            .iter()
            .flatten()
            .chain(self.disabled_skills.iter())
    }
}

/// Refuse a project view that names a server no `[[mcp.servers]]` entry defines, or a name in
/// both lists of one pair. A server name that matches nothing restricts nothing, and the reader
/// gets no signal: they asked for a narrowed project and would get a wider one believing
/// otherwise. Skills are not checked here, since they come and go without a config edit; the
/// session that opens on the project says which are missing.
pub(crate) fn validate_project_views<'a>(
    projects: &BTreeMap<String, ProjectConfig>,
    servers: impl IntoIterator<Item = &'a str>,
) -> Result<(), String> {
    let servers: HashSet<&str> = servers.into_iter().collect();
    for (project, config) in projects {
        for (key, names) in [
            ("allowed_servers", &config.allowed_servers),
            ("disabled_servers", &config.disabled_servers),
        ] {
            if let Some(unknown) = names
                .iter()
                .flatten()
                .find(|name| !servers.contains(name.as_str()))
            {
                return Err(format!(
                    "`[projects.{project}].{key}` names '{unknown}', which no `[[mcp.servers]]` \
                     entry defines"
                ));
            }
        }
        for (pair, allowed, disabled) in [
            ("servers", &config.allowed_servers, &config.disabled_servers),
            ("skills", &config.allowed_skills, &config.disabled_skills),
        ] {
            if let Some(both) = allowed.iter().flatten().find(|name| {
                disabled
                    .as_ref()
                    .is_some_and(|disabled| disabled.contains(name))
            }) {
                return Err(format!(
                    "`[projects.{project}]` names '{both}' in both allowed_{pair} and \
                     disabled_{pair}; remove it from one"
                ));
            }
        }
    }
    Ok(())
}

/// One configured project as a listing or a picker shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProjectSummary {
    pub(crate) name: String,
}

/// The project a name denotes, or [`crate::error::MekaError::ProjectNotConfigured`] naming the
/// ones there are. Every door that puts a session on a project asks this first, so a row never
/// names a project the file lacks except by the file changing afterwards.
pub(crate) fn require_project<'a>(
    name: &str,
    projects: &'a BTreeMap<String, ProjectConfig>,
) -> crate::error::Result<&'a ProjectConfig> {
    projects
        .get(name)
        .ok_or_else(|| crate::error::MekaError::ProjectNotConfigured {
            name: name.to_string(),
            known: projects.keys().cloned().collect(),
        })
}

/// Refuse a `[projects.*]` key that is not a valid project name, or two that differ only by case.
///
/// A project name is recorded on session and memory rows and compared exactly, the way a profile
/// name is, so two names equal under case folding would be two projects a user cannot tell apart
/// in a listing and a shell cannot always distinguish on disk; the rule refuses them at the file.
pub(crate) fn validate_project_names<'a>(
    names: impl IntoIterator<Item = &'a String>,
) -> Result<(), String> {
    let mut seen_folded: BTreeMap<String, &str> = BTreeMap::new();
    for name in names {
        crate::entry::validate_entry_name(name, "project")
            .map_err(|error| format!("`[projects]`: {error}"))?;
        if let Some(other) = seen_folded.insert(name.to_ascii_lowercase(), name) {
            return Err(format!(
                "projects '{other}' and '{name}' differ only by case; rename one"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_project_name_follows_the_entry_rule() {
        let names = ["webdev".to_string(), "Movie_2".to_string()];
        validate_project_names(&names).expect("plain names pass");
        let error =
            validate_project_names(&["web dev".to_string()]).expect_err("a space is refused");
        assert!(
            error.contains("`[projects]`") && error.contains("'web dev'"),
            "{error}"
        );
        let error = validate_project_names(&[String::new()]).expect_err("empty is refused");
        assert!(error.contains("cannot be empty"), "{error}");
    }

    #[test]
    fn two_projects_that_differ_only_by_case_are_refused() {
        let names = ["Webdev".to_string(), "webdev".to_string()];
        let error = validate_project_names(&names).expect_err("case-only duplicates are refused");
        assert!(
            error.contains("'Webdev'")
                && error.contains("'webdev'")
                && error.contains("rename one"),
            "{error}"
        );
    }

    fn projects_from(toml: &str) -> BTreeMap<String, ProjectConfig> {
        #[derive(Deserialize)]
        struct Probe {
            projects: BTreeMap<String, ProjectConfig>,
        }
        toml::from_str::<Probe>(toml).expect("parse").projects
    }

    /// Unset is everything, an allow-list is the whole view, a disabled list subtracts, a session
    /// on no project sees everything, and one on a project the file lacks sees nothing of either.
    #[test]
    fn a_project_s_view_follows_the_allowed_and_disabled_pairs() {
        let projects = projects_from(
            "[projects.open]\n[projects.narrow]\nallowed_servers = [\"fs\", \"exa\"]\n\
             disabled_servers = [\"exa\"]\nallowed_skills = [\"deploy\"]\n\
             [projects.minus]\ndisabled_skills = [\"release\"]\n",
        );
        let open = ProjectAccess::of(&projects, Some("open"));
        assert!(open.allowed_servers().is_none() && open.disabled_servers().is_empty());
        assert!(open.sees_skill("anything") && open.sees_every_skill());
        let narrow = ProjectAccess::of(&projects, Some("narrow"));
        assert_eq!(
            narrow.project(),
            Some("narrow"),
            "a refusal names the project"
        );
        assert_eq!(open.project(), Some("open"));
        assert_eq!(
            narrow.allowed_servers().map(|servers| servers.len()),
            Some(2),
            "an allow-list is the whole view"
        );
        assert!(narrow.disabled_servers().contains("exa"));
        assert!(narrow.sees_skill("deploy") && !narrow.sees_skill("release"));
        assert!(!narrow.sees_every_skill());
        let minus = ProjectAccess::of(&projects, Some("minus"));
        assert!(minus.allowed_servers().is_none());
        assert!(minus.sees_skill("deploy") && !minus.sees_skill("release"));
        assert_eq!(minus.skill_names().cloned().collect::<Vec<_>>(), [
            "release"
        ]);
        assert!(ProjectAccess::of(&projects, None).sees_skill("release"));
        assert!(ProjectAccess::of(&projects, None).project().is_none());
        assert!(ProjectAccess::of(&projects, Some("")).sees_every_skill());
        assert!(ProjectAccess::of(&projects, Some("")).project().is_none());
        let gone = ProjectAccess::of(&projects, Some("gone"));
        assert!(!gone.sees_skill("deploy") && !gone.sees_every_skill());
        assert_eq!(gone.allowed_servers().map(HashSet::len), Some(0));
    }

    /// A server the file does not configure, and a name in both lists of a pair, are refused with
    /// the project and the key named.
    #[test]
    fn a_project_view_naming_an_unknown_server_or_both_lists_is_refused() {
        let projects = projects_from("[projects.p]\nallowed_servers = [\"fs\", \"ghost\"]\n");
        let error = validate_project_views(&projects, ["fs"]).expect_err("unknown server");
        assert!(
            error.contains("`[projects.p].allowed_servers` names 'ghost'"),
            "{error}"
        );
        validate_project_views(&projects, ["fs", "ghost"]).expect("configured");
        let projects = projects_from(
            "[projects.p]\nallowed_skills = [\"a\", \"b\"]\ndisabled_skills = [\"b\"]\n",
        );
        let error = validate_project_views(&projects, []).expect_err("both lists");
        assert!(
            error.contains("names 'b' in both allowed_skills and disabled_skills"),
            "{error}"
        );
        let projects = projects_from(
            "[projects.p]\nallowed_servers = [\"fs\"]\ndisabled_servers = [\"fs\"]\n",
        );
        assert!(validate_project_views(&projects, ["fs"]).is_err());
    }

    #[test]
    fn an_empty_table_is_a_project_and_an_unknown_key_is_refused() {
        #[derive(Debug, Deserialize)]
        struct Probe {
            projects: BTreeMap<String, ProjectConfig>,
        }
        let probe: Probe = toml::from_str("[projects.webdev]\n").expect("empty table parses");
        assert!(probe.projects.contains_key("webdev"));
        let error = toml::from_str::<Probe>("[projects.webdev]\nservers = [\"a\"]\n")
            .expect_err("an unmodeled key is refused");
        assert!(error.to_string().contains("servers"), "{error}");
    }
}
