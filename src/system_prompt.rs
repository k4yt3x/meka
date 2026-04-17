use crate::permission::Permission;
use crate::provider::ToolDefinition;
use crate::skills::Skill;

pub fn build_system_prompt(
    permission: Permission,
    tools: &[ToolDefinition],
    sandboxed_shell: bool,
    deferred_tools: &[(String, String)],
    skills: &[Skill],
) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are agsh, an agentic shell assistant. The user communicates with you \
         in natural language, and you execute their requests using the available tools.\n\n",
    );

    prompt.push_str(&format!("## Current Permission Level: {}\n\n", permission));

    match permission {
        Permission::None => {
            prompt.push_str(
                "You have NO tools available. You can only respond with text. \
                 If the user asks you to perform an action, inform them that the current \
                 permission mode does not allow it and suggest they press Shift+Tab \
                 to cycle to a higher permission level.\n\n",
            );
        }
        Permission::Read => {
            if sandboxed_shell {
                prompt.push_str(
                    "You can use READ-ONLY tools: reading files, searching files and contents, \
                     fetching web pages, searching the web, and executing shell commands in a \
                     read-only sandboxed environment. Shell commands run with the filesystem \
                     mounted read-only — you can run commands like `ls`, `cat`, `df`, `ps`, \
                     `uname`, `grep`, `find`, `git log`, `git diff`, etc., but any command \
                     that writes to the filesystem will fail. You CANNOT write files or edit \
                     files directly. If the user asks you to perform a write operation, inform \
                     them that the current permission mode does not allow it and suggest they \
                     press Shift+Tab to cycle to 'write' mode.\n\n",
                );
            } else {
                prompt.push_str(
                    "You can use READ-ONLY tools: reading files, searching files and contents, \
                     fetching web pages, and searching the web. You CANNOT write files, \
                     edit files, or execute shell commands. If the user asks you to perform \
                     a write operation, inform them that the current permission mode does not \
                     allow it and suggest they press Shift+Tab to cycle to 'write' mode.\n\n",
                );
            }
        }
        Permission::Ask => {
            prompt.push_str(
                "You have access to all tools, but the user will be prompted to approve \
                 or deny each tool call before it executes. Proceed normally — the user \
                 will see each tool invocation and decide whether to allow it.\n\n",
            );
        }
        Permission::Write => {
            prompt.push_str(
                "You have FULL access to all tools, including file writing, editing, \
                 and shell command execution. For potentially destructive operations \
                 (e.g., deleting files, overwriting data, running dangerous commands), \
                 briefly explain what you will do before proceeding.\n\n",
            );
        }
    }

    if !tools.is_empty() {
        prompt.push_str("## Available Tools\n\n");
        for tool in tools {
            prompt.push_str(&format!("- **{}**: {}\n", tool.name, tool.description));
        }
        prompt.push('\n');
    }

    if !deferred_tools.is_empty() {
        prompt.push_str("## Additional Tools (loaded on first use)\n\n");
        prompt.push_str("These tools are available but their schemas are loaded on demand. ");
        prompt.push_str("Call them by name and they will be activated.\n\n");
        for (name, description) in deferred_tools {
            prompt.push_str(&format!("- **{}**: {}\n", name, description));
        }
        prompt.push('\n');
    }

    prompt.push_str("## Guidelines\n\n");
    prompt.push_str("- Format your responses in Markdown.\n");
    prompt.push_str("- When executing shell commands, show the command you are about to run.\n");
    prompt.push_str(
        "- For potentially destructive operations, explain what you will do before proceeding.\n",
    );
    prompt.push_str(
        "- If a tool returns an error, explain the error to the user and suggest alternatives.\n",
    );
    prompt.push_str("- Be concise but thorough.\n\n");

    if !matches!(permission, Permission::None) && !skills.is_empty() {
        prompt.push_str("## Skills\n\n");
        prompt.push_str(
            "The following skills are available. Call the `skill` tool with the \
             skill name to load its full content. Only invoke a skill when the \
             user's request matches its stated purpose.\n\n",
        );
        for skill in skills {
            prompt.push_str(&format!(
                "- **{}**: {} — {}\n",
                skill.name, skill.description, skill.when_to_use
            ));
        }
        prompt.push('\n');
    }

    if !matches!(permission, Permission::None) {
        prompt.push_str("## Environment\n\n");

        if let Ok(shell) = std::env::var("SHELL") {
            prompt.push_str(&format!("- Shell: {}\n", shell));
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(info) = std::fs::read_to_string("/etc/os-release") {
                for line in info.lines() {
                    if let Some(name) = line.strip_prefix("PRETTY_NAME=") {
                        let name = name.trim_matches('"');
                        prompt.push_str(&format!("- OS: {}\n", name));
                        break;
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("sw_vers")
                .arg("-productVersion")
                .output()
            {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !version.is_empty() {
                    prompt.push_str(&format!("- OS: macOS {}\n", version));
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            prompt.push_str("- OS: Windows\n");
        }
    }

    prompt
}

pub fn build_environment_context(permission: Permission) -> String {
    if permission == Permission::None {
        return String::new();
    }

    let mut context = String::from("[Environment context]\n");

    if let Ok(cwd) = std::env::current_dir() {
        context.push_str(&format!("Working directory: {}\n", cwd.display()));
    }

    let now = chrono::Local::now().to_rfc2822();
    context.push_str(&format!("Date: {}\n", now));

    context
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_skill(name: &str) -> Skill {
        Skill {
            name: name.to_string(),
            source_dir: std::path::PathBuf::from("/tmp").join(name),
            description: format!("{} description", name),
            when_to_use: format!("{} use case", name),
            allowed_tools: Vec::new(),
            version: None,
            user_invocable: true,
            body_path: std::path::PathBuf::from("/tmp").join(name).join("SKILL.md"),
        }
    }

    #[test]
    fn test_system_prompt_none_mode() {
        let prompt = build_system_prompt(Permission::None, &[], false, &[], &[]);
        assert!(prompt.contains("NO tools available"));
        assert!(prompt.contains("Shift+Tab"));
        assert!(!prompt.contains("## Environment"));
    }

    #[test]
    fn test_system_prompt_read_mode_with_sandbox() {
        let prompt = build_system_prompt(Permission::Read, &[], true, &[], &[]);
        assert!(prompt.contains("READ-ONLY"));
        assert!(prompt.contains("read-only sandboxed"));
        assert!(prompt.contains("CANNOT write"));
    }

    #[test]
    fn test_system_prompt_read_mode_without_sandbox() {
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &[]);
        assert!(prompt.contains("READ-ONLY"));
        assert!(prompt.contains("CANNOT write"));
        assert!(prompt.contains("execute shell commands"));
        assert!(!prompt.contains("sandboxed"));
    }

    #[test]
    fn test_system_prompt_write_mode() {
        let prompt = build_system_prompt(Permission::Write, &[], false, &[], &[]);
        assert!(prompt.contains("FULL access"));
        assert!(prompt.contains("destructive"));
    }

    #[test]
    fn test_system_prompt_with_tools() {
        let tools = vec![
            ToolDefinition {
                name: "read_file".to_string(),
                description: "Read file contents".to_string(),
                parameters: serde_json::json!({}),
            },
            ToolDefinition {
                name: "execute_command".to_string(),
                description: "Run a shell command".to_string(),
                parameters: serde_json::json!({}),
            },
        ];

        let prompt = build_system_prompt(Permission::Write, &tools, false, &[], &[]);
        assert!(prompt.contains("read_file"));
        assert!(prompt.contains("execute_command"));
        assert!(prompt.contains("Available Tools"));
    }

    #[test]
    fn test_system_prompt_has_environment() {
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &[]);
        assert!(prompt.contains("Environment"));
        assert!(!prompt.contains("Working Directory:"));
        assert!(!prompt.contains("Date:"));
    }

    #[test]
    fn test_system_prompt_lists_skills() {
        let skills = vec![sample_skill("setup-server"), sample_skill("deploy-app")];
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &skills);
        assert!(prompt.contains("## Skills"));
        assert!(prompt.contains("**setup-server**"));
        assert!(prompt.contains("setup-server description"));
        assert!(prompt.contains("setup-server use case"));
        assert!(prompt.contains("**deploy-app**"));
    }

    #[test]
    fn test_system_prompt_omits_skills_in_none_mode() {
        let skills = vec![sample_skill("setup-server")];
        let prompt = build_system_prompt(Permission::None, &[], false, &[], &skills);
        assert!(!prompt.contains("## Skills"));
        assert!(!prompt.contains("setup-server"));
    }

    #[test]
    fn test_system_prompt_omits_skills_section_when_empty() {
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &[]);
        assert!(!prompt.contains("## Skills"));
    }

    #[test]
    fn test_environment_context() {
        let context = build_environment_context(Permission::Read);
        assert!(context.contains("[Environment context]"));
        assert!(context.contains("Working directory:"));
        assert!(context.contains("Date:"));
    }

    #[test]
    fn test_environment_context_none_mode() {
        let context = build_environment_context(Permission::None);
        assert!(context.is_empty());
    }
}
