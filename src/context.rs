use crate::permission::Permission;
use crate::provider::ToolDefinition;
use crate::session::ToolOutputSummary;
use crate::skills::Skill;
use crate::tools::todo::{self, TodoItem};

/// Build the static, session-level system prompt: role, permission description,
/// user instructions, tools, skills, guidelines, and environment info.
pub fn build_system_prompt(
    permission: Permission,
    tools: &[ToolDefinition],
    sandboxed_shell: bool,
    deferred_tools: &[(String, String)],
    skills: &[Skill],
    user_instructions: Option<&str>,
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

    if let Some(instructions) = user_instructions
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        prompt.push_str("## User Instructions\n\n");
        prompt.push_str(
            "These are installation-specific rules set by the user. Treat them as \
             hard constraints unless they conflict with safety requirements.\n\n",
        );
        prompt.push_str(instructions);
        prompt.push_str("\n\n");
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

/// Build the per-turn environment context block (pwd, date).
/// Returns an empty string in `None` permission mode so system info isn't leaked.
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

/// Build the `<context>...</context>` block that wraps per-turn user input with
/// the active todo list and environment info. Returns `None` when neither
/// contributes anything (empty todos in `None` mode).
pub fn build_turn_context(permission: Permission, todos: &[TodoItem]) -> Option<String> {
    let todo_context = if todos.is_empty() {
        String::new()
    } else {
        todo::format_todo_for_context(todos)
    };
    let environment_context = build_environment_context(permission);
    let body = format!("{}{}", todo_context, environment_context);
    if body.trim().is_empty() {
        None
    } else {
        Some(format!("<context>\n{}</context>", body))
    }
}

/// Build the post-compaction context block summarizing live session state
/// (environment, todos, scratchpad inventory) that must persist across the
/// compacted message window.
pub fn build_post_compact_context(
    permission: Permission,
    todos: &[TodoItem],
    scratchpad_entries: &[ToolOutputSummary],
) -> String {
    let mut parts = Vec::new();

    let env = build_environment_context(permission);
    if !env.is_empty() {
        parts.push(env);
    }

    if !todos.is_empty() {
        parts.push(todo::format_todo_for_context(todos));
    }

    if !scratchpad_entries.is_empty() {
        let mut listing = String::from("[Scratchpad entries]\n");
        for entry in scratchpad_entries {
            listing.push_str(&format!(
                "- \"{}\" ({})\n",
                entry.name,
                crate::tools::scratchpad::format_size(entry.size),
            ));
        }
        parts.push(listing);
    }

    parts.join("\n")
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

    fn sample_todo(id: &str, description: &str, status: &str) -> TodoItem {
        TodoItem {
            id: id.to_string(),
            description: description.to_string(),
            status: status.to_string(),
        }
    }

    fn sample_scratchpad_entry(name: &str, size: usize) -> ToolOutputSummary {
        ToolOutputSummary {
            name: name.to_string(),
            size,
            created_at: "2026-04-17T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_system_prompt_none_mode() {
        let prompt = build_system_prompt(Permission::None, &[], false, &[], &[], None);
        assert!(prompt.contains("NO tools available"));
        assert!(prompt.contains("Shift+Tab"));
        assert!(!prompt.contains("## Environment"));
    }

    #[test]
    fn test_system_prompt_read_mode_with_sandbox() {
        let prompt = build_system_prompt(Permission::Read, &[], true, &[], &[], None);
        assert!(prompt.contains("READ-ONLY"));
        assert!(prompt.contains("read-only sandboxed"));
        assert!(prompt.contains("CANNOT write"));
    }

    #[test]
    fn test_system_prompt_read_mode_without_sandbox() {
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &[], None);
        assert!(prompt.contains("READ-ONLY"));
        assert!(prompt.contains("CANNOT write"));
        assert!(prompt.contains("execute shell commands"));
        assert!(!prompt.contains("sandboxed"));
    }

    #[test]
    fn test_system_prompt_write_mode() {
        let prompt = build_system_prompt(Permission::Write, &[], false, &[], &[], None);
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

        let prompt = build_system_prompt(Permission::Write, &tools, false, &[], &[], None);
        assert!(prompt.contains("read_file"));
        assert!(prompt.contains("execute_command"));
        assert!(prompt.contains("Available Tools"));
    }

    #[test]
    fn test_system_prompt_has_environment() {
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &[], None);
        assert!(prompt.contains("Environment"));
        assert!(!prompt.contains("Working Directory:"));
        assert!(!prompt.contains("Date:"));
    }

    #[test]
    fn test_system_prompt_lists_skills() {
        let skills = vec![sample_skill("setup-server"), sample_skill("deploy-app")];
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &skills, None);
        assert!(prompt.contains("## Skills"));
        assert!(prompt.contains("**setup-server**"));
        assert!(prompt.contains("setup-server description"));
        assert!(prompt.contains("setup-server use case"));
        assert!(prompt.contains("**deploy-app**"));
    }

    #[test]
    fn test_system_prompt_omits_skills_in_none_mode() {
        let skills = vec![sample_skill("setup-server")];
        let prompt = build_system_prompt(Permission::None, &[], false, &[], &skills, None);
        assert!(!prompt.contains("## Skills"));
        assert!(!prompt.contains("setup-server"));
    }

    #[test]
    fn test_system_prompt_omits_skills_section_when_empty() {
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &[], None);
        assert!(!prompt.contains("## Skills"));
    }

    #[test]
    fn test_system_prompt_includes_user_instructions() {
        let prompt = build_system_prompt(
            Permission::Read,
            &[],
            false,
            &[],
            &[],
            Some("Never use pip. Always prefer uv."),
        );
        assert!(prompt.contains("## User Instructions"));
        assert!(prompt.contains("Never use pip. Always prefer uv."));
        assert!(prompt.contains("installation-specific rules"));
    }

    #[test]
    fn test_system_prompt_includes_user_instructions_in_none_mode() {
        let prompt = build_system_prompt(Permission::None, &[], false, &[], &[], Some("Rule X"));
        assert!(prompt.contains("## User Instructions"));
        assert!(prompt.contains("Rule X"));
    }

    #[test]
    fn test_system_prompt_omits_user_instructions_when_none() {
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &[], None);
        assert!(!prompt.contains("## User Instructions"));
    }

    #[test]
    fn test_system_prompt_omits_user_instructions_when_whitespace() {
        let prompt = build_system_prompt(Permission::Read, &[], false, &[], &[], Some("   \n"));
        assert!(!prompt.contains("## User Instructions"));
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

    #[test]
    fn test_turn_context_empty_in_none_mode_with_no_todos() {
        assert!(build_turn_context(Permission::None, &[]).is_none());
    }

    #[test]
    fn test_turn_context_has_environment_in_read_mode() {
        let context = build_turn_context(Permission::Read, &[]).expect("should have content");
        assert!(context.starts_with("<context>\n"));
        assert!(context.ends_with("</context>"));
        assert!(context.contains("[Environment context]"));
    }

    #[test]
    fn test_turn_context_includes_todos() {
        let todos = vec![sample_todo("1", "write tests", "in_progress")];
        let context = build_turn_context(Permission::Read, &todos).expect("should have content");
        assert!(context.contains("write tests"));
        assert!(context.contains("[Environment context]"));
    }

    #[test]
    fn test_turn_context_none_mode_with_todos_still_includes_todos() {
        let todos = vec![sample_todo("1", "do a thing", "pending")];
        let context = build_turn_context(Permission::None, &todos).expect("should have content");
        assert!(context.contains("do a thing"));
        assert!(!context.contains("[Environment context]"));
    }

    #[test]
    fn test_post_compact_context_empty_in_none_mode_no_state() {
        let result = build_post_compact_context(Permission::None, &[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_post_compact_context_includes_env_todos_scratchpad() {
        let todos = vec![sample_todo("1", "keep working", "pending")];
        let entries = vec![sample_scratchpad_entry("notes", 1024)];
        let result = build_post_compact_context(Permission::Read, &todos, &entries);
        assert!(result.contains("[Environment context]"));
        assert!(result.contains("keep working"));
        assert!(result.contains("[Scratchpad entries]"));
        assert!(result.contains("\"notes\""));
    }

    #[test]
    fn test_post_compact_context_scratchpad_only() {
        let entries = vec![sample_scratchpad_entry("log", 500)];
        let result = build_post_compact_context(Permission::None, &[], &entries);
        assert!(result.contains("[Scratchpad entries]"));
        assert!(result.contains("\"log\""));
        assert!(!result.contains("[Environment context]"));
    }
}
