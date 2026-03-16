mod file;
mod search;
mod shell;
mod util;
mod web;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::permission::Permission;
use crate::provider::ToolDefinition;

#[derive(Debug)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn required_permission(&self) -> Permission;
    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput>;
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|tool| tool.definition().name == name)
            .map(|tool| tool.as_ref())
    }

    pub fn definitions_for_permission(&self, permission: Permission) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .filter(|tool| permission.allows(tool.required_permission()))
            .map(|tool| tool.definition())
            .collect()
    }

    pub fn build_default(
        user_agent: String,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
    ) -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(file::ReadFileTool));
        registry.register(Box::new(file::EditFileTool));
        registry.register(Box::new(file::WriteFileTool));
        registry.register(Box::new(search::FindFilesTool));
        registry.register(Box::new(search::SearchContentsTool));
        let web_client = reqwest::Client::builder()
            .user_agent(&user_agent)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        registry.register(Box::new(web::FetchUrlTool {
            client: web_client.clone(),
        }));
        registry.register(Box::new(web::WebSearchTool { client: web_client }));
        registry.register(Box::new(shell::ExecuteCommandTool {
            sandbox_capability,
            shared_permission,
            sandbox_enabled,
        }));
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_shared_permission() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(Permission::Write)
    }

    #[test]
    fn test_tool_registry() {
        let registry = ToolRegistry::build_default(
            "test-agent/0.1".to_string(),
            test_shared_permission(),
            true,
            crate::sandbox::detect(),
        );
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(registry.get("edit_file").is_some());
        assert!(registry.get("find_files").is_some());
        assert!(registry.get("search_contents").is_some());
        assert!(registry.get("execute_command").is_some());
        assert!(registry.get("fetch_url").is_some());
        assert!(registry.get("web_search").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_permission_filtering() {
        let registry = ToolRegistry::build_default(
            "test-agent/0.1".to_string(),
            test_shared_permission(),
            true,
            crate::sandbox::detect(),
        );

        let none_tools = registry.definitions_for_permission(Permission::None);
        assert!(none_tools.is_empty());

        let read_tools = registry.definitions_for_permission(Permission::Read);
        assert!(read_tools.iter().any(|t| t.name == "read_file"));
        assert!(read_tools.iter().any(|t| t.name == "find_files"));
        assert!(read_tools.iter().any(|t| t.name == "execute_command"));
        assert!(!read_tools.iter().any(|t| t.name == "write_file"));

        let write_tools = registry.definitions_for_permission(Permission::Write);
        assert!(write_tools.iter().any(|t| t.name == "read_file"));
        assert!(write_tools.iter().any(|t| t.name == "write_file"));
        assert!(write_tools.iter().any(|t| t.name == "execute_command"));
    }
}
