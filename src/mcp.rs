use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::config::{McpServerConfig, McpTransport};
use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::ToolDefinition;
use crate::tools::{Tool, ToolOutput};

type McpRunningService = rmcp::service::RunningService<rmcp::RoleClient, ()>;

pub struct McpClientManager {
    servers: HashMap<String, Arc<McpRunningService>>,
}

impl McpClientManager {
    pub async fn connect_all(configs: &[McpServerConfig]) -> Result<Self> {
        let mut servers = HashMap::new();

        for config in configs {
            if config.name.is_empty() {
                return Err(AgshError::McpConnection {
                    server_name: "(empty)".to_string(),
                    message: "server name must not be empty".to_string(),
                });
            }

            if config.name.contains("__") {
                return Err(AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: "server name must not contain '__' (reserved as namespace separator)"
                        .to_string(),
                });
            }

            if servers.contains_key(&config.name) {
                return Err(AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: "duplicate server name".to_string(),
                });
            }

            let service = connect_server(config).await?;
            tracing::info!("connected to MCP server '{}'", config.name);
            servers.insert(config.name.clone(), Arc::new(service));
        }

        Ok(Self { servers })
    }

    pub async fn discover_tools_for_server(
        &self,
        server_name: &str,
        permission_str: Option<&str>,
    ) -> Result<Vec<McpToolAdapter>> {
        let service = self
            .servers
            .get(server_name)
            .ok_or_else(|| AgshError::McpConnection {
                server_name: server_name.to_string(),
                message: "server not found".to_string(),
            })?;

        let permission = parse_server_permission(server_name, permission_str)?;

        let tools =
            service
                .peer()
                .list_all_tools()
                .await
                .map_err(|error| AgshError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!("list_tools failed: {}", error),
                })?;

        let mut adapters = Vec::new();
        for tool in tools {
            let namespaced_name = format!("{}__{}", server_name, tool.name);
            let description = tool.description.map(|d| d.into_owned()).unwrap_or_default();
            let parameters = serde_json::to_value(&*tool.input_schema)
                .unwrap_or_else(|_| serde_json::json!({"type": "object", "properties": {}}));

            adapters.push(McpToolAdapter {
                namespaced_name,
                server_name: server_name.to_string(),
                remote_tool_name: tool.name.into_owned(),
                description,
                parameters,
                permission,
                service: Arc::clone(service),
            });
        }

        Ok(adapters)
    }

    pub async fn shutdown(self) {
        for (server_name, service) in self.servers {
            match Arc::try_unwrap(service) {
                Ok(service) => {
                    if let Err(error) = service.cancel().await {
                        tracing::warn!(
                            "failed to shut down MCP server '{}': {}",
                            server_name,
                            error
                        );
                    }
                }
                Err(_arc) => {
                    tracing::warn!(
                        "MCP server '{}' still has outstanding references, dropping",
                        server_name
                    );
                }
            }
        }
    }
}

async fn connect_server(config: &McpServerConfig) -> Result<McpRunningService> {
    use rmcp::ServiceExt;

    match config.transport {
        McpTransport::Stdio => {
            let command_str =
                config
                    .command
                    .as_deref()
                    .ok_or_else(|| AgshError::McpConnection {
                        server_name: config.name.clone(),
                        message: "stdio transport requires 'command' field".to_string(),
                    })?;

            let mut command = Command::new(command_str);
            if let Some(args) = &config.args {
                command.args(args);
            }
            if let Some(env) = &config.env {
                command.envs(env);
            }

            let transport = rmcp::transport::TokioChildProcess::new(command).map_err(|error| {
                AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: format!("failed to spawn process: {}", error),
                }
            })?;

            ().serve(transport)
                .await
                .map_err(|error| AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: format!("handshake failed: {}", error),
                })
        }
        McpTransport::Http => {
            let url = config
                .url
                .as_deref()
                .ok_or_else(|| AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: "http transport requires 'url' field".to_string(),
                })?;

            let mut transport_config =
                rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(url);

            if let Some(token) = &config.auth_token {
                transport_config = transport_config.auth_header(token.clone());
            }

            if let Some(headers) = &config.headers {
                let mut header_map = std::collections::HashMap::new();
                for (key, value) in headers {
                    let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                        .map_err(|error| AgshError::McpConnection {
                            server_name: config.name.clone(),
                            message: format!("invalid header name '{}': {}", key, error),
                        })?;
                    let header_value =
                        reqwest::header::HeaderValue::from_str(value).map_err(|error| {
                            AgshError::McpConnection {
                                server_name: config.name.clone(),
                                message: format!("invalid header value for '{}': {}", key, error),
                            }
                        })?;
                    header_map.insert(header_name, header_value);
                }
                transport_config = transport_config.custom_headers(header_map);
            }

            let transport =
                rmcp::transport::StreamableHttpClientTransport::from_config(transport_config);

            ().serve(transport)
                .await
                .map_err(|error| AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: format!("HTTP connection failed: {}", error),
                })
        }
    }
}

fn parse_server_permission(server_name: &str, permission_str: Option<&str>) -> Result<Permission> {
    let permission_str = permission_str.unwrap_or("read");
    permission_str
        .parse::<Permission>()
        .map_err(|_| AgshError::McpConnection {
            server_name: server_name.to_string(),
            message: format!(
                "invalid permission '{}': expected 'none', 'read', or 'write'",
                permission_str
            ),
        })
}

pub struct McpToolAdapter {
    namespaced_name: String,
    server_name: String,
    remote_tool_name: String,
    description: String,
    parameters: serde_json::Value,
    permission: Permission,
    service: Arc<McpRunningService>,
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.namespaced_name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }

    fn required_permission(&self) -> Permission {
        self.permission
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let arguments = input.as_object().cloned();

        let mut call_params =
            rmcp::model::CallToolRequestParams::new(self.remote_tool_name.clone());
        call_params.arguments = arguments;

        let result = tokio::select! {
            result = self.service.peer().call_tool(call_params) => {
                result.map_err(|error| AgshError::McpToolExecution {
                    server_name: self.server_name.clone(),
                    tool_name: self.remote_tool_name.clone(),
                    message: error.to_string(),
                })?
            }
            _ = cancellation.cancelled() => {
                return Err(AgshError::Interrupted);
            }
        };

        let content = result
            .content
            .iter()
            .map(|content_item| match &content_item.raw {
                rmcp::model::RawContent::Text(text_content) => text_content.text.clone(),
                rmcp::model::RawContent::Image(_) => "[image content]".to_string(),
                rmcp::model::RawContent::Audio(_) => "[audio content]".to_string(),
                rmcp::model::RawContent::Resource(resource) => {
                    format!("[embedded resource: {:?}]", resource.resource)
                }
                rmcp::model::RawContent::ResourceLink(resource) => {
                    format!("[resource link: {}]", resource.uri)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let is_error = result.is_error.unwrap_or(false);

        Ok(ToolOutput { content, is_error })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_server_permission_defaults_to_read() {
        let permission = parse_server_permission("test", None).expect("should parse");
        assert_eq!(permission, Permission::Read);
    }

    #[test]
    fn test_parse_server_permission_valid_values() {
        assert_eq!(
            parse_server_permission("test", Some("read")).expect("should parse"),
            Permission::Read
        );
        assert_eq!(
            parse_server_permission("test", Some("write")).expect("should parse"),
            Permission::Write
        );
        assert_eq!(
            parse_server_permission("test", Some("none")).expect("should parse"),
            Permission::None
        );
    }

    #[test]
    fn test_parse_server_permission_invalid() {
        assert!(parse_server_permission("test", Some("invalid")).is_err());
    }
}
