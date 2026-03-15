use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rmcp::transport::auth::OAuthState;
use rmcp::transport::{
    AuthClient, AuthError, AuthorizationManager, ClientCredentialsConfig, CredentialStore,
    StoredCredentials,
};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::config::{McpAuthConfig, McpServerConfig, McpTransport};
use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::ToolDefinition;
use crate::session::TokenStore;
use crate::tools::{Tool, ToolOutput};

type McpRunningService = rmcp::service::RunningService<rmcp::RoleClient, ()>;

pub struct McpClientManager {
    servers: HashMap<String, Arc<McpRunningService>>,
}

impl McpClientManager {
    pub async fn connect_all(
        configs: &[McpServerConfig],
        token_store: Option<&TokenStore>,
    ) -> Result<Self> {
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

            let service = match connect_server(config, token_store).await {
                Ok(service) => service,
                Err(error) => {
                    tracing::warn!(
                        "failed to connect to MCP server '{}': {}",
                        config.name,
                        error
                    );
                    continue;
                }
            };
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
        let Some(service) = self.servers.get(server_name) else {
            return Ok(Vec::new());
        };

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

async fn connect_server(
    config: &McpServerConfig,
    token_store: Option<&TokenStore>,
) -> Result<McpRunningService> {
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

            if let Some(auth_config) = &config.auth {
                connect_http_with_oauth(
                    &config.name,
                    url,
                    auth_config,
                    transport_config,
                    token_store,
                )
                .await
            } else {
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
}

async fn connect_http_with_oauth(
    server_name: &str,
    url: &str,
    auth_config: &McpAuthConfig,
    transport_config: rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig,
    token_store: Option<&TokenStore>,
) -> Result<McpRunningService> {
    use rmcp::ServiceExt;

    let auth_manager = match auth_config {
        McpAuthConfig::ClientCredentials {
            client_id,
            client_secret,
            scopes,
            resource,
        } => {
            authenticate_client_credentials(
                server_name,
                url,
                client_id,
                client_secret,
                scopes.as_deref(),
                resource.as_deref(),
            )
            .await?
        }
        McpAuthConfig::ClientCredentialsJwt {
            client_id,
            signing_key_path,
            signing_algorithm,
            scopes,
            resource,
        } => {
            authenticate_client_credentials_jwt(
                server_name,
                url,
                client_id,
                signing_key_path,
                signing_algorithm.as_deref(),
                scopes.as_deref(),
                resource.as_deref(),
            )
            .await?
        }
        McpAuthConfig::OAuth {
            client_id,
            client_secret,
            scopes,
            redirect_port,
        } => {
            authenticate_oauth_authorization_code(
                server_name,
                url,
                client_id.as_deref(),
                client_secret.as_deref(),
                scopes.as_deref(),
                redirect_port.unwrap_or(8400),
                token_store,
            )
            .await?
        }
    };

    let auth_client = AuthClient::new(reqwest::Client::new(), auth_manager);
    let transport =
        rmcp::transport::StreamableHttpClientTransport::with_client(auth_client, transport_config);

    ().serve(transport)
        .await
        .map_err(|error| AgshError::McpConnection {
            server_name: server_name.to_string(),
            message: format!("HTTP connection failed: {}", error),
        })
}

async fn authenticate_client_credentials(
    server_name: &str,
    url: &str,
    client_id: &str,
    client_secret: &str,
    scopes: Option<&[String]>,
    resource: Option<&str>,
) -> Result<AuthorizationManager> {
    let config = ClientCredentialsConfig::ClientSecret {
        client_id: client_id.to_string(),
        client_secret: client_secret.to_string(),
        scopes: scopes.map(|s| s.to_vec()).unwrap_or_default(),
        resource: resource.map(|s| s.to_string()),
    };

    let mut oauth_state = OAuthState::new(url, None)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth: {}", error),
        })?;

    oauth_state
        .authenticate_client_credentials(config)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("client credentials authentication failed: {}", error),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after client credentials authentication".to_string(),
        })
}

async fn authenticate_client_credentials_jwt(
    server_name: &str,
    url: &str,
    client_id: &str,
    signing_key_path: &str,
    signing_algorithm: Option<&str>,
    scopes: Option<&[String]>,
    resource: Option<&str>,
) -> Result<AuthorizationManager> {
    let signing_key = std::fs::read(signing_key_path).map_err(|error| AgshError::McpAuth {
        server_name: server_name.to_string(),
        message: format!(
            "failed to read signing key '{}': {}",
            signing_key_path, error
        ),
    })?;

    let algorithm = parse_jwt_signing_algorithm(server_name, signing_algorithm)?;

    let config = ClientCredentialsConfig::PrivateKeyJwt {
        client_id: client_id.to_string(),
        signing_key,
        signing_algorithm: algorithm,
        scopes: scopes.map(|s| s.to_vec()).unwrap_or_default(),
        resource: resource.map(|s| s.to_string()),
        token_endpoint_audience: None,
    };

    let mut oauth_state = OAuthState::new(url, None)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth: {}", error),
        })?;

    oauth_state
        .authenticate_client_credentials(config)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("JWT client credentials authentication failed: {}", error),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after JWT authentication".to_string(),
        })
}

fn parse_jwt_signing_algorithm(
    server_name: &str,
    algorithm: Option<&str>,
) -> Result<rmcp::transport::JwtSigningAlgorithm> {
    use rmcp::transport::JwtSigningAlgorithm;

    match algorithm.unwrap_or("RS256") {
        "RS256" => Ok(JwtSigningAlgorithm::RS256),
        "RS384" => Ok(JwtSigningAlgorithm::RS384),
        "RS512" => Ok(JwtSigningAlgorithm::RS512),
        "ES256" => Ok(JwtSigningAlgorithm::ES256),
        "ES384" => Ok(JwtSigningAlgorithm::ES384),
        other => Err(AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!(
                "unsupported signing algorithm '{}': expected RS256, RS384, RS512, ES256, or ES384",
                other
            ),
        }),
    }
}

async fn authenticate_oauth_authorization_code(
    server_name: &str,
    url: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
    scopes: Option<&[String]>,
    redirect_port: u16,
    token_store: Option<&TokenStore>,
) -> Result<AuthorizationManager> {
    let redirect_uri = format!("http://127.0.0.1:{}/callback", redirect_port);
    let scope_strings: Vec<String> = scopes.map(|s| s.to_vec()).unwrap_or_default();

    let mut manager = AuthorizationManager::new(url)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth manager: {}", error),
        })?;

    // Set up persistent credential storage if available
    if let Some(store) = token_store {
        manager.set_credential_store(SqliteCredentialStore {
            token_store: store.clone(),
            server_name: server_name.to_string(),
        });
    }

    // Try loading existing credentials from the store
    let has_stored_credentials =
        manager
            .initialize_from_store()
            .await
            .map_err(|error| AgshError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to load stored credentials: {}", error),
            })?;

    if has_stored_credentials {
        tracing::info!(
            "loaded stored OAuth credentials for MCP server '{}'",
            server_name
        );
        return Ok(manager);
    }

    // No stored credentials — run the interactive browser flow
    tracing::info!(
        "starting OAuth authorization flow for MCP server '{}'",
        server_name
    );

    // Wrap in OAuthState to use its start_authorization flow which handles
    // metadata discovery, dynamic client registration, and PKCE setup
    let mut oauth_state = OAuthState::Unauthorized(manager);

    // If we have a pre-configured client_id, configure it before starting
    if let Some(id) = client_id {
        // We need to discover metadata first, then configure the client
        if let OAuthState::Unauthorized(ref mut manager) = oauth_state {
            let metadata =
                manager
                    .discover_metadata()
                    .await
                    .map_err(|error| AgshError::McpAuth {
                        server_name: server_name.to_string(),
                        message: format!("OAuth metadata discovery failed: {}", error),
                    })?;
            manager.set_metadata(metadata);

            let oauth_client_config = rmcp::transport::auth::OAuthClientConfig {
                client_id: id.to_string(),
                client_secret: client_secret.map(|s| s.to_string()),
                scopes: scope_strings.clone(),
                redirect_uri: redirect_uri.clone(),
            };
            manager
                .configure_client(oauth_client_config)
                .map_err(|error| AgshError::McpAuth {
                    server_name: server_name.to_string(),
                    message: format!("failed to configure OAuth client: {}", error),
                })?;

            let scope_refs: Vec<&str> = scope_strings.iter().map(|s| s.as_str()).collect();
            let auth_url = manager
                .get_authorization_url(&scope_refs)
                .await
                .map_err(|error| AgshError::McpAuth {
                    server_name: server_name.to_string(),
                    message: format!("failed to get authorization URL: {}", error),
                })?;

            let session = rmcp::transport::AuthorizationSession::for_scope_upgrade(
                std::mem::replace(
                    manager,
                    AuthorizationManager::new("http://localhost")
                        .await
                        .map_err(|error| AgshError::McpAuth {
                            server_name: server_name.to_string(),
                            message: format!("internal error: {}", error),
                        })?,
                ),
                auth_url,
                &redirect_uri,
            );
            oauth_state = OAuthState::Session(session);
        }
    } else {
        // No client_id configured — use dynamic registration via start_authorization
        let scope_refs: Vec<&str> = scope_strings.iter().map(|s| s.as_str()).collect();
        oauth_state
            .start_authorization(&scope_refs, &redirect_uri, Some("agsh"))
            .await
            .map_err(|error| AgshError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to start OAuth authorization: {}", error),
            })?;
    }

    let auth_url =
        oauth_state
            .get_authorization_url()
            .await
            .map_err(|error| AgshError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to get authorization URL: {}", error),
            })?;

    // Open browser for user authorization
    eprintln!(
        "Opening browser for MCP server '{}' OAuth authorization...",
        server_name
    );
    if let Err(error) = open::that(&auth_url) {
        tracing::warn!("failed to open browser: {}", error);
    }
    eprintln!("If the browser didn't open, visit:\n  {}", auth_url);

    // Start callback server and wait for the authorization code
    let (code, state) = run_oauth_callback_server(redirect_port)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("OAuth callback failed: {}", error),
        })?;

    // Exchange the authorization code for tokens
    oauth_state
        .handle_callback(&code, &state)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("OAuth token exchange failed: {}", error),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after authorization".to_string(),
        })
}

async fn run_oauth_callback_server(port: u16) -> std::result::Result<(String, String), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .map_err(|error| format!("failed to bind callback server on port {}: {}", port, error))?;

    let timeout = tokio::time::Duration::from_secs(120);

    let (mut stream, _addr) = tokio::time::timeout(timeout, listener.accept())
        .await
        .map_err(|_| "authorization timed out after 120 seconds".to_string())?
        .map_err(|error| format!("failed to accept connection: {}", error))?;

    let mut buffer = vec![0u8; 4096];
    let bytes_read = stream
        .read(&mut buffer)
        .await
        .map_err(|error| format!("failed to read request: {}", error))?;

    let request = String::from_utf8_lossy(&buffer[..bytes_read]);

    // Parse the GET request line to extract query parameters
    // Expected: GET /callback?code=...&state=... HTTP/1.1
    let (code, state) = parse_callback_query(&request)?;

    let response_body = "<!DOCTYPE html><html><body>\
        <h1>Authorization successful</h1>\
        <p>You can close this tab and return to agsh.</p>\
        </body></html>";
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response_body.len(),
        response_body
    );

    if let Err(error) = stream.write_all(response.as_bytes()).await {
        tracing::debug!("failed to send callback response: {}", error);
    }

    Ok((code, state))
}

fn parse_callback_query(request: &str) -> std::result::Result<(String, String), String> {
    let first_line = request.lines().next().ok_or("empty HTTP request")?;

    let path = first_line
        .split_whitespace()
        .nth(1)
        .ok_or("malformed HTTP request line")?;

    let query_string = path
        .split_once('?')
        .map(|(_, query)| query)
        .ok_or("no query parameters in callback URL")?;

    let mut code = None;
    let mut state = None;

    for pair in query_string.split('&') {
        if let Some((key, value)) = pair.split_once('=') {
            match key {
                "code" => code = Some(value.to_string()),
                "state" => state = Some(value.to_string()),
                _ => {}
            }
        }
    }

    let code = code.ok_or("missing 'code' parameter in callback")?;
    let state = state.ok_or("missing 'state' parameter in callback")?;

    Ok((code, state))
}

struct SqliteCredentialStore {
    token_store: TokenStore,
    server_name: String,
}

#[async_trait]
impl CredentialStore for SqliteCredentialStore {
    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        match self
            .token_store
            .load_mcp_credentials(&self.server_name)
            .await
        {
            Ok(Some(json)) => {
                let credentials: StoredCredentials =
                    serde_json::from_str(&json).map_err(|error| {
                        AuthError::InternalError(format!(
                            "failed to deserialize stored credentials: {}",
                            error
                        ))
                    })?;
                Ok(Some(credentials))
            }
            Ok(None) => Ok(None),
            Err(error) => Err(AuthError::InternalError(format!(
                "failed to load credentials from database: {}",
                error
            ))),
        }
    }

    async fn save(&self, credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        let json = serde_json::to_string(&credentials).map_err(|error| {
            AuthError::InternalError(format!("failed to serialize credentials: {}", error))
        })?;

        self.token_store
            .save_mcp_credentials(&self.server_name, &json)
            .await
            .map_err(|error| {
                AuthError::InternalError(format!(
                    "failed to save credentials to database: {}",
                    error
                ))
            })
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        self.token_store
            .clear_mcp_credentials(&self.server_name)
            .await
            .map_err(|error| {
                AuthError::InternalError(format!(
                    "failed to clear credentials from database: {}",
                    error
                ))
            })
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

    #[test]
    fn test_parse_callback_query_valid() {
        let request = "GET /callback?code=abc123&state=xyz789 HTTP/1.1\r\nHost: localhost\r\n";
        let (code, state) = parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz789");
    }

    #[test]
    fn test_parse_callback_query_reversed_order() {
        let request = "GET /callback?state=xyz789&code=abc123 HTTP/1.1\r\n";
        let (code, state) = parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz789");
    }

    #[test]
    fn test_parse_callback_query_missing_code() {
        let request = "GET /callback?state=xyz789 HTTP/1.1\r\n";
        let result = parse_callback_query(request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("code"));
    }

    #[test]
    fn test_parse_callback_query_missing_state() {
        let request = "GET /callback?code=abc123 HTTP/1.1\r\n";
        let result = parse_callback_query(request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("state"));
    }

    #[test]
    fn test_parse_callback_query_no_query_string() {
        let request = "GET /callback HTTP/1.1\r\n";
        let result = parse_callback_query(request);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_jwt_signing_algorithm_defaults() {
        let alg = parse_jwt_signing_algorithm("test", None).expect("should parse");
        assert!(matches!(alg, rmcp::transport::JwtSigningAlgorithm::RS256));
    }

    #[test]
    fn test_parse_jwt_signing_algorithm_all_values() {
        for name in &["RS256", "RS384", "RS512", "ES256", "ES384"] {
            assert!(parse_jwt_signing_algorithm("test", Some(name)).is_ok());
        }
    }

    #[test]
    fn test_parse_jwt_signing_algorithm_invalid() {
        assert!(parse_jwt_signing_algorithm("test", Some("HS256")).is_err());
    }
}
