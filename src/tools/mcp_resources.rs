//! Builtin tools exposing MCP resources and prompts to the agent: `mcp_resource_list`,
//! `mcp_resource_read`, `mcp_prompt_list`, and `mcp_prompt_get`. Each tool routes through a shared
//! [`McpClientManager`] so it can target any configured server by name.

use std::sync::Arc;

use async_trait::async_trait;

use super::{Tool, ToolDenials, ToolOutput, util::require_str};
use crate::{
    error::{MekaError, Result},
    mcp::{MAX_MCP_DESCRIPTION_CHARS, McpClientManager, truncate},
    permission::Permission,
    provider::ToolDefinition,
    text::sanitize_text,
};

/// Cap on total bytes returned by `mcp_resource_read` across all content chunks from a single
/// server response. Mirrors `MAX_MCP_IMAGE_BYTES`: servers can return large blob or text resources
/// that would otherwise be cloned verbatim into the provider request and blown through the user's
/// API quota (or OOM the agent).
pub(crate) const MAX_MCP_RESOURCE_BYTES: usize = 10 * crate::text::MIB;

fn no_such_server(tool_name: &str, server: &str, available: &[String]) -> MekaError {
    MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: crate::text::unknown_name("MCP server", server, available),
    }
}

/// Servers a denied sub-agent may still name. Every meta-tool routes its server argument through
/// this rather than [`McpClientManager::server_names`] directly.
///
/// Without it `disabled_servers` would be a tool-list filter and nothing more: the meta-tools take
/// a server by name and never consult the registry, so a worker denied `mekabridge` could still
/// read its resources and render its prompts. A denial that covers only one of the three surfaces a
/// server exposes is not a denial.
fn visible_servers(manager: &McpClientManager, denials: &ToolDenials) -> Vec<String> {
    manager
        .server_names()
        .into_iter()
        .filter(|name| !denials.denies_server(name))
        .collect()
}

/// Resolve a caller-named server, refusing denied ones the same way an unconfigured name is
/// refused. Deliberately indistinguishable: telling a worker "that server exists but you may not
/// have it" hands it the server list its denial was meant to withhold.
fn visible_server_entry(
    tool_name: &str,
    manager: &McpClientManager,
    denials: &ToolDenials,
    server: &str,
) -> Result<Arc<crate::mcp::ServerEntry>> {
    if denials.denies_server(server) {
        return Err(no_such_server(
            tool_name,
            server,
            &visible_servers(manager, denials),
        ));
    }
    manager
        .server_entry(server)
        .ok_or_else(|| no_such_server(tool_name, server, &visible_servers(manager, denials)))
}

pub(crate) struct ListMcpResourcesTool {
    pub(crate) manager: Arc<McpClientManager>,
    pub(crate) denials: Arc<ToolDenials>,
}

#[async_trait]
impl Tool for ListMcpResourcesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "mcp_resource_list".to_string(),
            description: "List resources advertised by MCP servers. If `server` is provided, \
                 list only that server's resources; otherwise list all configured \
                 servers. Each row is `<server>\\t<uri>\\t<name>\\t<mime>\\t<description>`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Optional MCP server name. If omitted, lists resources from every configured server."
                    }
                }
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        let server_filter = input
            .get("server")
            .and_then(|v| v.as_str())
            .map(String::from);

        let names: Vec<String> = if let Some(name) = &server_filter {
            visible_server_entry("mcp_resource_list", &self.manager, &self.denials, name)?;
            vec![name.clone()]
        } else {
            visible_servers(&self.manager, &self.denials)
        };

        if names.is_empty() {
            return Ok(ToolOutput::text(
                "(no MCP servers configured)".to_string(),
                false,
            ));
        }

        let mut lines = Vec::new();
        let mut any_error = false;

        for name in names {
            let Some(entry) = self.manager.server_entry(&name) else {
                continue;
            };
            match crate::mcp::list_resources(&entry, &cancellation).await {
                Ok(resources) => {
                    for resource in resources {
                        let raw = &resource;
                        let mime = raw.mime_type.as_deref().unwrap_or("");
                        let description = raw.description.as_deref().unwrap_or("");
                        let description =
                            truncate(&sanitize_text(description), MAX_MCP_DESCRIPTION_CHARS);
                        lines.push(format!(
                            "{}\t{}\t{}\t{}\t{}",
                            name,
                            sanitize_text(&raw.uri),
                            sanitize_text(&raw.name),
                            sanitize_text(mime),
                            description
                        ));
                    }
                }
                Err(error) => {
                    any_error = true;
                    lines.push(format!("{name}\t<error>\t\t\t{error}"));
                }
            }
        }

        if lines.is_empty() {
            return Ok(ToolOutput::text(
                "(no resources advertised)".to_string(),
                false,
            ));
        }

        Ok(ToolOutput::text(lines.join("\n"), any_error))
    }
}

pub(crate) struct ReadMcpResourceTool {
    pub(crate) manager: Arc<McpClientManager>,
    pub(crate) denials: Arc<ToolDenials>,
}

#[async_trait]
impl Tool for ReadMcpResourceTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "mcp_resource_read".to_string(),
            description: "Read an MCP resource by URI from a specific server. Text \
                          resources are returned inline; binary resources are \
                          returned base64-encoded with their declared MIME type."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server name that advertises the resource."
                    },
                    "uri": {
                        "type": "string",
                        "description": "Resource URI (e.g. file:///path/to/file). Exactly as listed by `mcp_resource_list`."
                    }
                },
                "required": ["server", "uri"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        let server = require_str(&input, "server", "mcp_resource_read")?;
        let uri = require_str(&input, "uri", "mcp_resource_read")?;

        let entry =
            visible_server_entry("mcp_resource_read", &self.manager, &self.denials, &server)?;

        let result = crate::mcp::read_resource(&entry, uri.clone(), &cancellation).await?;

        let chunks = format_resource_contents(&result.contents, MAX_MCP_RESOURCE_BYTES);

        if chunks.is_empty() {
            return Ok(ToolOutput::text(
                format!("resource '{uri}' returned no content"),
                false,
            ));
        }

        Ok(ToolOutput::text(chunks.join("\n\n"), false))
    }
}

/// Render MCP `ResourceContents` into formatted chunks with Unicode sanitization applied to all
/// server-supplied strings (URIs, MIME types, text bodies) and a hard byte budget across the whole
/// response. Split from `ReadMcpResourceTool::execute` so it's exercisable from tests.
fn format_resource_contents(
    contents: &[rmcp::model::ResourceContents],
    max_bytes: usize,
) -> Vec<String> {
    let mut chunks = Vec::with_capacity(contents.len());
    let mut total_bytes: usize = 0;
    let mut truncated = false;
    for entry in contents {
        if truncated {
            break;
        }
        match entry {
            rmcp::model::ResourceContents::TextResourceContents {
                uri: content_uri,
                mime_type,
                text,
                ..
            } => {
                if total_bytes.saturating_add(text.len()) > max_bytes {
                    chunks.push(format!(
                        "--- {} [truncated: would exceed {} byte limit]",
                        sanitize_text(content_uri),
                        max_bytes
                    ));
                    truncated = true;
                    continue;
                }
                total_bytes = total_bytes.saturating_add(text.len());
                chunks.push(format!(
                    "--- {} [{}]\n{}",
                    sanitize_text(content_uri),
                    sanitize_text(mime_type.as_deref().unwrap_or("text")),
                    sanitize_text(text)
                ));
            }
            rmcp::model::ResourceContents::BlobResourceContents {
                uri: content_uri,
                mime_type,
                blob,
                ..
            } => {
                if total_bytes.saturating_add(blob.len()) > max_bytes {
                    chunks.push(format!(
                        "--- {} [truncated: blob would exceed {} byte limit]",
                        sanitize_text(content_uri),
                        max_bytes
                    ));
                    truncated = true;
                    continue;
                }
                total_bytes = total_bytes.saturating_add(blob.len());
                chunks.push(format!(
                    "--- {} [{}] (base64, {} bytes encoded)\n{}",
                    sanitize_text(content_uri),
                    sanitize_text(mime_type.as_deref().unwrap_or("application/octet-stream")),
                    blob.len(),
                    blob
                ));
            }
            // `ResourceContents` is non-exhaustive; skip any content kind this build doesn't know.
            _ => {}
        }
    }
    chunks
}

pub(crate) struct ListMcpPromptsTool {
    pub(crate) manager: Arc<McpClientManager>,
    pub(crate) denials: Arc<ToolDenials>,
}

#[async_trait]
impl Tool for ListMcpPromptsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "mcp_prompt_list".to_string(),
            description: "List prompts advertised by MCP servers. If `server` is provided, \
                 list only that server's prompts; otherwise list all configured \
                 servers. Each row is `<server>\\t<name>\\t<description>\\t<args>`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Optional MCP server name. If omitted, lists prompts from every configured server."
                    }
                }
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        let server_filter = input
            .get("server")
            .and_then(|v| v.as_str())
            .map(String::from);

        let names: Vec<String> = if let Some(name) = &server_filter {
            visible_server_entry("mcp_prompt_list", &self.manager, &self.denials, name)?;
            vec![name.clone()]
        } else {
            visible_servers(&self.manager, &self.denials)
        };

        if names.is_empty() {
            return Ok(ToolOutput::text(
                "(no MCP servers configured)".to_string(),
                false,
            ));
        }

        let mut lines = Vec::new();
        let mut any_error = false;

        for name in names {
            let Some(entry) = self.manager.server_entry(&name) else {
                continue;
            };
            match crate::mcp::list_prompts(&entry, &cancellation).await {
                Ok(prompts) => {
                    for prompt in prompts {
                        let description = prompt.description.unwrap_or_default();
                        let description =
                            truncate(&sanitize_text(&description), MAX_MCP_DESCRIPTION_CHARS);
                        let args = prompt
                            .arguments
                            .unwrap_or_default()
                            .into_iter()
                            .map(|a| {
                                let sanitized = sanitize_text(&a.name);
                                if a.required == Some(true) {
                                    format!("{sanitized}!")
                                } else {
                                    sanitized
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(",");
                        lines.push(format!(
                            "{}\t{}\t{}\t{}",
                            name,
                            sanitize_text(&prompt.name),
                            description,
                            args
                        ));
                    }
                }
                Err(error) => {
                    any_error = true;
                    lines.push(format!("{name}\t<error>\t{error}\t"));
                }
            }
        }

        if lines.is_empty() {
            return Ok(ToolOutput::text(
                "(no prompts advertised)".to_string(),
                false,
            ));
        }

        Ok(ToolOutput::text(lines.join("\n"), any_error))
    }
}

pub(crate) struct GetMcpPromptTool {
    pub(crate) manager: Arc<McpClientManager>,
    pub(crate) denials: Arc<ToolDenials>,
}

#[async_trait]
impl Tool for GetMcpPromptTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "mcp_prompt_get".to_string(),
            description: "Render an MCP prompt by name from a specific server. \
                          Returns the prompt's messages serialized as `<role>: \
                          <text>` lines. `arguments` are passed verbatim to the \
                          server; see `mcp_prompt_list` for each prompt's \
                          declared arguments."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server name that advertises the prompt."
                    },
                    "name": {
                        "type": "string",
                        "description": "Prompt name, as returned by `mcp_prompt_list`."
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the prompt. Keys match the prompt's declared argument names.",
                        "additionalProperties": true
                    }
                },
                "required": ["server", "name"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        let server = require_str(&input, "server", "mcp_prompt_get")?;
        let name = require_str(&input, "name", "mcp_prompt_get")?;

        let arguments = input.get("arguments").and_then(|v| v.as_object()).cloned();

        let entry = visible_server_entry("mcp_prompt_get", &self.manager, &self.denials, &server)?;

        let result = crate::mcp::get_prompt(&entry, name.clone(), arguments, &cancellation).await?;

        // Sanitized before truncation, like every other server-supplied string in this file: it
        // reaches the model and the terminal.
        let description = result
            .description
            .map(|description| {
                truncate(
                    &crate::text::sanitize_text(&description),
                    MAX_MCP_DESCRIPTION_CHARS,
                )
            })
            .unwrap_or_default();

        let mut lines = Vec::new();
        if !description.is_empty() {
            lines.push(format!("# {description}"));
        }

        for message in &result.messages {
            let role_label = match message.role {
                rmcp::model::Role::User => "user",
                rmcp::model::Role::Assistant => "assistant",
            };
            match &message.content {
                rmcp::model::ContentBlock::Text(text) => {
                    lines.push(format!("{}: {}", role_label, sanitize_text(&text.text)));
                }
                rmcp::model::ContentBlock::Image(_) => {
                    lines.push(format!("{role_label}: [image content]"));
                }
                rmcp::model::ContentBlock::Audio(_) => {
                    lines.push(format!("{role_label}: [audio content]"));
                }
                rmcp::model::ContentBlock::Resource(embedded) => {
                    lines.push(format!(
                        "{}: [embedded resource: {:?}]",
                        role_label, embedded.resource
                    ));
                }
                rmcp::model::ContentBlock::ResourceLink(link) => {
                    lines.push(format!(
                        "{}: [resource link: {}]",
                        role_label,
                        sanitize_text(&link.uri)
                    ));
                }
                _ => lines.push(format!("{role_label}: [unsupported content]")),
            }
        }

        if lines.is_empty() {
            return Ok(ToolOutput::text(
                format!("prompt '{name}' returned no messages"),
                false,
            ));
        }

        Ok(ToolOutput::text(lines.join("\n"), false))
    }
}

pub(crate) fn register_all(registry: &super::ToolRegistry, manager: Arc<McpClientManager>) {
    // Skip registration if no servers are configured. These tools rely on the manager and there's
    // nothing useful to do without at least one. A sub-agent denied every configured server is in
    // exactly that position, so it takes the same exit: seven tools that can only answer "unknown
    // server" are worse than no tools, because the model spends turns discovering that.
    let denials = Arc::new(registry.denials().clone());
    if visible_servers(&manager, &denials).is_empty() {
        return;
    }
    // These seven are registered directly rather than through `register_builtin`, so they honor
    // the `[tools]` block-list and the sub-agent deny list only because this asks.
    // `admits_infrastructure` ignores `allowed_tools`, which would silently delete them from any
    // install that has one.
    //
    // All seven are discovery-style helpers, so each is marked deferred. Marking rides along in the
    // same macro: a deferred marker for a tool that was never registered is a name `load_tool`
    // would offer and then fail to find.
    macro_rules! register_meta {
        ($name:expr, $tool:expr) => {
            if registry.admits_infrastructure($name) {
                #[allow(
                    clippy::expect_used,
                    reason = "two builtins sharing a name is a bug the first build must surface"
                )]
                registry.register(Arc::new($tool)).expect(concat!(
                    "builtin ",
                    $name,
                    " tool name collision"
                ));
                registry.mark_deferred($name);
            }
        };
    }
    register_meta!("mcp_resource_list", ListMcpResourcesTool {
        manager: Arc::clone(&manager),
        denials: Arc::clone(&denials),
    });
    register_meta!("mcp_resource_read", ReadMcpResourceTool {
        manager: Arc::clone(&manager),
        denials: Arc::clone(&denials),
    });
    register_meta!("mcp_prompt_list", ListMcpPromptsTool {
        manager: Arc::clone(&manager),
        denials: Arc::clone(&denials),
    });
    register_meta!("mcp_prompt_get", GetMcpPromptTool {
        manager: Arc::clone(&manager),
        denials: Arc::clone(&denials),
    });
    register_meta!("mcp_resource_subscribe", SubscribeMcpResourceTool {
        manager: Arc::clone(&manager),
        denials: Arc::clone(&denials),
    });
    register_meta!("mcp_resource_unsubscribe", UnsubscribeMcpResourceTool {
        manager: Arc::clone(&manager),
        denials: Arc::clone(&denials),
    });
    register_meta!("mcp_resource_updates_list", ListMcpResourceUpdatesTool {
        manager: Arc::clone(&manager),
        denials
    });
    drop(manager);
}

pub(crate) struct SubscribeMcpResourceTool {
    pub(crate) manager: Arc<McpClientManager>,
    pub(crate) denials: Arc<ToolDenials>,
}

#[async_trait]
impl Tool for SubscribeMcpResourceTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "mcp_resource_subscribe".to_string(),
            description: "Subscribe to change notifications for an MCP resource. After \
                          subscribing, the server will send resources/updated notifications \
                          that meka records; query them with `mcp_resource_updates_list`."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "MCP server name that advertises the resource."},
                    "uri": {"type": "string", "description": "Resource URI to subscribe to."}
                },
                "required": ["server", "uri"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        let server = require_str(&input, "server", "mcp_resource_subscribe")?;
        let uri = require_str(&input, "uri", "mcp_resource_subscribe")?;
        let entry = visible_server_entry(
            "mcp_resource_subscribe",
            &self.manager,
            &self.denials,
            &server,
        )?;
        crate::mcp::subscribe_resource(&entry, uri.clone(), &cancellation)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "mcp_resource_subscribe".to_string(),
                message: format!("subscribe failed: {error}"),
            })?;
        Ok(ToolOutput::text(
            format!("subscribed to '{uri}' on server '{server}'"),
            false,
        ))
    }
}

pub(crate) struct UnsubscribeMcpResourceTool {
    pub(crate) manager: Arc<McpClientManager>,
    pub(crate) denials: Arc<ToolDenials>,
}

#[async_trait]
impl Tool for UnsubscribeMcpResourceTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "mcp_resource_unsubscribe".to_string(),
            description: "Cancel a prior subscription to an MCP resource.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "MCP server name that advertises the resource."},
                    "uri": {"type": "string", "description": "Resource URI to unsubscribe from."}
                },
                "required": ["server", "uri"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        let server = require_str(&input, "server", "mcp_resource_unsubscribe")?;
        let uri = require_str(&input, "uri", "mcp_resource_unsubscribe")?;
        let entry = visible_server_entry(
            "mcp_resource_unsubscribe",
            &self.manager,
            &self.denials,
            &server,
        )?;
        crate::mcp::unsubscribe_resource(&entry, uri.clone(), &cancellation)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "mcp_resource_unsubscribe".to_string(),
                message: format!("unsubscribe failed: {error}"),
            })?;
        Ok(ToolOutput::text(
            format!("unsubscribed from '{uri}' on server '{server}'"),
            false,
        ))
    }
}

pub(crate) struct ListMcpResourceUpdatesTool {
    pub(crate) manager: Arc<McpClientManager>,
    pub(crate) denials: Arc<ToolDenials>,
}

#[async_trait]
impl Tool for ListMcpResourceUpdatesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "mcp_resource_updates_list".to_string(),
            description: "List all resources that have been reported as updated since \
                          this meka session started. Rows are `<server>\\t<uri>\\t<unix_ts>`."
                .to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        // Reads an in-process ledger; there is no round-trip to bound or interrupt.
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        // The update log is shared by every session on this MCP context, so a parent's
        // subscription to a denied server would show that server's resource URIs to a worker that
        // cannot otherwise see it exists.
        let updates: Vec<_> = self
            .manager
            .client_context
            .resource_updates
            .snapshot()
            .into_iter()
            .filter(|(server, _uri, _stamp)| !self.denials.denies_server(server))
            .collect();
        if updates.is_empty() {
            return Ok(ToolOutput::text(
                "(no MCP resource updates recorded)".to_string(),
                false,
            ));
        }
        let body = updates
            .into_iter()
            .map(|(server, uri, stamp)| format!("{server}\t{uri}\t{stamp}"))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ToolOutput::text(body, false))
    }
}

#[cfg(test)]
mod tests {
    use rmcp::model::ResourceContents;

    use super::*;

    fn text_contents(uri: &str, mime: Option<&str>, body: &str) -> ResourceContents {
        ResourceContents::TextResourceContents {
            uri: uri.to_string(),
            mime_type: mime.map(str::to_string),
            text: body.to_string(),
            meta: None,
        }
    }

    fn blob_contents(uri: &str, mime: Option<&str>, blob: &str) -> ResourceContents {
        ResourceContents::BlobResourceContents {
            uri: uri.to_string(),
            mime_type: mime.map(str::to_string),
            blob: blob.to_string(),
            meta: None,
        }
    }

    #[test]
    fn format_resource_contents_strips_ansi_and_rtl_from_text() {
        let contents = vec![text_contents(
            "file:///evil.txt",
            Some("text/plain"),
            "before\x1b[2Jafter\u{202E}rtl",
        )];
        let out = format_resource_contents(&contents, 1_000_000).join("\n");
        assert!(!out.contains('\x1b'), "ANSI escape leaked: {out:?}");
        assert!(!out.contains('\u{202E}'), "RTL override leaked: {out:?}");
        assert!(out.contains("before"));
        assert!(out.contains("after"));
        assert!(out.contains("rtl"));
    }

    #[test]
    fn format_resource_contents_strips_control_chars_from_uri_and_mime() {
        let contents = vec![text_contents(
            "file:///a\u{200B}b",
            Some("text/\x07plain"),
            "body",
        )];
        let out = format_resource_contents(&contents, 1_000_000).join("\n");
        assert!(!out.contains('\u{200B}'));
        assert!(!out.contains('\x07'));
        assert!(out.contains("file:///ab"));
    }

    #[test]
    fn format_resource_contents_truncates_when_text_exceeds_cap() {
        // 1 MiB text body with a 1 KiB cap: must bail out, not emit the body.
        let body = "X".repeat(1024 * 1024);
        let contents = vec![text_contents("file:///big.txt", Some("text/plain"), &body)];
        let out = format_resource_contents(&contents, 1024).join("\n");
        assert!(out.contains("truncated"));
        assert!(out.contains("1024"));
        // The giant body itself must NOT have been emitted.
        assert!(
            !out.contains(&"X".repeat(2048)),
            "truncation failed to omit body"
        );
    }

    #[test]
    fn format_resource_contents_truncates_when_blob_exceeds_cap() {
        let blob = "B".repeat(1024 * 1024);
        let contents = vec![blob_contents(
            "file:///big.bin",
            Some("application/octet-stream"),
            &blob,
        )];
        let out = format_resource_contents(&contents, 1024).join("\n");
        assert!(out.contains("truncated"));
        assert!(out.contains("blob would exceed"));
        assert!(
            !out.contains(&"B".repeat(2048)),
            "blob leaked despite truncation"
        );
    }

    #[test]
    fn format_resource_contents_stops_after_first_truncation() {
        // First entry fills the budget, second should be dropped entirely.
        let first = "Y".repeat(2048);
        let contents = vec![
            text_contents("file:///first.txt", Some("text/plain"), &first),
            text_contents("file:///second.txt", Some("text/plain"), "short"),
        ];
        let out = format_resource_contents(&contents, 1024);
        // The truncation marker for the first chunk is present, and no second-chunk line should
        // appear.
        let joined = out.join("\n");
        assert!(joined.contains("first.txt"));
        assert!(joined.contains("truncated"));
        assert!(!joined.contains("second.txt"));
        assert!(!joined.contains("short"));
    }

    #[test]
    fn format_resource_contents_under_cap_emits_all_chunks() {
        let contents = vec![
            text_contents("file:///a.txt", Some("text/plain"), "alpha"),
            text_contents("file:///b.txt", Some("text/plain"), "beta"),
        ];
        let out = format_resource_contents(&contents, 1_000_000);
        assert_eq!(out.len(), 2);
        assert!(out[0].contains("alpha"));
        assert!(out[1].contains("beta"));
    }

    /// The seven MCP resource tools stay deferred after registration, or every MCP-using session
    /// would carry seven extra tool schemas in its tools array on the first turn.
    #[tokio::test]
    async fn mcp_resource_tools_remain_deferred() {
        use crate::{
            config::{McpServerConfig, McpTransport},
            mcp::{McpClientContext, McpClientManager},
            tools::ToolRegistry,
        };

        let server_config = McpServerConfig {
            name: "fixture-srv".to_string(),
            transport: McpTransport::Stdio,
            command: Some("/bin/false".to_string()),
            args: None,
            env: None,
            url: None,
            headers: None,
            headers_helper: None,
            auth: None,
            permission: None,
            allowed_tools: None,
            disabled_tools: None,
            eager_load_tools: None,
            tool_permissions: None,
            trust_read_only_hint: None,
            disabled: None,
            required: None,
        };
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[server_config], None, None, context)
            .await
            .expect("prepare with one server should succeed");

        let registry = ToolRegistry::new();
        register_all(&registry, manager);

        let entries = registry.tool_catalog();
        let by_name: std::collections::HashMap<_, _> =
            entries.iter().map(|(n, _, _, d)| (n.clone(), *d)).collect();

        for name in [
            "mcp_resource_list",
            "mcp_resource_read",
            "mcp_prompt_list",
            "mcp_prompt_get",
            "mcp_resource_subscribe",
            "mcp_resource_unsubscribe",
            "mcp_resource_updates_list",
        ] {
            assert!(
                by_name.contains_key(name),
                "MCP resource tool {name} not registered"
            );
            assert!(
                by_name[name],
                "MCP resource tool {name} should still be deferred (would otherwise \
                 bloat the tools array on every MCP-enabled session)",
            );
        }
    }
}
