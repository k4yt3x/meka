//! `agsh mcp …` management subcommands.

use std::sync::Arc;

use crate::config::{McpServerConfig, McpTransport};
use crate::error::{AgshError, Result};
use crate::mcp::{McpClientContext, McpClientManager};
use crate::session::TokenStore;

fn config_err(message: impl Into<String>) -> AgshError {
    AgshError::Config(message.into())
}

/// Run `agsh mcp list` — print configured servers + their transport/URL.
pub async fn run_list(servers: &[McpServerConfig]) -> Result<()> {
    if servers.is_empty() {
        println!("(no MCP servers configured)");
        return Ok(());
    }
    println!("{:<24} {:<8} {:<8} target", "name", "transport", "perm");
    println!("{}", "-".repeat(72));
    for config in servers {
        let target = match config.transport {
            McpTransport::Stdio => {
                let args = config
                    .args
                    .as_ref()
                    .map(|a| a.join(" "))
                    .unwrap_or_default();
                format!(
                    "{} {}",
                    config.command.as_deref().unwrap_or("(no command)"),
                    args
                )
                .trim()
                .to_string()
            }
            McpTransport::Http => config.url.clone().unwrap_or_else(|| "(no url)".to_string()),
        };
        println!(
            "{:<24} {:<8} {:<8} {}",
            config.name,
            match config.transport {
                McpTransport::Stdio => "stdio",
                McpTransport::Http => "http",
            },
            config.permission.as_deref().unwrap_or("read"),
            target
        );
    }
    Ok(())
}

/// Run `agsh mcp get <name>` — print a single server config in detail.
pub async fn run_get(servers: &[McpServerConfig], name: &str) -> Result<()> {
    let config = servers
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| config_err(format!("no MCP server named '{}'", name)))?;
    println!("name:        {}", config.name);
    println!(
        "transport:   {}",
        match config.transport {
            McpTransport::Stdio => "stdio",
            McpTransport::Http => "http",
        }
    );
    println!(
        "permission:  {}",
        config.permission.as_deref().unwrap_or("read")
    );
    if let Some(command) = &config.command {
        println!("command:     {}", command);
    }
    if let Some(args) = &config.args {
        println!("args:        {:?}", args);
    }
    if let Some(env) = &config.env {
        println!("env:         {} keys", env.len());
        for key in env.keys() {
            println!("  - {}", key);
        }
    }
    if let Some(url) = &config.url {
        println!("url:         {}", url);
    }
    if let Some(headers) = &config.headers {
        println!("headers:     {} entries", headers.len());
    }
    if config.auth_token.is_some() {
        println!("auth_token:  (set)");
    }
    if let Some(auth) = &config.auth {
        println!("auth:        {:?}", std::mem::discriminant(auth));
    }
    println!("sampling:    {}", config.sampling);
    Ok(())
}

/// Run `agsh mcp reconnect <name>` — connect once as a smoke test, print
/// `ok` on success and the error otherwise. Does not mutate config.
pub async fn run_reconnect(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    let config = servers
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| config_err(format!("no MCP server named '{}'", name)))?
        .clone();

    let context = McpClientContext::new();
    let manager = McpClientManager::connect_all(
        std::slice::from_ref(&config),
        Some(token_store),
        Arc::clone(&context),
    )
    .await?;

    if manager.server_names().contains(&config.name) {
        println!("ok: connected to '{}'", config.name);
        manager.shutdown().await;
        Ok(())
    } else {
        Err(config_err(format!(
            "failed to connect to '{}' — see logs above",
            config.name
        )))
    }
}

/// Run `agsh mcp logout <name>` — clear any stored OAuth credentials for
/// the given server, and clear the auth-probe cache entry (if any).
pub async fn run_logout(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    // Best-effort revocation — cleared from stored creds regardless.
    if let Some(config) = servers
        .iter()
        .find(|c| c.name == name && matches!(c.transport, McpTransport::Http))
        && let Err(error) = crate::mcp::revoke_stored_token(token_store, &config.name).await
    {
        tracing::warn!(
            "failed to revoke token at server '{}': {} (continuing)",
            config.name,
            error
        );
    }

    token_store.clear_mcp_credentials(name).await?;
    token_store.clear_auth_probe(name).await?;
    println!("ok: cleared credentials for '{}'", name);
    Ok(())
}

/// Run `agsh mcp login <name>` — force an interactive OAuth flow.
pub async fn run_login(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    let config = servers
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| config_err(format!("no MCP server named '{}'", name)))?
        .clone();
    if config.auth.is_none() {
        return Err(config_err(format!(
            "server '{}' has no 'auth' configured; nothing to log in to",
            name
        )));
    }

    token_store.clear_mcp_credentials(name).await?;
    token_store.clear_auth_probe(name).await?;

    let context = McpClientContext::new();
    let manager =
        McpClientManager::connect_all(std::slice::from_ref(&config), Some(token_store), context)
            .await?;

    if manager.server_names().contains(&config.name) {
        println!("ok: authorized '{}'", config.name);
        manager.shutdown().await;
        Ok(())
    } else {
        Err(config_err(format!(
            "OAuth flow did not complete for '{}'",
            config.name
        )))
    }
}

/// Run `agsh mcp add …` — persist a new server config into config.toml.
pub fn run_add(args: AddArgs) -> Result<()> {
    use crate::mcp::sanitize::{is_reserved_server_name, normalize_server_name};

    let normalized = normalize_server_name(&args.name);
    if normalized != args.name {
        return Err(config_err(format!(
            "server name '{}' contains invalid characters (would normalise to '{}')",
            args.name, normalized
        )));
    }
    if is_reserved_server_name(&args.name) {
        return Err(config_err(format!(
            "server name '{}' is reserved",
            args.name
        )));
    }

    let path = crate::config::config_file_path()
        .ok_or_else(|| config_err("could not determine config directory"))?;

    // Propagate every read error except "the file does not exist yet". The
    // previous `unwrap_or_default()` would happily silently overwrite a
    // file we merely lacked permission to read.
    let existing = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(config_err(format!(
                "failed to read existing config at {}: {}",
                path.display(),
                error
            )));
        }
    };
    let mut document = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| config_err(format!("failed to parse existing config: {}", error)))?;

    let servers_array = document
        .entry("mcp")
        .or_insert(toml_edit::Item::Table(toml_edit::Table::new()))
        .as_table_mut()
        .ok_or_else(|| config_err("config 'mcp' is not a table"))?
        .entry("servers")
        .or_insert(toml_edit::Item::ArrayOfTables(
            toml_edit::ArrayOfTables::new(),
        ))
        .as_array_of_tables_mut()
        .ok_or_else(|| config_err("config 'mcp.servers' is not an array of tables"))?;

    for existing_entry in servers_array.iter() {
        if existing_entry
            .get("name")
            .and_then(|v| v.as_str())
            .is_some_and(|n| n == args.name)
        {
            return Err(config_err(format!(
                "server '{}' already exists in config",
                args.name
            )));
        }
    }

    let mut table = toml_edit::Table::new();
    table.insert("name", toml_edit::value(args.name.clone()));
    table.insert(
        "transport",
        toml_edit::value(match args.transport {
            McpTransport::Stdio => "stdio",
            McpTransport::Http => "http",
        }),
    );
    if let Some(command) = &args.command {
        table.insert("command", toml_edit::value(command.clone()));
    }
    if let Some(cli_args) = &args.args {
        let mut arr = toml_edit::Array::new();
        for arg in cli_args {
            arr.push(arg);
        }
        table.insert("args", toml_edit::Item::Value(toml_edit::Value::Array(arr)));
    }
    if let Some(env) = &args.env {
        let mut env_table = toml_edit::InlineTable::new();
        for (k, v) in env {
            env_table.insert(k, toml_edit::Value::from(v.as_str()));
        }
        table.insert(
            "env",
            toml_edit::Item::Value(toml_edit::Value::InlineTable(env_table)),
        );
    }
    if let Some(url) = &args.url {
        table.insert("url", toml_edit::value(url.clone()));
    }
    if let Some(permission) = &args.permission {
        table.insert("permission", toml_edit::value(permission.clone()));
    }

    servers_array.push(table);

    crate::config::write_config_atomic(&path, &document.to_string())
        .map_err(|error| config_err(format!("failed to write config: {}", error)))?;
    println!("ok: added '{}' to {}", args.name, path.display());
    Ok(())
}

/// Run `agsh mcp remove <name>` — delete the entry from config.toml + clear
/// any stored credentials/probe cache.
pub async fn run_remove(name: &str, token_store: &TokenStore) -> Result<()> {
    let path = crate::config::config_file_path()
        .ok_or_else(|| config_err("could not determine config directory"))?;
    let existing = std::fs::read_to_string(&path)
        .map_err(|error| config_err(format!("failed to read config: {}", error)))?;
    let mut document = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| config_err(format!("failed to parse config: {}", error)))?;

    let servers = document
        .get_mut("mcp")
        .and_then(|m| m.as_table_mut())
        .and_then(|t| t.get_mut("servers"))
        .and_then(|s| s.as_array_of_tables_mut());

    let Some(servers) = servers else {
        return Err(config_err(
            "no MCP servers in config; nothing to remove".to_string(),
        ));
    };

    let original_len = servers.len();
    servers.retain(|entry| entry.get("name").and_then(|v| v.as_str()) != Some(name));
    if servers.len() == original_len {
        return Err(config_err(format!("no server named '{}' in config", name)));
    }

    crate::config::write_config_atomic(&path, &document.to_string())
        .map_err(|error| config_err(format!("failed to write config: {}", error)))?;

    token_store.clear_mcp_credentials(name).await?;
    token_store.clear_auth_probe(name).await?;
    // Drop any accumulated resource-update ledger entries for this server so
    // `list_mcp_resource_updates` doesn't keep reporting stale URIs after the
    // server's configuration has been removed.
    crate::mcp::resource_updates::clear_for_server(name);
    println!("ok: removed '{}' from {}", name, path.display());
    Ok(())
}

/// Arguments collected by the `agsh mcp add` CLI path.
pub struct AddArgs {
    pub name: String,
    pub transport: McpTransport,
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
    pub env: Option<Vec<(String, String)>>,
    pub url: Option<String>,
    pub permission: Option<String>,
}
