//! `meka mcp …` management subcommands.

use std::sync::Arc;

use crate::{
    config::{McpAuthConfig, McpServerConfig, McpTransport, ResolvedConfig},
    error::{MekaError, Result},
    mcp::{McpClientContext, McpClientManager},
    store::{Store, TokenStore},
    view::{McpServerDetail, McpServerView, McpToolView, McpToolsResponse},
};

fn config_err(message: impl Into<String>) -> MekaError {
    MekaError::Config(message.into())
}

/// `/mcp list` in the REPL: the configured servers with their transport and target.
///
/// With a `manager`, the table also carries a `State` column with each server's live lifecycle
/// state; `meka mcp list` has no running manager and gets the table without it.
pub(crate) async fn run_list(
    servers: &[McpServerConfig],
    manager: Option<&std::sync::Arc<crate::mcp::McpClientManager>>,
    token_store: &TokenStore,
) -> Result<()> {
    list_servers(
        servers,
        manager,
        token_store,
        crate::cli::OutputFormat::Plain,
    )
    .await
}

/// [`run_list`] under a chosen format; the REPL's `/mcp list` has no format flag and takes the door
/// above. `--format json` prints `{"servers": [...]}` and leaves the orphan report on stderr.
pub(crate) async fn list_servers(
    servers: &[McpServerConfig],
    manager: Option<&std::sync::Arc<crate::mcp::McpClientManager>>,
    token_store: &TokenStore,
    format: crate::cli::OutputFormat,
) -> Result<()> {
    // Computed before the early return below: "every server is gone but the OAuth bundles are still
    // here" is precisely the state worth reporting, and it is the one that return would hide.
    let orphans = orphaned_credentials(servers, token_store).await?;

    if format == crate::cli::OutputFormat::Json {
        let views: Vec<McpServerView> = servers.iter().map(McpServerView::from).collect();
        crate::cli::write_json_listing("servers", &views)?;
        report_orphaned_credentials(&orphans)?;
        return Ok(());
    }
    if servers.is_empty() {
        crate::streams::write_stderr_line("No MCP servers.");
        report_orphaned_credentials(&orphans)?;
        return Ok(());
    }

    // Resolve state once up front so the table output stays consistent even if a server connects
    // mid-print.
    let mut states: std::collections::HashMap<&str, &'static str> =
        std::collections::HashMap::new();
    if let Some(manager) = manager {
        for config in servers {
            let label = match manager.server_entry(&config.name) {
                Some(entry) => entry.state().await.label(),
                None => "(unknown)",
            };
            states.insert(config.name.as_str(), label);
        }
    }

    let with_state = manager.is_some();

    let rows: Vec<Vec<String>> = servers
        .iter()
        .map(|config| {
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
            let transport_label = config.transport.name();
            let perm_label = config.permission.map_or("(unset)", |level| level.name());

            // `required` is settled during config resolution, so `None` only shows up for a
            // config assembled outside that path; it means the same thing as false. A disabled
            // server is never started and so never gates, whatever `required` says, and this view
            // has no State column to reveal that on its own.
            let required_label = match (
                config.required.unwrap_or(false),
                config.disabled.unwrap_or(false),
            ) {
                (_, true) => "n/a (disabled)",
                (true, false) => "yes",
                (false, false) => "no",
            };

            let mut row = vec![config.name.clone(), transport_label.to_string()];
            if with_state {
                let state_label = states
                    .get(config.name.as_str())
                    .copied()
                    .unwrap_or("(unknown)");
                row.push(state_label.to_string());
            }
            row.push(required_label.to_string());
            row.push(perm_label.to_string());
            row.push(target);
            row
        })
        .collect();

    let headers: &[&str] = if with_state {
        &[
            "Name",
            "Transport",
            "State",
            "Required",
            "Permission",
            "Target",
        ]
    } else {
        &["Name", "Transport", "Required", "Permission", "Target"]
    };
    crate::render::write_stdout(crate::text::format_columns(headers, &rows))?;
    report_orphaned_credentials(&orphans)?;
    Ok(())
}

/// Server names holding stored OAuth credentials that no configured server claims.
///
/// Credentials are keyed by server name and nothing deletes them when the `[[mcp.servers]]` entry
/// goes away by hand, so an OAuth bundle can outlive its server indefinitely. This diff is the only
/// thing that can name one, and `meka mcp remove <name>` then clears it.
///
/// Safe to compute here only because every caller of `run_list` has already failed on an unreadable
/// config. An empty server list that came from a config meka could not parse would report every
/// credential in the database as an orphan.
async fn orphaned_credentials(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
) -> Result<Vec<String>> {
    Ok(token_store
        .list_mcp_credential_servers()
        .await?
        .into_iter()
        .filter(|name| !servers.iter().any(|config| &config.name == name))
        .collect())
}

/// Print the orphan block, if there is one. The names go to stderr with a hint: the listing is the
/// requested data and this is a diagnostic about the store, so `meka mcp list 2>/dev/null | awk`
/// must not see it as rows.
fn report_orphaned_credentials(orphans: &[String]) -> Result<()> {
    if orphans.is_empty() {
        return Ok(());
    }
    crate::streams::write_stderr_line("");
    crate::streams::write_stderr_line(format!(
        "Stored credentials with no server: {}",
        orphans.join(", ")
    ));
    // The action only: deleting the entry by hand is the usual cause, but a rollback that itself
    // failed leaves the same trace, so a hint naming one cause would mislead the other.
    crate::render::render_hint("delete one with `meka mcp remove <name>`");
    Ok(())
}

/// The configured server `name` denotes, or the refusal naming the ones that exist.
fn require_server<'a>(servers: &'a [McpServerConfig], name: &str) -> Result<&'a McpServerConfig> {
    servers
        .iter()
        .find(|server| server.name == name)
        .ok_or_else(|| {
            config_err(crate::text::unknown_name(
                "MCP server",
                name,
                servers.iter().map(|server| server.name.as_str()),
            ))
        })
}

/// Run `meka mcp get <name>`. Prints a single server config in detail.
pub(crate) async fn run_get(
    servers: &[McpServerConfig],
    name: &str,
    token_store: &TokenStore,
    format: crate::cli::OutputFormat,
) -> Result<()> {
    let config = require_server(servers, name)?;
    let mut credentials = Vec::new();
    for kind in [
        crate::store::McpCredentialKind::Bearer,
        crate::store::McpCredentialKind::ClientSecret,
        crate::store::McpCredentialKind::OAuth,
    ] {
        if token_store
            .load_mcp_credentials(name, kind)
            .await?
            .is_some()
        {
            credentials.push(kind.label());
        }
    }
    // The endpoint the stored grant was minted against, beside the one this config points at.
    //
    // A credential row is keyed by `(server_name, kind)` and by nothing else, so editing `url` in
    // place (a staging host promoted to production, a tenant moved) leaves a token issued by one
    // server being presented to another, with every other line of this report agreeing that the
    // server is configured and authorized. Reporting only, because a mismatch is not necessarily
    // wrong: an issuer legitimately differs from the resource server, and a host that moved behind
    // the same issuer is a rename rather than a new grant. What it cannot be is invisible.
    let credential_origin = crate::mcp::auth::stored_credential_origin(token_store, name).await;
    // A server with no `url` (a stdio one that was HTTP once, and still holds the bundle from then)
    // has nothing to disagree with, so it gets the origin and no verdict; comparing against `""`
    // would report every such server as a mismatch against an empty URL.
    let credential_origin_matches_url = credential_origin.as_deref().and_then(|origin| {
        config
            .url
            .as_deref()
            .map(|configured| crate::mcp::auth::same_origin(origin, configured))
    });
    let sorted_keys = |map: Option<&std::collections::HashMap<String, String>>| -> Vec<String> {
        let mut keys: Vec<String> = map
            .map(|map| map.keys().cloned().collect())
            .unwrap_or_default();
        keys.sort();
        keys
    };
    let detail = McpServerDetail {
        server: McpServerView::from(config),
        env_keys: sorted_keys(config.env.as_ref()),
        header_keys: sorted_keys(config.headers.as_ref()),
        credentials,
        credential_origin,
        credential_origin_matches_url,
        // The `type` value as written in config.toml, not `Debug` on a discriminant: that prints
        // an opaque `Discriminant(1)` and tells the reader nothing about which flow is configured.
        auth: config.auth.as_ref().map(|auth| match auth {
            McpAuthConfig::ClientCredentials { .. } => "client_credentials",
            McpAuthConfig::ClientCredentialsJwt { .. } => "client_credentials_jwt",
            McpAuthConfig::OAuth { .. } => "oauth",
        }),
        allowed_tools: config.allowed_tools.clone(),
        disabled_tools: config.disabled_tools.clone(),
        tool_permissions: config
            .tool_permissions
            .as_ref()
            .filter(|permissions| !permissions.is_empty())
            .map(|permissions| {
                permissions
                    .iter()
                    .map(|(tool, level)| (tool.clone(), *level))
                    .collect()
            }),
    };
    if format == crate::cli::OutputFormat::Json {
        crate::cli::write_json(&detail)?;
        return Ok(());
    }

    let server = &detail.server;
    let mut fields = vec![
        ("name", server.name.clone()),
        ("transport", server.transport.to_string()),
    ];
    // Same order as the `meka mcp list` columns. A disabled server is never started, so it never
    // reaches the turn gate no matter what `required` says; claiming it "gates the turn" would be
    // a flat falsehood, and `default_required = true` seeds `required` on disabled servers too.
    if server.disabled {
        fields.push(("required", "n/a (server is disabled)".to_string()));
        fields.push(("disabled", "yes (skipped at startup)".to_string()));
    } else {
        fields.push((
            "required",
            if server.required {
                "yes (gates the turn)"
            } else {
                "no"
            }
            .to_string(),
        ));
    }
    fields.push((
        "permission",
        server
            .permission
            .map_or("(unset)", |level| level.name())
            .to_string(),
    ));
    if let Some(command) = &server.command {
        fields.push(("command", command.clone()));
    }
    if let Some(args) = &server.args {
        fields.push(("args", args.join(" ")));
    }
    if !detail.env_keys.is_empty() {
        fields.push(("env", detail.env_keys.join(", ")));
    }
    if let Some(url) = &server.url {
        fields.push(("url", url.clone()));
    }
    if !detail.header_keys.is_empty() {
        fields.push(("headers", detail.header_keys.join(", ")));
    }
    if !detail.credentials.is_empty() {
        fields.push(("credentials", detail.credentials.join(", ")));
    }
    if let Some(origin) = &detail.credential_origin {
        fields.push((
            "issued for",
            match (detail.credential_origin_matches_url, &server.url) {
                (Some(false), Some(configured)) => {
                    format!("{origin} (does not match url: {configured})")
                }
                _ => origin.clone(),
            },
        ));
    }
    if let Some(auth) = detail.auth {
        fields.push(("auth", auth.to_string()));
    }
    if let Some(allowed) = &detail.allowed_tools {
        fields.push(("allowed_tools", allowed.join(", ")));
    }
    if let Some(disabled) = &detail.disabled_tools {
        fields.push(("disabled_tools", disabled.join(", ")));
    }
    if let Some(permissions) = &detail.tool_permissions {
        fields.push((
            "tool_permissions",
            permissions
                .iter()
                .map(|(tool, level)| format!("{tool} = {level}"))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    crate::render::write_stdout(crate::text::format_fields(&fields))?;
    Ok(())
}

/// Run `meka mcp tools <name>`: connect to the server, list every advertised tool, resolve
/// permissions, and print a column-aligned table. Disabled-by-allow/block tools are still shown
/// (marked `blocked`) so users can edit their config without leaving the CLI to discover names.
pub(crate) async fn run_tools(
    servers: &[McpServerConfig],
    mcp_default: Option<crate::permission::Permission>,
    token_store: &TokenStore,
    name: &str,
    format: crate::cli::OutputFormat,
) -> Result<()> {
    let config = require_server(servers, name)?.clone();

    let context = McpClientContext::new();
    let manager = McpClientManager::prepare(
        std::slice::from_ref(&config),
        mcp_default,
        Some(token_store.clone()),
        Arc::clone(&context),
    )
    .await?;
    manager.start_connector(crate::mcp::McpRuntimeConfig {
        connect_timeout: std::time::Duration::from_secs(30),
        stdio_concurrency: 1,
        http_concurrency: 1,
    });
    manager.await_settled().await;

    let connected = if let Some(entry) = manager.server_entry(&config.name) {
        matches!(
            &*entry.state.read().await,
            crate::mcp::ServerState::Connected { .. }
        )
    } else {
        false
    };
    if !connected {
        return Err(config_err(format!(
            "failed to connect to '{}'; see logs above",
            config.name
        )));
    }

    let tools = manager.list_advertised_tools(&config.name).await?;
    // Read before the shutdown consumes the manager.
    let dropped = manager
        .server_entry(&config.name)
        .map(|entry| {
            entry
                .dropped_tools
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .unwrap_or(0);
    manager.shutdown_arc().await;

    if format == crate::cli::OutputFormat::Json {
        crate::cli::write_json(&McpToolsResponse {
            server: config.name.clone(),
            tools: tools.iter().map(McpToolView::from).collect(),
        })?;
        warn_about_dropped_tools(dropped);
        return Ok(());
    }
    if tools.is_empty() {
        crate::streams::write_stderr_line("No tools.");
        return Ok(());
    }

    let rows: Vec<Vec<String>> = tools
        .iter()
        .map(|tool| {
            vec![
                // The server chose this name and it is never validated on the way in, so it
                // reaches this table exactly as sent. The one listing an operator reads to decide
                // what a server may do is not a place to reproduce a newline or an escape.
                //
                // Sanitized but never truncated, unlike every other authored cell here. This is
                // the string the user retypes into `tools`, a per-tool `permission` override or
                // `--eager-load-tool`, and no command prints it in full elsewhere, so a name cut
                // to fit a column matches nothing, silently.
                crate::text::sanitize_to_line(&tool.raw_name, usize::MAX),
                tool.resolved_permission.to_string(),
                // A declined hint has to say so here, because this table is where a user checks
                // what `trust_read_only_hint = false` moved, and the winning source alone cannot
                // tell "the server offered nothing" from "the server offered and meka refused".
                if tool.read_only_hint_declined {
                    format!(
                        "{} (readOnlyHint declined)",
                        tool.permission_source.as_str()
                    )
                } else {
                    tool.permission_source.as_str().to_string()
                },
                if tool.allowed { "allowed" } else { "blocked" }.to_string(),
                describe_one_line(&tool.description),
            ]
        })
        .collect();
    crate::render::write_stdout(crate::text::format_columns(
        &["Name", "Permission", "Source", "Status", "Description"],
        &rows,
    ))?;

    // Commentary on the table, not part of it: on stdout it would append a sentence to the data a
    // caller piped.
    let total = tools.len();
    let allowed = tools.iter().filter(|tool| tool.allowed).count();
    crate::streams::write_stderr_line("");
    crate::streams::write_stderr_line(format!(
        "{} tool{} total, {} allowed, {} blocked",
        total,
        if total == 1 { "" } else { "s" },
        allowed,
        total - allowed
    ));

    warn_about_dropped_tools(dropped);
    Ok(())
}

/// A tool meka dropped to stay under the per-server ceiling is, from here, indistinguishable from
/// one the server never offered. Say so: this listing is where someone goes to find out why the
/// model cannot call something the server's own docs advertise, and a `tracing::warn!` at connect
/// time is not where they will be looking.
fn warn_about_dropped_tools(dropped: usize) {
    if dropped > 0 {
        crate::streams::write_stderr_line(format!(
            "{} further tool{} advertised by this server {} not registered: the per-server ceiling \
             is {}",
            dropped,
            if dropped == 1 { "" } else { "s" },
            if dropped == 1 { "was" } else { "were" },
            crate::mcp::MAX_MCP_TOOLS_PER_SERVER,
        ));
    }
}

/// Collapse a (possibly multi-line) description into one short line so the table stays legible. MCP
/// descriptions can be kilobytes; the first sentence or ~80 chars is enough for a listing.
fn describe_one_line(description: &str) -> String {
    const MAX: usize = 80;
    let mut collapsed = String::with_capacity(description.len().min(MAX + 8));
    let mut prev_space = false;
    for ch in description.chars() {
        if ch.is_whitespace() {
            if !prev_space && !collapsed.is_empty() {
                collapsed.push(' ');
            }
            prev_space = true;
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
        if collapsed.chars().count() > MAX {
            break;
        }
    }
    let trimmed = collapsed.trim_end();
    if trimmed.chars().count() > MAX {
        let clipped: String = trimmed.chars().take(MAX).collect();
        format!("{}…", clipped.trim_end())
    } else {
        trimmed.to_string()
    }
}

/// Run `meka mcp reconnect <name>`: connect once as a smoke test, print `ok` on success and the
/// error otherwise. Does not mutate config.
pub(crate) async fn run_reconnect(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    let config = require_server(servers, name)?.clone();

    let context = McpClientContext::new();
    // No `[mcp].default_permission`: this is a smoke test with no `ResolvedConfig` in scope, and
    // the per-tool resolution falls through to its strict default.
    let manager = McpClientManager::prepare(
        std::slice::from_ref(&config),
        None,
        Some(token_store.clone()),
        Arc::clone(&context),
    )
    .await?;
    manager.start_connector(crate::mcp::McpRuntimeConfig {
        connect_timeout: std::time::Duration::from_secs(30),
        stdio_concurrency: 1,
        http_concurrency: 1,
    });
    manager.await_settled().await;

    let connected = if let Some(entry) = manager.server_entry(&config.name) {
        matches!(
            &*entry.state.read().await,
            crate::mcp::ServerState::Connected { .. }
        )
    } else {
        false
    };

    if connected {
        tracing::info!("connected to '{name}'", name = config.name);
        manager.shutdown_arc().await;
        Ok(())
    } else {
        Err(config_err(format!(
            "failed to connect to '{}'; see logs above",
            config.name
        )))
    }
}

/// Run `meka mcp logout <name>`: clear any stored OAuth credentials for the given server.
pub(crate) async fn run_logout(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    // Best-effort revocation; the stored credentials are cleared regardless.
    if let Some(config) = servers
        .iter()
        .find(|c| c.name == name && matches!(c.transport, McpTransport::Http))
        && let Err(error) = crate::mcp::auth::revoke_stored_token(token_store, &config.name).await
    {
        tracing::warn!(
            "failed to revoke the token at '{name}': {error}",
            name = config.name
        );
    }

    // Named, because the kinds are not equally replaceable: an OAuth bundle is reobtained by
    // logging in again, while a bearer or a client secret was typed by the user and meka is now its
    // only holder. `logout` still clears them all, but says which of the user's own secrets went.
    let cleared: Vec<&str> = {
        let mut found = Vec::new();
        for kind in [
            crate::store::McpCredentialKind::Bearer,
            crate::store::McpCredentialKind::ClientSecret,
            crate::store::McpCredentialKind::OAuth,
        ] {
            if token_store
                .load_mcp_credentials(name, kind)
                .await?
                .is_some()
            {
                found.push(kind.label());
            }
        }
        found
    };
    token_store.clear_mcp_credentials(name).await?;
    if cleared.is_empty() {
        tracing::info!("'{name}' had no stored credentials");
    } else {
        tracing::info!(
            "cleared {cleared} for '{name}'",
            cleared = cleared.join(", ")
        );
    }
    Ok(())
}

/// Run `meka mcp login <name> --auth-token-stdin` or `--client-secret-stdin`: record a secret the
/// user already holds.
///
/// Separate from [`run_login`] because the two do opposite things: that one goes and obtains a
/// credential, this one is handed one. A confidential OAuth client needs both, in this order:
/// deposit the client secret, then run the flow that presents it.
pub(crate) async fn run_store_secret(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
    kind: crate::store::McpCredentialKind,
    secret: &str,
) -> Result<()> {
    use crate::store::McpCredentialKind;

    let server = require_server(servers, name)?;
    // Checked against config rather than accepted for any name, so a typo leaves a stranded
    // credential under a server that will never exist rather than silently doing nothing useful.
    if server.transport != McpTransport::Http {
        return Err(config_err(format!(
            "server '{name}' is stdio and has no HTTP credential; its secrets go in `env`"
        )));
    }

    // The states `resolve_add_args` refuses at add time, refused again here, because this is the
    // other door onto the same state and a rule enforced at one of two doors is not enforced.
    match (kind, &server.auth) {
        (McpCredentialKind::Bearer, Some(_)) => {
            return Err(config_err(format!(
                "server '{name}' authenticates through its `[auth]` block, which a stored bearer \
                 would override; drop the block first"
            )));
        }
        (McpCredentialKind::ClientSecret, None) => {
            return Err(config_err(format!(
                "server '{name}' has no `[auth]` block to present a client secret; add one with \
                 `type = \"oauth\"` or `\"client_credentials\"` first"
            )));
        }
        (McpCredentialKind::ClientSecret, Some(McpAuthConfig::ClientCredentialsJwt { .. })) => {
            return Err(config_err(format!(
                "server '{name}' authenticates with a signed JWT, not a client secret; the key it \
                 signs with is `signing_key_path`"
            )));
        }
        _ => {}
    }

    token_store.save_mcp_credentials(name, kind, secret).await?;
    tracing::info!("stored {kind} for '{name}'", kind = kind.label());
    Ok(())
}

/// Connect `config` once with a terminal at hand for the browser step, and say whether it ended
/// connected. Split from `run_login` so every way the flow can end short of `Connected`, an error
/// included, reaches the one place that puts the previous bundle back.
async fn run_login_flow(config: &McpServerConfig, token_store: &TokenStore) -> Result<bool> {
    let context = McpClientContext::new();
    // The one place a person is at the terminal to finish a browser login, so the one place the
    // client is given a way to ask them.
    context.set_login_prompt(std::sync::Arc::new(TerminalLogin));
    // No `[mcp].default_permission`, for the reason `run_reconnect` gives.
    let manager = McpClientManager::prepare(
        std::slice::from_ref(config),
        None,
        Some(token_store.clone()),
        context,
    )
    .await?;
    manager.start_connector(crate::mcp::McpRuntimeConfig {
        connect_timeout: std::time::Duration::from_secs(30),
        stdio_concurrency: 1,
        http_concurrency: 1,
    });
    manager.await_settled().await;

    let connected = if let Some(entry) = manager.server_entry(&config.name) {
        matches!(
            &*entry.state.read().await,
            crate::mcp::ServerState::Connected { .. }
        )
    } else {
        false
    };
    manager.shutdown_arc().await;
    Ok(connected)
}

/// The terminal's side of an interactive MCP login: the URL on stderr, and the pasted callback read
/// from stdin when stdin is a terminal.
struct TerminalLogin;

#[async_trait::async_trait]
impl crate::mcp::auth::LoginPrompt for TerminalLogin {
    fn authorize_at(&self, url: &str) {
        // The URL, exactly once, and no browser launched: which browser, and on which machine, is
        // the user's to decide, and a launch from an SSH session or a container reached nothing or
        // the wrong desktop.
        crate::streams::write_stderr_line(format!(
            "To authorize, open this URL in your browser:\n\n{url}\n"
        ));
    }

    fn awaiting_callback(&self, seconds: u64, accepts_paste: bool) {
        if accepts_paste {
            crate::streams::write_stderr_line(format!(
                "Waiting up to {seconds}s for the callback, or paste the callback URL here and \
                 press Enter:"
            ));
        } else {
            crate::streams::write_stderr_line(format!(
                "Waiting up to {seconds}s for the callback."
            ));
        }
    }

    fn accepts_paste(&self) -> bool {
        std::io::IsTerminal::is_terminal(&std::io::stdin())
    }

    async fn read_pasted_line(&self) -> std::result::Result<Option<String>, String> {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let mut reader = BufReader::new(tokio::io::stdin());
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Err(error) => Err(format!("failed to read stdin: {error}")),
            // EOF: stdin was closed before the user pasted anything.
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(line)),
        }
    }
}

/// Run `meka mcp login <name>`: an interactive OAuth flow.
///
/// An explicit `[auth]` block is honored as written. An HTTP server without one is assumed to take
/// `type = "oauth"`, and on success that block is written back to `config.toml` so later runs do
/// not assume. A stdio server has nothing to log in to.
pub(crate) async fn run_login(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    let base_config = require_server(servers, name)?.clone();

    // Ahead of the branch below rather than inside it, because a stored bearer makes *any* login
    // incoherent, not only the one that would invent an `[auth]` block. rmcp sends the transport's
    // `auth_header` when it is set and consults the authorization flow only when it is not, so the
    // flow would run, deposit a bundle, and the bearer would go out on every request regardless.
    // The user would see a login succeed and nothing change.
    if token_store
        .load_mcp_credentials(name, crate::store::McpCredentialKind::Bearer)
        .await?
        .is_some()
    {
        return Err(config_err(format!(
            "server '{name}' has a stored bearer, which would override the login; drop it first \
             with `meka mcp logout {name}`"
        )));
    }

    let (config, needs_persist) = if base_config.auth.is_some() {
        (base_config, false)
    } else {
        match base_config.transport {
            McpTransport::Http => {
                let mut assumed = base_config.clone();
                assumed.auth = Some(McpAuthConfig::OAuth {
                    client_id: None,
                    scopes: None,
                    redirect_port: None,
                });
                tracing::info!("no [auth] block for '{name}'; assuming OAuth authorization_code");
                (assumed, true)
            }
            McpTransport::Stdio => {
                return Err(config_err(format!(
                    "server '{name}' is stdio; there is nothing to log in to"
                )));
            }
        }
    };

    // Only the bundle this flow is about to replace. A confidential client's stored `client_secret`
    // is an *input* here, so clearing every kind would delete the credential the login needs.
    //
    // Kept aside until the new flow has produced a bundle. Cleared and not restored, a login that
    // did not complete (a closed tab, a browser step outlasting the callback wait) logged the
    // server out everywhere, a running `meka serve` included on its next restart.
    let previous_bundle = token_store
        .load_mcp_credentials(name, crate::store::McpCredentialKind::OAuth)
        .await?;
    token_store
        .clear_mcp_credentials_of_kind(name, crate::store::McpCredentialKind::OAuth)
        .await?;

    let connected = run_login_flow(&config, token_store).await;
    let connected = match connected {
        Ok(true) => true,
        Ok(false) | Err(_) => {
            if let Some(bundle) = &previous_bundle
                && let Err(error) = token_store
                    .save_mcp_credentials(name, crate::store::McpCredentialKind::OAuth, bundle)
                    .await
            {
                tracing::warn!(
                    "failed to restore the previous OAuth bundle for '{name}' after an incomplete \
                     login: {error}"
                );
            }
            false
        }
    };
    if !connected {
        return Err(config_err(format!(
            "OAuth flow did not complete for '{}'",
            config.name
        )));
    }

    if needs_persist && let Err(error) = persist_auth_block_for(name) {
        // The login itself succeeded, so the write-back is a warning rather than a failure.
        tracing::warn!(
            "failed to write `[auth] type = \"oauth\"` for '{name}' to config.toml: {error}; the \
             login itself succeeded"
        );
    }

    tracing::info!("authorized '{name}'", name = config.name);
    Ok(())
}

/// Write `[mcp.servers.auth] type = "oauth"` for a named server that has no `auth` key, so the
/// assumption [`run_login`] made is recorded rather than reapplied on every later run.
fn persist_auth_block_for(name: &str) -> Result<()> {
    // Held to the end of the function, so this read and the write below cannot interleave with
    // another editor's, `device_id::persist` on an ordinary launch included.
    let _config_lock = crate::config::lock_config_file()
        .map_err(|error| config_err(format!("failed to lock config: {error}")))?;
    let path = crate::paths::config_file_path()
        .ok_or_else(|| config_err("failed to determine the config directory"))?;
    let existing = std::fs::read_to_string(&path)
        .map_err(|error| config_err(format!("failed to read config: {error}")))?;
    let mut document = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| config_err(format!("failed to parse config: {error}")))?;

    let mutated = {
        let servers = document
            .get_mut("mcp")
            .and_then(|m| m.as_table_mut())
            .and_then(|t| t.get_mut("servers"))
            .and_then(|s| s.as_array_of_tables_mut())
            .ok_or_else(|| config_err("config has no [[mcp.servers]] entries".to_string()))?;

        let mut target = None;
        for entry in servers.iter_mut() {
            if entry.get("name").and_then(|v| v.as_str()) == Some(name) {
                target = Some(entry);
                break;
            }
        }
        let entry = target.ok_or_else(|| {
            config_err(format!(
                "server '{name}' not found in [[mcp.servers]] after login"
            ))
        })?;
        if entry.contains_key("auth") {
            false
        } else {
            let mut auth_table = toml_edit::Table::new();
            auth_table.insert("type", toml_edit::value("oauth"));
            entry.insert("auth", toml_edit::Item::Table(auth_table));
            true
        }
    };

    if mutated {
        crate::fs::write_file_atomic(&path, &document.to_string())
            .map_err(|error| config_err(format!("failed to write config: {error}")))?;
    }
    Ok(())
}

/// Inputs for `meka mcp add`. Parsed into a [`ResolvedAddArgs`] by [`resolve_add_args`] which is
/// where transport auto-detection, flag compatibility, and the `McpAuthKind` → `McpAuthConfig`
/// mapping live. Keep this struct plain-data so the clap layer in `cli.rs` and the CLI integration
/// tests can both build one.
pub(crate) struct AddArgs {
    pub(crate) name: String,
    pub(crate) location: Option<String>,
    pub(crate) args: Vec<String>,
    pub(crate) transport: Option<McpTransport>,
    /// Raw `KEY=VALUE` entries from the CLI.
    pub(crate) env: Vec<String>,
    /// Raw `KEY=VALUE` entries from the CLI.
    pub(crate) header: Vec<String>,
    pub(crate) auth: Option<crate::cli::McpAuthKind>,
    pub(crate) auth_token: Option<String>,
    pub(crate) client_id: Option<String>,
    pub(crate) client_secret: Option<String>,
    pub(crate) signing_key: Option<String>,
    pub(crate) signing_algorithm: Option<String>,
    pub(crate) scope: Vec<String>,
    pub(crate) redirect_port: Option<u16>,
    pub(crate) permission: Option<String>,
    /// Skip the auto-login that runs when the probe reports auth-required or when `--auth oauth`
    /// was explicitly set.
    pub(crate) no_login: bool,
    /// Raw tool names to allow-list (only these register).
    pub(crate) allow_tool: Vec<String>,
    /// Raw tool names to block-list (never register).
    pub(crate) disable_tool: Vec<String>,
    /// Raw tool names to eager-load (skip `load_tool` round-trip).
    pub(crate) eager_load_tool: Vec<String>,
    /// Raw `NAME=LEVEL` pairs for per-tool permission overrides.
    pub(crate) tool_permission: Vec<String>,
    /// Persist with `required = true` so an unavailable server gates the turn instead of being
    /// skipped over. Omitted from the written table when false, which leaves the server inheriting
    /// `[mcp].default_required`.
    pub(crate) required: bool,
    /// Persist with `disabled = true` so the server is skipped at startup until the user runs
    /// `meka mcp enable <name>`.
    pub(crate) disabled: bool,
}

/// What `add` looks like after validation: transport is chosen, mutually-exclusive flag
/// combinations have been refused, and the `[auth]` block (if any) has been reduced to an
/// [`McpAuthConfig`] ready to be serialized into TOML.
#[cfg_attr(test, derive(Debug))]
struct ResolvedAddArgs {
    name: String,
    transport: McpTransport,
    /// Present iff `transport == Stdio`.
    command: Option<String>,
    /// Present iff `transport == Stdio` and there were trailing args.
    stdio_args: Vec<String>,
    /// Present iff `transport == Stdio` and `--env` was given.
    env: Vec<(String, String)>,
    /// Present iff `transport == Http`.
    url: Option<String>,
    /// Present iff `transport == Http` and `--header` was given.
    headers: Vec<(String, String)>,
    /// Secrets travel beside the config entry rather than in it, and [`run_add`] writes them to
    /// the store once the entry that names them is on disk. Neither ever reaches
    /// [`build_server_table`].
    auth_token: Option<String>,
    client_secret: Option<String>,
    auth: Option<McpAuthConfig>,
    permission: Option<crate::permission::Permission>,
    allowed_tools: Option<Vec<String>>,
    disabled_tools: Option<Vec<String>>,
    eager_load_tools: Option<Vec<String>>,
    tool_permissions: Option<std::collections::HashMap<String, crate::permission::Permission>>,
    no_login: bool,
    disabled: bool,
    required: bool,
}

/// Run `meka mcp add …`.
///
/// Persists the server into `config.toml`, then for an HTTP server probes the endpoint (RFC 6750 /
/// RFC 9728) and, when auth is required or `--auth oauth` was passed and `--no-login` was not, runs
/// the OAuth flow at once. A failed flow purges the entry just written and exits non-zero.
pub(crate) async fn run_add(args: AddArgs, token_store: &TokenStore) -> Result<()> {
    use crate::mcp::sanitize::{is_reserved_server_name, normalize_server_name};

    let normalized = normalize_server_name(&args.name);
    if normalized != args.name {
        return Err(config_err(format!(
            "server name '{}' has invalid characters; '{}' would be accepted",
            args.name, normalized
        )));
    }
    if is_reserved_server_name(&args.name) {
        return Err(config_err(format!(
            "server name '{}' is reserved",
            args.name
        )));
    }
    let resolved = resolve_add_args(args)?;

    // Held across this read and the write below, so the two cannot interleave with another
    // editor's (`device_id::persist` on an ordinary launch included), and released the moment the
    // write lands; see the `drop` below.
    let config_lock = crate::config::lock_config_file()
        .map_err(|error| config_err(format!("failed to lock config: {error}")))?;
    let path = crate::paths::config_file_path()
        .ok_or_else(|| config_err("failed to determine the config directory"))?;

    // Only a missing file starts from empty: treating any read failure as empty would overwrite a
    // file meka merely lacked permission to read.
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
        .map_err(|error| config_err(format!("failed to parse existing config: {error}")))?;

    let servers_array = document
        .entry("mcp")
        .or_insert(toml_edit::Item::Table(toml_edit::Table::new()))
        .as_table_mut()
        .ok_or_else(|| config_err("`mcp` in config.toml is not a table"))?
        .entry("servers")
        .or_insert(toml_edit::Item::ArrayOfTables(
            toml_edit::ArrayOfTables::new(),
        ))
        .as_array_of_tables_mut()
        .ok_or_else(|| config_err("`mcp.servers` in config.toml is not an array of tables"))?;

    for existing_entry in servers_array.iter() {
        if existing_entry
            .get("name")
            .and_then(|v| v.as_str())
            .is_some_and(|n| n == resolved.name)
        {
            return Err(config_err(format!(
                "server '{}' already exists in config.toml",
                resolved.name
            )));
        }
    }

    // A secret already filed under this name belongs to a server the loop above has just shown is
    // not in config: a hand-deleted `[[mcp.servers]]` entry leaves its credentials behind. Adding
    // the name again would attach the old bearer to whatever URL the new entry names, and clearing
    // it would destroy a credential without being asked, so neither: name what is in the way and
    // let `mcp remove` clear it properly, revoking at the provider on the way out.
    //
    // After the duplicate check so the message can say the entry is gone, and before the write so
    // the refusal needs no rollback.
    if token_store.has_mcp_credentials(&resolved.name).await? {
        return Err(config_err(format!(
            "'{0}' has a stored credential left over from a removed server; clear it first with \
             `meka mcp remove {0}`",
            resolved.name
        )));
    }

    let table = build_server_table(&resolved);
    servers_array.push(table);

    crate::fs::write_file_atomic(&path, &document.to_string())
        .map_err(|error| config_err(format!("failed to write config: {error}")))?;
    tracing::info!(
        "added '{name}' to {path}",
        name = resolved.name,
        path = path.display()
    );

    // Released before anything that talks to a network or waits on a human: `lock_config_file`
    // blocks with no timeout, so holding it across the OAuth flow would hang every other meka
    // launch that touches `config.toml` (`device_id::persist` does, on an ordinary start) for as
    // long as the browser login takes. `purge_server` takes the lock again for the rollback below.
    drop(config_lock);

    // Secrets are not config, so they land here rather than in the table just written. Straight
    // after that write, because a server whose entry exists without the credential it needs would
    // make unauthenticated requests and be told it is unauthorized, which says nothing about the
    // real cause. A failure rolls the entry back like every other post-persist failure.
    if let Err(error) = store_add_secrets(&resolved, token_store).await {
        tracing::warn!(
            "failed to store credentials for '{name}': {error}; rolling back the config entry",
            name = resolved.name
        );
        return Err(roll_back(&resolved.name, token_store, error).await);
    }

    // Stdio has no auth surface, and an HTTP server with a static bearer needs no login either;
    // everything else gets the probe. These short-circuits return before the Ctrl-C protection
    // below is needed.
    if resolved.transport != McpTransport::Http {
        return Ok(());
    }
    if resolved.auth_token.is_some() {
        return Ok(());
    }
    if resolved.url.is_none() {
        return Ok(());
    }

    // From here on, anything that goes wrong (natural failure, timeout, Ctrl-C) should leave
    // config.toml in the "never added" state. Race the probe + auto-login block against SIGINT so
    // an interrupted user ends up in exactly the same place as a clean `mcp remove`.
    let result: Result<()> = tokio::select! {
        biased;
        _ = tokio::signal::ctrl_c() => Err(MekaError::Interrupted),
        r = probe_then_login(&resolved, token_store) => r,
    };

    if let Err(error) = result {
        match &error {
            MekaError::Interrupted => {
                tracing::warn!("interrupted; rolling back '{name}'", name = resolved.name);
            }
            other => tracing::warn!(
                "authorization failed for '{name}': {other}; rolling back the config entry",
                name = resolved.name
            ),
        }
        return Err(roll_back(&resolved.name, token_store, error).await);
    }

    Ok(())
}

/// Write the secrets `add` was given to the store, under the kind each one is.
///
/// Two separate rows rather than one, because `--auth-token-stdin` and `--client-secret-stdin` are
/// mutually exclusive only by the flag rules above; the schema does not depend on that and neither
/// does this.
async fn store_add_secrets(resolved: &ResolvedAddArgs, token_store: &TokenStore) -> Result<()> {
    use crate::store::McpCredentialKind;

    if let Some(token) = &resolved.auth_token {
        token_store
            .save_mcp_credentials(&resolved.name, McpCredentialKind::Bearer, token)
            .await?;
    }
    if let Some(secret) = &resolved.client_secret {
        token_store
            .save_mcp_credentials(&resolved.name, McpCredentialKind::ClientSecret, secret)
            .await?;
    }
    Ok(())
}

/// Undo everything `add` wrote, so a failure anywhere after the config write leaves the user where
/// a clean `mcp remove` would. Returns `error` unchanged, for `return Err(roll_back(…).await)`.
///
/// The reason is logged by the caller, which is the only place that knows it; a rollback that
/// itself fails is reported here, because at that point the user has to finish it by hand.
async fn roll_back(name: &str, token_store: &TokenStore, error: MekaError) -> MekaError {
    if let Err(purge_error) = purge_server(name, token_store).await {
        tracing::warn!(
            "failed to roll back '{name}': {purge_error}; remove its entry from config.toml by hand"
        );
    }
    error
}

/// The "everything that can fail post-persist" block: probe, decide whether to auto-login, and run
/// the OAuth flow when warranted. Extracted so [`run_add`] can race it against a SIGINT handler and
/// roll back on either error path from one place.
async fn probe_then_login(resolved: &ResolvedAddArgs, token_store: &TokenStore) -> Result<()> {
    // The caller has already established HTTP transport, no `auth_token` and a URL; an absent URL
    // here is a bug in that caller, reported rather than panicked on so the rollback still runs.
    let Some(url) = resolved.url.as_deref() else {
        return Err(MekaError::Internal(format!(
            "probe_then_login for '{}' was reached without a URL",
            resolved.name
        )));
    };

    let wants_oauth_already = matches!(resolved.auth, Some(McpAuthConfig::OAuth { .. }));
    let should_login = match probe_and_announce(&resolved.name, url).await {
        // Probe says unauthenticated access works, and the user didn't explicitly ask for OAuth →
        // nothing to log in to.
        ProbeOutcome::Open => wants_oauth_already,
        ProbeOutcome::AuthRequired => true,
        ProbeOutcome::Inconclusive => wants_oauth_already,
    };

    if !should_login {
        return Ok(());
    }
    if resolved.no_login {
        tracing::info!(
            "skipping auto-login (--no-login); run `meka mcp login {name}` when ready",
            name = resolved.name
        );
        return Ok(());
    }

    // The entry just written, without a round trip through disk.
    let server_config = resolved_to_server_config(resolved);
    tracing::info!(
        "running OAuth authorization for '{name}' (use --no-login to skip)",
        name = resolved.name
    );
    run_login(
        std::slice::from_ref(&server_config),
        token_store,
        &resolved.name,
    )
    .await
}

/// The probe as `run_add` decides on it: the full `McpAuthProbe` is logged, and the auto-login
/// decision needs only these three states.
#[derive(Debug, PartialEq, Eq)]
enum ProbeOutcome {
    Open,
    AuthRequired,
    Inconclusive,
}

/// Probe the HTTP endpoint, print a one-line hint, and collapse the probe result into a
/// login-decision summary.
async fn probe_and_announce(name: &str, url: &str) -> ProbeOutcome {
    use crate::mcp::auth::McpAuthProbe;

    match crate::mcp::auth::probe_http_auth(url).await {
        McpAuthProbe::Open => {
            tracing::info!("probe: '{name}' reachable and does not require auth");
            ProbeOutcome::Open
        }
        McpAuthProbe::AuthRequired { resource_metadata } => {
            tracing::info!("probe: '{name}' requires OAuth");
            if let Some(meta) = resource_metadata {
                tracing::debug!("resource_metadata advertised by '{name}': {meta}");
            }
            ProbeOutcome::AuthRequired
        }
        McpAuthProbe::Unexpected { status } => {
            tracing::warn!("probe: '{name}' answered HTTP {status}; failed to infer auth state");
            ProbeOutcome::Inconclusive
        }
        McpAuthProbe::Unreachable { message } => {
            tracing::warn!("probe: failed to reach '{url}': {message}");
            ProbeOutcome::Inconclusive
        }
    }
}

/// Turn the raw CLI [`AddArgs`] into a validated [`ResolvedAddArgs`].
///
/// Auto-detects transport from the positional `location` when `--transport` is not given
/// (`http[s]://…` → http, anything else → stdio). Refuses every illegal flag combination at add
/// time, so a bad configuration never lands in `config.toml`.
fn resolve_add_args(args: AddArgs) -> Result<ResolvedAddArgs> {
    let AddArgs {
        name,
        location,
        args: tail,
        transport,
        env,
        header,
        auth,
        auth_token,
        client_id,
        client_secret,
        signing_key,
        signing_algorithm,
        scope,
        redirect_port,
        permission,
        no_login,
        allow_tool,
        disable_tool,
        eager_load_tool,
        tool_permission,
        disabled,
        required,
    } = args;

    let looks_like_url = location
        .as_deref()
        .map(|location| location.starts_with("http://") || location.starts_with("https://"))
        .unwrap_or(false);
    let transport = transport.unwrap_or(if looks_like_url {
        McpTransport::Http
    } else {
        McpTransport::Stdio
    });

    // Parsed through `Permission`'s own parser rather than a list restated here: keeping a second
    // copy of the vocabulary is how a retired spelling survives in one door after being removed
    // from the other.
    let permission = permission
        .as_deref()
        .map(|level| {
            level
                .parse::<crate::permission::Permission>()
                .map_err(|error| config_err(format!("`--permission`: {error}")))
        })
        .transpose()?;

    // Per-tool permission overrides arrive as `NAME=LEVEL` strings. Parse + validate here so bad
    // input never lands in config.toml.
    let tool_permissions = if tool_permission.is_empty() {
        None
    } else {
        let mut map = std::collections::HashMap::with_capacity(tool_permission.len());
        for entry in &tool_permission {
            let (tool, level) = entry.split_once('=').ok_or_else(|| {
                config_err(format!(
                    "`--tool-permission` takes TOOL=LEVEL, got '{entry}'"
                ))
            })?;
            let tool = tool.trim();
            let level = level.trim();
            if tool.is_empty() {
                return Err(config_err(format!(
                    "`--tool-permission` entry '{entry}' has an empty tool name"
                )));
            }
            let level = level
                .parse::<crate::permission::Permission>()
                .map_err(|error| {
                    config_err(format!("`--tool-permission` entry '{entry}': {error}"))
                })?;
            map.insert(tool.to_string(), level);
        }
        Some(map)
    };
    let allowed_tools = if allow_tool.is_empty() {
        None
    } else {
        Some(allow_tool)
    };
    let disabled_tools = if disable_tool.is_empty() {
        None
    } else {
        Some(disable_tool)
    };
    let eager_load_tools = if eager_load_tool.is_empty() {
        None
    } else {
        Some(eager_load_tool)
    };

    let auth_flags_present = client_id.is_some()
        || client_secret.is_some()
        || signing_key.is_some()
        || signing_algorithm.is_some()
        || !scope.is_empty()
        || redirect_port.is_some();

    if auth_token.is_some() && auth.is_some() {
        return Err(config_err(
            "`--auth-token-stdin` cannot be combined with `--auth`",
        ));
    }

    match transport {
        McpTransport::Stdio => {
            let command = location.ok_or_else(|| {
                config_err("stdio transport needs an executable as the positional argument")
            })?;
            if !header.is_empty() {
                return Err(config_err("`--header` is HTTP-only"));
            }
            if auth_token.is_some() || auth.is_some() || auth_flags_present {
                return Err(config_err("auth flags are HTTP-only"));
            }

            let env = parse_kv_pairs("--env", &env)?;

            Ok(ResolvedAddArgs {
                name,
                transport,
                command: Some(command),
                stdio_args: tail,
                env,
                url: None,
                headers: Vec::new(),
                auth_token: None,
                client_secret: None,
                auth: None,
                permission,
                allowed_tools,
                disabled_tools,
                eager_load_tools,
                tool_permissions,
                no_login,
                disabled,
                required,
            })
        }
        McpTransport::Http => {
            let url = location.ok_or_else(|| {
                config_err("http transport needs a URL as the positional argument")
            })?;
            if !tail.is_empty() {
                return Err(config_err("http transport takes no trailing arguments"));
            }
            if !env.is_empty() {
                return Err(config_err("`--env` is stdio-only"));
            }

            let headers = parse_kv_pairs("--header", &header)?;

            let auth_config = resolve_auth_config(
                auth,
                &auth_token,
                client_id,
                &client_secret,
                signing_key,
                signing_algorithm,
                scope,
                redirect_port,
            )?;

            Ok(ResolvedAddArgs {
                name,
                transport,
                command: None,
                stdio_args: Vec::new(),
                env: Vec::new(),
                url: Some(url),
                headers,
                auth_token,
                client_secret,
                auth: auth_config,
                permission,
                allowed_tools,
                disabled_tools,
                eager_load_tools,
                tool_permissions,
                no_login,
                disabled,
                required,
            })
        }
    }
}

/// Convert the CLI's auth-related flags into an [`McpAuthConfig`] (or `None` if the user chose
/// static-token / no auth). Validates the per-variant required fields so "oauth" doesn't silently
/// accept an unrelated `--signing-key` and ship a malformed config.
///
/// The two secrets are read but never returned: an [`McpAuthConfig`] is written to `config.toml`,
/// which is not where a secret goes. They are checked here because this is where the per-variant
/// rules live, and carried to the store by [`run_add`].
#[allow(
    clippy::too_many_arguments,
    reason = "the eight inputs are the independent auth-related CLI flags; a newtype would only move the destructuring one step upstream"
)]
fn resolve_auth_config(
    auth: Option<crate::cli::McpAuthKind>,
    auth_token: &Option<String>,
    client_id: Option<String>,
    client_secret: &Option<String>,
    signing_key: Option<String>,
    signing_algorithm: Option<String>,
    scope: Vec<String>,
    redirect_port: Option<u16>,
) -> Result<Option<McpAuthConfig>> {
    use crate::cli::McpAuthKind;

    let auth_flags_present = client_id.is_some()
        || client_secret.is_some()
        || signing_key.is_some()
        || signing_algorithm.is_some()
        || !scope.is_empty()
        || redirect_port.is_some();

    match auth {
        None => {
            if auth_flags_present && auth_token.is_none() {
                return Err(config_err("OAuth-family flags need `--auth`"));
            }
            Ok(None)
        }
        Some(McpAuthKind::OAuth) => {
            if signing_key.is_some() || signing_algorithm.is_some() {
                return Err(config_err(
                    "`--signing-key` and `--signing-algorithm` need `--auth client_credentials_jwt`",
                ));
            }
            Ok(Some(McpAuthConfig::OAuth {
                client_id,
                scopes: if scope.is_empty() { None } else { Some(scope) },
                redirect_port,
            }))
        }
        Some(McpAuthKind::ClientCredentials) => {
            let client_id = client_id
                .ok_or_else(|| config_err("`--auth client_credentials` needs `--client-id`"))?;
            if client_secret.is_none() {
                return Err(config_err(
                    "`--auth client_credentials` needs `--client-secret-stdin`",
                ));
            }
            if signing_key.is_some() || signing_algorithm.is_some() {
                return Err(config_err(
                    "`--signing-key` and `--signing-algorithm` need `--auth client_credentials_jwt`",
                ));
            }
            if redirect_port.is_some() {
                return Err(config_err("`--redirect-port` needs `--auth oauth`"));
            }
            Ok(Some(McpAuthConfig::ClientCredentials {
                client_id,
                scopes: if scope.is_empty() { None } else { Some(scope) },
                resource: None,
            }))
        }
        Some(McpAuthKind::ClientCredentialsJwt) => {
            let client_id = client_id
                .ok_or_else(|| config_err("`--auth client_credentials_jwt` needs `--client-id`"))?;
            let signing_key_path = signing_key.ok_or_else(|| {
                config_err("`--auth client_credentials_jwt` needs `--signing-key`")
            })?;
            if client_secret.is_some() {
                return Err(config_err(
                    "`--auth client_credentials_jwt` signs a JWT and takes no `--client-secret-stdin`",
                ));
            }
            if redirect_port.is_some() {
                return Err(config_err("`--redirect-port` needs `--auth oauth`"));
            }
            Ok(Some(McpAuthConfig::ClientCredentialsJwt {
                client_id,
                signing_key_path,
                signing_algorithm,
                scopes: if scope.is_empty() { None } else { Some(scope) },
                resource: None,
            }))
        }
    }
}

/// Parse a list of `KEY=VALUE` strings from the CLI into pairs. Surfaces the originating flag name
/// in errors so users can tell `--env` and `--header` apart when both are wrong at once.
fn parse_kv_pairs(flag: &str, pairs: &[String]) -> Result<Vec<(String, String)>> {
    let mut out = Vec::with_capacity(pairs.len());
    for entry in pairs {
        let (k, v) = entry
            .split_once('=')
            .ok_or_else(|| config_err(format!("`{flag}` takes KEY=VALUE, got '{entry}'")))?;
        if k.is_empty() {
            return Err(config_err(format!(
                "`{flag}` entry '{entry}' has an empty key"
            )));
        }
        out.push((k.to_string(), v.to_string()));
    }
    Ok(out)
}

/// An [`McpServerConfig`] equal to what parsing the entry just written to `config.toml` would
/// yield, so the auto-login path in [`run_add`] can call [`run_login`] without a round trip
/// through disk.
fn resolved_to_server_config(resolved: &ResolvedAddArgs) -> McpServerConfig {
    let env = if resolved.env.is_empty() {
        None
    } else {
        Some(resolved.env.iter().cloned().collect())
    };
    let headers = if resolved.headers.is_empty() {
        None
    } else {
        Some(resolved.headers.iter().cloned().collect())
    };
    let args = if resolved.stdio_args.is_empty() {
        None
    } else {
        Some(resolved.stdio_args.clone())
    };
    McpServerConfig {
        name: resolved.name.clone(),
        transport: resolved.transport,
        command: resolved.command.clone(),
        args,
        env,
        url: resolved.url.clone(),
        headers,
        headers_helper: None,
        auth: resolved.auth.clone(),
        permission: resolved.permission,
        allowed_tools: resolved.allowed_tools.clone(),
        disabled_tools: resolved.disabled_tools.clone(),
        eager_load_tools: resolved.eager_load_tools.clone(),
        tool_permissions: resolved.tool_permissions.clone(),
        // No `meka mcp add` flag drives this: declining a server's `readOnlyHint` is a standing
        // policy decision about a server you already distrust, not something to pick while adding
        // one. `None` means the default (trust it), and the knob is edited in `config.toml`.
        trust_read_only_hint: None,
        disabled: resolved.disabled.then_some(true),
        required: resolved.required.then_some(true),
    }
}

/// Serialize a validated [`ResolvedAddArgs`] into a TOML table ready to push onto `mcp.servers`.
/// Only the fields the user actually supplied are emitted so hand-edited config files stay
/// readable.
///
/// `auth_token` and `client_secret` are deliberately absent: they are secrets, so they go to the
/// store, and this function cannot write them because it is handed no way to.
fn build_server_table(resolved: &ResolvedAddArgs) -> toml_edit::Table {
    let mut table = toml_edit::Table::new();
    table.insert("name", toml_edit::value(resolved.name.clone()));
    table.insert(
        "transport",
        toml_edit::value(match resolved.transport {
            McpTransport::Stdio => "stdio",
            McpTransport::Http => "http",
        }),
    );

    if let Some(command) = &resolved.command {
        table.insert("command", toml_edit::value(command.clone()));
    }
    if !resolved.stdio_args.is_empty() {
        let mut arr = toml_edit::Array::new();
        for arg in &resolved.stdio_args {
            arr.push(arg.as_str());
        }
        table.insert("args", toml_edit::Item::Value(toml_edit::Value::Array(arr)));
    }
    if !resolved.env.is_empty() {
        let mut env_table = toml_edit::InlineTable::new();
        for (k, v) in &resolved.env {
            env_table.insert(k, toml_edit::Value::from(v.as_str()));
        }
        table.insert(
            "env",
            toml_edit::Item::Value(toml_edit::Value::InlineTable(env_table)),
        );
    }
    if let Some(url) = &resolved.url {
        table.insert("url", toml_edit::value(url.clone()));
    }
    if !resolved.headers.is_empty() {
        let mut h = toml_edit::InlineTable::new();
        for (k, v) in &resolved.headers {
            h.insert(k, toml_edit::Value::from(v.as_str()));
        }
        table.insert(
            "headers",
            toml_edit::Item::Value(toml_edit::Value::InlineTable(h)),
        );
    }
    if let Some(permission) = &resolved.permission {
        table.insert("permission", toml_edit::value(permission.name()));
    }
    if resolved.disabled {
        table.insert("disabled", toml_edit::value(true));
    }
    if resolved.required {
        table.insert("required", toml_edit::value(true));
    }
    if let Some(auth) = &resolved.auth {
        table.insert("auth", toml_edit::Item::Table(auth_to_toml(auth)));
    }
    if let Some(allowed) = resolved.allowed_tools.as_deref() {
        let mut arr = toml_edit::Array::new();
        for name in allowed {
            arr.push(name.as_str());
        }
        table.insert(
            "allowed_tools",
            toml_edit::Item::Value(toml_edit::Value::Array(arr)),
        );
    }
    if let Some(disabled) = resolved.disabled_tools.as_deref() {
        let mut arr = toml_edit::Array::new();
        for name in disabled {
            arr.push(name.as_str());
        }
        table.insert(
            "disabled_tools",
            toml_edit::Item::Value(toml_edit::Value::Array(arr)),
        );
    }
    if let Some(eager) = resolved.eager_load_tools.as_deref() {
        let mut arr = toml_edit::Array::new();
        for name in eager {
            arr.push(name.as_str());
        }
        table.insert(
            "eager_load_tools",
            toml_edit::Item::Value(toml_edit::Value::Array(arr)),
        );
    }
    if let Some(permissions) = resolved.tool_permissions.as_ref()
        && !permissions.is_empty()
    {
        let mut tpt = toml_edit::Table::new();
        // Stable key order so the TOML diff is review-friendly.
        let mut keys: Vec<&String> = permissions.keys().collect();
        keys.sort();
        for key in keys {
            tpt.insert(key, toml_edit::value(permissions[key].name()));
        }
        table.insert("tool_permissions", toml_edit::Item::Table(tpt));
    }
    table
}

fn auth_to_toml(auth: &McpAuthConfig) -> toml_edit::Table {
    let mut t = toml_edit::Table::new();
    match auth {
        McpAuthConfig::OAuth {
            client_id,
            scopes,
            redirect_port,
        } => {
            t.insert("type", toml_edit::value("oauth"));
            if let Some(id) = client_id {
                t.insert("client_id", toml_edit::value(id.clone()));
            }
            insert_string_array(&mut t, "scopes", scopes.as_deref());
            if let Some(port) = redirect_port {
                t.insert("redirect_port", toml_edit::value(*port as i64));
            }
        }
        McpAuthConfig::ClientCredentials {
            client_id,
            scopes,
            resource,
        } => {
            t.insert("type", toml_edit::value("client_credentials"));
            t.insert("client_id", toml_edit::value(client_id.clone()));
            insert_string_array(&mut t, "scopes", scopes.as_deref());
            if let Some(resource) = resource {
                t.insert("resource", toml_edit::value(resource.clone()));
            }
        }
        McpAuthConfig::ClientCredentialsJwt {
            client_id,
            signing_key_path,
            signing_algorithm,
            scopes,
            resource,
        } => {
            t.insert("type", toml_edit::value("client_credentials_jwt"));
            t.insert("client_id", toml_edit::value(client_id.clone()));
            t.insert(
                "signing_key_path",
                toml_edit::value(signing_key_path.clone()),
            );
            if let Some(alg) = signing_algorithm {
                t.insert("signing_algorithm", toml_edit::value(alg.clone()));
            }
            insert_string_array(&mut t, "scopes", scopes.as_deref());
            if let Some(resource) = resource {
                t.insert("resource", toml_edit::value(resource.clone()));
            }
        }
    }
    t
}

fn insert_string_array(table: &mut toml_edit::Table, key: &str, values: Option<&[String]>) {
    let Some(values) = values else {
        return;
    };
    if values.is_empty() {
        return;
    }
    let mut arr = toml_edit::Array::new();
    for v in values {
        arr.push(v.as_str());
    }
    table.insert(key, toml_edit::Item::Value(toml_edit::Value::Array(arr)));
}

/// What [`purge_server`] found to remove.
enum Purged {
    /// The `[[mcp.servers]]` entry was removed, along with any local state.
    Server(std::path::PathBuf),
    /// There was no config entry, but stored credentials were cleared: the leftover of a server
    /// deleted from `config.toml` by hand.
    CredentialsOnly,
}

/// Wipe every trace of `name`: the `[[mcp.servers]]` entry in `config.toml` and any stored
/// credentials (the OAuth bundle revoked server-side via RFC 7009 first, best-effort). Silent:
/// callers print their own user-facing line. Used by both `run_remove` (user-invoked) and
/// `run_add`'s auto-login rollback path (on OAuth failure after the config entry has already been
/// written).
///
/// A missing config entry is only an error when there is nothing else to remove either. This is the
/// one command that deletes an MCP credential as part of deleting a server, so refusing on a name
/// that is absent from `config.toml` would strand the OAuth bundle of a server whose entry was
/// deleted by hand: `mcp list` reports it, and nothing else would clear it.
async fn purge_server(name: &str, token_store: &TokenStore) -> Result<Purged> {
    // Held to the end of the function, so this read and the write below cannot interleave with
    // another editor's, `device_id::persist` on an ordinary launch included.
    let _config_lock = crate::config::lock_config_file()
        .map_err(|error| config_err(format!("failed to lock config: {error}")))?;
    let path = crate::paths::config_file_path()
        .ok_or_else(|| config_err("failed to determine the config directory"))?;
    let existing = std::fs::read_to_string(&path)
        .map_err(|error| config_err(format!("failed to read config: {error}")))?;
    let mut document = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| config_err(format!("failed to parse config: {error}")))?;

    let removed_from_config = document
        .get_mut("mcp")
        .and_then(|m| m.as_table_mut())
        .and_then(|t| t.get_mut("servers"))
        .and_then(|s| s.as_array_of_tables_mut())
        .is_some_and(|servers| {
            let original_len = servers.len();
            servers.retain(|entry| entry.get("name").and_then(|v| v.as_str()) != Some(name));
            servers.len() != original_len
        });

    if !removed_from_config {
        if !token_store.has_mcp_credentials(name).await? {
            return Err(config_err(crate::text::unknown_name(
                "MCP server",
                name,
                server_names(&document),
            )));
        }
        clear_server_state(name, token_store).await?;
        return Ok(Purged::CredentialsOnly);
    }

    crate::fs::write_file_atomic(&path, &document.to_string())
        .map_err(|error| config_err(format!("failed to write config: {error}")))?;

    clear_server_state(name, token_store).await?;
    Ok(Purged::Server(path))
}

/// The server names `config.toml` carries, off the raw document, for a refusal on a file the typed
/// config may not parse.
fn server_names(document: &toml_edit::DocumentMut) -> Vec<String> {
    document
        .get("mcp")
        .and_then(|mcp| mcp.get("servers"))
        .and_then(|servers| servers.as_array_of_tables())
        .map(|servers| {
            servers
                .iter()
                .filter_map(|entry| entry.get("name").and_then(|name| name.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Drop everything meka stores about `name` outside `config.toml`: every credential kind, the
/// OAuth bundle revoked server-side via RFC 7009 first, best-effort. Shared by both of
/// [`purge_server`]'s exits so a leftover credential is cleaned as thoroughly as one attached to a
/// real server.
async fn clear_server_state(name: &str, token_store: &TokenStore) -> Result<()> {
    // If the caller is rolling back a never-succeeded login there won't be any credentials;
    // `revoke_stored_token` early-returns in that case so the call is safe regardless.
    if let Err(error) = crate::mcp::auth::revoke_stored_token(token_store, name).await {
        tracing::warn!(
            "failed to revoke token at server '{name}' during purge: {error} (continuing)"
        );
    }

    token_store.clear_mcp_credentials(name).await?;
    Ok(())
}

/// Run `meka mcp remove <name>`: delete the entry from config.toml, best-effort revoke OAuth
/// tokens at the provider, clear local state.
pub(crate) async fn run_remove(name: &str, token_store: &TokenStore) -> Result<()> {
    match purge_server(name, token_store).await? {
        Purged::Server(path) => {
            tracing::info!("removed '{name}' from {path}", path = path.display())
        }
        Purged::CredentialsOnly => {
            tracing::info!("cleared stored credentials for '{name}'; no server was configured")
        }
    }
    Ok(())
}

/// Set `disabled = <value>` on a server entry in config.toml, preserving other fields and
/// formatting. Backs `meka mcp disable|enable`. Writes atomically via
/// [`crate::fs::write_file_atomic`].
fn set_server_disabled(name: &str, disabled: bool) -> Result<std::path::PathBuf> {
    // Held to the end of the function, so this read and the write below cannot interleave with
    // another editor's, `device_id::persist` on an ordinary launch included.
    let _config_lock = crate::config::lock_config_file()
        .map_err(|error| config_err(format!("failed to lock config: {error}")))?;
    let path = crate::paths::config_file_path()
        .ok_or_else(|| config_err("failed to determine the config directory"))?;
    let existing = std::fs::read_to_string(&path)
        .map_err(|error| config_err(format!("failed to read config: {error}")))?;
    let mut document = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| config_err(format!("failed to parse config: {error}")))?;

    let known = server_names(&document);
    let unknown = || config_err(crate::text::unknown_name("MCP server", name, known.iter()));
    let servers = document
        .get_mut("mcp")
        .and_then(|m| m.as_table_mut())
        .and_then(|t| t.get_mut("servers"))
        .and_then(|s| s.as_array_of_tables_mut())
        .ok_or_else(unknown)?;

    let entry = servers
        .iter_mut()
        .find(|entry| entry.get("name").and_then(|v| v.as_str()) == Some(name))
        .ok_or_else(unknown)?;

    if disabled {
        entry.insert("disabled", toml_edit::value(true));
    } else {
        // Remove the key entirely rather than setting `false`. This keeps minimal diffs for users
        // who never enabled the flag before.
        entry.remove("disabled");
    }

    crate::fs::write_file_atomic(&path, &document.to_string())
        .map_err(|error| config_err(format!("failed to write config: {error}")))?;
    Ok(path)
}

/// Run `meka mcp disable <name>`. Sets `disabled = true` in config.toml. The currently-running meka
/// session (if any) keeps its state; the change takes effect on the next start.
pub(crate) fn run_disable(name: &str) -> Result<()> {
    let path = set_server_disabled(name, true)?;
    tracing::info!("disabled '{name}' in {path}", path = path.display());
    Ok(())
}

/// Run `meka mcp enable <name>`. Clears `disabled` from the server entry in config.toml.
pub(crate) fn run_enable(name: &str) -> Result<()> {
    let path = set_server_disabled(name, false)?;
    tracing::info!("enabled '{name}' in {path}", path = path.display());
    Ok(())
}

pub(crate) async fn run_mcp_subcommand(
    store: &Store,
    action: &crate::cli::McpAction,
    cli_args: &crate::cli::Cli,
) -> anyhow::Result<()> {
    let config = ResolvedConfig::resolve(cli_args.overrides());
    // `validate()` never runs here, so an unparseable config has to be handled per action. The four
    // that edit `config.toml` through `toml_edit` never read `config.mcp_servers` and are how the
    // file gets repaired, so they run on a broken one; the rest would answer out of an empty server
    // list and state it as fact ("No MCP servers.", "no MCP server named 'x'").
    if matches!(
        action,
        crate::cli::McpAction::Add { .. }
            | crate::cli::McpAction::Remove { .. }
            | crate::cli::McpAction::Enable { .. }
            | crate::cli::McpAction::Disable { .. }
    ) {
        config.warn_if_config_unreadable();
    } else {
        config.require_readable_config()?;
    }
    let token_store = store.token_store();
    match action {
        crate::cli::McpAction::List { format } => {
            crate::cli::mcp::list_servers(&config.mcp_servers, None, &token_store, *format).await?
        }
        crate::cli::McpAction::Get { name, format } => {
            crate::cli::mcp::run_get(&config.mcp_servers, name, &token_store, *format).await?
        }
        crate::cli::McpAction::Reconnect { name } => {
            crate::cli::mcp::run_reconnect(&config.mcp_servers, &token_store, name).await?
        }
        crate::cli::McpAction::Tools { name, format } => {
            crate::cli::mcp::run_tools(
                &config.mcp_servers,
                config.mcp_default_permission,
                &token_store,
                name,
                *format,
            )
            .await?
        }
        crate::cli::McpAction::Login {
            name,
            auth_token_stdin,
            client_secret_stdin,
        } => {
            use crate::store::McpCredentialKind;

            // Clap refuses both flags at once, so at most one of these reads stdin.
            let stored = match (
                crate::cli::read_secret_from_stdin(*auth_token_stdin, "auth token")?,
                crate::cli::read_secret_from_stdin(*client_secret_stdin, "client secret")?,
            ) {
                (Some(token), _) => Some((McpCredentialKind::Bearer, token)),
                (None, Some(secret)) => Some((McpCredentialKind::ClientSecret, secret)),
                (None, None) => None,
            };

            match stored {
                Some((kind, secret)) => {
                    crate::cli::mcp::run_store_secret(
                        &config.mcp_servers,
                        &token_store,
                        name,
                        kind,
                        &secret,
                    )
                    .await?
                }
                None => crate::cli::mcp::run_login(&config.mcp_servers, &token_store, name).await?,
            }
        }
        crate::cli::McpAction::Logout { name } => {
            crate::cli::mcp::run_logout(&config.mcp_servers, &token_store, name).await?
        }
        crate::cli::McpAction::Add {
            name,
            location,
            args,
            transport,
            env,
            header,
            auth,
            auth_token_stdin,
            client_id,
            client_secret_stdin,
            signing_key,
            signing_algorithm,
            scope,
            redirect_port,
            permission,
            no_login,
            allow_tool,
            disable_tool,
            eager_load_tool,
            tool_permission,
            disabled,
            required,
        } => {
            crate::cli::mcp::run_add(
                crate::cli::mcp::AddArgs {
                    name: name.clone(),
                    location: location.clone(),
                    args: args.clone(),
                    transport: *transport,
                    env: env.clone(),
                    header: header.clone(),
                    auth: *auth,
                    // Read here rather than in `run_add` because stdin is a process-wide resource
                    // and this is the layer that owns it. Clap has already refused both flags at
                    // once, so at most one of these two reads the stream.
                    auth_token: crate::cli::read_secret_from_stdin(
                        *auth_token_stdin,
                        "auth token",
                    )?,
                    client_id: client_id.clone(),
                    client_secret: crate::cli::read_secret_from_stdin(
                        *client_secret_stdin,
                        "client secret",
                    )?,
                    signing_key: signing_key.clone(),
                    signing_algorithm: signing_algorithm.clone(),
                    scope: scope.clone(),
                    redirect_port: *redirect_port,
                    permission: permission.clone(),
                    no_login: *no_login,
                    allow_tool: allow_tool.clone(),
                    disable_tool: disable_tool.clone(),
                    eager_load_tool: eager_load_tool.clone(),
                    tool_permission: tool_permission.clone(),
                    disabled: *disabled,
                    required: *required,
                },
                &token_store,
            )
            .await?
        }
        crate::cli::McpAction::Remove { name } => {
            crate::cli::mcp::run_remove(name, &token_store).await?
        }
        crate::cli::McpAction::Disable { name } => crate::cli::mcp::run_disable(name)?,
        crate::cli::McpAction::Enable { name } => crate::cli::mcp::run_enable(name)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare_add(name: &str, location: Option<&str>) -> AddArgs {
        AddArgs {
            name: name.to_string(),
            location: location.map(str::to_string),
            args: Vec::new(),
            transport: None,
            env: Vec::new(),
            header: Vec::new(),
            auth: None,
            auth_token: None,
            client_id: None,
            client_secret: None,
            signing_key: None,
            signing_algorithm: None,
            scope: Vec::new(),
            redirect_port: None,
            permission: None,
            no_login: false,
            allow_tool: Vec::new(),
            disable_tool: Vec::new(),
            eager_load_tool: Vec::new(),
            tool_permission: Vec::new(),
            disabled: false,
            required: false,
        }
    }

    #[test]
    fn resolve_autodetects_http_from_url() {
        let resolved = resolve_add_args(bare_add("notion", Some("https://mcp.notion.com/mcp")))
            .expect("should resolve");
        assert_eq!(resolved.transport, McpTransport::Http);
        assert_eq!(resolved.url.as_deref(), Some("https://mcp.notion.com/mcp"));
        assert!(resolved.command.is_none());
        assert!(resolved.auth.is_none());
    }

    #[test]
    fn resolve_autodetects_stdio_from_command() {
        let mut args = bare_add("pg", Some("npx"));
        args.args = vec![
            "-y".to_string(),
            "@modelcontextprotocol/server-postgres".to_string(),
        ];
        let resolved = resolve_add_args(args).expect("should resolve");
        assert_eq!(resolved.transport, McpTransport::Stdio);
        assert_eq!(resolved.command.as_deref(), Some("npx"));
        assert_eq!(resolved.stdio_args.len(), 2);
        assert!(resolved.url.is_none());
    }

    #[test]
    fn resolve_http_requires_location() {
        let err = resolve_add_args({
            let mut args = bare_add("srv", None);
            args.transport = Some(McpTransport::Http);
            args
        })
        .expect_err("http with no URL should error");
        assert!(format!("{err}").contains("http transport needs a URL"));
    }

    #[test]
    fn resolve_stdio_requires_command() {
        let err = resolve_add_args({
            let mut args = bare_add("srv", None);
            args.transport = Some(McpTransport::Stdio);
            args
        })
        .expect_err("stdio with no command should error");
        assert!(format!("{err}").contains("stdio transport needs an executable"));
    }

    #[test]
    fn resolve_http_rejects_trailing_args() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.args = vec!["extra".to_string()];
        let err = resolve_add_args(args).expect_err("should reject trailing args on http");
        assert!(format!("{err}").contains("trailing arguments"));
    }

    #[test]
    fn resolve_rejects_env_on_http() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.env = vec!["K=V".to_string()];
        let err = resolve_add_args(args).expect_err("env on http should error");
        assert!(format!("{err}").contains("`--env` is stdio-only"));
    }

    #[test]
    fn resolve_rejects_header_on_stdio() {
        let mut args = bare_add("srv", Some("/usr/bin/mcp"));
        args.header = vec!["X-Custom=1".to_string()];
        let err = resolve_add_args(args).expect_err("header on stdio should error");
        assert!(format!("{err}").contains("`--header` is HTTP-only"));
    }

    #[test]
    fn resolve_rejects_auth_token_with_auth_flag() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth_token = Some("tok".to_string());
        args.auth = Some(crate::cli::McpAuthKind::OAuth);
        let err = resolve_add_args(args).expect_err("mutually exclusive flags");
        assert!(format!("{err}").contains("cannot be combined"));
    }

    #[test]
    fn resolve_client_credentials_requires_id_and_secret() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth = Some(crate::cli::McpAuthKind::ClientCredentials);
        let err = resolve_add_args(args).expect_err("missing client-id");
        assert!(format!("{err}").contains("--client-id"));
    }

    #[test]
    fn resolve_oauth_builds_empty_config_when_no_flags() {
        let mut args = bare_add("notion", Some("https://mcp.notion.com/mcp"));
        args.auth = Some(crate::cli::McpAuthKind::OAuth);
        let resolved = resolve_add_args(args).expect("should resolve");
        match resolved.auth {
            Some(McpAuthConfig::OAuth {
                client_id,
                scopes,
                redirect_port,
            }) => {
                assert!(client_id.is_none());
                assert!(scopes.is_none());
                assert!(redirect_port.is_none());
            }
            other => panic!("expected OAuth auth, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn resolve_rejects_oauth_flags_without_auth() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.client_id = Some("id".to_string());
        let err = resolve_add_args(args).expect_err("orphan oauth flag");
        assert!(format!("{err}").contains("OAuth-family flags"));
    }

    #[test]
    fn resolve_auth_token_works_alone() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth_token = Some("bearer-xyz".to_string());
        let resolved = resolve_add_args(args).expect("should resolve");
        assert_eq!(resolved.auth_token.as_deref(), Some("bearer-xyz"));
        assert!(resolved.auth.is_none());
    }

    /// Neither secret reaches `config.toml`. `run_add` carries them to the store instead, and this
    /// is the function that would put them in the file if anything did.
    #[test]
    fn the_written_table_never_carries_a_secret() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth_token = Some("bearer-not-a-real-token".to_string());
        let resolved = resolve_add_args(args).expect("should resolve");
        let rendered = build_server_table(&resolved).to_string();
        assert!(
            !rendered.contains("bearer-not-a-real-token"),
            "the bearer must not be written to config.toml, got:\n{rendered}"
        );

        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth = Some(crate::cli::McpAuthKind::OAuth);
        args.client_secret = Some("cs-not-a-real-secret".to_string());
        let resolved = resolve_add_args(args).expect("should resolve");
        let rendered = build_server_table(&resolved).to_string();
        assert!(
            !rendered.contains("cs-not-a-real-secret"),
            "the client secret must not be written to config.toml, got:\n{rendered}"
        );
        assert_eq!(
            resolved.client_secret.as_deref(),
            Some("cs-not-a-real-secret"),
            "it must still reach run_add, or it would be silently dropped"
        );
    }

    /// `--auth client_credentials` cannot authenticate without one, so `add` says so at add time
    /// rather than letting the first connect fail.
    #[test]
    fn client_credentials_without_a_secret_is_refused() {
        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth = Some(crate::cli::McpAuthKind::ClientCredentials);
        args.client_id = Some("id".to_string());
        let error = resolve_add_args(args).expect_err("no secret should be refused");
        assert!(
            format!("{error}").contains("--client-secret-stdin"),
            "the error should name the flag that supplies it, got: {error}"
        );
    }

    #[test]
    fn parse_kv_pairs_rejects_missing_separator() {
        let err = parse_kv_pairs("--env", &["bad".to_string()]).expect_err("no = should error");
        assert!(format!("{err}").contains("--env"));
        assert!(format!("{err}").contains("KEY=VALUE"));
    }

    #[test]
    fn parse_kv_pairs_rejects_empty_key() {
        let err = parse_kv_pairs("--header", &["=value".to_string()])
            .expect_err("empty key should error");
        assert!(format!("{err}").contains("empty key"));
    }

    #[test]
    fn build_server_table_emits_oauth_block() {
        let mut args = bare_add("notion", Some("https://mcp.notion.com/mcp"));
        args.auth = Some(crate::cli::McpAuthKind::OAuth);
        args.scope = vec!["read".to_string(), "write".to_string()];
        args.redirect_port = Some(8400);
        let resolved = resolve_add_args(args).expect("resolve");
        // Parse it back through toml to confirm the schema matches what ResolvedConfig expects.
        // Checking the textual rendering is fragile because toml_edit decides when to emit a
        // standalone `[mcp.servers.auth]` header.
        let mut doc = toml_edit::DocumentMut::new();
        let mut servers = toml_edit::ArrayOfTables::new();
        servers.push(build_server_table(&resolved));
        doc.insert(
            "mcp",
            toml_edit::Item::Table({
                let mut t = toml_edit::Table::new();
                t.insert("servers", toml_edit::Item::ArrayOfTables(servers));
                t
            }),
        );
        let parsed: crate::config::ConfigFile =
            toml::from_str(&doc.to_string()).expect("valid config");
        let servers = parsed.mcp.expect("mcp").servers.expect("servers");
        assert_eq!(servers.len(), 1);
        let server = &servers[0];
        assert_eq!(server.name, "notion");
        assert!(matches!(
            &server.auth,
            Some(crate::config::McpAuthConfig::OAuth { redirect_port: Some(8400), scopes: Some(s), .. })
            if s == &vec!["read".to_string(), "write".to_string()]
        ));
    }

    /// `--required` is written by exactly one `if` in `build_server_table`, and nothing else
    /// asserts it. Without this, dropping that branch leaves `meka mcp add --required` silently
    /// producing an optional server with the whole suite still green.
    #[test]
    fn build_server_table_persists_required_only_when_set() {
        let round_trip = |required: bool| {
            let mut args = bare_add("bridge", Some("http://127.0.0.1:9100/mcp"));
            args.required = required;
            let resolved = resolve_add_args(args).expect("resolve");
            let mut doc = toml_edit::DocumentMut::new();
            let mut servers = toml_edit::ArrayOfTables::new();
            servers.push(build_server_table(&resolved));
            doc.insert(
                "mcp",
                toml_edit::Item::Table({
                    let mut table = toml_edit::Table::new();
                    table.insert("servers", toml_edit::Item::ArrayOfTables(servers));
                    table
                }),
            );
            let parsed: crate::config::ConfigFile =
                toml::from_str(&doc.to_string()).expect("valid config");
            parsed.mcp.expect("mcp").servers.expect("servers")[0].required
        };
        assert_eq!(round_trip(true), Some(true));
        // Left out entirely when false, so the server keeps inheriting `[mcp].default_required`.
        assert_eq!(round_trip(false), None);
    }

    #[test]
    fn describe_one_line_collapses_whitespace() {
        assert_eq!(describe_one_line("one\n\ntwo  three"), "one two three");
    }

    #[test]
    fn describe_one_line_short_input_passes_through() {
        assert_eq!(describe_one_line("Read a file."), "Read a file.");
    }

    #[test]
    fn describe_one_line_caps_at_80_chars_with_ellipsis() {
        let long = "a".repeat(200);
        let out = describe_one_line(&long);
        assert!(out.ends_with('…'));
        assert!(out.chars().count() <= 81);
    }

    #[test]
    fn describe_one_line_empty_passes_through() {
        assert_eq!(describe_one_line(""), "");
    }

    fn server_named(name: &str) -> McpServerConfig {
        let resolved = resolve_add_args(bare_add(name, Some("https://example.test/mcp")))
            .expect("resolve add args");
        resolved_to_server_config(&resolved)
    }

    async fn memory_token_store() -> TokenStore {
        crate::store::Store::for_test().await.token_store()
    }

    /// The step between `add` deciding what the secrets are and the store holding them. If it
    /// writes nothing, `mcp add --auth-token-stdin` reports success and the server is
    /// unauthenticated, with the secret gone from stdin and from anywhere else.
    #[tokio::test]
    async fn add_carries_both_secrets_to_the_store_under_their_own_kinds() {
        use crate::store::McpCredentialKind;

        let store = memory_token_store().await;
        let mut args = bare_add("api", Some("https://example.test/mcp"));
        args.auth = Some(crate::cli::McpAuthKind::OAuth);
        args.client_secret = Some("cs-not-a-real-secret".to_string());
        let resolved = resolve_add_args(args).expect("resolve");
        store_add_secrets(&resolved, &store).await.expect("store");

        assert_eq!(
            store
                .load_mcp_credentials("api", McpCredentialKind::ClientSecret)
                .await
                .expect("load")
                .as_deref(),
            Some("cs-not-a-real-secret")
        );

        let mut args = bare_add("bear", Some("https://example.test/mcp"));
        args.auth_token = Some("bearer-not-a-real-token".to_string());
        let resolved = resolve_add_args(args).expect("resolve");
        store_add_secrets(&resolved, &store).await.expect("store");

        assert_eq!(
            store
                .load_mcp_credentials("bear", McpCredentialKind::Bearer)
                .await
                .expect("load")
                .as_deref(),
            Some("bearer-not-a-real-token")
        );
        assert!(
            store
                .load_mcp_credentials("bear", McpCredentialKind::ClientSecret)
                .await
                .expect("load")
                .is_none(),
            "a bearer must not be filed as a client secret"
        );
    }

    /// `meka mcp login --auth-token-stdin` is the second door onto the state `mcp add` refuses, so
    /// it applies the same rules. A bearer stored on a server with an `[auth]` block would never be
    /// sent: the flow's own `Authorization` header wins, and the user would be left with a secret
    /// in the store and a server that still says it is unauthorized.
    #[tokio::test]
    async fn a_bearer_is_refused_for_a_server_that_authenticates_by_flow() {
        use crate::store::McpCredentialKind;

        let store = memory_token_store().await;
        let mut server = server_named("api");
        server.auth = Some(McpAuthConfig::OAuth {
            client_id: None,
            scopes: None,
            redirect_port: None,
        });

        let error = run_store_secret(
            std::slice::from_ref(&server),
            &store,
            "api",
            McpCredentialKind::Bearer,
            "bearer-not-a-real-token",
        )
        .await
        .expect_err("a bearer beside an [auth] block should be refused");
        assert!(
            format!("{error}").contains("[auth]"),
            "the error should say what is in the way, got: {error}"
        );
        assert!(
            !store
                .has_mcp_credentials("api")
                .await
                .expect("has credentials"),
            "a refused store must write nothing"
        );
    }

    /// The mirror: nothing presents a client secret when there is no flow to present it.
    #[tokio::test]
    async fn a_client_secret_is_refused_without_an_auth_block() {
        let store = memory_token_store().await;
        let error = run_store_secret(
            std::slice::from_ref(&server_named("api")),
            &store,
            "api",
            crate::store::McpCredentialKind::ClientSecret,
            "cs-not-a-real-secret",
        )
        .await
        .expect_err("a client secret with no [auth] block should be refused");
        assert!(
            format!("{error}").contains("client_credentials"),
            "the error should name what to add, got: {error}"
        );
    }

    /// A JWT client signs an assertion; it has no client secret, and `mcp add` says so too.
    #[tokio::test]
    async fn a_client_secret_is_refused_for_a_jwt_client() {
        let store = memory_token_store().await;
        let mut server = server_named("api");
        server.auth = Some(McpAuthConfig::ClientCredentialsJwt {
            client_id: "id".to_string(),
            signing_key_path: "/key.pem".to_string(),
            signing_algorithm: None,
            scopes: None,
            resource: None,
        });

        let error = run_store_secret(
            std::slice::from_ref(&server),
            &store,
            "api",
            crate::store::McpCredentialKind::ClientSecret,
            "cs-not-a-real-secret",
        )
        .await
        .expect_err("a JWT client has no client secret");
        assert!(
            format!("{error}").contains("signing_key_path"),
            "the error should name what it signs with instead, got: {error}"
        );
    }

    /// `login` is refused for a server with a stored bearer **whichever branch it would take**.
    ///
    /// Both are covered because they fail differently. With no `[auth]` block, login invents one
    /// and persists `type = "oauth"` to config.toml, making the broken pairing permanent. With one
    /// hand-added to config, nothing is invented but the flow still runs and deposits a bundle the
    /// bearer overrides on every request. A guard sited inside the first branch, as this one first
    /// was, leaves the second wide open.
    #[tokio::test]
    async fn login_is_refused_for_a_server_that_has_a_stored_bearer() {
        for auth in [
            None,
            Some(McpAuthConfig::OAuth {
                client_id: None,
                scopes: None,
                redirect_port: None,
            }),
        ] {
            let store = memory_token_store().await;
            store
                .save_mcp_credentials(
                    "api",
                    crate::store::McpCredentialKind::Bearer,
                    "bearer-not-a-real-token",
                )
                .await
                .expect("save");

            let mut server = server_named("api");
            let had_auth = auth.is_some();
            server.auth = auth;

            let error = run_login(std::slice::from_ref(&server), &store, "api")
                .await
                .expect_err("a stored bearer must stop the login");
            assert!(
                format!("{error}").contains("meka mcp logout api"),
                "the error should name the command that drops it (auth block: {had_auth}), got: {error}"
            );
            assert_eq!(
                store
                    .load_mcp_credentials("api", crate::store::McpCredentialKind::Bearer)
                    .await
                    .expect("load")
                    .as_deref(),
                Some("bearer-not-a-real-token"),
                "and it must not have cleared the bearer it refused over (auth block: {had_auth})"
            );
        }
    }

    /// A stdio server has no HTTP request to attach a bearer to. Its secrets are `env`, which is
    /// config, and the error says so rather than storing something nothing will ever read.
    #[tokio::test]
    async fn a_secret_is_refused_for_a_stdio_server() {
        let store = memory_token_store().await;
        let stdio: McpServerConfig =
            toml::from_str("name = \"pg\"\ntransport = \"stdio\"\ncommand = \"pg-mcp\"\n")
                .expect("the fixture parses");

        let error = run_store_secret(
            std::slice::from_ref(&stdio),
            &store,
            "pg",
            crate::store::McpCredentialKind::Bearer,
            "bearer-not-a-real-token",
        )
        .await
        .expect_err("a stdio server has no HTTP credential");
        assert!(
            format!("{error}").contains("env"),
            "the error should point at where a stdio secret goes, got: {error}"
        );
        assert!(
            !store.has_mcp_credentials("pg").await.expect("has"),
            "a refused store must write nothing"
        );
    }

    /// A typo must not strand a secret under a name no server has.
    #[tokio::test]
    async fn a_secret_is_refused_for_a_server_that_is_not_configured() {
        let store = memory_token_store().await;
        let error = run_store_secret(
            std::slice::from_ref(&server_named("api")),
            &store,
            "apo",
            crate::store::McpCredentialKind::Bearer,
            "bearer-not-a-real-token",
        )
        .await
        .expect_err("an unknown server should be refused");
        assert!(format!("{error}").contains("apo"), "got: {error}");
        assert!(
            store
                .list_mcp_credential_servers()
                .await
                .expect("list")
                .is_empty(),
            "nothing should have been written under the typo"
        );
    }

    /// OAuth bundles are keyed by server name and nothing prunes them, so a hand-deleted
    /// `[[mcp.servers]]` entry leaves a live refresh token behind. This diff is the only thing that
    /// can name one.
    #[tokio::test]
    async fn orphaned_credentials_names_bundles_no_server_claims() {
        let store = memory_token_store().await;
        for name in ["linear", "retired"] {
            store
                .save_mcp_credentials(
                    name,
                    crate::store::McpCredentialKind::OAuth,
                    r#"{"tokens":{"access_token":"at"}}"#,
                )
                .await
                .expect("save");
        }

        // `notion` is configured but never logged in to, which is not an orphan: the diff runs in
        // one direction only.
        let servers = vec![server_named("linear"), server_named("notion")];
        let orphans = orphaned_credentials(&servers, &store)
            .await
            .expect("diff credentials against servers");
        assert_eq!(orphans, vec!["retired".to_string()]);
    }

    /// A name whose config entry was deleted by hand keeps its credentials, so the name is free
    /// while the secret is not. Re-adding it must not attach that secret to the new server.
    ///
    /// The leak this stops: `add api https://first…  --auth-token-stdin`, hand-delete the entry,
    /// then `add api https://second…` with no secret at all. The bearer issued for the first host
    /// would be loaded by server name and sent to the second on the first connect, with `mcp get`
    /// reporting a credential the user never gave this server.
    #[tokio::test]
    async fn add_refuses_a_name_whose_credential_outlived_its_config_entry() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "[mcp]\n").expect("write config");
        let store = memory_token_store().await;
        store
            .save_mcp_credentials(
                "api",
                crate::store::McpCredentialKind::Bearer,
                "bearer-not-a-real-token",
            )
            .await
            .expect("save the credential the deleted server left behind");

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set → run → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        // Loopback discard port: `run_add` must refuse before it can probe, so nothing dials out.
        let result = run_add(bare_add("api", Some("http://127.0.0.1:9/mcp")), &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = result.expect_err("a name with a stored credential must be refused");
        let message = format!("{error}");
        assert!(
            message.contains("meka mcp remove api"),
            "the error should name the command that clears it, got: {message}"
        );

        let written = std::fs::read_to_string(dir.path().join("config.toml")).expect("read config");
        assert!(
            !written.contains("127.0.0.1"),
            "the refusal must come before the config write, got:\n{written}"
        );
        assert_eq!(
            store
                .load_mcp_credentials("api", crate::store::McpCredentialKind::Bearer)
                .await
                .expect("load")
                .as_deref(),
            Some("bearer-not-a-real-token"),
            "and it must not have destroyed the credential it refused over"
        );
    }

    /// `logout` clears every kind, and says which ones it cleared.
    ///
    /// The naming matters because the kinds are not equally replaceable: an OAuth bundle is
    /// reobtained by logging in again, while a bearer or a client secret was typed by the user and
    /// meka is now its only holder. A logout that took those away in silence would send them back
    /// to the provider with nothing on screen to explain why.
    #[tokio::test]
    async fn logout_clears_every_kind_and_names_them() {
        use crate::store::McpCredentialKind;

        let store = memory_token_store().await;
        for (kind, secret) in [
            (McpCredentialKind::Bearer, "bearer-not-a-real-token"),
            (McpCredentialKind::ClientSecret, "cs-not-a-real-secret"),
            (McpCredentialKind::OAuth, r#"{"access_token":"at"}"#),
        ] {
            store
                .save_mcp_credentials("api", kind, secret)
                .await
                .expect("seed");
        }

        run_logout(std::slice::from_ref(&server_named("api")), &store, "api")
            .await
            .expect("logout succeeds");

        for kind in [
            McpCredentialKind::Bearer,
            McpCredentialKind::ClientSecret,
            McpCredentialKind::OAuth,
        ] {
            assert!(
                store
                    .load_mcp_credentials("api", kind)
                    .await
                    .expect("load")
                    .is_none(),
                "{} should have been cleared",
                kind.label()
            );
        }
        assert!(
            !store.has_mcp_credentials("api").await.expect("has"),
            "and the server should hold nothing at all"
        );
    }

    /// `remove` is the only command that clears an MCP credential, so requiring a config entry
    /// would leave a hand-deleted server's OAuth bundle unreachable from every surface meka has.
    #[tokio::test]
    async fn remove_clears_credentials_for_a_server_no_longer_in_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "[mcp]\nstrict = true\n")
            .expect("write config");
        let store = memory_token_store().await;
        store
            .save_mcp_credentials(
                "retired",
                crate::store::McpCredentialKind::OAuth,
                r#"{"tokens":{"access_token":"at"}}"#,
            )
            .await
            .expect("save");

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set → run → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("retired", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        result.expect("removing an orphaned credential succeeds");

        assert!(
            store
                .load_mcp_credentials("retired", crate::store::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .is_none(),
            "the stored OAuth bundle must be gone"
        );
    }

    /// The relaxation above must not turn every typo into a silent success: with no config entry
    /// *and* no stored credential there is genuinely nothing to remove.
    #[tokio::test]
    async fn remove_still_refuses_a_name_that_exists_nowhere() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "[mcp]\nstrict = true\n")
            .expect("write config");
        let store = memory_token_store().await;

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("typo", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("removing a name that exists nowhere must fail"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("no MCP server named 'typo'"), "{error}");
    }
}
