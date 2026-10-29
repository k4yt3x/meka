//! `meka mcp …` management subcommands: the flags, the probe on `add`, the listings and the
//! terminal's side of a login, over the doors in [`crate::installation::mcp`].

use std::sync::Arc;

use super::TerminalLogin;
use crate::{
    config::{McpAuthConfig, McpServerConfig, McpTransport, ResolvedConfig},
    error::{MekaError, Result},
    installation::{
        Patch,
        mcp::{
            PROBE_CONNECT_TIMEOUT, ServerChanges, ServerSecrets, add_server, connect_alone,
            edit_server, login_server, logout_server, remove_server, require_server, roll_back,
            server_detail, store_server_secret, usage_err,
        },
    },
    mcp::McpClientContext,
    store::{Store, TokenStore},
    view::{McpServerView, McpToolView, McpToolsResponse},
};

/// `meka mcp list` under a chosen format. The screen's `/mcp` is a picker built on
/// [`crate::mcp::McpClientManager::server_entry`] instead, so this path is the CLI's alone.
/// `--format json` prints `{"servers": [...]}` and leaves the orphan report on stderr.
pub(crate) async fn list_servers(
    servers: &[McpServerConfig],
    manager: Option<&std::sync::Arc<crate::mcp::McpClientManager>>,
    token_store: &TokenStore,
    format: crate::cli::OutputFormat,
    stream: crate::render::Stream,
) -> Result<()> {
    // Computed before the early return below: "every server is gone but the OAuth bundles are still
    // here" is precisely the state worth reporting, and it is the one that return would hide.
    let orphans = crate::installation::mcp::orphaned_credentials(servers, token_store).await?;

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
    // mid-print. The protocol column is a connected server's negotiated version, and empty for
    // every other state.
    let mut states: std::collections::HashMap<&str, (String, String)> =
        std::collections::HashMap::new();
    if let Some(manager) = manager {
        for config in servers {
            let columns = match manager.server_entry(&config.name) {
                Some(entry) => (
                    entry.state().await.label().to_string(),
                    entry
                        .protocol_version()
                        .await
                        .map(|version| version.to_string())
                        .unwrap_or_default(),
                ),
                None => ("(unknown)".to_string(), String::new()),
            };
            states.insert(config.name.as_str(), columns);
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
                let (state_label, protocol_version) = states
                    .get(config.name.as_str())
                    .cloned()
                    .unwrap_or_else(|| ("(unknown)".to_string(), String::new()));
                row.push(state_label);
                row.push(protocol_version);
            }
            row.push(required_label.to_string());
            row.push(perm_label.to_string());
            row.push(target);
            row
        })
        .collect();

    // The name is what every `meka mcp` command takes, so it is shown in full; the target is a URL
    // or a command line shown for information, and `get` has the whole one.
    let mut columns = vec![
        crate::text::Column::content("Name"),
        crate::text::Column::content("Transport"),
    ];
    if with_state {
        columns.push(crate::text::Column::content("State"));
        columns.push(crate::text::Column::content("Protocol"));
    }
    columns.extend([
        crate::text::Column::content("Required"),
        crate::text::Column::content("Permission"),
        crate::text::Column::remainder("Target"),
    ]);
    stream.write(crate::text::format_table(&columns, &rows))?;
    report_orphaned_credentials(&orphans)?;
    Ok(())
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

pub(crate) async fn run_get(
    servers: &[McpServerConfig],
    overlay: &crate::config::McpOverlay,
    name: &str,
    token_store: &TokenStore,
    format: crate::cli::OutputFormat,
) -> Result<()> {
    let config = require_server(servers, name)?;
    let detail = server_detail(config, overlay, token_store).await?;
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
            format!(
                "{}{}",
                if server.required {
                    "yes (gates the turn)"
                } else {
                    "no"
                },
                inherited_note(&detail, "required")
            ),
        ));
    }
    fields.push((
        "deferral",
        format!("{}{}", detail.deferral, inherited_note(&detail, "deferral")),
    ));
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
    if let Some(helper) = &detail.headers_helper {
        fields.push(("headers_helper", helper.clone()));
    }
    if !detail.credentials.is_empty() {
        fields.push(("credentials", detail.credentials.join(", ")));
    }
    if !detail.credentials_for_another_endpoint.is_empty() {
        let elsewhere: Vec<String> = detail
            .credentials_for_another_endpoint
            .iter()
            .map(|stray| match &stray.stored_for {
                Some(origin) => format!("{} (stored for {origin})", stray.kind),
                None => format!("{} (stored for no server)", stray.kind),
            })
            .collect();
        fields.push(("not presented", elsewhere.join(", ")));
    }
    if let Some(auth) = &detail.auth {
        fields.push(("auth", auth_summary(auth)));
    }
    if let Some(allowed) = &detail.allowed_tools {
        fields.push(("allowed_tools", allowed.join(", ")));
    }
    if let Some(disabled) = &detail.disabled_tools {
        fields.push(("disabled_tools", disabled.join(", ")));
    }
    if let Some(eager) = &detail.eager_load_tools {
        fields.push(("eager_load_tools", eager.join(", ")));
    }
    if let Some(trust) = detail.trust_read_only_hint {
        fields.push(("trust_read_only_hint", trust.to_string()));
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

/// ` (from [mcp])` after a value the entry leaves unsaid, or nothing.
fn inherited_note(detail: &crate::view::McpServerDetail, key: &str) -> &'static str {
    if detail.inherited.contains(&key) {
        " (from [mcp])"
    } else {
        ""
    }
}

/// An `auth` block on one line: its `type`, then each key it states.
fn auth_summary(auth: &McpAuthConfig) -> String {
    let mut keys: Vec<String> = Vec::new();
    let mut push = |key: &str, value: Option<String>| {
        if let Some(value) = value {
            keys.push(format!("{key} = {value}"));
        }
    };
    let kind = match auth {
        McpAuthConfig::ClientCredentials { client_id, scopes } => {
            push("client_id", Some(client_id.clone()));
            push("scopes", scopes.as_ref().map(|scopes| scopes.join(" ")));
            "client_credentials"
        }
        McpAuthConfig::ClientCredentialsJwt {
            client_id,
            signing_key_path,
            signing_algorithm,
            scopes,
        } => {
            push("client_id", Some(client_id.clone()));
            push("signing_key_path", Some(signing_key_path.clone()));
            push("signing_algorithm", signing_algorithm.clone());
            push("scopes", scopes.as_ref().map(|scopes| scopes.join(" ")));
            "client_credentials_jwt"
        }
        McpAuthConfig::OAuth {
            client_id,
            scopes,
            redirect_port,
        } => {
            push("client_id", client_id.clone());
            push("scopes", scopes.as_ref().map(|scopes| scopes.join(" ")));
            push("redirect_port", redirect_port.map(|port| port.to_string()));
            "oauth"
        }
    };
    if keys.is_empty() {
        kind.to_string()
    } else {
        format!("{kind} ({})", keys.join(", "))
    }
}

/// Run `meka mcp tools <name>`: connect to the server, list every advertised tool, resolve
/// permissions, and print a column-aligned table. A tool that `allowed_tools` or `disabled_tools`
/// filters out is still shown, marked `disabled`, so users can edit their config without leaving
/// the CLI to discover names.
pub(crate) async fn run_tools(
    servers: &[McpServerConfig],
    overlay: &crate::config::McpOverlay,
    mcp_default: Option<crate::permission::Permission>,
    token_store: &TokenStore,
    name: &str,
    format: crate::cli::OutputFormat,
) -> Result<()> {
    let config = require_server(servers, name)?.clone();

    let context = McpClientContext::new();
    let (manager, connected) = connect_alone(
        &config,
        overlay,
        mcp_default,
        token_store,
        Arc::clone(&context),
        PROBE_CONNECT_TIMEOUT,
    )
    .await?;
    if !connected {
        return Err(crate::installation::mcp::probe_failure(&manager, &config.name).await);
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
                tool.raw_name.clone(),
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
                // The same words `meka tool list` uses for a built-in, plus `hidden`, which only
                // an MCP tool can be.
                match (tool.allowed, tool.exposure) {
                    (false, _) => "disabled",
                    (true, crate::config::Deferral::Eager) => "enabled",
                    (true, crate::config::Deferral::Deferred) => "deferred",
                    (true, crate::config::Deferral::Hidden) => "hidden",
                }
                .to_string(),
                crate::text::prose_cell(&tool.description),
            ]
        })
        .collect();
    crate::render::write_stdout(crate::text::format_table(
        &crate::cli::tool::TOOL_TABLE_COLUMNS,
        &rows,
    ))?;

    warn_about_dropped_tools(dropped);
    Ok(())
}

/// A tool meka dropped to stay under the per-server ceiling is, from here, indistinguishable from
/// one the server never offered. Say so: this listing is where someone goes to find out why the
/// model cannot call something the server's own docs advertise, and a `tracing::warn!` at connect
/// time is not where they will be looking.
fn warn_about_dropped_tools(dropped: usize) {
    if dropped > 0 {
        crate::streams::write_stderr_line("");
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

/// Run `meka mcp reconnect <name>`: connect once as a smoke test, silent on success and failing
/// with the error otherwise. Does not mutate config.
pub(crate) async fn run_reconnect(
    servers: &[McpServerConfig],
    overlay: &crate::config::McpOverlay,
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    let config = require_server(servers, name)?.clone();

    let context = McpClientContext::new();
    // No `[mcp].default_permission`: this is a smoke test with no `ResolvedConfig` in scope, and
    // the per-tool resolution falls through to its strict default.
    let (manager, connected) = connect_alone(
        &config,
        overlay,
        None,
        token_store,
        Arc::clone(&context),
        PROBE_CONNECT_TIMEOUT,
    )
    .await?;

    if connected {
        let version = match manager.server_entry(&config.name) {
            Some(entry) => entry.protocol_version().await,
            None => None,
        };
        match version {
            Some(version) => {
                tracing::info!("connected to '{name}' on MCP {version}", name = config.name)
            }
            None => tracing::info!("connected to '{name}'", name = config.name),
        }
        manager.shutdown_arc().await;
        Ok(())
    } else {
        Err(crate::installation::mcp::probe_failure(&manager, &config.name).await)
    }
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
    /// Raw tool names to eager-load (skip `tool_load` round-trip).
    pub(crate) eager_load_tool: Vec<String>,
    /// `deferral` for the server's other tools.
    pub(crate) deferral: Option<crate::config::Deferral>,
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
    /// [`server_config_to_table`](crate::installation::mcp::server_config_to_table).
    auth_token: Option<String>,
    client_secret: Option<String>,
    auth: Option<McpAuthConfig>,
    permission: Option<crate::permission::Permission>,
    allowed_tools: Option<Vec<String>>,
    disabled_tools: Option<Vec<String>>,
    eager_load_tools: Option<Vec<String>>,
    deferral: Option<crate::config::Deferral>,
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
pub(crate) async fn run_add(
    args: AddArgs,
    overlay: &crate::config::McpOverlay,
    token_store: &TokenStore,
) -> Result<()> {
    let resolved = resolve_add_args(args)?;
    let config = resolved_to_server_config(&resolved);
    let secrets = ServerSecrets {
        auth_token: resolved.auth_token.clone(),
        client_secret: resolved.client_secret.clone(),
    };
    add_server(&config, &secrets, token_store).await?;

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
        r = probe_then_login(&resolved, overlay, token_store) => r,
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

/// The "everything that can fail post-persist" block: probe, decide whether to auto-login, and run
/// the OAuth flow when warranted. Extracted so [`run_add`] can race it against a SIGINT handler and
/// roll back on either error path from one place.
async fn probe_then_login(
    resolved: &ResolvedAddArgs,
    overlay: &crate::config::McpOverlay,
    token_store: &TokenStore,
) -> Result<()> {
    // The caller has already established HTTP transport, no `auth_token` and a URL; an absent URL
    // here is a bug in that caller, reported rather than panicked on so the rollback still runs.
    let Some(url) = resolved.url.as_deref() else {
        return Err(MekaError::Internal(format!(
            "probe_then_login for '{}' was reached without a URL",
            resolved.name
        )));
    };

    let wants_oauth_already = matches!(resolved.auth, Some(McpAuthConfig::OAuth { .. }));
    // Expanded as the connector expands it, or the probe would send a literal `${KEY}` to the
    // host: the one door the server's own rule does not reach. Skipped rather than refused when
    // this shell cannot supply a name, since the server's own runs may, and the entry is written.
    let expansion = crate::config::expand_env_vars(url, |name| std::env::var(name).ok());
    let probe = if expansion.missing.is_empty() && expansion.malformed.is_empty() {
        probe_and_announce(&resolved.name, &expansion.text).await
    } else {
        tracing::warn!(
            "skipping the auth probe of '{}': its URL names {:?}, which this shell does not supply",
            resolved.name,
            [expansion.missing, expansion.malformed].concat()
        );
        ProbeOutcome::Inconclusive
    };
    let should_login = match probe {
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
    login_server(
        std::slice::from_ref(&server_config),
        overlay,
        token_store,
        &resolved.name,
        Arc::new(TerminalLogin),
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
        deferral,
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
                .map_err(|error| usage_err(format!("`--permission`: {error}")))
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
                usage_err(format!(
                    "`--tool-permission` takes TOOL=LEVEL, got '{entry}'"
                ))
            })?;
            let tool = tool.trim();
            let level = level.trim();
            if tool.is_empty() {
                return Err(usage_err(format!(
                    "`--tool-permission` entry '{entry}' has an empty tool name"
                )));
            }
            let level = level
                .parse::<crate::permission::Permission>()
                .map_err(|error| {
                    usage_err(format!("`--tool-permission` entry '{entry}': {error}"))
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
        return Err(usage_err(
            "`--auth-token-stdin` cannot be combined with `--auth`",
        ));
    }

    match transport {
        McpTransport::Stdio => {
            let command = location.ok_or_else(|| {
                usage_err("stdio transport needs an executable as the positional argument")
            })?;
            if !header.is_empty() {
                return Err(usage_err("`--header` is HTTP-only"));
            }
            if auth_token.is_some() || auth.is_some() || auth_flags_present {
                return Err(usage_err("auth flags are HTTP-only"));
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
                deferral,
                tool_permissions,
                no_login,
                disabled,
                required,
            })
        }
        McpTransport::Http => {
            let url = location.ok_or_else(|| {
                usage_err("http transport needs a URL as the positional argument")
            })?;
            if !tail.is_empty() {
                return Err(usage_err("http transport takes no trailing arguments"));
            }
            if !env.is_empty() {
                return Err(usage_err("`--env` is stdio-only"));
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
                deferral,
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
                return Err(usage_err("OAuth-family flags need `--auth`"));
            }
            Ok(None)
        }
        Some(McpAuthKind::OAuth) => {
            if signing_key.is_some() || signing_algorithm.is_some() {
                return Err(usage_err(
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
                .ok_or_else(|| usage_err("`--auth client_credentials` needs `--client-id`"))?;
            if client_secret.is_none() {
                return Err(usage_err(
                    "`--auth client_credentials` needs `--client-secret-stdin`",
                ));
            }
            if signing_key.is_some() || signing_algorithm.is_some() {
                return Err(usage_err(
                    "`--signing-key` and `--signing-algorithm` need `--auth client_credentials_jwt`",
                ));
            }
            if redirect_port.is_some() {
                return Err(usage_err("`--redirect-port` needs `--auth oauth`"));
            }
            Ok(Some(McpAuthConfig::ClientCredentials {
                client_id,
                scopes: if scope.is_empty() { None } else { Some(scope) },
            }))
        }
        Some(McpAuthKind::ClientCredentialsJwt) => {
            let client_id = client_id
                .ok_or_else(|| usage_err("`--auth client_credentials_jwt` needs `--client-id`"))?;
            let signing_key_path = signing_key.ok_or_else(|| {
                usage_err("`--auth client_credentials_jwt` needs `--signing-key`")
            })?;
            if client_secret.is_some() {
                return Err(usage_err(
                    "`--auth client_credentials_jwt` signs a JWT and takes no `--client-secret-stdin`",
                ));
            }
            if redirect_port.is_some() {
                return Err(usage_err("`--redirect-port` needs `--auth oauth`"));
            }
            Ok(Some(McpAuthConfig::ClientCredentialsJwt {
                client_id,
                signing_key_path,
                signing_algorithm,
                scopes: if scope.is_empty() { None } else { Some(scope) },
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
            .ok_or_else(|| usage_err(format!("`{flag}` takes KEY=VALUE, got '{entry}'")))?;
        if k.is_empty() {
            return Err(usage_err(format!(
                "`{flag}` entry '{entry}' has an empty key"
            )));
        }
        out.push((k.to_string(), v.to_string()));
    }
    Ok(out)
}

/// An [`McpServerConfig`] equal to what parsing the entry just written to `config.toml` would
/// yield, so the auto-login path in [`run_add`] can call [`login_server`] without a round trip
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
        deferral: resolved.deferral,
        tool_permissions: resolved.tool_permissions.clone(),
        // No `meka mcp add` flag drives this: declining a server's `readOnlyHint` is a standing
        // policy decision about a server you already distrust, not something to pick while adding
        // one. `None` means the default (trust it), and the knob is edited in `config.toml`.
        trust_read_only_hint: None,
        disabled: resolved.disabled.then_some(true),
        required: resolved.required.then_some(true),
    }
}

/// Run `meka mcp disable <name>`. Sets `disabled = true` on the server entry in config.toml.
pub(crate) async fn run_disable(name: &str, token_store: &TokenStore) -> Result<()> {
    let (_, revocation) = edit_server(
        name,
        &ServerChanges {
            disabled: Patch::Set(true),
            ..ServerChanges::default()
        },
        token_store,
    )
    .await?;
    revocation.send().await;
    tracing::info!("disabled '{name}'");
    Ok(())
}

/// Run `meka mcp enable <name>`. Clears `disabled` from the server entry in config.toml.
pub(crate) async fn run_enable(name: &str, token_store: &TokenStore) -> Result<()> {
    let (_, revocation) = edit_server(
        name,
        &ServerChanges {
            disabled: Patch::Set(false),
            ..ServerChanges::default()
        },
        token_store,
    )
    .await?;
    revocation.send().await;
    tracing::info!("enabled '{name}'");
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
        // The raw-document editors run on a file meka cannot parse, since they are how it gets
        // repaired; `main` has already noted the failure once.
    } else {
        config.require_readable_config()?;
    }
    let token_store = store.token_store();
    match action {
        // Shown as the servers run: what an entry leaves unsaid, under the `[mcp]` defaults.
        crate::cli::McpAction::List { format } => {
            crate::cli::mcp::list_servers(
                &config.mcp_servers_as_served(),
                None,
                &token_store,
                *format,
                crate::render::Stream::Stdout,
            )
            .await?
        }
        crate::cli::McpAction::Get { name, format } => {
            crate::cli::mcp::run_get(
                &config.mcp_servers,
                &config.mcp_overlay,
                name,
                &token_store,
                *format,
            )
            .await?
        }
        crate::cli::McpAction::Reconnect { name } => {
            crate::cli::mcp::run_reconnect(
                &config.mcp_servers,
                &config.mcp_overlay,
                &token_store,
                name,
            )
            .await?
        }
        crate::cli::McpAction::Tools { name, format } => {
            crate::cli::mcp::run_tools(
                &config.mcp_servers,
                &config.mcp_overlay,
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
                    crate::cli::mcp::store_server_secret(
                        &config.mcp_servers,
                        &token_store,
                        name,
                        kind,
                        &secret,
                    )
                    .await?
                }
                None => {
                    crate::cli::mcp::login_server(
                        &config.mcp_servers,
                        &config.mcp_overlay,
                        &token_store,
                        name,
                        Arc::new(TerminalLogin),
                    )
                    .await?
                }
            }
        }
        crate::cli::McpAction::Logout { name } => {
            let revocation =
                crate::cli::mcp::logout_server(&config.mcp_servers, &token_store, name).await?;
            revocation.send().await;
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
            deferral,
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
                    deferral: *deferral,
                    tool_permission: tool_permission.clone(),
                    disabled: *disabled,
                    required: *required,
                },
                &config.mcp_overlay,
                &token_store,
            )
            .await?
        }
        crate::cli::McpAction::Remove { name } => {
            crate::cli::mcp::remove_server(name, &token_store)
                .await?
                .send()
                .await;
        }
        crate::cli::McpAction::Disable { name } => {
            crate::cli::mcp::run_disable(name, &token_store).await?
        }
        crate::cli::McpAction::Enable { name } => {
            crate::cli::mcp::run_enable(name, &token_store).await?
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::installation::mcp::{server_config_to_table, store_secrets};

    async fn memory_token_store() -> TokenStore {
        crate::store::Store::for_test().await.token_store()
    }

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
            deferral: None,
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
        let rendered = server_config_to_table(&resolved_to_server_config(&resolved)).to_string();
        assert!(
            !rendered.contains("bearer-not-a-real-token"),
            "the bearer must not be written to config.toml, got:\n{rendered}"
        );

        let mut args = bare_add("srv", Some("https://example.com"));
        args.auth = Some(crate::cli::McpAuthKind::OAuth);
        args.client_secret = Some("cs-not-a-real-secret".to_string());
        let resolved = resolve_add_args(args).expect("should resolve");
        let rendered = server_config_to_table(&resolved_to_server_config(&resolved)).to_string();
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
        servers.push(server_config_to_table(&resolved_to_server_config(
            &resolved,
        )));
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
            servers.push(server_config_to_table(&resolved_to_server_config(
                &resolved,
            )));
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

    /// `--deferral` is written as the key, and left out when the flag was not given, so the server
    /// keeps inheriting `[mcp].deferral`; a spelling outside the three is refused at the flag.
    #[test]
    fn build_server_table_persists_deferral_only_when_set() {
        let round_trip = |deferral: Option<crate::config::Deferral>| {
            let mut args = bare_add("bridge", Some("http://127.0.0.1:9100/mcp"));
            args.deferral = deferral;
            let resolved = resolve_add_args(args)?;
            let mut doc = toml_edit::DocumentMut::new();
            let mut servers = toml_edit::ArrayOfTables::new();
            servers.push(server_config_to_table(&resolved_to_server_config(
                &resolved,
            )));
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
            Ok::<_, anyhow::Error>(parsed.mcp.expect("mcp").servers.expect("servers")[0].deferral)
        };
        assert_eq!(
            round_trip(Some(crate::config::Deferral::Eager)).expect("resolve"),
            Some(crate::config::Deferral::Eager)
        );
        assert_eq!(round_trip(None).expect("resolve"), None);
        use clap::Parser as _;
        let refused = crate::cli::Cli::try_parse_from([
            "meka",
            "mcp",
            "add",
            "--deferral",
            "lazy",
            "bridge",
            "http://127.0.0.1:9100/mcp",
        ])
        .expect_err("refused at the flag");
        assert!(
            refused.to_string().contains("deferred, eager, hidden"),
            "{refused}"
        );
    }

    fn server_named(name: &str) -> McpServerConfig {
        let resolved = resolve_add_args(bare_add(name, Some("https://example.test/mcp")))
            .expect("resolve add args");
        resolved_to_server_config(&resolved)
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
        store_secrets(
            &resolved_to_server_config(&resolved),
            &ServerSecrets {
                auth_token: resolved.auth_token.clone(),
                client_secret: resolved.client_secret.clone(),
            },
            &store,
        )
        .await
        .expect("store");

        assert_eq!(
            store
                .load_mcp_credentials("api", McpCredentialKind::ClientSecret)
                .await
                .expect("load")
                .map(|stored| stored.secret)
                .as_deref(),
            Some("cs-not-a-real-secret")
        );

        let mut args = bare_add("bear", Some("https://example.test/mcp"));
        args.auth_token = Some("bearer-not-a-real-token".to_string());
        let resolved = resolve_add_args(args).expect("resolve");
        store_secrets(
            &resolved_to_server_config(&resolved),
            &ServerSecrets {
                auth_token: resolved.auth_token.clone(),
                client_secret: resolved.client_secret.clone(),
            },
            &store,
        )
        .await
        .expect("store");

        assert_eq!(
            store
                .load_mcp_credentials("bear", McpCredentialKind::Bearer)
                .await
                .expect("load")
                .map(|stored| stored.secret)
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
                    "https://example.test/mcp",
                )
                .await
                .expect("save");

            let mut server = server_named("api");
            let had_auth = auth.is_some();
            server.auth = auth;

            let error = login_server(
                std::slice::from_ref(&server),
                &crate::config::McpOverlay::default(),
                &store,
                "api",
                Arc::new(TerminalLogin),
            )
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
                    .map(|stored| stored.secret)
                    .as_deref(),
                Some("bearer-not-a-real-token"),
                "and it must not have cleared the bearer it refused over (auth block: {had_auth})"
            );
        }
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
                    r#"{"client_id":"c","token_response":{"access_token":"at","token_type":"bearer"}}"#,
                    "",
                )
                .await
                .expect("save");
        }

        // `notion` is configured but never logged in to, which is not an orphan: the diff runs in
        // one direction only.
        let servers = vec![server_named("linear"), server_named("notion")];
        let orphans = crate::installation::mcp::orphaned_credentials(&servers, &store)
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
                "",
            )
            .await
            .expect("save the credential the deleted server left behind");

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set → run → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        // Loopback discard port: `run_add` must refuse before it can probe, so nothing dials out.
        let result = run_add(
            bare_add("api", Some("http://127.0.0.1:9/mcp")),
            &crate::config::McpOverlay::default(),
            &store,
        )
        .await;
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
                .map(|stored| stored.secret)
                .as_deref(),
            Some("bearer-not-a-real-token"),
            "and it must not have destroyed the credential it refused over"
        );
    }
}
