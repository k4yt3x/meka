//! The `[[mcp.servers]]` entries and the credentials stored for them: the doors `meka mcp` and
//! `/v1/mcp` write through, and the logins and logouts both run.

use std::sync::Arc;

use super::Patch;
use crate::{
    config::{McpAuthConfig, McpServerConfig, McpTransport},
    error::{MekaError, Result},
    mcp::{McpClientContext, McpClientManager},
    store::TokenStore,
    view::{McpServerDetail, McpServerView},
};

/// `config.toml` holds something this command cannot work with.
fn config_err(message: impl Into<String>) -> MekaError {
    MekaError::Config(message.into())
}

/// The command was asked for something it cannot do as asked.
pub(crate) fn usage_err(message: impl Into<String>) -> MekaError {
    MekaError::Usage(message.into())
}

/// The configured server `name` denotes, or the refusal naming the ones that exist.
pub(crate) fn require_server<'a>(
    servers: &'a [McpServerConfig],
    name: &str,
) -> Result<&'a McpServerConfig> {
    servers
        .iter()
        .find(|server| server.name == name)
        .ok_or_else(|| {
            usage_err(crate::text::unknown_name(
                "MCP server",
                name,
                servers.iter().map(|server| server.name.as_str()),
            ))
        })
}

/// One server's configuration in full, as `meka mcp get` prints it and `GET /v1/mcp/{name}`
/// answers with: the entry, the kinds of credential stored under it, and where the stored grant
/// was minted against.
pub(crate) async fn server_detail(
    config: &McpServerConfig,
    token_store: &TokenStore,
) -> Result<McpServerDetail> {
    let name = config.name.as_str();
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
    Ok(McpServerDetail {
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
    })
}

/// The document as it will be written, refused ahead of the write when the edit would leave it
/// unreadable. A file that already fails the typed read is written as edited: these doors reach
/// only the entry they name, which they read as typed, and they are how such a file is repaired,
/// as `account remove` and `profile remove` are, so a key meka does not know elsewhere in it is
/// not theirs to refuse.
fn reparsed_edit(before: &str, document: &toml_edit::DocumentMut) -> Result<String> {
    let after = document.to_string();
    if toml::from_str::<crate::config::ConfigFile>(before).is_err() {
        return Ok(after);
    }
    super::account::reparse_after_edit(before, &after)
        .map_err(|error| MekaError::Config(error.to_string()))?;
    Ok(after)
}

/// How long a probe gives one server to connect: `meka mcp tools` and `meka mcp reconnect`.
pub(crate) const PROBE_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Connect `config` on its own, wait for the connector to settle, and say whether it ended
/// connected; the manager comes back for what the caller reads off it. The one probe `mcp tools`,
/// `mcp reconnect` and a login share, so the three cannot count differently; `connect_timeout`
/// bounds the connect, the browser wait of a login included.
pub(crate) async fn connect_alone(
    config: &McpServerConfig,
    overlay: &crate::config::McpOverlay,
    mcp_default: Option<crate::permission::Permission>,
    token_store: &TokenStore,
    context: Arc<McpClientContext>,
    connect_timeout: std::time::Duration,
) -> Result<(Arc<McpClientManager>, bool)> {
    // None of the three probes calls a tool, so the per-call cap never applies and the default
    // stands in rather than threading a `ResolvedConfig` these commands do not otherwise need;
    // the overlay is threaded, since what a probe lists is what the entry runs under.
    let manager = McpClientManager::prepare_with_overlay(
        std::slice::from_ref(config),
        overlay.clone(),
        mcp_default,
        Some(token_store.clone()),
        context,
        crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
    )
    .await?;
    manager.start_connector(crate::mcp::McpRuntimeConfig {
        connect_timeout,
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
    Ok((manager, connected))
}

/// The server a logout revokes at: the http server of that name, since a stdio server holds no
/// grant.
fn revokes_at<'a>(servers: &'a [McpServerConfig], name: &str) -> Option<&'a McpServerConfig> {
    servers
        .iter()
        .find(|server| server.name == name && matches!(server.transport, McpTransport::Http))
}

/// Clear every credential stored for a server, the OAuth grant revoked at the provider first
/// where it can be: `meka mcp logout` and `POST /v1/mcp/{name}/logout`.
pub(crate) async fn logout_server(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
) -> Result<()> {
    // Best-effort revocation; the stored credentials are cleared regardless.
    if let Some(config) = revokes_at(servers, name)
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

/// Store a secret the user already holds: `meka mcp login --auth-token-stdin` or
/// `--client-secret-stdin`, and `POST /v1/mcp/{name}/login` with one.
///
/// Separate from [`login_server`] because the two do opposite things: that one goes and obtains a
/// credential, this one is handed one. A confidential OAuth client needs both, in this order:
/// deposit the client secret, then run the flow that presents it.
pub(crate) async fn store_server_secret(
    servers: &[McpServerConfig],
    token_store: &TokenStore,
    name: &str,
    kind: crate::store::McpCredentialKind,
    secret: &str,
) -> Result<()> {
    let server = require_server(servers, name)?;
    refuse_a_misplaced_secret(server, kind)?;
    token_store.save_mcp_credentials(name, kind, secret).await?;
    tracing::info!("stored {kind} for '{name}'", kind = kind.label());
    Ok(())
}

/// Refuse a secret of `kind` on a server whose configuration cannot present it: the states
/// `resolve_add_args` refuses at add time, asked again by every other door onto the same state,
/// because a rule enforced at one door of several is not enforced.
fn refuse_a_misplaced_secret(
    server: &McpServerConfig,
    kind: crate::store::McpCredentialKind,
) -> Result<()> {
    use crate::store::McpCredentialKind;

    let name = server.name.as_str();
    // Checked against config rather than accepted for any name, so a typo leaves a stranded
    // credential under a server that will never exist rather than silently doing nothing useful.
    if server.transport != McpTransport::Http {
        return Err(usage_err(format!(
            "server '{name}' is stdio and has no HTTP credential; its secrets go in `env`"
        )));
    }
    match (kind, &server.auth) {
        (McpCredentialKind::Bearer, Some(_)) => Err(usage_err(format!(
            "server '{name}' authenticates through its `[auth]` block, which a stored bearer \
             would override; drop the block first"
        ))),
        (McpCredentialKind::ClientSecret, None) => Err(usage_err(format!(
            "server '{name}' has no `[auth]` block to present a client secret; add one with \
             `type = \"oauth\"` or `\"client_credentials\"` first"
        ))),
        (McpCredentialKind::ClientSecret, Some(McpAuthConfig::ClientCredentialsJwt { .. })) => {
            Err(usage_err(format!(
                "server '{name}' authenticates with a signed JWT, not a client secret; the key it \
                 signs with is `signing_key_path`"
            )))
        }
        _ => Ok(()),
    }
}

/// Connect `config` once with a person at hand for the browser step, and say why it ended short of
/// `Connected`, or `None` for a connect that landed. Split from `login_server` so every way the
/// flow can end short, an error included, reaches the one place that puts the previous bundle
/// back. The reason is the bare cause the connector recorded on the flow's own entry, URLs
/// redacted: the connect's error is answered as a state there, and a login that fails on it would
/// otherwise say only that it did not complete.
async fn login_flow(
    config: &McpServerConfig,
    overlay: &crate::config::McpOverlay,
    token_store: &TokenStore,
    prompt: Arc<dyn crate::oauth::LoginPrompt>,
) -> Result<Option<String>> {
    let context = McpClientContext::new();
    // The one place a person is there to finish a browser login, so the one place the client is
    // given a way to ask them.
    context.set_login_prompt(prompt);
    // No `[mcp].default_permission`, for the reason `run_reconnect` gives. The connect holds the
    // browser wait, so its bound is the callback's plus a probe's, or the wait the person is told
    // of would be cut short.
    let (manager, connected) = connect_alone(
        config,
        overlay,
        None,
        token_store,
        context,
        crate::mcp::auth::OAUTH_CALLBACK_TIMEOUT + PROBE_CONNECT_TIMEOUT,
    )
    .await?;
    let reason = if connected {
        None
    } else {
        let recorded = match manager.server_entry(&config.name) {
            Some(entry) => match entry.state().await {
                crate::mcp::ServerState::Failed { error, .. } => Some(error),
                _ => None,
            },
            None => None,
        };
        Some(recorded.unwrap_or_else(|| "the OAuth flow did not complete".to_string()))
    };
    manager.shutdown_arc().await;
    Ok(reason)
}

/// Put `name` back on a running manager from the entry as `config.toml` now states it, once a
/// login stored a credential: the running entry predates the `auth` block the login wrote, and a
/// server restarted from the file connects with the credential. `Ok(false)` when the file no
/// longer names the server.
pub(crate) async fn restart_from_file(
    manager: &std::sync::Arc<crate::mcp::McpClientManager>,
    name: &str,
) -> Result<bool> {
    let config = crate::config::load_config_file_or_err()?
        .mcp
        .and_then(|mcp| mcp.servers)
        .and_then(|servers| servers.into_iter().find(|server| server.name == name));
    let Some(config) = config else {
        return Ok(false);
    };
    manager.replace_server(&config).await?;
    Ok(true)
}

// Log a server in through its OAuth flow, with `prompt` as the person's side of it:
/// `meka mcp login` and `POST /v1/mcp/{name}/login` with no body.
///
/// An explicit `[auth]` block is honored as written. An HTTP server without one is assumed to take
/// `type = "oauth"`, and on success that block is written back to `config.toml` so later runs do
/// not assume. A stdio server has nothing to log in to.
pub(crate) async fn login_server(
    servers: &[McpServerConfig],
    overlay: &crate::config::McpOverlay,
    token_store: &TokenStore,
    name: &str,
    prompt: Arc<dyn crate::oauth::LoginPrompt>,
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
        return Err(usage_err(format!(
            "server '{name}' has a stored bearer, which would override the login; drop it first \
             with `meka mcp logout {name}`"
        )));
    }

    // A disabled entry is connected by nothing, so a login would end as "did not complete" with
    // nothing naming the cause.
    if base_config.disabled.unwrap_or(false) {
        return Err(usage_err(format!(
            "server '{name}' is disabled; enable it first"
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
                return Err(usage_err(format!(
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

    let failure = match login_flow(&config, overlay, token_store, prompt).await {
        Ok(None) => None,
        Ok(Some(reason)) => Some(Ok(reason)),
        Err(error) => Some(Err(error)),
    };
    if let Some(failure) = failure {
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
        return Err(match failure {
            // The flow's own word on why the login did not complete; an error is its own class.
            Ok(reason) => MekaError::McpAuth {
                server_name: config.name.clone(),
                message: reason,
            },
            Err(error) => error,
        });
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
/// assumption [`login_server`] made is recorded rather than reapplied on every later run.
fn persist_auth_block_for(name: &str) -> Result<()> {
    // Held to the end of the function, so this read and the write below cannot interleave with
    // another editor's, `device_id::persist` on an ordinary launch included.
    let (_config_lock, path, mut document) = super::account::open_document()?;
    let before = document.to_string();

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
        crate::fs::write_file_atomic(&path, &reparsed_edit(&before, &document)?)
            .map_err(|error| MekaError::io(format!("failed to write config: {error}")))?;
    }
    Ok(())
}

/// The secrets an entry is added or edited with, which go to the store and never to the table.
#[derive(Default)]
pub(crate) struct ServerSecrets {
    pub(crate) auth_token: Option<String>,
    pub(crate) client_secret: Option<String>,
}

/// Refuse a name no tool prefix can carry, one meka reserves, or none: the rule the running
/// manager holds an entry to, asked ahead of the write so the file never holds an entry the next
/// start refuses.
fn validate_server_name(name: &str) -> Result<()> {
    crate::mcp::sanitize::validate_server_name(name).map_err(usage_err)
}

/// Refuse an entry whose keys do not fit its transport: the rule `resolve_add_args` applies to
/// the flags, asked of the entry itself by every door that writes one, so a stdio server never
/// carries a URL and an HTTP one never a command.
pub(crate) fn validate_server_shape(config: &McpServerConfig) -> Result<()> {
    let name = config.name.as_str();
    match config.transport {
        McpTransport::Stdio => {
            if config.command.as_deref().is_none_or(str::is_empty) {
                return Err(usage_err(format!(
                    "server '{name}' is stdio and needs `command`"
                )));
            }
            if config.url.is_some() || config.headers.is_some() || config.auth.is_some() {
                return Err(usage_err(format!(
                    "server '{name}' is stdio; `url`, `headers` and `auth` are for an http server"
                )));
            }
        }
        McpTransport::Http => {
            if config.url.as_deref().is_none_or(str::is_empty) {
                return Err(usage_err(format!(
                    "server '{name}' is http and needs `url`"
                )));
            }
            if config.command.is_some() || config.args.is_some() || config.env.is_some() {
                return Err(usage_err(format!(
                    "server '{name}' is http; `command`, `args` and `env` are for a stdio server"
                )));
            }
        }
    }
    Ok(())
}

/// Write `config` as a new `[[mcp.servers]]` entry and store its secrets: `meka mcp add` once
/// its flags are resolved, and `POST /v1/mcp`.
///
/// The entry is refused when the name is in use or a credential is stored under it, and written
/// before the secrets so a failed credential write can roll it back; a failure anywhere leaves
/// the file as it was.
pub(crate) async fn add_server(
    config: &McpServerConfig,
    secrets: &ServerSecrets,
    token_store: &TokenStore,
) -> Result<()> {
    validate_server_name(&config.name)?;
    validate_server_shape(config)?;
    if secrets.auth_token.is_some() {
        refuse_a_misplaced_secret(config, crate::store::McpCredentialKind::Bearer)?;
    }
    if secrets.client_secret.is_some() {
        refuse_a_misplaced_secret(config, crate::store::McpCredentialKind::ClientSecret)?;
    }
    let name = config.name.as_str();

    // Held across this read and the write below, so the two cannot interleave with another
    // editor's (`device_id::persist` on an ordinary launch included), and released the moment the
    // write lands; see the `drop` below.
    // Read ahead of the lock, which is never held across an await: its depth is a thread-local,
    // and a task resuming on another worker would leave it wrong (see `ConfigFileLock`). Acted on
    // under the lock, after the duplicate check, so the refusal names what is in the way.
    let leftover_credential = token_store.has_mcp_credentials(name).await?;
    let (config_lock, path, mut document) = super::account::open_document()?;
    let before = document.to_string();

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
            .is_some_and(|n| n == name)
        {
            return Err(usage_err(format!(
                "server '{name}' already exists in config.toml"
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
    if leftover_credential {
        return Err(usage_err(format!(
            "'{name}' has a stored credential left over from a removed server; clear it first \
             with `meka mcp remove {name}`"
        )));
    }

    servers_array.push(server_config_to_table(config));

    crate::fs::write_file_atomic(&path, &reparsed_edit(&before, &document)?)
        .map_err(|error| MekaError::io(format!("failed to write config: {error}")))?;
    tracing::info!("added '{name}' to {path}", path = path.display());

    // Released before anything that talks to a network or waits on a human: `lock_config_file`
    // blocks with no timeout, so holding it across the OAuth flow would hang every other meka
    // launch that touches `config.toml` (`device_id::persist` does, on an ordinary start) for as
    // long as the browser login takes. `purge_server` takes the lock again for the rollback below.
    drop(config_lock);

    // Secrets are not config, so they land here rather than in the table just written. Straight
    // after that write, because a server whose entry exists without the credential it needs would
    // make unauthenticated requests and be told it is unauthorized, which says nothing about the
    // real cause. A failure rolls the entry back like every other post-persist failure.
    if let Err(error) = store_secrets(name, secrets, token_store).await {
        tracing::warn!(
            "failed to store credentials for '{name}': {error}; rolling back the config entry"
        );
        return Err(roll_back(name, token_store, error).await);
    }
    Ok(())
}

/// What `PATCH /v1/mcp/{name}` and `meka mcp enable|disable` change on an entry: each key kept,
/// removed or set, a string key reading a blank as a removal. The secrets go to the store: a
/// removal clears the kind.
#[derive(Default)]
pub(crate) struct ServerChanges {
    pub(crate) command: Patch<String>,
    pub(crate) args: Patch<Vec<String>>,
    pub(crate) env: Patch<std::collections::HashMap<String, String>>,
    pub(crate) url: Patch<String>,
    pub(crate) headers: Patch<std::collections::HashMap<String, String>>,
    pub(crate) headers_helper: Patch<String>,
    pub(crate) auth: Patch<McpAuthConfig>,
    pub(crate) permission: Patch<crate::permission::Permission>,
    pub(crate) allowed_tools: Patch<Vec<String>>,
    pub(crate) disabled_tools: Patch<Vec<String>>,
    pub(crate) eager_load_tools: Patch<Vec<String>>,
    pub(crate) deferral: Patch<crate::config::Deferral>,
    pub(crate) tool_permissions:
        Patch<std::collections::HashMap<String, crate::permission::Permission>>,
    pub(crate) trust_read_only_hint: Patch<bool>,
    pub(crate) disabled: Patch<bool>,
    pub(crate) required: Patch<bool>,
    pub(crate) auth_token: Patch<String>,
    pub(crate) client_secret: Patch<String>,
}

/// Apply `changes` to the `[[mcp.servers]]` entry named `name`, in place, and give back the entry
/// as the file now states it. One critical section: the entry is read, changed, validated whole
/// through [`validate_server_shape`] and written once under the config file lock, so two edits
/// cannot combine into an entry either would refuse. Every untouched key keeps its place and its
/// comment; a changed key keeps the comment beside it. The secrets land in the store after the
/// write, each refused on a shape that cannot present it.
pub(crate) async fn edit_server(
    name: &str,
    changes: &ServerChanges,
    token_store: &TokenStore,
) -> Result<McpServerConfig> {
    // Read ahead of the lock, which is never held across an await (see `ConfigFileLock`); the
    // rule it feeds is asked under the lock, of the entry as it stands then.
    let bearer_stored = token_store
        .load_mcp_credentials(name, crate::store::McpCredentialKind::Bearer)
        .await?
        .is_some();
    let auth_token = changes.auth_token.clone().blank_removes();
    let client_secret = changes.client_secret.clone().blank_removes();
    let (config, path, touched) = {
        let (_config_lock, path, mut document) = super::account::open_document()?;
        let before = document.to_string();
        // The entry alone is read as typed, off the raw document: this door is one a shell
        // repairs a file with, so a key meka does not know elsewhere in it is not its business.
        let known = server_names(&document);
        let unknown = || usage_err(crate::text::unknown_name("MCP server", name, known.iter()));
        let entry_table = document
            .get("mcp")
            .and_then(|mcp| mcp.get("servers"))
            .and_then(|servers| servers.as_array_of_tables())
            .and_then(|servers| {
                servers
                    .iter()
                    .find(|entry| entry.get("name").and_then(|v| v.as_str()) == Some(name))
            })
            .ok_or_else(unknown)?;
        let mut config: McpServerConfig =
            toml::from_str(&render_entry(entry_table)).map_err(|error| {
                config_err(format!(
                    "server '{name}' in config.toml does not parse: {error}"
                ))
            })?;
        let mut touched: Vec<&'static str> = Vec::new();
        macro_rules! apply {
            ($field:ident, $key:literal, $patch:expr) => {
                match $patch {
                    Patch::Keep => {}
                    Patch::Remove => {
                        config.$field = None;
                        touched.push($key);
                    }
                    Patch::Set(value) => {
                        config.$field = Some(value);
                        touched.push($key);
                    }
                }
            };
        }
        apply!(command, "command", changes.command.clone().blank_removes());
        apply!(args, "args", changes.args.clone());
        apply!(env, "env", changes.env.clone());
        apply!(url, "url", changes.url.clone().blank_removes());
        apply!(headers, "headers", changes.headers.clone());
        apply!(
            headers_helper,
            "headers_helper",
            changes.headers_helper.clone().blank_removes()
        );
        apply!(auth, "auth", changes.auth.clone());
        apply!(permission, "permission", changes.permission.clone());
        apply!(
            allowed_tools,
            "allowed_tools",
            changes.allowed_tools.clone()
        );
        apply!(
            disabled_tools,
            "disabled_tools",
            changes.disabled_tools.clone()
        );
        apply!(
            eager_load_tools,
            "eager_load_tools",
            changes.eager_load_tools.clone()
        );
        apply!(deferral, "deferral", changes.deferral.clone());
        apply!(
            tool_permissions,
            "tool_permissions",
            changes.tool_permissions.clone()
        );
        apply!(
            trust_read_only_hint,
            "trust_read_only_hint",
            changes.trust_read_only_hint.clone()
        );
        apply!(required, "required", changes.required.clone());
        match changes.disabled {
            Patch::Keep => {}
            // `false` is the key's absence: a server not disabled carries no key, which keeps
            // the diff minimal for one that never was.
            Patch::Remove | Patch::Set(false) => {
                config.disabled = None;
                touched.push("disabled");
            }
            Patch::Set(true) => {
                config.disabled = Some(true);
                touched.push("disabled");
            }
        }
        validate_server_shape(&config)?;
        // A bearer in the store overrides an `auth` block, so neither may be set while the other
        // stands; asked of the shape about to be written and the store as it was read.
        let bearer_after = match &auth_token {
            Patch::Set(_) => true,
            Patch::Remove => false,
            Patch::Keep => bearer_stored,
        };
        if bearer_after && config.auth.is_some() {
            return Err(usage_err(format!(
                "server '{name}' would hold both an `auth` block and a stored bearer, which \
                 overrides it; set one or the other"
            )));
        }
        if matches!(auth_token, Patch::Set(_)) {
            refuse_a_misplaced_secret(&config, crate::store::McpCredentialKind::Bearer)?;
        }
        if matches!(client_secret, Patch::Set(_)) {
            refuse_a_misplaced_secret(&config, crate::store::McpCredentialKind::ClientSecret)?;
        }
        let fresh = server_config_to_table(&config);
        {
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
            for key in &touched {
                match fresh.get(key) {
                    Some(item) => {
                        // The value's decor holds what follows it on the line, so a `# note`
                        // beside the old value stays beside the new one.
                        let mut item = item.clone();
                        if let (Some(existing), Some(value)) = (
                            entry.get(key).and_then(|item| item.as_value()),
                            item.as_value_mut(),
                        ) {
                            *value.decor_mut() = existing.decor().clone();
                        }
                        entry.insert(key, item);
                    }
                    None => {
                        entry.remove(key);
                    }
                }
            }
        }
        crate::fs::write_file_atomic(&path, &reparsed_edit(&before, &document)?)
            .map_err(|error| MekaError::io(format!("failed to write config: {error}")))?;
        (config, path, touched)
    };

    for (change, kind) in [
        (&auth_token, crate::store::McpCredentialKind::Bearer),
        (
            &client_secret,
            crate::store::McpCredentialKind::ClientSecret,
        ),
    ] {
        match change {
            Patch::Set(secret) => {
                token_store.save_mcp_credentials(name, kind, secret).await?;
                tracing::info!("stored {kind} for '{name}'", kind = kind.label());
            }
            Patch::Remove => {
                token_store
                    .clear_mcp_credentials_of_kind(name, kind)
                    .await?;
                tracing::info!("cleared {kind} for '{name}'", kind = kind.label());
            }
            Patch::Keep => {}
        }
    }
    tracing::info!(
        "changed '{name}' in {path}: {keys}",
        path = path.display(),
        keys = touched.join(", ")
    );
    Ok(config)
}

/// Write the secrets `add` was given to the store, under the kind each one is.
///
/// Two separate rows rather than one, because `--auth-token-stdin` and `--client-secret-stdin` are
/// mutually exclusive only by the flag rules above; the schema does not depend on that and neither
/// does this.
pub(crate) async fn store_secrets(
    name: &str,
    secrets: &ServerSecrets,
    token_store: &TokenStore,
) -> Result<()> {
    use crate::store::McpCredentialKind;

    if let Some(token) = &secrets.auth_token {
        token_store
            .save_mcp_credentials(name, McpCredentialKind::Bearer, token)
            .await?;
    }
    if let Some(secret) = &secrets.client_secret {
        token_store
            .save_mcp_credentials(name, McpCredentialKind::ClientSecret, secret)
            .await?;
    }
    Ok(())
}

/// Undo everything `add` wrote, so a failure anywhere after the config write leaves the user where
/// a clean `mcp remove` would. Returns `error` unchanged, for `return Err(roll_back(…).await)`.
///
/// The reason is logged by the caller, which is the only place that knows it; a rollback that
/// itself fails is reported here, because at that point the user has to finish it by hand.
pub(crate) async fn roll_back(name: &str, token_store: &TokenStore, error: MekaError) -> MekaError {
    if let Err(purge_error) = purge_server(name, token_store).await {
        tracing::warn!(
            "failed to roll back '{name}': {purge_error}; remove its entry from config.toml by hand"
        );
    }
    error
}

/// Serialize an entry into the TOML table `[[mcp.servers]]` carries, in the order the reference
/// documents the keys. Only the keys the entry states are emitted so hand-edited config files stay
/// readable.
///
/// No secret is here to write: a bearer and a client secret go to the store, and this function is
/// handed no way to write one.
pub(crate) fn server_config_to_table(config: &McpServerConfig) -> toml_edit::Table {
    let mut table = toml_edit::Table::new();
    table.insert("name", toml_edit::value(config.name.clone()));
    table.insert("transport", toml_edit::value(config.transport.name()));

    if let Some(command) = &config.command {
        table.insert("command", toml_edit::value(command.clone()));
    }
    insert_string_array(&mut table, "args", config.args.as_deref());
    if let Some(env) = config.env.as_ref().filter(|env| !env.is_empty()) {
        table.insert(
            "env",
            toml_edit::Item::Value(toml_edit::Value::InlineTable(inline_table(env))),
        );
    }
    if let Some(url) = &config.url {
        table.insert("url", toml_edit::value(url.clone()));
    }
    if let Some(headers) = config
        .headers
        .as_ref()
        .filter(|headers| !headers.is_empty())
    {
        table.insert(
            "headers",
            toml_edit::Item::Value(toml_edit::Value::InlineTable(inline_table(headers))),
        );
    }
    if let Some(helper) = &config.headers_helper {
        table.insert("headers_helper", toml_edit::value(helper.clone()));
    }
    if let Some(permission) = &config.permission {
        table.insert("permission", toml_edit::value(permission.name()));
    }
    if let Some(trust) = config.trust_read_only_hint {
        table.insert("trust_read_only_hint", toml_edit::value(trust));
    }
    if config.disabled.unwrap_or(false) {
        table.insert("disabled", toml_edit::value(true));
    }
    if let Some(required) = config.required {
        table.insert("required", toml_edit::value(required));
    }
    if let Some(auth) = &config.auth {
        table.insert("auth", toml_edit::Item::Table(auth_to_toml(auth)));
    }
    insert_string_array(&mut table, "allowed_tools", config.allowed_tools.as_deref());
    insert_string_array(
        &mut table,
        "disabled_tools",
        config.disabled_tools.as_deref(),
    );
    insert_string_array(
        &mut table,
        "eager_load_tools",
        config.eager_load_tools.as_deref(),
    );
    if let Some(deferral) = config.deferral {
        table.insert("deferral", toml_edit::value(deferral.name()));
    }
    if let Some(permissions) = config.tool_permissions.as_ref()
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

/// An entry's table rendered as the root of a document of its own: a table on its own prints its
/// keys and not the sub-tables under it, which an `auth` block is.
pub(crate) fn render_entry(table: &toml_edit::Table) -> String {
    let mut standalone = toml_edit::DocumentMut::new();
    *standalone.as_table_mut() = table.clone();
    standalone.to_string()
}

/// Whether two entries are one entry as `config.toml` states it: equal as the file would be
/// written from each, since a door may hand the manager a copy that says what the writer leaves
/// unsaid, a `false` or an empty list, and the file reads that back as unsaid.
pub(crate) fn same_entry(a: &McpServerConfig, b: &McpServerConfig) -> bool {
    render_entry(&server_config_to_table(a)) == render_entry(&server_config_to_table(b))
}

/// `map` as an inline table, keys in order so the TOML diff is review-friendly.
fn inline_table(map: &std::collections::HashMap<String, String>) -> toml_edit::InlineTable {
    let mut table = toml_edit::InlineTable::new();
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    for key in keys {
        table.insert(key, toml_edit::Value::from(map[key].as_str()));
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
/// callers print their own user-facing line. Used by both `remove_server` (user-invoked) and
/// `run_add`'s auto-login rollback path (on OAuth failure after the config entry has already been
/// written).
///
/// A missing config entry is only an error when there is nothing else to remove either. This is the
/// one command that deletes an MCP credential as part of deleting a server, so refusing on a name
/// that is absent from `config.toml` would strand the OAuth bundle of a server whose entry was
/// deleted by hand: `mcp list` reports it, and nothing else would clear it.
async fn purge_server(name: &str, token_store: &TokenStore) -> Result<Purged> {
    // The read and the write are one critical section, and the lock ends with it: never held
    // across an await, since its depth is a thread-local and a task resuming on another worker
    // would leave it wrong (see `ConfigFileLock`). The store and the provider are reached after.
    let (removed_from_config, path, known) = {
        let (_config_lock, path, mut document) = super::account::open_document()?;
        let before = document.to_string();
        let removed = document
            .get_mut("mcp")
            .and_then(|m| m.as_table_mut())
            .and_then(|t| t.get_mut("servers"))
            .and_then(|s| s.as_array_of_tables_mut())
            .is_some_and(|servers| {
                let original_len = servers.len();
                servers.retain(|entry| entry.get("name").and_then(|v| v.as_str()) != Some(name));
                servers.len() != original_len
            });
        if removed {
            crate::fs::write_file_atomic(&path, &reparsed_edit(&before, &document)?)
                .map_err(|error| MekaError::io(format!("failed to write config: {error}")))?;
        }
        (removed, path, server_names(&document))
    };

    if !removed_from_config {
        if !token_store.has_mcp_credentials(name).await? {
            return Err(usage_err(crate::text::unknown_name(
                "MCP server",
                name,
                known,
            )));
        }
        clear_server_state(name, token_store).await?;
        return Ok(Purged::CredentialsOnly);
    }

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

/// Remove a server: delete the entry from config.toml, revoke an OAuth grant at the provider
/// where it can be, clear every credential stored under the name. `meka mcp remove` and
/// `DELETE /v1/mcp/{name}`.
pub(crate) async fn remove_server(name: &str, token_store: &TokenStore) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    async fn memory_token_store() -> TokenStore {
        crate::store::Store::for_test().await.token_store()
    }

    /// The edit door reaches only the entry it changes: a key meka does not know elsewhere in the
    /// file, which a shell is repairing, does not stop `meka mcp enable|disable` from touching the
    /// server, as the reference promises of every editor; the entry's own sub-tables are read.
    #[tokio::test]
    async fn an_edit_tolerates_an_unknown_key_elsewhere_in_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[display]\nbogus_key = 1\n\n[[mcp.servers]]\nname = \"x\"\ntransport = \"http\"\n\
             url = \"https://example.test/mcp\"\n\n[mcp.servers.auth]\ntype = \"oauth\"\n",
        )
        .expect("write config");
        let store = memory_token_store().await;

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set → run → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = edit_server(
            "x",
            &ServerChanges {
                disabled: Patch::Set(true),
                ..ServerChanges::default()
            },
            &store,
        )
        .await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        let config = result.expect("the entry is edited whatever the rest of the file holds");
        assert_eq!(config.disabled, Some(true));
        assert!(
            matches!(config.auth, Some(McpAuthConfig::OAuth { .. })),
            "the entry's own sub-table is read"
        );
        let written = std::fs::read_to_string(dir.path().join("config.toml")).expect("read");
        assert!(
            written.contains("disabled = true") && written.contains("bogus_key = 1"),
            "{written}"
        );

        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let removed = remove_server("x", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        removed.expect("the entry is removed whatever the rest of the file holds");
        let written = std::fs::read_to_string(dir.path().join("config.toml")).expect("read");
        assert!(
            !written.contains("[[mcp.servers]]") && written.contains("bogus_key = 1"),
            "{written}"
        );
    }

    /// An http server with nothing but a URL, as `meka mcp add <name> --url` writes one.
    fn server_named(name: &str) -> McpServerConfig {
        let mut config = McpServerConfig::for_test(name);
        config.url = Some("https://example.test/mcp".to_string());
        config
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

        let error = store_server_secret(
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
        let error = store_server_secret(
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

        let error = store_server_secret(
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

    /// A stdio server has no HTTP request to attach a bearer to. Its secrets are `env`, which is
    /// config, and the error says so rather than storing something nothing will ever read.
    #[tokio::test]
    async fn a_secret_is_refused_for_a_stdio_server() {
        let store = memory_token_store().await;
        let stdio: McpServerConfig =
            toml::from_str("name = \"pg\"\ntransport = \"stdio\"\ncommand = \"pg-mcp\"\n")
                .expect("the fixture parses");

        let error = store_server_secret(
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
        let error = store_server_secret(
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

        logout_server(std::slice::from_ref(&server_named("api")), &store, "api")
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
        let result = remove_server("retired", &store).await;
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
        let result = remove_server("typo", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("removing a name that exists nowhere must fail"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("no MCP server named 'typo'"), "{error}");
    }

    fn http(name: &str) -> McpServerConfig {
        let mut config = McpServerConfig::for_test(name);
        config.transport = McpTransport::Http;
        config.url = Some(format!("http://{name}.example/mcp"));
        config.command = None;
        config.args = None;
        config.env = None;
        config
    }

    fn stdio(name: &str) -> McpServerConfig {
        let mut config = McpServerConfig::for_test(name);
        config.transport = McpTransport::Stdio;
        config.url = None;
        config.headers = None;
        config.command = Some("cmd".to_string());
        config
    }

    /// An http entry carries none of a stdio server's keys and a stdio entry none of an http
    /// server's, each refused on its own.
    #[test]
    fn a_stray_key_is_refused_on_its_own_for_either_transport() {
        let pair = |key: &str, value: &str| {
            std::collections::HashMap::from([(key.to_string(), value.to_string())])
        };
        let mut with_command = http("x");
        with_command.command = Some("cmd".to_string());
        let mut with_args = http("x");
        with_args.args = Some(vec!["a".to_string()]);
        let mut with_env = http("x");
        with_env.env = Some(pair("K", "v"));
        let mut with_url = stdio("x");
        with_url.url = Some("http://x.example".to_string());
        let mut with_headers = stdio("x");
        with_headers.headers = Some(pair("H", "v"));
        for (key, config) in [
            ("command", with_command),
            ("args", with_args),
            ("env", with_env),
            ("url", with_url),
            ("headers", with_headers),
        ] {
            let error = validate_server_shape(&config).expect_err(key);
            assert!(error.to_string().contains(key), "{key}: {error}");
        }
    }

    /// The entry written carries its `env` and `headers` as inline tables, keys in order, and
    /// leaves an empty map out.
    #[test]
    fn the_written_entry_carries_its_env_and_headers() {
        let mut config = http("x");
        config.headers = Some(std::collections::HashMap::from([
            ("B".to_string(), "2".to_string()),
            ("A".to_string(), "1".to_string()),
        ]));
        let rendered = server_config_to_table(&config).to_string().replace(' ', "");
        assert!(rendered.contains("headers={A=\"1\",B=\"2\"}"), "{rendered}");
        let mut config = stdio("y");
        config.env = Some(std::collections::HashMap::from([(
            "K".to_string(),
            "v".to_string(),
        )]));
        let rendered = server_config_to_table(&config).to_string().replace(' ', "");
        assert!(rendered.contains("env={K=\"v\"}"), "{rendered}");
        let mut config = http("z");
        config.headers = Some(std::collections::HashMap::new());
        let rendered = server_config_to_table(&config).to_string();
        assert!(!rendered.contains("headers"), "{rendered}");
    }

    /// A logout revokes at the http server of that name and no other, and at no stdio server,
    /// which holds no grant.
    #[test]
    fn a_logout_revokes_at_the_named_http_server_alone() {
        let servers = [http("a"), http("b"), stdio("c")];
        assert_eq!(
            revokes_at(&servers, "a").map(|server| server.name.as_str()),
            Some("a")
        );
        assert_eq!(
            revokes_at(&servers, "b").map(|server| server.name.as_str()),
            Some("b")
        );
        assert!(revokes_at(&servers, "c").is_none());
        assert!(revokes_at(&servers, "d").is_none());
    }
}
