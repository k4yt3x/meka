//! The installation as `config.toml` and the store hold it, managed over HTTP: the accounts, the
//! profiles and the MCP servers, the credentials stored for them, and their logins.
//!
//! Every write runs through the host's one write ([`crate::host::SharedDeps::write_installation`]):
//! under one lock the file is parsed once and refused when it could not be served, the handler
//! decides on that parse and calls the door `meka account`, `meka profile` or `meka mcp` uses,
//! keeping the file's comments and order and refusing in the same words, and the file is then
//! served whole, so the profile a client adds is the one its next `POST /v1/sessions` runs on,
//! without a restart. A browser login is a write whose input arrives late: it begins against the
//! file as written and commits through the same write, refused when what it began against has
//! changed meanwhile.
//!
//! Under `config:r` and `config:w`, the installation's own pair: a token that runs turns has no
//! business rebinding which account they bill.

use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    host::http::{
        errors::{ErrorKind, ProblemDetail},
        scope::{self},
        state::ServerState,
    },
    installation::{Patch, account, profile},
    view::{AccountView, ProfileDetailView, ProfileView},
};

/// What a door's error is on the wire, for every door: a refusal of what was sent is the 422
/// every refusal of a body is, worded by [`ProblemDetail::for_error`] with this API's remedy
/// where the shell's sentence names a command; a fault of the file or the store is the sanitized
/// 500 with the text in the log under `context`, where the operator reads it. An unknown name on
/// the path is a 404 ahead of the door, from the handler holding the parse.
fn door_problem(
    state: &ServerState,
    context: &str,
    error: crate::error::MekaError,
) -> ProblemDetail {
    if error.kind() == ErrorKind::Internal {
        return ProblemDetail::internal_sanitized(context, error);
    }
    ProblemDetail::for_error(&error, state.config.relay_provider_errors)
}

/// The 422 for a body the door would not be asked about.
fn refused(sentence: impl Into<String>) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::InvalidBody,
        StatusCode::UNPROCESSABLE_ENTITY,
        sentence,
    )
}

/// The 502 for a login the other side did not see through.
fn upstream(sentence: impl Into<String>) -> ProblemDetail {
    ProblemDetail::new(ErrorKind::Provider, StatusCode::BAD_GATEWAY, sentence)
}

/// The 503 for a login begun or committing after the server began shutting down.
fn shutting_down() -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::ShuttingDown,
        StatusCode::SERVICE_UNAVAILABLE,
        "the server is shutting down",
    )
}

/// The 404 for a name no account, profile or server has, in the words every door refuses one
/// with.
fn not_found(
    noun: &str,
    name: &str,
    known: impl IntoIterator<Item = impl AsRef<str>>,
) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::NotFound,
        StatusCode::NOT_FOUND,
        crate::text::unknown_name(noun, name, known),
    )
}

/// Write the installation over HTTP: the host's one write
/// ([`crate::host::SharedDeps::write_installation`]), with `edit` deciding on the parse it is
/// handed and answering in this API's terms, and every login pending on an account or a server
/// the served file no longer names ended after it, since its commit would be refused.
async fn write_installation<T>(
    state: &ServerState,
    edit: impl AsyncFnOnce(&crate::config::ConfigFile) -> Result<T, ProblemDetail>,
) -> Result<(T, crate::host::assembly::ServedServers), ProblemDetail> {
    use crate::host::assembly::{InstallationFailure, ServeFailure};
    match state.shared.write_installation(edit).await {
        Ok(written) => {
            end_logins_on_subjects_gone(state).await;
            Ok(written)
        }
        Err(InstallationFailure::Unservable(unservable)) => Err(refused(
            unservable.worded(crate::host::http::errors::api_wording),
        )),
        Err(InstallationFailure::Edit(problem)) => Err(problem),
        Err(InstallationFailure::Unserved(ServeFailure::Unservable(unservable))) => Err(refused(
            unservable.after_the_write(crate::host::http::errors::api_wording),
        )),
        Err(InstallationFailure::Unserved(ServeFailure::Fault(error))) => Err(
            ProblemDetail::internal_sanitized("failed to serve config.toml after the write", error),
        ),
    }
}

/// The file as written, for a door that begins something it commits through
/// [`write_installation`] later: refused as that write would refuse it.
fn installation_as_written(
    state: &ServerState,
) -> Result<crate::config::ConfigFile, ProblemDetail> {
    state
        .shared
        .installation_as_written()
        .map_err(|unservable| refused(unservable.worded(crate::host::http::errors::api_wording)))
}

/// Where the installation is and how what this server serves stands against `config.toml`, as
/// `GET /v1/config` answers.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ConfigResponse {
    /// `config.toml`'s path; absent when no config directory can be determined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) config_path: Option<String>,
    /// The data directory, where the store lives; absent when none can be determined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) data_dir: Option<String>,
    /// When this process last read `config.toml` whole, RFC 3339: its start, the last
    /// `POST /v1/config/reload`, or its own last write, each of which reads the whole file.
    pub(crate) config_loaded_at: String,
    /// When `config.toml` was last written, RFC 3339; absent when there is no file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) config_modified_at: Option<String>,
    /// Whether `config.toml` has changed since this process last read it whole: an edit made in a
    /// shell, which `POST /v1/config/reload` applies. A write of the server's own reads the file
    /// again, so it leaves this `false`. Compared on the file's modification time, so it holds
    /// where the two instants above, spelled to the second, read equal.
    pub(crate) config_changed: bool,
    /// The sections of `config.toml` that state something else than the file this server
    /// started on and are read at startup alone, so neither a reload nor a write applies them and
    /// only a restart does: `permissions`, `shell`, `[mcp]`'s own keys as `mcp`, and the rest.
    pub(crate) restart_required: Vec<&'static str>,
}

/// `GET /v1/config`: where the installation is, when `config.toml` was last read whole and
/// written, whether an edit awaits `POST /v1/config/reload`, and what only a restart applies.
#[utoipa::path(
    get,
    path = "/v1/config",
    tag = "installation",
    responses(
        (status = 200, description = "Where the installation is and how this server stands against it", body = ConfigResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:r"]))
)]
pub(crate) async fn config(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigRead>,
) -> Json<ConfigResponse> {
    // The record ahead of the file: a serving that lands between the two then reads as a change
    // the record predates, where the other order would call a file just read whole changed.
    let read = state.shared.whole_read();
    let modified = crate::config::config_file_modified();
    Json(ConfigResponse {
        config_path: crate::paths::config_file_path().map(|path| path.display().to_string()),
        data_dir: crate::paths::meka_data_dir().map(|path| path.display().to_string()),
        config_loaded_at: super::info::rfc3339(read.at),
        config_modified_at: modified
            .map(|modified| super::info::rfc3339(chrono::DateTime::<chrono::Utc>::from(modified))),
        config_changed: modified != read.modified,
        restart_required: state.shared.sections_awaiting_restart(),
    })
}

/// What `POST /v1/config/reload` applied.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ReloadResponse {
    /// RFC 3339, when this read of `config.toml` was made; `GET /v1/config` reports the same
    /// instant from now on.
    pub(crate) config_loaded_at: String,
    /// MCP servers the file names that were not running, started.
    pub(crate) servers_added: Vec<String>,
    /// MCP servers whose connection or credential changed, restarted on the file's entry.
    pub(crate) servers_restarted: Vec<String>,
    /// MCP servers whose tool policy alone changed, given it on the connection they hold.
    pub(crate) servers_reconfigured: Vec<String>,
    /// MCP servers the file no longer names, taken out of the running manager.
    pub(crate) servers_removed: Vec<String>,
    /// Sections of `config.toml` that state something else than the file this server started
    /// on and are read at startup alone, so only a restart applies them; as `GET /v1/config`
    /// reports.
    pub(crate) restart_required: Vec<&'static str>,
}

/// `POST /v1/config/reload`: read `config.toml` again and serve what a write over HTTP would
/// have: the accounts and profiles, the MCP servers brought to the file's entries, and the standing
/// instructions. Every other section is read at startup alone, and the answer names those that
/// changed.
#[utoipa::path(
    post,
    path = "/v1/config/reload",
    tag = "installation",
    responses(
        (status = 200, description = "What the reload applied, and what waits for a restart", body = ReloadResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 422, description = "`config.toml` fails to parse, or holds anything startup would refuse; nothing was applied", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn reload_config(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
) -> Result<Json<ReloadResponse>, ProblemDetail> {
    let ((), servers) = write_installation(&state, async |_| Ok(())).await?;
    tracing::info!(
        "reloaded config.toml via HTTP: {added} MCP server(s) added, {restarted} restarted, \
         {reconfigured} reconfigured, {removed} removed",
        added = servers.added.len(),
        restarted = servers.restarted.len(),
        reconfigured = servers.reconfigured.len(),
        removed = servers.removed.len()
    );
    let read_at = state.shared.whole_read().at;
    Ok(Json(ReloadResponse {
        // As `GET /v1/config` spells it, so the two instants compare equal.
        config_loaded_at: super::info::rfc3339(read_at),
        servers_added: servers.added,
        servers_restarted: servers.restarted,
        servers_reconfigured: servers.reconfigured,
        servers_removed: servers.removed,
        restart_required: state.shared.sections_awaiting_restart(),
    }))
}

// ----- Accounts ----------------------------------------------------------------------------------

/// What `GET /v1/accounts` answers.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct AccountsResponse {
    pub(crate) accounts: Vec<AccountView>,
    /// Names a credential is stored under that no account claims, which `DELETE
    /// /v1/accounts/{name}` clears. Absent when there are none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) orphaned_credentials: Vec<String>,
}

/// `GET /v1/accounts`: every account, with whether a credential is stored under it.
#[utoipa::path(
    get,
    path = "/v1/accounts",
    tag = "installation",
    responses(
        (status = 200, description = "Configured accounts", body = AccountsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:r"]))
)]
pub(crate) async fn accounts(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigRead>,
) -> Result<Json<AccountsResponse>, ProblemDetail> {
    let catalog = state.shared.providers.catalog();
    let token_store = state.shared.store.token_store();
    let accounts = account::account_views(&token_store, &catalog.accounts)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to list accounts", error))?;
    // Claimed by the file as well as by the served catalog: an account added in a shell and not
    // yet served is a configured account, which `DELETE` would remove whole, not a stray secret.
    let mut claimed = catalog.accounts.clone();
    if let Ok(config_file) = crate::config::load_config_file_or_err() {
        claimed.extend(config_file.accounts);
    }
    let orphaned_credentials = account::orphaned_accounts(&token_store, &claimed)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to list credentials", error))?;
    Ok(Json(AccountsResponse {
        accounts,
        orphaned_credentials,
    }))
}

/// `GET /v1/accounts/{name}`: one account as the server serves it, with whether a credential is
/// stored under it.
#[utoipa::path(
    get,
    path = "/v1/accounts/{name}",
    tag = "installation",
    params(("name" = String, Path, description = "Account name")),
    responses(
        (status = 200, description = "The account", body = AccountView),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such account", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:r"]))
)]
pub(crate) async fn get_account(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigRead>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<AccountView>, ProblemDetail> {
    let catalog = state.shared.providers.catalog();
    let token_store = state.shared.store.token_store();
    Ok(Json(account_view(&catalog, &token_store, &name).await?))
}

/// The body of `POST /v1/accounts`: the `[accounts.<name>]` settings, and the key for a backend
/// that takes one.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateAccountRequest {
    pub(crate) name: String,
    /// The backend, as `[accounts.<name>].backend` names it.
    pub(crate) backend: String,
    /// The API base URL; omitted, the backend's own applies.
    #[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]
    pub(crate) base_url: Option<String>,
    /// The OAuth token endpoint, read by the two subscription backends alone.
    #[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]
    pub(crate) oauth_token_url: Option<String>,
    /// The OAuth client id, read by the two subscription backends alone.
    #[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]
    pub(crate) client_id: Option<String>,
    /// Whether to send the interleaved-thinking beta, read by `anthropic-messages` alone.
    pub(crate) interleaved_thinking: Option<bool>,
    /// The key an API-key backend presents, stored in the database and never in `config.toml`.
    /// Required on such a backend, refused on one that logs in through the browser.
    #[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]
    pub(crate) api_key: Option<String>,
}

/// `POST /v1/accounts`: add an account, with its key when its backend takes one.
///
/// A subscription backend's account is added without a credential and logged in through
/// `POST /v1/accounts/{name}/login`, as `meka account add` would have run the flow.
#[utoipa::path(
    post,
    path = "/v1/accounts",
    tag = "installation",
    request_body = CreateAccountRequest,
    responses(
        (status = 201, description = "Account added", body = AccountView),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body: a name in use, an unknown backend, a setting the backend never reads, or a key where none is taken or none where one is; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn create_account(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<AccountView>), ProblemDetail> {
    let body: CreateAccountRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("account", error))?;
    let name = body.name.as_str();
    let backend =
        account::validate_backend(&body.backend).map_err(|error| refused(error.sentence()))?;
    let settings = account::OAuthSettings {
        oauth_token_url: body.oauth_token_url,
        client_id: body.client_id,
    };
    let credential = body.api_key.map(crate::store::AuthCredential::ApiKey);
    let token_store = state.shared.store.token_store();
    write_installation(&state, async |_| {
        account::add_account(
            &account::NewAccount {
                name,
                backend,
                base_url: body.base_url.as_deref(),
                settings: &settings,
                interleaved_thinking: body.interleaved_thinking,
            },
            credential.as_ref(),
            &token_store,
        )
        .await
        .map_err(|error| door_problem(&state, "failed to add the account", error))
    })
    .await?;
    tracing::info!("added account '{name}' via HTTP");
    let catalog = state.shared.providers.catalog();
    let view = account_view(&catalog, &token_store, name).await?;
    Ok((StatusCode::CREATED, Json(view)))
}

/// One account's view from the served catalog, or the 404 naming the accounts it holds.
async fn account_view(
    catalog: &crate::config::ProfileCatalog,
    token_store: &crate::store::TokenStore,
    name: &str,
) -> Result<AccountView, ProblemDetail> {
    let Some(config) = catalog.accounts.get(name) else {
        return Err(not_found("account", name, catalog.accounts.keys()));
    };
    let mut views = account::account_views(
        token_store,
        &std::collections::BTreeMap::from([(name.to_string(), config.clone())]),
    )
    .await
    .map_err(|error| ProblemDetail::internal_sanitized("failed to read the account", error))?;
    views
        .pop()
        .ok_or_else(|| ProblemDetail::internal_sanitized("the account view came back empty", name))
}

/// The body of `PATCH /v1/accounts/{name}`.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct RenameRequest {
    /// The name the account moves to.
    pub(crate) name: String,
}

/// `PATCH /v1/accounts/{name}`: rename an account, moving its credential and every profile on it.
///
/// A rename is all an account takes: its other settings are what a login was made against, so
/// another endpoint or backend is another account.
#[utoipa::path(
    patch,
    path = "/v1/accounts/{name}",
    tag = "installation",
    params(("name" = String, Path, description = "Account name")),
    request_body = RenameRequest,
    responses(
        (status = 200, description = "The account under its new name", body = AccountView),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such account", body = ProblemDetail),
        (status = 409, description = "A session or sub-agent running on this server may run on the account, or another meka is refreshing its credential, which refreshes under its name", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid new name, or one in use; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn rename_account(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
    raw_body: Bytes,
) -> Result<Json<AccountView>, ProblemDetail> {
    let body: RenameRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("account patch", error))?;
    let token_store = state.shared.store.token_store();
    write_installation(&state, async |config_file| {
        let Some(config) = config_file.accounts.get(&name) else {
            return Err(not_found("account", &name, config_file.accounts.keys()));
        };
        refuse_renaming_under_a_session(&state, &name, config).await?;
        account::rename_account(&name, &body.name, &token_store)
            .await
            .map_err(|error| door_problem(&state, "failed to rename the account", error))
    })
    .await?;
    tracing::info!(
        "renamed account '{name}' to '{new}' via HTTP",
        new = body.name
    );
    let catalog = state.shared.providers.catalog();
    let view = account_view(&catalog, &token_store, &body.name).await?;
    Ok(Json(view))
}

/// Refuse renaming `name` while a session resident here, or a sub-agent running here, may run on
/// it and its credential refreshes. A session keeps the provider it was built with, and a
/// subscription's credential refreshes under the account's name: renamed under the session, the
/// next refresh would write to a row that is gone and lose the token the issuer just rotated. A
/// session runs on its profile and on any profile that profile falls back to, so the whole chain
/// counts. An API key is held by its provider whole and has no row to refresh, and an account
/// with no credential stored has none either.
async fn refuse_renaming_under_a_session(
    state: &ServerState,
    name: &str,
    config: &crate::config::AccountConfig,
) -> Result<(), ProblemDetail> {
    let refreshes = account::validate_backend(&config.backend).is_ok_and(|backend| {
        !matches!(
            account::credential_kind(backend),
            account::CredentialKind::ApiKey
        )
    });
    if !refreshes {
        return Ok(());
    }
    let stored = state
        .shared
        .store
        .token_store()
        .list_credential_accounts()
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to read the stored credentials", error)
        })?;
    if !stored.iter().any(|account| account == name) {
        return Ok(());
    }
    // The served catalog, since it is what a running provider was built from.
    let catalog = state.shared.providers.catalog();
    let mut running: Vec<uuid::Uuid> = state.sessions.read().await.keys().copied().collect();
    running.extend(state.shared.running_subagents.running_ids());
    let mut on_the_account = 0;
    for id in running {
        let recorded = state
            .shared
            .store
            .recorded_profile(id)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to read a session's profile", error)
            })?;
        if recorded.is_some_and(|profile| chain_may_bill(&catalog, &profile, name)) {
            on_the_account += 1;
        }
    }
    if on_the_account == 0 {
        return Ok(());
    }
    Err(ProblemDetail::new(
        ErrorKind::AccountInUse,
        StatusCode::CONFLICT,
        format!(
            "{on_the_account} session(s) open on this server may run on '{name}'; rename it once \
             they have closed"
        ),
    ))
}

/// Whether a session on `profile` may run on `account`: the profile or one it falls back to bills
/// it, or one of them is no longer configured, since a session on a profile since removed still
/// runs on the provider it was built with and cannot be shown not to bill it.
fn chain_may_bill(catalog: &crate::config::ProfileCatalog, profile: &str, account: &str) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    let mut pending = vec![profile.to_string()];
    while let Some(current) = pending.pop() {
        if !seen.insert(current.clone()) {
            continue;
        }
        let Some(config) = catalog.profiles.get(&current) else {
            return true;
        };
        if config.account == account {
            return true;
        }
        pending.extend(config.fallbacks.iter().cloned());
    }
    false
}

/// `DELETE /v1/accounts/{name}`: remove an account and the credential stored under its name,
/// or a credential alone when no account claims it.
#[utoipa::path(
    delete,
    path = "/v1/accounts/{name}",
    tag = "installation",
    params(("name" = String, Path, description = "Account name")),
    responses(
        (status = 204, description = "Account removed"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such account, and no credential under the name", body = ProblemDetail),
        (status = 422, description = "A profile names the account; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn delete_account(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    let token_store = state.shared.store.token_store();
    write_installation(&state, async |config_file| {
        if !config_file.accounts.contains_key(&name) {
            // A credential no account claims is this door's to clear too, as it is `meka
            // account remove`'s, so the name is unknown only when neither side has it.
            let orphaned = account::orphaned_accounts(&token_store, &config_file.accounts)
                .await
                .map_err(|error| {
                    ProblemDetail::internal_sanitized("failed to list credentials", error)
                })?;
            if !orphaned.contains(&name) {
                let configured: Vec<String> = config_file.accounts.keys().cloned().collect();
                return Err(ProblemDetail::new(
                    ErrorKind::NotFound,
                    StatusCode::NOT_FOUND,
                    account::unknown_account(&name, &configured, &orphaned),
                ));
            }
        }
        account::remove_account(&name, &token_store)
            .await
            .map_err(|error| door_problem(&state, "failed to remove the account", error))
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// The body of `POST /v1/accounts/{name}/login`.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct LoginRequest {
    /// The key an API-key backend presents, replacing the one stored. Absent on a subscription
    /// backend, whose login runs in the browser.
    #[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]
    pub(crate) api_key: Option<String>,
}

/// How long a browser login begun over HTTP waits for the person: the life of its `login_id`,
/// and of the loopback listener a `chatgpt-subscription` login holds.
const BROWSER_LOGIN_LIFETIME: std::time::Duration = std::time::Duration::from_secs(600);

/// How long a login begun over HTTP may take to produce its URL: an account's is immediate, an
/// MCP server's follows the discovery and the metadata fetch at the server.
const LOGIN_URL_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

/// How long a login being ended may take to end on its own before it is aborted: enough for a
/// flow at an await to see the cancel and close the probe connection it opened.
const LOGIN_END_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// What finished a browser login: `Ok` once the credential is stored and served, `Err` with what
/// the request completing it is answered with; `None` while it waits.
type LoginOutcome = Arc<std::sync::Mutex<Option<Result<(), ProblemDetail>>>>;

/// The person's side of a browser login run over HTTP, an account's or an MCP server's: the URL
/// and what completes the login are kept for the request that began it to answer with, and the
/// pasted line arrives from the request that completes it. The screen and the terminal have
/// their own.
struct HttpLoginPrompt {
    authorization_url: std::sync::Mutex<Option<String>>,
    /// What completes the login, once the flow says: whether it holds a listener, and what a
    /// paste is read as.
    completion: std::sync::Mutex<Option<crate::oauth::Completion>>,
    /// Why the last paste was refused, for the request that posted it; taken by that request.
    paste_refusal: std::sync::Mutex<Option<String>>,
    /// Woken on every change above, and when the login ends however it ends.
    changed: tokio::sync::Notify,
    pasted: std::sync::Mutex<Option<String>>,
    paste_ready: tokio::sync::Notify,
    cancel: tokio_util::sync::CancellationToken,
    /// Set once the flow holds its credential and its commit has begun: from then on the commit
    /// decides how the login ends, so nothing that ends logins settles this one for it.
    committing: std::sync::atomic::AtomicBool,
}

impl HttpLoginPrompt {
    /// Wake every request waiting on this login: the ones registered now, and the one that
    /// checked just before registering, which the stored permit reaches.
    fn wake(&self) {
        self.changed.notify_waiters();
        self.changed.notify_one();
    }
}

#[async_trait::async_trait]
impl crate::oauth::LoginPrompt for HttpLoginPrompt {
    fn authorize_at(&self, url: &str) {
        *crate::sync::lock(&self.authorization_url) = Some(url.to_string());
        self.wake();
    }

    fn awaiting(&self, completion: crate::oauth::Completion) {
        *crate::sync::lock(&self.completion) = Some(completion);
        self.wake();
    }

    fn accepts_paste(&self) -> bool {
        true
    }

    async fn read_pasted_line(&self) -> std::result::Result<Option<String>, String> {
        loop {
            if let Some(line) = crate::sync::lock(&self.pasted).take() {
                return Ok(Some(line));
            }
            // `notify_one` keeps a permit when nothing waits, so a paste landing between the
            // check above and this wait is not lost.
            tokio::select! {
                () = self.paste_ready.notified() => {}
                () = self.cancel.cancelled() => {
                    return Err("the login was replaced or canceled".to_string());
                }
            }
        }
    }

    fn paste_refused(&self, reason: &str) {
        *crate::sync::lock(&self.paste_refusal) = Some(reason.to_string());
        self.wake();
    }
}

/// A browser login begun over HTTP, waiting for the person to finish it in the browser and for
/// the client to bring back what the browser gave, or for a loopback listener to be sent it.
pub(crate) struct PendingLogin {
    /// The account or the MCP server being logged in.
    subject: String,
    /// An account's login rather than an MCP server's: the two share the map, and one name may
    /// be both an account's and a server's.
    account: bool,
    /// The flow's side of its conversation with the person.
    prompt: Arc<HttpLoginPrompt>,
    expires_at: std::time::Instant,
    /// Written by the flow's task when it ends.
    outcome: LoginOutcome,
    /// Ends the flow's wait for a paste when the login is replaced or its subject is gone.
    cancel: tokio_util::sync::CancellationToken,
    /// The task running the flow, which holds its listener: aborted and waited out by the login
    /// replacing this one, so the port it held is free to bind.
    task: tokio::task::JoinHandle<()>,
}

/// Where a login stands, from its outcome.
fn login_status_of(outcome: &LoginOutcome) -> LoginStatus {
    match crate::sync::lock(outcome).clone() {
        None => LoginStatus {
            status: "pending",
            error: None,
        },
        Some(Ok(())) => LoginStatus {
            status: "completed",
            error: None,
        },
        Some(Err(problem)) => LoginStatus {
            status: "failed",
            error: Some(problem.detail.unwrap_or(problem.title)),
        },
    }
}

/// Whether `login`'s commit has begun and not yet settled it, which every path that ends logins
/// leaves to the commit: settled for it, the login would read as failed while its credential is
/// stored.
fn is_committing(login: &PendingLogin) -> bool {
    login
        .prompt
        .committing
        .load(std::sync::atomic::Ordering::Acquire)
        && crate::sync::lock(&login.outcome).is_none()
}

/// Settle a login that will not complete, so a request waiting on it is answered with `reason`
/// rather than left to the login's lifetime.
fn settle_login(login: &PendingLogin, reason: &str) {
    let mut outcome = crate::sync::lock(&login.outcome);
    if outcome.is_none() {
        *outcome = Some(Err(refused(reason)));
    }
    drop(outcome);
    login.prompt.wake();
}

/// End a login: its flow is told to stop, which it does at its next cancel-aware await and within
/// its own bounds either way, and a request waiting on it is answered with `reason`. Not aborted:
/// an abort skips the close of the probe connection an MCP login opened.
fn end_login(login: &PendingLogin, reason: &str) {
    login.cancel.cancel();
    settle_login(login, reason);
}

/// The pending logins, by `login_id`. One per subject: beginning another replaces it.
pub(crate) type PendingLogins =
    Arc<std::sync::Mutex<std::collections::HashMap<String, PendingLogin>>>;

/// What a caller is told of an exchange the authorization server rejected: its own body, which
/// `[serve] relay_provider_errors` decides whether a caller sees, with the whole in the log.
fn exchange_rejection(relay: bool, account: &str, error: &anyhow::Error) -> String {
    tracing::warn!("the browser login for '{account}' failed: {error}");
    if relay {
        error.to_string()
    } else {
        "the login did not complete; see the server log".to_string()
    }
}

/// Drop every pending login past its lifetime, ending the flow each ran.
fn sweep_expired_logins(pending: &mut std::collections::HashMap<String, PendingLogin>) {
    let now = std::time::Instant::now();
    pending.retain(|_, login| {
        if login.expires_at > now || is_committing(login) {
            return true;
        }
        end_login(login, "the login lapsed");
        false
    });
}

/// Drop the pending login on `subject`, an account's when `account` and an MCP server's
/// otherwise, and wait its task out, for the login replacing it, which binds the same port.
async fn forget_pending_logins(state: &ServerState, subject: &str, account: bool) {
    let removed: Vec<PendingLogin> = {
        let mut pending = crate::sync::lock(&state.pending_logins);
        sweep_expired_logins(&mut pending);
        let gone: Vec<String> = pending
            .iter()
            .filter(|(_, login)| {
                login.subject == subject && login.account == account && !is_committing(login)
            })
            .map(|(login_id, _)| login_id.clone())
            .collect();
        let removed = gone
            .into_iter()
            .filter_map(|login_id| pending.remove(&login_id))
            .collect();
        drop(pending);
        removed
    };
    end_logins(removed, "the login was replaced").await;
}

/// End every pending login on an account or an MCP server the file no longer names, after a
/// write: its commit would be refused, so it holds a listener for nothing. Read off the file,
/// which a write replaces whole, rather than off the running manager, which another write may be
/// in the middle of restarting a server on. A file that does not parse ends nothing.
async fn end_logins_on_subjects_gone(state: &ServerState) {
    let Ok(config_file) = crate::config::load_config_file_or_err() else {
        return;
    };
    let servers = config_file.mcp_servers();
    let removed: Vec<PendingLogin> = {
        let mut pending = crate::sync::lock(&state.pending_logins);
        let gone: Vec<String> = pending
            .iter()
            .filter(|(_, login)| !is_committing(login))
            .filter(|(_, login)| {
                if login.account {
                    !config_file.accounts.contains_key(&login.subject)
                } else {
                    !servers.iter().any(|server| server.name == login.subject)
                }
            })
            .map(|(login_id, _)| login_id.clone())
            .collect();
        let removed = gone
            .into_iter()
            .filter_map(|login_id| pending.remove(&login_id))
            .collect();
        drop(pending);
        removed
    };
    end_logins(removed, "the login's account or server was removed").await;
}

/// Drop every pending login and wait its task out, for a server shutting down: a completion
/// request waiting on one would otherwise hold the drain for as long as its flow takes, an
/// exchange in flight included.
pub(crate) async fn forget_every_pending_login(state: &ServerState) {
    let removed: Vec<PendingLogin> = crate::sync::lock(&state.pending_logins)
        .drain()
        .map(|(_, login)| login)
        .collect();
    end_logins(removed, "the server is shutting down").await;
}

/// End each of `removed` with `reason` and wait its task out, leaving one whose commit has begun
/// to that commit.
async fn end_logins(removed: Vec<PendingLogin>, reason: &str) {
    for login in removed {
        // Its commit ends it, and quickly: settled here, it would read as failed while the
        // credential is stored.
        if is_committing(&login) {
            continue;
        }
        // Canceled first and aborted only after a grace: a flow at an await that sees the cancel
        // ends on its own and closes the probe connection an MCP login opened, where an abort
        // drops it mid-step.
        login.cancel.cancel();
        settle_login(&login, reason);
        let mut task = login.task;
        let ended = match tokio::time::timeout(LOGIN_END_GRACE, &mut task).await {
            Ok(ended) => ended,
            Err(_elapsed) => {
                task.abort();
                task.await
            }
        };
        if let Err(error) = ended
            && !error.is_cancelled()
        {
            tracing::debug!("an earlier login's task ended abnormally: {error}");
        }
    }
}

/// What `POST /v1/accounts/{name}/login` answers when the login runs in the browser.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct LoginStarted {
    /// Names the login to `GET` and `POST /v1/accounts/{name}/login/{login_id}`.
    pub(crate) login_id: String,
    /// The URL the person opens.
    pub(crate) authorization_url: String,
    /// What the person brings back and the client posts as `code`: `code`, the one Anthropic's
    /// page shows after the login, or `callback_url`, the address the browser was sent to.
    pub(crate) completes_with: &'static str,
    /// Whether the server holds the loopback listener the browser is sent back to, so a browser
    /// on the server's own machine completes the login without the client posting anything.
    pub(crate) listening: bool,
    /// When the login lapses, RFC 3339.
    pub(crate) expires_at: String,
}

/// Where a browser login stands, as `GET /v1/accounts/{name}/login/{login_id}` answers.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct LoginStatus {
    /// `pending`, `completed` or `failed`.
    pub(crate) status: &'static str,
    /// Why a `failed` login failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

/// `POST /v1/accounts/{name}/login`: store a credential for an account, replacing the one it has.
///
/// An API-key backend takes the key in the body, as `meka account login` reads it from stdin, and
/// is answered 200 with the account. A subscription backend takes no body: the login runs in the
/// browser, and the answer is 202 with the URL to open and a `login_id` that
/// `POST /v1/accounts/{name}/login/{login_id}` completes with what the browser gave. A
/// `chatgpt-subscription` login also holds the loopback listener the browser is sent back to,
/// when the port is free, so a browser on this machine completes it on its own. Beginning a login
/// on an account that has one pending replaces it.
#[utoipa::path(
    post,
    path = "/v1/accounts/{name}/login",
    tag = "installation",
    params(("name" = String, Path, description = "Account name")),
    request_body = LoginRequest,
    responses(
        (status = 200, description = "Logged in with the key; the account, now authenticated", body = AccountView),
        (status = 202, description = "A browser login begun; open the URL, then complete the login", body = LoginStarted),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such account", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "No key for a backend that takes one, or one for a backend that logs in through the browser; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 502, description = "The login produced no authorization URL in time; the reason is in the meka log", body = ProblemDetail),
        (status = 503, description = "The server is shutting down", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn login_account(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
    raw_body: Bytes,
) -> Result<Response, ProblemDetail> {
    let body: LoginRequest = if raw_body.is_empty() {
        LoginRequest { api_key: None }
    } else {
        serde_json::from_slice(&raw_body)
            .map_err(|error| ProblemDetail::invalid_body("login", error))?
    };
    let token_store = state.shared.store.token_store();
    if let Some(key) = body.api_key {
        let credential = crate::store::AuthCredential::ApiKey(key);
        write_installation(&state, async |config_file| {
            if !config_file.accounts.contains_key(&name) {
                return Err(not_found("account", &name, config_file.accounts.keys()));
            }
            account::store_credential(config_file, &name, &credential, &token_store)
                .await
                .map_err(|error| {
                    door_problem(&state, "failed to store the account credential", error)
                })
        })
        .await?;
        tracing::info!("re-authenticated account '{name}' via HTTP");
        let catalog = state.shared.providers.catalog();
        let view = account_view(&catalog, &token_store, &name).await?;
        return Ok(Json(view).into_response());
    }
    // The login begins against the account as the file states it, which its commit holds it to.
    let config_file = installation_as_written(&state)?;
    let Some(config) = config_file.accounts.get(&name) else {
        return Err(not_found("account", &name, config_file.accounts.keys()));
    };
    let backend = account::validate_backend(&config.backend)
        .map_err(|error| door_problem(&state, "failed to read the account", error))?;
    if matches!(
        account::credential_kind(backend),
        account::CredentialKind::ApiKey
    ) {
        return Err(door_problem(
            &state,
            "failed to begin the login",
            crate::error::MekaError::AccountCredentialKind {
                account: name.clone(),
                backend: backend.name().to_string(),
                key_given: false,
            },
        ));
    }
    match begin_browser_login(&state, &name, backend, config).await? {
        Begun::Browser(started) => Ok((StatusCode::ACCEPTED, Json(started)).into_response()),
        Begun::Completed => {
            let catalog = state.shared.providers.catalog();
            Ok(Json(account_view(&catalog, &token_store, &name).await?).into_response())
        }
    }
}

/// Begin a browser login for `name`, replacing one pending: the flow `meka account login` runs
/// at a terminal, with the request's side as the person's, its credential committed through
/// [`write_installation`] against the terms the account stated as it began.
async fn begin_browser_login(
    state: &ServerState,
    name: &str,
    backend: crate::config::Backend,
    config: &crate::config::AccountConfig,
) -> Result<Begun, ProblemDetail> {
    let began = account::LoginTerms::of(config);
    let client_id = config.client_id.clone();
    let oauth_token_url = config.oauth_token_url.clone();
    let account = name.to_string();
    let relay = state.config.relay_provider_errors;
    let state_for_flow = state.clone();
    begin_login(
        state,
        name,
        true,
        BROWSER_LOGIN_LIFETIME,
        move |prompt| async move {
            let credential = account::browser_login(
                backend,
                client_id.as_deref(),
                oauth_token_url.as_deref(),
                BROWSER_LOGIN_LIFETIME,
                prompt.as_ref(),
            )
            .await
            .map_err(|error| {
                // The authorization server's own words are relayed as `[serve]
                // relay_provider_errors` decides; an exchange that got no answer meka can read is
                // the endpoint's failure, with the reason in the log; every other failure is
                // meka's own sentence about what the person did.
                if error.downcast_ref::<account::ExchangeRejected>().is_some() {
                    refused(exchange_rejection(relay, &account, &error))
                } else if error.downcast_ref::<account::ExchangeFailed>().is_some() {
                    tracing::warn!("the token exchange for '{account}' failed: {error}");
                    upstream("the token exchange got no answer; the reason is in the meka log")
                } else {
                    tracing::warn!("the browser login for '{account}' ended: {error}");
                    refused(error.to_string())
                }
            })?;
            commit_on_its_own_task(&state_for_flow, &prompt, {
                let state = state_for_flow.clone();
                let account = account.clone();
                async move {
                    let token_store = state.shared.store.token_store();
                    write_installation(&state, async |config_file| {
                        account::commit_login(
                            config_file,
                            &account,
                            &began,
                            &credential,
                            &token_store,
                        )
                        .await
                        .map_err(|error| {
                            door_problem(&state, "failed to store the account credential", error)
                        })
                    })
                    .await
                    .map(|_| ())
                }
            })
            .await?;
            tracing::info!("logged in account '{account}' from the browser");
            Ok(())
        },
    )
    .await
}

/// Run a login's commit on a task of its own and answer how it went, marking the login as
/// committing first so nothing that ends logins settles it for the commit: the login's task can be
/// aborted at its grace, and an abort landing inside a write's serving, between a server's
/// removal and its re-adding, would leave the manager without a server the file names. A spawned
/// task runs on when the one awaiting it is aborted. Refused once the server is shutting down,
/// whose drain would not wait for it.
async fn commit_on_its_own_task(
    state: &ServerState,
    prompt: &HttpLoginPrompt,
    commit: impl std::future::Future<Output = Result<(), ProblemDetail>> + Send + 'static,
) -> Result<(), ProblemDetail> {
    if state.shutdown.is_cancelled() {
        return Err(shutting_down());
    }
    prompt
        .committing
        .store(true, std::sync::atomic::Ordering::Release);
    match tokio::spawn(commit).await {
        Ok(committed) => committed,
        Err(error) => Err(ProblemDetail::internal_sanitized(
            "a login's commit ended abnormally",
            error,
        )),
    }
}

/// How a login begun over HTTP stands when its request is answered.
enum Begun {
    /// Waiting for the person: the URL to open and what completes the login.
    Browser(LoginStarted),
    /// Committed already, by a flow that needed no browser: a `client_credentials` server proves
    /// its client with the secret alone.
    Completed,
}

/// Begin a login on `subject` over HTTP: the earlier one on it goes first, `flow` runs as a task
/// with an [`HttpLoginPrompt`] as the person's side, and the answer is the URL and what the flow
/// said completes it, once both are out. A flow that ends first is answered with how it ended,
/// committed or refused, and one that produces no URL in time is given up.
async fn begin_login<F>(
    state: &ServerState,
    subject: &str,
    account: bool,
    lifetime: std::time::Duration,
    flow: impl FnOnce(Arc<HttpLoginPrompt>) -> F,
) -> Result<Begun, ProblemDetail>
where
    F: std::future::Future<Output = Result<(), ProblemDetail>> + Send + 'static,
{
    let noun = if account { "account" } else { "MCP server" };
    // The shutdown ends every pending login once; one begun after would store its credential
    // past the drain.
    if state.shutdown.is_cancelled() {
        return Err(shutting_down());
    }
    // The earlier login goes first, listener included: its task is waited out, since a
    // cancellation alone leaves a port held until that task is next polled, and the flow below
    // would find it taken.
    forget_pending_logins(state, subject, account).await;
    let cancel = tokio_util::sync::CancellationToken::new();
    let prompt = Arc::new(HttpLoginPrompt {
        authorization_url: std::sync::Mutex::new(None),
        completion: std::sync::Mutex::new(None),
        paste_refusal: std::sync::Mutex::new(None),
        changed: tokio::sync::Notify::new(),
        pasted: std::sync::Mutex::new(None),
        paste_ready: tokio::sync::Notify::new(),
        cancel: cancel.clone(),
        committing: std::sync::atomic::AtomicBool::new(false),
    });
    let outcome: LoginOutcome = Arc::new(std::sync::Mutex::new(None));
    let task = tokio::spawn({
        let prompt = Arc::clone(&prompt);
        let outcome = Arc::clone(&outcome);
        let running = flow(Arc::clone(&prompt));
        async move {
            let result = running.await;
            // A login ended from outside keeps the sentence it was ended with.
            let mut settled = crate::sync::lock(&outcome);
            if settled.is_none() {
                *settled = Some(result);
            }
            drop(settled);
            prompt.wake();
        }
    });
    // In the map before the URL is waited for: a client that drops the request meanwhile leaves
    // a login the next begin replaces and the lifetime sweeps, rather than a task nothing holds
    // with a listener on the port the next login needs.
    let login_id = uuid::Uuid::new_v4().to_string();
    let expires_at = (chrono::Utc::now()
        + chrono::Duration::from_std(lifetime).unwrap_or_default())
    .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    // One login per subject, asked again under the map's lock: another begun on the subject since
    // the forget above is replaced by this one, as an earlier one was.
    let replaced: Vec<PendingLogin> = {
        let mut pending = crate::sync::lock(&state.pending_logins);
        let others: Vec<String> = pending
            .iter()
            .filter(|(_, login)| {
                login.subject == subject && login.account == account && !is_committing(login)
            })
            .map(|(login_id, _)| login_id.clone())
            .collect();
        let replaced = others
            .into_iter()
            .filter_map(|login_id| pending.remove(&login_id))
            .collect();
        pending.insert(login_id.clone(), PendingLogin {
            subject: subject.to_string(),
            account,
            prompt: Arc::clone(&prompt),
            expires_at: std::time::Instant::now() + lifetime,
            outcome: Arc::clone(&outcome),
            cancel,
            task,
        });
        replaced
    };
    end_logins(replaced, "the login was replaced").await;
    // The URL and what completes the login arrive once the flow has bound what it binds, or the
    // flow ends without them.
    let deadline = tokio::time::Instant::now() + LOGIN_URL_WAIT;
    let (authorization_url, completion) = loop {
        let url = crate::sync::lock(&prompt.authorization_url).clone();
        let completion = *crate::sync::lock(&prompt.completion);
        if let (Some(url), Some(completion)) = (url, completion) {
            break (url, completion);
        }
        if let Some(settled) = crate::sync::lock(&outcome).clone() {
            crate::sync::lock(&state.pending_logins).remove(&login_id);
            return settled.map(|()| Begun::Completed);
        }
        tokio::select! {
            () = prompt.changed.notified() => {}
            () = tokio::time::sleep_until(deadline) => {
                if let Some(login) = crate::sync::lock(&state.pending_logins).remove(&login_id) {
                    end_login(&login, "the login produced no authorization URL in time");
                }
                return Err(upstream(format!(
                    "the login for {noun} '{subject}' produced no authorization URL within {}s; \
                     the reason is in the meka log",
                    LOGIN_URL_WAIT.as_secs()
                )));
            }
        }
    };
    let started = LoginStarted {
        login_id,
        authorization_url,
        completes_with: match completion.paste {
            Some(crate::oauth::Paste::AuthorizationCode) => "code",
            _ => "callback_url",
        },
        listening: completion.listener_seconds.is_some(),
        expires_at,
    };
    tracing::info!("began a browser login for {noun} '{subject}' via HTTP");
    Ok(Begun::Browser(started))
}

/// Complete a pending browser login with what the browser gave: the line goes to the flow as a
/// paste, and the answer is how the flow took it. A login already completed is completed again;
/// one that has failed is a 422 with why; a paste the flow refused is a 422 that leaves the
/// login pending for another; and the wait ends with the login's own lifetime.
async fn complete_pending_login(
    state: &ServerState,
    account: bool,
    subject: &str,
    login_id: &str,
    pasted: String,
) -> Result<(), ProblemDetail> {
    let noun = if account { "account" } else { "MCP server" };
    let (prompt, outcome, expires_at) = pending_login(state, account, subject, login_id)?;
    // A blank line is waited past by the flow, as at a terminal, so it is refused here rather
    // than held until the login lapses.
    if pasted.trim().is_empty() {
        return Err(refused("code cannot be empty"));
    }
    if crate::sync::lock(&outcome).is_none() {
        // A refusal an earlier paste earned is not this one's.
        crate::sync::lock(&prompt.paste_refusal).take();
        *crate::sync::lock(&prompt.pasted) = Some(pasted);
        prompt.paste_ready.notify_one();
    }
    loop {
        if let Some(settled) = crate::sync::lock(&outcome).clone() {
            return settled;
        }
        if let Some(reason) = crate::sync::lock(&prompt.paste_refusal).take() {
            return Err(refused(reason));
        }
        tokio::select! {
            () = prompt.changed.notified() => {}
            () = tokio::time::sleep_until(expires_at.into()) => {
                // A commit under way decides the answer; its end wakes this wait.
                if prompt.committing.load(std::sync::atomic::Ordering::Acquire) {
                    prompt.changed.notified().await;
                    continue;
                }
                return Err(upstream(format!(
                    "the login for {noun} '{subject}' did not complete in time"
                )));
            }
        }
    }
}

/// The pending login `login_id` names on `subject`, an account's when `account` and an MCP
/// server's otherwise, or the 404 for one that is not there or has lapsed: its prompt, its
/// outcome, and when it lapses.
fn pending_login(
    state: &ServerState,
    account: bool,
    subject: &str,
    login_id: &str,
) -> Result<(Arc<HttpLoginPrompt>, LoginOutcome, std::time::Instant), ProblemDetail> {
    let noun = if account { "account" } else { "MCP server" };
    let mut pending = crate::sync::lock(&state.pending_logins);
    sweep_expired_logins(&mut pending);
    match pending.get(login_id) {
        Some(login) if login.subject == subject && login.account == account => Ok((
            Arc::clone(&login.prompt),
            Arc::clone(&login.outcome),
            login.expires_at,
        )),
        _ => Err(ProblemDetail::new(
            ErrorKind::NotFound,
            StatusCode::NOT_FOUND,
            format!("no login '{login_id}' is pending for {noun} '{subject}'"),
        )),
    }
}

/// `GET /v1/accounts/{name}/login/{login_id}`: where a browser login stands.
#[utoipa::path(
    get,
    path = "/v1/accounts/{name}/login/{login_id}",
    tag = "installation",
    params(
        ("name" = String, Path, description = "Account name"),
        ("login_id" = String, Path, description = "The login, as `POST /v1/accounts/{name}/login` named it"),
    ),
    responses(
        (status = 200, description = "Where the login stands", body = LoginStatus),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such login pending on the account, or one that lapsed", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:r"]))
)]
pub(crate) async fn login_status(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigRead>,
    AxumPath((name, login_id)): AxumPath<(String, String)>,
) -> Result<Json<LoginStatus>, ProblemDetail> {
    let (_, outcome, _) = pending_login(&state, true, &name, &login_id)?;
    Ok(Json(login_status_of(&outcome)))
}

/// The body of `POST /v1/accounts/{name}/login/{login_id}`.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompleteLoginRequest {
    /// What the browser gave: the code Anthropic's page shows, as shown, or the callback URL a
    /// `chatgpt-subscription` login sent the browser to.
    pub(crate) code: String,
}

/// `POST /v1/accounts/{name}/login/{login_id}`: complete a browser login with what the browser
/// gave, storing the credential.
///
/// A login the loopback listener has already completed answers 200 as well; one it has already
/// failed answers 422 with why. A code the authorization server rejects ends the login, since the
/// code is spent; a paste that carries no code leaves it pending for another try.
#[utoipa::path(
    post,
    path = "/v1/accounts/{name}/login/{login_id}",
    tag = "installation",
    params(
        ("name" = String, Path, description = "Account name"),
        ("login_id" = String, Path, description = "The login, as `POST /v1/accounts/{name}/login` named it"),
    ),
    request_body = CompleteLoginRequest,
    responses(
        (status = 200, description = "Logged in; the account, now authenticated", body = AccountView),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such login pending on the account, or one that lapsed", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "No code in what was pasted, another login's state, or an exchange the authorization server rejected; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 502, description = "The token exchange got no answer meka can read, or the login lapsed; the reason is in the meka log", body = ProblemDetail),
        (status = 503, description = "The server is shutting down", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn complete_login(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath((name, login_id)): AxumPath<(String, String)>,
    raw_body: Bytes,
) -> Result<Json<AccountView>, ProblemDetail> {
    let body: CompleteLoginRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("login completion", error))?;
    complete_pending_login(&state, true, &name, &login_id, body.code).await?;
    let token_store = state.shared.store.token_store();
    let catalog = state.shared.providers.catalog();
    Ok(Json(account_view(&catalog, &token_store, &name).await?))
}

// ----- Profiles ----------------------------------------------------------------------------------

/// What `GET /v1/profiles` answers.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProfilesResponse {
    pub(crate) profiles: Vec<ProfileView>,
}

/// `GET /v1/profiles`: configured profiles, with their account, backend and model only.
///
/// Under any read scope, with the other discovery endpoints: a client that may run turns may see
/// which profiles it can run them on. No credentials are returned: they live in the database,
/// keyed by account name, and never transit this API.
#[utoipa::path(
    get,
    path = "/v1/profiles",
    tag = "installation",
    responses(
        (status = 200, description = "Configured profiles", body = ProfilesResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r", "mcp:r", "skills:r", "memory:r", "schedule:r", "config:r"]))
)]
pub(crate) async fn profiles(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::AnyRead>,
) -> Result<Json<ProfilesResponse>, ProblemDetail> {
    let catalog = state.shared.providers.catalog();
    let active = catalog.default_profile.as_deref().ok();
    let profiles = catalog
        .profiles
        .iter()
        .map(|(name, profile)| {
            ProfileView::from_config(
                name,
                profile,
                &catalog.accounts,
                Some(name.as_str()) == active,
            )
        })
        .collect();
    Ok(Json(ProfilesResponse { profiles }))
}

/// One profile's detail from the catalog, or the 404 for a name it lacks.
fn profile_detail(
    catalog: &crate::config::ProfileCatalog,
    name: &str,
) -> Result<ProfileDetailView, ProblemDetail> {
    let Some(profile) = catalog.profiles.get(name) else {
        return Err(not_found("profile", name, catalog.profiles.keys()));
    };
    let active = catalog.default_profile.as_deref().ok() == Some(name);
    Ok(ProfileDetailView::from_config(
        name,
        profile,
        &catalog.accounts,
        active,
    ))
}

/// `GET /v1/profiles/{name}`: one profile, with every setting it states.
#[utoipa::path(
    get,
    path = "/v1/profiles/{name}",
    tag = "installation",
    params(("name" = String, Path, description = "Profile name")),
    responses(
        (status = 200, description = "The profile", body = ProfileDetailView),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such profile", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:r"]))
)]
pub(crate) async fn get_profile(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigRead>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<ProfileDetailView>, ProblemDetail> {
    let catalog = state.shared.providers.catalog();
    Ok(Json(profile_detail(&catalog, &name)?))
}

/// The body of `POST /v1/profiles`: `[profiles.<name>]` as it will be written. A setting left
/// out is left out of the profile, so its documented default applies.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateProfileRequest {
    pub(crate) name: String,
    /// The account the profile bills, which must be configured.
    pub(crate) account: String,
    pub(crate) model: String,
    /// The profiles a turn moves to when this one cannot serve it, in order.
    #[serde(default)]
    pub(crate) fallbacks: Vec<String>,
    pub(crate) context_window: Option<u64>,
    pub(crate) max_output_tokens: Option<u64>,
    pub(crate) max_request_bytes: Option<u64>,
    #[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]
    pub(crate) effort: Option<String>,
    pub(crate) vision: Option<bool>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) deferred_tools: Option<crate::config::DeferredTools>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) thinking: Option<crate::config::ThinkingMode>,
    pub(crate) thinking_budget: Option<u64>,
    pub(crate) cache_control: Option<bool>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) thinking_display: Option<crate::config::ThinkingDisplay>,
    /// Make the profile the one a session gets when it names none.
    #[serde(default)]
    pub(crate) default: bool,
}

/// `POST /v1/profiles`: add a profile on a configured account.
#[utoipa::path(
    post,
    path = "/v1/profiles",
    tag = "installation",
    request_body = CreateProfileRequest,
    responses(
        (status = 201, description = "Profile added", body = ProfileDetailView),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body: a name in use, an account not configured, a setting the backend never sends, a profile that could not run, or `default: true` on a server started with `--profile`; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn create_profile(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<ProfileDetailView>), ProblemDetail> {
    let body: CreateProfileRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("profile", error))?;
    let name = body.name.as_str();
    let tuning = profile::ProfileTuning {
        fallbacks: body.fallbacks,
        context_window: body.context_window,
        max_output_tokens: body.max_output_tokens,
        max_request_bytes: body.max_request_bytes,
        effort: body.effort,
        vision: body.vision,
        deferred_tools: body.deferred_tools,
        thinking: body.thinking,
        thinking_budget: body.thinking_budget,
        cache_control: body.cache_control,
        thinking_display: body.thinking_display,
    };
    refuse_a_default_under_the_started_profile(&state, body.default)?;
    write_installation(&state, async |_| {
        profile::write_profile(name, &body.account, &body.model, &tuning, body.default)
            .map_err(|error| door_problem(&state, "failed to write the profile", error))
    })
    .await?;
    tracing::info!(
        "added profile '{name}' on account '{account}' via HTTP",
        account = body.account
    );
    let catalog = state.shared.providers.catalog();
    Ok((StatusCode::CREATED, Json(profile_detail(&catalog, name)?)))
}

/// The body of `PATCH /v1/profiles/{name}`: a new `name`, sent on its own; `default: true`; and
/// any of the settings `meka profile set` writes, each left out to keep, `null` to remove, or a
/// value to set.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct PatchProfileRequest {
    /// The name the profile moves to, taking every session recorded on it along. Sent on its
    /// own: a rename moves rows in the store, and is not combined with a change to the file.
    #[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]
    pub(crate) name: Option<String>,
    /// `true` makes the profile the one a session gets when it names none. `false` is refused:
    /// the default is unset by removing the profile or choosing another.
    #[serde(default)]
    pub(crate) default: Option<bool>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) model: Patch<String>,
    #[serde(default)]
    #[schema(value_type = Option<Vec<String>>)]
    pub(crate) fallbacks: Patch<Vec<String>>,
    #[serde(default)]
    #[schema(value_type = Option<u64>)]
    pub(crate) context_window: Patch<u64>,
    #[serde(default)]
    #[schema(value_type = Option<u64>)]
    pub(crate) max_output_tokens: Patch<u64>,
    #[serde(default)]
    #[schema(value_type = Option<u64>)]
    pub(crate) max_request_bytes: Patch<u64>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) effort: Patch<String>,
    #[serde(default)]
    #[schema(value_type = Option<bool>)]
    pub(crate) vision: Patch<bool>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) deferred_tools: Patch<crate::config::DeferredTools>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) thinking: Patch<crate::config::ThinkingMode>,
    #[serde(default)]
    #[schema(value_type = Option<u64>)]
    pub(crate) thinking_budget: Patch<u64>,
    #[serde(default)]
    #[schema(value_type = Option<bool>)]
    pub(crate) cache_control: Patch<bool>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) thinking_display: Patch<crate::config::ThinkingDisplay>,
}

impl PatchProfileRequest {
    /// The changes the body asks for, as the edit door takes them. The door reads each value and
    /// refuses one no profile may carry, in the words `meka profile set` would.
    fn into_changes(self) -> profile::ProfileChanges {
        profile::ProfileChanges {
            model: self.model,
            fallbacks: self.fallbacks,
            context_window: self.context_window,
            max_output_tokens: self.max_output_tokens,
            max_request_bytes: self.max_request_bytes,
            effort: self.effort,
            vision: self.vision,
            deferred_tools: self.deferred_tools,
            thinking: self.thinking,
            thinking_budget: self.thinking_budget,
            cache_control: self.cache_control,
            thinking_display: self.thinking_display,
        }
    }
}

/// `PATCH /v1/profiles/{name}`: change a profile's settings, make it the default, or rename it.
#[utoipa::path(
    patch,
    path = "/v1/profiles/{name}",
    tag = "installation",
    params(("name" = String, Path, description = "Profile name")),
    request_body = PatchProfileRequest,
    responses(
        (status = 200, description = "The profile as it now stands", body = ProfileDetailView),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such profile", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body: a value the key refuses, a setting the backend never sends, a profile that could not run on the result, `default: false`, `default: true` on a server started with `--profile`, a `name` sent beside other changes, or a rename of the profile the server was started with `--profile` naming; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn patch_profile(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
    raw_body: Bytes,
) -> Result<Json<ProfileDetailView>, ProblemDetail> {
    let body: PatchProfileRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("profile patch", error))?;
    if body.default == Some(false) {
        return Err(refused(
            "`default: false` unsets nothing; remove the profile or make another the default",
        ));
    }
    let make_default = body.default == Some(true);
    refuse_a_default_under_the_started_profile(&state, make_default)?;
    let new_name = body.name.clone();
    let changes = body.into_changes();
    if new_name.is_some() && (make_default || !changes.is_empty()) {
        return Err(refused(
            "`name` renames the profile and is sent on its own; change its settings in another \
             request",
        ));
    }
    let (moved, _) = write_installation(&state, async |config_file| {
        if !config_file.profiles.contains_key(&name) {
            return Err(not_found("profile", &name, config_file.profiles.keys()));
        }
        if let Some(new_name) = &new_name {
            refuse_losing_the_started_profile(&state, &name)?;
            let moved = profile::rename_profile(&name, new_name, &state.shared.store)
                .await
                .map_err(|error| door_problem(&state, "failed to rename the profile", error))?;
            // The name is also held live, by every resident session and running sub-agent, and a
            // worker spawned or a row written from those cells must name the profile as the file
            // now does: a rename moves the name everywhere it is recorded.
            for entry in state.sessions.read().await.values() {
                entry
                    .resident
                    .cells()
                    .profile
                    .rename_profile(&name, new_name);
            }
            state
                .shared
                .running_subagents
                .rename_profile(&name, new_name);
            tracing::info!("renamed profile '{name}' to '{new_name}' via HTTP");
            return Ok(moved);
        }
        let keys = changes.keys();
        profile::edit_profile(&name, &changes, make_default)
            .map_err(|error| door_problem(&state, "failed to write the profile", error))?;
        if make_default {
            tracing::info!("default profile set to '{name}' via HTTP");
        }
        if !keys.is_empty() {
            tracing::info!(
                "changed profile '{name}' via HTTP: {keys}",
                keys = keys.join(", ")
            );
        }
        Ok(Vec::new())
    })
    .await?;
    // After the catalog knows the new name, so a client that refetches on the announcement
    // finds the profile the record names.
    super::sessions::announce_sessions_moved(&state, &moved).await;
    let catalog = state.shared.providers.catalog();
    let final_name = new_name.unwrap_or(name);
    Ok(Json(profile_detail(&catalog, &final_name)?))
}

/// Refuse renaming or removing the profile `--profile` named when this server started: the
/// server selects it as the default for as long as it runs, so a session naming none would be
/// refused, and the file would no longer be one the server could start on.
fn refuse_losing_the_started_profile(state: &ServerState, name: &str) -> Result<(), ProblemDetail> {
    if state.shared.requested_profile() == Some(name) {
        return Err(refused(format!(
            "this server was started with `--profile {name}`, which selects it as the default; \
             restart the server without `--profile` first"
        )));
    }
    Ok(())
}

/// Refuse `"default": true` on a server `--profile` started: the flag selects the default for as
/// long as the server runs, so the file's `default_profile` would move and nothing a session
/// gets would.
fn refuse_a_default_under_the_started_profile(
    state: &ServerState,
    asked: bool,
) -> Result<(), ProblemDetail> {
    if asked && let Some(started) = state.shared.requested_profile() {
        return Err(refused(format!(
            "this server was started with `--profile {started}`, which selects the default; \
             restart the server without `--profile` first"
        )));
    }
    Ok(())
}

/// `DELETE /v1/profiles/{name}`: remove a profile. The sessions recorded on it keep the name and
/// refuse to resume until it is configured again or they are moved.
#[utoipa::path(
    delete,
    path = "/v1/profiles/{name}",
    tag = "installation",
    params(("name" = String, Path, description = "Profile name")),
    responses(
        (status = 204, description = "Profile removed"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such profile", body = ProblemDetail),
        (status = 422, description = "Another profile names it in `fallbacks`; it is the default while two or more profiles would remain; or the server was started with `--profile` naming it; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn delete_profile(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    write_installation(&state, async |config_file| {
        if !config_file.profiles.contains_key(&name) {
            return Err(not_found("profile", &name, config_file.profiles.keys()));
        }
        refuse_losing_the_started_profile(&state, &name)?;
        profile::remove_profile(&name, &state.shared.store)
            .await
            .map_err(|error| door_problem(&state, "failed to remove the profile", error))
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}

// ----- MCP servers -------------------------------------------------------------------------------

/// The manager every MCP door acts on. `meka serve` holds one from startup, servers or none.
fn mcp_manager(state: &ServerState) -> Result<Arc<crate::mcp::McpClientManager>, ProblemDetail> {
    state.shared.mcp_manager.clone().ok_or_else(|| {
        ProblemDetail::internal_sanitized("no MCP manager", "the host started without one")
    })
}

/// One server's entry as the running manager holds it, or the 404 naming the configured ones.
fn configured_server(
    manager: &crate::mcp::McpClientManager,
    name: &str,
) -> Result<crate::config::McpServerConfig, ProblemDetail> {
    let configs = manager.server_configs();
    configs
        .iter()
        .find(|config| config.name == name)
        .cloned()
        .ok_or_else(|| {
            not_found(
                "MCP server",
                name,
                configs.iter().map(|config| config.name.as_str()),
            )
        })
}

/// A configured server's entry and where its connection stands: `GET /v1/mcp/{name}`, and what
/// every write answers with.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct McpServerStatus {
    #[serde(flatten)]
    pub(crate) detail: crate::view::McpServerDetail,
    /// `pending`, `connected`, `failed` or `disabled`.
    pub(crate) state: String,
    /// Why the last connect failed, on a `failed` server, with URLs redacted as the connector
    /// records them: a client that manages servers without a shell has no log to read it from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

async fn server_status(
    state: &ServerState,
    manager: &crate::mcp::McpClientManager,
    name: &str,
) -> Result<McpServerStatus, ProblemDetail> {
    let detail = crate::installation::mcp::server_detail(
        &configured_server(manager, name)?,
        manager.overlay(),
        &state.shared.store.token_store(),
    )
    .await
    .map_err(|error| ProblemDetail::internal_sanitized("failed to read the server", error))?;
    let (connection, error) = match manager.server_entry(name) {
        Some(entry) => {
            let state = entry.state().await;
            let error = match &state {
                crate::mcp::ServerState::Failed {
                    needs: Some(need), ..
                } => Some(crate::host::http::errors::api_wording(&need.refusal(name))),
                crate::mcp::ServerState::Failed { error, .. } => Some(error.clone()),
                _ => None,
            };
            (state.label().to_string(), error)
        }
        None => (crate::mcp::ServerState::Disabled.label().to_string(), None),
    };
    Ok(McpServerStatus {
        detail,
        state: connection,
        error,
    })
}

/// `GET /v1/mcp/{name}`: one server's configuration and connection state.
#[utoipa::path(
    get,
    path = "/v1/mcp/{name}",
    tag = "installation",
    params(("name" = String, Path, description = "MCP server name")),
    responses(
        (status = 200, description = "The server as configured, with the kinds of credential stored for it and its connection state", body = McpServerStatus),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:r"]))
)]
pub(crate) async fn get_mcp_server(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigRead>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<McpServerStatus>, ProblemDetail> {
    let manager = mcp_manager(&state)?;
    Ok(Json(server_status(&state, &manager, &name).await?))
}

/// The body of `POST /v1/mcp`: a `[[mcp.servers]]` entry as `config.toml` holds it, with the
/// secrets the entry presents beside its keys, which go to the store and never to the file.
///
/// Parsed by [`Self::parse`] and described by hand rather than derived: serde does not combine
/// `flatten` with the entry's `deny_unknown_fields`, and the entry has to keep refusing a key it
/// does not model.
#[derive(Debug)]
pub(crate) struct McpServerRequest {
    pub(crate) server: crate::config::McpServerConfig,
    /// A bearer sent verbatim on every request, for an http server with no `auth` block.
    pub(crate) auth_token: Option<String>,
    /// The client secret a confidential OAuth client presents, for an `auth` block of type
    /// `oauth` or `client_credentials`.
    pub(crate) client_secret: Option<String>,
}

impl utoipa::PartialSchema for McpServerRequest {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::schema::{AllOfBuilder, ObjectBuilder, Type};
        AllOfBuilder::new()
            .item(utoipa::openapi::Ref::from_schema_name("McpServerConfig"))
            .item(
                ObjectBuilder::new()
                    .property(
                        "auth_token",
                        ObjectBuilder::new()
                            .schema_type(Type::String)
                            .description(Some(
                                "A bearer sent verbatim on every request, for an http server \
                                 with no `auth` block",
                            )),
                    )
                    .property(
                        "client_secret",
                        ObjectBuilder::new()
                            .schema_type(Type::String)
                            .description(Some(
                                "The client secret a confidential OAuth client presents, for an \
                                 `auth` block of type `oauth` or `client_credentials`",
                            )),
                    ),
            )
            .into()
    }
}

impl utoipa::ToSchema for McpServerRequest {}

impl McpServerRequest {
    /// The two secrets taken off the object, the rest read as the entry, which refuses a key it
    /// does not model. A blank secret is none.
    fn parse(raw_body: &[u8]) -> serde_json::Result<Self> {
        let mut object: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(raw_body)?;
        let mut secret = |key: &str| -> serde_json::Result<Option<String>> {
            match object.remove(key) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(value) => {
                    let text: String = serde_json::from_value(value)?;
                    Ok(Some(text).filter(|text| !text.trim().is_empty()))
                }
            }
        };
        let auth_token = secret("auth_token")?;
        let client_secret = secret("client_secret")?;
        let server = serde_json::from_value(serde_json::Value::Object(object))?;
        Ok(Self {
            server,
            auth_token,
            client_secret,
        })
    }
}

/// `POST /v1/mcp`: add a server to `config.toml` and to the running manager, which connects it
/// in the background.
///
/// The answer's `state` is where the connection stands at once, `pending` for a server about to
/// connect; `GET /v1/mcp` says when it has. No login runs: a server whose endpoint asks for one
/// is logged in through `POST /v1/mcp/{name}/login`.
#[utoipa::path(
    post,
    path = "/v1/mcp",
    tag = "installation",
    request_body = McpServerRequest,
    responses(
        (status = 201, description = "Server added", body = McpServerStatus),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body: a name in use or reserved, keys that do not fit the transport, or a secret the entry cannot present; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn create_mcp_server(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<McpServerStatus>), ProblemDetail> {
    let body = McpServerRequest::parse(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("MCP server", error))?;
    let manager = mcp_manager(&state)?;
    let secrets = crate::installation::mcp::ServerSecrets {
        auth_token: body.auth_token,
        client_secret: body.client_secret,
    };
    let token_store = state.shared.store.token_store();
    write_installation(&state, async |_| {
        crate::installation::mcp::add_server(&body.server, &secrets, &token_store)
            .await
            .map_err(|error| door_problem(&state, "failed to add the MCP server", error))
    })
    .await?;
    let name = body.server.name.clone();
    tracing::info!("added MCP server '{name}' via HTTP");
    Ok((
        StatusCode::CREATED,
        Json(server_status(&state, &manager, &name).await?),
    ))
}

/// The body of `PATCH /v1/mcp/{name}`: any of the entry's keys, each left out to keep, `null` to
/// remove, or a value to set, `env`, `headers` and `tool_permissions` merging key by key, and the
/// two secrets, each `null` to clear the kind. A string key reads `""` as `null`. `name` and
/// `transport` are not changed: another transport is another server.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct PatchMcpServerRequest {
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) command: Patch<String>,
    #[serde(default)]
    #[schema(value_type = Option<Vec<String>>)]
    pub(crate) args: Patch<Vec<String>>,
    /// Merged into the entry's: a key set to a string is set, one set to `null` removed.
    #[serde(default)]
    #[schema(value_type = Option<std::collections::HashMap<String, Option<String>>>)]
    pub(crate) env: Patch<std::collections::HashMap<String, Option<String>>>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) url: Patch<String>,
    /// Merged into the entry's, as `env` is.
    #[serde(default)]
    #[schema(value_type = Option<std::collections::HashMap<String, Option<String>>>)]
    pub(crate) headers: Patch<std::collections::HashMap<String, Option<String>>>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) headers_helper: Patch<String>,
    #[serde(default)]
    #[schema(value_type = Option<crate::config::McpAuthConfig>)]
    pub(crate) auth: Patch<crate::config::McpAuthConfig>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) permission: Patch<crate::permission::Permission>,
    #[serde(default)]
    #[schema(value_type = Option<Vec<String>>)]
    pub(crate) allowed_tools: Patch<Vec<String>>,
    #[serde(default)]
    #[schema(value_type = Option<Vec<String>>)]
    pub(crate) disabled_tools: Patch<Vec<String>>,
    #[serde(default)]
    #[schema(value_type = Option<Vec<String>>)]
    pub(crate) eager_load_tools: Patch<Vec<String>>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) deferral: Patch<crate::config::Deferral>,
    /// Merged into the entry's, as `env` is.
    #[serde(default)]
    #[schema(value_type = Option<std::collections::HashMap<String, Option<String>>>)]
    pub(crate) tool_permissions:
        Patch<std::collections::HashMap<String, Option<crate::permission::Permission>>>,
    #[serde(default)]
    #[schema(value_type = Option<bool>)]
    pub(crate) trust_read_only_hint: Patch<bool>,
    #[serde(default)]
    #[schema(value_type = Option<bool>)]
    pub(crate) disabled: Patch<bool>,
    #[serde(default)]
    #[schema(value_type = Option<bool>)]
    pub(crate) required: Patch<bool>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) auth_token: Patch<String>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) client_secret: Patch<String>,
}

/// `PATCH /v1/mcp/{name}`: change a server's entry in place. A change to what its connection is
/// made from, or to a credential it stores, restarts it; a change to its tool policy alone applies
/// on the connection it holds; `disabled: true` stops it and `false` starts it.
#[utoipa::path(
    patch,
    path = "/v1/mcp/{name}",
    tag = "installation",
    params(("name" = String, Path, description = "MCP server name")),
    request_body = PatchMcpServerRequest,
    responses(
        (status = 200, description = "The server as it now stands", body = McpServerStatus),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body: keys that do not fit the transport, an `auth` block beside a stored bearer, or a secret the entry cannot present; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn patch_mcp_server(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
    raw_body: Bytes,
) -> Result<Json<McpServerStatus>, ProblemDetail> {
    let body: PatchMcpServerRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("MCP server patch", error))?;
    let manager = mcp_manager(&state)?;
    let changes = crate::installation::mcp::ServerChanges {
        command: body.command,
        args: body.args,
        env: body.env,
        url: body.url,
        headers: body.headers,
        headers_helper: body.headers_helper,
        auth: body.auth,
        permission: body.permission,
        allowed_tools: body.allowed_tools,
        disabled_tools: body.disabled_tools,
        eager_load_tools: body.eager_load_tools,
        deferral: body.deferral,
        tool_permissions: body.tool_permissions,
        trust_read_only_hint: body.trust_read_only_hint,
        disabled: body.disabled,
        required: body.required,
        auth_token: body.auth_token,
        client_secret: body.client_secret,
    };
    let token_store = state.shared.store.token_store();
    let ((_, revocation), _) = write_installation(&state, async |config_file| {
        require_file_server(config_file, &name)?;
        crate::installation::mcp::edit_server(&name, &changes, &token_store)
            .await
            .map_err(|error| door_problem(&state, "failed to change the MCP server", error))
    })
    .await?;
    // Outside the write, once it has committed: the authorization server holds up nothing.
    revocation.send().await;
    tracing::info!("changed MCP server '{name}' via HTTP");
    Ok(Json(server_status(&state, &manager, &name).await?))
}

/// `DELETE /v1/mcp/{name}`: remove a server from `config.toml` and from the running manager,
/// closing its connection and dropping its tools from every session, and clear the credentials
/// stored under its name, the OAuth grant revoked at the provider once the removal is served,
/// where it can be. A name with credentials and no entry is cleared the same way.
#[utoipa::path(
    delete,
    path = "/v1/mcp/{name}",
    tag = "installation",
    params(("name" = String, Path, description = "MCP server name")),
    responses(
        (status = 204, description = "Server removed"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server, and no credentials under the name", body = ProblemDetail),
        (status = 422, description = "`config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn delete_mcp_server(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    mcp_manager(&state)?;
    let token_store = state.shared.store.token_store();
    let (revocation, _) = write_installation(&state, async |config_file| {
        refuse_a_name_nothing_holds(config_file, &name, &token_store).await?;
        crate::installation::mcp::remove_server(&name, &token_store)
            .await
            .map_err(|error| door_problem(&state, "failed to remove the MCP server", error))
    })
    .await?;
    revocation.send().await;
    tracing::info!("removed MCP server '{name}' via HTTP");
    Ok(StatusCode::NO_CONTENT)
}

/// The entry `name` in the parsed file, or the 404 naming the configured ones.
fn require_file_server(
    config_file: &crate::config::ConfigFile,
    name: &str,
) -> Result<crate::config::McpServerConfig, ProblemDetail> {
    let entries = config_file.mcp_servers();
    entries
        .iter()
        .find(|entry| entry.name == name)
        .cloned()
        .ok_or_else(|| {
            not_found(
                "MCP server",
                name,
                entries.iter().map(|entry| entry.name.as_str()),
            )
        })
}

/// The 404 for a name neither the file's entries nor the stored credentials hold, from the two
/// doors that take either: a removal and a logout.
async fn refuse_a_name_nothing_holds(
    config_file: &crate::config::ConfigFile,
    name: &str,
    token_store: &crate::store::TokenStore,
) -> Result<(), ProblemDetail> {
    let Err(missing) = require_file_server(config_file, name) else {
        return Ok(());
    };
    let has_credentials = token_store
        .has_mcp_credentials(name)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to read the credentials", error)
        })?;
    if has_credentials {
        return Ok(());
    }
    Err(missing)
}

/// The body of `POST /v1/mcp/{name}/login`: a secret to store, or none to begin the OAuth login
/// in the browser.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpLoginRequest {
    /// A bearer sent verbatim on every request, for a server with no `auth` block.
    #[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]
    pub(crate) auth_token: Option<String>,
    /// The client secret a confidential OAuth client presents.
    #[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]
    pub(crate) client_secret: Option<String>,
}

/// `POST /v1/mcp/{name}/login`: store a credential for a server, or begin its OAuth login in the
/// browser.
///
/// With `auth_token` or `client_secret`, the secret is stored, as `meka mcp login
/// --auth-token-stdin` or `--client-secret-stdin` stores it, and the server is restarted from its
/// entry; the answer is the server as it stands. With neither, the OAuth login `meka mcp
/// login` runs at a terminal begins: the answer is 202 with the URL to open and a `login_id`,
/// the server holds the loopback listener the browser is sent back to, and a browser elsewhere
/// is answered by posting the callback URL to `POST /v1/mcp/{name}/login/{login_id}`. The server
/// is restarted from the entry as the file then states it once the login completes. A
/// `client_credentials` server needs no browser: its login connects with the stored secret and is
/// answered 200 with the server, as a stored secret is.
#[utoipa::path(
    post,
    path = "/v1/mcp/{name}/login",
    tag = "installation",
    params(("name" = String, Path, description = "MCP server name")),
    request_body = McpLoginRequest,
    responses(
        (status = 200, description = "The secret stored, or a login that needed no browser completed: a `client_credentials` server's; the server as it stands", body = McpServerStatus),
        (status = 202, description = "A browser login begun; open the URL, then complete the login", body = LoginStarted),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "A secret the entry cannot present, both secrets at once, or a login the entry cannot run: a stdio server, or one holding a bearer; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 502, description = "The login did not begin: the server or its authorization server did not answer; the reason is in the meka log", body = ProblemDetail),
        (status = 503, description = "The server is shutting down", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn login_mcp_server(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
    raw_body: Bytes,
) -> Result<Response, ProblemDetail> {
    let body: McpLoginRequest = if raw_body.is_empty() {
        McpLoginRequest {
            auth_token: None,
            client_secret: None,
        }
    } else {
        serde_json::from_slice(&raw_body)
            .map_err(|error| ProblemDetail::invalid_body("MCP login", error))?
    };
    let manager = mcp_manager(&state)?;
    let token_store = state.shared.store.token_store();
    let secret = match (body.auth_token, body.client_secret) {
        (Some(_), Some(_)) => {
            return Err(refused(
                "`auth_token` and `client_secret` are two credentials; store one per request",
            ));
        }
        (Some(token), None) => Some((crate::store::McpCredentialKind::Bearer, token)),
        (None, Some(secret)) => Some((crate::store::McpCredentialKind::ClientSecret, secret)),
        (None, None) => None,
    };
    if let Some((kind, secret)) = secret {
        // Served after, so the server restarts on the secret.
        write_installation(&state, async |config_file| {
            require_file_server(config_file, &name)?;
            crate::installation::mcp::store_server_secret(
                &config_file.mcp_servers(),
                &token_store,
                &name,
                kind,
                &secret,
            )
            .await
            .map_err(|error| door_problem(&state, "failed to store the credential", error))
        })
        .await?;
        tracing::info!("stored a credential for MCP server '{name}' via HTTP");
        return Ok(Json(server_status(&state, &manager, &name).await?).into_response());
    }

    // The login begins against the entry as the file states it, which its commit holds it to.
    let config_file = installation_as_written(&state)?;
    require_file_server(&config_file, &name)?;
    let servers = config_file.mcp_servers();
    let overlay = manager.overlay().clone();
    let server = name.clone();
    let state_for_flow = state.clone();
    let begun = begin_login(
        &state,
        &name,
        false,
        // The flow's own bound: its callback wait starts after discovery, and the connect holds
        // both, so the login outlives neither.
        crate::mcp::auth::OAUTH_CALLBACK_TIMEOUT + crate::installation::mcp::PROBE_CONNECT_TIMEOUT,
        move |prompt| async move {
            let login_prompt: Arc<dyn crate::oauth::LoginPrompt> = Arc::clone(&prompt) as _;
            let grant = crate::installation::mcp::obtain_login(
                &servers,
                &overlay,
                &token_store,
                &server,
                login_prompt,
            )
            .await
            .map_err(|error| mcp_login_problem(&state_for_flow, &server, error))?;
            commit_on_its_own_task(&state_for_flow, &prompt, {
                let state = state_for_flow.clone();
                let server = server.clone();
                async move {
                    write_installation(&state, async |config_file| {
                        crate::installation::mcp::commit_login(config_file, grant, &token_store)
                            .await
                            .map_err(|error| mcp_login_problem(&state, &server, error))
                    })
                    .await
                    .map(|_| ())
                }
            })
            .await?;
            tracing::info!("logged in MCP server '{server}' from the browser");
            Ok(())
        },
    )
    .await?;
    match begun {
        Begun::Browser(started) => Ok((StatusCode::ACCEPTED, Json(started)).into_response()),
        Begun::Completed => Ok(Json(server_status(&state, &manager, &name).await?).into_response()),
    }
}

/// What an MCP login's failure is on the wire. An auth failure carries the reason the flow
/// recorded, with URLs redacted, since a URL may name a key, as a connector's reason may; its
/// kind's sentence names nothing, and the client managing servers over HTTP has no log to read
/// one from. A refusal is the door's own words, the rest the log's to say.
fn mcp_login_problem(
    state: &ServerState,
    server: &str,
    error: crate::error::MekaError,
) -> ProblemDetail {
    tracing::warn!("the browser login for MCP server '{server}' failed: {error}");
    match &error {
        crate::error::MekaError::McpCredentialInTheWay { .. } => {
            door_problem(state, "the MCP login failed", error)
        }
        _ if error.kind() == ErrorKind::InvalidBody => {
            refused(crate::text::redact_urls(&error.sentence()))
        }
        crate::error::MekaError::McpAuth { message, .. } => upstream(format!(
            "the login did not complete: {}",
            crate::text::redact_urls(message)
        )),
        crate::error::MekaError::Database(_) | crate::error::MekaError::Io(_) => {
            ProblemDetail::internal_sanitized("the MCP login failed", error)
        }
        _ => upstream("the login did not complete; the reason is in the meka log"),
    }
}

/// `GET /v1/mcp/{name}/login/{login_id}`: where a browser login stands.
#[utoipa::path(
    get,
    path = "/v1/mcp/{name}/login/{login_id}",
    tag = "installation",
    params(
        ("name" = String, Path, description = "MCP server name"),
        ("login_id" = String, Path, description = "The login, as `POST /v1/mcp/{name}/login` named it"),
    ),
    responses(
        (status = 200, description = "Where the login stands", body = LoginStatus),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such login pending on the server, or one that lapsed", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:r"]))
)]
pub(crate) async fn mcp_login_status(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigRead>,
    AxumPath((name, login_id)): AxumPath<(String, String)>,
) -> Result<Json<LoginStatus>, ProblemDetail> {
    let (_, outcome, _) = pending_login(&state, false, &name, &login_id)?;
    Ok(Json(login_status_of(&outcome)))
}

/// `POST /v1/mcp/{name}/login/{login_id}`: complete a browser login with the callback URL the
/// browser was sent to, when it could not reach the listener itself.
///
/// A URL with no code in it is refused and leaves the login pending, as an account's is. The
/// answer waits for the exchange and the commit; the restart connects in the background, so the
/// server it carries is `pending` until it has.
#[utoipa::path(
    post,
    path = "/v1/mcp/{name}/login/{login_id}",
    tag = "installation",
    params(
        ("name" = String, Path, description = "MCP server name"),
        ("login_id" = String, Path, description = "The login, as `POST /v1/mcp/{name}/login` named it"),
    ),
    request_body = CompleteLoginRequest,
    responses(
        (status = 200, description = "Logged in; the server as it now stands", body = McpServerStatus),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such login pending on the server, or one that lapsed", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "No code in what was posted, another login's state, or a login that had already ended; or `config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 502, description = "The exchange at the authorization server failed; the reason is in the meka log", body = ProblemDetail),
        (status = 503, description = "The server is shutting down", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn complete_mcp_login(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath((name, login_id)): AxumPath<(String, String)>,
    raw_body: Bytes,
) -> Result<Json<McpServerStatus>, ProblemDetail> {
    let body: CompleteLoginRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("login completion", error))?;
    let manager = mcp_manager(&state)?;
    // The flow exchanges the code and the task reconnects the server; both are bounded by the
    // flow's own timeout, which the login's lifetime matches.
    complete_pending_login(&state, false, &name, &login_id, body.code).await?;
    tracing::info!("completed the browser login for MCP server '{name}' via HTTP");
    Ok(Json(server_status(&state, &manager, &name).await?))
}

/// `POST /v1/mcp/{name}/logout`: clear every credential stored for a server and restart it
/// without them, the OAuth grant revoked at the provider once that is served, where it can be. A
/// server holding none is left as it is.
#[utoipa::path(
    post,
    path = "/v1/mcp/{name}/logout",
    tag = "installation",
    params(("name" = String, Path, description = "MCP server name")),
    responses(
        (status = 204, description = "Credentials cleared"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server, and no credentials under the name", body = ProblemDetail),
        (status = 422, description = "`config.toml` holds something startup would refuse, which refuses every write until it is repaired", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn logout_mcp_server(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    mcp_manager(&state)?;
    let token_store = state.shared.store.token_store();
    // Served after, so the server restarts without the credentials it held.
    let (revocation, _) = write_installation(&state, async |config_file| {
        refuse_a_name_nothing_holds(config_file, &name, &token_store).await?;
        crate::installation::mcp::logout_server(&config_file.mcp_servers(), &token_store, &name)
            .await
            .map_err(|error| door_problem(&state, "failed to clear the credentials", error))
    })
    .await?;
    revocation.send().await;
    tracing::info!("logged out MCP server '{name}' via HTTP");
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key left out keeps, `null` removes, a value sets: the three readings a form can send,
    /// told apart on the way to the door, which reads the values.
    #[test]
    fn a_profile_patch_tells_keep_remove_and_set_apart() {
        let body: PatchProfileRequest = serde_json::from_str(
            r#"{"context_window": null, "effort": "", "model": "m", "vision": false}"#,
        )
        .expect("parse");
        let changes = body.into_changes();
        assert_eq!(changes.model, Patch::Set("m".to_string()));
        assert_eq!(changes.fallbacks, Patch::Keep);
        assert_eq!(changes.context_window, Patch::Remove);
        assert_eq!(changes.effort, Patch::Set(String::new()));
        assert_eq!(changes.vision, Patch::Set(false));
        assert_eq!(changes.keys(), [
            "model",
            "context_window",
            "effort",
            "vision"
        ]);

        let rename: PatchProfileRequest =
            serde_json::from_str(r#"{"name": "primary"}"#).expect("parse");
        assert_eq!(rename.name.as_deref(), Some("primary"));
        assert_eq!(rename.default, None);
        assert!(rename.into_changes().is_empty());
    }

    /// The secrets ride beside the entry and are taken out of it: the entry keeps refusing a key
    /// it does not model, and a secret given is a secret the door is handed.
    #[test]
    fn an_mcp_server_request_keeps_its_secrets_beside_the_entry() {
        let body = McpServerRequest::parse(
            br#"{"name":"odd","transport":"stdio","command":"x","auth_token":"t"}"#,
        )
        .expect("parse");
        assert_eq!(body.auth_token.as_deref(), Some("t"));
        assert_eq!(body.server.command.as_deref(), Some("x"));
        let refused = McpServerRequest::parse(
            br#"{"name":"odd","transport":"stdio","command":"x","frobnicate":1}"#,
        );
        assert!(refused.is_err(), "an unknown key is refused");
    }
}
