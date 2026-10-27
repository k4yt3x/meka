//! The installation as `config.toml` holds it, managed over HTTP: the accounts and the profiles.
//!
//! Every write goes through the door `meka account` or `meka profile` uses, under the config file
//! lock, keeping the file's comments and order, and refusing in the same words; what differs is
//! the prompt, which a body answers up front. A write that landed replaces the catalog every
//! session on this server resolves against ([`crate::config::ProfileCatalog`]), so the profile
//! a client adds is the one its next `POST /v1/sessions` runs on, without a restart.
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

/// What a configuration door's refusal is on the wire. A fault of the file or the store is the
/// sanitized 500 with the text in the log, where the operator reads it; everything else is the
/// door's own sentence about what was sent, as the 422 every refusal of a body is. An unknown
/// name on the path is a 404 ahead of the door, from the handler holding the catalog.
fn door_refusal(context: &str, error: anyhow::Error) -> ProblemDetail {
    if error.downcast_ref::<std::io::Error>().is_some() {
        return ProblemDetail::internal_sanitized(context, error);
    }
    let sentence = match error.downcast_ref::<crate::error::MekaError>() {
        Some(crate::error::MekaError::Io(_) | crate::error::MekaError::Database(_)) => {
            return ProblemDetail::internal_sanitized(context, error);
        }
        // The sentence rather than the display, which prefixes its class.
        Some(refusal) => refusal.sentence(),
        None => error.to_string(),
    };
    ProblemDetail::new(
        ErrorKind::InvalidBody,
        StatusCode::UNPROCESSABLE_ENTITY,
        sentence,
    )
}

/// The 422 for a body the door would not be asked about.
fn refused(sentence: impl Into<String>) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::InvalidBody,
        StatusCode::UNPROCESSABLE_ENTITY,
        sentence,
    )
}

/// The 404 for a name no account or profile has, in the words every door refuses one with.
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

/// The catalog's reload after a write to `config.toml`, begun ahead of the write. The rename and
/// remove doors work on the raw document, so a file with a defect elsewhere can be repaired at a
/// terminal; here the catalog is rebuilt from what a write leaves, so a file that already fails to
/// parse is refused before it is written, naming the line, rather than written and then reported as
/// a fault.
#[must_use]
struct Reload<'a>(&'a ServerState);

impl<'a> Reload<'a> {
    fn ahead_of_write(state: &'a ServerState) -> Result<Self, ProblemDetail> {
        crate::config::load_config_file_or_err().map_err(|error| {
            refused(format!(
                "config.toml already fails to parse; repair it first: {sentence}",
                sentence = error.sentence()
            ))
        })?;
        Ok(Self(state))
    }

    /// Bring the catalog up to the file the write produced, or say that it could not be read.
    fn after_write(self) -> Result<(), ProblemDetail> {
        self.0.shared.reload_profiles().map_err(|error| {
            ProblemDetail::internal_sanitized("failed to reload the profiles after a write", error)
        })
    }
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
    let orphaned_credentials = account::orphaned_accounts(&token_store, &catalog.accounts)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to list credentials", error))?;
    Ok(Json(AccountsResponse {
        accounts,
        orphaned_credentials,
    }))
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
        (status = 422, description = "Invalid body: a name in use, an unknown backend, a setting the backend never reads, or a key where none is taken or none where one is", body = ProblemDetail),
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
    let name = body.name.trim();
    if name.is_empty() {
        return Err(refused("account name cannot be empty"));
    }
    let backend =
        account::validate_backend(&body.backend).map_err(|error| refused(error.to_string()))?;
    // Refused rather than dropped, as `meka profile set` refuses a key the backend never reads: a
    // body states exactly what it means to write, and reporting success for a setting that was
    // not written would be reporting nothing.
    for (key, given) in [
        ("client_id", body.client_id.is_some()),
        ("oauth_token_url", body.oauth_token_url.is_some()),
        ("interleaved_thinking", body.interleaved_thinking.is_some()),
    ] {
        if given && !backend.reads_account_key(key) {
            return Err(refused(format!(
                "a '{backend}' account never reads `{key}`; drop it"
            )));
        }
    }
    if backend == crate::config::Backend::ChatGptSubscription
        && let Some(url) = body.base_url.as_deref()
    {
        crate::provider::openai::subscription::chatgpt_base_url_shape(url)
            .map_err(|error| refused(error.to_string()))?;
    }
    let credential = match (account::credential_kind(backend), body.api_key) {
        (account::CredentialKind::ApiKey, Some(key)) => {
            Some(crate::store::AuthCredential::ApiKey(key))
        }
        (account::CredentialKind::ApiKey, None) => {
            return Err(refused(format!("a '{backend}' account needs `api_key`")));
        }
        (_, Some(_)) => {
            return Err(refused(format!(
                "'{backend}' logs in through the browser and takes no API key; add the account \
                 without one and log in at `POST /v1/accounts/{name}/login`"
            )));
        }
        (_, None) => None,
    };
    let settings = account::OAuthSettings {
        oauth_token_url: body.oauth_token_url,
        client_id: body.client_id,
    };
    let reload = Reload::ahead_of_write(&state)?;
    // The account before the secret, as `meka account add` orders them: a failed credential write
    // then leaves an account with no credential, which the listing shows and the login repairs,
    // rather than a credential no account names.
    account::write_account(
        name,
        backend,
        body.base_url.as_deref(),
        &settings,
        body.interleaved_thinking,
    )
    .map_err(|error| door_refusal("failed to write the account", error))?;
    let token_store = state.shared.store.token_store();
    if let Some(credential) = &credential {
        token_store
            .save_account_credential(name, credential)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to store the account credential", error)
            })?;
    }
    reload.after_write()?;
    tracing::info!("added account '{name}' via HTTP");
    let catalog = state.shared.providers.catalog();
    let view = account_view(&catalog, &token_store, name).await?;
    Ok((StatusCode::CREATED, Json(view)))
}

/// One account's view from the catalog a write just produced.
async fn account_view(
    catalog: &crate::config::ProfileCatalog,
    token_store: &crate::store::TokenStore,
    name: &str,
) -> Result<AccountView, ProblemDetail> {
    let Some(config) = catalog.accounts.get(name) else {
        return Err(ProblemDetail::internal_sanitized(
            "the account written is not in the reloaded catalog",
            name,
        ));
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

/// The body of `PATCH /v1/accounts/{name}` and of a `PATCH /v1/profiles/{name}` that renames.
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
        (status = 409, description = "A session resident on this server runs on the account, whose credential refreshes under its name", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid new name, or one in use", body = ProblemDetail),
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
    let catalog = state.shared.providers.catalog();
    let Some(config) = catalog.accounts.get(&name) else {
        return Err(not_found("account", &name, catalog.accounts.keys()));
    };
    // A session resident here keeps the provider it was built with, and a subscription's
    // credential refreshes under the account's name: renamed under the session, the next refresh
    // would write to a row that is gone and lose the token the issuer just rotated. Refused while
    // such a session is up; an API key is held by its provider whole and has no row to refresh.
    let refreshes = account::validate_backend(&config.backend).is_ok_and(|backend| {
        !matches!(
            account::credential_kind(backend),
            account::CredentialKind::ApiKey
        )
    });
    if refreshes {
        let resident: Vec<uuid::Uuid> = state.sessions.read().await.keys().copied().collect();
        let mut running = 0;
        for id in resident {
            let recorded = state
                .shared
                .store
                .recorded_profile(id)
                .await
                .map_err(|error| {
                    ProblemDetail::internal_sanitized("failed to read a session's profile", error)
                })?;
            // A session on a profile since removed still runs on the provider it was built
            // with, which may be this account's; counted, since it cannot be shown not to be.
            if recorded.is_some_and(|profile| {
                catalog
                    .profiles
                    .get(&profile)
                    .is_none_or(|profile| profile.account == name)
            }) {
                running += 1;
            }
        }
        if running > 0 {
            return Err(ProblemDetail::new(
                ErrorKind::AccountInUse,
                StatusCode::CONFLICT,
                format!(
                    "{running} session(s) resident on this server may run on '{name}', whose \
                     credential refreshes under its name; delete them or wait for them to be \
                     evicted"
                ),
            ));
        }
    }
    let reload = Reload::ahead_of_write(&state)?;
    // A login begun under the old name would store its credential there, beside the one the
    // rename moves.
    forget_pending_logins(&state, &name, true).await;
    let token_store = state.shared.store.token_store();
    account::rename_account(&name, &body.name, &token_store)
        .await
        .map_err(|error| door_refusal("failed to rename the account", error))?;
    reload.after_write()?;
    tracing::info!(
        "renamed account '{name}' to '{new}' via HTTP",
        new = body.name
    );
    let catalog = state.shared.providers.catalog();
    let view = account_view(&catalog, &token_store, &body.name).await?;
    Ok(Json(view))
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
        (status = 422, description = "A profile names the account", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn delete_account(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    let catalog = state.shared.providers.catalog();
    let token_store = state.shared.store.token_store();
    if !catalog.accounts.contains_key(&name) {
        // A credential no account claims is this door's to clear too, as it is `meka account
        // remove`'s, so the name is unknown only when neither side has it.
        let orphaned = account::orphaned_accounts(&token_store, &catalog.accounts)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to list credentials", error)
            })?;
        if !orphaned.contains(&name) {
            let configured: Vec<String> = catalog.accounts.keys().cloned().collect();
            return Err(ProblemDetail::new(
                ErrorKind::NotFound,
                StatusCode::NOT_FOUND,
                account::unknown_account(&name, &configured, &orphaned),
            ));
        }
    }
    let reload = Reload::ahead_of_write(&state)?;
    // A login still pending would store a credential under a name no account claims.
    forget_pending_logins(&state, &name, true).await;
    account::remove_account(&name, &token_store)
        .await
        .map_err(|error| door_refusal("failed to remove the account", error))?;
    reload.after_write()?;
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
/// flow at an await to see the cancel and put back what it set aside.
const LOGIN_END_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// Why a browser login failed: what was asked could not be done as asked, the other side did not
/// answer, or meka's own store failed, which a caller is told as a 422, a 502 and a 500.
#[derive(Debug, Clone)]
enum LoginFailure {
    Refused(String),
    Upstream(String),
    /// The sentence is sanitized; the fault is in the log.
    Internal(String),
}

impl LoginFailure {
    /// The problem a caller is answered with.
    fn problem(self) -> ProblemDetail {
        match self {
            Self::Refused(sentence) => refused(sentence),
            Self::Upstream(sentence) => {
                ProblemDetail::new(ErrorKind::Provider, StatusCode::BAD_GATEWAY, sentence)
            }
            Self::Internal(sentence) => ProblemDetail::new(
                ErrorKind::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                sentence,
            ),
        }
    }
}

/// What finished a browser login: `Ok` once the credential is stored, `Err` with why it failed;
/// `None` while it waits.
type LoginOutcome = Arc<std::sync::Mutex<Option<Result<(), LoginFailure>>>>;

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
        Some(Err(
            LoginFailure::Refused(error)
            | LoginFailure::Upstream(error)
            | LoginFailure::Internal(error),
        )) => LoginStatus {
            status: "failed",
            error: Some(error),
        },
    }
}

/// Settle a login that will not complete, so a request waiting on it is answered with `reason`
/// rather than left to the login's lifetime.
fn settle_login(login: &PendingLogin, reason: &str) {
    let mut outcome = crate::sync::lock(&login.outcome);
    if outcome.is_none() {
        *outcome = Some(Err(LoginFailure::Refused(reason.to_string())));
    }
    drop(outcome);
    login.prompt.wake();
}

/// End a login: its flow is told to stop, which it does at its next cancel-aware await and within
/// its own bounds either way, and a request waiting on it is answered with `reason`. Not aborted:
/// an abort skips what the flow puts back on its way out, as an MCP login restores the bundle it
/// cleared.
fn end_login(login: &PendingLogin, reason: &str) {
    login.cancel.cancel();
    settle_login(login, reason);
}

/// The pending logins, by `login_id`. One per subject: beginning another replaces it.
pub(crate) type PendingLogins =
    Arc<std::sync::Mutex<std::collections::HashMap<String, PendingLogin>>>;

/// What a caller is told of an exchange the authorization server refused: its own body, which
/// `[serve] relay_provider_errors` decides whether a caller sees, with the whole in the log.
fn exchange_refusal(relay: bool, account: &str, error: &anyhow::Error) -> String {
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
        if login.expires_at > now {
            return true;
        }
        end_login(login, "the login lapsed");
        false
    });
}

/// Drop the pending login on `subject`, an account's when `account` and an MCP server's
/// otherwise, and wait its task out: for the login replacing it, which binds the same port, and
/// for a rename or a removal of the subject, after which a credential the login stored would sit
/// under a name nothing claims.
async fn forget_pending_logins(state: &ServerState, subject: &str, account: bool) {
    let removed: Vec<PendingLogin> = {
        let mut pending = crate::sync::lock(&state.pending_logins);
        sweep_expired_logins(&mut pending);
        let gone: Vec<String> = pending
            .iter()
            .filter(|(_, login)| login.subject == subject && login.account == account)
            .map(|(login_id, _)| login_id.clone())
            .collect();
        let removed = gone
            .into_iter()
            .filter_map(|login_id| pending.remove(&login_id))
            .collect();
        drop(pending);
        removed
    };
    end_logins(removed, "the login was replaced, or its subject removed").await;
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

/// End each of `removed` with `reason` and wait its task out.
async fn end_logins(removed: Vec<PendingLogin>, reason: &str) {
    for login in removed {
        // Canceled first and aborted only after a grace: a flow at an await that sees the cancel
        // ends on its own and puts back what it set aside, as an MCP login restores the bundle it
        // cleared, where an abort leaves that undone.
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
        (status = 422, description = "No key for a backend that takes one, or one for a backend that logs in through the browser", body = ProblemDetail),
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
    let catalog = state.shared.providers.catalog();
    let Some(config) = catalog.accounts.get(&name) else {
        return Err(not_found("account", &name, catalog.accounts.keys()));
    };
    let backend =
        account::validate_backend(&config.backend).map_err(|error| refused(error.to_string()))?;
    let credential = match (account::credential_kind(backend), body.api_key) {
        (account::CredentialKind::ApiKey, Some(key)) => crate::store::AuthCredential::ApiKey(key),
        (account::CredentialKind::ApiKey, None) => {
            return Err(refused(format!(
                "'{name}' is a '{backend}' account, which logs in with `api_key`"
            )));
        }
        (_, Some(_)) => {
            return Err(refused(format!(
                "'{name}' is a '{backend}' account, which logs in through the browser and takes \
                 no API key"
            )));
        }
        (_, None) => {
            let started = begin_browser_login(&state, &name, backend, config).await?;
            return Ok((StatusCode::ACCEPTED, Json(started)).into_response());
        }
    };
    let token_store = state.shared.store.token_store();
    token_store
        .save_account_credential(&name, &credential)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to store the account credential", error)
        })?;
    tracing::info!("re-authenticated account '{name}' via HTTP");
    let view = account_view(&catalog, &token_store, &name).await?;
    Ok(Json(view).into_response())
}

/// Begin a browser login for `name`, replacing one pending: the flow `meka account login` runs
/// at a terminal, with the request's side as the person's.
async fn begin_browser_login(
    state: &ServerState,
    name: &str,
    backend: crate::config::Backend,
    config: &crate::config::AccountConfig,
) -> Result<LoginStarted, ProblemDetail> {
    let client_id = config.client_id.clone();
    let oauth_token_url = config.oauth_token_url.clone();
    let token_store = state.shared.store.token_store();
    let account = name.to_string();
    let relay = state.config.relay_provider_errors;
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
                // relay_provider_errors` decides; every other failure is meka's own sentence.
                LoginFailure::Refused(
                    if error.downcast_ref::<account::ExchangeRefused>().is_some() {
                        exchange_refusal(relay, &account, &error)
                    } else {
                        tracing::warn!("the browser login for '{account}' ended: {error}");
                        error.to_string()
                    },
                )
            })?;
            token_store
                .save_account_credential(&account, &credential)
                .await
                .map_err(|error| {
                    tracing::error!("failed to store the credential for '{account}': {error}");
                    LoginFailure::Internal(
                        "the credential could not be stored; see the server log".to_string(),
                    )
                })?;
            tracing::info!("logged in account '{account}' from the browser");
            Ok(())
        },
    )
    .await
}

/// Begin a browser login on `subject` over HTTP: the earlier one on it goes first, `flow` runs
/// as a task with an [`HttpLoginPrompt`] as the person's side, and the answer is the URL and
/// what the flow said completes it, once both are out. A flow that ends first is answered with
/// why, and one that produces no URL in time is given up.
async fn begin_login<F>(
    state: &ServerState,
    subject: &str,
    account: bool,
    lifetime: std::time::Duration,
    flow: impl FnOnce(Arc<HttpLoginPrompt>) -> F,
) -> Result<LoginStarted, ProblemDetail>
where
    F: std::future::Future<Output = Result<(), LoginFailure>> + Send + 'static,
{
    let noun = if account { "account" } else { "MCP server" };
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
    crate::sync::lock(&state.pending_logins).insert(login_id.clone(), PendingLogin {
        subject: subject.to_string(),
        account,
        prompt: Arc::clone(&prompt),
        expires_at: std::time::Instant::now() + lifetime,
        outcome: Arc::clone(&outcome),
        cancel,
        task,
    });
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
            return Err(match settled {
                Ok(()) => {
                    ProblemDetail::internal_sanitized("the login ended before it began", subject)
                }
                Err(failure) => failure.problem(),
            });
        }
        tokio::select! {
            () = prompt.changed.notified() => {}
            () = tokio::time::sleep_until(deadline) => {
                if let Some(login) = crate::sync::lock(&state.pending_logins).remove(&login_id) {
                    end_login(&login, "the login produced no authorization URL in time");
                }
                return Err(ProblemDetail::new(
                    ErrorKind::Provider,
                    StatusCode::BAD_GATEWAY,
                    format!(
                        "the login for {noun} '{subject}' produced no authorization URL within \
                         {}s; the reason is in the meka log",
                        LOGIN_URL_WAIT.as_secs()
                    ),
                ));
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
    Ok(started)
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
            return settled.map_err(LoginFailure::problem);
        }
        if let Some(reason) = crate::sync::lock(&prompt.paste_refusal).take() {
            return Err(refused(reason));
        }
        tokio::select! {
            () = prompt.changed.notified() => {}
            () = tokio::time::sleep_until(expires_at.into()) => {
                return Err(ProblemDetail::new(
                    ErrorKind::Provider,
                    StatusCode::BAD_GATEWAY,
                    format!("the login for {noun} '{subject}' did not complete in time"),
                ));
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
/// failed answers 422 with why. A code the authorization server refuses ends the login, since the
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
        (status = 422, description = "No code in what was pasted, another login's state, or an exchange the authorization server refused", body = ProblemDetail),
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

/// Refuse a setting on a profile whose account's backend never sends it, as `meka profile set`
/// does: a body states exactly what it means to write. An account the catalog lacks or whose
/// backend does not parse is left to the write door, which refuses the account by name.
fn refuse_inert_keys(
    catalog: &crate::config::ProfileCatalog,
    account: &str,
    keys: impl IntoIterator<Item = &'static str>,
) -> Result<(), ProblemDetail> {
    let Some(backend) = catalog
        .accounts
        .get(account)
        .and_then(|config| config.backend.parse::<crate::config::Backend>().ok())
    else {
        return Ok(());
    };
    for key in keys {
        if !backend.reads_profile_key(key) {
            return Err(refused(format!(
                "a profile on '{account}', a '{backend}' account, never sends `{key}`; drop it"
            )));
        }
    }
    Ok(())
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
        (status = 422, description = "Invalid body: a name in use, an account not configured, a setting the backend never sends, or a profile that could not run", body = ProblemDetail),
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
    let name = body.name.trim();
    if name.is_empty() {
        return Err(refused("profile name cannot be empty"));
    }
    let model = crate::config::require_model(name, Some(&body.model))
        .map_err(|error| refused(error.sentence()))?;
    let catalog = state.shared.providers.catalog();
    let stated: Vec<&'static str> = [
        ("context_window", body.context_window.is_some()),
        ("max_output_tokens", body.max_output_tokens.is_some()),
        ("max_request_bytes", body.max_request_bytes.is_some()),
        ("effort", body.effort.is_some()),
        ("vision", body.vision.is_some()),
        ("deferred_tools", body.deferred_tools.is_some()),
        ("thinking", body.thinking.is_some()),
        ("thinking_budget", body.thinking_budget.is_some()),
        ("cache_control", body.cache_control.is_some()),
        ("thinking_display", body.thinking_display.is_some()),
    ]
    .into_iter()
    .filter_map(|(key, given)| given.then_some(key))
    .collect();
    refuse_inert_keys(&catalog, &body.account, stated)?;
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
    let reload = Reload::ahead_of_write(&state)?;
    profile::write_profile(name, &body.account, model, &tuning, body.default)
        .map_err(|error| door_refusal("failed to write the profile", error))?;
    reload.after_write()?;
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
    /// refuses one no profile may carry, in the words `meka profile set` would; the one refusal
    /// made here is the one whose remedy is this surface's.
    fn into_changes(self) -> Result<profile::ProfileChanges, ProblemDetail> {
        if let Patch::Set(names) = &self.fallbacks
            && names.iter().all(|name| name.trim().is_empty())
        {
            return Err(refused(
                "fallbacks must name at least one profile; send null to remove the setting",
            ));
        }
        Ok(profile::ProfileChanges {
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
        })
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
        (status = 422, description = "Invalid body: a value the key refuses, a setting the backend never sends, a profile that could not run on the result, `default: false`, or a `name` sent beside other changes", body = ProblemDetail),
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
    let catalog = state.shared.providers.catalog();
    let Some(profile) = catalog.profiles.get(&name) else {
        return Err(not_found("profile", &name, catalog.profiles.keys()));
    };
    if body.default == Some(false) {
        return Err(refused(
            "`default: false` unsets nothing; remove the profile or make another the default",
        ));
    }
    let make_default = body.default == Some(true);
    let new_name = body.name.clone();
    let account = profile.account.clone();
    let changes = body.into_changes()?;
    let reload = Reload::ahead_of_write(&state)?;
    let final_name = match new_name {
        Some(new_name) => {
            if make_default || !changes.is_empty() {
                return Err(refused(
                    "`name` renames the profile and is sent on its own; change its settings in \
                     another request",
                ));
            }
            profile::rename_profile(&name, &new_name, &state.shared.store)
                .await
                .map_err(|error| door_refusal("failed to rename the profile", error))?;
            tracing::info!("renamed profile '{name}' to '{new_name}' via HTTP");
            new_name
        }
        None => {
            refuse_inert_keys(&catalog, &account, changes.set_keys())?;
            let keys = changes.keys();
            profile::edit_profile(&name, &changes, make_default)
                .map_err(|error| door_refusal("failed to write the profile", error))?;
            if make_default {
                tracing::info!("default profile set to '{name}' via HTTP");
            }
            if !keys.is_empty() {
                tracing::info!(
                    "changed profile '{name}' via HTTP: {keys}",
                    keys = keys.join(", ")
                );
            }
            name
        }
    };
    reload.after_write()?;
    let catalog = state.shared.providers.catalog();
    Ok(Json(profile_detail(&catalog, &final_name)?))
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
        (status = 422, description = "Another profile names it in `fallbacks`", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn delete_profile(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    let catalog = state.shared.providers.catalog();
    if !catalog.profiles.contains_key(&name) {
        return Err(not_found("profile", &name, catalog.profiles.keys()));
    }
    let reload = Reload::ahead_of_write(&state)?;
    profile::remove_profile(&name, &state.shared.store)
        .await
        .map_err(|error| door_refusal("failed to remove the profile", error))?;
    reload.after_write()?;
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

/// What an MCP door's refusal is on the wire: the door's own sentence for what was sent, the
/// sanitized 500 for a fault of the file or the store, through the one classification every
/// error has.
fn mcp_refusal(state: &ServerState, error: crate::error::MekaError) -> ProblemDetail {
    ProblemDetail::for_error(&error, state.config.relay_provider_errors)
}

/// A configured server's entry and where its connection stands: `GET /v1/mcp/{name}`, and what
/// every write answers with.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct McpServerStatus {
    #[serde(flatten)]
    pub(crate) detail: crate::view::McpServerDetail,
    /// `pending`, `connected`, `failed` or `disabled`.
    pub(crate) state: String,
}

async fn server_status(
    state: &ServerState,
    manager: &crate::mcp::McpClientManager,
    name: &str,
) -> Result<McpServerStatus, ProblemDetail> {
    let config = configured_server(manager, name)?;
    let detail =
        crate::installation::mcp::server_detail(&config, &state.shared.store.token_store())
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to read the server", error)
            })?;
    let connection = match manager.server_entry(name) {
        Some(entry) => entry.state().await.label().to_string(),
        None => crate::mcp::ServerState::Disabled.label().to_string(),
    };
    Ok(McpServerStatus {
        detail,
        state: connection,
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
        (status = 422, description = "Invalid body: a name in use or reserved, keys that do not fit the transport, or a secret the entry cannot present", body = ProblemDetail),
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
    let _doors = state.mcp_doors.lock().await;
    crate::installation::mcp::add_server(&body.server, &secrets, &state.shared.store.token_store())
        .await
        .map_err(|error| mcp_refusal(&state, error))?;
    let name = body.server.name.clone();
    manager
        .add_server(&body.server)
        .map_err(|error| ProblemDetail::internal_sanitized("failed to start the server", error))?;
    tracing::info!("added MCP server '{name}' via HTTP");
    Ok((
        StatusCode::CREATED,
        Json(server_status(&state, &manager, &name).await?),
    ))
}

/// The body of `PATCH /v1/mcp/{name}`: any of the entry's keys, each left out to keep, `null` to
/// remove, or a value to set, and the two secrets, each `null` to clear the kind. A string key
/// reads `""` as `null`. `name` and `transport` are not changed: another transport is another
/// server.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct PatchMcpServerRequest {
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) command: Patch<String>,
    #[serde(default)]
    #[schema(value_type = Option<Vec<String>>)]
    pub(crate) args: Patch<Vec<String>>,
    #[serde(default)]
    #[schema(value_type = Option<std::collections::HashMap<String, String>>)]
    pub(crate) env: Patch<std::collections::HashMap<String, String>>,
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) url: Patch<String>,
    #[serde(default)]
    #[schema(value_type = Option<std::collections::HashMap<String, String>>)]
    pub(crate) headers: Patch<std::collections::HashMap<String, String>>,
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
    #[serde(default)]
    #[schema(value_type = Option<std::collections::HashMap<String, String>>)]
    pub(crate) tool_permissions:
        Patch<std::collections::HashMap<String, crate::permission::Permission>>,
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

/// `PATCH /v1/mcp/{name}`: change a server's entry in place and put the changed server in place
/// of the running one, which reconnects it; a `disabled: true` stops it, `false` starts it.
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
        (status = 422, description = "Invalid body: keys that do not fit the transport, an `auth` block beside a stored bearer, or a secret the entry cannot present", body = ProblemDetail),
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
    configured_server(&manager, &name)?;
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
    // A login still pending was begun for the entry as it was: a bundle it minted would be for an
    // endpoint or an `auth` the entry no longer states.
    forget_pending_logins(&state, &name, false).await;
    let _doors = state.mcp_doors.lock().await;
    let config =
        crate::installation::mcp::edit_server(&name, &changes, &state.shared.store.token_store())
            .await
            .map_err(|error| mcp_refusal(&state, error))?;
    manager.replace_server(&config).await.map_err(|error| {
        ProblemDetail::internal_sanitized("failed to restart the server", error)
    })?;
    tracing::info!("changed MCP server '{name}' via HTTP");
    Ok(Json(server_status(&state, &manager, &name).await?))
}

/// `DELETE /v1/mcp/{name}`: remove a server from `config.toml` and from the running manager,
/// closing its connection and dropping its tools from every session, and clear the credentials
/// stored under its name, the OAuth grant revoked at the provider first where it can be. A name
/// with credentials and no entry is cleared the same way.
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
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn delete_mcp_server(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    let manager = mcp_manager(&state)?;
    let token_store = state.shared.store.token_store();
    let configured = configured_server(&manager, &name).is_ok();
    if !configured {
        let has_credentials = token_store
            .has_mcp_credentials(&name)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to read the credentials", error)
            })?;
        if !has_credentials {
            return Err(not_found("MCP server", &name, manager.server_names()));
        }
    }
    // A login still pending would store a credential under a name no entry claims.
    forget_pending_logins(&state, &name, false).await;
    let _doors = state.mcp_doors.lock().await;
    crate::installation::mcp::remove_server(&name, &token_store)
        .await
        .map_err(|error| mcp_refusal(&state, error))?;
    if configured && let Err(error) = manager.remove_server(&name).await {
        tracing::warn!(
            "MCP server '{name}' was removed from config.toml but not from the running manager: {error}"
        );
    }
    tracing::info!("removed MCP server '{name}' via HTTP");
    Ok(StatusCode::NO_CONTENT)
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

/// Put `name` back on the running manager from the entry as the file now states it, once a
/// credential is stored: an `auth` block the login wrote is in force, and the login shows as a
/// connection without the client reaching for the reconnect route.
async fn restart_after_login(state: &ServerState, name: &str) {
    let Ok(manager) = mcp_manager(state) else {
        return;
    };
    let _doors = state.mcp_doors.lock().await;
    match crate::installation::mcp::restart_from_file(&manager, name).await {
        Ok(true) => tracing::info!("restarted MCP server '{name}' after its login"),
        Ok(false) => {
            tracing::debug!("MCP server '{name}' is no longer configured after its login");
        }
        Err(error) => {
            tracing::warn!("failed to restart MCP server '{name}' after its login: {error}");
        }
    }
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
/// is restarted from the entry as the file then states it once the login completes.
#[utoipa::path(
    post,
    path = "/v1/mcp/{name}/login",
    tag = "installation",
    params(("name" = String, Path, description = "MCP server name")),
    request_body = McpLoginRequest,
    responses(
        (status = 200, description = "The secret stored; the server as it stands", body = McpServerStatus),
        (status = 202, description = "A browser login begun; open the URL, then complete the login", body = LoginStarted),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "A secret the entry cannot present, both secrets at once, or a login the entry cannot run: a stdio server, or one holding a bearer", body = ProblemDetail),
        (status = 502, description = "The login did not begin: the server or its authorization server did not answer; the reason is in the meka log", body = ProblemDetail),
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
    configured_server(&manager, &name)?;
    let configs = manager.server_configs();
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
        crate::installation::mcp::store_server_secret(&configs, &token_store, &name, kind, &secret)
            .await
            .map_err(|error| mcp_refusal(&state, error))?;
        tracing::info!("stored a credential for MCP server '{name}' via HTTP");
        // In the request, so the answer carries the restarted server's state.
        restart_after_login(&state, &name).await;
        return Ok(Json(server_status(&state, &manager, &name).await?).into_response());
    }

    let server = name.clone();
    let state_for_flow = state.clone();
    let started = begin_login(
        &state,
        &name,
        false,
        // The flow's own bound: its callback wait starts after discovery, and the connect holds
        // both, so the login outlives neither.
        crate::mcp::auth::OAUTH_CALLBACK_TIMEOUT + crate::installation::mcp::PROBE_CONNECT_TIMEOUT,
        move |prompt| async move {
            let login_prompt: Arc<dyn crate::oauth::LoginPrompt> = prompt;
            let result = crate::installation::mcp::login_server(
                &configs,
                &token_store,
                &server,
                login_prompt,
            )
            .await;
            match result {
                Ok(()) => {
                    tracing::info!("logged in MCP server '{server}' from the browser");
                    restart_after_login(&state_for_flow, &server).await;
                    Ok(())
                }
                Err(error) => {
                    tracing::warn!("the browser login for MCP server '{server}' failed: {error}");
                    Err(match error.kind() {
                        // What was asked cannot be done as asked, in the door's own words; a
                        // URL in them may name a key, as a connector's reason may.
                        ErrorKind::InvalidBody => {
                            LoginFailure::Refused(crate::text::redact_urls(&error.sentence()))
                        }
                        _ => LoginFailure::Upstream(
                            "the login did not complete; the reason is in the meka log".to_string(),
                        ),
                    })
                }
            }
        },
    )
    .await?;
    Ok((StatusCode::ACCEPTED, Json(started)).into_response())
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
/// answer waits for the exchange and the restart, so the server it carries is the one the
/// credential connected.
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
        (status = 422, description = "No code in what was posted, another login's state, or a login that had already ended", body = ProblemDetail),
        (status = 502, description = "The exchange at the authorization server failed; the reason is in the meka log", body = ProblemDetail),
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

/// `POST /v1/mcp/{name}/logout`: clear every credential stored for a server, the OAuth grant
/// revoked at the provider first where it can be. A connection already open stays until the
/// server is reconnected.
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
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["config:w"]))
)]
pub(crate) async fn logout_mcp_server(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ConfigWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    let manager = mcp_manager(&state)?;
    let token_store = state.shared.store.token_store();
    if configured_server(&manager, &name).is_err() {
        let has_credentials = token_store
            .has_mcp_credentials(&name)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to read the credentials", error)
            })?;
        if !has_credentials {
            return Err(not_found("MCP server", &name, manager.server_names()));
        }
    }
    // A login still pending would store the credential this clears.
    forget_pending_logins(&state, &name, false).await;
    crate::installation::mcp::logout_server(&manager.server_configs(), &token_store, &name)
        .await
        .map_err(|error| mcp_refusal(&state, error))?;
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
        let changes = body.into_changes().expect("changes");
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
        assert!(rename.into_changes().expect("changes").is_empty());
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
