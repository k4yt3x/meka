//! `meka account` subcommand suite: the accounts in `[accounts.<name>]`, the credential each holds
//! in the database, the OAuth login flows that acquire one, and the read-only account views.
//!
//! An account is where a request goes and who meka is when it arrives: a backend, an endpoint, and
//! the OAuth settings a login needs. What is asked of it is a profile's business, in `profile.rs`.

use std::io::{self, IsTerminal, Write};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};

use crate::{
    cli::AccountAction,
    config,
    oauth::{generate_pkce_pair, generate_state},
    provider::{
        DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID, DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID,
        DEFAULT_CLAUDE_SUBSCRIPTION_TOKEN_URL,
    },
    store::{AuthCredential, TokenStore},
};

/// Claude Code 2.1.263's `REDIRECT_URI`: the hosted page that shows the code the user pastes back.
const REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
/// Claude Code 2.1.263's `CLAUDE_AI_AUTHORIZE_URL`: the consumer login for a claude.ai account, as
/// opposed to its console login at `platform.claude.com/oauth/authorize`.
const AUTHORIZE_URL: &str = "https://claude.com/cai/oauth/authorize";
/// The scope set Claude Code 2.1.263 requests for a claude.ai login, in its order.
const SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code \
                      user:mcp_servers user:file_upload";

/// `chatgpt-subscription` OAuth flow constants. Mirror Codex's first-party CLI: the authorization
/// server lives at `auth.openai.com`, the redirect listener binds on `localhost:1455`.
const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_REDIRECT_PORT: u16 = 1455;
const CODEX_SCOPES: &str =
    "openid profile email offline_access api.connectors.read api.connectors.invoke";
const CODEX_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// Dispatch a `meka account` subcommand.
pub(crate) async fn run(
    action: &AccountAction,
    store: &crate::store::Store,
    cli_args: &crate::cli::Cli,
) -> anyhow::Result<()> {
    let token_store = &store.token_store();
    match action {
        AccountAction::Add {
            name,
            backend,
            base_url,
            oauth_token_url,
            client_id,
            api_key_stdin,
        } => {
            run_add(
                name,
                backend.as_deref(),
                base_url.clone(),
                OAuthSettings {
                    oauth_token_url: oauth_token_url.clone(),
                    client_id: client_id.clone(),
                },
                *api_key_stdin,
                token_store,
            )
            .await
        }
        AccountAction::List { format } => run_list(token_store, *format).await,
        AccountAction::Remove { name } => run_remove(name, token_store).await,
        AccountAction::Login {
            name,
            api_key_stdin,
        } => run_login(name, *api_key_stdin, token_store).await,
        AccountAction::Usage { .. }
        | AccountAction::Whoami { .. }
        | AccountAction::Stats { .. } => run_introspection(store, action, cli_args).await,
    }
}

/// The two OAuth overrides an account may state. Both are flag-only: they replace values meka has
/// to hardcode but does not own, and almost nobody states them.
#[derive(Default)]
struct OAuthSettings {
    oauth_token_url: Option<String>,
    client_id: Option<String>,
}

async fn run_add(
    name: &str,
    backend_flag: Option<&str>,
    base_url_flag: Option<String>,
    settings_flags: OAuthSettings,
    api_key_stdin: bool,
    token_store: &TokenStore,
) -> anyhow::Result<()> {
    if name.trim().is_empty() {
        anyhow::bail!("account name cannot be empty");
    }
    // Every prompt below reads the stdin the key is piped on, so under `--api-key-stdin` a prompt
    // would eat the secret, or write a second piped line to `config.toml` as `base_url`. The
    // backend is required up front and the base URL takes the backend default.
    if api_key_stdin {
        let Some(backend) = backend_flag else {
            anyhow::bail!("`--api-key-stdin` needs `--backend`");
        };
        let backend = validate_backend(backend)?;
        // The same refusal `meka account login` makes: `acquire_credential` ignores this flag for
        // a backend that logs in through the browser, so a script piping a key would hang on a
        // flow it cannot see while the key it piped went unread.
        if !matches!(credential_kind(backend), CredentialKind::ApiKey) {
            anyhow::bail!(
                "'{backend}' logs in through the browser and takes no API key; drop \
                 `--api-key-stdin`"
            );
        }
    }
    // Hard-fails on an unparseable config rather than warning: this guard is the only thing
    // standing between `account add <existing>` and `upsert_account_document` replacing the
    // account's table, and an empty parsed map defeats it.
    let existing = config::load_config_file_or_err()?;
    if existing.accounts.contains_key(name) {
        anyhow::bail!(
            "an account named '{name}' already exists; re-authenticate it with `meka account login \
             {name}`"
        );
    }

    let backend = match backend_flag {
        Some(value) => validate_backend(value)?,
        None => prompt_backend()?,
    };

    let base_url = match base_url_flag {
        Some(url) => Some(url),
        // Not prompted into the pipe the key is on.
        None if api_key_stdin => None,
        None => {
            // Shows the endpoint an empty answer accepts. An empty answer writes nothing: pinning
            // the default into the account would freeze it, and this is the one setting where
            // meka's own value is the right one for almost everybody.
            let prompt = format!(
                "API base URL [{}]: ",
                crate::provider::default_base_url(backend)
            );
            let input = prompt_line(&prompt)?;
            (!input.is_empty()).then_some(input)
        }
    };

    // Ahead of the login, which is the expensive step, and ahead of the write: an endpoint the
    // appended request paths cannot reach would otherwise fail on the first turn, with the account
    // already recorded. Only this backend builds its paths by appending; the others post to
    // whatever `base_url` names.
    if backend == config::Backend::ChatGptSubscription
        && let Some(url) = base_url.as_deref()
    {
        crate::provider::openai::subscription::chatgpt_base_url_shape(url)?;
    }

    let settings = drop_inert_settings(settings_flags, backend);

    // Last, after every prompt: the Codex login races a pasted-callback reader against the loopback
    // callback, and a callback win can leave a stdin read parked, so nothing may read stdin after.
    //
    // Minted under the `client_id` and at the `oauth_token_url` the account is about to record,
    // because those are what every later refresh presents and posts to; a grant issued to the
    // default client and then claimed by a custom one dies at its first refresh.
    let credential = acquire_credential(
        backend,
        api_key_stdin,
        settings.client_id.as_deref(),
        settings.oauth_token_url.as_deref(),
    )
    .await?;

    // The account before the secret, so the half that lands first is the visible half: a failed
    // config write then leaves an account with no credential, which the next run refuses by name,
    // rather than a credential no account names, which only `account list` reports.
    write_account(name, backend, base_url.as_deref(), &settings)?;
    token_store
        .save_account_credential(name, &credential)
        .await?;

    tracing::info!("added account '{name}'");
    Ok(())
}

/// Drop an OAuth override aimed at a backend that never reads it, saying so.
///
/// Through [`config::Backend::reads_account_key`], the one place that knows which backend reads
/// what. Writing the key anyway would produce a setting that reads plausibly and does nothing,
/// which the load-time warning would then report on every start.
fn drop_inert_settings(mut settings: OAuthSettings, backend: config::Backend) -> OAuthSettings {
    let mut dropped: Vec<&str> = Vec::new();
    if !backend.reads_account_key("client_id") && settings.client_id.take().is_some() {
        dropped.push("--client-id");
    }
    if !backend.reads_account_key("oauth_token_url") && settings.oauth_token_url.take().is_some() {
        dropped.push("--oauth-token-url");
    }
    if !dropped.is_empty() {
        tracing::warn!(
            "ignoring {dropped}: a '{backend}' account never reads it",
            dropped = dropped.join(", "),
        );
    }
    settings
}

async fn run_login(
    name: &str,
    api_key_stdin: bool,
    token_store: &TokenStore,
) -> anyhow::Result<()> {
    let config_file = config::load_config_file_or_err()?;
    let Some(account) = config_file.accounts.get(name) else {
        anyhow::bail!(crate::text::unknown_name(
            "account",
            name,
            config_file.accounts.keys()
        ));
    };
    // Before the guard below, which asks `credential_kind` a question it answers `None` to for a
    // backend it does not recognize. Otherwise a typo'd `backend` is diagnosed as a browser login
    // and the user is sent to run the command again without the flag, only to meet the real error
    // then. `run_add` validates first too.
    let backend = validate_backend(&account.backend)?;
    // `acquire_credential` ignores the flag for a backend that logs in through the browser, and a
    // script piping a key to one would hang on an OAuth flow it cannot see. The account names its
    // backend, so this is answerable before anything opens.
    if api_key_stdin && !matches!(credential_kind(backend), CredentialKind::ApiKey) {
        anyhow::bail!(
            "'{name}' is a '{backend}' account, which logs in through the browser and takes no API \
             key; drop `--api-key-stdin`"
        );
    }
    let credential = acquire_credential(
        backend,
        api_key_stdin,
        account.client_id.as_deref(),
        account.oauth_token_url.as_deref(),
    )
    .await?;
    token_store
        .save_account_credential(name, &credential)
        .await?;
    tracing::info!("re-authenticated account '{name}'");
    Ok(())
}

async fn run_remove(name: &str, token_store: &TokenStore) -> anyhow::Result<()> {
    // No configured account required: this is the only path that deletes a credential, so it has
    // to work on one whose `[accounts.<name>]` block was deleted by hand. Both sides are read
    // before either is touched, so a typo fails instead of reporting a removal that removed
    // nothing. `open_document` rather than the parsed config, because `remove` is one of the
    // ways a config.toml meka cannot deserialize gets repaired.
    //
    // Whether the row is *there*, not whether it parses: `load_account_credential` fails on a row
    // it cannot deserialize, and the one surface that removes a corrupt row must not refuse it for
    // being corrupt.
    let has_credential = token_store
        .list_credential_accounts()
        .await?
        .iter()
        .any(|account| account == name);

    // Under its own short-lived guard, dropped before the `await` below. `ConfigFileLock` tracks
    // reentrancy in a thread-local depth counter, so a guard held across an await on a
    // multi-threaded runtime can resume on a worker where the depth reads zero, and a nested
    // acquisition then self-deadlocks on the file lock this process already holds.
    let has_account = {
        let (_lock, _path, document) = open_document()?;
        // Ahead of every write. A profile that names this account cannot run without it, and
        // every session on that profile would refuse to resume; refusing here names what to move
        // first rather than cascading a deletion the user did not ask for.
        refuse_a_referenced_account(&document, name)?;
        table_names(&document, "accounts")
            .iter()
            .any(|account| account == name)
    };
    if !has_account && !has_credential {
        anyhow::bail!("no account or stored credential named '{name}'");
    }

    // Delete the credential first; the config write can still fail, but the secret should go
    // regardless so a `remove` that gets this far always logs you out. A config meka cannot *read*
    // stops the command above instead, with nothing done: an error that leaves the secret deleted
    // reads to the user as "nothing happened", which is the one thing it must not mean.
    token_store.delete_account_credential(name).await?;
    let removed_account = match remove_account_under_lock(name) {
        Ok(removed) => removed,
        Err(error) => {
            // The credential is already gone, so a refusal here must not read as "nothing
            // happened": that is the one thing this command's errors must never mean.
            tracing::warn!(
                "the stored credential for '{name}' is already cleared; log in again with `meka \
                 account login {name}`"
            );
            return Err(error);
        }
    };

    // What the write actually did, not what a probe predicted.
    if removed_account {
        tracing::info!("removed account '{name}'");
    } else {
        tracing::info!("cleared the stored credential for '{name}'; no account was configured");
    }
    Ok(())
}

/// Refuse to remove an account while a profile names it.
///
/// Asked twice by `run_remove`, of the file as it stands each time: ahead of the credential
/// deletion, and again under the lock the config write takes, because a `profile add` naming this
/// account can land between the two and the first answer is stale by then.
fn refuse_a_referenced_account(
    document: &toml_edit::DocumentMut,
    name: &str,
) -> anyhow::Result<()> {
    let referenced_by = profiles_on_account(document, name);
    if referenced_by.is_empty() {
        return Ok(());
    }
    anyhow::bail!(
        "account '{}' is named by profile(s) {}; remove them first with `meka profile remove \
         <name>`",
        name,
        referenced_by.join(", ")
    )
}

/// Remove `[accounts.<name>]` from the file, answering whether there was one to remove.
///
/// The check and the write are one critical section: `_lock` is held to the end, and the document
/// is read under it rather than carried over from `run_remove`'s probe, so the edit applies to the
/// current file. No `await` inside, for the reason `run_remove` gives about `ConfigFileLock`.
fn remove_account_under_lock(name: &str) -> anyhow::Result<bool> {
    let (_lock, path, mut document) = open_document()?;
    refuse_a_referenced_account(&document, name)?;
    let removed = remove_account_document(&mut document, name);
    crate::fs::write_file_atomic(&path, &document.to_string())?;
    Ok(removed)
}

/// The profiles naming `account`, read off the raw document so `remove` can refuse on a config
/// meka cannot otherwise parse.
fn profiles_on_account(document: &toml_edit::DocumentMut, account: &str) -> Vec<String> {
    document
        .get("profiles")
        .and_then(|item| item.as_table_like())
        .map(|profiles| {
            profiles
                .iter()
                .filter(|(_, profile)| {
                    profile
                        .as_table_like()
                        .and_then(|profile| profile.get("account"))
                        .and_then(|item| item.as_str())
                        == Some(account)
                })
                .map(|(name, _)| name.to_string())
                .collect()
        })
        .unwrap_or_default()
}

async fn run_list(
    token_store: &TokenStore,
    format: crate::cli::OutputFormat,
) -> anyhow::Result<()> {
    let config_file = config::load_config_file_or_err()?;
    // Computed before the early return below: "every account is gone but the secrets are still
    // here" is precisely the state worth reporting, and it is the one an early return would hide.
    let orphans = orphaned_accounts(token_store, &config_file).await?;

    let mut views: Vec<crate::view::AccountView> = Vec::with_capacity(config_file.accounts.len());
    for (name, account) in &config_file.accounts {
        // The error arm is its own answer, not a "no". `load_account_credential` fails on a row it
        // cannot deserialize, and reporting that as "never logged in" sends the user to `meka
        // account login`, which writes a new credential over a row they were never told was
        // corrupt.
        let authenticated = match token_store.load_account_credential(name).await {
            Ok(Some(_)) => "yes",
            Ok(None) => "no",
            Err(error) => {
                tracing::warn!("failed to read the stored credential for '{name}': {error}");
                "unreadable"
            }
        };
        views.push(crate::view::AccountView::new(name, account, authenticated));
    }
    if format == crate::cli::OutputFormat::Json {
        crate::cli::write_json_listing("accounts", &views)?;
        report_orphaned_accounts(&orphans)?;
        return Ok(());
    }
    if views.is_empty() {
        crate::streams::write_stderr_line("No accounts.");
        report_orphaned_accounts(&orphans)?;
        return Ok(());
    }
    let rows: Vec<Vec<String>> = views
        .iter()
        .map(|view| {
            vec![
                view.name.clone(),
                view.backend.clone(),
                view.base_url.clone().unwrap_or_else(|| "-".to_string()),
                view.authenticated.to_string(),
            ]
        })
        .collect();
    crate::render::write_stdout(crate::text::format_columns(
        &["Name", "Backend", "Base URL", "Authenticated"],
        &rows,
    ))?;
    report_orphaned_accounts(&orphans)?;
    Ok(())
}

/// Account names holding a stored credential that no configured account claims.
///
/// A credential is keyed by account name and nothing deletes it when the `[accounts.<name>]` block
/// goes away by hand, so an API key or OAuth refresh token can outlive its account indefinitely.
/// This diff is the only thing that can name one, and `account list` is where a name is worth
/// something: `meka account remove <name>` then clears it.
///
/// Safe to compute here only because the caller already failed on an unreadable config. An empty
/// account map that came from a config meka could not parse would report every credential in the
/// database as an orphan.
async fn orphaned_accounts(
    token_store: &TokenStore,
    config_file: &config::ConfigFile,
) -> anyhow::Result<Vec<String>> {
    Ok(token_store
        .list_credential_accounts()
        .await?
        .into_iter()
        .filter(|account| !config_file.accounts.contains_key(account))
        .collect())
}

/// Print the orphan block, if there is one. The names go to stderr with a hint, because the listing
/// is the requested data and this is a diagnostic about the store, so `meka account list
/// 2>/dev/null | awk` must not see it as rows.
fn report_orphaned_accounts(orphans: &[String]) -> anyhow::Result<()> {
    if orphans.is_empty() {
        return Ok(());
    }
    crate::streams::write_stderr_line("");
    crate::streams::write_stderr_line(format!(
        "Stored credentials with no account: {}",
        orphans.join(", ")
    ));
    // The action only. Deleting the block by hand is the usual cause, but an `add` that stored the
    // secret and then failed to write the account leaves the same trace, so a hint that named one
    // cause would send that user looking for an edit they never made.
    crate::render::render_hint("delete one with `meka account remove <name>`");
    Ok(())
}

pub(super) fn validate_backend(value: &str) -> anyhow::Result<config::Backend> {
    value
        .parse::<config::Backend>()
        .map_err(|message| anyhow::anyhow!(message))
}

/// How a backend proves who it is: an interactive OAuth flow, or a key the user pastes.
///
/// Split out from [`acquire_credential`] so the flow can be named without running a login. An
/// exhaustive match over [`config::Backend`], so a backend added there and forgotten here fails to
/// build rather than panicking at the credential step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CredentialKind {
    /// A Claude subscription login, against Anthropic's authorization server.
    ClaudeLogin,
    /// A ChatGPT subscription login, against OpenAI's.
    ChatGptLogin,
    /// A key the user supplies, for any endpoint the backend's protocol reaches.
    ApiKey,
}

/// Which flow acquires this backend's credential, naming the *specific* login rather than "OAuth".
///
/// The distinction is load-bearing: the two subscription flows use different client ids, scopes and
/// callback handling, and they are not interchangeable. The variant carries the vendor so the match
/// decides it: one `OAuth` variant with the vendor picked by a string test inside the dispatch arm
/// would let a third OAuth backend silently receive OpenAI's login while still satisfying a test
/// that only asserted *some* kind existed.
fn credential_kind(backend: config::Backend) -> CredentialKind {
    match backend {
        config::Backend::ClaudeSubscription => CredentialKind::ClaudeLogin,
        config::Backend::ChatGptSubscription => CredentialKind::ChatGptLogin,
        config::Backend::AnthropicMessages
        | config::Backend::OpenAiChatCompletions
        | config::Backend::OpenAiResponses => CredentialKind::ApiKey,
    }
}

/// Acquire a credential for `backend`: run the OAuth flow for OAuth backends, or read an API key
/// (from stdin when `api_key_stdin`, else an interactive prompt) for key backends.
async fn acquire_credential(
    backend: config::Backend,
    api_key_stdin: bool,
    client_id: Option<&str>,
    // The account's `oauth_token_url`, or `None` for the backend's own. Threaded for the same
    // reason `client_id` is: a value that overrides a constant has to override it everywhere the
    // constant appears, the mint included.
    oauth_token_url: Option<&str>,
) -> anyhow::Result<AuthCredential> {
    match credential_kind(backend) {
        CredentialKind::ClaudeLogin => claude_login(client_id, oauth_token_url).await,
        CredentialKind::ChatGptLogin => codex_login(client_id, oauth_token_url).await,
        CredentialKind::ApiKey => {
            let key = match crate::cli::read_secret_from_stdin(api_key_stdin, "API key")? {
                Some(key) => key,
                None => prompt_secret("Enter your API key: ")?,
            };
            if key.is_empty() {
                anyhow::bail!("API key cannot be empty");
            }
            Ok(AuthCredential::ApiKey(key))
        }
    }
}

// ----- Config file editing (toml_edit, comment-preserving) ---------------------------------------

/// Returns the lock alongside the document so a caller cannot read, mutate and write without
/// holding it: the whole point is that the read and the write are one critical section, and a
/// separate `lock_config_file()` call would be a step someone eventually forgets.
///
/// Shared with `profile.rs`, which edits the same file under the same lock.
pub(super) fn open_document() -> anyhow::Result<(
    config::ConfigFileLock,
    std::path::PathBuf,
    toml_edit::DocumentMut,
)> {
    let lock = config::lock_config_file()?;
    let path = crate::paths::config_file_path()
        .ok_or_else(|| anyhow::anyhow!("failed to determine the config directory"))?;
    // Only a genuinely absent file starts from empty. Treating *any* read failure as "" turns
    // "I couldn't read your config" into "your config is blank", and the caller writes that blank
    // document straight back over the real file: one non-UTF-8 byte or a mode-000 file, and
    // `account remove` truncates config.toml to nothing, accounts and MCP servers included. The
    // `meka mcp` editors already tolerate `NotFound` only; this matches them.
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            anyhow::bail!("failed to read config at {}: {}", path.display(), error);
        }
    };
    let document = contents.parse::<toml_edit::DocumentMut>()?;
    Ok((lock, path, document))
}

/// Borrow `[<section>]` as a real (header) table, creating it implicit if absent. Without this,
/// auto-vivifying `document[section][name]` produces an *inline* table, which renders the whole
/// block on one line.
///
/// A section that is present but written some other way is refused, not replaced. Both spellings
/// deserialize, so a listing and the duplicate guard see the same entries either way; treating "not
/// a header table" as "absent" overwrote the lot. `profiles = { work = … }` plus one `meka profile
/// add home` would leave a file naming only `home`, silently and with exit 0.
pub(super) fn ensure_section_table<'a>(
    document: &'a mut toml_edit::DocumentMut,
    section: &str,
) -> anyhow::Result<&'a mut toml_edit::Table> {
    if document.get(section).is_none() {
        let mut table = toml_edit::Table::new();
        // Implicit so the parent emits `[<section>.<name>]` headers rather than a bare
        // `[<section>]`.
        table.set_implicit(true);
        document[section] = toml_edit::Item::Table(table);
    }
    document
        .get_mut(section)
        .and_then(|item| item.as_table_mut())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`{section}` in config.toml is not a section; spell each entry as its own \
                 `[{section}.<name>]` section"
            )
        })
}

/// The names under `[<section>]`, whichever way the section is spelled.
///
/// `as_table_like` rather than `as_table` because an inline `accounts = { … }` deserializes and so
/// is a config meka runs on; a probe that could not see it reported entries as absent while they
/// were in the file.
pub(super) fn table_names(document: &toml_edit::DocumentMut, section: &str) -> Vec<String> {
    document
        .get(section)
        .and_then(|item| item.as_table_like())
        .map(|table| table.iter().map(|(name, _)| name.to_string()).collect())
        .unwrap_or_default()
}

/// Refuse a write that leaves `config.toml` unreadable, before the file is touched, and hand back
/// the parsed result for the caller's own checks.
///
/// `before` is what the file said when the lock was taken. The reparse covers the whole document,
/// so an unrelated defect anywhere in it would otherwise be reported as something this command
/// did; asking whether it already failed is the difference between naming the line the user must
/// fix and blaming an edit that had nothing to do with it. It stays a refusal either way, because
/// writing on top of a file meka cannot read would strand the rest of its contents.
pub(super) fn reparse_after_edit(before: &str, after: &str) -> anyhow::Result<config::ConfigFile> {
    toml::from_str(after).map_err(|error| {
        if toml::from_str::<config::ConfigFile>(before).is_err() {
            anyhow::anyhow!(
                "config.toml already fails to parse, so this is not what broke it: {error}"
            )
        } else {
            anyhow::anyhow!("that change makes config.toml unreadable: {error}")
        }
    })
}

/// Insert `[accounts.<name>]` into `document`. Pure mutation so it can be unit-tested without
/// touching the filesystem.
fn upsert_account_document(
    document: &mut toml_edit::DocumentMut,
    name: &str,
    backend: config::Backend,
    base_url: Option<&str>,
    settings: &OAuthSettings,
) -> anyhow::Result<()> {
    let mut account = toml_edit::Table::new();
    // Written in [`config::AccountConfig`]'s canonical order. An unset setting is left out rather
    // than written at its default, so the account records only what the user actually chose and a
    // later change to a default reaches existing accounts.
    account.insert("backend", toml_edit::value(backend.name()));
    if let Some(url) = base_url {
        account.insert("base_url", toml_edit::value(url));
    }
    if let Some(url) = settings.oauth_token_url.as_deref() {
        account.insert("oauth_token_url", toml_edit::value(url));
    }
    if let Some(client_id) = settings.client_id.as_deref() {
        account.insert("client_id", toml_edit::value(client_id));
    }
    // Redundant here, since the inserts above already run in order, and deliberately kept: it is
    // the one line that makes "every writer leaves the canonical order" true of this writer too.
    let mut account = toml_edit::Item::Table(account);
    config::sort_account_keys(&mut account);
    ensure_section_table(document, "accounts")?.insert(name, account);
    Ok(())
}

/// Remove `[accounts.<name>]` from `document`, answering whether there was one to remove.
fn remove_account_document(document: &mut toml_edit::DocumentMut, name: &str) -> bool {
    document
        .get_mut("accounts")
        .and_then(|item| item.as_table_like_mut())
        .and_then(|accounts| accounts.remove(name))
        .is_some()
}

fn write_account(
    name: &str,
    backend: config::Backend,
    base_url: Option<&str>,
    settings: &OAuthSettings,
) -> anyhow::Result<()> {
    // `_lock` is held to the end of the function, so the read above and the write below are one
    // critical section.
    let (_lock, path, mut document) = open_document()?;
    // The same refusal `run_add` gave before its prompts, asked again now that the lock is held:
    // the prompts and the OAuth flow between the two can take minutes, and an account written in
    // that window by another `account add` or by hand was replaced wholesale.
    if document
        .get("accounts")
        .and_then(|accounts| accounts.get(name))
        .is_some()
    {
        anyhow::bail!(
            "an account named '{name}' already exists; re-authenticate it with `meka account login \
             {name}`"
        );
    }
    let before = document.to_string();
    upsert_account_document(&mut document, name, backend, base_url, settings)?;
    let after = document.to_string();
    reparse_after_edit(&before, &after)?;
    crate::fs::write_file_atomic(&path, &after)?;
    Ok(())
}

// ----- Interactive prompts -----------------------------------------------------------------------

/// A `[y/N]` prompt: anything but `y` / `yes` (case-insensitively) is a no, **including end of
/// input**. An optional prompt must not be able to fail a run that would otherwise succeed: with
/// stdin closed or redirected, `profile add` takes the defaults and carries on.
pub(super) fn prompt_yes_no(prompt: &str) -> io::Result<bool> {
    match prompt_line(prompt) {
        Ok(answer) => Ok(matches!(answer.to_ascii_lowercase().as_str(), "y" | "yes")),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(error) => Err(error),
    }
}

/// Read a line without echoing it, so a pasted API key is not left on screen, in a scrollback
/// buffer, or in a screen recording.
///
/// Falls back to a visible prompt where echo cannot be suppressed (not a tty, or a platform without
/// termios), and says so, because silently echoing a secret the caller asked to hide is worse than
/// the visible prompt they can decide about.
fn prompt_secret(prompt: &str) -> io::Result<String> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;

        let fd = io::stdin().as_raw_fd();
        // SAFETY: `fd` is stdin's descriptor and `termios` is a valid out-parameter for the
        // duration of the call. `tcgetattr` only writes through it.
        let mut original: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(fd, &mut original) } == 0 {
            let mut quiet = original;
            quiet.c_lflag &= !libc::ECHO;
            // SAFETY: `quiet` is a termios obtained from this same descriptor with one flag
            // cleared.
            if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &quiet) } == 0 {
                // Restored on every exit, including the one the ordinary code path cannot see:
                // Ctrl-C at this prompt is normal, and SIGINT's default disposition kills the
                // process where it stands, leaving a shell that shows nothing typed until
                // `stty sane`.
                let _echo = EchoGuard::install(fd, original);
                let result = prompt_line(prompt);
                // The Enter the user pressed was not echoed either, so the cursor is still on the
                // prompt line.
                drop(_echo);
                crate::streams::write_stderr_line("");
                return result;
            }
        }
        // `warn!`, not `debug!`: this is a secret about to appear on screen, and the doc above
        // promises meka says so rather than quietly echoing it.
        tracing::warn!("failed to disable terminal echo; the API key will be visible as typed");
    }
    prompt_line(prompt)
}

/// Restores terminal echo when dropped *and* when the process is interrupted.
///
/// The handler is what makes this more than a `Drop` impl. `Drop` covers a return and a panic;
/// SIGINT bypasses both. It runs only `tcsetattr` and `_exit`, which are async-signal-safe, and
/// reads the saved settings through an `AtomicPtr` because a lock is not.
#[cfg(unix)]
struct EchoGuard {
    fd: std::os::unix::io::RawFd,
    original: libc::termios,
    previous_handler: libc::sighandler_t,
}

#[cfg(unix)]
static SAVED_TERMIOS: std::sync::atomic::AtomicPtr<libc::termios> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
#[cfg(unix)]
static SAVED_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

#[cfg(unix)]
extern "C" fn restore_echo_on_interrupt(_signal: libc::c_int) {
    let saved = SAVED_TERMIOS.load(std::sync::atomic::Ordering::Acquire);
    let fd = SAVED_FD.load(std::sync::atomic::Ordering::Acquire);
    if !saved.is_null() && fd >= 0 {
        // SAFETY: `saved` was published by `EchoGuard::install` from a live `Box` that outlives the
        // guard, and `fd` is the descriptor those settings came from. `tcsetattr` is
        // async-signal-safe.
        unsafe { libc::tcsetattr(fd, libc::TCSANOW, saved) };
    }
    // 128 + SIGINT, the conventional status, and `_exit` rather than `exit` because only the
    // former is async-signal-safe.
    unsafe { libc::_exit(130) };
}

#[cfg(unix)]
impl EchoGuard {
    fn install(fd: std::os::unix::io::RawFd, original: libc::termios) -> Self {
        // Leaked deliberately: the handler may read it at any point until the guard is dropped, and
        // freeing it on the drop path would race a signal arriving in the same instant. One
        // termios per prompt is a rounding error against a process that is about to hold a
        // conversation in memory.
        let saved = Box::into_raw(Box::new(original));
        SAVED_TERMIOS.store(saved, std::sync::atomic::Ordering::Release);
        SAVED_FD.store(fd, std::sync::atomic::Ordering::Release);
        // SAFETY: installing a handler for SIGINT; the function pointer is a valid `extern "C"`
        // handler and the returned value is the previous disposition, restored on drop.
        let previous_handler = unsafe {
            libc::signal(
                libc::SIGINT,
                restore_echo_on_interrupt as *const () as libc::sighandler_t,
            )
        };
        Self {
            fd,
            original,
            previous_handler,
        }
    }
}

#[cfg(unix)]
impl Drop for EchoGuard {
    fn drop(&mut self) {
        // SAFETY: `self.original` came from this descriptor, and the handler is being put back to
        // whatever it was before `install`.
        unsafe {
            libc::tcsetattr(self.fd, libc::TCSANOW, &self.original);
            libc::signal(libc::SIGINT, self.previous_handler);
        }
        SAVED_TERMIOS.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
        SAVED_FD.store(-1, std::sync::atomic::Ordering::Release);
    }
}

pub(super) fn prompt_line(prompt: &str) -> io::Result<String> {
    crate::streams::write_stderr(prompt);
    // The prompt went to stderr, so that is what has to be flushed; flushing stdout left the
    // prompt sitting in stderr's buffer and the user staring at a blank line.
    io::stderr().flush()?;
    let mut input = String::new();
    // `read_line` reports end of input as `Ok(0)` with an empty buffer, which is indistinguishable
    // from a bare Enter unless the count is checked. A caller that re-prompts on a bad answer would
    // otherwise spin forever against a closed stdin, which `prompt_backend` did.
    if io::stdin().read_line(&mut input)? == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "no more input to read",
        ));
    }
    Ok(input.trim().to_string())
}

/// The `account add` menu.
///
/// Ordered subscription-then-key within each vendor, and `openai-responses` directly under
/// `openai-chat-completions` because the two are a protocol choice against the same key: Responses
/// is what new work should use, Chat Completions is what a server that doesn't serve Responses
/// still speaks. The menu entries, in the order they are offered. Separate from [`prompt_backend`]
/// so a test can check it against [`config::Backend::ALL`]: this is the last hand-written backend
/// list, and a backend missing from it is simply never offered interactively, with nothing failing.
fn backend_menu() -> [(config::Backend, &'static str); 5] {
    [
        (
            config::Backend::ClaudeSubscription,
            "Claude subscription login",
        ),
        (
            config::Backend::AnthropicMessages,
            "Anthropic Messages API key",
        ),
        (
            config::Backend::ChatGptSubscription,
            "ChatGPT subscription login",
        ),
        (
            config::Backend::OpenAiChatCompletions,
            "OpenAI-compatible Chat Completions API key",
        ),
        (
            config::Backend::OpenAiResponses,
            "OpenAI-compatible Responses API key",
        ),
    ]
}

fn prompt_backend() -> anyhow::Result<config::Backend> {
    let options = backend_menu();
    crate::streams::write_stderr_line("Select a backend:");
    for (index, (id, label)) in options.iter().enumerate() {
        crate::streams::write_stderr_line(format!("  {}. {} ({})", index + 1, id, label));
    }
    loop {
        let input = prompt_line("> ")?;
        if let Ok(choice) = input.parse::<usize>()
            && let Some((backend, _)) = choice.checked_sub(1).and_then(|index| options.get(index))
        {
            return Ok(*backend);
        }
        crate::streams::write_stderr_line(format!(
            "Enter a number between 1 and {}.",
            options.len()
        ));
    }
}

// ----- Claude OAuth (paste-back) -----------------------------------------------------------------

fn build_authorize_url(
    client_id: &str,
    code_challenge: &str,
    state: &str,
) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(AUTHORIZE_URL)?;
    url.query_pairs_mut()
        .append_pair("code", "true")
        .append_pair("client_id", client_id)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPES)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state);
    Ok(url.to_string())
}

async fn claude_login(
    client_id: Option<&str>,
    oauth_token_url: Option<&str>,
) -> anyhow::Result<AuthCredential> {
    let client_id = client_id.unwrap_or(DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID);
    let token_url = oauth_token_url.unwrap_or(DEFAULT_CLAUDE_SUBSCRIPTION_TOKEN_URL);
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_state();
    let url = build_authorize_url(client_id, &code_challenge, &state)?;

    crate::streams::write_stderr_line("\nTo authorize, open this URL in your browser:");
    crate::streams::write_stderr_line(format!("    {url}\n"));

    let code_input = prompt_line("After authorizing, paste the authorization code here:\n> ")?;
    if code_input.is_empty() {
        anyhow::bail!("authorization code cannot be empty");
    }
    // Anthropic's page hands back `code#state`.
    let code = code_input.split('#').next().unwrap_or(&code_input);

    exchange_claude_code(code, &code_verifier, client_id, &state, token_url).await
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
    /// Anthropic's OAuth token response carries the subscriber's account here; its `uuid` is what
    /// Claude Code sends as `metadata.user_id.account_uuid` on every request.
    account: Option<OAuthAccount>,
}

#[derive(serde::Deserialize)]
struct OAuthAccount {
    uuid: String,
}

async fn exchange_claude_code(
    code: &str,
    code_verifier: &str,
    client_id: &str,
    state: &str,
    token_url: &str,
) -> anyhow::Result<AuthCredential> {
    let client = reqwest::Client::new();
    let response = client
        .post(token_url)
        .json(&serde_json::json!({
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "state": state,
        }))
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "token exchange failed ({}): {}",
            status,
            crate::error::render_error_body(&body)
        );
    }

    let token: TokenResponse = response.json().await?;
    let expires_at = token.expires_in.map(|seconds| {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as i64)
            .unwrap_or(0);
        // Saturating: a nonsense `expires_in` should read as "far future" and let the 401 correct
        // it, not overflow to a past instant and refresh on every request.
        seconds
            .checked_mul(1000)
            .map_or(i64::MAX, |millis| now_millis.saturating_add(millis))
    });

    Ok(AuthCredential::OAuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at,
        account_id: token.account.map(|account| account.uuid),
    })
}

// ----- OpenAI Codex OAuth (localhost callback) ---------------------------------------------------

async fn codex_login(
    client_id: Option<&str>,
    oauth_token_url: Option<&str>,
) -> anyhow::Result<AuthCredential> {
    let client_id = client_id.unwrap_or(DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID);
    let token_url = oauth_token_url.unwrap_or(CODEX_TOKEN_URL);
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_state();
    let redirect_uri = format!("http://localhost:{CODEX_REDIRECT_PORT}/auth/callback");
    let url = build_codex_authorize_url(client_id, &code_challenge, &state, &redirect_uri)?;

    // The loopback callback only works when the browser runs on this machine. On a remote/headless
    // box the redirect lands on the user's laptop instead, so a TTY session can still finish by
    // pasting the callback URL; a bind failure there degrades to paste-only rather than aborting.
    let paste_enabled = io::stdin().is_terminal();
    let listener =
        match tokio::net::TcpListener::bind(format!("127.0.0.1:{CODEX_REDIRECT_PORT}")).await {
            Ok(listener) => Some(listener),
            Err(error) if paste_enabled => {
                tracing::warn!(
                    "failed to bind the callback listener on 127.0.0.1:{CODEX_REDIRECT_PORT}: \
                     {error}; falling back to pasting the callback URL"
                );
                None
            }
            Err(error) => {
                anyhow::bail!(
                    "failed to bind the callback listener on 127.0.0.1:{CODEX_REDIRECT_PORT}: \
                     {error}"
                );
            }
        };

    crate::streams::write_stderr_line("\nTo authorize, open this URL in your browser:");
    crate::streams::write_stderr_line(format!("    {url}\n"));
    if listener.is_some() {
        crate::streams::write_stderr_line(format!(
            "Waiting up to {}s for the callback on 127.0.0.1:{}...",
            CODEX_CALLBACK_TIMEOUT.as_secs(),
            CODEX_REDIRECT_PORT
        ));
    }
    if paste_enabled {
        crate::streams::write_stderr_line(
            "If your browser is on another machine, paste the full callback URL here and press Enter.",
        );
    }

    // Race the loopback callback against a pasted-URL reader when both are viable. The accept
    // future carries the timeout that bounds the whole wait; the paste reader parks on EOF so a
    // non-interactive stdin can never win the race against a real callback.
    let (received_code, received_state) = match (listener, paste_enabled) {
        (Some(listener), true) => tokio::select! {
            result = accept_codex_callback(listener, CODEX_CALLBACK_TIMEOUT) => result?,
            result = read_pasted_codex_callback() => result?,
        },
        (Some(listener), false) => accept_codex_callback(listener, CODEX_CALLBACK_TIMEOUT).await?,
        (None, _) => read_pasted_codex_callback().await?,
    };
    if received_state != state {
        anyhow::bail!("OAuth state mismatch; refusing the callback");
    }
    exchange_codex_code(
        &received_code,
        &code_verifier,
        client_id,
        &redirect_uri,
        token_url,
    )
    .await
}

/// Read a manually pasted callback URL from stdin, the fallback for when the loopback callback
/// can't reach this machine. Blank lines re-prompt; an unparseable line prints a hint and
/// re-prompts (a mis-paste must not abort while a real callback may still arrive); EOF parks the
/// future forever so a non-interactive stdin can't win the `select!` against the callback branch.
async fn read_pasted_codex_callback() -> anyhow::Result<(String, String)> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let mut reader = BufReader::new(tokio::io::stdin());
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Err(error) => anyhow::bail!("failed to read pasted callback URL: {error}"),
            Ok(0) => std::future::pending::<()>().await,
            Ok(_) => {
                if line.trim().is_empty() {
                    continue;
                }
                match extract_codex_paste(&line) {
                    CodexCallback::Match { code, state } => return Ok((code, state)),
                    CodexCallback::AuthError(message) => {
                        anyhow::bail!("the authorization server rejected the login: {message}")
                    }
                    CodexCallback::NotCallback | CodexCallback::Malformed(_) => {
                        crate::streams::write_stderr_line(
                            "No 'code' and 'state' in that input; paste the full callback URL \
                             and press Enter."
                                .to_string(),
                        );
                        continue;
                    }
                }
            }
        }
    }
}

fn build_codex_authorize_url(
    client_id: &str,
    code_challenge: &str,
    state: &str,
    redirect_uri: &str,
) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(CODEX_AUTHORIZE_URL)?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", client_id)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", CODEX_SCOPES)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("state", state)
        .append_pair("originator", "meka_cli");
    Ok(url.to_string())
}

async fn accept_codex_callback(
    listener: tokio::net::TcpListener,
    timeout: std::time::Duration,
) -> anyhow::Result<(String, String)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            anyhow::bail!("authorization timed out after {}s", timeout.as_secs());
        }
        let (mut stream, _) = match tokio::time::timeout(remaining, listener.accept()).await {
            Ok(Ok(pair)) => pair,
            Ok(Err(error)) => {
                anyhow::bail!("failed to accept the OAuth callback connection: {error}")
            }
            Err(_) => anyhow::bail!("authorization timed out after {}s", timeout.as_secs()),
        };

        const MAX_BYTES: usize = 64 * crate::text::KIB;
        let mut buffer = Vec::with_capacity(4096);
        let mut temp = [0u8; 4096];
        let headers_complete = loop {
            if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                break true;
            }
            if buffer.len() >= MAX_BYTES {
                break false;
            }
            let read_remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if read_remaining.is_zero() {
                anyhow::bail!("authorization timed out after {}s", timeout.as_secs());
            }
            match tokio::time::timeout(read_remaining, stream.read(&mut temp)).await {
                Ok(Ok(0)) => break buffer.windows(4).any(|window| window == b"\r\n\r\n"),
                Ok(Ok(n)) => buffer.extend_from_slice(&temp[..n]),
                Ok(Err(error)) => {
                    anyhow::bail!("failed to read the OAuth callback request: {error}")
                }
                Err(_) => anyhow::bail!("authorization timed out after {}s", timeout.as_secs()),
            }
        };

        if !headers_complete {
            if let Err(error) = stream
                .write_all(b"HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await {
                tracing::debug!("failed to answer the callback client: {error}");
            }
            continue;
        }

        let request = String::from_utf8_lossy(&buffer);
        match parse_codex_callback_query(&request) {
            CodexCallback::Match { code, state } => {
                let body = b"<!DOCTYPE html><html><body>\
                    <h1>Codex authorization successful</h1>\
                    <p>You can close this tab and return to meka.</p>\
                    </body></html>";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                if let Err(error) = stream.write_all(response.as_bytes()).await {
                    tracing::debug!("failed to answer the callback client: {error}");
                }
                if let Err(error) = stream.write_all(body).await {
                    tracing::debug!("failed to answer the callback client: {error}");
                }
                return Ok((code, state));
            }
            CodexCallback::NotCallback => {
                if let Err(error) = stream
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await
                {
                    tracing::debug!("failed to answer the callback client: {error}");
                }
                continue;
            }
            CodexCallback::Malformed(message) => {
                if let Err(error) = stream
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await {
                    tracing::debug!("failed to answer the callback client: {error}");
                }
                anyhow::bail!(message);
            }
            CodexCallback::AuthError(message) => {
                if let Err(error) = stream
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await {
                    tracing::debug!("failed to answer the callback client: {error}");
                }
                anyhow::bail!("the authorization server rejected the login: {message}");
            }
        }
    }
}

enum CodexCallback {
    Match { code: String, state: String },
    NotCallback,
    Malformed(String),
    AuthError(String),
}

fn parse_codex_callback_query(request: &str) -> CodexCallback {
    let Some(first_line) = request.lines().next() else {
        return CodexCallback::Malformed("empty HTTP request".to_string());
    };
    let Some(path) = first_line.split_whitespace().nth(1) else {
        return CodexCallback::Malformed("malformed HTTP request line".to_string());
    };
    let (path_component, query_string) = path.split_once('?').unwrap_or((path, ""));
    if !path_component.eq_ignore_ascii_case("/auth/callback") {
        return CodexCallback::NotCallback;
    }
    if query_string.is_empty() {
        return CodexCallback::Malformed("no query parameters in callback URL".to_string());
    }
    code_state_from_query(query_string)
}

/// Extract `(code, state)` from a URL query string (percent-decoded). Shared by the loopback
/// callback and the pasted-URL fallback so both validate identically. An `error` param wins over
/// `code`/`state`, so an explicit authorization denial surfaces as [`CodexCallback::AuthError`].
fn code_state_from_query(query: &str) -> CodexCallback {
    let mut code = None;
    let mut state = None;
    let mut error_param: Option<String> = None;
    // `form_urlencoded` rather than a hand-rolled split, for the reason given at the matching site
    // in `mcp::auth`: a redirect query is form-encoded, so `+` is a space and decoding it as a
    // literal `+` corrupts any value that contains one.
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        let decoded = value.into_owned();
        match key.as_ref() {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error_param = Some(decoded),
            _ => {}
        }
    }

    if let Some(message) = error_param {
        return CodexCallback::AuthError(message);
    }
    match (code, state) {
        (Some(code), Some(state)) => CodexCallback::Match { code, state },
        _ => CodexCallback::Malformed("callback missing 'code' or 'state' parameter".to_string()),
    }
}

/// Parse a manually pasted callback URL, or a bare `code=...&state=...` query. Accepts the full URL
/// the browser tried to load: strips any `#fragment` and takes the substring after the first `?`.
fn extract_codex_paste(input: &str) -> CodexCallback {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return CodexCallback::Malformed("no callback URL pasted".to_string());
    }
    let before_hash = trimmed.split('#').next().unwrap_or(trimmed);
    let query = match before_hash.split_once('?') {
        Some((_, query)) => query,
        None => before_hash,
    };
    code_state_from_query(query)
}

async fn exchange_codex_code(
    code: &str,
    code_verifier: &str,
    client_id: &str,
    redirect_uri: &str,
    token_url: &str,
) -> anyhow::Result<AuthCredential> {
    use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
    let encode = |value: &str| utf8_percent_encode(value, NON_ALPHANUMERIC).to_string();
    let body = format!(
        "grant_type=authorization_code&code={}&redirect_uri={}&client_id={}&code_verifier={}",
        encode(code),
        encode(redirect_uri),
        encode(client_id),
        encode(code_verifier),
    );

    let client = reqwest::Client::new();
    let response = client
        .post(token_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "Codex token exchange failed ({}): {}",
            status,
            crate::error::render_error_body(&body)
        );
    }

    #[derive(serde::Deserialize)]
    struct CodexTokenResponse {
        id_token: Option<String>,
        access_token: String,
        refresh_token: Option<String>,
    }

    let token: CodexTokenResponse = response.json().await?;
    let account_id = token.id_token.as_deref().and_then(extract_codex_account_id);
    let expires_at = extract_jwt_expiration_millis(&token.access_token);

    Ok(AuthCredential::OAuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at,
        account_id,
    })
}

/// Decode an OpenAI id_token JWT and extract `chatgpt_account_id` from the nested
/// `https://api.openai.com/auth` claim. Returns `None` on any failure.
fn extract_codex_account_id(jwt: &str) -> Option<String> {
    let payload = jwt.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get("https://api.openai.com/auth")
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(|id| id.as_str())
        .map(|id| id.to_string())
}

/// Decode the `exp` claim of a JWT (seconds) and return millis, or `None` if missing/malformed.
fn extract_jwt_expiration_millis(jwt: &str) -> Option<i64> {
    let payload = jwt.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    Some(value.get("exp")?.as_i64()? * 1000)
}

/// Local (no-network) auth status for a stored credential: is the token valid, and when does it
/// expire. Serialized as the `auth` block of `meka account whoami --format json`.
#[derive(serde::Serialize)]
struct AuthStatus {
    valid: bool,
    /// Token expiry as Unix seconds (`None` for API keys / no expiry).
    expires_at: Option<i64>,
    /// Seconds until expiry (negative if already expired).
    expires_in_seconds: Option<i64>,
}

impl AuthStatus {
    fn from_credential(credential: &AuthCredential) -> Self {
        match credential {
            AuthCredential::OAuthToken { expires_at, .. } => {
                // `expires_at` is stored as epoch milliseconds.
                let expires_at = expires_at.map(|millis| millis / 1000);
                let expires_in_seconds =
                    expires_at.map(|secs| secs - chrono::Utc::now().timestamp());
                AuthStatus {
                    valid: expires_in_seconds.is_none_or(|remaining| remaining > 0),
                    expires_at,
                    expires_in_seconds,
                }
            }
            AuthCredential::ApiKey(_) => AuthStatus {
                valid: true,
                expires_at: None,
                expires_in_seconds: None,
            },
        }
    }
}

#[derive(serde::Serialize)]
struct UsageOutput<'a> {
    profile: &'a str,
    account: &'a str,
    #[serde(flatten)]
    usage: &'a crate::provider::AccountUsage,
}

#[derive(serde::Serialize)]
struct WhoamiOutput<'a> {
    profile: &'a str,
    account: &'a str,
    backend: &'a str,
    auth: AuthStatus,
    identity: Option<crate::provider::AccountIdentity>,
}

// ----- `meka account usage` / `whoami` / `stats` ------------------------------------------------

/// Which read-only view `run_introspection` renders.
enum View {
    Usage,
    Whoami,
    Stats,
}

/// The read-only account views. Each builds a provider through a profile, because a request needs
/// a model, and reports on the account that profile bills.
async fn run_introspection(
    store: &crate::store::Store,
    action: &crate::cli::AccountAction,
    cli_args: &crate::cli::Cli,
) -> anyhow::Result<()> {
    let (view, profile_arg, format) = match action {
        crate::cli::AccountAction::Usage { profile, format } => {
            (View::Usage, profile.clone(), *format)
        }
        crate::cli::AccountAction::Whoami { profile, format } => {
            (View::Whoami, profile.clone(), *format)
        }
        crate::cli::AccountAction::Stats { profile, format } => {
            (View::Stats, profile.clone(), *format)
        }
        crate::cli::AccountAction::Add { .. }
        | crate::cli::AccountAction::List { .. }
        | crate::cli::AccountAction::Login { .. }
        | crate::cli::AccountAction::Remove { .. } => {
            anyhow::bail!("not an introspection command")
        }
    };

    // `--profile` selects the way it does for a run, and the provider is built the way a run
    // builds one: through the registry, so the account view is of exactly the provider a session
    // on this profile would talk to, credential rotation and all.
    let mut overrides = cli_args.overrides();
    if let Some(name) = profile_arg {
        overrides.profile = Some(name);
    }
    let config = config::ResolvedConfig::resolve(overrides);
    // A parse error leaves `profiles` empty, and the refusal below would then blame a missing
    // profile for a typo in the file; every sibling subcommand fails on the parse error instead.
    config.require_readable_config()?;
    let name = config.default_profile.clone().ok_or_else(|| {
        anyhow::anyhow!(
            config
                .provider_error
                .clone()
                .unwrap_or_else(|| "no profile configured".to_string())
        )
    })?;
    let registry = crate::provider::ProviderRegistry::new(&config, store.token_store());
    let (provider, settings) = registry.resolve(&name).await?;

    match view {
        View::Usage => match provider.fetch_usage().await? {
            Some(usage) => match format {
                crate::cli::OutputFormat::Plain => {
                    crate::render::write_stdout(crate::render::format_account_usage(&usage))?;
                }
                crate::cli::OutputFormat::Json => {
                    let out = UsageOutput {
                        profile: &name,
                        account: &settings.account,
                        usage: &usage,
                    };
                    crate::render::write_stdout_line(&serde_json::to_string_pretty(&out)?)?;
                }
            },
            None => {
                crate::streams::write_stderr_line(format!(
                    "Account usage is not available for account '{}'.",
                    settings.account
                ));
                return Err(crate::AlreadyReported.into());
            }
        },
        View::Whoami => {
            // The identity call may refresh + rotate the token; re-read afterwards so the auth
            // block reflects the current expiry. A failed identity fetch (e.g. re-login needed)
            // still prints the local auth status so scripts can detect it.
            let identity = provider.fetch_identity().await;
            let fresh = store
                .token_store()
                .load_account_credential(&settings.account)
                .await?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no stored credential for account '{0}'; log in with `meka account login \
                         {0}`",
                        settings.account
                    )
                })?;
            let auth = AuthStatus::from_credential(&fresh);
            let identity = match identity {
                Ok(identity) => identity,
                Err(error) => {
                    tracing::warn!("failed to fetch identity: {error}");
                    None
                }
            };
            let out = WhoamiOutput {
                profile: &name,
                account: &settings.account,
                backend: settings.backend.name(),
                auth,
                identity,
            };
            let written = match format {
                crate::cli::OutputFormat::Plain => {
                    crate::render::write_stdout(format_whoami_plain(&out))
                }
                crate::cli::OutputFormat::Json => {
                    crate::render::write_stdout_line(&serde_json::to_string_pretty(&out)?)
                }
            };
            // Checked before the write is, because on this one command the status *is* the answer:
            // a script reads it to decide whether to re-authenticate. Propagating a failed write
            // first would hand back the code for "stdout went away", and a reader that hung up
            // turns that into success, which says the credential is fine when it is not.
            if !out.auth.valid {
                return Err(crate::AlreadyReported.into());
            }
            written?;
        }
        View::Stats => match provider.fetch_history().await? {
            Some(history) => {
                let out = StatsOutput {
                    profile: &name,
                    account: &settings.account,
                    history: &history,
                };
                match format {
                    crate::cli::OutputFormat::Plain => {
                        crate::render::write_stdout(format_stats_plain(&out))?
                    }
                    crate::cli::OutputFormat::Json => {
                        crate::render::write_stdout_line(&serde_json::to_string_pretty(&out)?)?
                    }
                }
            }
            None => {
                crate::streams::write_stderr_line(format!(
                    "Account history is not available for account '{}'.",
                    settings.account
                ));
                return Err(crate::AlreadyReported.into());
            }
        },
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct StatsOutput<'a> {
    profile: &'a str,
    account: &'a str,
    #[serde(flatten)]
    history: &'a crate::provider::UsageHistory,
}

/// Plain-text (ANSI-free) rendering of `meka account stats`.
fn format_stats_plain(out: &StatsOutput<'_>) -> String {
    let history = out.history;
    let mut text = format!("Account history: {} ({})\n", out.account, out.profile);
    let row_tokens = |text: &mut String, label: &str, value: Option<i64>| {
        if let Some(value) = value {
            text.push_str(&format!(
                "  {label:<18} {}\n",
                crate::text::format_token_count(value.max(0) as u64)
            ));
        }
    };
    let row_days = |text: &mut String, label: &str, value: Option<i64>| {
        if let Some(value) = value {
            text.push_str(&format!("  {label:<18} {value} days\n"));
        }
    };
    if let Some(first) = &history.first_used {
        // Trim an RFC 3339 timestamp to just the date for the human view.
        let date = first.split('T').next().unwrap_or(first);
        text.push_str(&format!("  {:<18} {date}\n", "First used:"));
    }
    row_tokens(&mut text, "Lifetime tokens:", history.lifetime_tokens);
    row_tokens(&mut text, "Peak daily:", history.peak_daily_tokens);
    row_days(&mut text, "Current streak:", history.current_streak_days);
    row_days(&mut text, "Longest streak:", history.longest_streak_days);
    if !history.daily.is_empty() {
        text.push_str("  Recent:\n");
        for day in history.daily.iter().rev().take(7) {
            text.push_str(&format!(
                "    {}  {}\n",
                day.date,
                crate::text::format_token_count(day.tokens.max(0) as u64)
            ));
        }
    }
    text
}

/// Plain-text (ANSI-free) rendering of `meka account whoami`.
fn format_whoami_plain(out: &WhoamiOutput<'_>) -> String {
    let mut text = format!(
        "Account: {} ({}, via profile {})\n",
        out.account, out.backend, out.profile
    );
    let auth = match (out.auth.valid, out.auth.expires_in_seconds) {
        (true, Some(secs)) => format!(
            "valid ({})",
            crate::text::format_duration_short(secs.max(0))
        ),
        (true, None) => "valid".to_string(),
        (false, _) => "EXPIRED: run `meka account login`".to_string(),
    };
    text.push_str(&format!("  Auth:          {auth}\n"));
    if let Some(identity) = &out.identity {
        let row = |text: &mut String, label: &str, value: &Option<String>| {
            if let Some(value) = value {
                text.push_str(&format!("  {label:<14} {value}\n"));
            }
        };
        row(&mut text, "Name:", &identity.display_name);
        row(&mut text, "Email:", &identity.email);
        row(&mut text, "Plan:", &identity.plan);
        row(&mut text, "Tier:", &identity.tier);
        row(&mut text, "Subscription:", &identity.subscription_status);
        row(&mut text, "Organization:", &identity.organization);
        row(&mut text, "Role:", &identity.role);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_status_from_credential() {
        let future = crate::store::AuthCredential::OAuthToken {
            access_token: "t".into(),
            refresh_token: None,
            // 1 hour out, in epoch millis.
            expires_at: Some((chrono::Utc::now().timestamp() + 3600) * 1000),
            account_id: None,
        };
        let status = AuthStatus::from_credential(&future);
        assert!(status.valid);
        assert!(status.expires_in_seconds.unwrap() > 3000);

        let expired = crate::store::AuthCredential::OAuthToken {
            access_token: "t".into(),
            refresh_token: None,
            expires_at: Some((chrono::Utc::now().timestamp() - 60) * 1000),
            account_id: None,
        };
        assert!(!AuthStatus::from_credential(&expired).valid);

        // API keys never expire.
        let api = crate::store::AuthCredential::ApiKey("k".into());
        let status = AuthStatus::from_credential(&api);
        assert!(status.valid);
        assert_eq!(status.expires_at, None);
    }

    /// A config meka cannot read must never be treated as a config that is empty: `open_document`
    /// hands its result to `write_file_atomic`, so "" here truncates the user's real file.
    #[test]
    fn open_document_refuses_an_unreadable_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), b"# caf\xe9\n").expect("write");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set → read → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = open_document();
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(_) => panic!("an unreadable config must not parse as empty"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("failed to read config"),
            "{error}"
        );
    }

    /// An absent file is the one case that legitimately starts from empty: `account add` on a
    /// fresh install has no config.toml to read yet.
    #[test]
    fn open_document_starts_empty_when_there_is_no_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = open_document();
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let (_lock, _, document) = result.expect("a missing config is not an error");
        assert!(document.as_table().is_empty());
    }

    /// The credential table is keyed by account name and nothing prunes it, so a hand-deleted
    /// `[accounts.<name>]` block leaves a live API key or refresh token behind. This diff is the
    /// only thing that can name one.
    #[tokio::test]
    async fn orphaned_accounts_names_credentials_no_account_claims() {
        let store = crate::store::Store::for_test().await.token_store();
        for account in ["work", "archive"] {
            store
                .save_account_credential(account, &AuthCredential::ApiKey("key".to_string()))
                .await
                .expect("save");
        }

        // `personal` is configured but never logged in to: the `Authenticated` column's job, not
        // an orphan. The diff runs in one direction only.
        let config_file: config::ConfigFile = toml::from_str(
            "[accounts.work]\nbackend = \"anthropic-messages\"\n\
             [accounts.personal]\nbackend = \"openai-chat-completions\"\n",
        )
        .expect("parse config");

        let orphans = orphaned_accounts(&store, &config_file)
            .await
            .expect("diff credentials against accounts");
        assert_eq!(orphans, vec!["archive".to_string()]);
    }

    /// The sibling of `login_refuses_a_piped_key_for_a_browser_backend`, and asserted together
    /// because the two commands taking the same flag must answer it the same way. `add` knows the
    /// backend too: `--api-key-stdin` requires `--backend`, so the refusal can come before the
    /// browser opens rather than after a script has hung on it.
    #[tokio::test]
    async fn add_refuses_a_piped_key_for_a_browser_backend() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "").expect("write config");
        let store = crate::store::Store::for_test().await.token_store();

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_add(
            "sub",
            Some("claude-subscription"),
            None,
            OAuthSettings::default(),
            true,
            &store,
        )
        .await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("a piped key for an OAuth backend must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("claude-subscription"), "{error}");
        assert!(error.contains("browser"), "{error}");
        // Nothing was written: the refusal lands before the config document is even opened.
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.is_empty(), "{contents}");
    }

    /// A `chatgpt-subscription` `base_url` of neither accepted shape is refused before the browser
    /// opens and before anything is written, by the same predicate the provider builds on.
    #[tokio::test]
    async fn add_refuses_a_chatgpt_base_url_of_neither_shape() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "").expect("write config");
        let store = crate::store::Store::for_test().await.token_store();

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_add(
            "sub",
            Some("chatgpt-subscription"),
            Some("https://chatgpt.com/backend-api".to_string()),
            OAuthSettings::default(),
            false,
            &store,
        )
        .await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("a base_url the request paths cannot be built on must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("/backend-api/codex"), "{error}");
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.is_empty(), "{contents}");
    }

    /// `acquire_credential` ignores `--api-key-stdin` for a backend that logs in through the
    /// browser, so without this a script piping a key to a subscription account would sit on an
    /// OAuth flow it cannot see while its key went unread. `login` knows the backend from the
    /// account, so it can say so before anything opens.
    #[tokio::test]
    async fn login_refuses_a_piped_key_for_a_browser_backend() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[accounts.sub]\nbackend = \"claude-subscription\"\n",
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await.token_store();

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_login("sub", true, &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("a piped key for an OAuth backend must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("claude-subscription"), "{error}");
        assert!(error.contains("browser"), "{error}");
    }

    /// `remove` is the only path that deletes a credential, so requiring a configured account would
    /// leave a hand-deleted account's secret unreachable from every surface meka has.
    #[tokio::test]
    async fn remove_deletes_a_credential_whose_account_is_gone() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "").expect("write config");
        let store = crate::store::Store::for_test().await.token_store();
        store
            .save_account_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save");

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set → run → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("work", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        result.expect("removing an orphaned credential succeeds");

        assert!(
            store
                .load_account_credential("work")
                .await
                .expect("load")
                .is_none(),
            "the stored credential must be gone"
        );
    }

    /// An account a profile still names is not removable: the profile could not run without it and
    /// every session on the profile would refuse to resume. The refusal comes before the credential
    /// is touched, so nothing has been logged out by the time the user reads it.
    #[tokio::test]
    async fn remove_is_refused_while_a_profile_names_the_account() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[accounts.work]\nbackend = \"anthropic-messages\"\n\n\
             [profiles.daily]\naccount = \"work\"\nmodel = \"m\"\n\n\
             [profiles.other]\naccount = \"elsewhere\"\nmodel = \"m\"\n",
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await.token_store();
        store
            .save_account_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("work", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("an account in use must not be removed"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("profile(s) daily;"),
            "the refusal names exactly the profiles on this account: {error}"
        );
        assert!(
            store
                .load_account_credential("work")
                .await
                .expect("load")
                .is_some(),
            "a refused remove must not have logged the account out"
        );
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.contains("[accounts.work]"), "{contents}");
    }

    /// The locked write asks the referenced-by question again, of the file as it stands then.
    ///
    /// `run_remove` probes under one guard and writes under another, with the credential deletion
    /// awaited between them, so a `profile add` naming the account can land in the window and the
    /// probe's answer is stale by the write. Only a refusal under the write's own lock holds.
    #[test]
    fn the_locked_write_refuses_an_account_a_profile_has_since_named() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[accounts.work]\nbackend = \"anthropic-messages\"\n\n\
             [profiles.daily]\naccount = \"work\"\nmodel = \"m\"\n",
        )
        .expect("write config");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = remove_account_under_lock("work");
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(_) => panic!("an account a profile names must not be removed"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("profile(s) daily;"), "{error}");
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.contains("[accounts.work]"), "{contents}");
    }

    /// The read-only views fail on an unreadable `config.toml` the way every sibling does, rather
    /// than reading its empty parse as "no profile configured" and sending the user to add one.
    #[tokio::test]
    async fn an_introspection_command_reports_an_unreadable_config_rather_than_a_missing_profile() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "bogus = 1\n").expect("write config");
        let store = crate::store::Store::for_test().await;
        let cli = <crate::cli::Cli as clap::Parser>::parse_from(["meka"]);
        let action = crate::cli::AccountAction::Whoami {
            profile: None,
            format: crate::cli::OutputFormat::Plain,
        };

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_introspection(&store, &action, &cli).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("an unreadable config must fail the command"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("failed to parse") && error.contains("bogus"),
            "the parse error is the answer: {error}"
        );
        assert!(
            !error.contains("meka account add"),
            "an unreadable file is not a missing profile: {error}"
        );
    }

    /// Without this, `account remove typo` would report `removed account 'typo'` and exit 0 having
    /// done nothing at all.
    #[tokio::test]
    async fn remove_refuses_a_name_with_neither_account_nor_credential() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[accounts.work]\nbackend = \"anthropic-messages\"\n",
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await.token_store();

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("typo", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("removing a name that exists nowhere must fail"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("no account or stored credential"), "{error}");
        // The untouched account is still there: a failed remove must not rewrite anything.
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.contains("[accounts.work]"), "{contents}");
    }

    /// The OAuth overrides reach the account, and only when given; the backend name always does.
    #[test]
    fn the_account_writer_records_what_was_chosen_and_nothing_else() {
        let mut bare = toml_edit::DocumentMut::new();
        upsert_account_document(
            &mut bare,
            "work",
            config::Backend::AnthropicMessages,
            None,
            &OAuthSettings::default(),
        )
        .expect("upsert");
        let rendered = bare.to_string();
        for key in ["base_url", "oauth_token_url", "client_id", "device_id"] {
            assert!(!rendered.contains(key), "{key} written unasked: {rendered}");
        }

        let mut tuned = toml_edit::DocumentMut::new();
        upsert_account_document(
            &mut tuned,
            "sub",
            config::Backend::ClaudeSubscription,
            Some("https://proxy.invalid/v1"),
            &OAuthSettings {
                oauth_token_url: Some("https://auth.invalid/token".to_string()),
                client_id: Some("my-client".to_string()),
            },
        )
        .expect("upsert");
        // Read back through the real parser, so a key written under the wrong name or type is
        // caught here rather than by `deny_unknown_fields` on the user's next run.
        let parsed: config::ConfigFile = toml::from_str(&tuned.to_string()).expect("parses");
        let account = parsed.accounts.get("sub").expect("the account");
        assert_eq!(account.backend, "claude-subscription");
        assert_eq!(
            account.base_url.as_deref(),
            Some("https://proxy.invalid/v1")
        );
        assert_eq!(
            account.oauth_token_url.as_deref(),
            Some("https://auth.invalid/token")
        );
        assert_eq!(account.client_id.as_deref(), Some("my-client"));
    }

    /// An OAuth override aimed at an API-key backend is dropped with a warning rather than written
    /// where nothing reads it; the subscription backends keep it.
    #[test]
    fn an_oauth_override_aimed_at_a_key_backend_is_dropped() {
        let flags = || OAuthSettings {
            oauth_token_url: Some("https://auth.invalid/token".to_string()),
            client_id: Some("my-client".to_string()),
        };
        let key = drop_inert_settings(flags(), config::Backend::OpenAiResponses);
        assert!(key.client_id.is_none() && key.oauth_token_url.is_none());
        let subscription = drop_inert_settings(flags(), config::Backend::ChatGptSubscription);
        assert_eq!(subscription.client_id.as_deref(), Some("my-client"));
        assert_eq!(
            subscription.oauth_token_url.as_deref(),
            Some("https://auth.invalid/token")
        );
    }

    /// `accounts = { work = { … } }` is valid TOML that serde reads. Treating "not a header table"
    /// as "absent" and overwriting would destroy it, so it is refused instead.
    #[test]
    fn adding_an_account_refuses_an_inline_accounts_table_rather_than_replacing_it() {
        let mut document = "accounts = { work = { backend = \"anthropic-messages\" } }\n"
            .parse::<toml_edit::DocumentMut>()
            .expect("parse");
        let error = upsert_account_document(
            &mut document,
            "home",
            config::Backend::OpenAiResponses,
            None,
            &OAuthSettings::default(),
        )
        .expect_err("an inline `accounts` must be refused, not overwritten");
        assert!(
            error.to_string().contains("is not a section"),
            "the refusal must say what to do about it: {error}"
        );
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert!(
            config.accounts.contains_key("work"),
            "the existing account must survive a refused add"
        );
        assert!(
            remove_account_document(&mut document, "work"),
            "removal reaches either spelling"
        );
        assert!(!remove_account_document(&mut document, "work"));
    }

    #[test]
    fn token_response_extracts_account_uuid() {
        let json = serde_json::json!({
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
            "account": { "uuid": "0f0e7b2c-1d3a-4b5c-8e9f-a1b2c3d4e5f6" },
        });
        let token: TokenResponse = serde_json::from_value(json).unwrap();
        assert_eq!(
            token.account.map(|account| account.uuid).as_deref(),
            Some("0f0e7b2c-1d3a-4b5c-8e9f-a1b2c3d4e5f6"),
        );
    }

    #[test]
    fn token_response_without_account_is_none() {
        let json = serde_json::json!({
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
        });
        let token: TokenResponse = serde_json::from_value(json).unwrap();
        assert!(token.account.is_none());
    }

    #[test]
    fn build_authorize_url_contains_params() {
        let url = build_authorize_url("cid", "challenge", "state").unwrap();
        assert!(url.starts_with(AUTHORIZE_URL));
        // The scope set as Claude Code 2.1.263 sends it, captured from its own authorization URL:
        // the same names in the same order, so a login is indistinguishable from the first-party
        // client's.
        assert!(
            url.contains(
                "scope=org%3Acreate_api_key+user%3Aprofile+user%3Ainference+user%3Asessions%3Aclaude_code+user%3Amcp_servers+user%3Afile_upload&"
            ),
            "{url}"
        );
        assert!(url.contains("client_id=cid"));
        assert!(url.contains("code_challenge=challenge"));
        assert!(url.contains("state=state"));
        assert!(url.contains("code_challenge_method=S256"));
    }

    #[test]
    fn build_codex_authorize_url_contains_required_params() {
        let url = build_codex_authorize_url(
            "app_test",
            "ch",
            "st",
            "http://localhost:1455/auth/callback",
        )
        .unwrap();
        assert!(url.starts_with(CODEX_AUTHORIZE_URL));
        assert!(url.contains("client_id=app_test"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("id_token_add_organizations=true"));
        assert!(url.contains("originator=meka_cli"));
    }

    #[test]
    fn a_backend_name_parses_only_when_meka_has_it() {
        assert!(validate_backend("claude-subscription").is_ok());
        assert!(validate_backend("bogus").is_err());
    }

    #[test]
    fn parse_codex_callback_query_match_and_decode() {
        let request = "GET /auth/callback?code=hello%20world&state=s%23t HTTP/1.1\r\n\r\n";
        match parse_codex_callback_query(request) {
            CodexCallback::Match { code, state } => {
                assert_eq!(code, "hello world");
                assert_eq!(state, "s#t");
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn parse_codex_callback_query_non_callback_and_error() {
        assert!(matches!(
            parse_codex_callback_query("GET /favicon.ico HTTP/1.1\r\n\r\n"),
            CodexCallback::NotCallback
        ));
        match parse_codex_callback_query("GET /auth/callback?error=access_denied HTTP/1.1\r\n\r\n")
        {
            CodexCallback::AuthError(message) => assert_eq!(message, "access_denied"),
            _ => panic!("expected AuthError"),
        }
    }

    #[test]
    fn extract_codex_paste_full_url() {
        // A real redirect URL: base64url code with '.'/'-'/'_', '+' in scope, all left intact.
        let url = "http://localhost:1455/auth/callback?code=ac_K0pPCDWiHyd5jWO_FqWeDB8-52rj9Dw-YxgEnqp5HAo.zDNcq5vdPneTieVAgYERv7yr5AhoiUbMAgpMjEuGnhs&scope=openid+profile+email+offline_access+api.connectors.read+api.connectors.invoke&state=uw-OxzrtaH6ZqtaJtuN7dvDZY0eM5ka7yn_zshisEi0";
        match extract_codex_paste(url) {
            CodexCallback::Match { code, state } => {
                assert_eq!(
                    code,
                    "ac_K0pPCDWiHyd5jWO_FqWeDB8-52rj9Dw-YxgEnqp5HAo.zDNcq5vdPneTieVAgYERv7yr5AhoiUbMAgpMjEuGnhs"
                );
                assert_eq!(state, "uw-OxzrtaH6ZqtaJtuN7dvDZY0eM5ka7yn_zshisEi0");
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn extract_codex_paste_bare_query_and_trim_and_fragment() {
        match extract_codex_paste("  code=abc&state=xyz#frag \n") {
            CodexCallback::Match { code, state } => {
                assert_eq!(code, "abc");
                assert_eq!(state, "xyz");
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn extract_codex_paste_error_and_missing() {
        match extract_codex_paste("http://localhost:1455/auth/callback?error=access_denied") {
            CodexCallback::AuthError(message) => assert_eq!(message, "access_denied"),
            _ => panic!("expected AuthError"),
        }
        assert!(matches!(
            extract_codex_paste("http://localhost:1455/auth/callback?code=only"),
            CodexCallback::Malformed(_)
        ));
    }

    #[test]
    fn extract_codex_account_id_and_expiration() {
        let payload = serde_json::json!({
            "exp": 1_700_000_000,
            "https://api.openai.com/auth": { "chatgpt_account_id": "ws-1" }
        });
        let body = URL_SAFE_NO_PAD.encode(payload.to_string().as_bytes());
        let jwt = format!("h.{body}.s");
        assert_eq!(extract_codex_account_id(&jwt).as_deref(), Some("ws-1"));
        assert_eq!(extract_jwt_expiration_millis(&jwt), Some(1_700_000_000_000));
    }

    /// Every supported backend must be offered by the interactive menu.
    ///
    /// `meka account add` with no `--backend` is how most people meet the backend list, and a name
    /// missing from the menu is unreachable that way while failing nothing: no error, no warning,
    /// just a backend the wizard never mentions.
    #[test]
    fn every_supported_backend_is_offered_by_the_menu() {
        let offered: Vec<config::Backend> = backend_menu().iter().map(|(id, _)| *id).collect();
        for backend in config::Backend::ALL {
            assert!(
                offered.contains(&backend),
                "{backend} is supported but is not in the `account add` menu"
            );
        }
        assert_eq!(
            offered.len(),
            config::Backend::ALL.len(),
            "the menu offers a backend twice: {offered:?}"
        );
    }

    /// Each subscription backend gets its *own* vendor's login, not merely "an OAuth flow": the
    /// client ids, scopes and callbacks differ and are not interchangeable.
    ///
    /// That every backend *has* a flow is the compiler's to check: `credential_kind` matches
    /// [`config::Backend`] exhaustively, so no arm can end in `unreachable!()`.
    #[test]
    fn each_subscription_backend_gets_its_own_vendors_login() {
        assert_eq!(
            credential_kind(config::Backend::ClaudeSubscription),
            CredentialKind::ClaudeLogin
        );
        assert_eq!(
            credential_kind(config::Backend::ChatGptSubscription),
            CredentialKind::ChatGptLogin
        );
        assert_eq!(
            credential_kind(config::Backend::OpenAiResponses),
            CredentialKind::ApiKey
        );
    }

    /// The same as the Claude test below, for the other subscription backend.
    ///
    /// A sibling rather than a duplicate: `codex_login` reaches a *different* exchange helper with
    /// a different body encoding and its own hardcoded constant, so the Claude test says nothing
    /// about it. Threading the endpoint into one of two mints and not the other is exactly the
    /// one-door-of-two shape the rest of the suite closes.
    #[tokio::test]
    async fn the_codex_exchange_posts_to_the_profiles_token_endpoint() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let received = std::thread::spawn(move || {
            use std::io::Read;
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buffer = [0_u8; 1024];
            let read = stream.read(&mut buffer).unwrap_or(0);
            String::from_utf8_lossy(&buffer[..read]).to_string()
        });

        let endpoint = format!("http://127.0.0.1:{port}/codex/own/token");
        let _ = exchange_codex_code(
            "code",
            "verifier",
            "a-client",
            "http://localhost:1455/auth/callback",
            &endpoint,
        )
        .await;

        let request = received.join().expect("the stub thread");
        assert!(
            request.starts_with("POST /codex/own/token "),
            "the exchange must post to the account's endpoint, got: {}",
            request.lines().next().unwrap_or_default()
        );
    }

    /// The code exchange goes to the account's endpoint, not the built-in one.
    ///
    /// The account's `client_id` reaches the mint because a grant issued to the default client and
    /// then claimed by a custom one dies at its first refresh. The endpoint is the same rule one
    /// field over: if refresh read `oauth_token_url` while the mint posted to a constant, the
    /// documented pair (`--client-id` with `--oauth-token-url`) could not complete a login at all.
    ///
    /// A real socket rather than a URL assertion, because a helper that took the endpoint and
    /// posted somewhere else would satisfy any signature check. The stub answers nothing useful and
    /// the exchange fails afterwards; that the request *arrived* there is the whole claim.
    ///
    /// The browser leg means no test can drive `claude_login` end to end, so the link between the
    /// account's field and this argument is held by the compiler instead: dropping it makes
    /// `oauth_token_url` an unused parameter, which CI's `-D warnings` turns into a build failure.
    #[tokio::test]
    async fn the_code_exchange_posts_to_the_profiles_token_endpoint() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let received = std::thread::spawn(move || {
            use std::io::Read;
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buffer = [0_u8; 1024];
            let read = stream.read(&mut buffer).unwrap_or(0);
            String::from_utf8_lossy(&buffer[..read]).to_string()
        });

        let endpoint = format!("http://127.0.0.1:{port}/its/own/token");
        // Fails once the stub hangs up; the assertion is about where it went.
        let _ = exchange_claude_code("code", "verifier", "a-client", "state", &endpoint).await;

        let request = received.join().expect("the stub thread");
        assert!(
            request.starts_with("POST /its/own/token "),
            "the exchange must post to the account's endpoint, got: {}",
            request.lines().next().unwrap_or_default()
        );
    }
}
