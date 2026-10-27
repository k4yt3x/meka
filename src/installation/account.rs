//! The accounts in `[accounts.<name>]`, the credential each holds in the database, and the
//! browser login that acquires one: the doors `meka account` and `/v1/accounts` write through,
//! and the shared editor of `config.toml` every door in this module tree uses.
//!
//! An account is where a request goes and who meka is when it arrives: a backend, an endpoint, and
//! the OAuth settings a login needs. What is asked of it is a profile's business, in `profile.rs`.

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};

use crate::{
    config,
    oauth::{Completion, LoginPrompt, Paste, generate_pkce_pair, generate_state},
    provider::{
        DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID, DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID,
        DEFAULT_CLAUDE_SUBSCRIPTION_TOKEN_URL,
    },
    store::{AuthCredential, TokenStore},
};

/// Claude Code 2.1.284's `MANUAL_REDIRECT_URL`: the hosted page that shows the code the user
/// pastes back.
const REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
/// Claude Code 2.1.284's `CLAUDE_AI_AUTHORIZE_URL`: the consumer login for a claude.ai account, as
/// opposed to its console login at `platform.claude.com/oauth/authorize`.
const AUTHORIZE_URL: &str = "https://claude.com/cai/oauth/authorize";
/// The scope set Claude Code 2.1.284 requests for a claude.ai login, in its order.
const SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code \
                      user:mcp_servers user:file_upload user:plugins";

/// `chatgpt-subscription` OAuth flow constants. Mirror Codex's first-party CLI: the authorization
/// server lives at `auth.openai.com`, the redirect listener binds on `localhost:1455`.
const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_REDIRECT_PORT: u16 = 1455;
const CODEX_SCOPES: &str =
    "openid profile email offline_access api.connectors.read api.connectors.invoke";
/// The two OAuth overrides an account may state. Both are flag-only: they replace values meka has
/// to hardcode but does not own, and almost nobody states them.
#[derive(Default)]
pub(crate) struct OAuthSettings {
    pub(crate) oauth_token_url: Option<String>,
    pub(crate) client_id: Option<String>,
}

/// Remove `[accounts.<name>]` and the credential stored under the name, either of which may be
/// all there is: `meka account remove` and `DELETE /v1/accounts/{name}`.
pub(crate) async fn remove_account(name: &str, token_store: &TokenStore) -> anyhow::Result<()> {
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
    let configured = {
        let (_lock, _path, document) = open_document()?;
        // Ahead of every write. A profile that names this account cannot run without it, and
        // every session on that profile would refuse to resume; refusing here names what to move
        // first rather than cascading a deletion the user did not ask for.
        refuse_a_referenced_account(&document, name)?;
        table_names(&document, "accounts")
    };
    let has_account = configured.iter().any(|account| account == name);
    if !has_account && !has_credential {
        let orphaned: Vec<String> = token_store
            .list_credential_accounts()
            .await?
            .into_iter()
            .filter(|account| !configured.contains(account))
            .collect();
        anyhow::bail!(unknown_account(name, &configured, &orphaned));
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
/// Asked twice by `remove_account`, of the file as it stands each time: ahead of the credential
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
        "account '{}' is named by profile(s) {}; remove them first",
        name,
        referenced_by.join(", ")
    )
}

/// Remove `[accounts.<name>]` from the file, answering whether there was one to remove.
///
/// The check and the write are one critical section: `_lock` is held to the end, and the document
/// is read under it rather than carried over from `remove_account`'s probe, so the edit applies
/// to the current file. No `await` inside, for the reason `remove_account` gives about
/// `ConfigFileLock`.
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

/// Rename `[accounts.<name>]`, moving the credential and every profile on it: `meka account
/// rename` and `PATCH /v1/accounts/{name}`.
pub(crate) async fn rename_account(
    name: &str,
    new_name: &str,
    token_store: &TokenStore,
) -> anyhow::Result<()> {
    if new_name.trim().is_empty() {
        anyhow::bail!("account name cannot be empty");
    }
    // Probed under a short guard, dropped before the `await` below, for the reason
    // `remove_account` gives about `ConfigFileLock`; the write asks again under its own.
    {
        let (_lock, _path, document) = open_document()?;
        refuse_an_unrenameable_account(&document, name, new_name)?;
    }
    // A leftover credential under the new name would collide with the moved one, and it is a
    // secret nobody configured: refused by name rather than overwritten.
    if token_store
        .list_credential_accounts()
        .await?
        .iter()
        .any(|account| account == new_name)
    {
        anyhow::bail!("a stored credential named '{new_name}' already exists; remove it first");
    }

    // Held until the row and the file both carry the new name. A refresh in flight in another meka
    // stores its result against the row it read, and with that row moved it finds nothing to
    // update and drops the rotated token, leaving the moved row holding a refresh token the issuer
    // has already spent.
    let Some(_refresh_guard) = token_store.try_lock_account_credential(name)? else {
        anyhow::bail!(
            "account '{name}' is refreshing its credential in another meka; retry shortly"
        );
    };

    // The credential moves first, and moves back if the config write then fails, so the two never
    // disagree for longer than this function runs. The other order has no undo: a config already
    // renamed makes the second half unrepeatable.
    token_store
        .rename_account_credential(name, new_name)
        .await?;
    if let Err(error) = rename_account_under_lock(name, new_name) {
        if let Err(undo) = token_store.rename_account_credential(new_name, name).await {
            tracing::warn!(
                "failed to move the credential back to '{name}': {undo}; log in again with `meka \
                 account login {name}`"
            );
        }
        return Err(error);
    }
    tracing::info!("renamed account '{name}' to '{new_name}'");
    Ok(())
}

/// Refuse a rename that names no account or takes a name in use.
///
/// Asked twice by `rename_account`, of the file as it stands each time, because an `account add`
/// taking the new name can land between the probe and the write.
fn refuse_an_unrenameable_account(
    document: &toml_edit::DocumentMut,
    name: &str,
    new_name: &str,
) -> anyhow::Result<()> {
    let accounts = table_names(document, "accounts");
    if !accounts.iter().any(|account| account == name) {
        anyhow::bail!(crate::text::unknown_name("account", name, &accounts));
    }
    if accounts.iter().any(|account| account == new_name) {
        anyhow::bail!("an account named '{new_name}' already exists");
    }
    Ok(())
}

/// Rename `[accounts.<name>]` and repoint every profile on it, as one critical section: `_lock` is
/// held to the end, and the document is read under it rather than carried over from
/// `rename_account`'s probe. No `await` inside, for the reason `remove_account` gives about
/// `ConfigFileLock`.
fn rename_account_under_lock(name: &str, new_name: &str) -> anyhow::Result<()> {
    let (_lock, path, mut document) = open_document()?;
    refuse_an_unrenameable_account(&document, name, new_name)?;
    rename_table_entry(&mut document, "accounts", name, new_name)?;
    for profile in profiles_on_account(&document, name) {
        if let Some(item) = document
            .get_mut("profiles")
            .and_then(|item| item.as_table_like_mut())
            .and_then(|profiles| profiles.get_mut(&profile))
            .and_then(|item| item.as_table_like_mut())
            .and_then(|profile| profile.get_mut("account"))
        {
            repoint_name(item, new_name);
        }
    }
    crate::fs::write_file_atomic(&path, &document.to_string())?;
    Ok(())
}

/// Rename `[<section>.<name>]` to `[<section>.<new_name>]` where it stands. A header table's place
/// in the file and the comments above it travel with the table itself, not with the key, so taking
/// the entry out and putting it back under the new key moves nothing else; an inline spelling has
/// no position to keep, and its entry lands at the end.
///
/// Shared with `profile.rs`, which renames under `[profiles]` the same way.
pub(super) fn rename_table_entry(
    document: &mut toml_edit::DocumentMut,
    section: &str,
    name: &str,
    new_name: &str,
) -> anyhow::Result<()> {
    let table = document
        .get_mut(section)
        .and_then(|item| item.as_table_like_mut())
        .ok_or_else(|| anyhow::anyhow!("no `[{section}]` table in config.toml"))?;
    let item = table
        .remove(name)
        .ok_or_else(|| anyhow::anyhow!("no `[{section}.{name}]` table in config.toml"))?;
    table.insert(new_name, item);
    Ok(())
}

/// Point a name-valued key at `name`, keeping whatever comment sat beside the old value.
pub(super) fn repoint_name(item: &mut toml_edit::Item, name: &str) {
    let decor = item.as_value().map(|value| value.decor().clone());
    let mut value = toml_edit::Value::from(name);
    if let Some(decor) = decor {
        *value.decor_mut() = decor;
    }
    *item = toml_edit::Item::Value(value);
}

/// Every account as either listing shows it, with whether a credential is stored under its name.
pub(crate) async fn account_views(
    token_store: &TokenStore,
    accounts: &std::collections::BTreeMap<String, config::AccountConfig>,
) -> anyhow::Result<Vec<crate::view::AccountView>> {
    let mut views: Vec<crate::view::AccountView> = Vec::with_capacity(accounts.len());
    for (name, account) in accounts {
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
    Ok(views)
}

/// The sentence for a name that is neither an account nor a stored credential, from a door that
/// takes either: the accounts configured, then the credentials no account claims, each as what it
/// is.
pub(crate) fn unknown_account(name: &str, configured: &[String], orphaned: &[String]) -> String {
    let mut sentence = crate::text::unknown_name("account", name, configured);
    if !orphaned.is_empty() {
        sentence.push_str(&format!(
            "; stored credentials with no account: {}",
            orphaned.join(", ")
        ));
    }
    sentence
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
pub(crate) async fn orphaned_accounts(
    token_store: &TokenStore,
    accounts: &std::collections::BTreeMap<String, config::AccountConfig>,
) -> anyhow::Result<Vec<String>> {
    Ok(token_store
        .list_credential_accounts()
        .await?
        .into_iter()
        .filter(|account| !accounts.contains_key(account))
        .collect())
}

pub(crate) fn validate_backend(value: &str) -> anyhow::Result<config::Backend> {
    value
        .parse::<config::Backend>()
        .map_err(|message| anyhow::anyhow!(message))
}

/// How a backend proves who it is: an interactive OAuth flow, or a key the user pastes.
///
/// Named apart from the CLI's `acquire_credential` so the flow can be asked about without running
/// a login. An exhaustive match over [`config::Backend`], so a backend added there and forgotten
/// here fails to build rather than panicking at the credential step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CredentialKind {
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
pub(crate) fn credential_kind(backend: config::Backend) -> CredentialKind {
    match backend {
        config::Backend::ClaudeSubscription => CredentialKind::ClaudeLogin,
        config::Backend::ChatGptSubscription => CredentialKind::ChatGptLogin,
        config::Backend::AnthropicMessages
        | config::Backend::OpenAiChatCompletions
        | config::Backend::OpenAiResponses => CredentialKind::ApiKey,
    }
}

/// Returns the lock alongside the document so a caller cannot read, mutate and write without
/// holding it: the whole point is that the read and the write are one critical section, and a
/// separate `lock_config_file()` call would be a step someone eventually forgets.
///
/// Shared with `profile.rs`, which edits the same file under the same lock.
pub(super) fn open_document() -> crate::error::Result<(
    config::ConfigFileLock,
    std::path::PathBuf,
    toml_edit::DocumentMut,
)> {
    let lock = config::lock_config_file()
        .map_err(|error| crate::error::MekaError::io(format!("failed to lock config: {error}")))?;
    let path = crate::paths::config_file_path()
        .ok_or_else(|| crate::error::MekaError::io("failed to determine the config directory"))?;
    // Only a genuinely absent file starts from empty. Treating *any* read failure as "" turns
    // "I couldn't read your config" into "your config is blank", and the caller writes that blank
    // document straight back over the real file: one non-UTF-8 byte or a mode-000 file, and
    // `account remove` truncates config.toml to nothing, accounts and MCP servers included. The
    // one opener for every editor of the raw document, `meka mcp`'s included.
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(crate::error::MekaError::io(format!(
                "failed to read config at {}: {}",
                path.display(),
                error
            )));
        }
    };
    let document = contents
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| {
            crate::error::MekaError::Config(format!("failed to parse config: {error}"))
        })?;
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
    interleaved_thinking: Option<bool>,
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
    if let Some(interleaved_thinking) = interleaved_thinking {
        account.insert(
            "interleaved_thinking",
            toml_edit::value(interleaved_thinking),
        );
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

/// Write `[accounts.<name>]`: `meka account add` once it holds the credential, and
/// `POST /v1/accounts`.
pub(crate) fn write_account(
    name: &str,
    backend: config::Backend,
    base_url: Option<&str>,
    settings: &OAuthSettings,
    interleaved_thinking: Option<bool>,
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
        anyhow::bail!("an account named '{name}' already exists");
    }
    let before = document.to_string();
    upsert_account_document(
        &mut document,
        name,
        backend,
        base_url,
        settings,
        interleaved_thinking,
    )?;
    let after = document.to_string();
    reparse_after_edit(&before, &after)?;
    crate::fs::write_file_atomic(&path, &after)?;
    Ok(())
}

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

/// A login the person finishes in a browser, begun: the URL they open, and what completes it
/// once they have.
///
/// One flow for both subscription backends, which `meka account add`, `meka account login` and
/// `POST /v1/accounts/{name}/login` run alike and differ in only one thing, how the person's
/// answer reaches them: a line at the terminal, or a request. The PKCE verifier and the state are
/// held here from the URL to the exchange, so a request completing a login can only complete the
/// one that minted its URL.
struct BrowserLogin {
    kind: CredentialKind,
    /// The URL the person opens.
    authorization_url: String,
    code_verifier: String,
    state: String,
    client_id: String,
    token_url: String,
    /// Where the browser is sent back to: the loopback listener for `chatgpt-subscription`,
    /// Anthropic's hosted page for `claude-subscription`.
    redirect_uri: String,
}

impl BrowserLogin {
    /// Begin a login for `backend`, minted under the account's `client_id` and against its
    /// `oauth_token_url` where it states them, for the reason `run_add` gives: a grant issued to
    /// the default client and then claimed by a custom one dies at its first refresh.
    fn begin(
        backend: config::Backend,
        client_id: Option<&str>,
        oauth_token_url: Option<&str>,
    ) -> anyhow::Result<Self> {
        let kind = credential_kind(backend);
        let (code_verifier, code_challenge) = generate_pkce_pair()?;
        let state = generate_state()?;
        let (client_id, token_url, redirect_uri, authorization_url) = match kind {
            CredentialKind::ClaudeLogin => {
                let client_id = client_id.unwrap_or(DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID);
                let token_url = oauth_token_url.unwrap_or(DEFAULT_CLAUDE_SUBSCRIPTION_TOKEN_URL);
                let url = build_authorize_url(client_id, &code_challenge, &state)?;
                (client_id, token_url, REDIRECT_URI.to_string(), url)
            }
            CredentialKind::ChatGptLogin => {
                let client_id = client_id.unwrap_or(DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID);
                let token_url = oauth_token_url.unwrap_or(CODEX_TOKEN_URL);
                let redirect_uri = format!("http://localhost:{CODEX_REDIRECT_PORT}/auth/callback");
                let url =
                    build_codex_authorize_url(client_id, &code_challenge, &state, &redirect_uri)?;
                (client_id, token_url, redirect_uri, url)
            }
            CredentialKind::ApiKey => {
                anyhow::bail!("'{backend}' takes an API key and has no browser login")
            }
        };
        Ok(Self {
            kind,
            authorization_url,
            code_verifier,
            state,
            client_id: client_id.to_string(),
            token_url: token_url.to_string(),
            redirect_uri,
        })
    }

    /// Whether the browser is sent back to a loopback listener this process can hold, so the
    /// login completes on its own when the browser runs on this machine.
    fn listens_on_loopback(&self) -> bool {
        matches!(self.kind, CredentialKind::ChatGptLogin)
    }

    /// What a pasted line is read as: the code Anthropic's page shows, or the URL the browser was
    /// sent to.
    const fn paste(&self) -> Paste {
        match self.kind {
            CredentialKind::ClaudeLogin => Paste::AuthorizationCode,
            CredentialKind::ChatGptLogin | CredentialKind::ApiKey => Paste::CallbackUrl,
        }
    }

    /// The code, and the state where the paste carries one, out of what the person pasted:
    /// Anthropic's page hands back `code#state`; a Codex login gets the callback URL the browser
    /// was sent to, or its bare query.
    fn parse_pasted(&self, input: &str) -> anyhow::Result<(String, Option<String>)> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            anyhow::bail!("authorization code cannot be empty");
        }
        match self.kind {
            CredentialKind::ClaudeLogin => {
                let (code, state) = match trimmed.split_once('#') {
                    Some((code, state)) => (code, Some(state.to_string())),
                    None => (trimmed, None),
                };
                Ok((code.to_string(), state))
            }
            CredentialKind::ChatGptLogin | CredentialKind::ApiKey => {
                match extract_codex_paste(trimmed) {
                    CodexCallback::Match { code, state } => Ok((code, Some(state))),
                    CodexCallback::AuthError(message) => {
                        anyhow::bail!("the authorization server rejected the login: {message}")
                    }
                    CodexCallback::NotCallback | CodexCallback::Malformed(_) => anyhow::bail!(
                        "no 'code' and 'state' in that input; paste the full callback URL the \
                         browser was sent to"
                    ),
                }
            }
        }
    }

    /// Whether `state`, as the browser brought it back, is this login's: a code presented under
    /// another login's state is a code this login did not ask for.
    fn state_matches(&self, state: &str) -> bool {
        state == self.state
    }

    /// Exchange the code the browser brought back for the credential, refusing a state that is
    /// not this login's.
    async fn complete(&self, code: &str, state: Option<&str>) -> anyhow::Result<AuthCredential> {
        if state.is_some_and(|state| !self.state_matches(state)) {
            anyhow::bail!("OAuth state mismatch; refusing the callback");
        }
        match self.kind {
            CredentialKind::ClaudeLogin => {
                exchange_claude_code(
                    code,
                    &self.code_verifier,
                    &self.client_id,
                    &self.state,
                    &self.token_url,
                )
                .await
            }
            CredentialKind::ChatGptLogin | CredentialKind::ApiKey => {
                exchange_codex_code(
                    code,
                    &self.code_verifier,
                    &self.client_id,
                    &self.redirect_uri,
                    &self.token_url,
                )
                .await
            }
        }
    }
}

/// Bind the loopback listener a `chatgpt-subscription` login is sent back to. Fails when another
/// process, or another login, holds the port.
async fn bind_codex_listener() -> std::io::Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind(format!("127.0.0.1:{CODEX_REDIRECT_PORT}")).await
}

/// How long the token endpoint has to answer an exchange. Bounded here because the exchange's
/// client bounds its connect alone, and a login left on a silent endpoint would outlive the
/// lifetime that ends it.
const TOKEN_EXCHANGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// A token exchange the authorization server refused, told apart from every other way a login
/// fails: what it says is the server's, which a surface decides whether to relay, where a timeout
/// or a listener's failure is meka's own sentence.
#[derive(Debug)]
pub(crate) struct ExchangeRefused(anyhow::Error);

impl std::fmt::Display for ExchangeRefused {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for ExchangeRefused {}

/// Log an account in through the browser, with `prompt` as the person's side of it: `meka account
/// add` and `meka account login` at a terminal, and `POST /v1/accounts/{name}/login` on the API.
///
/// The URL is minted under the account's `client_id` and against its `oauth_token_url` where it
/// states them, for the reason [`BrowserLogin::begin`] gives. A `chatgpt-subscription` login
/// holds the loopback listener the browser is sent back to for `callback_timeout`, raced against
/// a pasted callback URL where the prompt reads one; when the port is taken, the paste is all
/// there is. A `claude-subscription` login takes the code Anthropic's page shows, pasted. A paste
/// that carries no code, or another login's state, is refused to the prompt and waited past.
pub(crate) async fn browser_login(
    backend: config::Backend,
    client_id: Option<&str>,
    oauth_token_url: Option<&str>,
    callback_timeout: std::time::Duration,
    prompt: &dyn LoginPrompt,
) -> anyhow::Result<AuthCredential> {
    let login = BrowserLogin::begin(backend, client_id, oauth_token_url)?;
    let listener = if login.listens_on_loopback() {
        match bind_codex_listener().await {
            Ok(listener) => Some(listener),
            // The loopback callback only works when the browser runs on this machine. On a
            // remote or headless box the redirect lands on the user's laptop instead, so a
            // prompt that reads a paste can still finish; a bind failure there degrades to
            // paste-only rather than aborting.
            Err(error) if prompt.accepts_paste() => {
                tracing::warn!(
                    "failed to bind the callback listener on 127.0.0.1:{CODEX_REDIRECT_PORT}: \
                     {error}; falling back to pasting the callback URL"
                );
                None
            }
            Err(error) => anyhow::bail!(
                "failed to bind the callback listener on 127.0.0.1:{CODEX_REDIRECT_PORT}: {error}"
            ),
        }
    } else {
        None
    };
    // A login with no listener is completed by a paste alone, read whatever the input is: the
    // one line a script pipes in is still the code.
    let pastes = listener.is_none() || prompt.accepts_paste();
    prompt.authorize_at(&login.authorization_url);
    prompt.awaiting(Completion {
        listener_seconds: listener.as_ref().map(|_| callback_timeout.as_secs()),
        paste: pastes.then(|| login.paste()),
    });
    let (code, state) = match listener {
        // The accept carries the timeout that bounds the whole wait; the paste reader parks on a
        // closed input so a non-interactive stdin can never win the race against a real callback.
        Some(listener) if pastes => tokio::select! {
            result = accept_codex_callback(listener, callback_timeout) => {
                let (code, state) = result?;
                (code, Some(state))
            }
            result = read_paste(&login, prompt, true) => result?,
        },
        Some(listener) => {
            let (code, state) = accept_codex_callback(listener, callback_timeout).await?;
            (code, Some(state))
        }
        // Bounded like the listener's wait: a login nobody completes ends, rather than holding
        // the prompt for the life of the process.
        None => {
            match tokio::time::timeout(callback_timeout, read_paste(&login, prompt, false)).await {
                Ok(result) => result?,
                Err(_elapsed) => anyhow::bail!(
                    "authorization timed out after {}s",
                    callback_timeout.as_secs()
                ),
            }
        }
    };
    match tokio::time::timeout(
        TOKEN_EXCHANGE_TIMEOUT,
        login.complete(&code, state.as_deref()),
    )
    .await
    {
        Ok(exchanged) => exchanged.map_err(|error| anyhow::Error::new(ExchangeRefused(error))),
        Err(_elapsed) => anyhow::bail!(
            "the token exchange did not answer within {}s",
            TOKEN_EXCHANGE_TIMEOUT.as_secs()
        ),
    }
}

/// A pasted line that carries the code, read until one does: a blank line is waited past, one
/// with no code or with another login's state is refused to the prompt and waited past, and a
/// closed input parks the wait while a listener may still answer, else ends it.
async fn read_paste(
    login: &BrowserLogin,
    prompt: &dyn LoginPrompt,
    listening: bool,
) -> anyhow::Result<(String, Option<String>)> {
    loop {
        let line = match prompt.read_pasted_line().await {
            Err(error) => anyhow::bail!(error),
            Ok(None) if listening => std::future::pending().await,
            Ok(None) => anyhow::bail!("nothing was pasted"),
            Ok(Some(line)) => line,
        };
        if line.trim().is_empty() {
            continue;
        }
        match login.parse_pasted(&line) {
            // Refused ahead of the exchange and waited past: the code was never this login's to
            // spend, so the person can paste the right one.
            Ok((_, Some(state))) if !login.state_matches(&state) => {
                prompt.paste_refused("OAuth state mismatch; refusing the callback");
            }
            Ok(pair) => return Ok(pair),
            Err(error) => prompt.paste_refused(&error.to_string()),
        }
    }
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
    // The provider client, so the grant goes to `token_url` and nowhere a redirect points, and
    // the reply is read under the same bound as every other.
    let client = crate::provider::build_http_client("claude-subscription", |builder| builder)?;
    let never = tokio_util::sync::CancellationToken::new();
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
        let body = crate::error::read_whole_reply(response, None, &never)
            .await
            .unwrap_or_default();
        anyhow::bail!(
            "token exchange failed ({}): {}",
            status,
            crate::error::render_error_body(&body)
        );
    }

    let body = crate::error::read_whole_reply(response, None, &never).await?;
    let token: TokenResponse = serde_json::from_str(&body)?;
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

/// Wait on the loopback listener for the browser to be sent back, answering it a page, and give
/// the code and state it carried; refuses after `timeout`, and on an authorization the server
/// refused.
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
                Ok(Ok(n)) => buffer.extend(temp.iter().take(n)),
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

    // The provider client, for the reason `exchange_claude_code` gives.
    let client = crate::provider::build_http_client("chatgpt-subscription", |builder| builder)?;
    let never = tokio_util::sync::CancellationToken::new();
    let response = client
        .post(token_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = crate::error::read_whole_reply(response, None, &never)
            .await
            .unwrap_or_default();
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

    let body = crate::error::read_whole_reply(response, None, &never).await?;
    let token: CodexTokenResponse = serde_json::from_str(&body)?;
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The hint tells the accounts configured from the credentials no account claims, since the
    /// door takes a name of either kind.
    #[test]
    fn an_unknown_account_hint_tells_configured_from_orphaned() {
        let configured = vec!["work".to_string()];
        assert_eq!(
            unknown_account("x", &configured, &[]),
            "no account named 'x' (configured: work)"
        );
        assert_eq!(
            unknown_account("x", &configured, &["old".to_string()]),
            "no account named 'x' (configured: work); stored credentials with no account: old"
        );
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

        let orphans = orphaned_accounts(&store, &config_file.accounts)
            .await
            .expect("diff credentials against accounts");
        assert_eq!(orphans, vec!["archive".to_string()]);
    }

    /// The refusal for a name neither side has tells the accounts configured from the credentials
    /// no account claims, since this door takes a name of either kind.
    #[tokio::test]
    async fn an_unknown_account_is_refused_naming_each_kind_of_name_the_door_takes() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[accounts.work]\nbackend = \"anthropic-messages\"\n",
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await.token_store();
        for account in ["work", "old"] {
            store
                .save_account_credential(account, &AuthCredential::ApiKey("key".to_string()))
                .await
                .expect("save");
        }
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set → run → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = remove_account("nope", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        let error = result.expect_err("an unknown name is refused");
        assert_eq!(
            error.to_string(),
            "no account named 'nope' (configured: work); stored credentials with no account: old"
        );
    }

    /// A prompt whose input has closed hands over nothing.
    struct ClosedInput;

    #[async_trait::async_trait]
    impl crate::oauth::LoginPrompt for ClosedInput {
        fn authorize_at(&self, _url: &str) {}

        fn awaiting(&self, _completion: crate::oauth::Completion) {}

        fn accepts_paste(&self) -> bool {
            true
        }

        async fn read_pasted_line(&self) -> std::result::Result<Option<String>, String> {
            Ok(None)
        }

        fn paste_refused(&self, _reason: &str) {}
    }

    /// A closed input parks the paste wait while a listener may still answer the login, and ends
    /// it when nothing else can.
    #[tokio::test]
    async fn a_closed_input_parks_the_paste_while_a_listener_may_answer() {
        let login = BrowserLogin {
            kind: CredentialKind::ChatGptLogin,
            authorization_url: String::new(),
            code_verifier: String::new(),
            state: "s".to_string(),
            client_id: String::new(),
            token_url: String::new(),
            redirect_uri: String::new(),
        };
        let parked = tokio::time::timeout(
            std::time::Duration::from_millis(300),
            read_paste(&login, &ClosedInput, true),
        )
        .await;
        assert!(
            parked.is_err(),
            "with a listener the wait parks: {parked:?}"
        );
        let ended = read_paste(&login, &ClosedInput, false)
            .await
            .expect_err("without one it ends");
        assert_eq!(ended.to_string(), "nothing was pasted");
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
        let result = remove_account("work", &store).await;
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
        let result = remove_account("work", &store).await;
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

    /// A rename moves the name everywhere it is recorded and nothing else: the table keeps its
    /// place and the comment above it, every profile on the account follows, and the credential
    /// moves with it, so no login is needed.
    #[tokio::test]
    async fn rename_moves_the_credential_and_every_profile_on_the_account() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "# The work account.\n[accounts.work]\nbackend = \"anthropic-messages\"\n\n\
             [accounts.other]\nbackend = \"openai-responses\"\n\n\
             [profiles.daily]\naccount = \"work\" # bills work\nmodel = \"m\"\n\n\
             [profiles.side]\naccount = \"other\"\nmodel = \"m\"\n",
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
        let result = rename_account("work", "corp", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        result.expect("the rename succeeds");

        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(
            contents.contains("# The work account.\n[accounts.corp]\n"),
            "the table keeps its comment and its place: {contents}"
        );
        assert!(!contents.contains("[accounts.work]"), "{contents}");
        assert!(
            contents.find("[accounts.corp]") < contents.find("[accounts.other]"),
            "the tables keep their order: {contents}"
        );
        assert!(
            contents.contains("account = \"corp\" # bills work"),
            "the profile follows, its comment kept: {contents}"
        );
        assert!(
            contents.contains("account = \"other\""),
            "a profile on another account is untouched: {contents}"
        );
        assert!(
            store
                .load_account_credential("corp")
                .await
                .expect("load")
                .is_some(),
            "the credential moved with the account"
        );
        assert!(
            store
                .load_account_credential("work")
                .await
                .expect("load")
                .is_none(),
            "and nothing is left under the old name"
        );
    }

    /// Refused by name before anything moves: an account that does not exist, a name already in
    /// use, or a leftover credential stored under the new name, which is a secret nobody
    /// configured and must not be overwritten.
    #[tokio::test]
    async fn rename_refuses_an_unknown_account_a_taken_name_and_a_leftover_credential() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[accounts.work]\nbackend = \"anthropic-messages\"\n\n\
             [accounts.other]\nbackend = \"openai-responses\"\n",
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await.token_store();
        for account in ["work", "stale"] {
            store
                .save_account_credential(account, &AuthCredential::ApiKey("key".to_string()))
                .await
                .expect("save");
        }

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let unknown = rename_account("nope", "fresh", &store).await;
        let taken = rename_account("work", "other", &store).await;
        let leftover = rename_account("work", "stale", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let message = |result: anyhow::Result<()>| match result {
            Ok(()) => panic!("the rename must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(message(unknown).contains("no account named 'nope'"));
        assert!(message(taken).contains("an account named 'other' already exists"));
        assert!(message(leftover).contains("stored credential named 'stale'"));
        assert!(
            store
                .load_account_credential("work")
                .await
                .expect("load")
                .is_some(),
            "a refused rename moves nothing"
        );
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.contains("[accounts.work]"), "{contents}");
    }

    /// A refresh in flight owns the account's credential row for its duration; a rename that moved
    /// the row under it would have the rotated token dropped and the spent one kept.
    #[tokio::test]
    async fn rename_is_refused_while_the_credential_is_being_refreshed() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[accounts.work]\nbackend = \"anthropic-messages\"\n",
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await.token_store();
        store
            .save_account_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save");
        let _refreshing = store
            .try_lock_account_credential("work")
            .expect("lock")
            .expect("nobody else holds it");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = rename_account("work", "corp", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("a rename under a live refresh must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("refreshing its credential"), "{error}");
        assert!(
            store
                .load_account_credential("work")
                .await
                .expect("load")
                .is_some(),
            "nothing moved"
        );
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.contains("[accounts.work]"), "{contents}");
    }

    /// The credential moves first and comes back when the config write fails, so the two halves
    /// agree again by the time the error is read.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_config_write_that_fails_moves_the_credential_back() {
        use std::os::unix::fs::PermissionsExt as _;

        // Root writes through a read-only directory, so there is nothing to observe there.
        // SAFETY: `geteuid` reads a process attribute and has no preconditions.
        if unsafe { libc::geteuid() } == 0 {
            return;
        }
        // The file is a link into a read-only directory: the atomic write follows the link and
        // creates its temporary file beside the target, which is where it fails. The config
        // directory itself stays writable, since the write re-modes a directory it owns.
        let dir = tempfile::tempdir().expect("tempdir");
        let managed = dir.path().join("managed");
        std::fs::create_dir(&managed).expect("managed dir");
        std::fs::write(
            managed.join("config.toml"),
            "[accounts.work]\nbackend = \"anthropic-messages\"\n",
        )
        .expect("write config");
        std::os::unix::fs::symlink(managed.join("config.toml"), dir.path().join("config.toml"))
            .expect("link");
        let store = crate::store::Store::for_test().await.token_store();
        store
            .save_account_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save");
        std::fs::set_permissions(&managed, std::fs::Permissions::from_mode(0o555))
            .expect("read-only dir");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = rename_account("work", "corp", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        std::fs::set_permissions(&managed, std::fs::Permissions::from_mode(0o755))
            .expect("writable again");

        assert!(result.is_err(), "the config write must fail");
        assert!(
            store
                .load_account_credential("work")
                .await
                .expect("load")
                .is_some(),
            "the credential is back under the name the file still has"
        );
        assert!(
            store
                .load_account_credential("corp")
                .await
                .expect("load")
                .is_none()
        );
    }

    /// The locked write asks the referenced-by question again, of the file as it stands then.
    ///
    /// `remove_account` probes under one guard and writes under another, with the credential
    /// deletion awaited between them, so a `profile add` naming the account can land in the
    /// window and the probe's answer is stale by the write. Only a refusal under the write's
    /// own lock holds.
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
        let result = remove_account("typo", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("removing a name that exists nowhere must fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("no account named 'typo'") && error.contains("work"),
            "the one unknown-name sentence, naming what is known: {error}"
        );
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
            None,
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
            None,
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
            None,
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
        // The scope set as Claude Code 2.1.284 sends it, read off its own authorization constants:
        // the same names in the same order, so a login is indistinguishable from the first-party
        // client's.
        assert!(
            url.contains(
                "scope=org%3Acreate_api_key+user%3Aprofile+user%3Ainference+user%3Asessions%3Aclaude_code+user%3Amcp_servers+user%3Afile_upload+user%3Aplugins&"
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
        assert_eq!(
            validate_backend("claude-subscription").expect("a backend meka has"),
            config::Backend::ClaudeSubscription
        );
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
    /// A sibling rather than a duplicate: the Codex login reaches a *different* exchange helper
    /// with a different body encoding and its own hardcoded constant, so the Claude test says
    /// nothing about it. Threading the endpoint into one of two mints and not the other is
    /// exactly the one-door-of-two shape the rest of the suite closes.
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
    /// The browser leg means no test can drive the Claude login end to end, so the link between the
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
