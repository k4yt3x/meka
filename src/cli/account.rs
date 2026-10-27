//! `meka account` subcommand suite: the prompts, the listing, the login at a terminal and the
//! read-only account views, over the doors in [`crate::installation::account`].

use std::io::{self, Write};

use crate::{
    cli::AccountAction,
    config,
    installation::account::{
        CredentialKind, OAuthSettings, account_views, browser_login, credential_kind,
        orphaned_accounts, remove_account, rename_account, validate_backend, write_account,
    },
    store::{AuthCredential, TokenStore},
};

/// How long `meka account add` and `meka account login` wait at the terminal for the browser to
/// come back, by the listener or by a paste.
const BROWSER_LOGIN_WAIT: std::time::Duration = std::time::Duration::from_secs(600);

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
            interleaved_thinking,
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
                *interleaved_thinking,
                *api_key_stdin,
                token_store,
            )
            .await
        }
        AccountAction::List { format } => run_list(token_store, *format).await,
        AccountAction::Remove { name } => remove_account(name, token_store).await,
        AccountAction::Rename { name, new_name } => {
            rename_account(name, new_name, token_store).await
        }
        AccountAction::Login {
            name,
            api_key_stdin,
        } => run_login(name, *api_key_stdin, token_store).await,
        AccountAction::Usage { .. }
        | AccountAction::Whoami { .. }
        | AccountAction::Stats { .. } => run_introspection(store, action, cli_args).await,
    }
}

async fn run_add(
    name: &str,
    backend_flag: Option<&str>,
    base_url_flag: Option<String>,
    settings_flags: OAuthSettings,
    interleaved_thinking: Option<bool>,
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
    // Refused rather than written and ignored, the way a key the backend never reads is warned
    // about when hand-written: a flag that does nothing on this backend is a mistake to name.
    if interleaved_thinking.is_some() && !backend.reads_account_key("interleaved_thinking") {
        anyhow::bail!("`--interleaved-thinking` is read by `anthropic-messages` alone; drop it");
    }

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
    write_account(
        name,
        backend,
        base_url.as_deref(),
        &settings,
        interleaved_thinking,
    )?;
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

async fn run_list(
    token_store: &TokenStore,
    format: crate::cli::OutputFormat,
) -> anyhow::Result<()> {
    let config_file = config::load_config_file_or_err()?;
    // Computed before the early return below: "every account is gone but the secrets are still
    // here" is precisely the state worth reporting, and it is the one an early return would hide.
    let orphans = orphaned_accounts(token_store, &config_file.accounts).await?;

    let views = account_views(token_store, &config_file.accounts).await?;
    if format == crate::cli::OutputFormat::Json {
        crate::cli::write_json_listing("accounts", &views)?;
        report_account_problems(&orphans, &config_file)?;
        return Ok(());
    }
    if views.is_empty() {
        crate::streams::write_stderr_line("No accounts.");
        report_account_problems(&orphans, &config_file)?;
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
    crate::render::write_stdout(crate::text::format_table(&ACCOUNT_COLUMNS, &rows))?;
    report_account_problems(&orphans, &config_file)?;
    Ok(())
}

/// The listing's columns: the name every `meka account` command takes, the backend, the endpoint
/// for information, and whether a credential is stored.
const ACCOUNT_COLUMNS: [crate::text::Column; 4] = [
    crate::text::Column::content("Name"),
    crate::text::Column::content("Backend"),
    crate::text::Column::remainder("Base URL"),
    crate::text::Column::content("Authenticated"),
];

/// What this listing reveals about the store and the file beyond the table, said on stderr under
/// either format.
fn report_account_problems(
    orphans: &[String],
    config_file: &config::ConfigFile,
) -> anyhow::Result<()> {
    report_orphaned_accounts(orphans)?;
    report_unknown_backends(config_file);
    Ok(())
}

/// Accounts whose `backend` is not one meka knows, each with the parser's refusal.
///
/// `[accounts.<name>]` keeps `backend` as written, so a hand-edited typo loads and lists, and fails
/// only when a session on the account is built. A listing is where the user comes to check, so it
/// is where the discrepancy belongs; `meka profile list` reports it too, because every profile on
/// the account refuses to run.
fn unknown_backends(config_file: &config::ConfigFile) -> Vec<(String, String)> {
    config_file
        .accounts
        .iter()
        .filter_map(|(name, account)| {
            account
                .backend
                .parse::<config::Backend>()
                .err()
                .map(|error| (name.clone(), error))
        })
        .collect()
}

/// Say each of [`unknown_backends`] on stderr with its remedy, in the shape of the orphan report.
pub(super) fn report_unknown_backends(config_file: &config::ConfigFile) {
    for (name, error) in unknown_backends(config_file) {
        // Both halves are text from the file: the name is a key the user chose, and the refusal
        // quotes the value as written.
        let name = crate::text::sanitize_for_display(&name);
        let error = crate::text::sanitize_for_display(&error);
        crate::streams::write_stderr_line("");
        crate::streams::write_stderr_line(format!("Account '{name}': {error}"));
        crate::render::render_hint(&format!(
            "set `backend` under `[accounts.{name}]` to one of those"
        ));
    }
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
        CredentialKind::ClaudeLogin | CredentialKind::ChatGptLogin => {
            browser_login(
                backend,
                client_id,
                oauth_token_url,
                BROWSER_LOGIN_WAIT,
                &crate::cli::TerminalLogin,
            )
            .await
        }
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

/// Local (no-network) auth status for a stored credential: is the token valid, and when does it
/// expire. Serialized as the `auth` block of `meka account whoami --format json`.
#[derive(serde::Serialize)]
struct AuthStatus {
    valid: bool,
    /// Token expiry as RFC 3339, the form every wire carries a time in (`None` for API keys and
    /// for a token with no expiry).
    expires_at: Option<String>,
    /// Seconds until expiry (negative if already expired).
    expires_in_seconds: Option<i64>,
}

impl AuthStatus {
    fn from_credential(credential: &AuthCredential) -> Self {
        match credential {
            AuthCredential::OAuthToken { expires_at, .. } => {
                // `expires_at` is stored as epoch milliseconds.
                let expires_in_seconds =
                    expires_at.map(|millis| millis / 1000 - chrono::Utc::now().timestamp());
                let expires_at = expires_at
                    .and_then(chrono::DateTime::<chrono::Utc>::from_timestamp_millis)
                    .map(|expiry| expiry.to_rfc3339());
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
    account: &'a str,
    #[serde(flatten)]
    usage: &'a crate::provider::AccountUsage,
}

#[derive(serde::Serialize)]
struct WhoamiOutput<'a> {
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

/// The read-only account views, each over the account's own provider: they ask the account and
/// never a model, so no profile is needed.
async fn run_introspection(
    store: &crate::store::Store,
    action: &crate::cli::AccountAction,
    cli_args: &crate::cli::Cli,
) -> anyhow::Result<()> {
    let (view, named, format) = match action {
        crate::cli::AccountAction::Usage { name, format } => (View::Usage, name.clone(), *format),
        crate::cli::AccountAction::Whoami { name, format } => (View::Whoami, name.clone(), *format),
        crate::cli::AccountAction::Stats { name, format } => (View::Stats, name.clone(), *format),
        crate::cli::AccountAction::Add { .. }
        | crate::cli::AccountAction::List { .. }
        | crate::cli::AccountAction::Login { .. }
        | crate::cli::AccountAction::Remove { .. }
        | crate::cli::AccountAction::Rename { .. } => {
            anyhow::bail!("not an introspection command")
        }
    };

    let config = config::ResolvedConfig::resolve(cli_args.overrides());
    // A parse error leaves `accounts` empty, and the refusal below would then blame a missing
    // account for a typo in the file; every sibling subcommand fails on the parse error instead.
    config.require_readable_config()?;
    // An explicit `--profile` that selects nothing is refused as a run refuses it, rather than
    // falling silently to the one account.
    if cli_args.overrides().profile.is_some() && config.default_profile.is_none() {
        anyhow::bail!(
            config
                .provider_error
                .clone()
                .unwrap_or_else(|| "no profile configured".to_string())
        );
    }
    let account = account_to_ask(
        named,
        config.default_profile.as_deref(),
        &config.accounts,
        &config.profiles,
    )?;
    let registry = crate::provider::ProviderRegistry::new(&config, store.token_store());
    let (provider, backend) = registry.resolve_account(&account).await?;

    match view {
        View::Usage => match provider.fetch_usage().await? {
            Some(usage) => match format {
                crate::cli::OutputFormat::Plain => {
                    crate::render::write_stdout(crate::render::format_account_usage(&usage))?;
                }
                crate::cli::OutputFormat::Json => {
                    crate::cli::write_json(&UsageOutput {
                        account: &account,
                        usage: &usage,
                    })?;
                }
            },
            None => {
                crate::streams::write_stderr_line(format!(
                    "Account usage is not available for account '{account}'."
                ));
                return Err(crate::error::AlreadyReported.into());
            }
        },
        View::Whoami => {
            // The identity call may refresh + rotate the token; re-read afterwards so the auth
            // block reflects the current expiry. A failed identity fetch (e.g. re-login needed)
            // still prints the local auth status so scripts can detect it.
            let identity = provider.fetch_identity().await;
            let fresh = store
                .token_store()
                .load_account_credential(&account)
                .await?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no stored credential for account '{account}'; log in with `meka account \
                         login {account}`"
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
                account: &account,
                backend: backend.name(),
                auth,
                identity,
            };
            let written = match format {
                crate::cli::OutputFormat::Plain => {
                    crate::render::write_stdout(format_whoami_plain(&out))
                }
                crate::cli::OutputFormat::Json => crate::cli::write_json(&out),
            };
            // Checked before the write is, because on this one command the status *is* the answer:
            // a script reads it to decide whether to re-authenticate. Propagating a failed write
            // first would hand back the code for "stdout went away", and a reader that hung up
            // turns that into success, which says the credential is fine when it is not.
            if !out.auth.valid {
                return Err(crate::error::AlreadyReported.into());
            }
            written?;
        }
        View::Stats => match provider.fetch_history().await? {
            Some(history) => {
                let out = StatsOutput {
                    account: &account,
                    history: &history,
                };
                match format {
                    crate::cli::OutputFormat::Plain => {
                        crate::render::write_stdout(format_stats_plain(&out))?
                    }
                    crate::cli::OutputFormat::Json => crate::cli::write_json(&out)?,
                }
            }
            None => {
                crate::streams::write_stderr_line(format!(
                    "Account history is not available for account '{account}'."
                ));
                return Err(crate::error::AlreadyReported.into());
            }
        },
    }
    Ok(())
}

/// The account a view asks: the one named, else the account the default profile bills, which is
/// what a run reaches, else the one account when there is exactly one. Refused otherwise, naming
/// the accounts, since choosing among several would report on an account nobody chose.
fn account_to_ask(
    named: Option<String>,
    default_profile: Option<&str>,
    accounts: &std::collections::BTreeMap<String, crate::config::AccountConfig>,
    profiles: &std::collections::BTreeMap<String, crate::config::ProfileConfig>,
) -> anyhow::Result<String> {
    if let Some(name) = named {
        if !accounts.contains_key(&name) {
            anyhow::bail!(crate::text::unknown_name("account", &name, accounts.keys()));
        }
        return Ok(name);
    }
    if let Some(profile) = default_profile.and_then(|profile| profiles.get(profile)) {
        return Ok(profile.account.clone());
    }
    let mut names = accounts.keys();
    match (names.next(), names.next()) {
        (Some(only), None) => Ok(only.clone()),
        (None, _) => {
            anyhow::bail!("no accounts configured; create one with `meka account add <name>`")
        }
        (Some(_), Some(_)) => anyhow::bail!(
            "name the account to ask (configured: {})",
            accounts.keys().cloned().collect::<Vec<_>>().join(", ")
        ),
    }
}

#[derive(serde::Serialize)]
struct StatsOutput<'a> {
    account: &'a str,
    #[serde(flatten)]
    history: &'a crate::provider::UsageHistory,
}

/// `meka account stats` as `label: value` lines, the recent days indented under a bare `recent:`
/// heading the way `schedule show` prints a prompt.
fn format_stats_plain(out: &StatsOutput<'_>) -> String {
    let history = out.history;
    let tokens = |value: Option<i64>| {
        value.map(|value| crate::text::format_token_count(value.max(0) as u64))
    };
    let days = |value: Option<i64>| value.map(|value| format!("{value} days"));
    let mut fields = vec![("account", out.account.to_string())];
    if let Some(first) = &history.first_used {
        // An RFC 3339 timestamp trimmed to its date: the time of day says nothing here.
        fields.push((
            "first used",
            first.split('T').next().unwrap_or(first).to_string(),
        ));
    }
    for (label, value) in [
        ("lifetime tokens", tokens(history.lifetime_tokens)),
        ("peak daily", tokens(history.peak_daily_tokens)),
        ("current streak", days(history.current_streak_days)),
        ("longest streak", days(history.longest_streak_days)),
    ] {
        if let Some(value) = value {
            fields.push((label, value));
        }
    }
    if !history.daily.is_empty() {
        fields.push(("recent", String::new()));
    }
    let mut text = crate::text::format_fields(&fields);
    // Indented, because a date on its own would read as one more field.
    for day in history.daily.iter().rev().take(7) {
        text.push_str(&format!(
            "  {}  {}\n",
            day.date,
            crate::text::format_token_count(day.tokens.max(0) as u64)
        ));
    }
    text
}

/// `meka account whoami` as `label: value` lines; an identity field the backend did not report
/// is left out rather than printed empty.
fn format_whoami_plain(out: &WhoamiOutput<'_>) -> String {
    let auth = match (out.auth.valid, out.auth.expires_in_seconds) {
        (true, Some(secs)) => format!(
            "valid ({})",
            crate::text::format_duration_short(secs.max(0))
        ),
        (true, None) => "valid".to_string(),
        (false, _) => "expired; run `meka account login`".to_string(),
    };
    let mut fields = vec![
        ("account", out.account.to_string()),
        ("backend", out.backend.to_string()),
        ("auth", auth),
    ];
    if let Some(identity) = &out.identity {
        for (label, value) in [
            ("name", &identity.display_name),
            ("email", &identity.email),
            ("plan", &identity.plan),
            ("tier", &identity.tier),
            ("subscription", &identity.subscription_status),
            ("organization", &identity.organization),
            ("role", &identity.role),
        ] {
            if let Some(value) = value {
                fields.push((label, value.clone()));
            }
        }
    }
    crate::text::format_fields(&fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_meka_does_not_know_is_named_with_the_parser_s_refusal() {
        let config_file: config::ConfigFile = toml::from_str(
            "[accounts.good]\nbackend = \"anthropic-messages\"\n\n[accounts.typo]\nbackend = \"anthropic\"\n",
        )
        .expect("parses");
        let unknown = unknown_backends(&config_file);
        assert_eq!(unknown.len(), 1, "{unknown:?}");
        assert_eq!(unknown[0].0, "typo");
        assert!(
            unknown[0].1.contains("'anthropic' is not a backend"),
            "{unknown:?}"
        );
    }

    #[test]
    fn stats_indent_the_recent_days_under_a_bare_heading() {
        let history = crate::provider::UsageHistory {
            lifetime_tokens: Some(1_200_000),
            peak_daily_tokens: None,
            current_streak_days: Some(3),
            longest_streak_days: None,
            first_used: Some("2026-04-01T17:36:16Z".to_string()),
            daily: vec![crate::provider::DailyUsage {
                date: "2026-09-10".to_string(),
                tokens: 950,
            }],
        };
        let text = format_stats_plain(&StatsOutput {
            account: "codex",
            history: &history,
        });
        assert_eq!(
            text,
            "account:          codex\n\
             first used:       2026-04-01\n\
             lifetime tokens:  1.2M\n\
             current streak:   3 days\n\
             recent:\n\
             \x20 2026-09-10  950\n"
        );
    }

    /// A view asks the account named; named nothing, the account the default profile bills, else
    /// the one account; several with no default are refused by name, as is a name not configured.
    #[test]
    fn a_view_asks_the_named_account_else_the_default_profiles_else_the_only_one() {
        let account = |backend: &str| crate::config::AccountConfig {
            backend: backend.to_string(),
            ..Default::default()
        };
        let accounts: std::collections::BTreeMap<String, crate::config::AccountConfig> = [
            ("claude".to_string(), account("claude-subscription")),
            ("work".to_string(), account("anthropic-messages")),
        ]
        .into_iter()
        .collect();
        let profiles: std::collections::BTreeMap<String, crate::config::ProfileConfig> =
            [("daily".to_string(), crate::config::ProfileConfig {
                account: "claude".to_string(),
                ..Default::default()
            })]
            .into_iter()
            .collect();
        assert_eq!(
            account_to_ask(
                Some("work".to_string()),
                Some("daily"),
                &accounts,
                &profiles
            )
            .expect("named"),
            "work"
        );
        assert_eq!(
            account_to_ask(None, Some("daily"), &accounts, &profiles)
                .expect("the default profile's"),
            "claude"
        );
        let refused = account_to_ask(None, None, &accounts, &profiles)
            .expect_err("two accounts and no default");
        assert!(
            refused.to_string().contains("(configured: claude, work)"),
            "{refused}"
        );
        let unknown = account_to_ask(Some("nope".to_string()), None, &accounts, &profiles)
            .expect_err("not configured");
        assert!(unknown.to_string().contains("'nope'"), "{unknown}");
        let only: std::collections::BTreeMap<String, crate::config::AccountConfig> =
            [("work".to_string(), account("anthropic-messages"))]
                .into_iter()
                .collect();
        assert_eq!(
            account_to_ask(None, None, &only, &profiles).expect("the only one"),
            "work"
        );
    }

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

    /// `--interleaved-thinking` is `anthropic-messages`'s key; on any other backend it would be
    /// written and never read, so the door refuses it in the words the hand-written warning uses.
    #[tokio::test]
    async fn add_refuses_the_interleaved_thinking_flag_on_a_backend_that_never_reads_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "").expect("write config");
        let store = crate::store::Store::for_test().await.token_store();

        // SAFETY: as in the sibling tests.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_add(
            "home",
            Some("openai-responses"),
            None,
            OAuthSettings::default(),
            Some(false),
            true,
            &store,
        )
        .await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        let error = result
            .expect_err("refused ahead of the key prompt")
            .to_string();
        assert!(error.contains("`--interleaved-thinking`"), "{error}");
        assert!(error.contains("anthropic-messages"), "{error}");
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
            None,
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
            None,
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

    /// The read-only views fail on an unreadable `config.toml` the way every sibling does, rather
    /// than reading its empty parse as "no accounts configured" and sending the user to add one.
    #[tokio::test]
    async fn an_introspection_command_reports_an_unreadable_config_rather_than_a_missing_account() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "bogus = 1\n").expect("write config");
        let store = crate::store::Store::for_test().await;
        let cli = <crate::cli::Cli as clap::Parser>::parse_from(["meka"]);
        let action = crate::cli::AccountAction::Whoami {
            name: None,
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
}
