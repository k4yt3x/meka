//! `meka profile` subcommand suite: the profiles in `[profiles.<name>]`, each naming an account
//! and the model and every model-tied setting meka asks that account's endpoint for.
//!
//! A profile holds no secret and runs no login; that is the account's business, in `account.rs`,
//! whose config-editing and prompt helpers this module shares.

use super::account::{
    ensure_section_table, open_document, prompt_line, prompt_yes_no, reparse_after_edit,
    table_names, validate_backend,
};
use crate::{cli::ProfileAction, config};

/// Dispatch a `meka profile` subcommand.
pub(crate) async fn run(
    action: &ProfileAction,
    // Taken whole rather than as a bare token store because `remove` has to say how many sessions
    // it is about to strand, and only the session table can answer that.
    store: &crate::store::Store,
) -> anyhow::Result<()> {
    match action {
        ProfileAction::Add {
            name,
            account,
            model,
            context_window,
            max_output_tokens,
            effort,
            vision,
            thinking,
            thinking_budget,
            max_request_bytes,
            thinking_display,
        } => run_add(name, account.as_deref(), model.clone(), ProfileTuning {
            context_window: *context_window,
            max_output_tokens: *max_output_tokens,
            effort: effort.clone(),
            vision: *vision,
            thinking: *thinking,
            thinking_budget: *thinking_budget,
            max_request_bytes: *max_request_bytes,
            thinking_display: *thinking_display,
        }),
        ProfileAction::List { format } => run_list(*format),
        ProfileAction::Set {
            name,
            key,
            value,
            unset,
        } => run_set(name, key, value.as_deref(), *unset),
        ProfileAction::Use { name } => run_use(name),
        ProfileAction::Remove { name } => run_remove(name, store).await,
    }
}

/// The settings `profile add` can write beyond the account and model it always asks for. Each is
/// `None` when the flag was absent and the user declined (or skipped) the advanced prompt, which
/// leaves the key out of the profile entirely so the documented default applies.
///
/// Only `context_window`, `effort` and `thinking` are ever *prompted* for, plus the budget when
/// the answer is `budgeted`. The rest are flag-only, so a profile of any shape can be created in
/// one non-interactive command without making the interactive path a nine-step wizard for settings
/// most users never state. [`resolve_tuning`] says the same thing from the other side: its
/// short-circuit tests those three and no others, because a rare flag must not silently skip the
/// prompt for the common ones.
///
/// Ordered as [`config::ProfileConfig`] is, minus the two `profile add` takes as its own arguments
/// (`account`, `model`).
#[derive(Default)]
struct ProfileTuning {
    context_window: Option<u64>,
    max_output_tokens: Option<u64>,
    effort: Option<String>,
    vision: Option<bool>,
    thinking: Option<config::ThinkingMode>,
    thinking_budget: Option<u64>,
    max_request_bytes: Option<u64>,
    thinking_display: Option<crate::config::ThinkingDisplay>,
}

fn run_add(
    name: &str,
    account_flag: Option<&str>,
    model_flag: Option<String>,
    tuning_flags: ProfileTuning,
) -> anyhow::Result<()> {
    if name.trim().is_empty() {
        anyhow::bail!("profile name cannot be empty");
    }
    // Hard-fails on an unparseable config rather than warning: this guard is the only thing
    // standing between `profile add <existing>` and `upsert_profile_document` replacing the
    // profile's table, and an empty parsed map defeats it.
    let existing = config::load_config_file_or_err()?;
    if existing.profiles.contains_key(name) {
        anyhow::bail!(
            "a profile named '{name}' already exists; change it with `meka profile set {name} <key> \
             <value>`"
        );
    }

    let account_name = match account_flag {
        Some(account) => account.to_string(),
        None => prompt_account(&existing)?,
    };
    let Some(account) = existing.accounts.get(&account_name) else {
        return Err(unknown_account(&account_name, existing.accounts.keys()));
    };
    // The backend decides which settings are worth asking about and which flags are inert, so it
    // is read before the prompts rather than left to fail at the first run.
    let backend = validate_backend(&account.backend)?;

    let model = match model_flag {
        // Refused here rather than left to the write door, whose remedy is `profile set` on a
        // profile this command has not created; at this door the remedy is the flag.
        Some(model) if config::require_model(name, Some(&model)).is_err() => {
            anyhow::bail!("`--model` cannot be empty")
        }
        Some(model) => model,
        None => {
            let default_model = default_model_for(backend);
            let input = prompt_line(&format!("Model name [{default_model}]: "))?;
            // Empty entry accepts the backend's default.
            if input.is_empty() {
                default_model.to_string()
            } else {
                input
            }
        }
    };

    let tuning = resolve_tuning(
        tuning_flags,
        backend,
        name,
        existing
            .session
            .as_ref()
            .and_then(|session| session.context_window),
        existing
            .thinking
            .as_ref()
            .and_then(|thinking| thinking.budget),
    )?;

    write_profile(name, &account_name, model.as_str(), &tuning)?;
    tracing::info!("added profile '{name}' on account '{account_name}'");
    Ok(())
}

/// The refusal for a profile naming an account that is not configured, worded once for the door
/// ahead of the prompts and the one under the lock.
fn unknown_account(
    account: &str,
    known: impl IntoIterator<Item = impl AsRef<str>>,
) -> anyhow::Error {
    anyhow::anyhow!(
        "{}; create it with `meka account add {account}`",
        crate::text::unknown_name("account", account, known)
    )
}

/// Ask which account a new profile bills. A sole account is offered as the default; several are
/// listed and one must be named.
fn prompt_account(config_file: &config::ConfigFile) -> anyhow::Result<String> {
    let names: Vec<&str> = config_file.accounts.keys().map(String::as_str).collect();
    match names.as_slice() {
        [] => anyhow::bail!("no accounts configured; create one with `meka account add <name>`"),
        [only] => {
            let input = prompt_line(&format!("Account [{only}]: "))?;
            Ok(if input.is_empty() {
                (*only).to_string()
            } else {
                input
            })
        }
        _ => {
            let input = prompt_line(&format!("Account ({}): ", names.join(", ")))?;
            if input.is_empty() {
                anyhow::bail!("an account is required; pass `--account <name>`");
            }
            Ok(input)
        }
    }
}

/// Default model offered at the `profile add` prompt for a given backend. The user can override it
/// by typing a different name; an empty entry accepts the default.
fn default_model_for(backend: config::Backend) -> &'static str {
    match backend {
        config::Backend::AnthropicMessages | config::Backend::ClaudeSubscription => "claude-opus-5",
        config::Backend::OpenAiChatCompletions
        | config::Backend::OpenAiResponses
        | config::Backend::ChatGptSubscription => "gpt-5.6-sol",
    }
}

async fn run_remove(name: &str, store: &crate::store::Store) -> anyhow::Result<()> {
    // `open_document` rather than the parsed config on purpose: `remove` must still run on a
    // config.toml that meka can't deserialize, since it is one of the ways such a file gets
    // repaired. The guard is dropped before the `await` below, for the reason `account remove`
    // gives about `ConfigFileLock`'s thread-local depth counter.
    let (was_default, removed, remaining) = {
        let (_lock, path, mut document) = open_document()?;
        let was_default = document
            .get("default_profile")
            .and_then(|item| item.as_str())
            == Some(name);
        // Written even with no profile to remove: `default_profile` can still point at the name,
        // and dropping that dangling pointer is exactly the cleanup this case is for.
        let removed = remove_profile_document(&mut document, name);
        if !removed && !was_default {
            // The document's own table names rather than `require_profile`: this command edits
            // through `toml_edit` so it works on a file the typed config no longer parses.
            anyhow::bail!(crate::text::unknown_name(
                "profile",
                name,
                table_names(&document, "profiles")
            ));
        }
        crate::fs::write_file_atomic(&path, &document.to_string())?;
        (was_default, removed, table_names(&document, "profiles"))
    };

    // Losing the default is not a detail: with two or more profiles left, nothing picks one, and
    // the next `meka` with no `--profile` stops with an error about a setting the user did not
    // knowingly change. Said at `warn!` rather than `info!` because it needs a follow-up action
    // and is visible at the default verbosity.
    if was_default && remaining.len() > 1 {
        tracing::warn!(
            "'{name}' was the default profile, so `default_profile` is now unset; pick one of \
             {remaining} with `meka profile use <name>`",
            remaining = remaining.join(", ")
        );
    }

    // Sessions pinned to the profile are what makes a removal consequential, and they are silent
    // otherwise: the refusal arrives whenever the user next resumes one, which may be days later
    // and in another directory. The store is already open, so this costs one count.
    match store.count_sessions_on_profile(name).await {
        Ok(0) => {}
        Ok(pinned) => tracing::warn!(
            "{pinned} session(s) run on '{name}' and will refuse to resume; move one with \
             `meka -r <id> --profile <name>`"
        ),
        // Not worth failing the removal over: the profile is already gone, and this is advisory.
        Err(error) => tracing::warn!("failed to count sessions on '{name}': {error}"),
    }

    if removed {
        tracing::info!("removed profile '{name}'");
    } else {
        tracing::info!(
            "cleared `default_profile`, which named '{name}'; no such profile was configured"
        );
    }
    Ok(())
}

/// The settings `meka profile set` will write, in [`config::ProfileConfig`]'s canonical order
/// minus `account`.
///
/// `account` is absent. Moving a profile to another account moves every session on it to another
/// credential and possibly another backend, silently, which is the one thing this command must not
/// be able to do; a new profile on that account is one command away.
const SETTABLE_PROFILE_KEYS: &[&str] = &[
    "model",
    "context_window",
    "max_output_tokens",
    "effort",
    "vision",
    "thinking",
    "thinking_budget",
    "max_request_bytes",
    "thinking_display",
];

/// Parse one `key`/`value` pair into the TOML value the profile should carry.
///
/// Split from the write so the whole vocabulary is testable without a filesystem, and so a value
/// that cannot be parsed is refused before the config lock is taken rather than after.
///
/// Each key parses the way the matching `profile add` flag does, `thinking` through the same
/// `ValueEnum` for the reason [`resolve_tuning`]'s prompt gives: a second hand-written match would
/// be a second thing to keep in step with the enum. `name` is only for the refusal a value earns.
fn parse_profile_value(name: &str, key: &str, value: &str) -> anyhow::Result<toml_edit::Value> {
    let integer = |what: &str| -> anyhow::Result<toml_edit::Value> {
        let parsed: u64 = value
            .parse()
            .map_err(|_| anyhow::anyhow!("{what} must be a whole number, got '{value}'"))?;
        Ok(toml_edit::Value::from(toml_integer(what, parsed)?))
    };
    let boolean = |what: &str| -> anyhow::Result<toml_edit::Value> {
        match value {
            "true" => Ok(toml_edit::Value::from(true)),
            "false" => Ok(toml_edit::Value::from(false)),
            other => anyhow::bail!("{what} must be true or false, got '{other}'"),
        }
    };
    match key {
        // The predicate the write doors and a run apply, asked of the value alone: `""` is valid
        // TOML and a model no provider has, and it would otherwise be written and then sent.
        "model" => Ok(toml_edit::Value::from(config::require_model(
            name,
            Some(value),
        )?)),
        "effort" => Ok(toml_edit::Value::from(value)),
        "context_window" => integer("context_window"),
        "max_output_tokens" => integer("max_output_tokens"),
        "vision" => boolean("vision"),
        "thinking_budget" => integer("thinking_budget"),
        "max_request_bytes" => integer("max_request_bytes"),
        "thinking_display" => {
            let display = value
                .parse::<crate::config::ThinkingDisplay>()
                .map_err(anyhow::Error::msg)?;
            Ok(toml_edit::Value::from(display.name()))
        }
        "thinking" => {
            let mode = value
                .parse::<crate::config::ThinkingMode>()
                .map_err(anyhow::Error::msg)?;
            Ok(toml_edit::Value::from(mode.name()))
        }
        other => anyhow::bail!(
            "'{}' is not a profile setting (settable: {}).{}",
            other,
            SETTABLE_PROFILE_KEYS.join(", "),
            unsettable_key_hint(other)
        ),
    }
}

/// Why a key the user may reasonably reach for still cannot be written here.
///
/// An empty string for anything else, so the refusal above reads the same either way. Worth saying
/// rather than leaving to the list: a user who typed `account` did not typo, and "settable: model,
/// effort, ..." alone would read as though meka had simply forgotten it.
fn unsettable_key_hint(key: &str) -> String {
    match key {
        "account" => " Changing `account` would move every session on this profile onto another \
             credential; add a profile on that account instead."
            .to_string(),
        "backend" | "base_url" | "oauth_token_url" | "client_id" | "device_id" => {
            format!(" `{key}` is an account setting, under `[accounts.<name>]`.")
        }
        _ => String::new(),
    }
}

/// Write one key into `[profiles.<name>]`, or remove it when `value` is `None`.
///
/// Answers whether the profile was there to change, which the caller turns into the refusal: a
/// silent no-op on a mistyped profile name is the failure `meka profile remove` reports too.
///
/// A field-level edit rather than [`upsert_profile_document`]'s whole-table replace, so `toml_edit`
/// keeps every other key, its ordering, and any comment the user wrote beside it. Replacing the
/// table would silently eat all three on every `set`.
fn set_profile_field(
    document: &mut toml_edit::DocumentMut,
    name: &str,
    key: &str,
    value: Option<toml_edit::Value>,
) -> bool {
    let Some(item) = document
        .get_mut("profiles")
        .and_then(|item| item.as_table_like_mut())
        .and_then(|profiles| profiles.get_mut(name))
    else {
        return false;
    };
    let Some(profile) = item.as_table_like_mut() else {
        return false;
    };
    match value {
        Some(mut value) => {
            // A key's surroundings live in two places, and `insert` clears both. The *value*'s
            // decor holds what follows it on the line, so losing it deleted the trailing `# note`
            // beside the model being changed. The *key*'s leaf decor holds everything before it,
            // which is where `toml_edit` puts whole-line comments and blank lines above the key --
            // so losing that deleted the paragraph a user had written to explain the setting, and
            // the blank line separating it from the one above. Both are captured before the insert
            // and put back after, because `insert` is what clears them.
            if let Some(existing) = profile.get(key).and_then(|item| item.as_value()) {
                *value.decor_mut() = existing.decor().clone();
            }
            let key_decor = profile.key(key).map(|key| key.leaf_decor().clone());
            profile.insert(key, toml_edit::Item::Value(value));
            if let (Some(decor), Some(mut written)) = (key_decor, profile.key_mut(key)) {
                *written.leaf_decor_mut() = decor;
            }
        }
        None => {
            profile.remove(key);
        }
    }
    // A key `set` adds was appended, so it would otherwise sit wherever it landed. Sorting here
    // rather than only on the added key is what makes the order an invariant instead of a habit: a
    // profile written by hand, or by a meka that ordered its keys differently, is normalized the
    // first time this command touches it.
    crate::config::sort_profile_keys(item);
    true
}

/// Refuse a key on a profile whose account's backend will never send it.
///
/// [`resolve_tuning`] drops such a flag on `profile add` with a warning, on the grounds that
/// writing it "would produce a setting that reads plausibly and does nothing". `set` reaches the
/// same outcome by refusing rather than dropping, and the difference is deliberate: `add` is
/// building a bundle out of many flags, where ignoring one inapplicable knob and saying so is
/// proportionate, whereas `set` exists to write exactly one key, so dropping it would mean
/// reporting success for a command that did nothing at all.
///
/// Every key, through [`config::Backend::reads_profile_key`], which is the one place that knows.
/// A list of the thinking keys here let `max_request_bytes` onto an `openai-responses` profile,
/// reported as set, where nothing ever read it.
///
/// Read from the document being written rather than from a parameter, because `set account` is
/// refused and so the backend cannot be the thing this command is changing. Asked only of a write:
/// `run_set` skips it for `--unset`, which can only be removing such a key.
fn refuse_an_inert_key(
    document: &toml_edit::DocumentMut,
    name: &str,
    key: &str,
) -> anyhow::Result<()> {
    let account = document
        .get("profiles")
        .and_then(|item| item.as_table_like())
        .and_then(|profiles| profiles.get(name))
        .and_then(|item| item.as_table_like())
        .and_then(|profile| profile.get("account"))
        .and_then(|item| item.as_str())
        .unwrap_or_default();
    let backend = document
        .get("accounts")
        .and_then(|item| item.as_table_like())
        .and_then(|accounts| accounts.get(account))
        .and_then(|item| item.as_table_like())
        .and_then(|account| account.get("backend"))
        .and_then(|item| item.as_str())
        .unwrap_or_default();
    // An unrecognized backend, or a profile whose account is missing, is left alone:
    // `validate_backend` and `account_for` are the doors that report those, and guessing here
    // would refuse a key over a typo in a different one.
    let Ok(backend) = backend.parse::<config::Backend>() else {
        return Ok(());
    };
    if backend.reads_profile_key(key) {
        return Ok(());
    }
    anyhow::bail!(
        "profile '{name}' bills '{account}', a '{backend}' account, which never sends `{key}`; \
         nothing was written"
    )
}

/// Refuse a key this command will not write, naming the ones it will.
///
/// Asked before the value is looked at, so `--unset modle` is refused for the same reason and in
/// the same words as `set modle x`. Checking it inside the value branch instead left `--unset` to
/// accept any spelling and report success, having changed nothing.
fn ensure_settable_key(key: &str) -> anyhow::Result<()> {
    if SETTABLE_PROFILE_KEYS.contains(&key) {
        return Ok(());
    }
    anyhow::bail!(
        "'{}' is not a profile setting (settable: {}).{}",
        key,
        SETTABLE_PROFILE_KEYS.join(", "),
        unsettable_key_hint(key)
    )
}

/// `meka profile set <name> <key> <value>`, or `--unset` to drop the key.
fn run_set(name: &str, key: &str, value: Option<&str>, unset: bool) -> anyhow::Result<()> {
    ensure_settable_key(key)?;
    // Clap's `conflicts_with` rules out "both"; this is the other half, which it cannot express.
    let parsed = match (value, unset) {
        (Some(value), _) => Some(parse_profile_value(name, key, value)?),
        (None, true) => None,
        (None, false) => anyhow::bail!(
            "`meka profile set {name} {key}` needs a value, or `--unset` to remove the setting"
        ),
    };

    // `_lock` is held to the end of the function, so the read, the validation and the write are one
    // critical section.
    let (_lock, path, mut document) = open_document()?;
    let before = document.to_string();
    if !set_profile_field(&mut document, name, key, parsed) {
        // The document's own table names, for the reason `run_remove` gives.
        anyhow::bail!(crate::text::unknown_name(
            "profile",
            name,
            table_names(&document, "profiles")
        ));
    }
    // Only a write is refused. `--unset` on such a key moves the profile *towards* the state this
    // guards, and a hand-edited file is the one place an inert key can already be sitting, so
    // refusing to remove it would leave no door that could.
    if value.is_some() {
        refuse_an_inert_key(&document, name, key)?;
    }

    // Asked of the document this command is about to write, not of the one on disk, so neither
    // write door can leave behind a profile the other would have refused. Checked here rather than
    // at the next run because a config that fails to start is a worse answer than a command that
    // declines, and the value that broke it is in front of the user right now.
    let after = document.to_string();
    refuse_a_profile_that_cannot_run(&before, &after, name)?;

    crate::fs::write_file_atomic(&path, &after)?;
    match value {
        Some(value) => tracing::info!("set {key} = {value} on profile '{name}'"),
        None => tracing::info!("cleared {key} on profile '{name}'"),
    }
    Ok(())
}

fn run_use(name: &str) -> anyhow::Result<()> {
    set_default_profile(name)?;
    tracing::info!("default profile set to '{name}'");
    Ok(())
}

fn run_list(format: crate::cli::OutputFormat) -> anyhow::Result<()> {
    let config_file = config::load_config_file_or_err()?;
    let views = profile_views(&config_file);
    match format {
        crate::cli::OutputFormat::Json => crate::cli::write_json_listing("profiles", &views)?,
        crate::cli::OutputFormat::Plain if views.is_empty() => {
            crate::streams::write_stderr_line("No profiles.");
        }
        crate::cli::OutputFormat::Plain => {
            let rows: Vec<Vec<String>> = views
                .iter()
                .map(|view| {
                    vec![
                        view.name.clone(),
                        view.account.clone(),
                        view.backend.clone().unwrap_or_else(|| "-".to_string()),
                        view.model.clone().unwrap_or_else(|| "-".to_string()),
                        if view.active { "*" } else { "" }.to_string(),
                    ]
                })
                .collect();
            crate::render::write_stdout(crate::text::format_columns(
                &["Name", "Account", "Backend", "Model", "Default"],
                &rows,
            ))?;
        }
    }
    // After every branch, the empty one included: a `default_profile` naming nothing over zero
    // profiles is exactly the state worth reporting, and only this listing reports it.
    report_broken_profiles(&config_file);
    Ok(())
}

/// Every configured profile as either format prints it, the default marked by the selection rule a
/// run applies, so a sole profile with no `default_profile` is the default under both.
///
/// One list for both formats, so the marker cannot be computed one way for plain and another for
/// JSON. A profile on a missing account is listed without a backend and [`report_broken_profiles`]
/// says so on stderr.
fn profile_views(config_file: &config::ConfigFile) -> Vec<crate::view::ProfileView> {
    let (active, _) = config::select_profile(
        config_file.default_profile.clone(),
        config::ProfileRequest::DefaultProfile,
        &config_file.profiles,
    );
    config_file
        .profiles
        .iter()
        .map(|(name, profile)| {
            crate::view::ProfileView::from_config(
                name,
                profile,
                &config_file.accounts,
                Some(name.as_str()) == active.as_deref(),
            )
        })
        .collect()
}

/// The two states a listing can reveal and a run then fails on, said on stderr under either format.
fn report_broken_profiles(config_file: &config::ConfigFile) {
    let default = config_file.default_profile.as_deref();
    // A `default_profile` naming nothing renders as a table with no `*`, which is exactly what "no
    // default set" looks like, and the next `meka` run then fails on a setting the user believes is
    // fine. This listing is where they come to check, so it is where the discrepancy belongs.
    if let Some(default) = default
        && !config_file.profiles.contains_key(default)
    {
        crate::streams::write_stderr_line(format!(
            "`default_profile` names '{}': {}; point it at one with `meka profile use <name>`",
            crate::text::sanitize_for_display(default),
            crate::text::unknown_name(
                "profile",
                &crate::text::sanitize_for_display(default),
                config_file.profiles.keys()
            )
        ));
    }
    // A profile whose account is gone is listed with `-` for its backend, and said aloud: it is the
    // state a hand-deleted `[accounts.<name>]` leaves, and every session on it refuses to run.
    for (name, profile) in &config_file.profiles {
        if !config_file.accounts.contains_key(&profile.account) {
            let account = crate::text::sanitize_for_display(&profile.account);
            crate::streams::write_stderr_line(format!(
                "profile '{name}': {}; create it with `meka account add {account}`",
                crate::text::unknown_name("account", &account, config_file.accounts.keys())
            ));
        }
    }
}

// ----- Config file editing (toml_edit, comment-preserving) ---------------------------------------

/// Narrow a profile's `u64` setting to the `i64` a TOML integer actually is.
///
/// TOML has one integer type and it is signed 64-bit, so a `u64` past `i64::MAX` has no
/// representation at all; `as i64` would wrap it silently into a file every later command refuses,
/// the `profile set` that would repair it included.
fn toml_integer(field: &str, value: u64) -> anyhow::Result<i64> {
    i64::try_from(value).map_err(|_| {
        anyhow::anyhow!(
            "{} must be at most {} (the largest TOML integer), got {}",
            field,
            i64::MAX,
            value
        )
    })
}

/// Insert `[profiles.<name>]` into `document`. Pure mutation so it can be unit-tested without
/// touching the filesystem.
fn upsert_profile_document(
    document: &mut toml_edit::DocumentMut,
    name: &str,
    account: &str,
    model: &str,
    tuning: &ProfileTuning,
) -> anyhow::Result<()> {
    let mut profile = toml_edit::Table::new();
    // Written in [`config::ProfileConfig`]'s canonical order, so a profile meka authors reads the
    // way the reference documents it.
    //
    // An unset knob is left out rather than written at its default, so the profile records only
    // what the user actually chose and a later change to a default reaches existing profiles.
    profile.insert("account", toml_edit::value(account));
    profile.insert("model", toml_edit::value(model));
    if let Some(window) = tuning.context_window {
        profile.insert(
            "context_window",
            toml_edit::value(toml_integer("context_window", window)?),
        );
    }
    if let Some(cap) = tuning.max_output_tokens {
        profile.insert(
            "max_output_tokens",
            toml_edit::value(toml_integer("max_output_tokens", cap)?),
        );
    }
    if let Some(effort) = tuning.effort.as_deref() {
        profile.insert("effort", toml_edit::value(effort));
    }
    if let Some(vision) = tuning.vision {
        profile.insert("vision", toml_edit::value(vision));
    }
    if let Some(mode) = tuning.thinking {
        profile.insert("thinking", toml_edit::value(mode.name()));
    }
    if let Some(budget) = tuning.thinking_budget {
        profile.insert(
            "thinking_budget",
            toml_edit::value(toml_integer("thinking_budget", budget)?),
        );
    }
    if let Some(bytes) = tuning.max_request_bytes {
        profile.insert(
            "max_request_bytes",
            toml_edit::value(toml_integer("max_request_bytes", bytes)?),
        );
    }
    if let Some(display) = tuning.thinking_display {
        profile.insert("thinking_display", toml_edit::value(display.name()));
    }
    // Redundant here, since the inserts above already run in order, and deliberately kept: it is
    // the one line that makes "every writer leaves the canonical order" true of this writer too,
    // rather than true only for as long as nobody adds an insert in the wrong place.
    let mut profile = toml_edit::Item::Table(profile);
    config::sort_profile_keys(&mut profile);
    ensure_section_table(document, "profiles")?.insert(name, profile);
    // No `default_profile`. A sole profile is the default by the selection rule, and `use` is the
    // one command that sets the key; written here too, `add` on a config whose key had been
    // deleted by hand silently rebound every new session to the profile just added.
    Ok(())
}

/// Remove `[profiles.<name>]` from `document`, clearing `default_profile` if it pointed at the
/// removed profile. Pure mutation, unit-testable.
///
/// Answers whether a profile was actually removed, because the caller cannot tell from anywhere
/// else. Inferring it from a separate `as_table` probe made `remove` report the opposite of what it
/// did on an inline `profiles`: the profile probed as absent, so the command dropped
/// `default_profile`, left `[profiles.<name>]` untouched in the file, and said "no profile was
/// configured" about one `meka profile list` still shows.
fn remove_profile_document(document: &mut toml_edit::DocumentMut, name: &str) -> bool {
    let removed = document
        .get_mut("profiles")
        .and_then(|item| item.as_table_like_mut())
        .and_then(|profiles| profiles.remove(name))
        .is_some();
    // If this profile was the default, drop the dangling pointer.
    if document
        .get("default_profile")
        .and_then(|item| item.as_str())
        == Some(name)
    {
        document.as_table_mut().remove("default_profile");
    }
    removed
}

fn write_profile(
    name: &str,
    account: &str,
    model: &str,
    tuning: &ProfileTuning,
) -> anyhow::Result<()> {
    // `_lock` is held to the end of the function, so the read above and the write below are one
    // critical section.
    let (_lock, path, mut document) = open_document()?;
    // The same refusal `run_add` gave before its prompts, asked again now that the lock is held:
    // the prompts between the two take as long as the user takes, and a profile written in that
    // window by another `profile add` or by hand was replaced wholesale.
    if document
        .get("profiles")
        .and_then(|profiles| profiles.get(name))
        .is_some()
    {
        anyhow::bail!(
            "a profile named '{name}' already exists; change it with `meka profile set {name} <key> \
             <value>`"
        );
    }
    // The account too, for the same reason: `account remove` re-asks its own side under this lock,
    // and the pair only holds if this side asks as well. Read off the document rather than the
    // config `run_add` parsed, because that parse is what has gone stale.
    if !table_names(&document, "accounts")
        .iter()
        .any(|configured| configured == account)
    {
        return Err(unknown_account(account, table_names(&document, "accounts")));
    }
    let before = document.to_string();
    upsert_profile_document(&mut document, name, account, model, tuning)?;
    let after = document.to_string();
    refuse_a_profile_that_cannot_run(&before, &after, name)?;
    crate::fs::write_file_atomic(&path, &after)?;
    Ok(())
}

/// Refuse a write that would leave `name` unusable, before the file is touched.
///
/// Both write doors ask this, and that is the point of it being one function. With `set` asking and
/// `add` not, `profile add work --thinking budgeted --thinking-budget 32000 --max-output-tokens
/// 8000` would exit 0 and write a profile that fails at *startup* on every later run, and then be a
/// trap, because the `profile set` sent to repair it would re-derive the same refusal from the two
/// keys already on disk and decline an unrelated edit.
///
/// The parse itself is [`reparse_after_edit`]'s, shared with the account writer.
fn refuse_a_profile_that_cannot_run(before: &str, after: &str, name: &str) -> anyhow::Result<()> {
    let candidate = reparse_after_edit(before, after)?;
    if let Some(profile) = candidate.profiles.get(name) {
        // `--unset model` and `--model ""` both arrive here as a profile a run would refuse by
        // name; the predicate the run applies refuses the write first.
        config::require_model(name, profile.model.as_deref())?;
        config::validate_max_output_tokens(
            name,
            candidate
                .accounts
                .get(&profile.account)
                .and_then(|account| account.backend.parse::<config::Backend>().ok()),
            profile.max_output_tokens,
            profile.thinking.unwrap_or_default(),
            profile
                .thinking_budget
                .or(candidate.thinking.as_ref().and_then(|it| it.budget))
                .unwrap_or(config::DEFAULT_THINKING_BUDGET_TOKENS),
        )?;
    }
    Ok(())
}

/// Point `default_profile` at `name`, refusing a name that is not a profile.
///
/// The refusal is asked under the lock the write takes, of the file as it stands then. Asked of a
/// parse taken before the lock, a `profile remove` landing between the two leaves `default_profile`
/// naming a profile that is gone, and the next run fails on a key this command just wrote.
fn set_default_profile(name: &str) -> anyhow::Result<()> {
    // `_lock` is held to the end of the function, so the check and the write below are one
    // critical section.
    let (_lock, path, mut document) = open_document()?;
    // The typed parse rather than the document's table names: `use` writes a pointer for the typed
    // reader to follow, so it has nothing to do on a file that reader refuses, and
    // `require_profile` is the one definition of a name that is a profile.
    let config_file = config::load_config_file_or_err()?;
    config::require_profile(name, &config_file.profiles)?;
    document["default_profile"] = toml_edit::value(name);
    crate::fs::write_file_atomic(&path, &document.to_string())?;
    Ok(())
}

// ----- Interactive prompts -----------------------------------------------------------------------

/// What declining the advanced step leaves in force, or `None` when the flags already pinned every
/// setting so there is no default left to report.
///
/// Only the still-unset ones are named: stating a default for something the flags set would
/// contradict the file this same command is about to write. Split out from the prompt so the
/// composition is testable without stdin: the prompt is the one part of `resolve_tuning` a test
/// cannot drive.
fn unset_defaults_summary(
    tuning: &ProfileTuning,
    takes_thinking: bool,
    effective_window: u64,
) -> Option<String> {
    let mut defaults: Vec<String> = Vec::new();
    if takes_thinking && tuning.thinking.is_none() {
        defaults.push(format!(
            "thinking {}",
            crate::config::ThinkingMode::default().name()
        ));
    }
    if tuning.context_window.is_none() {
        defaults.push(format!("context window {effective_window}"));
    }
    if tuning.effort.is_none() {
        defaults.push("the provider's own reasoning effort".to_string());
    }
    (!defaults.is_empty()).then(|| defaults.join(", "))
}

/// Whether the advanced step should ask for a thinking budget.
///
/// The one flag-only setting that is also prompted for, and only in the case that creates it. A
/// budget means nothing under `adaptive` (the default) or `off`, which send no `budget_tokens` at
/// all, so asking unconditionally would put a fourth question in front of every user to serve the
/// one who just answered "budgeted", and that user needs it now, because this is where the
/// `max_output_tokens` pairing starts to matter.
///
/// Split out from the prompt for the reason [`unset_defaults_summary`] is: the prompt itself is the
/// one part of [`resolve_tuning`] a test cannot drive, so the condition deciding whether it fires
/// has to be reachable on its own or it is guarded by nothing.
fn budget_is_worth_asking_about(thinking: Option<crate::config::ThinkingMode>) -> bool {
    thinking == Some(crate::config::ThinkingMode::Budgeted)
}

/// Fill in whatever the flags didn't set, offering one opt-in prompt rather than three
/// unconditional ones. Declining leaves every unset knob out of the profile, and says which
/// defaults that implies, because these are settings meka never infers: nothing else will tell the
/// user they exist.
fn resolve_tuning(
    flags: ProfileTuning,
    backend: config::Backend,
    profile_name: &str,
    session_window: Option<u64>,
    global_budget: Option<u64>,
) -> anyhow::Result<ProfileTuning> {
    // What an unset profile would actually budget against: `[session].context_window` if the user
    // already set one, else the built-in default. Showing the constant unconditionally would state
    // a window the run is not going to use.
    let effective_window = session_window.unwrap_or(crate::provider::DEFAULT_CONTEXT_WINDOW);
    // The same question one field over: a budget prompt showing the built-in constant while the
    // profile falls back to `[thinking].budget` would make pressing Enter to "accept the default"
    // produce a different number from typing the one on screen.
    let effective_budget = global_budget.unwrap_or(crate::config::DEFAULT_THINKING_BUDGET_TOKENS);
    // `thinking` is an Anthropic Messages request field, so an OpenAI profile is neither asked
    // about it nor told what it defaults to: writing the key there would produce a setting that
    // reads plausibly and does nothing.
    let takes_thinking = backend.takes_thinking();
    let mut flags = flags;
    // The flags are dropped, not just left unprompted. Guarding only the prompt produced a run that
    // printed "Using defaults:" without thinking *and* wrote `thinking` into the profile. Every
    // backend-specific flag, through the one function that knows which backend reads what: a list
    // of the thinking three here let `--max-request-bytes` onto an `openai-responses` profile,
    // reported as set, where nothing ever read it.
    let mut dropped: Vec<&str> = Vec::new();
    if !backend.reads_profile_key("thinking") && flags.thinking.take().is_some() {
        dropped.push("--thinking");
    }
    if !backend.reads_profile_key("thinking_budget") && flags.thinking_budget.take().is_some() {
        dropped.push("--thinking-budget");
    }
    if !backend.reads_profile_key("thinking_display") && flags.thinking_display.take().is_some() {
        dropped.push("--thinking-display");
    }
    if !dropped.is_empty() {
        tracing::warn!(
            "ignoring {dropped}: a '{backend}' account never sends the field",
            dropped = dropped.join(", ")
        );
    }
    let complete = flags.context_window.is_some()
        && flags.effort.is_some()
        && (!takes_thinking || flags.thinking.is_some());
    if complete {
        return Ok(flags);
    }
    if !prompt_yes_no("Configure advanced settings? [y/N]: ")? {
        if let Some(defaults) = unset_defaults_summary(&flags, takes_thinking, effective_window) {
            crate::streams::write_stderr_line(format!(
                "Using defaults: {defaults}; change one with `meka profile set {profile_name} \
                 <key> <value>`."
            ));
        }
        return Ok(flags);
    }

    let thinking = match flags.thinking {
        Some(mode) => Some(mode),
        None if !takes_thinking => None,
        None => {
            let input = prompt_line("  Thinking mode (adaptive, budgeted, off) [adaptive]: ")?;
            match input.as_str() {
                "" => None,
                // Parsed through the enum's own `FromStr` rather than a fourth hand-written match,
                // so this prompt accepts exactly what `--thinking` accepts.
                other => Some(
                    other
                        .parse::<crate::config::ThinkingMode>()
                        .map_err(anyhow::Error::msg)?,
                ),
            }
        }
    };

    let context_window =
        match flags.context_window {
            Some(window) => Some(window),
            None => {
                let input = prompt_line(&format!(
                    "  Context window in tokens [{effective_window}]: "
                ))?;
                match input.as_str() {
                    "" => None,
                    other => Some(other.parse::<u64>().map_err(|_| {
                        anyhow::anyhow!("'{other}' is not a whole number of tokens")
                    })?),
                }
            }
        };

    let effort = match flags.effort {
        Some(effort) => Some(effort),
        None => {
            let input = prompt_line("  Reasoning effort (empty for the provider's default): ")?;
            (!input.is_empty()).then_some(input)
        }
    };

    let thinking_budget =
        match flags.thinking_budget {
            Some(budget) => Some(budget),
            None if budget_is_worth_asking_about(thinking) => {
                let input = prompt_line(&format!(
                    "  Thinking budget in tokens [{effective_budget}]: "
                ))?;
                match input.as_str() {
                    "" => None,
                    other => Some(other.parse::<u64>().map_err(|_| {
                        anyhow::anyhow!("'{other}' is not a whole number of tokens")
                    })?),
                }
            }
            None => None,
        };

    Ok(ProfileTuning {
        thinking,
        context_window,
        effort,
        thinking_budget,
        // Flag-only, so they pass through whatever the caller set. Prompting for them would make
        // the advanced step twice as long for settings that are stated far less often.
        vision: flags.vision,
        max_output_tokens: flags.max_output_tokens,
        thinking_display: flags.thinking_display,
        max_request_bytes: flags.max_request_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_ACCOUNTS: &str = "[accounts.work]\nbackend = \"anthropic-messages\"\n\n\
                                [accounts.oai]\nbackend = \"openai-responses\"\n\n";

    /// `use` is the one command that sets `default_profile`, and it refuses a name that is not a
    /// profile rather than writing a pointer at nothing.
    #[test]
    fn use_sets_the_default_and_refuses_an_unknown_profile() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            format!(
                "{TWO_ACCOUNTS}[profiles.work]\naccount = \"work\"\nmodel = \"m\"\n\n\
                 [profiles.personal]\naccount = \"work\"\nmodel = \"m\"\n"
            ),
        )
        .expect("write config");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set → run → clear cycle.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let refused = run_use("ghost");
        let set = run_use("personal");
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        assert!(refused.is_err(), "an unknown profile is not a default");
        set.expect("a configured profile becomes the default");
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(
            contents.contains("default_profile = \"personal\""),
            "{contents}"
        );
    }

    /// Removing the default among several profiles unsets the pointer and leaves the siblings, so
    /// the next run falls back to the selection rule rather than failing on a name that is gone.
    #[tokio::test]
    async fn remove_of_the_default_unsets_it_and_keeps_the_siblings() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            format!(
                "default_profile = \"work\"\n\n{TWO_ACCOUNTS}[profiles.work]\naccount = \
                 \"work\"\nmodel = \"m\"\n\n[profiles.personal]\naccount = \"work\"\nmodel = \
                 \"m\"\n"
            ),
        )
        .expect("write config");
        let manager = crate::store::Store::for_test().await;

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("work", &manager).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        result.expect("removing the default succeeds");

        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(!contents.contains("default_profile"), "{contents}");
        assert!(!contents.contains("[profiles.work]"), "{contents}");
        assert!(contents.contains("[profiles.personal]"), "{contents}");
        assert!(
            contents.contains("[accounts.work]"),
            "the account the profile billed is not the profile's to remove: {contents}"
        );
    }

    /// Without this, `profile remove typo` would report `removed profile 'typo'` and exit 0 having
    /// done nothing at all.
    #[tokio::test]
    async fn remove_refuses_a_name_that_is_not_a_profile() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            format!("{TWO_ACCOUNTS}[profiles.work]\naccount = \"work\"\nmodel = \"m\"\n"),
        )
        .expect("write config");
        let manager = crate::store::Store::for_test().await;

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_remove("typo", &manager).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("removing a name that exists nowhere must fail"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("no profile named"), "{error}");
        // The untouched profile is still there: a failed remove must not rewrite anything.
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.contains("[profiles.work]"), "{contents}");
    }

    #[test]
    fn default_model_for_known_backends() {
        assert_eq!(
            default_model_for(config::Backend::AnthropicMessages),
            "claude-opus-5"
        );
        assert_eq!(
            default_model_for(config::Backend::ClaudeSubscription),
            "claude-opus-5"
        );
        assert_eq!(
            default_model_for(config::Backend::OpenAiChatCompletions),
            "gpt-5.6-sol"
        );
        assert_eq!(
            default_model_for(config::Backend::ChatGptSubscription),
            "gpt-5.6-sol"
        );
    }

    /// Every thinking flag must be *dropped* on a backend whose requests have no thinking field,
    /// not merely left unprompted.
    ///
    /// Guarding only the prompt is the bug this closes: the run then printed a "Using defaults:"
    /// line with no thinking in it *and* wrote `thinking` into the profile anyway, so the file
    /// disagreed with what the user had just been told, and the key sat there reading plausibly
    /// while doing nothing. Both halves are asserted, because either one alone passes the wrong
    /// implementation.
    ///
    /// All three flags: the three are one request field between them, so a guard that names one of
    /// them is a guard for none of them, and `--thinking-budget` walks straight through a guard on
    /// `thinking` alone.
    #[test]
    fn a_thinking_flag_aimed_at_a_backend_without_thinking_is_dropped() {
        let flags = || ProfileTuning {
            thinking: Some(config::ThinkingMode::Budgeted),
            thinking_budget: Some(2_048),
            thinking_display: Some(crate::config::ThinkingDisplay::Redacted),
            context_window: Some(1_024),
            effort: Some("low".to_string()),
            ..Default::default()
        };
        // Every setting is pinned, so this returns before any prompt: no stdin involved.
        let openai = resolve_tuning(
            flags(),
            config::Backend::OpenAiChatCompletions,
            "oai",
            None,
            None,
        )
        .expect("resolve");
        assert_eq!(openai.thinking, None, "written into an OpenAI profile");
        assert_eq!(
            openai.thinking_budget, None,
            "the budget is the same request field, one key over"
        );
        assert_eq!(
            openai.thinking_display, None,
            "and so is the redaction of what it produces"
        );
        assert_eq!(openai.context_window, Some(1_024));
        assert_eq!(openai.effort.as_deref(), Some("low"));

        let claude = resolve_tuning(
            flags(),
            config::Backend::AnthropicMessages,
            "work",
            None,
            None,
        )
        .expect("resolve");
        assert_eq!(
            claude.thinking,
            Some(config::ThinkingMode::Budgeted),
            "the same flag must still reach a Claude profile"
        );
        assert_eq!(
            claude.thinking_budget,
            Some(2_048),
            "and so must the budget"
        );
        // Not the redaction flag, which is a narrower question: `thinking_display` gates a beta
        // header only `claude-subscription` sends, so an `anthropic-messages` profile that stored
        // it would carry a setting that reads plausibly and is never consulted. See
        // `Backend::reads_profile_key`.
        assert_eq!(
            claude.thinking_display, None,
            "anthropic-messages takes a thinking field but never sends the redaction beta"
        );
        let subscription = resolve_tuning(
            flags(),
            config::Backend::ClaudeSubscription,
            "work",
            None,
            None,
        )
        .expect("resolve");
        assert_eq!(
            subscription.thinking_display,
            Some(crate::config::ThinkingDisplay::Redacted),
            "the one backend that does send it keeps the flag"
        );
    }

    /// The defaults line names what is still unset, and nothing else.
    ///
    /// It is the only thing that tells a user these settings exist: meka never infers them, so
    /// nothing else in the run mentions them. Naming one the flags already pinned would state a
    /// default that contradicts the file being written in the same breath.
    #[test]
    fn the_defaults_line_reports_only_the_settings_still_unset() {
        let bare = ProfileTuning::default();
        let claude = unset_defaults_summary(&bare, true, 1_000_000).expect("something is unset");
        assert!(claude.contains("thinking adaptive"), "{claude}");
        assert!(claude.contains("context window 1000000"), "{claude}");
        assert!(claude.contains("reasoning effort"), "{claude}");

        // No thinking on a backend that has no such field, so it is not reported either.
        let openai = unset_defaults_summary(&bare, false, 1_000_000).expect("something is unset");
        assert!(!openai.contains("thinking"), "{openai}");

        // The window reported is the one an unset profile would actually budget against, which is
        // `[session].context_window` when the user already set one.
        let session_window = unset_defaults_summary(&bare, false, 262_144).expect("unset");
        assert!(
            session_window.contains("context window 262144"),
            "{session_window}"
        );

        // A pinned setting is not reported as a default.
        let pinned = ProfileTuning {
            thinking: Some(config::ThinkingMode::Off),
            context_window: Some(8_192),
            effort: None,
            ..Default::default()
        };
        let partial = unset_defaults_summary(&pinned, true, 1_000_000).expect("effort is unset");
        assert!(!partial.contains("thinking"), "{partial}");
        assert!(!partial.contains("context window"), "{partial}");
        assert!(partial.contains("reasoning effort"), "{partial}");

        // Nothing unset, nothing to say.
        assert_eq!(
            unset_defaults_summary(
                &ProfileTuning {
                    thinking: Some(config::ThinkingMode::Off),
                    context_window: Some(8_192),
                    effort: Some("low".to_string()),
                    ..Default::default()
                },
                true,
                1_000_000,
            ),
            None
        );
    }

    /// `set` edits one key and leaves the rest of the file exactly as the user wrote it.
    ///
    /// The whole reason this is not [`upsert_profile_document`]: replacing the table would rewrite
    /// every key in meka's order and drop the comments beside them, so a user who ran `set` once
    /// would find their annotated config quietly reformatted and their notes gone. Nothing else
    /// would fail, which is why the guard is here.
    ///
    /// A key's surroundings live in two decors and `insert` clears both, so this asserts on both:
    /// carrying the *value*'s decor alone saves the trailing `# note` and still deletes the
    /// whole-line comment and blank line above the key, since those belong to the key's leaf decor.
    /// Half a fix passes an assertion on either half alone.
    #[test]
    fn set_edits_one_key_and_preserves_everything_around_it() {
        let mut document = r#"default_profile = "work"

# Why this profile exists.
[profiles.work]
account = "work"

# The 1M-window model; keep context_window in step with it.
model = "old-model"       # the model, annotated
context_window = 200000

[profiles.other]
account = "oai"
model = "untouched"
"#
        .parse::<toml_edit::DocumentMut>()
        .expect("parse");

        assert!(set_profile_field(
            &mut document,
            "work",
            "model",
            Some(toml_edit::Value::from("new-model"))
        ));

        let rendered = document.to_string();
        assert!(rendered.contains("model = \"new-model\""), "{rendered}");
        assert!(
            rendered.contains("# Why this profile exists."),
            "the comment above the table survives: {rendered}"
        );
        assert!(
            rendered.contains("# the model, annotated"),
            "the comment beside the edited key survives: {rendered}"
        );
        assert!(
            rendered.contains("# The 1M-window model; keep context_window in step with it."),
            "and so does the one above it, which lives in the key's decor rather than the \
             value's: {rendered}"
        );
        assert!(
            rendered.contains("\n\n# The 1M-window model"),
            "including the blank line that separated it from the key before: {rendered}"
        );
        assert!(
            rendered.contains("context_window = 200000"),
            "the profile's other keys survive: {rendered}"
        );
        assert!(
            rendered.contains("model = \"untouched\""),
            "another profile is not touched: {rendered}"
        );
        assert!(
            rendered.find("account").unwrap() < rendered.find("context_window").unwrap(),
            "key order is the user's, not meka's: {rendered}"
        );
    }

    /// A profile setting past `i64::MAX` is refused rather than wrapped into the file.
    ///
    /// TOML has one integer type and it is signed, so `as i64` turned `u64::MAX` into `-1` and
    /// wrote it. `profile add brick --context-window 18446744073709551615` then exited 0 having
    /// bricked the config: every later command, including the `profile set` that would have
    /// repaired it, refused the file with `invalid value: integer -1, expected u64`.
    #[test]
    fn an_integer_too_large_for_toml_is_refused_rather_than_wrapped() {
        for field in ["context_window", "thinking_budget", "max_output_tokens"] {
            let error = toml_integer(field, u64::MAX).expect_err("u64::MAX has no TOML form");
            let message = error.to_string();
            assert!(
                message.contains(field) && message.contains(&i64::MAX.to_string()),
                "the refusal names the setting and the ceiling: {message}"
            );
        }
        assert_eq!(
            toml_integer("context_window", 1_000_000).expect("an ordinary window fits"),
            1_000_000
        );
        // The boundary itself, so the refusal is `>` and not `>=`.
        assert_eq!(
            toml_integer("context_window", i64::MAX as u64).expect("the largest legal value fits"),
            i64::MAX
        );

        // And through the door `set` actually uses, since that is where a user meets it.
        let error = parse_profile_value("work", "context_window", &u64::MAX.to_string())
            .expect_err("`set` refuses it too");
        assert!(
            error.to_string().contains("context_window"),
            "{}",
            error.to_string()
        );
    }

    /// The shared refusal both write doors ask, checked directly.
    ///
    /// Reached only through `write_profile` and `run_set`, which both touch the real config path,
    /// so nothing exercised it: `cargo mutants` replaced the whole function with `Ok(())` and the
    /// suite stayed green. It is a pure function of the two documents and a name, so it does not
    /// need the filesystem to be tested, only to be called.
    #[test]
    fn a_profile_that_could_not_start_is_refused_by_both_write_doors() {
        let good = "[accounts.work]\nbackend = \"anthropic-messages\"\n\n[profiles.work]\naccount = \
                    \"work\"\nmodel = \"m\"\nthinking = \"budgeted\"\nthinking_budget = \
                    8000\nmax_output_tokens = 32000\n";
        let cap_below_budget = "[accounts.work]\nbackend = \
                                \"anthropic-messages\"\n\n[profiles.work]\naccount = \
                                \"work\"\nmodel = \"m\"\nthinking = \"budgeted\"\n\
                                thinking_budget = 32000\nmax_output_tokens = 8000\n";
        refuse_a_profile_that_cannot_run(good, good, "work").expect("a workable pairing passes");
        let error = refuse_a_profile_that_cannot_run(good, cap_below_budget, "work")
            .expect_err("a cap at or below the budget cannot produce a valid request");
        let message = error.to_string();
        assert!(
            message.contains("work") && message.contains("32000"),
            "the refusal names the profile and the budget it must exceed: {message}"
        );

        // The budget falls back to the global when the profile states none, so the same pairing is
        // refused even though neither number is written beside the other.
        let global_budget = "[thinking]\nbudget = 64000\n\n[accounts.work]\nbackend = \
                             \"anthropic-messages\"\n\n[profiles.work]\naccount = \
                             \"work\"\nmodel = \"m\"\nthinking = \"budgeted\"\n\
                             max_output_tokens = 16000\n";
        refuse_a_profile_that_cannot_run(good, global_budget, "work")
            .expect_err("the fallback budget is checked too, not skipped for being absent");

        // A profile the edit did not touch is not this call's business.
        refuse_a_profile_that_cannot_run(good, cap_below_budget, "other")
            .expect("only the named profile is judged");
    }

    /// A profile that names no model, or an empty one, is refused by both write doors.
    ///
    /// `--unset model` and `--model ""` reach the same place: a profile every run refuses by name
    /// and every session on it refuses to resume through. `""` is the half `is_none()` misses, and
    /// the one that would otherwise go to the provider as the model's name.
    #[test]
    fn a_profile_without_a_model_is_refused_by_both_write_doors() {
        let with_model = "[accounts.work]\nbackend = \"anthropic-messages\"\n\n[profiles.work]\n\
                          account = \"work\"\nmodel = \"m\"\n";
        let without_model = "[accounts.work]\nbackend = \"anthropic-messages\"\n\n\
                             [profiles.work]\naccount = \"work\"\n";
        let empty_model = "[accounts.work]\nbackend = \"anthropic-messages\"\n\n[profiles.work]\n\
                           account = \"work\"\nmodel = \"\"\n";
        refuse_a_profile_that_cannot_run(without_model, with_model, "work")
            .expect("a profile that gains a model can run");
        for broken in [without_model, empty_model] {
            let message = refuse_a_profile_that_cannot_run(with_model, broken, "work")
                .expect_err("a profile with no model cannot run")
                .to_string();
            assert!(
                message.contains("names no model") && message.contains("meka profile set work"),
                "the refusal names the profile and the command that fixes it: {message}"
            );
        }
        // And at the parse door, before the lock is taken.
        assert!(parse_profile_value("work", "model", "").is_err());
        assert!(parse_profile_value("work", "model", "   ").is_err());
    }

    /// `--model ""` is refused at the flag, ahead of the prompts and with the flag as the remedy.
    ///
    /// The write door refuses it too, but its remedy is `meka profile set <name> model`, on a
    /// profile this command has not created.
    #[test]
    fn add_refuses_an_empty_model_flag_before_anything_is_written() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), TWO_ACCOUNTS).expect("write config");
        // Every prompted setting is pinned, so `resolve_tuning` would return before any prompt if
        // the refusal were missing: no stdin involved either way.
        let pinned = ProfileTuning {
            context_window: Some(1_000),
            effort: Some("low".to_string()),
            thinking: Some(config::ThinkingMode::Off),
            ..Default::default()
        };

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = run_add("blank", Some("work"), Some(String::new()), pinned);
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = result.expect_err("an empty model").to_string();
        assert!(error.contains("--model"), "the remedy is the flag: {error}");
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(!contents.contains("blank"), "{contents}");
    }

    /// The account is checked under the lock the write takes, not only before the prompts.
    ///
    /// `run_add` asks of the config it parsed before prompting, and `account remove` re-asks its
    /// own side under this same lock; the pair holds only if the write asks too, or a removal that
    /// landed during the prompts leaves a profile naming an account that is gone.
    #[test]
    fn write_profile_refuses_an_account_the_locked_file_does_not_have() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[accounts.work]\nbackend = \"anthropic-messages\"\n",
        )
        .expect("write config");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let refused = write_profile("daily", "gone", "m", &ProfileTuning::default());
        let written = write_profile("daily", "work", "m", &ProfileTuning::default());
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = refused
            .expect_err("an account that is not configured")
            .to_string();
        assert!(error.contains("no account named 'gone'"), "{error}");
        written.expect("a configured account");
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.contains("account = \"work\""), "{contents}");
        assert!(
            !contents.contains("gone"),
            "a refused write leaves nothing: {contents}"
        );
    }

    /// `default_profile` is written only for a profile the locked file has.
    ///
    /// Asked under the lock rather than of a parse taken before it, so a `profile remove` landing
    /// between the two cannot leave the key naming a profile that is gone.
    #[test]
    fn set_default_profile_refuses_a_name_the_locked_file_does_not_have() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            format!("{TWO_ACCOUNTS}[profiles.work]\naccount = \"work\"\nmodel = \"m\"\n"),
        )
        .expect("write config");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let refused = set_default_profile("ghost");
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = refused
            .expect_err("a name that is not a profile")
            .to_string();
        assert!(error.contains("no profile named 'ghost'"), "{error}");
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(!contents.contains("default_profile"), "{contents}");
    }

    /// Plain and JSON mark the same profile as the default: the one the selection rule picks, so a
    /// sole profile with no `default_profile` is marked under both rather than only under JSON.
    #[test]
    fn the_default_marker_follows_the_selection_rule_under_either_format() {
        let sole: config::ConfigFile = toml::from_str(&format!(
            "{TWO_ACCOUNTS}[profiles.work]\naccount = \"work\"\nmodel = \"m\"\n"
        ))
        .expect("parse");
        let views = profile_views(&sole);
        assert_eq!(views.len(), 1);
        assert!(
            views[0].active,
            "a sole profile is the default by the selection rule"
        );

        let two: config::ConfigFile = toml::from_str(&format!(
            "{TWO_ACCOUNTS}[profiles.work]\naccount = \"work\"\nmodel = \"m\"\n\n\
             [profiles.side]\naccount = \"oai\"\nmodel = \"m\"\n"
        ))
        .expect("parse");
        assert!(
            profile_views(&two).iter().all(|view| !view.active),
            "two profiles and no `default_profile` pick nothing"
        );

        let pointed: config::ConfigFile = toml::from_str(&format!(
            "default_profile = \"side\"\n\n{TWO_ACCOUNTS}[profiles.work]\naccount = \
             \"work\"\nmodel = \"m\"\n\n[profiles.side]\naccount = \"oai\"\nmodel = \"m\"\n"
        ))
        .expect("parse");
        let active: Vec<String> = profile_views(&pointed)
            .into_iter()
            .filter(|view| view.active)
            .map(|view| view.name)
            .collect();
        assert_eq!(active, ["side"]);
    }

    /// A file that was already unreadable is not blamed on the edit that found it.
    ///
    /// The reparse covers the whole document, so any unrelated defect anywhere in it blocks the
    /// write. Reporting that as "that change makes config.toml unreadable" would send the user
    /// looking at the key they had just set, which is not the problem. Still a refusal either way;
    /// only the sentence changes, and the sentence is the whole value.
    #[test]
    fn a_pre_existing_config_error_is_not_blamed_on_this_edit() {
        let already_broken = "[typo_section]\nfoo = 1\n";
        let error = refuse_a_profile_that_cannot_run(already_broken, already_broken, "work")
            .expect_err("an unreadable file is refused");
        assert!(
            error.to_string().contains("not what broke it"),
            "the message must say the edit is not what broke it: {error}"
        );

        let fine = "[profiles.work]\naccount = \"work\"\n";
        let error = refuse_a_profile_that_cannot_run(fine, already_broken, "work")
            .expect_err("an edit that breaks parsing is refused");
        assert!(
            error.to_string().contains("makes config.toml unreadable"),
            "and when the edit *is* what broke it, it says so: {error}"
        );
    }

    /// A thinking key is refused on a profile whose account's backend never sends one.
    ///
    /// `profile add` drops such a flag with a warning; `set` refuses, because a command whose
    /// entire job is one key cannot report success having written nothing. Both doors reach the
    /// same place: the key does not end up in a profile that ignores it.
    #[test]
    fn setting_a_thinking_key_on_a_backend_without_thinking_is_refused() {
        let document = r#"[accounts.oai]
backend = "openai-responses"

[accounts.work]
backend = "anthropic-messages"

[accounts.typo]
backend = "not-a-backend"

[profiles.oai]
account = "oai"
model = "gpt-5.6"

[profiles.work]
account = "work"
model = "claude-opus-5"

[profiles.typo]
account = "typo"
model = "m"

[profiles.orphan]
account = "gone"
model = "m"
"#
        .parse::<toml_edit::DocumentMut>()
        .expect("parse");

        for key in ["thinking", "thinking_budget", "thinking_display"] {
            let error = refuse_an_inert_key(&document, "oai", key)
                .expect_err("inert on a Responses profile");
            let message = error.to_string();
            assert!(
                message.contains(key) && message.contains("openai-responses"),
                "the refusal names the key and the backend: {message}"
            );
            // `anthropic-messages` sends a thinking field but not the redaction beta, so the two
            // groups part company here rather than moving together.
            let on_messages = refuse_an_inert_key(&document, "work", key);
            if key == "thinking_display" {
                let message = on_messages
                    .expect_err("the redaction beta is claude-subscription's alone")
                    .to_string();
                assert!(
                    message.contains(key) && message.contains("anthropic-messages"),
                    "the refusal names the key and the backend: {message}"
                );
            } else {
                on_messages.expect("the thinking field itself is live on a Messages profile");
            }
            // An unrecognized backend is `validate_backend`'s to report, and a missing account is
            // `account_for`'s. Refusing here would answer a typo in one key with a complaint about
            // a different one.
            refuse_an_inert_key(&document, "typo", key)
                .expect("an unknown backend is not this function's to judge");
            refuse_an_inert_key(&document, "orphan", key)
                .expect("a missing account is not this function's to judge");
        }
        refuse_an_inert_key(&document, "oai", "model")
            .expect("a key every backend reads passes on any backend");
        // The request ceiling is every backend's: a default on the Anthropic ones, applied only
        // when stated on the others.
        refuse_an_inert_key(&document, "oai", "max_request_bytes")
            .expect("a Responses profile may state its endpoint's ceiling");
    }

    /// `--unset` removes the key rather than writing an empty value.
    ///
    /// The two are not the same: an absent key follows whatever meka's default becomes, which is
    /// the documented meaning of an unstated setting, while `model = ""` is a model named the empty
    /// string and would be sent as one.
    #[test]
    fn unsetting_removes_the_key_rather_than_emptying_it() {
        let mut document = r#"[profiles.work]
account = "work"
effort = "high"
"#
        .parse::<toml_edit::DocumentMut>()
        .expect("parse");

        assert!(set_profile_field(&mut document, "work", "effort", None));
        let rendered = document.to_string();
        assert!(!rendered.contains("effort"), "the key is gone: {rendered}");
        assert!(rendered.contains("account ="), "the rest stays: {rendered}");
    }

    /// A profile that is not there is reported, never silently created.
    ///
    /// Answering `true` here would have `set` write a `[profiles.<typo>]` table with one key in
    /// it, report success, and leave the user's real profile unchanged with no indication why.
    #[test]
    fn setting_a_key_on_an_absent_profile_says_so() {
        let mut document = r#"[profiles.work]
account = "work"
"#
        .parse::<toml_edit::DocumentMut>()
        .expect("parse");

        assert!(!set_profile_field(
            &mut document,
            "ghost",
            "model",
            Some(toml_edit::Value::from("x"))
        ));
        assert!(
            !document.to_string().contains("ghost"),
            "a refused set writes nothing at all"
        );
    }

    /// Every settable key parses its own value type, and refuses what it cannot mean.
    #[test]
    fn each_profile_key_parses_the_way_its_add_flag_does() {
        assert!(parse_profile_value("work", "model", "claude-opus-5").is_ok());
        assert!(parse_profile_value("work", "context_window", "200000").is_ok());
        assert!(parse_profile_value("work", "context_window", "lots").is_err());
        assert!(parse_profile_value("work", "vision", "false").is_ok());
        assert!(parse_profile_value("work", "vision", "no").is_err());
        assert!(parse_profile_value("work", "thinking", "budgeted").is_ok());
        assert!(parse_profile_value("work", "thinking", "sideways").is_err());
        assert!(parse_profile_value("work", "thinking_budget", "2048").is_ok());

        // Every key in the advertised list has an arm. A key listed but unhandled would fall to the
        // catch-all and be refused as unknown, which reads as meka having forgotten its own field.
        for key in SETTABLE_PROFILE_KEYS {
            let sample = match *key {
                "context_window" | "max_output_tokens" | "thinking_budget"
                | "max_request_bytes" => "1000",
                "vision" => "true",
                "thinking_display" => "updates",
                "thinking" => "adaptive",
                _ => "value",
            };
            assert!(
                parse_profile_value("work", key, sample).is_ok(),
                "'{key}' is advertised as settable but does not parse"
            );
        }
    }

    /// An unknown key is refused whichever way it arrived, including behind `--unset`.
    ///
    /// The refusal must not live inside the value branch, where `--unset modle` skips it entirely:
    /// the command exits 0, removes nothing, and tells the user nothing. Nothing else would notice,
    /// because the write it did not do is indistinguishable from a key that was already absent.
    #[test]
    fn an_unknown_key_is_refused_whether_or_not_a_value_came_with_it() {
        let error = ensure_settable_key("modle").expect_err("a typo is not a setting");
        assert!(
            error.to_string().contains("is not a profile setting"),
            "{}",
            error
        );
        for key in SETTABLE_PROFILE_KEYS {
            assert!(
                ensure_settable_key(key).is_ok(),
                "'{key}' is advertised as settable but the door refuses it"
            );
        }
        assert!(
            ensure_settable_key("account").is_err() && ensure_settable_key("base_url").is_err(),
            "the deliberate exclusions are refused by the same door"
        );
    }

    /// The advanced step asks for a budget exactly when one will be sent, and never otherwise.
    ///
    /// `adaptive` and `off` send no `budget_tokens` at all, so a budget collected under them is a
    /// question asked for nothing and a key written into the profile that does nothing. Inverting
    /// this, or pinning it to `true`, is invisible to every other test: the prompt is the one part
    /// of `resolve_tuning` a test cannot drive.
    #[test]
    fn the_budget_is_asked_for_under_budgeted_and_nothing_else() {
        use crate::config::ThinkingMode;
        assert!(budget_is_worth_asking_about(Some(ThinkingMode::Budgeted)));
        assert!(!budget_is_worth_asking_about(Some(ThinkingMode::Adaptive)));
        assert!(!budget_is_worth_asking_about(Some(ThinkingMode::Off)));
        assert!(
            !budget_is_worth_asking_about(None),
            "an unstated thinking mode takes the default, which is adaptive and sends no budget"
        );
    }

    /// `account` is refused with the reason, not merely omitted from the list, and an account key
    /// says where it lives.
    ///
    /// A user who typed it did not typo, so "settable: model, effort, …" on its own would read as
    /// an oversight rather than a decision, and the obvious next move would be to hand-edit the key
    /// that meka is declining to change for them.
    #[test]
    fn changing_a_profiles_account_is_refused_with_its_reason() {
        let error = parse_profile_value("work", "account", "oai")
            .expect_err("the account is not settable in place");
        let message = error.to_string();
        assert!(
            message.contains("credential") && message.contains("add a profile"),
            "the refusal must say why and where to go: {message}"
        );

        let base_url = parse_profile_value("work", "base_url", "https://x.invalid")
            .expect_err("an account setting is not a profile setting");
        assert!(
            base_url.to_string().contains("[accounts.<name>]"),
            "{}",
            base_url.to_string()
        );
    }

    /// Every flag-only setting reaches the profile, and none is written when its flag was absent.
    ///
    /// These never pass through a prompt, so the writer is the only thing standing between the
    /// flag and the file: a missing `insert` would make `--vision false` exit 0 and change nothing,
    /// and the profile would keep advertising images the model cannot take. The absent half matters
    /// for the reason the sibling test gives: an unstated key follows the documented default, and
    /// writing it eagerly would freeze today's default into every profile.
    #[test]
    fn the_flag_only_settings_reach_the_profile_and_only_when_given() {
        let mut bare = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut bare,
            "local",
            "work",
            "some-local-model",
            &ProfileTuning::default(),
        )
        .expect("upsert");
        let rendered = bare.to_string();
        for key in [
            "thinking_budget",
            "vision",
            "max_output_tokens",
            "thinking_display",
            "max_request_bytes",
        ] {
            assert!(!rendered.contains(key), "{key} written unasked: {rendered}");
        }
        // `use` sets the default; a sole profile is the default by the selection rule, and writing
        // the key here rebound every new session on a config whose key had been deleted by hand.
        assert!(
            !rendered.contains("default_profile"),
            "add must not set the default: {rendered}"
        );

        let mut tuned = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut tuned,
            "local",
            "work",
            "some-local-model",
            &ProfileTuning {
                thinking_budget: Some(4_096),
                vision: Some(false),
                max_output_tokens: Some(32_000),
                thinking_display: Some(crate::config::ThinkingDisplay::Summarized),
                max_request_bytes: Some(8_388_608),
                ..Default::default()
            },
        )
        .expect("upsert");

        // Read back through the real parser, so a key written under the wrong name or type is
        // caught here rather than by `deny_unknown_fields` on the user's next run.
        let parsed: config::ConfigFile = toml::from_str(&tuned.to_string()).expect("parses");
        let profile = parsed.profiles.get("local").expect("the profile");
        assert_eq!(profile.account, "work");
        assert_eq!(profile.thinking_budget, Some(4_096));
        assert_eq!(profile.vision, Some(false));
        assert_eq!(profile.max_output_tokens, Some(32_000));
        assert_eq!(
            profile.thinking_display,
            Some(crate::config::ThinkingDisplay::Summarized)
        );
        assert_eq!(profile.max_request_bytes, Some(8_388_608));
    }

    /// The advanced prompt covers exactly the three settings it has always covered.
    ///
    /// `complete` is what decides whether the prompt fires at all. Widening it to the flag-only
    /// settings would mean `--vision false` silently skips the thinking / window / effort
    /// questions, so a user who set one advanced thing would never be asked about the three that
    /// matter most, and would get their defaults without being told. Nothing else would fail.
    #[test]
    fn a_flag_only_setting_does_not_suppress_the_advanced_prompt() {
        let all_three = ProfileTuning {
            thinking: Some(config::ThinkingMode::Off),
            context_window: Some(1_000),
            effort: Some("low".to_string()),
            ..Default::default()
        };
        assert!(
            unset_defaults_summary(&all_three, true, 1_000).is_none(),
            "with the prompted three set there is no default left to report"
        );

        let only_flag_only = ProfileTuning {
            vision: Some(false),
            max_request_bytes: Some(1_024),
            ..Default::default()
        };
        let summary = unset_defaults_summary(&only_flag_only, true, 1_000)
            .expect("the prompted three are still unset, so their defaults are still worth naming");
        for expected in ["thinking", "context window", "reasoning effort"] {
            assert!(
                summary.contains(expected),
                "'{expected}' missing from: {summary}"
            );
        }
    }

    #[test]
    fn the_advanced_settings_are_written_only_when_chosen() {
        // Unset knobs stay out of the profile rather than being written at their defaults, so a
        // profile records the user's choices and a later change to a default still reaches it.
        let mut bare = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut bare,
            "local",
            "work",
            "some-local-model",
            &ProfileTuning::default(),
        )
        .expect("upsert");
        let rendered = bare.to_string();
        for key in ["thinking", "context_window", "effort"] {
            assert!(!rendered.contains(key), "{key} in: {rendered}");
        }

        // Chosen ones round-trip through the runtime config, which is what actually reads them.
        let mut tuned = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut tuned,
            "local",
            "work",
            "some-local-model",
            &ProfileTuning {
                thinking: Some(config::ThinkingMode::Budgeted),
                context_window: Some(262_144),
                effort: Some("medium".to_string()),
                ..Default::default()
            },
        )
        .expect("upsert");
        let config: config::ConfigFile =
            toml::from_str(&tuned.to_string()).expect("re-parse config");
        let profile = config.profiles.get("local").expect("profile present");
        assert_eq!(profile.thinking, Some(config::ThinkingMode::Budgeted));
        assert_eq!(profile.context_window, Some(262_144));
        assert_eq!(profile.effort.as_deref(), Some("medium"));
    }

    #[test]
    fn upsert_profile_document_sets_no_default() {
        let mut document = toml_edit::DocumentMut::new();
        upsert_profile_document(
            &mut document,
            "work",
            "oai",
            "gpt-4o",
            &ProfileTuning::default(),
        )
        .expect("upsert");
        // The rendered TOML must parse back into the runtime config with the profile. No default
        // is written: a sole profile is the default by the selection rule, and `use` is the one
        // command that sets the key.
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert_eq!(config.default_profile, None);
        let profile = config.profiles.get("work").expect("profile present");
        assert_eq!(profile.account, "oai");
        assert_eq!(profile.model.as_deref(), Some("gpt-4o"));

        let mut with_default: toml_edit::DocumentMut = "default_profile = \"work\"\n"
            .parse()
            .expect("a document with a default");
        upsert_profile_document(
            &mut with_default,
            "personal",
            "oai",
            "gpt-4o",
            &ProfileTuning::default(),
        )
        .expect("upsert personal");
        let config: config::ConfigFile =
            toml::from_str(&with_default.to_string()).expect("re-parse config");
        // The default must remain the first profile, not silently flip to the newest one.
        assert_eq!(config.default_profile.as_deref(), Some("work"));
        assert!(config.profiles.contains_key("personal"));
    }

    /// `profiles = { work = { … } }` is valid TOML that serde reads, so `profile list` and the
    /// duplicate guard both see `work`. Treating "not a header table" as "absent" and overwriting
    /// would destroy it: one `profile add home` would leave a config naming only `home`, silently,
    /// with exit 0.
    #[test]
    fn adding_a_profile_refuses_an_inline_profiles_table_rather_than_replacing_it() {
        let mut document = "default_profile = \"work\"\n\
             profiles = { work = { account = \"a\", model = \"m1\" } }\n"
            .parse::<toml_edit::DocumentMut>()
            .expect("parse");
        let error =
            upsert_profile_document(&mut document, "home", "a", "m2", &ProfileTuning::default())
                .expect_err("an inline `profiles` must be refused, not overwritten");
        assert!(
            error.to_string().contains("is not a section"),
            "the refusal must say what to do about it: {error}"
        );
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert!(
            config.profiles.contains_key("work"),
            "the existing profile must survive a refused add"
        );
        assert_eq!(config.default_profile.as_deref(), Some("work"));
    }

    /// The other half of the same rule. Removal *is* safe on either spelling, and the caller cannot
    /// infer whether it happened from a separate `as_table` probe: such a probe says "absent" while
    /// the profile is in the file, so `remove` would drop `default_profile`, leave
    /// `[profiles.work]` in place, and report "no profile was configured" about one `meka profile
    /// list` still shows.
    #[test]
    fn removing_a_profile_reaches_an_inline_profiles_table_and_says_it_did() {
        let mut document = "default_profile = \"work\"\n\
             profiles = { work = { account = \"a\", model = \"m1\" }, \
             side = { account = \"a\", model = \"m2\" } }\n"
            .parse::<toml_edit::DocumentMut>()
            .expect("parse");
        assert!(
            remove_profile_document(&mut document, "work"),
            "a profile that was there must be reported as removed"
        );
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert!(!config.profiles.contains_key("work"));
        assert!(config.profiles.contains_key("side"));
        assert_eq!(config.default_profile, None);

        assert!(
            !remove_profile_document(&mut document, "work"),
            "a second removal has nothing to remove and must say so"
        );
    }

    #[test]
    fn remove_profile_document_clears_dangling_default() {
        let mut document: toml_edit::DocumentMut = "default_profile = \"work\"\n"
            .parse()
            .expect("a document with a default");
        upsert_profile_document(
            &mut document,
            "work",
            "a",
            "claude-x",
            &ProfileTuning::default(),
        )
        .expect("upsert work");
        upsert_profile_document(
            &mut document,
            "personal",
            "a",
            "gpt-4o",
            &ProfileTuning::default(),
        )
        .expect("upsert personal");
        remove_profile_document(&mut document, "work");
        let config: config::ConfigFile =
            toml::from_str(&document.to_string()).expect("re-parse config");
        assert!(!config.profiles.contains_key("work"));
        assert!(config.profiles.contains_key("personal"));
        // `work` was the default; removing it must drop the dangling pointer rather than leave it.
        assert!(config.default_profile.is_none());
    }

    /// The two key lists cannot drift, because one is stated as a function of the other; two
    /// separate lists in separate files diverge, under a doc comment claiming they match.
    #[test]
    fn the_settable_keys_are_the_canonical_order_minus_what_set_refuses() {
        let expected: Vec<&str> = crate::config::PROFILE_KEY_ORDER
            .iter()
            .copied()
            .filter(|key| *key != "account")
            .collect();
        assert_eq!(SETTABLE_PROFILE_KEYS, expected.as_slice());
    }
}
