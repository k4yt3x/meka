//! `meka profile` subcommand suite: the flags, the prompts and the listing, over the doors in
//! [`crate::installation::profile`].

use super::account::{prompt_line, prompt_yes_no};
use crate::{
    cli::ProfileAction,
    config,
    installation::{
        Patch,
        account::validate_backend,
        profile::{
            ProfileChanges, ProfileTuning, edit_profile, remove_profile, rename_profile,
            write_profile,
        },
    },
};

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
            fallback,
            context_window,
            max_output_tokens,
            max_request_bytes,
            effort,
            vision,
            deferred_tools,
            thinking,
            thinking_budget,
            cache_control,
            thinking_display,
        } => run_add(name, account.as_deref(), model.clone(), ProfileTuning {
            fallbacks: fallback.clone(),
            context_window: *context_window,
            max_output_tokens: *max_output_tokens,
            max_request_bytes: *max_request_bytes,
            effort: effort.clone(),
            vision: *vision,
            deferred_tools: *deferred_tools,
            thinking: *thinking,
            thinking_budget: *thinking_budget,
            cache_control: *cache_control,
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
        ProfileAction::Remove { name } => remove_profile(name, store).await,
        ProfileAction::Rename { name, new_name } => {
            rename_profile(name, new_name, store).await.map(|_| ())
        }
    }
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

    write_profile(name, &account_name, model.as_str(), &tuning, false)?;
    tracing::info!("added profile '{name}' on account '{account_name}'");
    Ok(())
}

/// The refusal `profile add` gives for a profile naming an account that is not configured, ahead
/// of its prompts; the door under the lock refuses in the bare words every surface shares.
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
        config::Backend::AnthropicMessages | config::Backend::ClaudeSubscription => {
            "claude-opus-5-5"
        }
        config::Backend::OpenAiChatCompletions
        | config::Backend::OpenAiResponses
        | config::Backend::ChatGptSubscription => "gpt-6-astra",
    }
}

/// The settings `meka profile set` will write, in [`config::ProfileConfig`]'s canonical order
/// minus `account`.
///
/// `account` is absent. Moving a profile to another account moves every session on it to another
/// credential and possibly another backend, silently, which is the one thing this command must not
/// be able to do; a new profile on that account is one command away.
pub(super) const SETTABLE_PROFILE_KEYS: &[&str] = &[
    "model",
    "fallbacks",
    "context_window",
    "max_output_tokens",
    "max_request_bytes",
    "effort",
    "vision",
    "deferred_tools",
    "thinking",
    "thinking_budget",
    "cache_control",
    "thinking_display",
];

/// One `key`/`value` pair as typed at a shell, as the edit door takes it; `None` removes the key.
///
/// Each key parses the way the matching `profile add` flag does, `thinking` through the same
/// `ValueEnum` for the reason `resolve_tuning`'s prompt gives: a second hand-written match would
/// be a second thing to keep in step with the enum. What a value may be is the door's to decide,
/// where every surface meets the same answer; this only reads the shell's spelling of it.
fn parse_profile_change(key: &str, value: Option<&str>) -> anyhow::Result<ProfileChanges> {
    fn patch<T>(
        value: Option<&str>,
        parse: impl FnOnce(&str) -> anyhow::Result<T>,
    ) -> anyhow::Result<Patch<T>> {
        Ok(match value {
            None => Patch::Remove,
            Some(value) => Patch::Set(parse(value)?),
        })
    }
    let integer = |what: &str| {
        patch(value, |value| {
            value
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("{what} must be a whole number, got '{value}'"))
        })
    };
    let boolean = |what: &str| {
        patch(value, |value| match value {
            "true" => Ok(true),
            "false" => Ok(false),
            other => anyhow::bail!("{what} must be true or false, got '{other}'"),
        })
    };
    let mut changes = ProfileChanges::default();
    match key {
        "model" => changes.model = patch(value, |value| Ok(value.to_string()))?,
        "fallbacks" => {
            changes.fallbacks = patch(value, |value| {
                let names = fallback_names(value);
                // The door refuses an empty list too; the remedy is this surface's to name.
                if names.is_empty() {
                    anyhow::bail!(
                        "fallbacks must name at least one profile, comma-separated; pass `--unset` \
                         to remove the setting"
                    );
                }
                Ok(names)
            })?;
        }
        "effort" => changes.effort = patch(value, |value| Ok(value.to_string()))?,
        "context_window" => changes.context_window = integer("context_window")?,
        "max_output_tokens" => changes.max_output_tokens = integer("max_output_tokens")?,
        "max_request_bytes" => changes.max_request_bytes = integer("max_request_bytes")?,
        "vision" => changes.vision = boolean("vision")?,
        "thinking_budget" => changes.thinking_budget = integer("thinking_budget")?,
        "cache_control" => changes.cache_control = boolean("cache_control")?,
        "deferred_tools" => {
            changes.deferred_tools = patch(value, |value| {
                value
                    .parse::<crate::config::DeferredTools>()
                    .map_err(anyhow::Error::msg)
            })?;
        }
        "thinking_display" => {
            changes.thinking_display = patch(value, |value| {
                value
                    .parse::<crate::config::ThinkingDisplay>()
                    .map_err(anyhow::Error::msg)
            })?;
        }
        "thinking" => {
            changes.thinking = patch(value, |value| {
                value
                    .parse::<crate::config::ThinkingMode>()
                    .map_err(anyhow::Error::msg)
            })?;
        }
        other => return Err(unsettable_key_refusal(other)),
    }
    Ok(changes)
}

/// The names a `fallbacks` value on the command line carries: comma-separated, blanks dropped.
pub(crate) fn fallback_names(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}

/// The refusal for a key `profile set` cannot write: one sentence, with the remedy where there
/// is one. The one spelling for the check ahead of parsing and the parser's own fall-through.
fn unsettable_key_refusal(key: &str) -> anyhow::Error {
    match key {
        // A user who typed it did not typo, and "settable: model, effort, ..." alone would read
        // as though meka had forgotten it rather than declined it.
        "account" => anyhow::anyhow!(
            "`account` names the credential every session on the profile bills; add a profile on \
             the other account instead"
        ),
        "backend" | "base_url" | "oauth_token_url" | "client_id" | "device_id" => {
            anyhow::anyhow!("`{key}` is set on the account, under `[accounts.<name>]`")
        }
        _ => anyhow::anyhow!(
            "'{key}' is not a profile setting (settable: {})",
            SETTABLE_PROFILE_KEYS.join(", ")
        ),
    }
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
    Err(unsettable_key_refusal(key))
}

/// `meka profile set <name> <key> <value>`, or `--unset` to drop the key.
fn run_set(name: &str, key: &str, value: Option<&str>, unset: bool) -> anyhow::Result<()> {
    ensure_settable_key(key)?;
    // Clap's `conflicts_with` rules out "both"; this is the other half, which it cannot express.
    let value = match (value, unset) {
        (Some(value), _) => Some(value),
        (None, true) => None,
        (None, false) => anyhow::bail!(
            "`meka profile set {name} {key}` needs a value, or `--unset` to remove the setting"
        ),
    };
    edit_profile(name, &parse_profile_change(key, value)?, false)?;
    match value {
        Some(value) => tracing::info!("set {key} = {value} on profile '{name}'"),
        None => tracing::info!("cleared {key} on profile '{name}'"),
    }
    Ok(())
}

fn run_use(name: &str) -> anyhow::Result<()> {
    edit_profile(name, &ProfileChanges::default(), true)?;
    tracing::info!("default profile set to '{name}'");
    Ok(())
}

/// The listing's columns: the name `--profile` and `/profile` take, the account and the model for
/// information, and the two facts.
const PROFILE_COLUMNS: [crate::text::Column; 6] = [
    crate::text::Column::content("Name"),
    crate::text::Column::capped("Account", crate::text::NAME_WIDTH),
    crate::text::Column::content("Backend"),
    crate::text::Column::remainder("Model"),
    crate::text::Column::capped("Fallbacks", crate::text::NAME_WIDTH),
    crate::text::Column::content("Default"),
];

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
                        if view.fallbacks.is_empty() {
                            "-".to_string()
                        } else {
                            view.fallbacks.join(", ")
                        },
                        if view.active { "yes" } else { "no" }.to_string(),
                    ]
                })
                .collect();
            crate::render::write_stdout(crate::text::format_table(&PROFILE_COLUMNS, &rows))?;
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

/// Every state a listing can reveal and a run then fails on, said on stderr under either format.
fn report_broken_profiles(config_file: &config::ConfigFile) {
    let default = config_file.default_profile.as_deref();
    // A `default_profile` naming nothing renders as a table with `no` in every `Default` cell,
    // which is exactly what "no default set" looks like, and the next `meka` run then fails on
    // a setting the user believes is fine. This listing is where they come to check, so it is
    // where the discrepancy belongs.
    if let Some(default) = default
        && !config_file.profiles.contains_key(default)
    {
        let default = crate::text::sanitize_for_display(default);
        crate::streams::write_stderr_line("");
        crate::streams::write_stderr_line(format!(
            "`default_profile` names '{default}': {}",
            crate::text::unknown_name("profile", &default, config_file.profiles.keys())
        ));
        crate::render::render_hint("point it at one with `meka profile use <name>`");
    }
    // A profile whose account is gone is listed with `-` for its backend, and said aloud: it is the
    // state a hand-deleted `[accounts.<name>]` leaves, and every session on it refuses to run.
    for (name, profile) in &config_file.profiles {
        if !config_file.accounts.contains_key(&profile.account) {
            let name = crate::text::sanitize_for_display(name);
            let account = crate::text::sanitize_for_display(&profile.account);
            crate::streams::write_stderr_line("");
            crate::streams::write_stderr_line(format!(
                "Profile '{name}': {}",
                crate::text::unknown_name("account", &account, config_file.accounts.keys())
            ));
            crate::render::render_hint(&format!("create it with `meka account add {account}`"));
        }
    }
    crate::cli::account::report_unknown_backends(config_file);
}

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
    let effective_window = session_window.unwrap_or(crate::config::DEFAULT_CONTEXT_WINDOW);
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
    if !backend.reads_profile_key("cache_control") && flags.cache_control.take().is_some() {
        dropped.push("--cache-control");
    }
    if !backend.reads_profile_key("deferred_tools") && flags.deferred_tools.take().is_some() {
        dropped.push("--deferred-tools");
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
        fallbacks: flags.fallbacks,
        vision: flags.vision,
        max_output_tokens: flags.max_output_tokens,
        thinking_display: flags.thinking_display,
        max_request_bytes: flags.max_request_bytes,
        cache_control: flags.cache_control,
        deferred_tools: flags.deferred_tools,
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

    #[test]
    fn default_model_for_known_backends() {
        assert_eq!(
            default_model_for(config::Backend::AnthropicMessages),
            "claude-opus-5-5"
        );
        assert_eq!(
            default_model_for(config::Backend::ClaudeSubscription),
            "claude-opus-5-5"
        );
        assert_eq!(
            default_model_for(config::Backend::OpenAiChatCompletions),
            "gpt-6-astra"
        );
        assert_eq!(
            default_model_for(config::Backend::ChatGptSubscription),
            "gpt-6-astra"
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

    /// `--cache-control` is dropped on every backend but `anthropic-messages`: the OpenAI wires
    /// have no breakpoint to withhold, and `claude-subscription` pins its own to the captured
    /// wire.
    #[test]
    fn a_cache_control_flag_aimed_at_a_backend_that_never_reads_it_is_dropped() {
        let flags = || ProfileTuning {
            cache_control: Some(false),
            thinking: Some(config::ThinkingMode::Off),
            context_window: Some(1_024),
            effort: Some("low".to_string()),
            ..Default::default()
        };
        for (backend, kept) in [
            (config::Backend::AnthropicMessages, Some(false)),
            (config::Backend::ClaudeSubscription, None),
            (config::Backend::OpenAiChatCompletions, None),
            (config::Backend::OpenAiResponses, None),
            (config::Backend::ChatGptSubscription, None),
        ] {
            // Every prompted setting is pinned, so this returns before any prompt: no stdin.
            let tuning = resolve_tuning(flags(), backend, "work", None, None).expect("resolve");
            assert_eq!(tuning.cache_control, kept, "{backend}");
        }
    }

    /// Chat Completions has no declaration to make, so the flag is dropped there and kept on the
    /// four wires that have one.
    #[test]
    fn a_deferred_tools_flag_is_dropped_only_for_chat_completions() {
        let flags = || ProfileTuning {
            deferred_tools: Some(config::DeferredTools::Native),
            thinking: Some(config::ThinkingMode::Off),
            context_window: Some(1_024),
            effort: Some("low".to_string()),
            ..Default::default()
        };
        for backend in config::Backend::ALL {
            let tuning = resolve_tuning(flags(), backend, "work", None, None).expect("resolve");
            let kept = (backend != config::Backend::OpenAiChatCompletions)
                .then_some(config::DeferredTools::Native);
            assert_eq!(tuning.deferred_tools, kept, "{backend}");
        }
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

    /// Every settable key parses its own value type, and refuses what it cannot mean.
    #[test]
    fn each_profile_key_parses_the_way_its_add_flag_does() {
        let parsed = |key: &str, value: &str| parse_profile_change(key, Some(value)).expect(key);
        assert_eq!(
            parsed("model", "claude-opus-5-5").model,
            Patch::Set("claude-opus-5-5".to_string())
        );
        assert_eq!(
            parsed("context_window", "200000").context_window,
            Patch::Set(200_000)
        );
        assert!(parse_profile_change("context_window", Some("lots")).is_err());
        assert_eq!(parsed("vision", "false").vision, Patch::Set(false));
        assert!(parse_profile_change("vision", Some("no")).is_err());
        assert_eq!(
            parsed("thinking", "budgeted").thinking,
            Patch::Set(crate::config::ThinkingMode::Budgeted)
        );
        assert!(parse_profile_change("thinking", Some("sideways")).is_err());
        assert_eq!(
            parsed("thinking_budget", "2048").thinking_budget,
            Patch::Set(2048)
        );
        assert_eq!(
            parse_profile_change("effort", None).expect("unset").effort,
            Patch::Remove
        );

        // Every key in the advertised list has an arm. A key listed but unhandled would fall to the
        // catch-all and be refused as unknown, which reads as meka having forgotten its own field.
        for key in SETTABLE_PROFILE_KEYS {
            let sample = match *key {
                "context_window" | "max_output_tokens" | "thinking_budget"
                | "max_request_bytes" => "1000",
                "vision" | "cache_control" => "true",
                "thinking_display" => "updates",
                "thinking" => "adaptive",
                "deferred_tools" => "native",
                "fallbacks" => "backup",
                _ => "value",
            };
            assert!(
                parse_profile_change(key, Some(sample)).is_ok(),
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
        let error = parse_profile_change("account", Some("oai"))
            .expect_err("the account is not settable in place");
        let message = error.to_string();
        assert!(
            message.contains("credential") && message.contains("add a profile"),
            "the refusal must say why and where to go: {message}"
        );

        let base_url = parse_profile_change("base_url", Some("https://x.invalid"))
            .expect_err("an account setting is not a profile setting");
        assert!(
            base_url.to_string().contains("[accounts.<name>]"),
            "{}",
            base_url.to_string()
        );
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

    /// A `fallbacks` value on the command line is comma-separated, blanks dropped, and an empty
    /// one is refused rather than written as `[]`.
    #[test]
    fn a_fallbacks_value_is_a_comma_separated_list() {
        let parsed = parse_profile_change("fallbacks", Some(" backup, last ,")).expect("list");
        assert_eq!(
            parsed.fallbacks,
            Patch::Set(vec!["backup".to_string(), "last".to_string()])
        );
        let error = parse_profile_change("fallbacks", Some(" , ")).expect_err("empty");
        assert!(error.to_string().contains("--unset"), "{error}");
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
