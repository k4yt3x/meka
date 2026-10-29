//! The profiles in `[profiles.<name>]`, each naming an account and the model and every
//! model-tied setting meka asks that account's endpoint for: the doors `meka profile` and
//! `/v1/profiles` write through.
//!
//! A profile holds no secret and runs no login; that is the account's business, in `account.rs`,
//! whose config-editing helpers this module shares.

use super::{
    Patch, PatchState,
    account::{
        ensure_section_table, open_document, rename_table_entry, reparse_after_edit, repoint_name,
        table_names,
    },
};
use crate::config;

/// The settings `profile add` can write beyond the account and model it always asks for. Each is
/// `None` when the flag was absent and the user declined (or skipped) the advanced prompt, which
/// leaves the key out of the profile entirely so the documented default applies.
///
/// Only `context_window`, `effort` and `thinking` are ever *prompted* for, plus the budget when
/// the answer is `budgeted`. The rest are flag-only, so a profile of any shape can be created in
/// one non-interactive command without making the interactive path a nine-step wizard for settings
/// most users never state. `resolve_tuning` says the same thing from the other side: its
/// short-circuit tests those three and no others, because a rare flag must not silently skip the
/// prompt for the common ones.
///
/// Ordered as [`config::ProfileConfig`] is, minus the two `profile add` takes as its own arguments
/// (`account`, `model`).
#[derive(Default)]
pub(crate) struct ProfileTuning {
    pub(crate) fallbacks: Vec<String>,
    pub(crate) context_window: Option<u64>,
    pub(crate) max_output_tokens: Option<u64>,
    pub(crate) max_request_bytes: Option<u64>,
    pub(crate) effort: Option<String>,
    pub(crate) vision: Option<bool>,
    pub(crate) deferred_tools: Option<crate::config::DeferredTools>,
    pub(crate) thinking: Option<config::ThinkingMode>,
    pub(crate) thinking_budget: Option<u64>,
    pub(crate) cache_control: Option<bool>,
    pub(crate) thinking_display: Option<crate::config::ThinkingDisplay>,
}

impl ProfileTuning {
    /// The keys this states a value for, which the account's backend must read.
    fn stated_keys(&self) -> Vec<&'static str> {
        [
            ("fallbacks", !self.fallbacks.is_empty()),
            ("context_window", self.context_window.is_some()),
            ("max_output_tokens", self.max_output_tokens.is_some()),
            ("max_request_bytes", self.max_request_bytes.is_some()),
            ("effort", self.effort.is_some()),
            ("vision", self.vision.is_some()),
            ("deferred_tools", self.deferred_tools.is_some()),
            ("thinking", self.thinking.is_some()),
            ("thinking_budget", self.thinking_budget.is_some()),
            ("cache_control", self.cache_control.is_some()),
            ("thinking_display", self.thinking_display.is_some()),
        ]
        .into_iter()
        .filter_map(|(key, stated)| stated.then_some(key))
        .collect()
    }
}

/// Remove `[profiles.<name>]`, or a dangling `default_profile` naming it: `meka profile remove`
/// and `DELETE /v1/profiles/{name}`. The default is not removed while two or more profiles would
/// remain, since nothing would pick one; with one left, the key is dropped and the sole profile
/// is the default by the selection rule.
pub(crate) async fn remove_profile(
    name: &str,
    store: &crate::store::Store,
) -> crate::error::Result<()> {
    // `open_document` rather than the parsed config on purpose: `remove` must still run on a
    // config.toml that meka can't deserialize, since it is one of the ways such a file gets
    // repaired. The guard is dropped before the `await` below, for the reason `account remove`
    // gives about `ConfigFileLock`'s thread-local depth counter.
    let (was_default, removed) = {
        let (_lock, path, mut document) = open_document()?;
        let was_default = document
            .get("default_profile")
            .and_then(|item| item.as_str())
            == Some(name);
        // A profile another's `fallbacks` names could not be removed without leaving that list
        // naming nothing, which the next run refuses; refused here, where the name is in front
        // of the user, as `account remove` refuses an account a profile names.
        let naming = profiles_naming_as_fallback(&document, name);
        if !naming.is_empty() {
            return Err(super::refusal(format!(
                "profile(s) {} name '{name}' in `fallbacks`; remove it from them first",
                naming.join(", ")
            )));
        }
        // A dangling `default_profile`, one naming no profile, is cleanup rather than a choice
        // lost, so only a default that exists is held back.
        let profiles = table_names(&document, "profiles");
        let exists = profiles.iter().any(|profile| profile == name);
        let others: Vec<String> = profiles
            .into_iter()
            .filter(|profile| profile != name)
            .collect();
        if was_default && exists && others.len() > 1 {
            return Err(crate::error::MekaError::ProfileDefaultStillNeeded {
                name: name.to_string(),
                remaining: others,
            });
        }
        // Written even with no profile to remove: `default_profile` can still point at the name,
        // and dropping that dangling pointer is exactly the cleanup this case is for.
        let removed = remove_profile_document(&mut document, name);
        if !removed && !was_default {
            // The document's own table names rather than `require_profile`: this command edits
            // through `toml_edit` so it works on a file the typed config no longer parses.
            return Err(super::refusal(crate::text::unknown_name(
                "profile",
                name,
                table_names(&document, "profiles"),
            )));
        }
        crate::fs::write_file_atomic(&path, &document.to_string())?;
        (was_default, removed)
    };
    if was_default {
        tracing::info!("'{name}' was the default profile; `default_profile` is unset");
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

/// Rename `[profiles.<name>]`, moving every session recorded on it and answering their ids:
/// `meka profile rename` and `PATCH /v1/profiles/{name}` with a `name`.
pub(crate) async fn rename_profile(
    name: &str,
    new_name: &str,
    store: &crate::store::Store,
) -> crate::error::Result<Vec<uuid::Uuid>> {
    super::refuse_an_unusable_name("profile", new_name)?;
    // Probed under a short guard, dropped before the `await` below, for the reason `account
    // remove` gives about `ConfigFileLock`; the write asks again under its own.
    {
        let (_lock, _path, document) = open_document()?;
        refuse_an_unrenameable_profile(&document, name, new_name)?;
    }

    // What runs on the new name already, which `profile remove` leaves behind, would run on the
    // renamed profile instead: refused by name, since only an explicit act may move a session
    // between profiles. A root session first, which a move takes off the name, then a sub-agent
    // pinned to it, which only removing its session does.
    let run_on = store.sessions_run_on_profile_name(new_name).await?;
    if run_on.roots > 0 {
        return Err(crate::error::MekaError::ProfileRecordedBySessions {
            profile: new_name.to_string(),
            sessions: run_on.roots,
        });
    }
    if !run_on.pinned_under.is_empty() {
        return Err(crate::error::MekaError::ProfilePinnedBySubagents {
            profile: new_name.to_string(),
            under: run_on.pinned_under,
        });
    }

    // The rows move first, and move back if the config write then fails, so a session never names
    // a profile the file does not have for longer than this function runs. The other order has no
    // undo: a config already renamed makes the second half unrepeatable. The undo puts back what
    // moved and nothing else, so a sub-agent row that followed its parent onto the new name before
    // is not carried onto the old one.
    let moved = store.rename_profile(name, new_name).await?;
    let sessions = moved.sessions.clone();
    if let Err(error) = rename_profile_under_lock(name, new_name) {
        if let Err(undo) = store.undo_profile_move(moved, name).await {
            tracing::warn!(
                "failed to move {count} session(s) back to '{name}': {undo}; they now run on \
                 '{new_name}'",
                count = sessions.len()
            );
        }
        return Err(error);
    }
    tracing::info!(
        "renamed profile '{name}' to '{new_name}'; {count} session(s) follow",
        count = sessions.len()
    );
    Ok(sessions)
}

/// Refuse a rename that names no profile or takes a name in use.
///
/// Asked twice by `rename_profile`, of the file as it stands each time, because a `profile add`
/// taking the new name can land between the probe and the write.
fn refuse_an_unrenameable_profile(
    document: &toml_edit::DocumentMut,
    name: &str,
    new_name: &str,
) -> crate::error::Result<()> {
    let profiles = table_names(document, "profiles");
    if !profiles.iter().any(|profile| profile == name) {
        return Err(super::refusal(crate::text::unknown_name(
            "profile", name, &profiles,
        )));
    }
    if profiles.iter().any(|profile| profile == new_name) {
        return Err(super::refusal(format!(
            "a profile named '{new_name}' already exists"
        )));
    }
    Ok(())
}

/// Rename `[profiles.<name>]` and repoint `default_profile` if it named it, as one critical
/// section under the config lock. No `await` inside, for the reason `account remove` gives about
/// `ConfigFileLock`.
fn rename_profile_under_lock(name: &str, new_name: &str) -> crate::error::Result<()> {
    let (_lock, path, mut document) = open_document()?;
    refuse_an_unrenameable_profile(&document, name, new_name)?;
    rename_table_entry(&mut document, "profiles", name, new_name)?;
    repoint_fallbacks(&mut document, name, new_name);
    if document
        .get("default_profile")
        .and_then(|item| item.as_str())
        == Some(name)
        && let Some(item) = document.get_mut("default_profile")
    {
        repoint_name(item, new_name);
    }
    crate::fs::write_file_atomic(&path, &document.to_string())?;
    Ok(())
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
/// `resolve_tuning` drops such a flag on `profile add` with a warning, on the grounds that
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
/// `edit_profile` skips it for a removal, which can only be taking such a key out.
fn refuse_an_inert_key(
    document: &toml_edit::DocumentMut,
    name: &str,
    key: &str,
) -> crate::error::Result<()> {
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
    Err(super::refusal(format!(
        "profile '{name}' bills '{account}', a '{backend}' account, which never sends `{key}`; \
         nothing was written"
    )))
}

/// What an edit of `[profiles.<name>]` changes: each key `meka profile set` writes, kept, removed
/// or set. `account` is not here, because moving it moves every session on the profile onto
/// another credential, and `device_id` is not, because meka resolves it. The shell's strings and a
/// request's JSON both arrive as this, so the door reads a value once, in one way.
#[derive(Debug, Default)]
pub(crate) struct ProfileChanges {
    pub(crate) model: Patch<String>,
    pub(crate) fallbacks: Patch<Vec<String>>,
    pub(crate) context_window: Patch<u64>,
    pub(crate) max_output_tokens: Patch<u64>,
    pub(crate) max_request_bytes: Patch<u64>,
    pub(crate) effort: Patch<String>,
    pub(crate) vision: Patch<bool>,
    pub(crate) deferred_tools: Patch<crate::config::DeferredTools>,
    pub(crate) thinking: Patch<crate::config::ThinkingMode>,
    pub(crate) thinking_budget: Patch<u64>,
    pub(crate) cache_control: Patch<bool>,
    pub(crate) thinking_display: Patch<crate::config::ThinkingDisplay>,
}

impl ProfileChanges {
    /// Every key with what happens to it, in the order the file carries the keys.
    fn states(&self) -> [(&'static str, PatchState); 12] {
        [
            ("model", self.model.state()),
            ("fallbacks", self.fallbacks.state()),
            ("context_window", self.context_window.state()),
            ("max_output_tokens", self.max_output_tokens.state()),
            ("max_request_bytes", self.max_request_bytes.state()),
            ("effort", self.effort.state()),
            ("vision", self.vision.state()),
            ("deferred_tools", self.deferred_tools.state()),
            ("thinking", self.thinking.state()),
            ("thinking_budget", self.thinking_budget.state()),
            ("cache_control", self.cache_control.state()),
            ("thinking_display", self.thinking_display.state()),
        ]
    }

    /// The keys changed, set or removed.
    pub(crate) fn keys(&self) -> Vec<&'static str> {
        self.states()
            .into_iter()
            .filter(|(_, state)| *state != PatchState::Keep)
            .map(|(key, _)| key)
            .collect()
    }

    /// Whether nothing is changed.
    pub(crate) fn is_empty(&self) -> bool {
        self.keys().is_empty()
    }

    /// The changes as the document takes them: each changed key with the TOML value it is set to,
    /// or `None` for one removed. A value no profile may carry is refused here, ahead of the lock:
    /// a blank `model`, which is the model no provider has; a `fallbacks` naming nothing; an
    /// integer past the largest TOML can hold. A blank string on a string key is a removal.
    fn to_document(
        &self,
        name: &str,
    ) -> crate::error::Result<Vec<(&'static str, Option<toml_edit::Value>)>> {
        let mut changes: Vec<(&'static str, Option<toml_edit::Value>)> = Vec::new();
        let integer = |key: &'static str, patch: &Patch<u64>| -> crate::error::Result<_> {
            Ok(match patch {
                Patch::Keep => None,
                Patch::Remove => Some((key, None)),
                Patch::Set(value) => Some((
                    key,
                    Some(toml_edit::Value::from(toml_integer(key, *value)?)),
                )),
            })
        };
        let boolean = |key: &'static str, patch: &Patch<bool>| match patch {
            Patch::Keep => None,
            Patch::Remove => Some((key, None)),
            Patch::Set(value) => Some((key, Some(toml_edit::Value::from(*value)))),
        };
        let text = |key: &'static str, patch: &Patch<String>| match patch.clone().blank_removes() {
            Patch::Keep => None,
            Patch::Remove => Some((key, None)),
            Patch::Set(value) => Some((key, Some(toml_edit::Value::from(value)))),
        };
        let named = |key: &'static str, state: PatchState, name: Option<&'static str>| match state {
            PatchState::Keep => None,
            PatchState::Remove => Some((key, None)),
            PatchState::Set => name.map(|name| (key, Some(toml_edit::Value::from(name)))),
        };
        match &self.model {
            Patch::Keep => {}
            Patch::Remove => changes.push(("model", None)),
            // The predicate the write door applies, asked of the value alone so `""` is refused
            // as the model no provider has rather than written and then sent.
            Patch::Set(model) => {
                config::require_model(name, Some(model))?;
                changes.push(("model", Some(toml_edit::Value::from(model.as_str()))));
            }
        }
        match &self.fallbacks {
            Patch::Keep => {}
            Patch::Remove => changes.push(("fallbacks", None)),
            Patch::Set(names) => {
                let names = fallback_names(names)?;
                changes.push((
                    "fallbacks",
                    Some(toml_edit::Value::Array(fallbacks_array(&names))),
                ));
            }
        }
        changes.extend(integer("context_window", &self.context_window)?);
        changes.extend(integer("max_output_tokens", &self.max_output_tokens)?);
        changes.extend(integer("max_request_bytes", &self.max_request_bytes)?);
        changes.extend(text("effort", &self.effort));
        changes.extend(boolean("vision", &self.vision));
        changes.extend(named(
            "deferred_tools",
            self.deferred_tools.state(),
            match &self.deferred_tools {
                Patch::Set(setting) => Some(setting.name()),
                _ => None,
            },
        ));
        changes.extend(named(
            "thinking",
            self.thinking.state(),
            match &self.thinking {
                Patch::Set(mode) => Some(mode.name()),
                _ => None,
            },
        ));
        changes.extend(integer("thinking_budget", &self.thinking_budget)?);
        changes.extend(boolean("cache_control", &self.cache_control));
        changes.extend(named(
            "thinking_display",
            self.thinking_display.state(),
            match &self.thinking_display {
                Patch::Set(display) => Some(display.name()),
                _ => None,
            },
        ));
        Ok(changes)
    }
}

/// Write `changes` into `[profiles.<name>]` and point `default_profile` at the profile when
/// `make_default`: one critical section, one validation of the result, one write.
///
/// The one door for every edit of a profile that exists. `meka profile set` sends one change,
/// `meka profile use` none, and `PATCH /v1/profiles/{name}` any number with or without the
/// default, so the three cannot disagree about what a profile may be left as. A key the profile's
/// backend never sends is refused on a write and allowed on a removal, for the reason
/// [`refuse_an_inert_key`] gives, and the result is refused whole when the profile could not run
/// on it or a fallback chain breaks, for the reason [`refuse_a_profile_that_cannot_run`] gives.
pub(crate) fn edit_profile(
    name: &str,
    changes: &ProfileChanges,
    make_default: bool,
) -> crate::error::Result<()> {
    let changes = changes.to_document(name)?;
    // `_lock` is held to the end of the function, so the read, the validation and the write are one
    // critical section.
    let (_lock, path, mut document) = open_document()?;
    let before = document.to_string();
    // The document's own table names, for the reason `remove_profile` gives.
    if !table_names(&document, "profiles")
        .iter()
        .any(|profile| profile == name)
    {
        return Err(super::refusal(crate::text::unknown_name(
            "profile",
            name,
            table_names(&document, "profiles"),
        )));
    }
    for (key, value) in &changes {
        set_profile_field(&mut document, name, key, value.clone());
        // Only a write is refused. A removal of such a key moves the profile *towards* the state
        // this guards, and a hand-edited file is the one place an inert key can already be
        // sitting, so refusing to remove it would leave no door that could.
        if value.is_some() {
            refuse_an_inert_key(&document, name, key)?;
        }
    }
    // Asked of the document about to be written, not of the one on disk, so no write door can
    // leave behind a profile another would have refused. Checked here rather than at the next run
    // because a config that fails to start is a worse answer than a command that declines, and the
    // value that broke it is in front of the user right now.
    let after = document.to_string();
    refuse_a_profile_that_cannot_run(&before, &after, name)?;
    if make_default {
        // `entry().or_insert()` is what an index assignment does: a present key keeps its decor,
        // where `insert` would reset it.
        *document
            .entry("default_profile")
            .or_insert(toml_edit::Item::None) = toml_edit::value(name);
    }
    crate::fs::write_file_atomic(&path, &document.to_string())?;
    Ok(())
}

/// Narrow a profile's `u64` setting to the `i64` a TOML integer actually is.
///
/// TOML has one integer type and it is signed 64-bit, so a `u64` past `i64::MAX` has no
/// representation at all; `as i64` would wrap it silently into a file every later command refuses,
/// the `profile set` that would repair it included.
pub(crate) fn toml_integer(field: &str, value: u64) -> crate::error::Result<i64> {
    i64::try_from(value).map_err(|_| {
        super::refusal(format!(
            "{} must be at most {} (the largest TOML integer), got {}",
            field,
            i64::MAX,
            value
        ))
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
) -> crate::error::Result<()> {
    let mut profile = toml_edit::Table::new();
    // Written in [`config::ProfileConfig`]'s canonical order, so a profile meka authors reads the
    // way the reference documents it.
    //
    // An unset knob is left out rather than written at its default, so the profile records only
    // what the user actually chose and a later change to a default reaches existing profiles.
    profile.insert("account", toml_edit::value(account));
    profile.insert("model", toml_edit::value(model));
    if !tuning.fallbacks.is_empty() {
        profile.insert(
            "fallbacks",
            toml_edit::value(fallbacks_array(&fallback_names(&tuning.fallbacks)?)),
        );
    }
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
    if let Some(bytes) = tuning.max_request_bytes {
        profile.insert(
            "max_request_bytes",
            toml_edit::value(toml_integer("max_request_bytes", bytes)?),
        );
    }
    if let Some(effort) = tuning.effort.as_deref() {
        profile.insert("effort", toml_edit::value(effort));
    }
    if let Some(vision) = tuning.vision {
        profile.insert("vision", toml_edit::value(vision));
    }
    if let Some(setting) = tuning.deferred_tools {
        profile.insert("deferred_tools", toml_edit::value(setting.name()));
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
    if let Some(cache_control) = tuning.cache_control {
        profile.insert("cache_control", toml_edit::value(cache_control));
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
    // No `default_profile` for the profile written. A sole profile is the default by the selection
    // rule, and `use` is the command that points the key at a profile; written here too, `add` on
    // a config whose key had been deleted by hand silently rebound every new session to the
    // profile just added. `write_profile` names the one that was alone when this one joins it,
    // which rebinds nothing.
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

/// Write `[profiles.<name>]`, and point `default_profile` at it when `make_default`: `meka
/// profile add` once its prompts are answered, and `POST /v1/profiles`.
pub(crate) fn write_profile(
    name: &str,
    account: &str,
    model: &str,
    tuning: &ProfileTuning,
    make_default: bool,
) -> crate::error::Result<()> {
    super::refuse_an_unusable_name("profile", name)?;
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
        return Err(super::refusal(format!(
            "a profile named '{name}' already exists"
        )));
    }
    // The account too, for the same reason: `account remove` re-asks its own side under this lock,
    // and the pair only holds if this side asks as well. Read off the document rather than the
    // config `run_add` parsed, because that parse is what has gone stale.
    if !table_names(&document, "accounts")
        .iter()
        .any(|configured| configured == account)
    {
        return Err(super::refusal(crate::text::unknown_name(
            "account",
            account,
            table_names(&document, "accounts"),
        )));
    }
    let alone = table_names(&document, "profiles");
    let before = document.to_string();
    upsert_profile_document(&mut document, name, account, model, tuning)?;
    for key in tuning.stated_keys() {
        refuse_an_inert_key(&document, name, key)?;
    }
    let after = document.to_string();
    refuse_a_profile_that_cannot_run(&before, &after, name)?;
    if make_default {
        *document
            .entry("default_profile")
            .or_insert(toml_edit::Item::None) = toml_edit::value(name);
    } else if document.get("default_profile").is_none()
        && let [alone] = alone.as_slice()
    {
        // A sole profile is the default by the selection rule, so a second one joining it would
        // leave the installation with none, and nothing in the answer to the add would say so.
        // Naming the one that was alone moves no session: every new one was already on it.
        *document
            .entry("default_profile")
            .or_insert(toml_edit::Item::None) = toml_edit::value(alone);
    }
    crate::fs::write_file_atomic(&path, &document.to_string())?;
    Ok(())
}

/// A `fallbacks` list as a door writes it, whichever door: each name trimmed and a blank one
/// dropped, and a list that names nothing then refused, since a list stated is a list meant.
fn fallback_names(names: &[String]) -> crate::error::Result<Vec<String>> {
    let names: Vec<String> = names
        .iter()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .collect();
    if names.is_empty() {
        return Err(super::refusal(
            "fallbacks must name at least one profile; remove the setting for none",
        ));
    }
    Ok(names)
}

/// The TOML array a `fallbacks` list is written as.
pub(crate) fn fallbacks_array(fallbacks: &[String]) -> toml_edit::Array {
    fallbacks.iter().map(String::as_str).collect()
}

/// The profiles whose `fallbacks` name `name`, which is what keeps `name` from being removed.
fn profiles_naming_as_fallback(document: &toml_edit::DocumentMut, name: &str) -> Vec<String> {
    let Some(profiles) = document
        .get("profiles")
        .and_then(|item| item.as_table_like())
    else {
        return Vec::new();
    };
    profiles
        .iter()
        .filter(|(_, profile)| {
            profile
                .as_table_like()
                .and_then(|profile| profile.get("fallbacks"))
                .and_then(|item| item.as_array())
                .is_some_and(|array| array.iter().any(|value| value.as_str() == Some(name)))
        })
        .map(|(profile, _)| profile.to_string())
        .collect()
}

/// Point every `fallbacks` entry naming `name` at `new_name`, keeping whatever comment sat beside
/// it: a rename moves the name everywhere it is recorded.
fn repoint_fallbacks(document: &mut toml_edit::DocumentMut, name: &str, new_name: &str) {
    let Some(profiles) = document
        .get_mut("profiles")
        .and_then(|item| item.as_table_like_mut())
    else {
        return;
    };
    for (_, profile) in profiles.iter_mut() {
        let Some(array) = profile
            .as_table_like_mut()
            .and_then(|profile| profile.get_mut("fallbacks"))
            .and_then(|item| item.as_array_mut())
        else {
            continue;
        };
        for value in array.iter_mut() {
            if value.as_str() == Some(name) {
                let decor = value.decor().clone();
                *value = toml_edit::Value::from(new_name);
                *value.decor_mut() = decor;
            }
        }
    }
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
pub(crate) fn refuse_a_profile_that_cannot_run(
    before: &str,
    after: &str,
    name: &str,
) -> crate::error::Result<()> {
    let candidate = reparse_after_edit(before, after)?;
    if let Some(profile) = candidate.profiles.get(name) {
        // `--unset model` and `--model ""` both arrive here as a profile a run would refuse by
        // name; the predicate the run applies refuses the write first.
        config::refuse_an_unrunnable_profile(
            name,
            profile,
            &candidate.accounts,
            candidate
                .thinking
                .as_ref()
                .and_then(|thinking| thinking.budget),
        )?;
    }
    // The chains, over the whole file rather than `name` alone: an edit to one profile's window or
    // `vision` changes the answer for every chain that names it, which no per-profile check sees.
    // A chain the file already broke by hand is not blamed on an edit that leaves it as it was,
    // since the run refuses either way and the repair has to come through this door.
    if let Err(reason) = fallback_chains_of(&candidate)
        && toml::from_str::<config::ConfigFile>(before)
            .is_ok_and(|previous| fallback_chains_of(&previous).is_ok())
    {
        return Err(super::refusal(reason));
    }
    Ok(())
}

/// The startup rule for every `fallbacks` chain, asked of a parsed file.
fn fallback_chains_of(config_file: &config::ConfigFile) -> Result<(), String> {
    config::validate_fallback_chains(
        &config_file.accounts,
        &config_file.profiles,
        config_file
            .session
            .as_ref()
            .and_then(|session| session.context_window),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The door reads a change the same way whichever surface sent it: a blank string on a string
    /// key is a removal, a blank `model` is the model no provider has, a `fallbacks` naming nothing
    /// is refused, and the keys come out in the order the file carries them.
    #[test]
    fn the_edit_door_reads_each_change_the_same_way_for_every_surface() {
        let changes = ProfileChanges {
            effort: Patch::Set("  ".to_string()),
            context_window: Patch::Set(1_000_000),
            vision: Patch::Remove,
            ..Default::default()
        };
        assert_eq!(changes.keys(), ["context_window", "effort", "vision"]);
        let described: Vec<(&str, Option<String>)> = changes
            .to_document("work")
            .expect("document")
            .iter()
            .map(|(key, value)| (*key, value.as_ref().map(|value| value.to_string())))
            .collect();
        assert_eq!(described, vec![
            ("context_window", Some("1000000".to_string())),
            ("effort", None),
            ("vision", None),
        ]);
        let refused = ProfileChanges {
            fallbacks: Patch::Set(vec![" ".to_string()]),
            ..Default::default()
        };
        assert!(refused.to_document("work").is_err());
        assert!(ProfileChanges::default().is_empty());
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

        // And through the door every surface uses, since that is where a user meets it.
        let error = ProfileChanges {
            context_window: Patch::Set(u64::MAX),
            ..Default::default()
        }
        .to_document("work")
        .expect_err("the door refuses it too");
        assert!(
            error.to_string().contains("context_window"),
            "{}",
            error.to_string()
        );
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
        // And as a change, before the lock is taken.
        for blank in ["", "   "] {
            let changes = ProfileChanges {
                model: Patch::Set(blank.to_string()),
                ..Default::default()
            };
            assert!(changes.to_document("work").is_err(), "{blank:?}");
        }
    }

    const TWO_ACCOUNTS: &str = "[accounts.work]\nbackend = \"anthropic-messages\"\n\n\
                                [accounts.oai]\nbackend = \"openai-responses\"\n\n";

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
        let result = remove_profile("work", &manager).await;
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

    /// A rename moves the name everywhere it is recorded: the table keeps its place and comment,
    /// `default_profile` follows, and every session on the profile moves, a pinned sub-agent's
    /// spawn terms included, while a session on another profile stays.
    #[tokio::test]
    async fn rename_moves_the_default_and_every_session_and_keeps_the_table_in_place() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            format!(
                "default_profile = \"work\"\n\n{TWO_ACCOUNTS}# Daily driver.\n[profiles.work]\n\
                 account = \"work\"\nmodel = \"m\"\n\n[profiles.personal]\naccount = \
                 \"work\"\nmodel = \"m\"\n"
            ),
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await;
        let root = store
            .create_session(None, "work".to_string())
            .await
            .expect("root");
        let (pinned, _lock) = store
            .create_child_session_for_test(
                root,
                None,
                Vec::new(),
                Some(r#"{"permission":"read","tools":[],"profile":"work"}"#.to_string()),
                "read".to_string(),
                "work".to_string(),
            )
            .await
            .expect("pinned child");
        let (following, _lock) = store
            .create_child_session_for_test(
                root,
                None,
                Vec::new(),
                Some(r#"{"permission":"read","tools":[]}"#.to_string()),
                "read".to_string(),
                "work".to_string(),
            )
            .await
            .expect("following child");
        let elsewhere = store
            .create_session(None, "personal".to_string())
            .await
            .expect("other");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = rename_profile("work", "main", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        result.expect("the rename succeeds");

        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(
            contents.contains("default_profile = \"main\""),
            "{contents}"
        );
        assert!(
            contents.contains("# Daily driver.\n[profiles.main]\n"),
            "the table keeps its comment and its place: {contents}"
        );
        assert!(!contents.contains("[profiles.work]"), "{contents}");
        assert!(
            contents.find("[profiles.main]") < contents.find("[profiles.personal]"),
            "the tables keep their order: {contents}"
        );
        for id in [root, pinned, following] {
            assert_eq!(
                store.recorded_profile(id).await.expect("read").as_deref(),
                Some("main"),
                "every row on the profile moves"
            );
        }
        assert_eq!(
            store
                .recorded_profile(elsewhere)
                .await
                .expect("read")
                .as_deref(),
            Some("personal"),
            "a session on another profile stays"
        );
        assert!(
            store
                .load_subagent_spec(pinned)
                .await
                .expect("read")
                .is_some_and(|spec| spec.contains("\"profile\":\"main\"")),
            "the pinned spawn terms follow"
        );
        assert!(
            store
                .load_subagent_spec(following)
                .await
                .expect("read")
                .is_some_and(|spec| !spec.contains("profile")),
            "unpinned spawn terms gain no pin"
        );
    }

    /// `default_profile` follows the renamed profile and no other: one naming a sibling is left
    /// exactly as it was.
    #[tokio::test]
    async fn rename_leaves_a_default_that_names_another_profile_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            format!(
                "default_profile = \"personal\" # the usual\n\n{TWO_ACCOUNTS}[profiles.work]\n\
                 account = \"work\"\nmodel = \"m\"\n\n[profiles.personal]\naccount = \
                 \"work\"\nmodel = \"m\"\n"
            ),
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await;

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = rename_profile("work", "main", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        result.expect("the rename succeeds");

        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(
            contents.contains("default_profile = \"personal\" # the usual"),
            "{contents}"
        );
        assert!(contents.contains("[profiles.main]"), "{contents}");
    }

    /// Rows already recording the new name are what `profile remove` leaves behind. A rename onto
    /// that name would adopt them, and its undo would carry them onto the old name, so it is
    /// refused before any row moves.
    #[tokio::test]
    async fn rename_refuses_a_name_that_sessions_already_record() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            format!("{TWO_ACCOUNTS}[profiles.work]\naccount = \"work\"\nmodel = \"m\"\n"),
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await;
        let on_work = store
            .create_session(None, "work".to_string())
            .await
            .expect("on work");
        let orphan = store
            .create_session(None, "main".to_string())
            .await
            .expect("left behind by a removal");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = rename_profile("work", "main", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(_) => panic!("a rename onto a recorded name must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("1 session(s) already record the profile 'main'"),
            "{error}"
        );
        for (id, profile) in [(on_work, "work"), (orphan, "main")] {
            assert_eq!(
                store.recorded_profile(id).await.expect("read").as_deref(),
                Some(profile),
                "no row moved"
            );
        }
    }

    /// Refused by name before anything moves: a profile that does not exist, or a name in use.
    #[tokio::test]
    async fn rename_refuses_an_unknown_profile_and_a_taken_name_without_moving_a_row() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            format!(
                "{TWO_ACCOUNTS}[profiles.work]\naccount = \"work\"\nmodel = \
                 \"m\"\n\n[profiles.personal]\naccount = \"work\"\nmodel = \"m\"\n"
            ),
        )
        .expect("write config");
        let store = crate::store::Store::for_test().await;
        let root = store
            .create_session(None, "work".to_string())
            .await
            .expect("root");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let unknown = rename_profile("nope", "fresh", &store).await;
        let taken = rename_profile("work", "personal", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let message = |result: crate::error::Result<Vec<uuid::Uuid>>| match result {
            Ok(_) => panic!("the rename must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(message(unknown).contains("no profile named 'nope'"));
        assert!(message(taken).contains("a profile named 'personal' already exists"));
        assert_eq!(
            store.recorded_profile(root).await.expect("read").as_deref(),
            Some("work"),
            "a refused rename moves no row"
        );
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(contents.contains("[profiles.work]"), "{contents}");
    }

    /// The rows move first and come back when the config write fails, so no session names a
    /// profile the file does not have by the time the error is read.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_config_write_that_fails_moves_the_sessions_back() {
        use std::os::unix::fs::PermissionsExt as _;

        // Root writes through a read-only directory, so there is nothing to observe there.
        // SAFETY: `geteuid` reads a process attribute and has no preconditions.
        if unsafe { libc::geteuid() } == 0 {
            return;
        }
        // A link into a read-only directory, for the reason the account test gives: the atomic
        // write fails beside the link's target, while the config directory stays writable.
        let dir = tempfile::tempdir().expect("tempdir");
        let managed = dir.path().join("managed");
        std::fs::create_dir(&managed).expect("managed dir");
        std::fs::write(
            managed.join("config.toml"),
            format!("{TWO_ACCOUNTS}[profiles.work]\naccount = \"work\"\nmodel = \"m\"\n"),
        )
        .expect("write config");
        std::os::unix::fs::symlink(managed.join("config.toml"), dir.path().join("config.toml"))
            .expect("link");
        let store = crate::store::Store::for_test().await;
        let root = store
            .create_session(None, "work".to_string())
            .await
            .expect("root");
        std::fs::set_permissions(&managed, std::fs::Permissions::from_mode(0o555))
            .expect("read-only dir");

        // SAFETY: as above.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = rename_profile("work", "main", &store).await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        std::fs::set_permissions(&managed, std::fs::Permissions::from_mode(0o755))
            .expect("writable again");

        assert!(result.is_err(), "the config write must fail");
        assert_eq!(
            store.recorded_profile(root).await.expect("read").as_deref(),
            Some("work"),
            "the row is back on the name the file still has"
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
        let result = remove_profile("typo", &manager).await;
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
        let refused = write_profile("daily", "gone", "m", &ProfileTuning::default(), false);
        let written = write_profile("daily", "work", "m", &ProfileTuning::default(), false);
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

    /// A sole profile is the default by the selection rule, so the second profile added names the
    /// first as `default_profile`, and new sessions stay where they were; a third changes nothing,
    /// and `--default` on the second wins.
    #[test]
    fn a_second_profile_keeps_the_first_as_the_default() {
        let add = |dir: &std::path::Path, name: &str, make_default: bool| {
            // SAFETY: as above.
            let _guard = crate::config::CONFIG_DIR_ENV_LOCK.blocking_lock();
            unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir) };
            let written = write_profile(name, "work", "m", &ProfileTuning::default(), make_default);
            unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
            written.expect("a configured account");
            std::fs::read_to_string(dir.join("config.toml")).expect("read back")
        };
        let account = "[accounts.work]\nbackend = \"anthropic-messages\"\n";

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), account).expect("write config");
        let first = add(dir.path(), "work", false);
        assert!(!first.contains("default_profile"), "{first}");
        let second = add(dir.path(), "cheap", false);
        assert!(second.contains("default_profile = \"work\""), "{second}");
        assert!(
            second.find("default_profile") < second.find("[accounts"),
            "the key precedes every table: {second}"
        );
        let third = add(dir.path(), "third", false);
        assert_eq!(third.matches("default_profile").count(), 1, "{third}");
        assert!(third.contains("default_profile = \"work\""), "{third}");

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), account).expect("write config");
        add(dir.path(), "work", false);
        let chosen = add(dir.path(), "cheap", true);
        assert!(chosen.contains("default_profile = \"cheap\""), "{chosen}");
        assert_eq!(chosen.matches("default_profile").count(), 1, "{chosen}");
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
        let refused = edit_profile("ghost", &ProfileChanges::default(), true);
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = refused
            .expect_err("a name that is not a profile")
            .to_string();
        assert!(error.contains("no profile named 'ghost'"), "{error}");
        let contents = std::fs::read_to_string(dir.path().join("config.toml")).expect("read back");
        assert!(!contents.contains("default_profile"), "{contents}");
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
model = "claude-opus-5-5"

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
        // The breakpoint is `anthropic-messages`'s alone: the OpenAI wires have no such field.
        refuse_an_inert_key(&document, "work", "cache_control")
            .expect("a Messages profile may withhold the breakpoint");
        let message = refuse_an_inert_key(&document, "oai", "cache_control")
            .expect_err("a Responses profile has no breakpoint to withhold")
            .to_string();
        assert!(
            message.contains("cache_control") && message.contains("openai-responses"),
            "the refusal names the key and the backend: {message}"
        );
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
            "cache_control",
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
                cache_control: Some(false),
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
        assert_eq!(profile.cache_control, Some(false));
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

    /// `fallbacks` is written as an array in the canonical place, after `model`, and reads back
    /// as the list it was given.
    #[test]
    fn upsert_profile_document_writes_fallbacks_after_model() {
        let mut document = toml_edit::DocumentMut::new();
        upsert_profile_document(&mut document, "main", "a", "m", &ProfileTuning {
            fallbacks: vec!["backup".to_string(), "last".to_string()],
            context_window: Some(1_024),
            ..Default::default()
        })
        .expect("upsert");
        let rendered = document.to_string();
        let model_at = rendered.find("model").expect("model");
        let fallbacks_at = rendered.find("fallbacks").expect("fallbacks");
        let window_at = rendered.find("context_window").expect("window");
        assert!(
            model_at < fallbacks_at && fallbacks_at < window_at,
            "{rendered}"
        );
        let config: config::ConfigFile = toml::from_str(&rendered).expect("re-parse config");
        assert_eq!(config.profiles["main"].fallbacks, vec!["backup", "last"]);
    }

    /// The write doors apply the startup rule to the file they are about to write: a list may not
    /// name the profile itself, a profile the file does not have, or one name twice, and a
    /// fallback must carry what its profile holds. A chain the file already broke is not blamed on
    /// an edit that leaves it as it was.
    #[test]
    fn a_bad_chain_is_refused_at_the_write_door() {
        let file = |main_extra: &str, backup_extra: &str| {
            format!(
                "[accounts.a]\nbackend = \"anthropic-messages\"\n\n[profiles.main]\naccount = \
                 \"a\"\nmodel = \"m\"\n{main_extra}\n[profiles.backup]\naccount = \"a\"\nmodel = \
                 \"m\"\n{backup_extra}"
            )
        };
        let before = file("", "");
        refuse_a_profile_that_cannot_run(&before, &file("fallbacks = [\"backup\"]\n", ""), "main")
            .expect("a good chain");
        for (after, word) in [
            (file("fallbacks = [\"main\"]\n", ""), "itself"),
            (file("fallbacks = [\"nowhere\"]\n", ""), "nowhere"),
            (file("fallbacks = [\"backup\", \"backup\"]\n", ""), "twice"),
            (
                file("fallbacks = [\"backup\"]\n", "vision = false\n"),
                "vision",
            ),
            (
                file("fallbacks = [\"backup\"]\n", "context_window = 200000\n"),
                "200000",
            ),
        ] {
            let error = refuse_a_profile_that_cannot_run(&before, &after, "main").expect_err(word);
            assert!(error.to_string().contains(word), "{error}");
        }

        let broken = file("fallbacks = [\"backup\"]\n", "vision = false\n");
        let unrelated = file(
            "fallbacks = [\"backup\"]\neffort = \"high\"\n",
            "vision = false\n",
        );
        refuse_a_profile_that_cannot_run(&broken, &unrelated, "main")
            .expect("a chain broken on disk is not this edit's to answer for");
    }

    /// A profile another's `fallbacks` names is found by the removal door, and a rename moves the
    /// name inside every list, comment and all.
    #[test]
    fn fallback_lists_follow_a_remove_and_a_rename() {
        let mut document: toml_edit::DocumentMut = "[profiles.main]\naccount = \"a\"\n\
             fallbacks = [\"backup\", \"last\"] # in order\n\n[profiles.other]\naccount = \"a\"\n\
             fallbacks = [\"backup\"]\n\n[profiles.backup]\naccount = \"a\"\n"
            .parse()
            .expect("a document");
        assert_eq!(profiles_naming_as_fallback(&document, "backup"), vec![
            "main", "other"
        ]);
        assert!(profiles_naming_as_fallback(&document, "main").is_empty());

        repoint_fallbacks(&mut document, "backup", "reserve");
        let rendered = document.to_string();
        assert!(
            rendered.contains("fallbacks = [\"reserve\", \"last\"] # in order"),
            "{rendered}"
        );
        assert!(rendered.contains("fallbacks = [\"reserve\"]"), "{rendered}");
        assert!(profiles_naming_as_fallback(&document, "backup").is_empty());
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
}
