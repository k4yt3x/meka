//! Accounts and profiles: [`Backend`], the vocabulary an account's `backend` names;
//! [`AccountConfig`] and [`ProfileConfig`] as `config.toml` writes them and [`ProfileSettings`] as
//! a run reads a profile through its account; and which profile a run selects when several are
//! configured.

use serde::Deserialize;

use super::*;

/// The driver an account's `backend` names.
///
/// One vocabulary for every reader: profile resolution, the provider builder, the `account add`
/// prompts and `/status` all match on this rather than on the string, so a backend added here is a
/// compile error at every site that has to know about it rather than a name that falls through a
/// catch-all at runtime.
///
/// An API-key backend is named for the protocol it speaks, because `base_url` decides the endpoint
/// and the same protocol is served by many vendors. A subscription backend is named for the
/// product, because the endpoint is fixed and what the account holds is a billing relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Backend {
    /// Anthropic's Messages API, billed to an API key.
    AnthropicMessages,
    /// OpenAI's Responses API against `chatgpt.com`, billed to a ChatGPT subscription.
    ChatGptSubscription,
    /// Anthropic's Messages API, billed to a Claude subscription.
    ClaudeSubscription,
    /// An OpenAI-compatible Chat Completions endpoint, billed to an API key.
    OpenAiChatCompletions,
    /// An OpenAI-compatible Responses endpoint, billed to an API key.
    OpenAiResponses,
}
/// One named account from `[accounts.<name>]`: where a request goes and who meka is when it
/// arrives. Holds only non-secret settings; the credential (API key or OAuth bundle) is stored in
/// the DB keyed by the account name and is acquired via `meka account add` / `login`.
///
/// **Field order is canonical and load-bearing**, and every surface that lists these keys repeats
/// it: [`ACCOUNT_KEY_ORDER`], `upsert_account_document`, the `account add` flags, and the
/// `config.toml` reference. Widest reach first, ending with the narrowest: `backend` leads and
/// `device_id` closes as the one key meka writes rather than the user.
///
/// The division from [`ProfileConfig`] is the one "Whose fact is it" draws, which is why a new
/// field has an obvious home: would two models on one account state it differently? Then it
/// belongs on the profile. Would two accounts on one endpoint? Here.
#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct AccountConfig {
    /// The name of a [`Backend`]. Kept as written so a typo is reported against the account that
    /// carries it, by [`resolve_profile`], rather than failing the whole file.
    pub(crate) backend: String,
    pub(crate) base_url: Option<String>,
    pub(crate) oauth_token_url: Option<String>,
    /// OAuth client id override (advanced; `claude-subscription` / `chatgpt-subscription`).
    pub(crate) client_id: Option<String>,
    pub(crate) device_id: Option<String>,
}
/// One named profile from `[profiles.<name>]`: which account it bills, and what meka asks that
/// account's endpoint for and how.
///
/// **Field order is canonical and load-bearing**, and every surface that lists these keys repeats
/// it: [`ProfileSettings`], [`resolve_profile`], `ProfileTuning`, `upsert_profile_document`,
/// `SETTABLE_PROFILE_KEYS`, the `profile add` flags, and the `config.toml` reference. `account`
/// leads, then `model`, then every model-tied knob, widest reach first and ending with
/// `redact_thinking` as the only `claude-subscription` setting.
#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProfileConfig {
    /// The name of an [`AccountConfig`]. Kept as written so a profile naming a missing account is
    /// reported against the profile, by [`account_for`], rather than failing the whole file.
    pub(crate) account: String,
    pub(crate) model: Option<String>,
    /// The model's context window (total tokens it can hold), used for the context gauge and
    /// auto-compaction. Falls back to `[session].context_window`, then to
    /// [`crate::provider::DEFAULT_CONTEXT_WINDOW`]. meka never infers this from the model name or
    /// asks the provider for it, so this is where a model smaller than the default gets stated.
    pub(crate) context_window: Option<u64>,
    /// Override the per-request output (completion) token cap. When unset, each backend keeps its
    /// built-in default. On Claude the value must exceed the thinking budget.
    pub(crate) max_output_tokens: Option<u64>,
    /// The reasoning-effort knob for every backend: Claude's `output_config.effort` and OpenAI's
    /// `reasoning.effort`. Passed through verbatim (trimmed and lowercased); when unset the field
    /// is omitted entirely, which is how meka asks for the provider's own default. meka picks no
    /// tier of its own; see [`crate::provider::resolve_effort_level`].
    pub(crate) effort: Option<String>,
    /// Whether this profile's model accepts image input. Defaults to `true`; set `false` to stop
    /// the ACP frontend from advertising / accepting images for a text-only model.
    pub(crate) vision: Option<bool>,
    /// Claude-only: how this profile encodes extended thinking, `adaptive`, `budgeted`, or `off`.
    /// Defaults to `adaptive`. The right value depends on the model *and* on what the endpoint
    /// implements, which meka can't infer, so it is stated rather than guessed; a profile that
    /// later changes `model` is the user's to keep correct.
    pub(crate) thinking: Option<crate::config::ThinkingMode>,
    /// How many tokens `thinking = "budgeted"` asks the model to spend reasoning. Falls back to
    /// `[thinking].budget`, then to [`DEFAULT_THINKING_BUDGET_TOKENS`]. Ignored by the other two
    /// modes, which send no budget at all.
    ///
    /// Per profile because it is a parameter of `thinking`, and `thinking` is per profile: a
    /// process-wide value would have [`validate_max_output_tokens`] refuse a profile over a number
    /// it did not state, and tell the user to lower a global every other profile is also budgeting
    /// against.
    pub(crate) thinking_budget: Option<u64>,
    /// Largest request body, in bytes, before the oldest tool-result images are redacted to fit.
    /// Anthropic's endpoint caps at 32 MiB and the Anthropic backends default to 30 MiB, but an
    /// `anthropic-messages` account reaches whatever `base_url` names, whose cap is its own fact.
    pub(crate) max_request_bytes: Option<u64>,
    /// `claude-subscription` only: when true, meka sends the `redact-thinking-2026-02-12` beta
    /// header so the server returns `redacted_thinking` blocks instead of full thinking summaries
    /// (saves bandwidth, but the redacted payloads can't be replayed back to the server in
    /// multi-turn conversations). Defaults to true.
    pub(crate) redact_thinking: Option<bool>,
}
/// [`AccountConfig`]'s field order, as the key names a `config.toml` account table carries.
pub(crate) const ACCOUNT_KEY_ORDER: &[&str] = &[
    "backend",
    "base_url",
    "oauth_token_url",
    "client_id",
    "device_id",
];
/// [`ProfileConfig`]'s field order, as the key names a `config.toml` profile table carries.
///
/// The list exists because the order is *enforced* rather than merely produced: every writer runs
/// [`sort_profile_keys`] before saving, so a profile meka has touched is in this order however it
/// got there, written whole by `profile add` or edited one key at a time by `profile set`.
pub(crate) const PROFILE_KEY_ORDER: &[&str] = &[
    "account",
    "model",
    "context_window",
    "max_output_tokens",
    "effort",
    "vision",
    "thinking",
    "thinking_budget",
    "max_request_bytes",
    "redact_thinking",
];
/// Put one table's keys into `order`.
///
/// Comments ride along, because `toml_edit` hangs them off the key they belong to: whole-line
/// comments and blank lines above a key live in its `leaf_decor`, and a trailing `# note` lives in
/// the value's decor. Moving the entry moves both, which is what makes sorting safe on a file a
/// user has annotated.
///
/// A key the current build does not model sorts last, alphabetically among its peers, rather than
/// wherever the sort happened to leave it. `deny_unknown_fields` means such a key cannot survive a
/// load, so this only decides what an unreachable case looks like; but "unreachable" and
/// "arbitrary" are different, and the second one produces a diff that changes between runs.
///
/// Takes a [`toml_edit::Item`] rather than a `TableLike`, which is the shape every caller has,
/// because only the concrete types carry a comparator-taking sort and their comparators differ
/// (`Table` compares `Item`s, `InlineTable` compares `Value`s). Both are reachable: a table is
/// ordinarily its own section, but `profiles` written as one inline table is a shape meka has had
/// to handle before.
fn sort_keys(item: &mut toml_edit::Item, order: &[&str]) {
    let rank = |key: &str| -> (usize, String) {
        let position = order
            .iter()
            .position(|known| *known == key)
            .unwrap_or(order.len());
        (position, key.to_string())
    };
    if let Some(table) = item.as_table_mut() {
        table.sort_values_by(|left, _, right, _| rank(left.get()).cmp(&rank(right.get())));
    } else if let Some(inline) = item.as_inline_table_mut() {
        inline.sort_values_by(|left, _, right, _| rank(left.get()).cmp(&rank(right.get())));
    }
}
/// Put one account table's keys into [`ACCOUNT_KEY_ORDER`].
pub(crate) fn sort_account_keys(account: &mut toml_edit::Item) {
    sort_keys(account, ACCOUNT_KEY_ORDER);
}
/// Put one profile table's keys into [`PROFILE_KEY_ORDER`].
pub(crate) fn sort_profile_keys(profile: &mut toml_edit::Item) {
    sort_keys(profile, PROFILE_KEY_ORDER);
}
/// One profile resolved through its account into everything needed to build a provider for it.
///
/// This exists because a provider is a property of a session, not of the process. A session names
/// the profile it runs with, so any given turn may need one that is not the configured default:
/// resolution has to be callable per profile name rather than once at startup.
///
/// `context_window` and `vision` ride along despite not reaching
/// [`crate::provider::ProviderBuilder`]: both are stated per profile, so a session that switches
/// profile has to re-derive them or it keeps gauging its context against the window of a model it
/// is no longer talking to.
#[derive(Debug, Clone)]
pub(crate) struct ProfileSettings {
    /// The account the profile bills, which is the row its credential is stored under.
    pub(crate) account: String,
    pub(crate) backend: Backend,
    pub(crate) base_url: Option<String>,
    pub(crate) oauth_token_url: Option<String>,
    pub(crate) client_id: Option<String>,
    pub(crate) device_id: String,
    pub(crate) model: Option<String>,
    pub(crate) context_window: Option<u64>,
    pub(crate) max_output_tokens: Option<u64>,
    pub(crate) effort: Option<String>,
    pub(crate) vision: bool,
    pub(crate) thinking: crate::config::ThinkingMode,
    /// Resolved with the default already applied, unlike [`Self::context_window`]. No caller needs
    /// to tell "unset" from "16000": the budget is only ever sent as a number, whereas an absent
    /// window is what [`crate::provider::profile_context_window`] reports rather than dividing by
    /// a figure meka has no reason to believe.
    pub(crate) thinking_budget: u64,
    pub(crate) redact_thinking: bool,
    /// See [`ProfileConfig::max_request_bytes`]; `None` leaves the backend's default.
    pub(crate) max_request_bytes: Option<usize>,
}
/// The selected profile's name, read straight off disk without resolving anything else.
///
/// Exists for the one caller that opens the store *before* config is resolved: the `meka account`
/// and `meka profile` subcommands deliberately skip [`ResolvedConfig::resolve`], and opening the
/// store is what carries an older one forward. Without this, which command a user happened to run
/// first would decide what their existing sessions record as their profile.
///
/// *Selection* errors are dropped rather than reported: the caller is on its way to a command that
/// works fine with no profile configured, and neither of the two things this feeds needs the
/// reason. The migration context takes an absent name as "nothing resolved" and leaves those rows
/// alone; `crate::store::export::plan_import` refuses the import rather than writing a session that
/// cannot run. A config that could not be **read** is a different answer and is raised, via
/// [`load_config_file_or_err`], for the reason that function already gives: "empty" and "your
/// profiles are gone" are indistinguishable to a caller with no `validate()` behind it.
///
/// The ledger is the sharpest instance of that. It stamps whatever this returns onto every session
/// that predates meka recording a profile, once and irreversibly, so swallowing a parse error here
/// converted one typo in `config.toml` into every session permanently recorded against no profile,
/// with `default_profile` naming a perfectly good one three lines above the typo. Nothing said so:
/// the store was stamped at head, the step never ran again, and the command exited 0.
pub(crate) fn default_profile_on_disk(
    requested: Option<&str>,
) -> crate::error::Result<Option<String>> {
    let config_file = load_config_file_or_err()?;
    let (source, requested) = match requested {
        Some(name) => (ProfileRequest::Flag, Some(name.to_string())),
        None => (
            ProfileRequest::DefaultProfile,
            config_file.default_profile.clone(),
        ),
    };
    let (active, _error) = select_profile(requested, source, &config_file.profiles);
    Ok(active)
}
/// The account a profile names, or why it cannot be found.
///
/// The one place a profile's `account` is followed, so every reader refuses a dangling one in the
/// same words. Refuses rather than falling back: a profile whose account is gone cannot bill
/// anything, and quietly running it on another account is the failure the split exists to prevent.
pub(crate) fn account_for<'a>(
    profile_name: &str,
    profile: &ProfileConfig,
    accounts: &'a std::collections::BTreeMap<String, AccountConfig>,
) -> std::result::Result<&'a AccountConfig, String> {
    accounts.get(&profile.account).ok_or_else(|| {
        let refusal = crate::text::unknown_name("account", &profile.account, accounts.keys());
        if accounts.is_empty() {
            format!(
                "profile '{profile_name}': {refusal}; create it with `meka account add {}`",
                profile.account
            )
        } else {
            format!("profile '{profile_name}': {refusal}")
        }
    })
}
/// Resolve one named profile through its account, applying the process-level fallbacks for the
/// two settings that have them. Nothing overrides a field inside a profile or an account, so there
/// is nothing else to apply.
///
/// `session_context_window` is `[session].context_window` and `default_thinking_budget` is
/// `[thinking].budget`; a profile's own value takes precedence over either. The two are spelled
/// differently on the way out because they are consumed differently: the call sites in `main.rs`
/// apply [`crate::provider::DEFAULT_CONTEXT_WINDOW`] when the window is still unset, while the
/// budget is defaulted here because every consumer needs a number.
pub(crate) fn resolve_profile(
    profile: &ProfileConfig,
    account: &AccountConfig,
    session_context_window: Option<u64>,
    default_thinking_budget: Option<u64>,
    // Received rather than resolved here, because resolving it is not a pure function: for a
    // `claude-subscription` account that states none, [`resolve_device_id`] mints one and rewrites
    // the user's `config.toml` under the config lock. This runs per request (every
    // `resolved_profile`, every `profile_context_window`, every `/status`) against the snapshot of
    // accounts the registry holds, so a resolver called from in here would never see the value it
    // had just persisted and would mint and write a new identifier every time.
    device_id: String,
) -> std::result::Result<ProfileSettings, String> {
    // The one place an account's `backend` is read as a `Backend`, so every later reader holds the
    // enum.
    let backend = account
        .backend
        .parse::<Backend>()
        .map_err(|error| format!("account '{}': {error}", profile.account))?;
    Ok(ProfileSettings {
        account: profile.account.clone(),
        backend,
        base_url: account.base_url.clone(),
        oauth_token_url: account.oauth_token_url.clone(),
        client_id: account.client_id.clone(),
        device_id,
        model: profile.model.clone(),
        context_window: profile.context_window.or(session_context_window),
        max_output_tokens: profile.max_output_tokens,
        max_request_bytes: profile
            .max_request_bytes
            .and_then(|bytes| usize::try_from(bytes).ok()),
        // Pure passthrough for every backend: whatever the profile sets goes to the provider
        // verbatim (the provider trims and lowercases it). Unset means the field is omitted and
        // the provider applies its own default. An invalid value is the user's to own.
        effort: profile.effort.clone(),
        vision: profile.vision.unwrap_or(true),
        thinking: profile.thinking.unwrap_or_default(),
        thinking_budget: profile
            .thinking_budget
            .or(default_thinking_budget)
            .unwrap_or(DEFAULT_THINKING_BUDGET_TOKENS),
        // Default on to match Claude Code, which sends `redact-thinking` for every capable model.
        // Profiles opt out with `redact_thinking = false` to keep interleaved thinking visible.
        redact_thinking: profile.redact_thinking.unwrap_or(true),
    })
}
/// The `claude-subscription` device identifier for one account, seeding and persisting one when
/// the account states none.
///
/// Empty for every other backend. Deliberately separate from [`resolve_profile`], which callers
/// reach on every request: this one writes `config.toml`, so it belongs where a caller can arrange
/// to ask exactly once. [`crate::provider::ProviderRegistry`] memoises it per account.
pub(crate) fn resolve_device_id(
    backend: Backend,
    account_name: &str,
    configured: Option<&str>,
) -> String {
    device_id::resolve(Some(backend), Some(account_name), configured)
}
/// The profile a name denotes, or [`crate::error::MekaError::ProfileNotConfigured`] naming the
/// ones that exist.
///
/// The one predicate for "is this a configured profile", asked by every door that takes a profile
/// name: `--profile` and `default_profile` selection, a resume repin, `/profile` and its HTTP and
/// ACP twins, `meka profile use`, and the registry resolving the profile a session's row records.
/// Refuses rather than falling back to the default, because a session names the profile it runs
/// on and quietly running it elsewhere bills another account and loses the reasoning replay.
///
/// Unconditional on the name, including the empty one a store carried forward from before this
/// column existed: that row is the migration's to explain, and a branch here would be this reader
/// knowing an older meka wrote the store.
pub(crate) fn require_profile<'a>(
    name: &str,
    profiles: &'a std::collections::BTreeMap<String, ProfileConfig>,
) -> crate::error::Result<&'a ProfileConfig> {
    profiles
        .get(name)
        .ok_or_else(|| crate::error::MekaError::ProfileNotConfigured {
            name: name.to_string(),
            known: profiles.keys().cloned().collect(),
        })
}
/// The model a profile names, refused when it names none or an empty one.
///
/// One predicate for every door that needs a model in hand: startup validation of the default
/// profile, the registry resolving the profile a session names, and both `meka profile` write
/// doors. `""` is what makes this more than an `is_none()`: TOML admits it, the option check does
/// not see it, and the empty string would go to the provider as the model's name.
pub(crate) fn require_model<'a>(
    profile: &str,
    model: Option<&'a str>,
) -> crate::error::Result<&'a str> {
    match model {
        Some(model) if !model.trim().is_empty() => Ok(model),
        _ => Err(crate::error::MekaError::Config(format!(
            "profile '{profile}' names no model; set one with `meka profile set {profile} model \
             <model>`"
        ))),
    }
}
/// Which tier asked for the profile [`select_profile`] could not find.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileRequest {
    /// `--profile <name>` on this run.
    Flag,
    /// `default_profile` in `config.toml`, which is also what `meka profile use` writes.
    DefaultProfile,
}
/// Resolve which profile is active given an explicit request (from `--profile` or
/// `default_profile`) and the configured profiles. Returns `(default_profile, profile_error)`:
/// exactly one is `Some`. A deferred error string (rather than a hard failure) keeps
/// [`ResolvedConfig::resolve`] infallible; `validate()` surfaces it later with guidance.
pub(crate) fn select_profile(
    requested: Option<String>,
    // Where `requested` came from, because the remedy differs. Telling someone who just typed
    // `--profile typo` to "pass `--profile`" is advice they have already taken; telling someone
    // whose `default_profile` names a deleted profile to run `meka profile list` is not enough.
    source: ProfileRequest,
    profiles: &std::collections::BTreeMap<String, ProfileConfig>,
) -> (Option<String>, Option<String>) {
    match requested {
        Some(name) => match require_profile(&name, profiles) {
            Ok(_) => (Some(name), None),
            Err(refusal) => (
                None,
                Some(format!(
                    "{refusal}; {}",
                    match (source, profiles.is_empty()) {
                        (_, true) => "create one with `meka profile add <name>`",
                        (ProfileRequest::Flag, false) => "pass a configured name to `--profile`",
                        (ProfileRequest::DefaultProfile, false) =>
                            "point `default_profile` at one with `meka profile use <name>`",
                    }
                )),
            ),
        },
        None => match profiles.len() {
            0 => (
                None,
                Some(
                    "no profile configured; run `meka account add <name>`, then `meka profile add \
                     <name>`"
                        .to_string(),
                ),
            ),
            1 => (profiles.keys().next().cloned(), None),
            // Names `meka profile use`, which is what actually writes `default_profile`. Saying
            // only "set `default_profile`" sends the reader to hand-edit a file for something a
            // command does, and this state is most often reached by `meka profile remove`
            // dropping a pointer to the profile it deleted.
            _ => (
                None,
                Some(format!(
                    "multiple profiles configured ({}) and no `default_profile`; pick one with \
                     `meka profile use <name>`",
                    profiles.keys().cloned().collect::<Vec<_>>().join(", ")
                )),
            ),
        },
    }
}
/// Refuse a `max_output_tokens` override that can't produce a valid Claude request: under
/// [`crate::config::ThinkingMode::Budgeted`] the budget is drawn from `max_tokens`, so the cap
/// must exceed it. Surfaced as a config error with clear guidance rather than a provider 400
/// mid-turn. The other two modes send no `budget_tokens` and have no such constraint, and neither
/// do the OpenAI backends.
///
/// Asked once per *profile*, never once per process: the values it reads are stated per profile, so
/// a run that resumes a session onto one profile must not be refused over another's settings, and
/// that session's own pairing must still be checked. [`ResolvedConfig::validate_default_profile`]
/// asks it of the process default; [`crate::provider::ProviderRegistry::resolve`] asks it of
/// whichever profile a session actually names. Both arguments come from the same
/// [`ProfileSettings`], so a profile is never refused over a number stated nowhere in it.
pub(crate) fn validate_max_output_tokens(
    profile: &str,
    backend: Option<Backend>,
    max_output_tokens: Option<u64>,
    thinking: crate::config::ThinkingMode,
    thinking_budget: u64,
) -> crate::error::Result<()> {
    let Some(max_output) = max_output_tokens else {
        return Ok(());
    };
    // The same predicate `/status` and `meka profile add` use: the invariant belongs to the
    // Messages protocol, so every backend that speaks it is in scope and a future one is in scope
    // automatically. Missing a backend here is invisible until a real request is rejected mid-turn,
    // which is what this check exists to pre-empt.
    let speaks_messages = backend.is_some_and(Backend::takes_thinking);
    let budgeted = matches!(thinking, crate::config::ThinkingMode::Budgeted);
    if speaks_messages && budgeted && max_output <= thinking_budget {
        // The remedy names this profile's own keys, never the `[thinking].budget` the budget may
        // have been inherited from. Both fix the refusal; only one of them leaves every other
        // profile budgeting against what it did before.
        return Err(crate::error::MekaError::Config(format!(
            "profile '{profile}': `max_output_tokens` ({max_output}) must exceed `thinking_budget` \
             ({thinking_budget}); raise it in `[profiles.{profile}]`",
        )));
    }
    Ok(())
}
/// Stable per-device identity for `claude-subscription` (embedded in `metadata.user_id`). Other
/// backends get an empty string, so no stub config file is written for an unused value.
pub(crate) mod device_id {
    use std::path::Path;

    use super::{config_file_path, write_file_atomic};

    /// Lookup order: configured → Claude Code's `~/.claude.json` userID → freshly generated. The
    /// claude.json fallback lets meka and Claude Code on the same machine share a device identity.
    /// A freshly seeded value is persisted into the account's `[accounts.<name>].device_id` so it
    /// stays stable across runs.
    pub(crate) fn resolve(
        backend: Option<super::Backend>,
        account_name: Option<&str>,
        configured: Option<&str>,
    ) -> String {
        if backend != Some(super::Backend::ClaudeSubscription) {
            return String::new();
        }

        // The account's `device_id`, already deserialized, wins.
        if let Some(id) = configured
            && !id.is_empty()
        {
            return id.to_string();
        }

        let (id, source) = match read_claude_code_user_id(dirs::home_dir()) {
            Some(id) => (id, "~/.claude.json"),
            None => (generate(), "random"),
        };
        tracing::info!("seeded claude-subscription device_id from {source}");

        // With no config path or account name the id is still used for this run, just not saved.
        if let (Some(account), Some(path)) = (account_name, config_file_path())
            && let Err(error) = persist(&path, account, &id)
        {
            tracing::warn!("failed to persist device_id: {error}");
        }
        id
    }

    /// A fresh device id: 32 random bytes as lowercase hex.
    pub(super) fn generate() -> String {
        use rand::RngExt;
        let mut bytes = [0u8; 32];
        rand::rng().fill(&mut bytes);
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Claude Code's `userID`, read from the `.claude.json` under `home`.
    pub(super) fn read_claude_code_user_id(home: Option<std::path::PathBuf>) -> Option<String> {
        read_user_id_from(&home?.join(".claude.json"))
    }

    pub(crate) fn read_user_id_from(path: &Path) -> Option<String> {
        let contents = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            // Absent is the ordinary state of a machine without Claude Code, and nothing about it
            // is actionable; a file that is there but cannot be read is.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                tracing::debug!(
                    "no user-id file at '{path}': {error}",
                    path = path.display()
                );
                return None;
            }
            Err(error) => {
                tracing::warn!(
                    "failed to read user-id file '{path}': {error}",
                    path = path.display()
                );
                return None;
            }
        };
        let document: serde_json::Value = match serde_json::from_str(&contents) {
            Ok(document) => document,
            Err(error) => {
                tracing::warn!(
                    "failed to parse user-id file '{path}': {error}",
                    path = path.display()
                );
                return None;
            }
        };
        let id = document.get("userID")?.as_str()?.trim();
        if id.is_empty() {
            return None;
        }
        Some(id.to_string())
    }

    pub(crate) fn persist(path: &Path, account: &str, id: &str) -> std::io::Result<()> {
        // Under the config lock because this runs the first time `ProviderRegistry::device_id_for`
        // resolves a `claude-subscription` account that states no device id, so a plain `meka`
        // launch competes with whatever `meka mcp add` is doing in the next terminal.
        let _lock = super::lock_config_file()?;
        let contents = std::fs::read_to_string(path).unwrap_or_default();
        let mut document: toml_edit::DocumentMut = contents
            .parse()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

        // The account's table already exists, since the profile was resolved through it; a shape
        // that does not match is left alone rather than rewritten.
        let Some(item) = document
            .get_mut("accounts")
            .and_then(|accounts| accounts.get_mut(account))
        else {
            return Ok(());
        };
        let Some(table) = item.as_table_like_mut() else {
            return Ok(());
        };
        table.insert("device_id", toml_edit::value(id));
        // `insert` appends, so without this the one key meka writes for itself would trail whatever
        // the table already held. Sorting is cheap and leaves the file in the order the reference
        // documents, which is what every other writer guarantees.
        super::sort_account_keys(item);

        write_file_atomic(path, &document.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model is a non-empty name. `""` is the half `is_none()` misses, and the half that would
    /// otherwise be written by `profile set` and sent to the provider verbatim.
    #[test]
    fn a_profile_needs_a_model_that_is_not_empty() {
        assert_eq!(
            require_model("work", Some("claude-opus-5")).expect("a model"),
            "claude-opus-5"
        );
        for missing in [None, Some(""), Some("   ")] {
            let message = require_model("work", missing)
                .expect_err("no model to run on")
                .to_string();
            assert!(
                message.contains("'work'") && message.contains("meka profile set work model"),
                "the refusal names the profile and the command that fixes it: {message}"
            );
        }
    }

    fn profiles_from(
        backends: &[(&str, &str)],
    ) -> std::collections::BTreeMap<String, ProfileConfig> {
        backends
            .iter()
            .map(|(name, _backend)| {
                (name.to_string(), ProfileConfig {
                    account: name.to_string(),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn account_on(backend: &str) -> AccountConfig {
        AccountConfig {
            backend: backend.to_string(),
            ..Default::default()
        }
    }

    /// An override beats the profile, and an absent one leaves the profile's value alone.
    ///
    /// The two fields that default *on*, and the one that falls back to `[session]`.
    ///
    /// A profile saying nothing must resolve the way a profile-less run did, or switching a session
    /// onto a bare profile would silently turn images or redacted thinking off.
    #[test]
    fn an_unstated_profile_resolves_to_the_documented_defaults() {
        let bare = ProfileConfig::default();
        let settings = resolve_profile(
            &bare,
            &account_on("openai-chat-completions"),
            Some(200_000),
            None,
            String::new(),
        )
        .expect("resolves");

        assert!(settings.vision, "vision defaults on");
        assert!(settings.redact_thinking, "redact_thinking defaults on");
        assert_eq!(
            settings.context_window,
            Some(200_000),
            "falls back to `[session].context_window`"
        );
        assert_eq!(settings.effort, None, "no effort means the provider's own");
    }

    /// A profile's own window beats `[session].context_window`, which is the whole reason a
    /// per-profile value exists: one small model among several must not drag the rest down.
    #[test]
    fn a_profiles_context_window_beats_the_session_default() {
        let profile = ProfileConfig {
            context_window: Some(32_000),
            ..Default::default()
        };
        let settings = resolve_profile(
            &profile,
            &account_on("anthropic-messages"),
            Some(1_000_000),
            None,
            String::new(),
        )
        .expect("resolves");
        assert_eq!(settings.context_window, Some(32_000));
    }

    /// The budget resolves like the window: profile first, then the installation-wide fallback,
    /// then the built-in default.
    ///
    /// The middle rung is the one worth pinning. Dropping it silently sends every profile the
    /// built-in 16000 and ignores the `[thinking].budget` most configs state, which no request
    /// would refuse and no other test would notice.
    #[test]
    fn a_profiles_thinking_budget_beats_the_installation_fallback() {
        let budgeted = |thinking_budget| ProfileConfig {
            thinking_budget,
            ..Default::default()
        };
        let resolve = |profile: &ProfileConfig, fallback| {
            resolve_profile(
                profile,
                &account_on("anthropic-messages"),
                None,
                fallback,
                String::new(),
            )
            .expect("resolves")
            .thinking_budget
        };

        assert_eq!(
            resolve(&budgeted(Some(8_000)), Some(64_000)),
            8_000,
            "the profile's own budget wins over `[thinking].budget`"
        );
        assert_eq!(
            resolve(&budgeted(None), Some(64_000)),
            64_000,
            "a profile stating none falls back to `[thinking].budget`"
        );
        assert_eq!(
            resolve(&budgeted(None), None),
            DEFAULT_THINKING_BUDGET_TOKENS,
            "with neither stated, the documented default"
        );
    }

    /// Two profiles resolve two budgets from one registry, so [`validate_max_output_tokens`] judges
    /// each against its own number rather than one the process holds for both.
    #[test]
    fn two_profiles_can_hold_different_thinking_budgets() {
        let big = ProfileConfig {
            thinking_budget: Some(60_000),
            ..Default::default()
        };
        let small = ProfileConfig {
            thinking_budget: Some(2_048),
            ..Default::default()
        };
        let resolve = |profile: &ProfileConfig| {
            resolve_profile(
                profile,
                &account_on("anthropic-messages"),
                None,
                Some(16_000),
                String::new(),
            )
            .expect("resolves")
            .thinking_budget
        };
        assert_eq!(resolve(&big), 60_000);
        assert_eq!(resolve(&small), 2_048);
    }

    /// A profile is judged against its own budget, not the installation's.
    ///
    /// `max_output_tokens = 8000` is fine beside a 2048 budget and impossible beside a 64000 one,
    /// and the second profile's number must not decide the first profile's fate.
    #[test]
    fn max_output_tokens_is_checked_against_the_profiles_own_budget() {
        let check = |thinking_budget| {
            validate_max_output_tokens(
                "work",
                Some(Backend::AnthropicMessages),
                Some(8_000),
                crate::config::ThinkingMode::Budgeted,
                thinking_budget,
            )
        };
        assert!(
            check(2_048).is_ok(),
            "8000 output over a 2048 budget is fine"
        );
        let error = check(64_000).expect_err("8000 output cannot carry a 64000 budget");
        let message = error.to_string();
        assert!(
            message.contains("[profiles.work]"),
            "the remedy names the profile's own table, not a global: {message}"
        );
        assert!(
            !message.contains("[thinking].budget"),
            "fixing one profile must not be described as editing every profile's fallback: \
             {message}"
        );
    }

    #[test]
    fn select_active_profile_explicit_request_wins() {
        let providers = profiles_from(&[
            ("work", "claude-subscription"),
            ("personal", "openai-chat-completions"),
        ]);
        let (active, error) = select_profile(
            Some("personal".to_string()),
            ProfileRequest::Flag,
            &providers,
        );
        assert_eq!(active.as_deref(), Some("personal"));
        assert!(error.is_none());
    }

    #[test]
    fn select_active_profile_sole_profile_when_no_request() {
        let providers = profiles_from(&[("only", "anthropic-messages")]);
        let (active, error) = select_profile(None, ProfileRequest::DefaultProfile, &providers);
        assert_eq!(active.as_deref(), Some("only"));
        assert!(error.is_none());
    }

    #[test]
    fn select_active_profile_zero_profiles_errors() {
        let providers = profiles_from(&[]);
        let (active, error) = select_profile(None, ProfileRequest::DefaultProfile, &providers);
        assert!(active.is_none());
        assert!(error.expect("error expected").contains("profile add"));
    }

    #[test]
    fn select_active_profile_ambiguous_without_default_errors() {
        let providers = profiles_from(&[
            ("work", "claude-subscription"),
            ("personal", "openai-chat-completions"),
        ]);
        let (active, error) = select_profile(None, ProfileRequest::DefaultProfile, &providers);
        assert!(active.is_none());
        let error = error.expect("error expected");
        assert!(error.contains("multiple profiles"));
        // The error lists the configured names so the user knows what to pick.
        assert!(error.contains("work") && error.contains("personal"));
    }

    #[test]
    fn select_active_profile_unknown_name_errors() {
        let providers = profiles_from(&[("work", "claude-subscription")]);
        let (active, error) = select_profile(
            Some("missing".to_string()),
            ProfileRequest::Flag,
            &providers,
        );
        assert!(active.is_none());
        assert!(
            error
                .expect("error expected")
                .contains("no profile named 'missing'")
        );
    }

    #[test]
    fn a_device_id_is_empty_for_every_backend_but_claude_subscription() {
        // Should not generate / persist anything when the provider doesn't need a device_id. Empty
        // string flows through but is ignored by non-claude-subscription providers.
        assert_eq!(
            device_id::resolve(Some(Backend::OpenAiChatCompletions), Some("work"), None),
            ""
        );
        assert_eq!(
            device_id::resolve(Some(Backend::AnthropicMessages), Some("work"), None),
            ""
        );
        assert_eq!(device_id::resolve(None, None, None), "");
        // Even an explicit configured value is suppressed when the provider isn't
        // claude-subscription; the field is provider-scoped.
        assert_eq!(
            device_id::resolve(
                Some(Backend::OpenAiChatCompletions),
                Some("work"),
                Some("explicit")
            ),
            ""
        );
    }

    #[test]
    fn resolve_device_id_uses_configured_value_for_claude_oauth() {
        // A configured value returns before the persist branch, so this never touches the FS.
        let id = "deadbeef".repeat(8);
        assert_eq!(
            device_id::resolve(Some(Backend::ClaudeSubscription), Some("work"), Some(&id)),
            id,
            "configured value must be used verbatim for claude-subscription"
        );
    }

    /// Every key in [`ACCOUNT_KEY_ORDER`] and [`PROFILE_KEY_ORDER`] is one the matching struct
    /// actually models, in order.
    ///
    /// `deny_unknown_fields` does the proving: a list naming a key the struct has since renamed or
    /// dropped stops this fixture parsing. Without that, the sorter would quietly treat the real
    /// key as unmodeled and file it last, a drift that surfaces as a diff nobody asked for rather
    /// than as an error, which is the hardest kind to trace back.
    ///
    /// The fixture is also the order, written out, so each list has one readable statement of
    /// itself that a person can check against the reference without running anything.
    #[test]
    fn every_canonical_key_is_one_the_tables_model() {
        let fixture = concat!(
            "[accounts.work]\n",
            "backend = \"claude-subscription\"\n",
            "base_url = \"https://api.anthropic.com\"\n",
            "oauth_token_url = \"https://example.invalid/token\"\n",
            "client_id = \"a-client\"\n",
            "device_id = \"a-device\"\n",
            "\n",
            "[profiles.work]\n",
            "account = \"work\"\n",
            "model = \"claude-opus-5\"\n",
            "context_window = 200000\n",
            "max_output_tokens = 64000\n",
            "effort = \"high\"\n",
            "vision = true\n",
            "thinking = \"budgeted\"\n",
            "thinking_budget = 32000\n",
            "max_request_bytes = 8388608\n",
            "redact_thinking = true\n",
        );

        let parsed: ConfigFile =
            toml::from_str(fixture).expect("every canonical key must be one the tables model");
        assert!(parsed.accounts.contains_key("work"));
        assert!(parsed.profiles.contains_key("work"));

        let document: toml_edit::DocumentMut = fixture.parse().expect("parse");
        let keys_of = |table: &str| -> Vec<String> {
            document[table]["work"]
                .as_table()
                .expect("the table")
                .iter()
                .map(|(key, _)| key.to_string())
                .collect()
        };
        assert_eq!(
            keys_of("accounts"),
            ACCOUNT_KEY_ORDER,
            "the fixture must cover the account list exactly, in order"
        );
        assert_eq!(
            keys_of("profiles"),
            PROFILE_KEY_ORDER,
            "the fixture must cover the profile list exactly, in order"
        );
    }

    /// A scrambled profile comes back in [`PROFILE_KEY_ORDER`], with its annotations intact.
    ///
    /// The comments are the point, not decoration. Sorting moves whole entries, and `toml_edit`
    /// hangs a whole-line comment off the *following* key's `leaf_decor` and a trailing one off the
    /// value, so if either were held somewhere else, sorting would silently re-attach a user's
    /// explanation to a different setting. That is a worse failure than the disorder it fixes,
    /// because the file still parses and still reads as if it meant something.
    #[test]
    fn sorting_a_profile_orders_its_keys_and_carries_the_comments_with_them() {
        let mut document: toml_edit::DocumentMut = concat!(
            "[profiles.work]\n",
            "redact_thinking = false\n",
            "\n",
            "# the window this model really has\n",
            "context_window = 200000\n",
            "model = \"claude-opus-5\" # pinned deliberately\n",
            "account = \"work\"\n",
        )
        .parse()
        .expect("parse");

        let profile = document
            .get_mut("profiles")
            .and_then(|profiles| profiles.get_mut("work"))
            .expect("the profile table");
        sort_profile_keys(profile);

        let rendered = document.to_string();
        let keys: Vec<&str> = rendered
            .lines()
            .filter_map(|line| line.split_once(" = "))
            .map(|(key, _)| key.trim())
            .collect();
        assert_eq!(
            keys,
            ["account", "model", "context_window", "redact_thinking"],
            "keys should follow PROFILE_KEY_ORDER: {rendered}"
        );
        assert!(
            rendered.contains("# the window this model really has\ncontext_window = 200000"),
            "the comment should still sit above the key it explains: {rendered}"
        );
        assert!(
            rendered.contains("model = \"claude-opus-5\" # pinned deliberately"),
            "a trailing comment should still sit beside its value: {rendered}"
        );
    }

    /// The account sorter is the profile sorter's twin: a scrambled account comes back in
    /// [`ACCOUNT_KEY_ORDER`], with its annotations still on the keys they explain.
    #[test]
    fn sorting_an_account_orders_its_keys_and_carries_the_comments_with_them() {
        let mut document: toml_edit::DocumentMut = concat!(
            "[accounts.work]\n",
            "device_id = \"a-device\"\n",
            "\n",
            "# the endpoint this account bills\n",
            "base_url = \"https://api.anthropic.com\"\n",
            "backend = \"anthropic-messages\" # pinned deliberately\n",
        )
        .parse()
        .expect("parse");

        let account = document
            .get_mut("accounts")
            .and_then(|accounts| accounts.get_mut("work"))
            .expect("the account table");
        sort_account_keys(account);

        let rendered = document.to_string();
        let keys: Vec<&str> = rendered
            .lines()
            .filter_map(|line| line.split_once(" = "))
            .map(|(key, _)| key.trim())
            .collect();
        assert_eq!(
            keys,
            ["backend", "base_url", "device_id"],
            "keys should follow ACCOUNT_KEY_ORDER: {rendered}"
        );
        assert!(
            rendered.contains(
                "# the endpoint this account bills\nbase_url = \"https://api.anthropic.com\""
            ),
            "the comment should still sit above the key it explains: {rendered}"
        );
        assert!(
            rendered.contains("backend = \"anthropic-messages\" # pinned deliberately"),
            "a trailing comment should still sit beside its value: {rendered}"
        );
    }

    /// A key this build does not model sorts last rather than wherever the sort left it.
    ///
    /// `deny_unknown_fields` means it cannot survive a load, so this pins what an unreachable case
    /// looks like. Worth pinning anyway: the alternative is a write whose output depends on the
    /// sort's internals, which is the kind of thing that produces a diff on a file nobody edited.
    #[test]
    fn an_unmodeled_profile_key_sorts_last() {
        let mut document: toml_edit::DocumentMut =
            "[profiles.work]\nzzz_future = 1\nmodel = \"m\"\naaa_future = 2\naccount = \"a\"\n"
                .parse()
                .expect("parse");
        let profile = document
            .get_mut("profiles")
            .and_then(|profiles| profiles.get_mut("work"))
            .expect("the profile table");
        sort_profile_keys(profile);

        let rendered = document.to_string();
        let keys: Vec<&str> = rendered
            .lines()
            .filter_map(|line| line.split_once(" = "))
            .map(|(key, _)| key.trim())
            .collect();
        assert_eq!(
            keys,
            ["account", "model", "aaa_future", "zzz_future"],
            "{rendered}"
        );
    }

    #[test]
    fn persist_device_id_writes_into_account_table() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "default_profile = \"work\"\n\n[accounts.work]\nbackend = \"claude-subscription\"\n\n[profiles.work]\naccount = \"work\"\nmodel = \"claude-opus-4-8\"\n",
        )
        .expect("seed config");

        device_id::persist(&path, "work", "abc123").expect("persist");

        let contents = std::fs::read_to_string(&path).expect("read back");
        let config: ConfigFile = toml::from_str(&contents).expect("re-parse");
        let persisted = config
            .accounts
            .get("work")
            .and_then(|account| account.device_id.as_deref());
        assert_eq!(
            persisted,
            Some("abc123"),
            "device_id must be stored under the account"
        );
        // Existing account fields are preserved.
        assert_eq!(
            config
                .accounts
                .get("work")
                .map(|account| account.backend.as_str()),
            Some("claude-subscription")
        );
        // Closing the loop: the reader feeds the persisted value back through `resolve`, which must
        // return it verbatim rather than regenerate a fresh id on the next run.
        assert_eq!(
            device_id::resolve(Some(Backend::ClaudeSubscription), Some("work"), persisted),
            "abc123",
            "a persisted device_id must be picked up on the next run, not regenerated"
        );
    }

    #[test]
    fn read_user_id_from_valid_claude_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(
            &path,
            r#"{"userID": "af5986c7cb3b5e8d00eaf3da3b81730c6f523b1e68e1720c7128a96167534be3", "other": "stuff"}"#,
        )
        .expect("write");
        assert_eq!(
            device_id::read_user_id_from(&path).as_deref(),
            Some("af5986c7cb3b5e8d00eaf3da3b81730c6f523b1e68e1720c7128a96167534be3")
        );
    }

    #[test]
    fn read_user_id_from_missing_file_returns_none() {
        let path = std::path::Path::new("/nonexistent/path/claude.json");
        assert!(device_id::read_user_id_from(path).is_none());
    }

    #[test]
    fn read_user_id_from_malformed_json_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, "{not valid json").expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn read_user_id_from_missing_field_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"foo": "bar"}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn read_user_id_from_empty_string_returns_none() {
        // An empty `userID` in claude.json shouldn't override meka's own random-generation
        // fallback; treat it as "not configured".
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": ""}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn read_user_id_from_whitespace_only_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": "   "}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn read_user_id_from_non_string_returns_none() {
        // A non-string `userID` (number, object, …) shouldn't crash; just decline to use the
        // value.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": 12345}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn read_user_id_from_trims_surrounding_whitespace() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": "  abcdef123  "}"#).expect("write");
        assert_eq!(
            device_id::read_user_id_from(&path).as_deref(),
            Some("abcdef123")
        );
    }

    /// A generated device id has the shape of the one it stands in for, and no two runs share one.
    #[test]
    fn a_generated_device_id_is_64_lowercase_hex_characters_and_never_repeats() {
        let first = device_id::generate();
        let second = device_id::generate();
        for id in [&first, &second] {
            assert_eq!(id.len(), 64, "{id}");
            assert!(
                id.chars()
                    .all(|character| matches!(character, '0'..='9' | 'a'..='f')),
                "{id}"
            );
        }
        assert_ne!(first, second);
    }

    /// Claude Code's id comes from the `.claude.json` in the home directory and from nowhere else:
    /// a machine without one yields nothing, never a placeholder.
    #[test]
    fn claude_code_s_user_id_is_read_from_the_home_directory_only_when_the_file_is_there() {
        let home = tempfile::tempdir().expect("tempdir");
        let under_home = || Some(home.path().to_path_buf());
        assert_eq!(device_id::read_claude_code_user_id(under_home()), None);
        assert_eq!(device_id::read_claude_code_user_id(None), None);

        std::fs::write(home.path().join(".claude.json"), r#"{"userID": "abc123"}"#).expect("write");
        assert_eq!(
            device_id::read_claude_code_user_id(under_home()).as_deref(),
            Some("abc123")
        );
    }

    #[test]
    fn the_budget_invariant_is_checked_only_under_budgeted_thinking() {
        // Budgeted draws the thinking budget out of `max_tokens`, so a cap at or below it can only
        // produce a 400. Catch it at config load, where the message can say what to change.
        //
        // Both Anthropic Messages backends, because the invariant belongs to the protocol rather
        // than to the account: dropping either from the gate is invisible until a real request is
        // rejected mid-turn, which is exactly what this check exists to pre-empt.
        for backend in [Backend::ClaudeSubscription, Backend::AnthropicMessages] {
            let result = validate_max_output_tokens(
                "work",
                Some(backend),
                Some(5_000),
                ThinkingMode::Budgeted,
                10_000,
            );
            assert!(result.is_err(), "{backend}");
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("`thinking_budget`"),
                "{backend}: the error must name the budget it conflicts with"
            );
        }
        assert!(
            validate_max_output_tokens(
                "work",
                Some(Backend::ClaudeSubscription),
                Some(20_000),
                ThinkingMode::Budgeted,
                10_000
            )
            .is_ok()
        );

        // The other two modes send no `budget_tokens` at all, so the same cap is fine. Reading the
        // mode rather than guessing it from the model name is the point: a budget-free request is
        // stated, not inferred.
        for mode in [ThinkingMode::Adaptive, ThinkingMode::Off] {
            assert!(
                validate_max_output_tokens(
                    "work",
                    Some(Backend::ClaudeSubscription),
                    Some(5_000),
                    mode,
                    10_000
                )
                .is_ok(),
                "{mode:?}"
            );
        }
        // Non-Claude backends have no such constraint either, whatever the mode.
        assert!(
            validate_max_output_tokens(
                "work",
                Some(Backend::OpenAiChatCompletions),
                Some(100),
                ThinkingMode::Budgeted,
                10_000
            )
            .is_ok()
        );
        // No override at all.
        assert!(
            validate_max_output_tokens(
                "work",
                Some(Backend::AnthropicMessages),
                None,
                ThinkingMode::Budgeted,
                10_000
            )
            .is_ok()
        );
    }

    /// The selected profile's window must not become every other profile's fallback.
    ///
    /// `session_context_window` is the seed `ProviderRegistry` hands `resolve_profile` for *each*
    /// profile, so folding the selected profile into it here applies that profile twice: a profile
    /// stating no window inherited the default profile's, which is the exact defect per-profile
    /// windows exist to prevent, observable as `GET /v1/sessions/{id}/context` reporting the
    /// default profile's number for a session on another profile.
    ///
    /// Touches process env, so it serializes against any other env-var test in this file via
    /// [`CONFIG_DIR_ENV_LOCK`].
    #[test]
    fn the_session_window_seed_is_not_the_active_profiles_window() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        use clap::Parser;

        let resolve_with = |config: &str| {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::write(dir.path().join("config.toml"), config).expect("write config.toml");
            // SAFETY: `CONFIG_DIR_ENV_LOCK` serializes this with any other env-var test.
            unsafe {
                std::env::set_var("MEKA_CONFIG_DIR", dir.path());
            }
            let resolved =
                ResolvedConfig::resolve(crate::cli::Cli::parse_from(["meka"]).overrides());
            // SAFETY: same as above; the guard is held for the full set→read→clear cycle.
            unsafe {
                std::env::remove_var("MEKA_CONFIG_DIR");
            }
            resolved
        };

        let profile_only = resolve_with(
            r#"
default_profile = "big"

[accounts.big]
backend = "anthropic-messages"

[profiles.big]
account = "big"
model = "big-model"
context_window = 999000

[accounts.small]
backend = "anthropic-messages"

[profiles.small]
account = "small"
model = "small-model"
"#,
        );
        assert_eq!(
            profile_only.session_context_window, None,
            "a profile's own window is not a `[session]` default for the other profiles"
        );
        assert_eq!(
            profile_only
                .profiles
                .get("small")
                .map(|profile| resolve_profile(
                    profile,
                    profile_only.accounts.get("small").expect("the account"),
                    profile_only.session_context_window,
                    profile_only.default_thinking_budget,
                    String::new(),
                )
                .expect("resolves"))
                .and_then(|settings| settings.context_window),
            None,
            "a profile stating no window resolves to none, so the documented default applies"
        );

        let with_session_block = resolve_with(
            r#"
default_profile = "big"

[session]
context_window = 200000

[accounts.big]
backend = "anthropic-messages"

[profiles.big]
account = "big"
model = "big-model"
context_window = 999000
"#,
        );
        assert_eq!(
            with_session_block.session_context_window,
            Some(200_000),
            "`[session].context_window` is still carried through for profiles that state none"
        );
    }
}
