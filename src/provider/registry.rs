//! From a profile name to a running provider: building one, caching it per key, and resolving what
//! a session is bound to.

use super::*;

/// Builds the [`Provider`] a [`Backend`] names. Each setter documents which backends read it; the
/// others ignore it.
pub(crate) struct ProviderBuilder {
    pub(super) backend: Backend,
    pub(super) credential: AuthCredential,
    pub(super) model: String,
    pub(super) base_url: Option<String>,
    pub(super) client_id: Option<String>,
    pub(super) oauth_token_url: Option<String>,
    pub(super) token_store: Option<Arc<TokenStore>>,
    /// Account name the credential is stored under; OAuth providers use it to write refreshed
    /// tokens back to the right `account_credentials` row. Required by both subscription backends;
    /// see [`Self::resolve_credential_key`].
    pub(super) credential_key: Option<String>,
    pub(super) thinking: ThinkingMode,
    pub(super) thinking_budget_tokens: u64,
    pub(super) device_id: String,
    pub(super) effort: Option<String>,
    pub(super) thinking_display: crate::config::ThinkingDisplay,
    pub(super) context_window: Option<u64>,
    pub(super) max_output_tokens: Option<u64>,
    pub(super) max_request_bytes: Option<usize>,
}
impl ProviderBuilder {
    pub(crate) fn new(
        backend: Backend,
        credential: AuthCredential,
        model: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            credential,
            model: model.into(),
            base_url: None,
            client_id: None,
            oauth_token_url: None,
            token_store: None,
            credential_key: None,
            thinking: ThinkingMode::Off,
            thinking_budget_tokens: 0,
            device_id: String::new(),
            effort: None,
            thinking_display: crate::config::ThinkingDisplay::default(),
            context_window: None,
            max_output_tokens: None,
            max_request_bytes: None,
        }
    }

    /// Override the HTTP endpoint. Applies to every provider variant; defaults to the Claude or
    /// OpenAI production URL.
    pub(crate) fn base_url(mut self, value: Option<String>) -> Self {
        self.base_url = value;
        self
    }

    /// OAuth client ID. Consumed by both subscription backends.
    pub(crate) fn client_id(mut self, value: Option<String>) -> Self {
        self.client_id = value;
        self
    }

    /// OAuth token endpoint. Consumed by both subscription backends.
    pub(crate) fn oauth_token_url(mut self, value: Option<String>) -> Self {
        self.oauth_token_url = value;
        self
    }

    /// Sink for refreshed OAuth tokens. Consumed by both subscription backends; when `None`,
    /// refreshed tokens are held in memory only.
    pub(crate) fn token_store(mut self, value: Option<Arc<TokenStore>>) -> Self {
        self.token_store = value;
        self
    }

    /// Account name the credential is stored under (OAuth refresh write-back key). A subscription
    /// backend does not build without one.
    pub(crate) fn credential_key(mut self, value: Option<String>) -> Self {
        self.credential_key = value;
        self
    }

    /// Claude-only: the profile's thinking mode, plus the budget cap [`ThinkingMode::Budgeted`]
    /// uses. Ignored by the OpenAI backends.
    pub(crate) fn thinking(mut self, mode: ThinkingMode, budget_tokens: u64) -> Self {
        self.thinking = mode;
        self.thinking_budget_tokens = budget_tokens;
        self
    }

    /// Stable device identity embedded in `metadata.user_id`. Only consumed by
    /// `claude-subscription`.
    pub(crate) fn device_id(mut self, value: String) -> Self {
        self.device_id = value;
        self
    }

    /// The user's explicit reasoning-effort override (`low` / `medium` / `high` / `xhigh` / `max`),
    /// or `None` to leave the field off, so the provider applies its own. Consumed by every
    /// backend: Claude maps it to `output_config.effort`, OpenAI to `reasoning.effort`.
    pub(crate) fn effort(mut self, value: Option<String>) -> Self {
        self.effort = value;
        self
    }

    /// How thinking is presented. Only consumed by `claude-subscription`.
    pub(crate) fn thinking_display(mut self, value: crate::config::ThinkingDisplay) -> Self {
        self.thinking_display = value;
        self
    }

    /// The profile's context window, which decides the 1M-context beta. Only consumed by
    /// `claude-subscription`.
    pub(crate) fn context_window(mut self, value: Option<u64>) -> Self {
        self.context_window = value;
        self
    }

    /// Per-request output (completion) token cap. When `None`, each backend keeps its built-in
    /// default. Consumed by every backend.
    pub(crate) fn max_output_tokens(mut self, value: Option<u64>) -> Self {
        self.max_output_tokens = value;
        self
    }

    /// Largest request body before old images are redacted. Anthropic backends only; `None` keeps
    /// their default.
    pub(crate) fn max_request_bytes(mut self, value: Option<usize>) -> Self {
        self.max_request_bytes = value;
        self
    }

    /// The row a subscription backend writes refreshed tokens to.
    ///
    /// No default in its place: the registry is the one authority on which row a profile's
    /// credential lives in, and a builder that fell back to the backend name would write refreshed
    /// tokens to a row no account names, leaving the real one stale and `account list` reporting an
    /// orphan. Only meka's own code reaches a build, so a missing name is a defect here, not a
    /// configuration error.
    pub(super) fn resolve_credential_key(&self) -> Result<String> {
        self.credential_key.clone().ok_or_else(|| {
            MekaError::Internal(format!(
                "a '{}' provider was built without the account its credential is stored under",
                self.backend
            ))
        })
    }

    pub(crate) fn build(self) -> Result<Arc<dyn Provider>> {
        match self.backend {
            Backend::OpenAiResponses => {
                // Same credential handling as `openai-chat-completions`: these two differ by
                // protocol, not by how they authenticate.
                let api_key = match &self.credential {
                    AuthCredential::ApiKey(key) => key.clone(),
                    AuthCredential::OAuthToken { .. } => {
                        return Err(MekaError::Config(
                            "backend 'openai-responses' takes an API key, not an OAuth token; \
                             'chatgpt-subscription' bills a subscription"
                                .to_string(),
                        ));
                    }
                };
                Ok(Arc::new(OpenAiResponsesProvider::new(api_key, self)?))
            }
            Backend::OpenAiChatCompletions => {
                let api_key = match &self.credential {
                    AuthCredential::ApiKey(key) => key.clone(),
                    AuthCredential::OAuthToken { .. } => {
                        return Err(MekaError::Config(
                            "backend 'openai-chat-completions' takes an API key, not an OAuth \
                             token; 'chatgpt-subscription' bills a subscription"
                                .to_string(),
                        ));
                    }
                };
                Ok(Arc::new(OpenAiChatCompletionsProvider::new(api_key, self)?))
            }
            Backend::AnthropicMessages => {
                let api_key = match &self.credential {
                    AuthCredential::ApiKey(key) => key.clone(),
                    AuthCredential::OAuthToken { .. } => {
                        return Err(MekaError::Config(
                            "backend 'anthropic-messages' takes an API key, not an OAuth token; \
                             'claude-subscription' bills a subscription"
                                .to_string(),
                        ));
                    }
                };
                Ok(Arc::new(AnthropicMessagesProvider::new(api_key, self)?))
            }
            Backend::ClaudeSubscription => {
                if matches!(self.credential, AuthCredential::ApiKey(_)) {
                    return Err(MekaError::Config(
                        "backend 'claude-subscription' takes an OAuth token, not an API key; use \
                         'anthropic-messages' to bill an API key"
                            .to_string(),
                    ));
                }
                Ok(Arc::new(ClaudeSubscriptionProvider::new(self)?))
            }
            Backend::ChatGptSubscription => {
                if matches!(self.credential, AuthCredential::ApiKey(_)) {
                    return Err(MekaError::Config(
                        "backend 'chatgpt-subscription' takes an OAuth token, not an API key; use \
                         'openai-responses' to bill an API key"
                            .to_string(),
                    ));
                }
                Ok(Arc::new(ChatGptSubscriptionProvider::new(self)?))
            }
        }
    }
}
/// What distinguishes one built provider from another.
///
/// The profile name determines the rest: nothing can rewrite a field inside a profile or the
/// account it names, so two providers built from one name are built from the same values. The
/// fields beside it are therefore redundant rather than load-bearing, and kept for one reason: they
/// make the memo key state what a provider is actually built from, so a future field that is *not*
/// a function of the name cannot be added without someone noticing this struct.
///
/// The credential is deliberately *not* part of this, even though the provider is built from one.
/// See [`CachedProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct ProviderKey {
    pub(super) profile: String,
    pub(super) account: String,
    pub(super) model: Option<String>,
    pub(super) base_url: Option<String>,
    pub(super) thinking: ThinkingMode,
}
/// A provider kept for reuse, beside the credential it was built from.
///
/// The credential is a *tag* rather than part of [`ProviderKey`] because it is the one input to a
/// build that supersedes its predecessor rather than distinguishing a sibling. Keyed, every
/// rotation would mint an entry and none would ever be dropped, so a long-lived `meka serve`
/// against an OAuth profile (which rewrites the credential on every token refresh) would
/// accumulate one `reqwest` client and its connection pool per hour, forever. Tagged, there is
/// exactly one entry per key and a rotation replaces it.
///
/// [`TokenStore::account_credential_version`] rather than the credential itself: the comparison
/// wants to know *whether* the credential moved, and nothing here needs a secret in memory to
/// answer that.
///
/// **What this reaches, and what it does not.** It reaches every later `build()`, which is every
/// session assembled or re-attached after the rotation. It does not reach a session already
/// resident: that one holds its own `Arc<dyn Provider>` on [`crate::agent::Agent`] until
/// `set_provider` replaces it, so it keeps presenting the old credential until it is evicted. That
/// is a property of resolving the profile once per session rather than once per turn, which is a
/// separate decision from this memo; it is documented for users at
/// `docs/book/src/configuration/overview.md` § "When edits take effect".
pub(super) struct CachedProvider {
    /// What [`TokenStore::account_credential_version`] said *before* this provider's credential
    /// was loaded. That order matters and is the fail-safe direction: read after, a rotation
    /// landing between the two reads would tag a provider built from the old credential with
    /// the new version and pin it forever. Read before, the same race tags a provider built
    /// from the new credential with the old version, and costs one needless rebuild that then
    /// converges.
    pub(super) credential_version: Option<String>,
    pub(super) provider: Arc<dyn Provider>,
}
/// Providers, built on demand for whichever profile is asked for and kept for reuse.
///
/// A single `Arc<dyn Provider>` built once at startup cannot serve sessions on different profiles:
/// a session records the one it runs with, and two sessions in one `meka serve` may name different
/// profiles and both have to work.
///
/// Reuse is the reason this caches rather than building per turn: an `Arc<dyn Provider>` owns a
/// `reqwest::Client` and therefore a connection pool, and rebuilding one per turn would throw away
/// every kept-alive connection.
///
/// **An account's credential is checked at first use, not at startup.** Only the process
/// default could be validated up front, and validating every configured account would make an
/// unused one with a stale token fail a launch it has nothing to do with. The cost is that a typo
/// in a profile a session names shows up when that session runs.
///
/// Reuse does not extend to a credential that has moved: see [`CachedProvider`].
pub(crate) struct ProviderRegistry {
    /// `config.toml`'s `[accounts.*]`, as they stood when this process started. A snapshot for
    /// the reason [`Self::profiles`] gives.
    pub(super) accounts: std::collections::BTreeMap<String, crate::config::AccountConfig>,
    /// `config.toml`'s `[profiles.*]`, as they stood when this process started.
    ///
    /// A snapshot on purpose, and the one place in this file where that is the right answer.
    /// `config.toml` is hand-edited and has no change feed; re-reading it per request would mean a
    /// half-written file could take a running server's sessions down, and would make which account
    /// a turn bills depend on when the turn happened to run. So a `meka profile add` while
    /// `meka serve` is up is invisible to it until restart, and `POST /v1/sessions` answers "not
    /// configured" about a profile `meka profile list` shows. Documented for users at
    /// `docs/book/src/configuration/overview.md` § "When edits take effect"; the credential behind
    /// an account is *not* snapshotted (see [`CachedProvider`]), because that one has a writer
    /// meka owns.
    pub(super) profiles: std::collections::BTreeMap<String, crate::config::ProfileConfig>,
    /// `[session].context_window`, which a profile's own value takes precedence over.
    pub(super) session_context_window: Option<u64>,
    /// `[thinking].budget`, the seed a profile stating no `thinking_budget` of its own
    /// falls back to. Held unresolved for the reason [`Self::session_context_window`] is:
    /// resolving it through any one profile would make that profile's budget everyone else's.
    pub(super) default_thinking_budget: Option<u64>,
    /// Device ids, resolved at most once per account.
    ///
    /// `claude-subscription` mints one and writes it into `config.toml` when the account states
    /// none, and `accounts` above is a snapshot taken when this was built, so a resolver reached
    /// per request never sees the value it just persisted: it mints another, and rewrites the
    /// config, on every session create, context poll and status query, for an identifier whose
    /// entire purpose is to stay the same. Lazy rather than filled in `new` so an account this
    /// process never uses is never seeded.
    pub(super) device_ids: std::sync::Mutex<std::collections::HashMap<String, String>>,
    pub(super) token_store: Arc<TokenStore>,
    pub(super) built: std::sync::Mutex<std::collections::HashMap<ProviderKey, CachedProvider>>,
    /// Debug-only: a scripted provider that stands in for every profile.
    ///
    /// Set by the three hosts when `MEKA_MOCK_PROVIDER=1`. One override here covers every profile,
    /// which is what a harness driving a session on any profile wants.
    #[cfg(any(debug_assertions, feature = "mock-provider"))]
    pub(super) scripted: std::sync::Mutex<Option<Arc<dyn Provider>>>,
}
impl ProviderRegistry {
    pub(crate) fn new(config: &crate::config::ResolvedConfig, token_store: TokenStore) -> Self {
        Self {
            accounts: config.accounts.clone(),
            profiles: config.profiles.clone(),
            session_context_window: config.session_context_window,
            default_thinking_budget: config.default_thinking_budget,
            device_ids: std::sync::Mutex::new(std::collections::HashMap::new()),
            token_store: Arc::new(token_store),
            built: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(any(debug_assertions, feature = "mock-provider"))]
            scripted: std::sync::Mutex::new(None),
        }
    }

    /// Resolve every `claude-subscription` account's device id ahead of the first ask, on a
    /// blocking thread.
    ///
    /// [`Self::settings`] is synchronous and resolves an id it has not seen on the spot, which for
    /// an account that states none writes `config.toml` under the config lock: a `flock` that
    /// waits on whatever other meka holds it, then two fsyncs, all on whichever runtime thread
    /// happened to ask and while holding the registry's mutex. Done once here, every later ask
    /// is a map read. An account whose `backend` does not parse is skipped; `settings` reports
    /// it.
    pub(crate) async fn preload_device_ids(&self) {
        let pending: Vec<(String, crate::config::AccountConfig)> = {
            let cache = crate::sync::lock(&self.device_ids);
            self.accounts
                .iter()
                .filter(|(name, account)| {
                    !cache.contains_key(*name)
                        && account
                            .backend
                            .parse::<Backend>()
                            .is_ok_and(|backend| backend == Backend::ClaudeSubscription)
                })
                .map(|(name, account)| (name.clone(), account.clone()))
                .collect()
        };
        for (name, account) in pending {
            let resolved = tokio::task::spawn_blocking({
                let name = name.clone();
                move || {
                    crate::config::resolve_device_id(
                        Backend::ClaudeSubscription,
                        &name,
                        account.device_id.as_deref(),
                    )
                }
            })
            .await;
            match resolved {
                Ok(device_id) => {
                    crate::sync::lock(&self.device_ids)
                        .entry(name)
                        .or_insert(device_id);
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to resolve the device id for account '{name}' ahead of time: {error}"
                    );
                }
            }
        }
    }

    /// Every configured profile's name, sorted, as `config.toml` stood when this process started:
    /// what `agent_spawn` offers when the profile a sub-agent runs on is the agent's to choose.
    pub(crate) fn profile_names(&self) -> Vec<String> {
        self.profiles.keys().cloned().collect()
    }

    /// The settings of the profile a session's row names, refused by name when `config.toml` no
    /// longer has it: a session runs on the profile it recorded, never on the default in its place,
    /// because quietly running the conversation somewhere else is the failure this arrangement
    /// exists to prevent.
    pub(crate) fn settings(&self, profile: &str) -> Result<crate::config::ProfileSettings> {
        let configured = crate::config::require_profile(profile, &self.profiles)?;
        let account = crate::config::account_for(profile, configured, &self.accounts)
            .map_err(MekaError::Config)?;
        crate::config::resolve_profile(
            configured,
            account,
            self.session_context_window,
            self.default_thinking_budget,
            self.device_id_for(&configured.account, account),
        )
        .map_err(MekaError::Config)
    }

    /// This account's `claude-subscription` device id, resolved on the first ask and remembered.
    pub(super) fn device_id_for(
        &self,
        name: &str,
        account: &crate::config::AccountConfig,
    ) -> String {
        let mut cache = crate::sync::lock(&self.device_ids);
        cache
            .entry(name.to_string())
            .or_insert_with(|| {
                // A `backend` that does not parse resolves no device id; `settings` refuses the
                // profile a moment later with the message that names the typo.
                account
                    .backend
                    .parse::<Backend>()
                    .map(|backend| {
                        crate::config::resolve_device_id(
                            backend,
                            name,
                            account.device_id.as_deref(),
                        )
                    })
                    .unwrap_or_default()
            })
            .clone()
    }

    /// The provider for one profile, built on first ask and reused after, with the settings it was
    /// built from.
    ///
    /// Both, because the caller wants both and resolving is not free: it reads the profile and
    /// looks up a device id. Returning only the provider would have [`resolved_profile`] resolve a
    /// second time to learn the window and the vision flag this call already computed.
    pub(crate) async fn resolve(
        &self,
        profile: &str,
    ) -> Result<(Arc<dyn Provider>, crate::config::ProfileSettings)> {
        let settings = self.settings(profile)?;

        #[cfg(any(debug_assertions, feature = "mock-provider"))]
        if let Ok(scripted) = self.scripted.lock()
            && let Some(scripted) = scripted.as_ref()
        {
            return Ok((Arc::clone(scripted), settings));
        }

        // Asked of the profile this session actually names, not of the process default. A pairing
        // that cannot produce a valid request is worth catching before the request, and only this
        // profile's own values can say whether it can.
        crate::config::validate_max_output_tokens(
            profile,
            Some(settings.backend),
            settings.max_output_tokens,
            settings.thinking,
            settings.thinking_budget,
        )?;
        let key = ProviderKey {
            profile: profile.to_string(),
            account: settings.account.clone(),
            model: settings.model.clone(),
            base_url: settings.base_url.clone(),
            thinking: settings.thinking,
        };
        // Pulled and compared rather than pushed, because the writer that supersedes a credential
        // is usually a *different process* (`meka account login work` run against a store a
        // `meka serve` is already using) and no invalidation hook can reach across that. Without
        // the comparison a rotation is invisible for the life of the process, and every later
        // build serves the provider holding the revoked key.
        let credential_version = self
            .token_store
            .account_credential_version(&settings.account)
            .await?;
        if let Ok(built) = self.built.lock()
            && let Some(existing) = built.get(&key)
            && existing.credential_version == credential_version
        {
            return Ok((Arc::clone(&existing.provider), settings));
        }

        let credential = self.credential_for(&settings.account).await?;
        let model = crate::config::require_model(profile, settings.model.as_deref())?.to_string();
        let needs_token_store = matches!(credential, AuthCredential::OAuthToken { .. });
        let provider = ProviderBuilder::new(settings.backend, credential, model)
            .base_url(settings.base_url.clone())
            .client_id(settings.client_id.clone())
            .credential_key(Some(settings.account.clone()))
            .oauth_token_url(settings.oauth_token_url.clone())
            .token_store(needs_token_store.then(|| Arc::clone(&self.token_store)))
            .thinking(settings.thinking, settings.thinking_budget)
            .device_id(settings.device_id.clone())
            .effort(settings.effort.clone())
            .thinking_display(settings.thinking_display)
            .context_window(settings.context_window)
            .max_output_tokens(settings.max_output_tokens)
            .max_request_bytes(settings.max_request_bytes)
            .build()?;

        match self.built.lock() {
            Ok(mut built) => {
                let cached = built.entry(key).or_insert_with(|| CachedProvider {
                    credential_version: credential_version.clone(),
                    provider: Arc::clone(&provider),
                });
                // Whoever got here first wins, and a loser drops its own build rather than
                // replacing a provider another turn may already be using, unless what is there
                // was built from a credential that has since been superseded, which is the case
                // this call exists to serve.
                //
                // Two builds spanning two rotations can land out of order and leave the older one
                // cached. That converges rather than sticking: the tag records which credential the
                // entry was built from, so the next ask compares it against the row and rebuilds.
                if cached.credential_version != credential_version {
                    *cached = CachedProvider {
                        credential_version,
                        provider: Arc::clone(&provider),
                    };
                }
                Ok((Arc::clone(&cached.provider), settings))
            }
            // A poisoned cache costs reuse, not correctness: the provider just built is complete
            // and usable, and the next ask builds another.
            Err(_) => Ok((provider, settings)),
        }
    }

    pub(super) async fn credential_for(&self, account: &str) -> Result<AuthCredential> {
        // Debug-only: the scripted provider replaces whatever this returns, so a harness need not
        // seed a credential it will never use. Reached only when `ProviderRegistry::build` was
        // called before a host installed the script.
        #[cfg(any(debug_assertions, feature = "mock-provider"))]
        if std::env::var("MEKA_MOCK_PROVIDER").as_deref() == Ok("1") {
            return Ok(AuthCredential::ApiKey("mock-provider".to_string()));
        }
        match self.token_store.load_account_credential(account).await? {
            Some(credential) => Ok(credential),
            None => Err(MekaError::Config(format!(
                "account '{account}' has no stored credential; run `meka account login {account}`"
            ))),
        }
    }

    /// Install a scripted provider in place of every profile's. Debug builds only.
    #[cfg(any(debug_assertions, feature = "mock-provider"))]
    pub(crate) fn install_scripted(&self, provider: Arc<dyn Provider>) {
        if let Ok(mut scripted) = self.scripted.lock() {
            *scripted = Some(provider);
        }
    }
}

#[cfg(test)]
impl ProviderRegistry {
    /// A registry over `names`, each an `anthropic-messages` profile on an account of the same
    /// name, with everything a resolution needs and nothing it does not. Tests that run a worker
    /// on one of them install a scripted provider, which stands in for every profile.
    pub(crate) fn for_test(token_store: TokenStore, names: &[&str]) -> Self {
        Self::for_test_over(
            names
                .iter()
                .map(|name| {
                    (name.to_string(), crate::config::AccountConfig {
                        backend: "anthropic-messages".to_string(),
                        ..Default::default()
                    })
                })
                .collect(),
            names
                .iter()
                .map(|name| {
                    (name.to_string(), crate::config::ProfileConfig {
                        account: name.to_string(),
                        ..Default::default()
                    })
                })
                .collect(),
            token_store,
        )
    }

    /// A registry over exactly these accounts and profiles, with everything a resolution needs
    /// and nothing it does not. The one test constructor; [`Self::for_test`] and the provider
    /// tests' own builder both go through it.
    pub(crate) fn for_test_over(
        accounts: std::collections::BTreeMap<String, crate::config::AccountConfig>,
        profiles: std::collections::BTreeMap<String, crate::config::ProfileConfig>,
        token_store: TokenStore,
    ) -> Self {
        Self {
            accounts,
            profiles,
            session_context_window: None,
            default_thinking_budget: Some(4_096),
            device_ids: std::sync::Mutex::new(std::collections::HashMap::new()),
            token_store: Arc::new(token_store),
            built: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(any(debug_assertions, feature = "mock-provider"))]
            scripted: std::sync::Mutex::new(None),
        }
    }
}
/// A session's profile, resolved into everything that follows from it.
///
/// One struct with one producer ([`resolved_profile`]) because these are not independent facts:
/// they all come from the same profile and its account, and a caller that took the provider and
/// left the window behind would gauge the new model against the old one's size. Building a session
/// and switching one mid-conversation both go through it, so neither can derive a subset the other
/// does not.
#[derive(Clone)]
pub(crate) struct ResolvedProfile {
    pub(crate) provider: Arc<dyn Provider>,
    pub(crate) profile: String,
    /// What the context gauge and the auto-compaction threshold read.
    pub(crate) context_window: u64,
    /// Whether this profile's model accepts image input. The agent does not read it: admitting an
    /// attachment is the host's decision, made before a turn exists, and every host asks
    /// `ResidentSession::accepts_images`, which reads this cell once the session's recorded
    /// profile has reached the agent.
    pub(crate) vision: bool,
}
/// The profile a session runs on.
///
/// One door, so every place that runs a turn answers the question the same way. A session that
/// exists names its profile on its row and that is what it gets; anything else would move the
/// conversation to a provider it was not having, drop the reasoning it recorded (a thinking block
/// is not replayed across providers) and bill a different account.
///
/// `None` is a session that does not exist yet, which takes the configured default and records it
/// the moment its row is written.
///
/// Takes the value it decides between rather than the whole [`ResolvedConfig`], so the decision can
/// be exercised on its own. Which of a recorded profile and the process default wins is the entire
/// question here, and a door that could only be reached through a fully resolved config could not
/// be asked it directly.
pub(crate) async fn resolve_session_profile(
    store: &Store,
    // The process default, or the reason there is not one. A reason rather than a bare absence
    // because it is the only useful thing to say when this falls through: "no profile could be
    // picked" is not actionable, while "multiple profiles configured (work, side); run
    // `meka profile use <name>`" is. `validate()` does not raise it for a resume, so this is where
    // it surfaces.
    default_profile: std::result::Result<&str, &str>,
    session_id: Option<Uuid>,
) -> Result<String> {
    if let Some(session_id) = session_id
        && let Some(recorded) = store.recorded_profile(session_id).await?
    {
        return Ok(recorded);
    }
    Ok(default_profile
        .map_err(|reason| MekaError::Config(reason.to_string()))?
        .to_string())
}
/// [`resolve_session_profile`] for a caller that has a whole [`ResolvedConfig`] to hand.
pub(crate) async fn profile_for_config(
    store: &Store,
    config: &ResolvedConfig,
    session_id: Option<Uuid>,
) -> Result<String> {
    resolve_session_profile(
        store,
        // Exactly one of the two is set; see `select_profile`. The fallback text is for a
        // shape that pairing rules out rather than for a case anyone expects to hit.
        config.default_profile.as_deref().ok_or_else(|| {
            config
                .provider_error
                .as_deref()
                .unwrap_or("no profile is configured; run `meka profile add`")
        }),
        session_id,
    )
    .await
}
/// The profile a session records, when `config.toml` no longer has it.
///
/// The one failure `--profile` is the fix for, and the only one worth naming a session in a hint
/// about: a profile that is configured but unusable (no stored credential, an endpoint that
/// refuses) is not moved by repinning the row.
///
/// The recorded name is compared against the configured set and nothing else, with no test for the
/// empty one a migrated store can hold. `""` is a name that resolves to nothing, which is exactly
/// what this asks, so it answers correctly without this function having to know where it came
/// from.
///
/// The name itself is not returned, because nothing needs it: the refusal already printed names the
/// profile, and the hint this gates adds only the repin command.
///
/// A read failure answers `false`: this runs only to decorate an error that has already been
/// printed, and failing the process over the decoration would replace a useful message with a
/// useless one.
pub(crate) async fn recorded_profile_is_gone(
    store: &Store,
    config: &ResolvedConfig,
    session_id: Uuid,
) -> bool {
    match store.recorded_profile(session_id).await {
        Ok(Some(profile)) => !config.profiles.contains_key(&profile),
        Ok(None) => false,
        Err(error) => {
            tracing::debug!(
                "failed to read session {session_id}'s recorded profile for the setup hint: {error}"
            );
            false
        }
    }
}
/// Turn a session's profile into the provider it names and the per-profile facts that come with it.
///
/// The one producer of [`ResolvedProfile`], so building a session and moving one mid-conversation
/// cannot disagree about what a profile means. Read once per process from the *default* profile
/// instead, a session pinned to a 32k profile gauges itself against the default's window, so
/// auto-compaction never fires and the provider rejects the turn.
pub(crate) async fn resolved_profile(
    providers: &ProviderRegistry,
    profile: String,
) -> Result<ResolvedProfile> {
    let (provider, settings) = providers.resolve(&profile).await?;
    Ok(ResolvedProfile {
        provider,
        // The documented default, not a guess at the model: meka does not infer a window from a
        // model name, so a profile that states none gets the one value the docs name.
        context_window: settings.context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
        vision: settings.vision,
        profile,
    })
}
/// A session's context window, answered without building its provider.
///
/// For a host that reports occupancy without reaching through the runtime mutex an in-flight turn
/// is holding. Same source, so the reported window is the one the agent gauges against.
///
/// `None` for a profile that cannot resolve, which is not the same as the documented default: that
/// session's next turn is going to be refused by name, and answering `1000000` beside a refusal
/// invites a client to divide by a number meka has no reason to believe.
pub(crate) fn profile_context_window(providers: &ProviderRegistry, profile: &str) -> Option<u64> {
    providers
        .settings(profile)
        .ok()
        .map(|settings| settings.context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW))
}

/// What a session runs on, published for the collaborators that outlive a single turn.
///
/// [`crate::agent::Agent::set_provider`] is the only writer, which is why this is a handle rather
/// than a second copy of the truth: it is the same arrangement `SharedPermission` and the
/// context-token counter already use for values the agent owns and others must watch.
///
/// It exists because a mid-session switch has to reach two things the agent does not own.
/// `agent_spawn` and `agent_followup` build an unpinned sub-agent from the parent's provider, and
/// the `context_*` tools report the window the model is being gauged against. A copy taken when the
/// session was assembled would be left behind by `/profile`, `PATCH /v1/sessions/{id}` and ACP's
/// `session/set_config_option`: a sub-agent spawned afterwards would run on, and bill, the profile
/// the user had just left, while the child's own row recorded the new one.
#[derive(Clone)]
pub(crate) struct PublishedProfile {
    resolved: Arc<std::sync::RwLock<ResolvedProfile>>,
    /// Held separately from `resolved` because `ContextGauge` reads it on every `context_check`
    /// and has no business knowing what a provider is.
    window: Arc<std::sync::atomic::AtomicU64>,
}
impl PublishedProfile {
    /// `window` is supplied rather than made here, for the reason
    /// [`crate::session::SessionCells`] gives about its `context_tokens`: a frontend gauge (the
    /// REPL prompt indicator, ACP's `usage_update`) is built before the agent exists and has to
    /// hold the same cell. Made internally, each host would keep a second copy and re-store it by
    /// hand beside every `set_provider` call. The caller's seed value is irrelevant; this
    /// overwrites it.
    ///
    /// Two hosts pass a throwaway instead. `serve`'s `SessionEntry` is built *after* the agent, so
    /// it goes the other way and reads the window back out of the agent's cells; `--oneshot`
    /// prints one answer and exits, so nothing watches its window at all.
    pub(crate) fn new(
        resolved: &ResolvedProfile,
        window: Arc<std::sync::atomic::AtomicU64>,
    ) -> Self {
        window.store(
            resolved.context_window,
            std::sync::atomic::Ordering::Release,
        );
        Self {
            resolved: Arc::new(std::sync::RwLock::new(resolved.clone())),
            window,
        }
    }

    /// A cell nobody outside watches, for a sub-agent: it has no prompt gauge and no session entry.
    pub(crate) fn detached(resolved: &ResolvedProfile) -> Self {
        Self::new(resolved, Arc::new(std::sync::atomic::AtomicU64::new(0)))
    }

    /// A poisoned lock costs nothing here: the guarded value is one owner's `ResolvedProfile`, and
    /// a writer that panicked mid-store left it whole either way.
    pub(crate) fn current(&self) -> ResolvedProfile {
        crate::sync::read(&self.resolved).clone()
    }

    /// The handle [`crate::tools::context::ContextGauge`] holds.
    pub(crate) fn window(&self) -> Arc<std::sync::atomic::AtomicU64> {
        Arc::clone(&self.window)
    }

    /// The current window, for a reader that wants the number rather than the cell.
    pub(crate) fn context_window(&self) -> u64 {
        self.window.load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn store(&self, resolved: &ResolvedProfile) {
        *crate::sync::write(&self.resolved) = resolved.clone();
        self.window.store(
            resolved.context_window,
            std::sync::atomic::Ordering::Release,
        );
    }
}

/// A provider for a session that never asks the model anything: the reference registry `meka
/// tool list` builds, and a gate probe's. Every request is refused, so a call reaching it is a
/// bug that surfaces rather than a silent hang.
struct Unbound;

#[async_trait]
impl Provider for Unbound {
    async fn complete(
        &self,
        _request: CompletionRequest<'_>,
        _cancellation: CancellationToken,
    ) -> Result<Completion> {
        Err(MekaError::Provider(
            "no provider is bound to this session".to_string(),
        ))
    }

    async fn stream(
        &self,
        _request: CompletionRequest<'_>,
        _event_sender: mpsc::Sender<StreamEvent>,
        _cancellation: CancellationToken,
    ) -> Result<()> {
        Err(MekaError::Provider(
            "no provider is bound to this session".to_string(),
        ))
    }
}

impl PublishedProfile {
    /// A cell bound to nothing, for a session that only lists its tools.
    pub(crate) fn unbound() -> Self {
        Self::detached(&ResolvedProfile {
            provider: Arc::new(Unbound),
            profile: String::new(),
            context_window: 0,
            vision: false,
        })
    }
}
