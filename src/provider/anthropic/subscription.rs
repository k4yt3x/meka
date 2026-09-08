//! `claude-subscription`: the Anthropic Messages API billed to a Claude subscription.
//!
//! Uses Claude Code's attestation and billing-header machinery to send requests as the official
//! CLI, and manages OAuth token refresh against the Claude token endpoint. The protocol itself is
//! shared with the API-key [`super::messages`] backend through [`super::shared`]; what is
//! particular here is the credential, the client persona, and the beta headers that persona
//! implies.

mod attestation;

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::shared::{
    self, DEFAULT_EFFORT, convert_messages_to_claude_content, convert_tools_to_claude_tools,
    model_is_haiku, model_supports_effort, model_supports_mid_conversation_system,
    model_supports_modern_features, model_supports_temperature,
};
use crate::{
    config::ThinkingMode,
    conversation::Message,
    error::{MekaError, Result},
    provider::{
        AccountIdentity, AccountUsage, CompletionRequest, DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID,
        ExtraUsage, Provider, StreamEvent, ThinkingOverride, ToolDefinition, UsageHistory,
        UsageWindow,
    },
    store::{AuthCredential, TokenStore},
};

/// Claude Code system prompt prefix.
const CC_SYSTEM_PROMPT_PREFIX: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

/// The `claude-subscription` backend: one profile's model and the OAuth credential it bills.
/// The window at which a profile asks for the 1M-context beta.
const CONTEXT_1M_TOKENS: u64 = 1_000_000;

/// The two values `thinking.display` can carry on the wire; the `redacted` mode sends none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireThinkingDisplay {
    /// A running token count in place of the text, under the display-updates beta.
    Updates,
    /// A short readable summary of the reasoning.
    Summarized,
}

impl WireThinkingDisplay {
    const fn name(self) -> &'static str {
        match self {
            Self::Updates => "updates",
            Self::Summarized => "summarized",
        }
    }
}

pub(crate) struct ClaudeSubscriptionProvider {
    client: reqwest::Client,
    credential: tokio::sync::RwLock<AuthCredential>,
    /// Serializes refreshes without blocking readers. Held across the database and network awaits
    /// a refresh performs; `credential` is not.
    refresh_gate: tokio::sync::Mutex<()>,
    base_url: String,
    model: String,
    client_id: String,
    oauth_token_url: String,
    token_store: Option<Arc<TokenStore>>,
    /// Account name this provider's credential is stored under, so refreshed tokens are written
    /// back to the correct `account_credentials` row.
    credential_key: String,
    session_id: String,
    device_id: String,
    /// The subscriber's account UUID, sent as `metadata.user_id.account_uuid` (matching Claude
    /// Code). Captured from the credential at construction; empty for pre-existing logins until a
    /// refresh persists it and the session restarts, or the user re-logs in.
    account_uuid: String,
    thinking: ThinkingMode,
    thinking_budget_tokens: u64,
    /// The access token a request site got a 401 for, read by every `ensure_valid_credential`
    /// until a refresh installs a replacement.
    ///
    /// A rejection is the backend saying the stored expiry is wrong: the token was revoked, or
    /// its issuer shortened lifetimes. The expiry alone would keep presenting the dead token until
    /// it passed, which can be hours or, for an expiry meka had to assume, a full lifetime. The
    /// token's identity rather than a flag, because a flag was spent by whichever read came first:
    /// two requests refused in the same window left one of them re-sending the dead bearer and
    /// failing with the login remedy while the other's refresh succeeded beside it. A second
    /// rejection of the replacement is the account, not the token.
    rejected_access_token: std::sync::Mutex<Option<String>>,
    /// The settled `output_config.effort` for the request body, resolved once at construction: the
    /// profile's value if it set one, otherwise [`DEFAULT_EFFORT`]. `None` only where the model
    /// takes no effort at all, and then the `effort-2025-11-24` beta is withheld too: both read
    /// this one slot, so they stay in lockstep the way Claude Code's `KHE` keeps them.
    resolved_effort: Option<String>,
    /// When true, request `redacted_thinking` blocks via the `redact-thinking-2026-02-12` beta
    /// header.
    thinking_display: crate::config::ThinkingDisplay,
    /// The profile's window; a million or more asks for the 1M-context beta.
    context_window: Option<u64>,
    /// Per-request output token cap from the profile; `None` keeps the built-in default.
    max_output_tokens: Option<u64>,
    /// See [`crate::config::ProfileConfig::max_request_bytes`].
    max_request_bytes: Option<usize>,
}

impl ClaudeSubscriptionProvider {
    /// Build from a profile's settings, whose credential the builder has checked is an OAuth one.
    pub(crate) fn new(settings: crate::provider::ProviderBuilder) -> Result<Self> {
        let credential_key = settings.resolve_credential_key()?;
        let crate::provider::ProviderBuilder {
            credential,
            model,
            base_url,
            client_id,
            oauth_token_url,
            token_store,
            thinking,
            thinking_budget_tokens,
            device_id,
            effort,
            thinking_display,
            context_window,
            max_output_tokens,
            max_request_bytes,
            ..
        } = settings;
        let account_uuid = match &credential {
            AuthCredential::OAuthToken { account_id, .. } => account_id.clone().unwrap_or_default(),
            AuthCredential::ApiKey(_) => String::new(),
        };
        let configured_effort = crate::provider::resolve_effort_level(effort.as_deref());
        // Settled once, because it is a property of the profile and the model rather than of a
        // request. A model that takes no effort drops a configured one rather than earning a 400
        // for it, which is what Claude Code's `KHE` does, and says so once instead of per turn.
        let resolved_effort = if model_supports_effort(&model) {
            Some(configured_effort.unwrap_or_else(|| DEFAULT_EFFORT.to_string()))
        } else {
            if let Some(configured) = &configured_effort {
                tracing::warn!(
                    "model '{model}' takes no reasoning effort; ignoring the profile's effort = '{configured}'"
                );
            }
            None
        };
        Ok(Self {
            client: crate::provider::build_http_client("claude-subscription", |builder| builder)?,
            credential: tokio::sync::RwLock::new(credential),
            refresh_gate: tokio::sync::Mutex::new(()),
            base_url: super::shared::normalize_claude_base_url(
                base_url
                    .as_deref()
                    .unwrap_or(crate::provider::DEFAULT_ANTHROPIC_BASE_URL),
            ),
            model,
            client_id: client_id
                .unwrap_or_else(|| DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID.to_string()),
            oauth_token_url: oauth_token_url
                .unwrap_or_else(|| "https://api.anthropic.com/v1/oauth/token".to_string()),
            token_store,
            credential_key,
            session_id: Uuid::new_v4().to_string(),
            device_id,
            account_uuid,
            thinking,
            thinking_budget_tokens,
            rejected_access_token: std::sync::Mutex::new(None),
            resolved_effort,
            thinking_display,
            context_window,
            max_output_tokens,
            max_request_bytes,
        })
    }

    fn effective_thinking(&self, thinking: ThinkingOverride) -> ThinkingMode {
        shared::effective_thinking(thinking, self.thinking)
    }

    /// The settled effort to send as `output_config.effort` (see [`Self::resolved_effort`]). The
    /// `effort-2025-11-24` beta gate in [`Self::compute_betas`] reads the same value, keeping the
    /// body field and the beta header in lockstep.
    fn wire_effort(&self) -> Option<String> {
        self.resolved_effort.clone()
    }

    /// Whether this model takes `output_config.effort` at all, which is also what decides the
    /// `effort-2025-11-24` beta. Reads [`Self::resolved_effort`], which is `None` only in that
    /// case, so the two can never disagree.
    fn model_takes_effort(&self) -> bool {
        self.resolved_effort.is_some()
    }

    /// Mirrors Claude Code 2.1.263's beta assembly for a first-party OAuth turn, validated against
    /// a live capture of its interactive CLI: opus-5 with tools, thinking on and display updates,
    /// fourteen betas in this order.
    ///
    /// `context-1m-2025-08-07` is sent when the profile's `context_window` is a million tokens or
    /// more. Claude Code sends it for the `[1m]` model variant its user selected, and the profile's
    /// window is where a meka user states the same choice.
    ///
    /// `redact-thinking-2026-02-12` and `thinking-display-updates-2026-08-18` are the two halves
    /// of Claude Code's display switch; see [`Self::sends_redaction_beta`] and
    /// [`Self::wire_thinking_display`].
    ///
    /// `has_tools` gates `advanced-tool-use-2025-11-20`. Claude Code's own gate is narrower (it
    /// sends that beta when its *tool search* is active rather than merely when tools are present),
    /// but tool search is on for every agentic CLI turn, so the wire is the same, and meka has
    /// tools on every turn anyway.
    ///
    /// `fallback-credit-2026-06-01` is sent unconditionally. Claude Code latches it whenever a
    /// model is visible in its UI, which is every interactive turn, and it only advertises that the
    /// server may answer with a fallback credit; meka sends no `fallbacks` or
    /// `fallback_credit_token` of its own, exactly like the captured turns that carry the beta.
    ///
    /// `cache-diagnosis-2026-04-07` is sent unconditionally too, paired with the body's
    /// `diagnostics.previous_message_id`.
    fn compute_betas(&self, has_tools: bool, thinking_on: bool) -> Option<String> {
        let model = self.model.as_str();
        let mut parts: Vec<&'static str> = Vec::with_capacity(14);

        if !model_is_haiku(model) {
            parts.push("claude-code-20250219");
        }
        parts.push("oauth-2025-04-20");
        if self.asks_for_1m_context() {
            parts.push("context-1m-2025-08-07");
        }

        if model_supports_modern_features(model) {
            parts.push("interleaved-thinking-2025-05-14");
            if self.sends_redaction_beta(thinking_on) {
                parts.push("redact-thinking-2026-02-12");
            }
            parts.push("thinking-token-count-2026-05-13");
            parts.push("context-management-2025-06-27");
        }

        parts.push("prompt-caching-scope-2026-01-05");

        if model_supports_mid_conversation_system(model) {
            parts.push("mid-conversation-system-2026-04-07");
        }

        if has_tools {
            parts.push("advanced-tool-use-2025-11-20");
        }

        // Keep the beta in lockstep with the body field: both read the same settled effort slot, so
        // the beta fires exactly when `output_config.effort` will be sent, which is on every model
        // that takes an effort at all. Only a model that takes none advertises neither.
        if self.model_takes_effort() {
            parts.push("effort-2025-11-24");
        }

        parts.push("fallback-credit-2026-06-01");
        if self.wire_thinking_display(thinking_on) == Some(WireThinkingDisplay::Updates) {
            parts.push("thinking-display-updates-2026-08-18");
        }
        parts.push("extended-cache-ttl-2025-04-11");
        parts.push("cache-diagnosis-2026-04-07");

        Some(parts.join(","))
    }

    /// Whether the profile asks for the 1M context window: a `context_window` of
    /// [`CONTEXT_1M_TOKENS`] or more.
    fn asks_for_1m_context(&self) -> bool {
        self.context_window
            .is_some_and(|window| window >= CONTEXT_1M_TOKENS)
    }

    /// The `display` the thinking field carries, or `None` where Claude Code sends none: thinking
    /// off, a model that cannot think, or the `redacted` mode.
    fn wire_thinking_display(&self, thinking_on: bool) -> Option<WireThinkingDisplay> {
        if !thinking_on || !model_supports_modern_features(&self.model) {
            return None;
        }
        match self.thinking_display {
            crate::config::ThinkingDisplay::Updates => Some(WireThinkingDisplay::Updates),
            crate::config::ThinkingDisplay::Summarized => Some(WireThinkingDisplay::Summarized),
            crate::config::ThinkingDisplay::Redacted => None,
        }
    }

    /// Claude Code pushes the redaction beta for every model that can think unless summaries are
    /// on, then removes it again on the turns where display updates apply, which is every turn
    /// with thinking on. Collapsed here to what reaches the wire.
    fn sends_redaction_beta(&self, thinking_on: bool) -> bool {
        match self.thinking_display {
            crate::config::ThinkingDisplay::Redacted => true,
            crate::config::ThinkingDisplay::Summarized => false,
            crate::config::ThinkingDisplay::Updates => !thinking_on,
        }
    }

    /// Resolve a valid Authorization header, refreshing the OAuth token if it's within 5 minutes of
    /// expiry.
    ///
    /// Concurrency contract (relevant under multi-session ACP where two sessions may call this in
    /// parallel): `refresh_gate` serializes refreshers, and `credential` is held only across the
    /// reads and writes themselves, never across an await on the network or the database. Two tasks
    /// that both observe an expiring token queue on the gate; the loser re-checks after acquiring
    /// it and finds the winner's fresh token. Exactly one refresh API call fires under
    /// contention and both callers return a valid token.
    ///
    /// The gate is what makes that true. With the `credential` write lock as the gate instead,
    /// every *reader* queues behind the refresh too, so a provider endpoint that accepts the
    /// connection and then goes silent wedges every session in the process, not just the one
    /// refreshing. A stalled refresh blocks only another refresh, and the bounded HTTP timeout
    /// ends even that.
    async fn ensure_valid_credential(&self) -> Result<(&'static str, String)> {
        // Compared, not consumed: the rejection stands until a refresh replaces the token it
        // names, so a second request carrying the same refused bearer refreshes too instead of
        // finding a flag the first one spent.
        let refused = crate::sync::lock(&self.rejected_access_token).clone();
        let (rejected, entry_access_token) = {
            let credential = self.credential.read().await;
            match &*credential {
                AuthCredential::ApiKey(_) => {
                    return Err(MekaError::Provider(
                        "claude-subscription requires an OAuth token, not an API key".to_string(),
                    ));
                }
                AuthCredential::OAuthToken {
                    access_token,
                    expires_at,
                    refresh_token,
                    ..
                } => {
                    let rejected = refused.as_deref() == Some(access_token.as_str());
                    if !rejected
                        && !crate::oauth::oauth_needs_refresh(
                            *expires_at,
                            refresh_token.is_some(),
                            crate::oauth::now_epoch_millis(),
                        )
                    {
                        return Ok(("Authorization", crate::text::bearer(access_token)));
                    }
                    (rejected, access_token.clone())
                }
            }
        };

        // Token expired: attempt refresh. Only refreshers queue here; readers are untouched.
        let _refreshing = self.refresh_gate.lock().await;

        // And the same thing one layer out. `refresh_gate` is a `tokio::sync::Mutex`, so it
        // serializes the tasks in *this* process and says nothing about the meka in the next
        // terminal, which is holding the same refresh token and is just as due. Bounded, and
        // advisory: the compare-and-swap on the write is what makes the outcome correct whether or
        // not this is held.
        let _across_processes = match &self.token_store {
            Some(store) => crate::oauth::await_credential_lock(store, &self.credential_key).await,
            None => None,
        };

        // Re-read the latest credential from the store. Refresh tokens rotate on each successful
        // refresh, and a sibling meka process may have rotated this one since startup; without
        // the re-read a stale refresh token is posted and the issuer rejects it with
        // `invalid_grant`.
        //
        // The store call is awaited with no credential lock held, and the result installed under a
        // write lock that spans an assignment and nothing else.
        //
        // Installed only when it is at least as new as what memory holds. The row is behind in one
        // case, a refresh in this process whose persist failed: adopting it would spend a refresh
        // token the issuer has already retired while the live one sits here. The row's version is
        // kept either way: the refresh below replaces the row on it, so a stale row catches up.
        let mut observed_version = None;
        if let Some(store) = &self.token_store {
            match store
                .load_account_credential_versioned(&self.credential_key)
                .await
            {
                Ok(Some(latest)) => {
                    let crate::store::StoredCredential {
                        credential: latest,
                        version,
                    } = latest;
                    observed_version = Some(version);
                    let mut credential = self.credential.write().await;
                    if crate::oauth::row_is_at_least_as_new(&credential, &latest) {
                        *credential = latest;
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!("failed to re-read Claude OAuth token before refresh: {error}");
                }
            }
        }

        // Double-check after the DB re-read: another task or process may have already rotated and
        // persisted a new access token that is still valid.
        let derived_from = {
            let credential = self.credential.read().await;
            if let AuthCredential::OAuthToken {
                access_token,
                expires_at,
                refresh_token,
                ..
            } = &*credential
            {
                // After a rejection the expiry is not trusted, but a token that is not the refused
                // one is: a sibling process may already have rotated past it.
                let usable = if rejected {
                    *access_token != entry_access_token
                } else {
                    !crate::oauth::oauth_needs_refresh(
                        *expires_at,
                        refresh_token.is_some(),
                        crate::oauth::now_epoch_millis(),
                    )
                };
                if usable {
                    return Ok(("Authorization", crate::text::bearer(access_token)));
                }
            }
            credential.clone()
        };
        let (refresh_token, prior_account_id) = match &derived_from {
            AuthCredential::OAuthToken {
                refresh_token,
                account_id,
                ..
            } => (refresh_token.clone(), account_id.clone()),
            AuthCredential::ApiKey(_) => (None, None),
        };

        // With nothing to refresh with, a rejected or expired token has one remedy, which every
        // exit from the refresh path names.
        let Some(refresh_token) = refresh_token else {
            return Err(crate::oauth::with_login_remedy(
                MekaError::Provider(
                    "OAuth access token expired and no refresh token available".to_string(),
                ),
                &self.credential_key,
            ));
        };

        let refreshed = self
            .refresh_oauth_token(&refresh_token, prior_account_id)
            .await?;

        // A refresh rotates the refresh token, so the one in the database is now dead, but only
        // if the database still holds the one this was derived from. Where it does not, what comes
        // back is the newer credential to use instead of this one.
        let new_credential = match &self.token_store {
            Some(store) => {
                crate::oauth::store_refreshed_credential(
                    store,
                    &self.credential_key,
                    observed_version.as_deref(),
                    refreshed,
                )
                .await
            }
            None => refreshed,
        };

        // Re-checked, because `new_credential` need not be the one this refresh minted: a swap the
        // row has moved past hands back what the row holds instead, and the check at the top of
        // this function saw the credential as it was on entry.
        //
        // Unreachable as things stand (`store_refreshed_credential` adopts only a credential of
        // the same kind this refresh was derived from, and that is an `OAuthToken` on every path
        // that reaches here), and kept for what it costs if that stops being true: `auth_header`
        // would otherwise put an API key in an `x-api-key` header on a subscription endpoint that
        // does not take one, which is the shape this backend refuses outright a hundred lines
        // above. The Codex provider has the same guard.
        let AuthCredential::OAuthToken { .. } = &new_credential else {
            return Err(MekaError::Provider(
                "claude-subscription requires an OAuth token, not an API key".to_string(),
            ));
        };

        let (header_name, header_value) = new_credential.auth_header();
        *self.credential.write().await = new_credential;
        // The refused token is gone, whatever the issuer minted; a rejection of the replacement is
        // recorded afresh by the request that meets it.
        crate::sync::lock(&self.rejected_access_token).take();
        Ok((header_name, header_value))
    }

    /// Record that the backend refused the current access token, so every credential read until
    /// it is replaced refreshes it instead of trusting the stored expiry.
    async fn note_credential_rejected(&self) {
        tracing::warn!(
            "claude-subscription rejected the access token; refreshing it and retrying once"
        );
        let refused = match &*self.credential.read().await {
            AuthCredential::OAuthToken { access_token, .. } => Some(access_token.clone()),
            AuthCredential::ApiKey(_) => None,
        };
        *crate::sync::lock(&self.rejected_access_token) = refused;
    }

    /// GET one of the OAuth account endpoints (usage, profile, history) as text.
    ///
    /// The three share everything but the path and the word in their error messages, and each
    /// needs the same one retry after a 401 that a completion gets. `what` names the call in the
    /// transport and read errors, which is how a user tells a usage probe from a profile read.
    async fn fetch_oauth_endpoint(&self, path: &str, what: &str) -> Result<String> {
        let response = crate::oauth::send_with_one_refresh(
            self,
            crate::error::ProviderRequest::Auxiliary,
            |error| {
                crate::error::provider_transport_error(
                    &format!("{what} request failed"),
                    error,
                    None,
                )
            },
            || async {
                let (auth_name, auth_value) = self.ensure_valid_credential().await?;
                Ok(attestation::apply_headers(
                    self.client.get(format!("{}{}", self.base_url, path)),
                    auth_name,
                    &auth_value,
                    &self.session_id,
                    Some("oauth-2025-04-20"),
                ))
            },
        )
        .await?;
        let status = response.status();
        let retry_after = crate::error::parse_retry_after(response.headers());
        let text = response.text().await.map_err(|error| {
            crate::error::provider_transport_error(
                &format!("failed to read {what} response"),
                &error,
                retry_after,
            )
        })?;
        if !status.is_success() {
            return Err(crate::error::provider_http_error(
                status,
                &text,
                retry_after,
                crate::error::ProviderRequest::Auxiliary,
            ));
        }
        Ok(text)
    }

    async fn refresh_oauth_token(
        &self,
        refresh_token: &str,
        prior_account_id: Option<String>,
    ) -> Result<AuthCredential> {
        tracing::info!("refreshing Claude OAuth token");

        #[derive(Deserialize)]
        struct RefreshResponse {
            access_token: String,
            refresh_token: Option<String>,
            expires_in: Option<u64>,
            account: Option<RefreshAccount>,
        }

        #[derive(Deserialize)]
        struct RefreshAccount {
            uuid: String,
        }

        // Only the shape of the answer is this backend's own; posting the grant and judging what
        // came back is the same act for every issuer, and lives in one place so it cannot drift
        // between the two subscription backends (see `crate::oauth::exchange_refresh_token`).
        let data: RefreshResponse =
            crate::oauth::exchange_refresh_token(crate::oauth::RefreshExchange {
                client: &self.client,
                token_url: &self.oauth_token_url,
                client_id: &self.client_id,
                refresh_token,
                account: &self.credential_key,
                context: "OAuth token refresh",
            })
            .await?;

        // An absent or nonsensical `expires_in` gets the assumed lifetime rather than `None` or
        // "never". `None` reads as due, so a token whose issuer never states an expiry would send
        // every later request back through this whole path, rotating the refresh token each time.
        // And a far-future stamp pins whatever the issuer actually minted for the rest of the
        // process: the 401 that would correct it forces one refresh, but a bounded guess is what
        // makes that the exception rather than the only path off a dead token.
        let now = crate::oauth::now_epoch_millis();
        let expires_at = Some(data.expires_in.map_or_else(
            || crate::oauth::oauth_assumed_expiry(now),
            |seconds| {
                // `try_from` rather than `as`: the cast wraps, and it happens *before* the
                // `checked_mul` that makes this saturating, so an `expires_in` past `i64::MAX`
                // would produce a negative and land the expiry in the past, the exact
                // refresh-every-request loop the comment above says this avoids.
                i64::try_from(seconds)
                    .ok()
                    .and_then(|seconds| seconds.checked_mul(1000))
                    .map_or_else(
                        || crate::oauth::oauth_assumed_expiry(now),
                        |millis| now.saturating_add(millis),
                    )
            },
        ));

        Ok(AuthCredential::OAuthToken {
            access_token: data.access_token,
            refresh_token: data
                .refresh_token
                .or_else(|| Some(refresh_token.to_string())),
            expires_at,
            // The freshly returned account wins, but an account already known is never blanked.
            account_id: data
                .account
                .map(|account| account.uuid)
                .or(prior_account_id),
        })
    }

    /// The request body in Claude Code's key order, carrying the `cch=00000` placeholder that
    /// [`attestation::patch_request_body`] fills after serialization.
    pub(super) fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
        thinking: ThinkingOverride,
        attribution: &crate::provider::Attribution,
    ) -> serde_json::Value {
        let claude_messages =
            convert_messages_to_claude_content(messages, super::shared::CacheBreakpoint::OneHour);

        let metadata_user_id = serde_json::json!({
            "device_id": self.device_id,
            "account_uuid": self.account_uuid,
            "session_id": self.session_id,
        })
        .to_string();

        // Keys go in Claude Code's order, which `serde_json`'s `preserve_order` feature carries
        // through to the wire:
        //
        //     model, messages, system, tools, metadata, max_tokens, thinking,
        //     [temperature], [context_management], [output_config], diagnostics, stream
        //
        // Nothing depends on this ordering, which is the point: the attestation locates its
        // placeholder structurally (see [`attestation::patch_request_body`]) rather than by
        // assuming `system` comes first, so the order is free to match the capture exactly.
        let mut body = serde_json::Map::new();

        body.insert("model".to_string(), serde_json::json!(self.model));
        body.insert("messages".to_string(), serde_json::json!(claude_messages));

        if !system_prompt.is_empty() {
            let billing_header = attestation::generate_billing_header(messages, attribution);
            // Matches Claude Code's wire shape: only the user system prompt carries
            // `cache_control`, the billing header and identity prefix are unmarked, and the `1h`
            // ttl is what an OAuth subscriber's turn carries. `scope: "global"` mirrors the
            // captured CLI breakpoint (the `prompt-caching-scope-2026-01-05` beta), sharing the
            // cached prefix across sessions.
            body.insert(
                "system".to_string(),
                serde_json::json!([
                    {
                        "type": "text",
                        "text": billing_header
                    },
                    {
                        "type": "text",
                        "text": CC_SYSTEM_PROMPT_PREFIX
                    },
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": { "type": "ephemeral", "ttl": "1h", "scope": "global" }
                    }
                ]),
            );
        }

        if !tools.is_empty() {
            body.insert(
                "tools".to_string(),
                serde_json::json!(convert_tools_to_claude_tools(tools)),
            );
        }

        body.insert(
            "metadata".to_string(),
            serde_json::json!({ "user_id": metadata_user_id }),
        );

        shared::insert_thinking_fields(
            &mut body,
            self.effective_thinking(thinking),
            self.thinking_budget_tokens,
            self.max_output_tokens,
            self.wire_thinking_display(self.effective_thinking(thinking).is_on())
                .map(WireThinkingDisplay::name),
        );

        // Claude Code sends `temperature: 1` only when thinking is off AND the model is on the
        // sampling-params allowlist (see `model_supports_temperature`). Opus 4.7+, the 5 line, and
        // Fable/Mythos reject `temperature` with a 400.
        if !self.effective_thinking(thinking).is_on() && model_supports_temperature(&self.model) {
            body.insert("temperature".to_string(), serde_json::json!(1));
        }

        // Mirrors `Yph`, which returns exactly this edit when thinking is on and nothing
        // otherwise: it preserves thinking blocks across previous assistant turns.
        if self.effective_thinking(thinking).is_on() && model_supports_modern_features(&self.model)
        {
            body.insert(
                "context_management".to_string(),
                serde_json::json!({
                    "edits": [{ "type": "clear_thinking_20251015", "keep": "all" }]
                }),
            );
        }

        if let Some(effort) = self.wire_effort() {
            body.insert(
                "output_config".to_string(),
                serde_json::json!({ "effort": effort }),
            );
        }

        // Only a conversation turn carries the field, `null` on its first request and after a
        // resume: Claude Code sends nothing here on its side queries, and meka's compaction is one.
        if attribution.is_conversation_turn() {
            body.insert(
                "diagnostics".to_string(),
                serde_json::json!({ "previous_message_id": attribution.previous_message_id() }),
            );
        }

        body.insert("stream".to_string(), serde_json::json!(stream));

        serde_json::Value::Object(body)
    }
}

#[async_trait]
impl crate::oauth::RefreshesCredential for ClaudeSubscriptionProvider {
    /// Sent once more after a 401, on a credential refreshed for the purpose; see
    /// `rejected_access_token`. A second refusal gets the login remedy instead.
    async fn refresh_after_rejection(&self) -> bool {
        self.note_credential_rejected().await;
        true
    }

    fn with_login_remedy(&self, error: MekaError) -> MekaError {
        crate::oauth::with_login_remedy(error, &self.credential_key)
    }
}

#[async_trait]
impl shared::ClaudeBackend for ClaudeSubscriptionProvider {
    fn client(&self) -> &reqwest::Client {
        &self.client
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/messages?beta=true", self.base_url)
    }

    fn max_request_bytes(&self) -> usize {
        self.max_request_bytes
            .unwrap_or(super::shared::MAX_REQUEST_BYTES)
    }

    fn request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
        thinking: ThinkingOverride,
        attribution: &crate::provider::Attribution,
    ) -> serde_json::Value {
        self.build_request_body(
            system_prompt,
            messages,
            tools,
            stream,
            thinking,
            attribution,
        )
    }

    fn finish_body(&self, system_prompt: &str, body_json: String) -> Result<String> {
        if system_prompt.is_empty() {
            Ok(body_json)
        } else {
            attestation::patch_request_body(&body_json)
        }
    }

    async fn authenticated_request(
        &self,
        request: reqwest::RequestBuilder,
        has_tools: bool,
        _stream: bool,
        thinking: ThinkingOverride,
    ) -> Result<reqwest::RequestBuilder> {
        let (auth_header_name, auth_header_value) = self.ensure_valid_credential().await?;
        Ok(attestation::apply_headers(
            request,
            auth_header_name,
            &auth_header_value,
            &self.session_id,
            self.compute_betas(has_tools, self.effective_thinking(thinking).is_on())
                .as_deref(),
        ))
    }

    fn remember_request_id(
        &self,
        attribution: &crate::provider::Attribution,
        headers: &reqwest::header::HeaderMap,
    ) {
        remember_request_id(attribution, headers);
    }
}

#[async_trait]
impl Provider for ClaudeSubscriptionProvider {
    async fn complete(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<crate::provider::Completion> {
        shared::complete(self, request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest<'_>,
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        shared::stream(self, request, event_sender, cancellation).await
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }

    async fn fetch_usage(&self) -> Result<Option<AccountUsage>> {
        // Reuse the full Claude Code header set; the `oauth-2025-04-20` beta is what unlocks the
        // usage endpoint for OAuth tokens.
        let response_text = self
            .fetch_oauth_endpoint("/api/oauth/usage", "usage")
            .await?;
        let parsed: OAuthUsageResponse = serde_json::from_str(&response_text)
            .map_err(|error| MekaError::Provider(format!("invalid usage JSON: {error}")))?;
        Ok(Some(parsed.into_account_usage()))
    }

    async fn fetch_identity(&self) -> Result<Option<AccountIdentity>> {
        // Required: the profile (identity + plan/tier/org).
        let profile_text = self
            .fetch_oauth_endpoint("/api/oauth/profile", "profile")
            .await?;
        // The credential the profile read succeeded with, for the best-effort call below.
        let (auth_name, auth_value) = self.ensure_valid_credential().await?;
        let profile: OAuthProfileResponse = serde_json::from_str(&profile_text)
            .map_err(|error| MekaError::Provider(format!("invalid profile JSON: {error}")))?;

        // Best-effort: the org/workspace role. A failure here (missing scope, network) must not
        // sink the whole command, so any error degrades `role` to `None`.
        let role = {
            let request = attestation::apply_headers(
                self.client
                    .get(format!("{}/api/oauth/claude_cli/roles", self.base_url)),
                auth_name,
                &auth_value,
                &self.session_id,
                Some("oauth-2025-04-20"),
            );
            match request.send().await {
                Ok(response) if response.status().is_success() => response
                    .text()
                    .await
                    .ok()
                    .and_then(|text| serde_json::from_str::<OAuthRolesResponse>(&text).ok())
                    .and_then(|roles| roles.organization_role),
                _ => None,
            }
        };

        Ok(Some(profile.into_identity(role)))
    }

    async fn fetch_history(&self) -> Result<Option<UsageHistory>> {
        // Anthropic exposes only a first-used date, not the rich Codex-style stats.
        let text = self
            .fetch_oauth_endpoint("/api/organization/claude_code_first_token_date", "history")
            .await?;
        #[derive(Deserialize)]
        struct FirstTokenDate {
            first_token_date: Option<String>,
        }
        let parsed: FirstTokenDate = serde_json::from_str(&text)
            .map_err(|error| MekaError::Provider(format!("invalid history JSON: {error}")))?;
        Ok(Some(UsageHistory {
            lifetime_tokens: None,
            peak_daily_tokens: None,
            current_streak_days: None,
            longest_streak_days: None,
            first_used: parsed.first_token_date,
            daily: Vec::new(),
        }))
    }
}

/// The subset of Anthropic's `GET /api/oauth/usage` body that is rendered. The live response
/// carries many more (feature-flagged, mostly-null) buckets plus a newer `limits[]` array; only the
/// stable flat windows are deserialized and the rest is ignored.
#[derive(Deserialize)]
struct OAuthUsageResponse {
    five_hour: Option<OAuthRateLimit>,
    seven_day: Option<OAuthRateLimit>,
    seven_day_opus: Option<OAuthRateLimit>,
    seven_day_sonnet: Option<OAuthRateLimit>,
    extra_usage: Option<OAuthExtraUsage>,
}

#[derive(Deserialize)]
struct OAuthRateLimit {
    /// Percentage of the window consumed, `0.0..=100.0`.
    utilization: Option<f64>,
    /// RFC 3339 timestamp of the next reset.
    resets_at: Option<String>,
}

/// The `extra_usage` block from `GET /api/oauth/usage`. The credit figures are carried inline here
/// (not in a separate top-level object): `used_credits` and `monthly_limit` are both in cents (the
/// API's minor unit), so divide by 100 for dollars. Credits are billed in USD, so no currency is
/// reported. Typed `f64` because the live API sends them as JSON floats (e.g. `1832.0`), which a
/// stricter `i64` field would reject.
#[derive(Deserialize)]
struct OAuthExtraUsage {
    is_enabled: Option<bool>,
    utilization: Option<f64>,
    used_credits: Option<f64>,
    monthly_limit: Option<f64>,
}

impl OAuthUsageResponse {
    fn into_account_usage(self) -> AccountUsage {
        let mut windows = Vec::new();
        push_oauth_window(&mut windows, "5-hour (session)", self.five_hour);
        push_oauth_window(&mut windows, "Weekly", self.seven_day);
        push_oauth_window(&mut windows, "Weekly (Opus)", self.seven_day_opus);
        push_oauth_window(&mut windows, "Weekly (Sonnet)", self.seven_day_sonnet);
        AccountUsage {
            windows,
            extra_usage: oauth_extra_usage(self.extra_usage),
            note: None,
        }
    }
}

/// Hand the response's `request-id` to the conversation, so the next request can name it as
/// `cc_prev_req`.
///
/// Called the moment the response head arrives rather than after the body is read, because on the
/// streaming path the body outlives this function by the whole length of the turn, and because
/// Claude Code stamps the id onto the assistant message as soon as the request resolves, error or
/// not. A response with no `request-id` (a proxy that drops it) simply leaves the previous value in
/// place, which is what Claude Code's "last assistant message that has one" does too.
fn remember_request_id(
    attribution: &crate::provider::Attribution,
    headers: &reqwest::header::HeaderMap,
) {
    if let Some(request_id) = headers
        .get("request-id")
        .and_then(|value| value.to_str().ok())
    {
        attribution.record_request_id(request_id);
    }
}

/// Normalize Anthropic's `extra_usage` block into [`ExtraUsage`]. `used` is `used_credits` in
/// dollars; `balance` is the remaining allowance (`monthly_limit - used_credits`) in dollars.
fn oauth_extra_usage(extra_usage: Option<OAuthExtraUsage>) -> Option<ExtraUsage> {
    let extra_usage = extra_usage?;
    let used = extra_usage.used_credits.map(|cents| cents / 100.0);
    let balance = extra_usage
        .monthly_limit
        .map(|limit| (limit - extra_usage.used_credits.unwrap_or(0.0)) / 100.0);
    Some(ExtraUsage {
        enabled: extra_usage.is_enabled.unwrap_or(false),
        utilization: extra_usage.utilization,
        used,
        balance,
        currency: None,
    })
}

/// Append a window iff the bucket is present and carries a utilization figure. `resets_at` is
/// parsed from RFC 3339 into Unix seconds; an unparseable value degrades to `None`, not an error.
fn push_oauth_window(windows: &mut Vec<UsageWindow>, label: &str, limit: Option<OAuthRateLimit>) {
    if let Some(limit) = limit
        && let Some(used_percent) = limit.utilization
    {
        let resets_at = limit
            .resets_at
            .as_deref()
            .and_then(|raw| chrono::DateTime::parse_from_rfc3339(raw).ok())
            .map(|dt| dt.timestamp());
        windows.push(UsageWindow {
            label: label.to_string(),
            used_percent,
            resets_at,
        });
    }
}

/// The subset of `GET /api/oauth/profile` that is rendered. Verified live against a `claude_max`
/// account.
#[derive(Deserialize)]
struct OAuthProfileResponse {
    account: Option<OAuthProfileAccount>,
    organization: Option<OAuthProfileOrg>,
}

#[derive(Deserialize)]
struct OAuthProfileAccount {
    display_name: Option<String>,
    email: Option<String>,
}

#[derive(Deserialize)]
struct OAuthProfileOrg {
    name: Option<String>,
    /// e.g. `claude_max` / `claude_pro` / `claude_enterprise`.
    organization_type: Option<String>,
    /// e.g. `default_claude_max_20x`.
    rate_limit_tier: Option<String>,
    subscription_status: Option<String>,
}

/// Subset of `GET /api/oauth/claude_cli/roles`.
#[derive(Deserialize)]
struct OAuthRolesResponse {
    organization_role: Option<String>,
}

impl OAuthProfileResponse {
    fn into_identity(self, role: Option<String>) -> AccountIdentity {
        let (display_name, email) = self
            .account
            .map(|account| (account.display_name, account.email))
            .unwrap_or((None, None));
        let (organization, plan, tier, subscription_status) = self
            .organization
            .map(|org| {
                (
                    org.name,
                    org.organization_type,
                    org.rate_limit_tier,
                    org.subscription_status,
                )
            })
            .unwrap_or((None, None, None, None));
        AccountIdentity {
            display_name,
            email,
            plan,
            tier,
            subscription_status,
            organization,
            role,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{attestation::CC_VERSION, shared::parse_non_streaming_response, *};
    use crate::{
        conversation::{ContentBlock, Role, ToolResultContent},
        provider::StopReason,
    };

    /// The body's keys in the order they will be serialized, which is the order they go on the
    /// wire: `serde_json`'s `preserve_order` feature makes `Map` insertion-ordered, and Claude
    /// Code's own order is what meka reproduces.
    fn claude_code_key_order(body: &serde_json::Value) -> Vec<&str> {
        body.as_object()
            .expect("body is an object")
            .keys()
            .map(String::as_str)
            .collect()
    }

    fn provider_for_test() -> ClaudeSubscriptionProvider {
        provider_with_base(None)
    }

    fn provider_with_base(base_url: Option<&str>) -> ClaudeSubscriptionProvider {
        ClaudeSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ClaudeSubscription,
                AuthCredential::ApiKey("test-key".to_string()),
                "claude-sonnet-4-20250514".to_string(),
            )
            .base_url(base_url.map(str::to_string))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .thinking(ThinkingMode::Off, 10000)
            .device_id("a".repeat(64))
            .effort(Some("high".to_string()))
            .thinking_display(crate::config::ThinkingDisplay::Summarized)
            .max_output_tokens(None)
            .max_request_bytes(None),
        )
        .expect("build test provider")
    }

    /// A stream that never started is retryable on the backend a real session lost a turn to.
    ///
    /// This is the exact site of the incident: an HTTP/2 stream reset while uploading a 2 MiB body
    /// was reported as a bare `MekaError::Provider`, which the agent loop discards, so the turn
    /// ended on the first attempt.
    ///
    /// An OAuth token rather than the API key the other tests use, because this backend refuses an
    /// API key outright and would never reach the send. Its expiry is a day out so
    /// `ensure_valid_credential` takes no refresh round trip of its own first.
    #[tokio::test]
    async fn a_stream_that_could_not_start_reports_a_retryable_failure() {
        let provider = provider_on_a_dead_port();
        let (sender, _receiver) = mpsc::channel(8);
        let error = provider
            .stream(
                CompletionRequest::new("you are a test", &[Message::user("hello")], &[]),
                sender,
                CancellationToken::new(),
            )
            .await
            .expect_err("nothing is listening there");

        assert!(
            matches!(error, MekaError::RetryableProvider { .. }),
            "a stream that never started must be retryable, got: {error}"
        );
    }

    /// The account probes classify a dead endpoint the same way the turn path does.
    ///
    /// Three functions in one test because they are one property: each has its own `.send()` and
    /// response-read, and each is a place the shared classifier could be bypassed.
    ///
    /// Nothing retries these -- their callers are `meka account usage` / `whoami` / `stats` and the
    /// REPL's `/usage`, none inside a retry loop -- so classifying them changes only how the
    /// failure reads. Still worth pinning: it is the same classifier the turn path depends on, and
    /// a probe that stops calling it is a probe that has grown its own private rules.
    #[tokio::test]
    async fn the_account_probes_report_a_dead_endpoint_as_retryable() {
        let provider = provider_on_a_dead_port();

        let usage = provider.fetch_usage().await.expect_err("nothing is there");
        assert!(
            matches!(usage, MekaError::RetryableProvider { .. }),
            "{usage}"
        );

        let identity = provider
            .fetch_identity()
            .await
            .expect_err("nothing is there");
        assert!(
            matches!(identity, MekaError::RetryableProvider { .. }),
            "{identity}"
        );

        let history = provider
            .fetch_history()
            .await
            .expect_err("nothing is there");
        assert!(
            matches!(history, MekaError::RetryableProvider { .. }),
            "{history}"
        );
    }

    /// A provider pointed at a port nothing is listening on, with a credential that needs no
    /// refresh round trip of its own first.
    ///
    /// An OAuth token rather than the API key the other tests use, because this backend refuses an
    /// API key outright and would never reach a send. The port is bound and dropped, so connecting
    /// is refused rather than answered or left hanging.
    fn provider_on_a_dead_port() -> ClaudeSubscriptionProvider {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        ClaudeSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ClaudeSubscription,
                AuthCredential::OAuthToken {
                    access_token: "access-test".to_string(),
                    refresh_token: Some("refresh-test".to_string()),
                    expires_at: Some(crate::oauth::now_epoch_millis() + 86_400_000),
                    account_id: Some("workspace-test".to_string()),
                },
                "claude-sonnet-4-20250514".to_string(),
            )
            .base_url(Some(format!("http://127.0.0.1:{port}")))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .thinking(ThinkingMode::Off, 10000)
            .device_id("a".repeat(64))
            .effort(Some("high".to_string()))
            .thinking_display(crate::config::ThinkingDisplay::Summarized)
            .max_output_tokens(None)
            .max_request_bytes(None),
        )
        .expect("build test provider")
    }

    #[test]
    fn claude_request_body_simple() {
        let provider = provider_for_test();

        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body(
            "system prompt",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &conversation_attribution(),
        );

        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["stream"], false);

        let system = body["system"].as_array().expect("system should be array");
        assert_eq!(system.len(), 3);

        assert_eq!(system[0]["type"], "text");
        let billing = system[0]["text"].as_str().unwrap();
        let expected_prefix = format!("x-anthropic-billing-header: cc_version={CC_VERSION}.");
        assert!(billing.starts_with(&expected_prefix), "{}", billing);
        assert!(billing.contains("cc_entrypoint=cli"));
        assert!(billing.contains("cch=00000"));
        assert!(system[0].get("cache_control").is_none());

        assert_eq!(system[1]["type"], "text");
        assert_eq!(system[1]["text"], CC_SYSTEM_PROMPT_PREFIX);
        // Identity prefix carries no cache_control, matching the captured Claude Code wire.
        assert!(system[1].get("cache_control").is_none());

        assert_eq!(system[2]["type"], "text");
        assert_eq!(system[2]["text"], "system prompt");
        // User system prompt carries cache_control with ttl=1h and scope=global (matches the
        // captured Claude Code CLI breakpoint).
        assert_eq!(
            system[2]["cache_control"],
            serde_json::json!({"type": "ephemeral", "ttl": "1h", "scope": "global"})
        );

        assert_eq!(claude_code_key_order(&body), vec![
            "model",
            "messages",
            "system",
            "metadata",
            "max_tokens",
            // No `thinking`: this profile has it off, and meka omits the key where Claude Code
            // would send `{"type":"disabled"}`. See `insert_thinking_fields`, which is shared
            // with the `anthropic-messages` backend and its arbitrary endpoints.
            "temperature",
            "diagnostics",
            "stream",
        ]);

        let user_id_str = body["metadata"]["user_id"].as_str().unwrap();
        let user_id_parsed: serde_json::Value = serde_json::from_str(user_id_str).unwrap();
        assert!(user_id_parsed.get("device_id").is_some());
        assert!(user_id_parsed.get("session_id").is_some());

        let claude_messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(claude_messages.len(), 1);
        assert_eq!(claude_messages[0]["role"], "user");

        let content = claude_messages[0]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "hello");
        assert!(content[0].get("cache_control").is_some());
    }

    #[test]
    fn claude_request_body_with_tools() {
        let provider = provider_for_test();

        let tools = vec![ToolDefinition::new(
            "read_file".to_string(),
            "Read a file".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        )];

        let body = provider.build_request_body(
            "",
            &[],
            &tools,
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let claude_tools = body["tools"].as_array().expect("tools should be array");
        assert_eq!(claude_tools.len(), 1);
        assert_eq!(claude_tools[0]["name"], "read_file");
        assert_eq!(claude_tools[0]["description"], "Read a file");
        assert!(claude_tools[0].get("input_schema").is_some());
        // Tools carry no cache_control (matches the captured CLI wire).
        assert!(claude_tools[0].get("cache_control").is_none());
    }

    #[test]
    fn claude_request_body_with_tool_calls() {
        let provider = provider_for_test();

        let messages = vec![
            Message::user("read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "file contents here".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];

        let body = provider.build_request_body(
            "",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let claude_messages = body["messages"]
            .as_array()
            .expect("messages should be array");

        assert_eq!(claude_messages.len(), 3);
        assert_eq!(claude_messages[0]["role"], "user");

        assert_eq!(claude_messages[1]["role"], "assistant");
        let assistant_content = claude_messages[1]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(assistant_content[0]["type"], "tool_use");
        assert_eq!(assistant_content[0]["id"], "toolu_1");
        assert_eq!(assistant_content[0]["name"], "read_file");

        assert_eq!(claude_messages[2]["role"], "user");
        let result_content = claude_messages[2]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(result_content[0]["type"], "tool_result");
        assert_eq!(result_content[0]["tool_use_id"], "toolu_1");

        let first_content = claude_messages[0]["content"]
            .as_array()
            .expect("content should be array");
        assert!(first_content[0].get("cache_control").is_none());
        assert!(assistant_content[0].get("cache_control").is_none());
        assert!(result_content[0].get("cache_control").is_some());
    }

    #[test]
    fn claude_parse_non_streaming_text() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Hello there!"
            }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        });

        let (message, stop_reason, _) =
            parse_non_streaming_response(&response).expect("should parse");

        assert_eq!(message.text_content(), "Hello there!");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn claude_parse_non_streaming_tool_use() {
        let response = serde_json::json!({
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll read that file for you."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "read_file",
                    "input": {"path": "/tmp/test.txt"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 20, "output_tokens": 15 }
        });

        let (message, stop_reason, _) =
            parse_non_streaming_response(&response).expect("should parse");

        assert_eq!(stop_reason, StopReason::ToolUse);
        assert_eq!(message.text_content(), "I'll read that file for you.");

        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);

        if let ContentBlock::ToolUse { id, name, input } = &tool_uses[0] {
            assert_eq!(id, "toolu_abc");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "/tmp/test.txt");
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn patch_request_body_replaces_placeholder() {
        let messages = vec![Message::user("hello")];
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "system prompt",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let body_json = serde_json::to_string(&body).unwrap();

        assert!(body_json.contains("cch=00000"));

        let patched = attestation::patch_request_body(&body_json).unwrap();
        assert!(!patched.contains("cch=00000"));
        let cch_index = patched.find("cch=").expect("cch= must be present");
        let token = &patched[cch_index + 4..cch_index + 9];
        assert_eq!(token.len(), 5);
        assert!(token.chars().all(|c| c.is_ascii_hexdigit()), "{}", token);

        let patched2 = attestation::patch_request_body(&body_json).unwrap();
        assert_eq!(patched, patched2);
    }

    #[test]
    fn patch_request_body_ignores_cch_in_messages() {
        let messages = vec![Message::user(
            "The billing header contains cch=00000 as a placeholder.",
        )];
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "system prompt",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let body_json = serde_json::to_string(&body).unwrap();

        let count = body_json.matches("cch=00000").count();
        assert_eq!(count, 2, "expected 2 occurrences of cch=00000 in body");

        let patched = attestation::patch_request_body(&body_json).unwrap();

        let billing_start = patched.find("x-anthropic-billing-header:").unwrap();
        let billing_region = &patched[billing_start..billing_start + 200];
        assert!(!billing_region.contains("cch=00000"));
        assert!(patched.contains("cch=00000"));
    }

    #[test]
    fn claude_no_system_prompt_when_empty() {
        let provider = provider_for_test();

        let body = provider.build_request_body(
            "",
            &[],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert!(body.get("system").is_none());
    }

    #[test]
    fn claude_parse_missing_tool_use_id() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "name": "read_file",
                "input": {"path": "/tmp/test.txt"}
            }],
            "stop_reason": "tool_use"
        });

        let result = parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn claude_parse_missing_tool_use_name() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_abc",
                "input": {"path": "/tmp/test.txt"}
            }],
            "stop_reason": "tool_use"
        });

        let result = parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn patch_request_body_cch_in_tool_result() {
        let messages = vec![
            Message::user("run the tool"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "echo cch=00000"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "output: cch=00000".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "system prompt",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let body_json = serde_json::to_string(&body).unwrap();
        assert!(body_json.matches("cch=00000").count() >= 2);

        let patched = attestation::patch_request_body(&body_json).unwrap();

        let billing_start = patched.find("x-anthropic-billing-header:").unwrap();
        let billing_end = patched[billing_start..].find(';').unwrap() + billing_start;
        let billing_region = &patched[billing_start..billing_end + 30];
        assert!(!billing_region.contains("cch=00000"));
        assert!(patched.contains("output: cch=00000"));
    }

    #[test]
    fn patch_request_body_preserves_length() {
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "prompt",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let body_json = serde_json::to_string(&body).unwrap();
        let patched = attestation::patch_request_body(&body_json).unwrap();
        assert_eq!(body_json.len(), patched.len());
    }

    #[test]
    fn claude_request_body_stream_true() {
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "prompt",
            &[Message::user("hi")],
            &[],
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn claude_request_body_system_and_tools_together() {
        let provider = provider_for_test();
        let tools = vec![ToolDefinition::new(
            "bash".to_string(),
            "Run a shell command".to_string(),
            serde_json::json!({"type": "object", "properties": {}}),
        )];
        let body = provider.build_request_body(
            "system prompt",
            &[Message::user("hi")],
            &tools,
            true,
            ThinkingOverride::Inherit,
            &conversation_attribution(),
        );

        assert!(body.get("system").is_some());
        assert!(body.get("tools").is_some());
        assert_eq!(body["stream"], true);

        assert_eq!(claude_code_key_order(&body), vec![
            "model",
            "messages",
            "system",
            "tools",
            "metadata",
            "max_tokens",
            // No `thinking`: this profile has it off, and meka omits the key where Claude Code
            // would send `{"type":"disabled"}`. See `insert_thinking_fields`, which is shared
            // with the `anthropic-messages` backend and its arbitrary endpoints.
            "temperature",
            "diagnostics",
            "stream",
        ]);

        let tools_array = body["tools"].as_array().unwrap();
        assert!(tools_array.last().unwrap().get("cache_control").is_none());
    }

    #[test]
    fn claude_request_body_metadata_fields() {
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "prompt",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        let user_id_str = body["metadata"]["user_id"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(user_id_str).unwrap();

        assert_eq!(parsed["device_id"], "a".repeat(64));
        assert_eq!(parsed["account_uuid"], "");
        let session_id = parsed["session_id"].as_str().unwrap();
        assert!(Uuid::parse_str(session_id).is_ok(), "{}", session_id);
    }

    #[test]
    fn oauth_profile_maps_identity() {
        // Trimmed from the live-verified `GET /api/oauth/profile` body.
        let body = r#"{
            "account": {"display_name": "Alice", "email": "a@example.com", "has_claude_max": true},
            "organization": {"name": "Acme", "organization_type": "claude_max",
                             "rate_limit_tier": "default_claude_max_20x",
                             "subscription_status": "active"}
        }"#;
        let identity = serde_json::from_str::<OAuthProfileResponse>(body)
            .expect("parse")
            .into_identity(Some("admin".to_string()));
        assert_eq!(identity.display_name.as_deref(), Some("Alice"));
        assert_eq!(identity.email.as_deref(), Some("a@example.com"));
        assert_eq!(identity.plan.as_deref(), Some("claude_max"));
        assert_eq!(identity.tier.as_deref(), Some("default_claude_max_20x"));
        assert_eq!(identity.subscription_status.as_deref(), Some("active"));
        assert_eq!(identity.organization.as_deref(), Some("Acme"));
        assert_eq!(identity.role.as_deref(), Some("admin"));
    }

    #[test]
    fn oauth_profile_tolerates_missing_fields() {
        let identity = serde_json::from_str::<OAuthProfileResponse>(r#"{"account": {}}"#)
            .unwrap()
            .into_identity(None);
        assert_eq!(identity.display_name, None);
        assert_eq!(identity.plan, None);
        assert_eq!(identity.role, None);
    }

    #[test]
    fn oauth_usage_maps_windows() {
        // Trimmed from a real `GET /api/oauth/usage` body: flat windows plus null/extra buckets we
        // must tolerate.
        let body = r#"{
            "five_hour": {"utilization": 8.0, "resets_at": "2026-07-02T02:10:00.621901+00:00"},
            "seven_day": {"utilization": 2.0, "resets_at": "2026-07-02T13:00:00.621920+00:00"},
            "seven_day_opus": null,
            "seven_day_sonnet": null,
            "extra_usage": {"is_enabled": false},
            "limits": [{"kind": "session", "percent": 8}]
        }"#;
        let parsed: OAuthUsageResponse = serde_json::from_str(body).expect("parse usage");
        let usage = parsed.into_account_usage();
        assert_eq!(usage.windows.len(), 2);
        assert_eq!(usage.windows[0].label, "5-hour (session)");
        assert_eq!(usage.windows[0].used_percent, 8.0);
        assert_eq!(usage.windows[1].label, "Weekly");
        assert_eq!(usage.windows[1].used_percent, 2.0);
        // RFC 3339 with fractional seconds + offset parses to Unix seconds.
        assert_eq!(
            usage.windows[0].resets_at,
            Some(
                chrono::DateTime::parse_from_rfc3339("2026-07-02T02:10:00.621901+00:00")
                    .unwrap()
                    .timestamp()
            )
        );
    }

    #[test]
    fn oauth_extra_usage_maps_credits() {
        // Credit dollars are carried inline in `extra_usage`, in cents. The live API sends them as
        // JSON floats (e.g. `1832.0`), so the fields must be `f64`: used_credits=1832.0 -> $18.32
        // spent, monthly_limit=5000.0 -> $50.00 cap, so balance = (5000 - 1832)/100 = $31.68.
        let body = r#"{
            "five_hour": {"utilization": 8.0, "resets_at": null},
            "extra_usage": {"is_enabled": true, "utilization": 35.0, "used_credits": 1832.0, "monthly_limit": 5000.0}
        }"#;
        let extra = serde_json::from_str::<OAuthUsageResponse>(body)
            .expect("parse usage")
            .into_account_usage()
            .extra_usage
            .expect("extra_usage");
        assert!(extra.enabled);
        assert_eq!(extra.utilization, Some(35.0));
        assert_eq!(extra.used, Some(18.32));
        assert_eq!(extra.balance, Some(31.68));
        assert_eq!(extra.currency, None);
    }

    #[test]
    fn oauth_usage_includes_opus_when_present() {
        let body = r#"{
            "five_hour": {"utilization": 10.0, "resets_at": null},
            "seven_day": {"utilization": 5.0, "resets_at": null},
            "seven_day_opus": {"utilization": 42.0, "resets_at": null}
        }"#;
        let usage = serde_json::from_str::<OAuthUsageResponse>(body)
            .unwrap()
            .into_account_usage();
        assert_eq!(usage.windows.len(), 3);
        assert_eq!(usage.windows[2].label, "Weekly (Opus)");
        assert_eq!(usage.windows[2].used_percent, 42.0);
        assert_eq!(usage.windows[0].resets_at, None);
    }

    #[test]
    fn a_claude_subscription_base_url_keeps_the_root_the_oauth_endpoints_hang_off() {
        let provider = ClaudeSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ClaudeSubscription,
                AuthCredential::OAuthToken {
                    access_token: "token".to_string(),
                    refresh_token: None,
                    expires_at: None,
                    account_id: None,
                },
                "claude-opus-4-8".to_string(),
            )
            .base_url(Some(
                "https://gateway.example.com/anthropic/v1/".to_string(),
            ))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .thinking(ThinkingMode::Off, 10000)
            .device_id("a".repeat(64))
            .effort(None)
            .thinking_display(crate::config::ThinkingDisplay::Summarized)
            .max_output_tokens(None)
            .max_request_bytes(None),
        )
        .expect("build test provider");
        // This provider reaches `/v1/messages` *and* `/api/oauth/usage` off the same root, so the
        // base has to stay the root: a stored `.../v1` would put the OAuth endpoints out of reach.
        assert_eq!(provider.base_url, "https://gateway.example.com/anthropic");
    }

    #[test]
    fn account_uuid_sourced_from_oauth_credential() {
        let provider = ClaudeSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ClaudeSubscription,
                AuthCredential::OAuthToken {
                    access_token: "token".to_string(),
                    refresh_token: None,
                    expires_at: None,
                    account_id: Some("7194a774-10cb-47f6-a031-78078f9054c9".to_string()),
                },
                "claude-opus-4-8".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .thinking(ThinkingMode::Off, 10000)
            .device_id("a".repeat(64))
            .effort(Some("high".to_string()))
            .thinking_display(crate::config::ThinkingDisplay::Summarized)
            .max_output_tokens(None)
            .max_request_bytes(None),
        )
        .expect("build test provider");
        let body = provider.build_request_body(
            "prompt",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let user_id_str = body["metadata"]["user_id"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(user_id_str).unwrap();
        assert_eq!(
            parsed["account_uuid"],
            "7194a774-10cb-47f6-a031-78078f9054c9"
        );
    }

    #[test]
    fn claude_request_body_no_tools_key_when_empty() {
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "prompt",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn claude_parse_missing_content_array() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn"
        });
        let result = parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn claude_parse_missing_stop_reason_defaults_to_end_turn() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}]
        });
        let (_, stop_reason, _) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn claude_parse_max_tokens_stop_reason() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "truncated"}],
            "stop_reason": "max_tokens"
        });
        let (_, stop_reason, _) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn claude_parse_unknown_stop_reason() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "something_new"
        });
        let (_, stop_reason, _) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(
            stop_reason,
            StopReason::Unknown("something_new".to_string())
        );
    }

    #[test]
    fn claude_parse_empty_content_array() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": "end_turn"
        });
        let (message, ..) = parse_non_streaming_response(&response).expect("should parse");
        assert!(message.content.is_empty());
        assert_eq!(message.text_content(), "");
    }

    #[test]
    fn claude_parse_thinking_block() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm..."},
                {"type": "text", "text": "answer"}
            ],
            "stop_reason": "end_turn"
        });
        let (message, ..) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(message.content.len(), 2);
        assert!(
            matches!(&message.content[0], ContentBlock::Thinking { thinking, .. } if thinking == "hmm...")
        );
        assert_eq!(message.text_content(), "answer");
    }

    #[test]
    fn claude_parse_unknown_block_type_skipped() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "totally_unknown", "data": "xyz"},
                {"type": "text", "text": "answer"}
            ],
            "stop_reason": "end_turn"
        });
        let (message, ..) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.text_content(), "answer");
    }

    #[test]
    fn claude_parse_tool_use_missing_input_defaults() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "list_files"
            }],
            "stop_reason": "tool_use"
        });
        let (message, ..) = parse_non_streaming_response(&response).expect("should parse");
        if let ContentBlock::ToolUse { input, .. } = &message.content[0] {
            assert_eq!(*input, serde_json::json!({}));
        } else {
            panic!("expected ToolUse block");
        }
    }

    fn provider_with(model: &str, thinking: bool) -> ClaudeSubscriptionProvider {
        provider_full(
            model,
            thinking,
            "high",
            crate::config::ThinkingDisplay::Summarized,
        )
    }

    fn provider_effort(model: &str, effort: Option<&str>) -> ClaudeSubscriptionProvider {
        ClaudeSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ClaudeSubscription,
                AuthCredential::ApiKey("test-key".to_string()),
                model.to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .thinking(ThinkingMode::Off, 10000)
            .device_id("a".repeat(64))
            .effort(effort.map(str::to_string))
            .thinking_display(crate::config::ThinkingDisplay::Summarized)
            .max_output_tokens(None)
            .max_request_bytes(None),
        )
        .expect("build test provider")
    }

    /// `thinking` is a bool here because every caller is asserting on a beta or on `temperature`,
    /// and those gate on whether thinking is on at all rather than on which encoding it uses.
    fn provider_full(
        model: &str,
        thinking: bool,
        effort: &str,
        thinking_display: crate::config::ThinkingDisplay,
    ) -> ClaudeSubscriptionProvider {
        ClaudeSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ClaudeSubscription,
                AuthCredential::ApiKey("test-key".to_string()),
                model.to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .thinking(
                if thinking {
                    ThinkingMode::Adaptive
                } else {
                    ThinkingMode::Off
                },
                10000,
            )
            .device_id("a".repeat(64))
            .effort(Some(effort.to_string()))
            .thinking_display(thinking_display)
            .max_output_tokens(None)
            .max_request_bytes(None),
        )
        .expect("build test provider")
    }

    #[test]
    fn betas_no_adaptive_thinking_beta() {
        // `adaptive-thinking-2026-01-28` does not exist in Claude Code; adaptive thinking is GA and
        // selected via the body `thinking` param, never a beta header.
        for (model, thinking) in [
            ("claude-opus-4-6-20250514", true),
            ("claude-opus-4-8", true),
            ("claude-opus-4-6-20250514", false),
        ] {
            let betas = provider_with(model, thinking)
                .compute_betas(true, true)
                .unwrap();
            assert!(
                !betas.contains("adaptive-thinking"),
                "{model} (thinking={thinking}) must not send an adaptive-thinking beta: {betas}"
            );
        }
    }

    #[test]
    fn betas_modern_thinking_model_full_set() {
        // Tools + thinking + the default display: matches the live Claude Code 2.1.263 interactive
        // CLI wire capture exactly, minus `context-1m-2025-08-07`, which this profile's window does
        // not ask for. Display updates replace the redaction beta on a turn with thinking on.
        let betas = provider_full(
            "claude-opus-4-8",
            true,
            "high",
            crate::config::ThinkingDisplay::Updates,
        )
        .compute_betas(true, true)
        .unwrap();
        let parts: Vec<&str> = betas.split(',').collect();
        assert_eq!(
            parts,
            vec![
                "claude-code-20250219",
                "oauth-2025-04-20",
                "interleaved-thinking-2025-05-14",
                "thinking-token-count-2026-05-13",
                "context-management-2025-06-27",
                "prompt-caching-scope-2026-01-05",
                "mid-conversation-system-2026-04-07",
                "advanced-tool-use-2025-11-20",
                "effort-2025-11-24",
                "fallback-credit-2026-06-01",
                "thinking-display-updates-2026-08-18",
                "extended-cache-ttl-2025-04-11",
                "cache-diagnosis-2026-04-07",
            ],
            "Claude Code 2.1.263 CLI beta set"
        );
    }

    #[test]
    fn betas_thinking_family_independent_of_toggle() {
        // Claude Code gates interleaved-thinking / thinking-token-count on model capability, not
        // the thinking toggle, so they appear whether thinking is on or off.
        for thinking in [true, false] {
            let betas = provider_with("claude-opus-4-6-20250514", thinking)
                .compute_betas(true, true)
                .unwrap();
            assert!(betas.contains("interleaved-thinking-2025-05-14"), "{betas}");
            assert!(betas.contains("thinking-token-count-2026-05-13"), "{betas}");
        }
    }

    #[test]
    fn the_1m_context_beta_is_not_sent_without_a_1m_window() {
        // Claude Code sends `context-1m-2025-08-07` only for the `[1m]` variant its user picked,
        // and meka only for a profile whose window is a million or more; these profiles state no
        // window, so the beta is absent across the lineup.
        for model in [
            "claude-opus-4-8",
            "claude-opus-4-6-20250514",
            "claude-sonnet-4-6",
            "claude-fable-5",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
        ] {
            assert!(
                !provider_with(model, false)
                    .compute_betas(true, true)
                    .unwrap()
                    .contains("context-1m-2025-08-07"),
                "{model} must not send context-1m"
            );
        }
    }

    #[test]
    fn betas_extended_cache_ttl_always_present() {
        // meka always sends a 1h cache TTL, so the extended-cache-ttl beta is unconditional.
        for model in [
            "claude-opus-4-6-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
        ] {
            assert!(
                provider_with(model, false)
                    .compute_betas(true, true)
                    .unwrap()
                    .contains("extended-cache-ttl-2025-04-11"),
                "{model} must send extended-cache-ttl"
            );
        }
    }

    #[test]
    fn betas_haiku_skips_claude_code_and_effort() {
        // Effort unset: Haiku (pre-4.6) omits the effort beta by default. (An explicit override is
        // absolute and would send it; see
        // output_config_omitted_when_unset_and_model_lacks_effort.)
        let betas = provider_effort("claude-haiku-4-5-20251001", None)
            .compute_betas(true, true)
            .unwrap();
        assert!(!betas.contains("claude-code-20250219"), "{betas}");
        assert!(!betas.contains("effort-2025-11-24"), "{betas}");
        // OAuth, prompt-caching-scope, and extended-cache-ttl are unconditional; Haiku still has
        // modern features (interleaved-thinking, context-management).
        assert!(betas.contains("oauth-2025-04-20"), "{betas}");
        assert!(betas.contains("prompt-caching-scope-2026-01-05"), "{betas}");
        assert!(betas.contains("interleaved-thinking-2025-05-14"), "{betas}");
    }

    #[test]
    fn betas_oauth_and_prompt_caching_scope_always_present() {
        for model in [
            "claude-opus-4-6-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
        ] {
            let provider = provider_with(model, false);
            let betas = provider.compute_betas(true, true).unwrap();
            assert!(betas.contains("oauth-2025-04-20"), "{model} → {betas}");
            assert!(
                betas.contains("prompt-caching-scope-2026-01-05"),
                "{model} → {betas}"
            );
        }
    }

    #[test]
    fn context_management_body_when_thinking_enabled() {
        let provider = provider_with("claude-opus-4-6-20250514", true);
        let body = provider.build_request_body(
            "system prompt",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let cm = body
            .get("context_management")
            .expect("context_management should be present when thinking is on");
        assert_eq!(cm["edits"][0]["type"], "clear_thinking_20251015");
        assert_eq!(cm["edits"][0]["keep"], "all");
    }

    #[test]
    fn output_config_effort_uses_configured_value() {
        for value in ["low", "medium", "high"] {
            let provider = provider_full(
                "claude-opus-4-6-20250514",
                false,
                value,
                crate::config::ThinkingDisplay::Summarized,
            );
            let body = provider.build_request_body(
                "system prompt",
                &[Message::user("hi")],
                &[],
                false,
                ThinkingOverride::Inherit,
                &crate::provider::Attribution::default(),
            );
            let oc = body
                .get("output_config")
                .unwrap_or_else(|| panic!("output_config missing for effort={value}"));
            assert_eq!(
                oc["effort"], value,
                "effort body field must reflect configured value"
            );
        }
    }

    #[test]
    fn an_unconfigured_profile_sends_claude_codes_default_effort() {
        // Claude Code never leaves `output_config.effort` to the server on a model that takes one:
        // it resolves a default and sends that. meka sends one value for all of them, and that
        // uniformity is the assertion -- a per-model table would pass a laxer version of this test
        // while adding a fact about Anthropic's data that nothing here would notice going stale.
        for model in [
            "claude-opus-4-8",
            "claude-opus-4-6-20250514",
            "claude-opus-5",
            "claude-sonnet-5",
            // Claude Code's bundled table happens to give this one `xhigh`. meka does not carry
            // per-model figures: the server cannot tell meka's default from a configured value, so
            // a table would only add something to go stale.
            "claude-opus-4-7",
        ] {
            let provider = provider_effort(model, None);
            let body = provider.build_request_body(
                "s",
                &[Message::user("hi")],
                &[],
                false,
                ThinkingOverride::Inherit,
                &crate::provider::Attribution::default(),
            );
            assert_eq!(body["output_config"]["effort"], "high", "{model}");
            assert!(
                provider
                    .compute_betas(true, true)
                    .unwrap_or_default()
                    .contains("effort-2025-11-24"),
                "{model} takes an effort, so the beta rides with it"
            );
        }

        // A model that takes no effort gets neither the field nor the beta, however the profile is
        // configured -- sending it would be a 400 rather than a stronger request.
        for model in ["claude-haiku-4-5-20251001", "claude-3-5-sonnet-20241022"] {
            for configured in [None, Some("high")] {
                let provider = provider_effort(model, configured);
                let body = provider.build_request_body(
                    "s",
                    &[Message::user("hi")],
                    &[],
                    false,
                    ThinkingOverride::Inherit,
                    &crate::provider::Attribution::default(),
                );
                assert!(
                    body.get("output_config").is_none(),
                    "{model} {configured:?}"
                );
                assert!(
                    !provider
                        .compute_betas(true, true)
                        .unwrap_or_default()
                        .contains("effort-2025-11-24"),
                    "{model} {configured:?}"
                );
            }
        }

        // A configured value is absolute: it replaces the default and is never clamped down to what
        // a bundled table thinks the model supports.
        let forced = provider_effort("claude-sonnet-4-6", Some("max"));
        let body = forced.build_request_body(
            "system prompt",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert_eq!(body["output_config"]["effort"], "max");
        assert!(
            forced
                .compute_betas(true, true)
                .unwrap()
                .contains("effort-2025-11-24"),
            "effort beta must accompany an explicit override in the body"
        );
    }

    #[test]
    fn temperature_present_when_model_supports_it() {
        // Opus 4.6 accepts sampling params; with thinking off, `temperature: 1` is sent.
        let provider = provider_with("claude-opus-4-6-20250514", false);
        let body = provider.build_request_body(
            "system prompt",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert_eq!(body["temperature"], 1);
    }

    #[test]
    fn temperature_omitted_when_model_rejects_it() {
        // Opus 4.8 rejects `temperature` (400); meka must omit it even with thinking off.
        let provider = provider_with("claude-opus-4-8", false);
        let body = provider.build_request_body(
            "system prompt",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert!(
            body.get("temperature").is_none(),
            "temperature must be omitted for sampling-param-removed models"
        );
    }

    #[test]
    fn betas_advanced_tool_use_gated_on_tools() {
        let provider = provider_with("claude-opus-4-8", true);
        assert!(
            provider
                .compute_betas(true, true)
                .unwrap()
                .contains("advanced-tool-use-2025-11-20"),
            "advanced-tool-use must be sent when the request carries tools"
        );
        assert!(
            !provider
                .compute_betas(false, true)
                .unwrap()
                .contains("advanced-tool-use-2025-11-20"),
            "advanced-tool-use must be omitted when there are no tools"
        );
    }

    #[test]
    fn betas_mid_conversation_system_gated_on_model() {
        // The gate is a denylist, so the newer models get it and the named older ones do not
        // (mirrors Claude Code 2.1.263's own list).
        for model in ["claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"] {
            assert!(
                provider_with(model, true)
                    .compute_betas(true, true)
                    .unwrap()
                    .contains("mid-conversation-system-2026-04-07"),
                "{model} must send mid-conversation-system"
            );
        }
        for model in ["claude-opus-4-6-20250514", "claude-haiku-4-5-20251001"] {
            assert!(
                !provider_with(model, true)
                    .compute_betas(true, true)
                    .unwrap()
                    .contains("mid-conversation-system-2026-04-07"),
                "{model} must not send mid-conversation-system"
            );
        }
    }

    #[test]
    fn a_model_newer_than_the_denylist_still_gets_mid_conversation_system() {
        // The direction of the gate, not a fact about these names: Claude Code's own list excludes
        // the older models and sends the beta to everything else, so a model meka has never heard
        // of has to come out on the sending side. Flipping the gate to an allowlist passes every
        // other test in this file and silently drops mid-conversation system messages the first
        // time Anthropic ships a model.
        for model in [
            "claude-opus-6",
            "claude-sonnet-7-20270101",
            "claude-haiku-9",
        ] {
            assert!(
                provider_with(model, true)
                    .compute_betas(true, true)
                    .unwrap()
                    .contains("mid-conversation-system-2026-04-07"),
                "{model} is newer than the denylist and must still send mid-conversation-system"
            );
        }
    }

    #[tokio::test]
    async fn billing_header_marks_subagent_under_scope() {
        let provider = provider_for_test();
        let messages = vec![Message::user("hi")];

        // Outside any sub-agent scope: no subagent segment.
        let main_body = provider.build_request_body(
            "system prompt",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let main_billing = main_body["system"][0]["text"].as_str().unwrap();
        assert!(!main_billing.contains("cc_is_subagent"), "{main_billing}");

        // Attributed to a sub-agent: the billing header carries `cc_is_subagent=true;` after `cch`.
        let sub_billing = {
            let body = provider.build_request_body(
                "system prompt",
                &messages,
                &[],
                false,
                ThinkingOverride::Inherit,
                &crate::provider::Attribution {
                    subagent: true,
                    ..Default::default()
                },
            );
            body["system"][0]["text"].as_str().unwrap().to_string()
        };
        assert!(
            sub_billing.contains("cch=00000; cc_is_subagent=true;"),
            "{sub_billing}"
        );
    }

    #[test]
    fn a_message_quoting_a_billing_header_cannot_steal_the_attestation() {
        // `messages` now precedes `system` on the wire, so a substring search for the billing
        // header would find the conversation's copy first and patch the token into a message --
        // leaving the real header reading `cch=00000`. A session about this code is exactly the
        // conversation that contains one, so this is a live case, not a contrived one.
        let forgery = "x-anthropic-billing-header: cc_version=9.9.9.aaa; \
                       cc_entrypoint=cli; cch=00000; cc_prompt_id=deadbeef;";
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "system prompt",
            &[Message::user(forgery)],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let body_json = serde_json::to_string(&body).unwrap();
        assert!(
            body_json.find(forgery).unwrap() < body_json.find("\"system\"").unwrap(),
            "the forged header must come first, or this proves nothing"
        );

        let patched = attestation::patch_request_body(&body_json).expect("patched");
        let patched: serde_json::Value = serde_json::from_str(&patched).unwrap();

        // The message is untouched and the real header carries the token.
        assert_eq!(patched["messages"][0]["content"][0]["text"], forgery);
        let billing = patched["system"][0]["text"].as_str().unwrap();
        assert!(!billing.contains("cch=00000"), "{billing}");
        assert!(billing.contains("cch="), "{billing}");
    }

    #[tokio::test]
    async fn the_response_header_anthropic_actually_sends_is_the_one_read() {
        // `request-id`, not `x-request-id`: pinning the exact name matters because reading the
        // wrong one leaves `cc_prev_req` silently absent on every turn, which is also what a
        // conversation's first request looks like.
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "request-id",
            "req_011CeJwF1NDYzXkUu6cyFJp2".parse().unwrap(),
        );
        let slot = crate::provider::PreviousRequestSlot::default();
        remember_request_id(
            &crate::provider::Attribution {
                previous_request: Some(Arc::clone(&slot)),
                ..Default::default()
            },
            &headers,
        );
        assert_eq!(
            slot.lock().expect("slot").as_deref(),
            Some("req_011CeJwF1NDYzXkUu6cyFJp2")
        );
    }

    #[tokio::test]
    async fn the_billing_header_names_the_prompt_and_the_response_before_it() {
        let provider = provider_for_test();
        let messages = vec![Message::user("hi")];

        // Outside a turn there is no prompt and no previous response to name, which is how meka's
        // own side queries go out -- matching the Claude Code path that passes neither.
        let bare = provider.build_request_body(
            "system prompt",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let bare = bare["system"][0]["text"].as_str().unwrap();
        assert!(!bare.contains("cc_prompt_id"), "{bare}");
        assert!(!bare.contains("cc_prev_req"), "{bare}");

        let attribution = crate::provider::Attribution {
            prompt_id: Some(uuid::Uuid::new_v4()),
            previous_request: Some(crate::provider::PreviousRequestSlot::default()),
            ..Default::default()
        };
        let first = provider.build_request_body(
            "system prompt",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &attribution,
        );
        let first = first["system"][0]["text"].as_str().unwrap().to_string();
        attribution.record_request_id("req_011CeJwF1NDYzXkUu6cyFJp2");
        let second = provider.build_request_body(
            "system prompt",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &attribution,
        );
        let second = second["system"][0]["text"].as_str().unwrap().to_string();
        assert!(first.contains("cch=00000; cc_prompt_id="), "{first}");
        assert!(!first.contains("cc_prev_req"), "{first}");

        // The next one names it, in Claude Code's segment order.
        assert!(
            second.contains("cch=00000; cc_prev_req=req_011CeJwF1NDYzXkUu6cyFJp2; cc_prompt_id="),
            "{second}"
        );

        // And it is the same prompt throughout, because it is the same prompt.
        let id_of = |header: &str| {
            header
                .split("cc_prompt_id=")
                .nth(1)
                .and_then(|rest| rest.split(';').next())
                .map(str::to_string)
        };
        assert_eq!(id_of(&first), id_of(&second));
    }

    /// An attribution as a conversation turn carries it: with a message slot, so the body has the
    /// `diagnostics` field a side query lacks.
    fn conversation_attribution() -> crate::provider::Attribution {
        crate::provider::Attribution {
            previous_message: Some(crate::provider::PreviousMessageSlot::default()),
            ..Default::default()
        }
    }

    fn body_for(provider: &ClaudeSubscriptionProvider) -> serde_json::Value {
        provider.build_request_body(
            "system prompt",
            &[Message::user("hi")],
            &[],
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        )
    }

    /// Claude Code's default: a turn with thinking on asks for display updates and drops the
    /// redaction beta; a turn with thinking off has nothing to display and keeps the redaction
    /// beta, as its own turns do.
    #[test]
    fn display_updates_follows_whether_thinking_is_on() {
        let on = provider_full(
            "claude-opus-4-6-20250514",
            true,
            "high",
            crate::config::ThinkingDisplay::Updates,
        );
        let betas = on.compute_betas(true, true).unwrap();
        assert!(
            betas.contains("thinking-display-updates-2026-08-18"),
            "{betas}"
        );
        assert!(!betas.contains("redact-thinking-2026-02-12"), "{betas}");
        assert_eq!(
            body_for(&on)["thinking"],
            serde_json::json!({"type": "adaptive", "display": "updates"})
        );

        let off = provider_full(
            "claude-opus-4-6-20250514",
            false,
            "high",
            crate::config::ThinkingDisplay::Updates,
        );
        let betas = off.compute_betas(true, false).unwrap();
        assert!(betas.contains("redact-thinking-2026-02-12"), "{betas}");
        assert!(
            !betas.contains("thinking-display-updates-2026-08-18"),
            "{betas}"
        );
        assert!(body_for(&off).get("thinking").is_none());
    }

    #[test]
    fn display_summarized_asks_for_summaries_and_never_redacts() {
        let on = provider_full(
            "claude-opus-4-6-20250514",
            true,
            "high",
            crate::config::ThinkingDisplay::Summarized,
        );
        let betas = on.compute_betas(true, true).unwrap();
        assert!(!betas.contains("redact-thinking-2026-02-12"), "{betas}");
        assert!(
            !betas.contains("thinking-display-updates-2026-08-18"),
            "{betas}"
        );
        assert_eq!(
            body_for(&on)["thinking"],
            serde_json::json!({"type": "adaptive", "display": "summarized"})
        );
        let off = provider_full(
            "claude-opus-4-6-20250514",
            false,
            "high",
            crate::config::ThinkingDisplay::Summarized,
        );
        assert!(
            !off.compute_betas(true, false)
                .unwrap()
                .contains("redact-thinking")
        );
    }

    #[test]
    fn display_redacted_keeps_the_redaction_beta_and_sends_no_display() {
        let provider = provider_full(
            "claude-opus-4-6-20250514",
            true,
            "high",
            crate::config::ThinkingDisplay::Redacted,
        );
        let betas = provider.compute_betas(true, true).unwrap();
        assert!(betas.contains("redact-thinking-2026-02-12"), "{betas}");
        assert!(
            !betas.contains("thinking-display-updates-2026-08-18"),
            "{betas}"
        );
        assert_eq!(
            body_for(&provider)["thinking"],
            serde_json::json!({"type": "adaptive"})
        );
    }

    /// Claude Code sends the 1M beta for the `[1m]` model variant its user picked; the profile's
    /// window is where a meka user states the same choice.
    #[test]
    fn the_1m_context_beta_follows_the_profiles_window() {
        let with_window = |window: Option<u64>| {
            ClaudeSubscriptionProvider::new(
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::ClaudeSubscription,
                    AuthCredential::ApiKey("test-key".to_string()),
                    "claude-opus-5".to_string(),
                )
                .credential_key(Some("test".to_string()))
                .thinking(ThinkingMode::Adaptive, 10000)
                .device_id("a".repeat(64))
                .context_window(window),
            )
            .expect("provider")
        };
        let betas = with_window(Some(1_000_000))
            .compute_betas(true, true)
            .unwrap();
        assert_eq!(
            betas.split(',').nth(2),
            Some("context-1m-2025-08-07"),
            "third, after oauth, as Claude Code orders it: {betas}"
        );
        for window in [None, Some(200_000), Some(999_999)] {
            let betas = with_window(window).compute_betas(true, true).unwrap();
            assert!(
                !betas.contains("context-1m-2025-08-07"),
                "{window:?}: {betas}"
            );
        }
    }

    /// `diagnostics.previous_message_id` is `null` until a response has carried a message id, and
    /// then names the last one; a malformed id is ignored rather than sent.
    #[test]
    fn diagnostics_carry_the_last_recorded_message_id() {
        let provider = provider_for_test();
        let slot = crate::provider::PreviousMessageSlot::default();
        let attribution = crate::provider::Attribution {
            previous_message: Some(Arc::clone(&slot)),
            ..Default::default()
        };
        let body = |attribution: &crate::provider::Attribution| {
            provider.build_request_body(
                "system prompt",
                &[Message::user("hi")],
                &[],
                true,
                ThinkingOverride::Inherit,
                attribution,
            )
        };
        assert_eq!(
            body(&attribution)["diagnostics"],
            serde_json::json!({"previous_message_id": null})
        );
        attribution.record_message_id("not an id");
        assert_eq!(
            body(&attribution)["diagnostics"]["previous_message_id"],
            serde_json::Value::Null
        );
        attribution.record_message_id("msg_011Cepw4KgcVvSiBJbcdRbqv");
        assert_eq!(
            body(&attribution)["diagnostics"]["previous_message_id"],
            "msg_011Cepw4KgcVvSiBJbcdRbqv"
        );
        // A side query has no slot and no field, as Claude Code's own side queries have none.
        assert!(
            body(&crate::provider::Attribution::default())
                .get("diagnostics")
                .is_none()
        );
    }

    #[test]
    fn context_management_body_absent_when_thinking_disabled() {
        let provider = provider_with("claude-opus-4-6-20250514", false);
        let body = provider.build_request_body(
            "system prompt",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert!(
            body.get("context_management").is_none(),
            "context_management must be omitted when thinking is off"
        );
    }

    /// All `cache_control` markers carry `ttl: "1h"` to match recent Claude Code's
    /// `getCacheControl` (returns `{type:"ephemeral", ttl:"1h"}` for OAuth subscribers via
    /// `should1hCacheTTL`).
    #[test]
    fn cache_control_uses_one_hour_ttl_everywhere() {
        let provider = provider_for_test();
        let tools = vec![ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        )];
        let body = provider.build_request_body(
            "user system prompt",
            &[Message::user("hi")],
            &tools,
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        let expected = serde_json::json!({"type": "ephemeral", "ttl": "1h"});

        // System: only the user prompt block (system[2]) has cache_control; it adds scope=global.
        let system = body["system"].as_array().unwrap();
        assert!(system[0].get("cache_control").is_none());
        assert!(system[1].get("cache_control").is_none());
        assert_eq!(
            system[2]["cache_control"],
            serde_json::json!({"type": "ephemeral", "ttl": "1h", "scope": "global"})
        );

        // Tools: no cache_control (the rolling message breakpoint caches the prefix).
        let tools_arr = body["tools"].as_array().unwrap();
        assert!(tools_arr.last().unwrap().get("cache_control").is_none());

        // Messages: last block of the last message carries cache_control with ttl=1h.
        let messages_arr = body["messages"].as_array().unwrap();
        let last_message = messages_arr.last().unwrap();
        let last_block = last_message["content"].as_array().unwrap().last().unwrap();
        assert_eq!(last_block["cache_control"], expected);
    }

    #[test]
    fn now_epoch_millis_reasonable() {
        let ms = crate::oauth::now_epoch_millis();
        assert!(ms > 1_577_836_800_000);
        assert!(ms < 4_102_444_800_000);
    }

    // Cache prefix stability tests. These tests simulate multi-turn conversations and tool-use
    // loops to verify that the serialized request bodies share a stable prefix across successive
    // API calls, which is the fundamental requirement for KV cache reuse. A "prefix" here means:
    // the system prompt, tool schemas, and all previously-sent messages must serialize identically
    // (ignoring the `cache_control` marker, which intentionally moves to the newest tail).

    /// Strips every `cache_control` key from every content block in a message so two messages can
    /// be compared purely on semantic content.
    fn strip_cache_control(message: &serde_json::Value) -> serde_json::Value {
        let mut message = message.clone();
        if let Some(content) = message.get_mut("content").and_then(|c| c.as_array_mut()) {
            for block in content.iter_mut() {
                if let Some(obj) = block.as_object_mut() {
                    obj.remove("cache_control");
                }
            }
        }
        message
    }

    /// Strips `cache_control` from every tool schema in an array.
    fn strip_tool_cache_control(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|tool| {
                let mut tool = tool.clone();
                if let Some(obj) = tool.as_object_mut() {
                    obj.remove("cache_control");
                }
                tool
            })
            .collect()
    }

    /// Asserts that the first `shared_count` messages in two request bodies are semantically
    /// identical (ignoring `cache_control` movement), and that the system prompt and tool schemas
    /// are identical.
    fn assert_prefix_stable(
        body_a: &serde_json::Value,
        body_b: &serde_json::Value,
        shared_message_count: usize,
    ) {
        // System prompt must be byte-identical (before cch patching).
        assert_eq!(
            body_a["system"], body_b["system"],
            "system prompt diverged between requests"
        );

        // Tool schemas must be identical (content-wise, ignoring cache_control which is always on
        // the last tool and doesn't affect tokens).
        let tools_a = body_a["tools"].as_array();
        let tools_b = body_b["tools"].as_array();
        match (tools_a, tools_b) {
            (Some(a), Some(b)) => {
                assert_eq!(
                    strip_tool_cache_control(a),
                    strip_tool_cache_control(b),
                    "tool schemas diverged between requests"
                );
            }
            (None, None) => {}
            _ => panic!("tools presence diverged between requests"),
        }

        let msgs_a = body_a["messages"]
            .as_array()
            .expect("messages array in body_a");
        let msgs_b = body_b["messages"]
            .as_array()
            .expect("messages array in body_b");

        assert!(
            msgs_a.len() >= shared_message_count,
            "body_a has {} messages, expected at least {}",
            msgs_a.len(),
            shared_message_count
        );
        assert!(
            msgs_b.len() >= shared_message_count,
            "body_b has {} messages, expected at least {}",
            msgs_b.len(),
            shared_message_count
        );

        for i in 0..shared_message_count {
            let a = strip_cache_control(&msgs_a[i]);
            let b = strip_cache_control(&msgs_b[i]);
            assert_eq!(a, b, "message at index {i} diverged between requests");
        }
    }

    /// Counts the total number of `cache_control` markers across all content blocks in the messages
    /// array.
    fn count_message_cache_controls(body: &serde_json::Value) -> usize {
        let mut count = 0;
        if let Some(messages) = body["messages"].as_array() {
            for message in messages {
                if let Some(content) = message["content"].as_array() {
                    for block in content {
                        if block.get("cache_control").is_some() {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    fn tools_for_test() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "read_file".to_string(),
                "Read a file".to_string(),
                serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            ),
            ToolDefinition::new(
                "execute_command".to_string(),
                "Run a shell command".to_string(),
                serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            ),
        ]
    }

    #[test]
    fn multi_turn_prefix_is_stable() {
        let provider = provider_for_test();
        let system = "You are a helpful assistant.";
        let tools = tools_for_test();

        // Turn 1: single user message
        let messages_t1 = vec![Message::user("What files are in /tmp?")];
        let body_t1 = provider.build_request_body(
            system,
            &messages_t1,
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // Turn 2: previous turn + assistant response + new user message
        let messages_t2 = vec![
            Message::user("What files are in /tmp?"),
            Message::assistant_text("There are 3 files in /tmp."),
            Message::user("Show me the first one."),
        ];
        let body_t2 = provider.build_request_body(
            system,
            &messages_t2,
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // Turn 3: previous turns + another exchange
        let messages_t3 = vec![
            Message::user("What files are in /tmp?"),
            Message::assistant_text("There are 3 files in /tmp."),
            Message::user("Show me the first one."),
            Message::assistant_text("Here is the content of file1.txt."),
            Message::user("Delete it."),
        ];
        let body_t3 = provider.build_request_body(
            system,
            &messages_t3,
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // The first message is shared across all three requests.
        assert_prefix_stable(&body_t1, &body_t2, 1);
        // The first three messages are shared between turn 2 and turn 3.
        assert_prefix_stable(&body_t2, &body_t3, 3);
        // The first message is shared across turn 1 and turn 3.
        assert_prefix_stable(&body_t1, &body_t3, 1);
    }

    /// Simulates a two-turn conversation where the user toggles the permission level between turns
    /// and verifies that the cacheable prefix (system prompt + tools array + historical messages)
    /// is byte-identical across the toggle. This is the regression guard for Option 1 of the
    /// higher-permission-visibility work; it proves that `/permission <level>` mid-session does
    /// not invalidate the Claude prompt cache.
    ///
    /// Covers the full agent request-body assembly:
    ///   - [`ToolRegistry::tool_catalog`] / [`ToolRegistry::definitions_active`]
    ///   - [`crate::prompt::build_system_prompt`]
    ///   - [`crate::prompt::build_turn_context`]
    ///   - [`ClaudeSubscriptionProvider::build_request_body`]
    #[tokio::test]
    async fn permission_toggle_preserves_cache_prefix() {
        use crate::{
            permission::{Permission, SharedPermission},
            prompt::{TurnContext, build_system_prompt, build_turn_context},
            store::Store,
            tools::ToolRegistry,
        };

        let store = Store::for_test().await;
        let shared_permission =
            SharedPermission::new(Permission::Read, crate::permission::EnabledPermissions::ALL);
        let shared_session_id = crate::session::SharedSessionId::default();
        let todo_list = crate::todo::SharedTodoList::default();
        let registry = ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability: crate::sandbox::detect(),
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe: crate::sandbox::BackendProbe::Missing {
                        reason: "test fixture".to_string(),
                    },
                    builtin_filter: crate::config::BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                skills_agent_managed: false,
                memories: crate::store::memory::MemoryStore::detached(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: shared_session_id,
                todo_list,
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    shared_permission,
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    std::sync::Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("default web client config should build cleanly");

        let provider = provider_for_test();

        // The agent builds these once per turn. Neither takes the current permission; that's the
        // invariant we're testing.
        let system = build_system_prompt(true, None);
        let tools = registry.definitions_active(&[]);

        let u1_text = {
            let block = build_turn_context(TurnContext {
                permission: Permission::Read,
                approvals: false,
                todos: &crate::todo::TodoState::default(),
                cwd: std::path::Path::new("."),
                roots: &[],
                world_state: "",
                budget: None,
                background: &[],
                outcomes: None,
                resumed: false,
            });
            format!("{}\n\n{}", block, "list files under /tmp")
        };
        let messages_t1 = vec![Message::user(&u1_text)];
        let body_t1 = provider.build_request_body(
            &system,
            &messages_t1,
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // Simulate a `/permission unrestricted` toggle: in real code this happens on a different
        // thread via `SharedPermission::set`; here we just re-read the catalog and rebuild
        // everything to prove the outputs don't depend on the live permission state.

        let system_t2 = build_system_prompt(true, None);
        let tools_t2 = registry.definitions_active(&[]);

        let u2_text = {
            let block = build_turn_context(TurnContext {
                permission: Permission::Unrestricted,
                approvals: false,
                todos: &crate::todo::TodoState::default(),
                cwd: std::path::Path::new("."),
                roots: &[],
                world_state: "",
                budget: None,
                background: &[],
                outcomes: None,
                resumed: false,
            });
            format!("{}\n\n{}", block, "now write 'hi' to /tmp/out.txt")
        };
        let messages_t2 = vec![
            Message::user(&u1_text),
            Message::assistant_text("There are three files in /tmp."),
            Message::user(&u2_text),
        ];
        let body_t2 = provider.build_request_body(
            &system_t2,
            &messages_t2,
            &tools_t2,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // 1. The system prompt is identical. (Breakpoint 2 cache-hit.)
        assert_eq!(
            body_t1["system"], body_t2["system"],
            "system prompt diverged across /permission toggle: cache prefix invalidated"
        );

        // 2. The tools array is identical. (Breakpoint 3 cache-hit.) Reuse the existing helper
        //    which tolerates cache_control movement between the last-tool position across requests.
        assert_prefix_stable(&body_t1, &body_t2, 1);

        // 3. The turn-1 user message is preserved verbatim in turn-2's history. Historical messages
        //    must never mutate on toggle; otherwise breakpoint 4 (messages cache) cascades.
        let t1_msg = strip_cache_control(&body_t1["messages"][0]);
        let t2_msg0 = strip_cache_control(&body_t2["messages"][0]);
        assert_eq!(
            t1_msg, t2_msg0,
            "turn-1 user message changed after permission toggle"
        );

        // 4. Sanity: the two user messages do differ in their permission context (fresh content on
        //    each turn, not cached yet).
        assert!(u1_text.contains("Current permission level: read"));
        assert!(u2_text.contains("Current permission level: unrestricted"));
        assert_ne!(u1_text, u2_text);
    }

    /// `load_tool` activation must NOT mutate the cacheable system prompt. This is the regression
    /// guard for the deferred-tool refactor: when the model invokes `load_tool` to expose a
    /// deferred tool's schema, the system prompt block stays byte-identical (so breakpoint 2 cache
    /// hits); the tools array is what grows, append-only, so its prior entries also cache
    /// (breakpoint 3).
    ///
    /// Mirrors [`permission_toggle_preserves_cache_prefix`] structurally.
    #[tokio::test]
    async fn load_tool_preserves_system_prompt_cache() {
        use crate::{
            permission::{Permission, SharedPermission},
            prompt::{build_system_prompt, build_turn_context},
            store::Store,
            tools::ToolRegistry,
        };

        let store = Store::for_test().await;
        let shared_permission = SharedPermission::new(
            Permission::Unrestricted,
            crate::permission::EnabledPermissions::ALL,
        );
        let shared_session_id = crate::session::SharedSessionId::default();
        let todo_list = crate::todo::SharedTodoList::default();
        let registry = ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability: crate::sandbox::detect(),
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe: crate::sandbox::BackendProbe::Missing {
                        reason: "test fixture".to_string(),
                    },
                    builtin_filter: crate::config::BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                skills_agent_managed: false,
                memories: crate::store::memory::MemoryStore::detached(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: shared_session_id,
                todo_list,
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    shared_permission,
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    std::sync::Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("default web client config should build cleanly");
        // Register a deferred fixture *after* `build_default` so it lands at the tail of the tools
        // vector. Loading it later appends to the end of the API tools array, which is the
        // append-only growth shape the cache prefix invariant relies on.
        registry.register_deferred_fixture("fixture_deferred");

        let provider = provider_for_test();
        let system = build_system_prompt(true, None);

        // Turn 1: empty history, fixture_deferred not yet exposed.
        let u1_text = {
            let block = build_turn_context(crate::prompt::TurnContext {
                permission: Permission::Unrestricted,
                approvals: false,
                todos: &crate::todo::TodoState::default(),
                cwd: std::path::Path::new("."),
                roots: &[],
                world_state: "",
                budget: None,
                background: &[],
                outcomes: None,
                resumed: false,
            });
            format!("{}\n\n{}", block, "investigate scratchpad")
        };
        let messages_t1 = vec![Message::user(&u1_text)];
        let tools_t1 = registry.definitions_active(&messages_t1);
        let body_t1 = provider.build_request_body(
            &system,
            &messages_t1,
            &tools_t1,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        assert!(
            !tools_t1.iter().any(|t| t.name == "fixture_deferred"),
            "fixture_deferred should be deferred in turn 1"
        );

        // Turn 2: the model has called `load_tool` for fixture_deferred, so the next request should
        // expose its schema.
        let messages_t2 = vec![
            Message::user(&u1_text),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "load_tool".to_string(),
                    input: serde_json::json!({"name": "fixture_deferred"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "schema available".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        // System prompt is rebuilt the same way every turn; its content is a function of the
        // catalog, not the messages, so it must not shift when load_tool is invoked.
        let system_t2 = build_system_prompt(true, None);
        let tools_t2 = registry.definitions_active(&messages_t2);
        let body_t2 = provider.build_request_body(
            &system_t2,
            &messages_t2,
            &tools_t2,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // 1. The system prompt is byte-identical. (Breakpoint 2 cache-hit.)
        assert_eq!(
            body_t1["system"], body_t2["system"],
            "system prompt diverged across load_tool invocation: cache prefix invalidated"
        );

        // 2. The tools array gained fixture_deferred (append-only growth).
        assert!(
            tools_t2.iter().any(|t| t.name == "fixture_deferred"),
            "fixture_deferred should be active in turn 2 after load_tool"
        );
        assert_eq!(
            tools_t2.len(),
            tools_t1.len() + 1,
            "tools array should grow by exactly one entry after load_tool"
        );

        // 3. The prior tools (turn-1 set) are present in turn-2 in the same relative order, i.e.,
        //    the prefix is preserved. Stripping cache_control because the marker moves to the new
        //    last tool.
        let tools_arr_t1 =
            strip_tool_cache_control(body_t1["tools"].as_array().expect("tools array in body_t1"));
        let tools_arr_t2 =
            strip_tool_cache_control(body_t2["tools"].as_array().expect("tools array in body_t2"));
        for (index, tool) in tools_arr_t1.iter().enumerate() {
            assert_eq!(
                &tools_arr_t2[index], tool,
                "tool at index {index} mutated between turns: cache prefix invalidated"
            );
        }
    }

    /// An MCP server hot-swapping its tools (`tools/list_changed`) is the loudest mid-session
    /// change there is. It legitimately rewrites part of the tools array, but it must not touch the
    /// system prompt: that heads the cached prefix, so a byte moving there would re-cache the whole
    /// conversation on top of the tools-array cost.
    #[tokio::test]
    async fn mcp_tool_swap_leaves_the_system_prompt_untouched() {
        use crate::{
            permission::{Permission, SharedPermission},
            prompt::build_system_prompt,
            store::Store,
            tools::ToolRegistry,
        };

        let store = Store::for_test().await;
        let shared_permission = SharedPermission::new(
            Permission::Unrestricted,
            crate::permission::EnabledPermissions::ALL,
        );
        let registry = ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability: crate::sandbox::detect(),
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe: crate::sandbox::BackendProbe::Missing {
                        reason: "test fixture".to_string(),
                    },
                    builtin_filter: crate::config::BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                skills_agent_managed: false,
                memories: crate::store::memory::MemoryStore::detached(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: crate::session::SharedSessionId::default(),
                todo_list: crate::todo::SharedTodoList::default(),
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    shared_permission,
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    std::sync::Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("default web client config should build cleanly");
        registry.register_deferred_fixture("mcp__fs__old_tool");

        let provider = provider_for_test();
        let before_system = build_system_prompt(true, None);
        let before_catalog = registry.tool_catalog();

        // The server reconnects and advertises a different tool set.
        registry.replace_server_tools("fs", vec![std::sync::Arc::new(
            crate::tools::FixtureDeferredTool {
                name: "mcp__fs__new_tool".to_string(),
            },
        )]);

        let after_system = build_system_prompt(true, None);
        let after_catalog = registry.tool_catalog();

        assert_eq!(
            before_system, after_system,
            "the system prompt must survive an MCP tool swap: it heads the cached prefix",
        );
        assert_ne!(
            before_catalog, after_catalog,
            "the swap must actually be visible somewhere, or this test proves nothing",
        );

        // ...and the change is announced in the block that gets appended instead.
        let memories: Vec<crate::memory::Memory> = Vec::new();
        let delta = crate::prompt::render_world_state(
            &crate::prompt::WorldSnapshot::new(
                &after_catalog,
                &crate::skills::SkillIndex::default(),
                &memories,
                &[],
                &[],
            ),
            Some(&crate::prompt::WorldSnapshot::new(
                &before_catalog,
                &crate::skills::SkillIndex::default(),
                &memories,
                &[],
                &[],
            )),
        );
        assert!(
            delta.contains("mcp__fs__new_tool"),
            "the new tool must be announced; got: {delta}",
        );
        assert!(
            delta.contains("No longer available, do not call: `mcp__fs__old_tool`"),
            "the withdrawn tool must be retracted, or the model keeps calling it; got: {delta}",
        );

        // Both request bodies still agree on the cached head.
        let body_before = provider.build_request_body(
            &before_system,
            &[Message::user("hi")],
            &registry.definitions_active(&[]),
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let body_after = provider.build_request_body(
            &after_system,
            &[Message::user("hi")],
            &registry.definitions_active(&[]),
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert_eq!(body_before["system"], body_after["system"]);
    }

    /// Compaction must not silently drop the deferred-tool active set. Pre-compaction, the model
    /// loads a deferred fixture via `load_tool`; post-compaction, the
    /// `Event::CompactBoundary::loaded_tools_snapshot` must keep the loaded tool in the API tools
    /// array even though the pre-compaction `load_tool` rows have moved below the materialized
    /// view's logical start.
    #[tokio::test]
    async fn compaction_preserves_loaded_tools_active_set() {
        use crate::{
            conversation::{Conversation, Event},
            permission::{Permission, SharedPermission},
            store::Store,
            tools::{ToolRegistry, load_tool::extract_loaded_tool_names_from_events},
        };

        let store = Store::for_test().await;
        let shared_permission = SharedPermission::new(
            Permission::Unrestricted,
            crate::permission::EnabledPermissions::ALL,
        );
        let shared_session_id = crate::session::SharedSessionId::default();
        let todo_list = crate::todo::SharedTodoList::default();
        let registry = ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability: crate::sandbox::detect(),
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe: crate::sandbox::BackendProbe::Missing {
                        reason: "test fixture".to_string(),
                    },
                    builtin_filter: crate::config::BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                skills_agent_managed: false,
                memories: crate::store::memory::MemoryStore::detached(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: shared_session_id,
                todo_list,
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    shared_permission,
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    std::sync::Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("default web client config should build cleanly");
        registry.register_deferred_fixture("fixture_deferred");

        // Pre-compaction: load fixture_deferred via load_tool.
        let mut log = Conversation::new();
        log.append(Message::user("question 1"));
        log.append(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "u1".to_string(),
                name: "load_tool".to_string(),
                input: serde_json::json!({"name": "fixture_deferred"}),
            }],
        });
        log.append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "loaded".to_string(),
                }],
                is_error: false,
            }],
        });

        let pre_loaded = extract_loaded_tool_names_from_events(log.events());
        assert!(pre_loaded.iter().any(|name| name == "fixture_deferred"));
        let pre_tools = registry.definitions_active_with_loaded(&pre_loaded);
        assert!(pre_tools.iter().any(|t| t.name == "fixture_deferred"));

        // Compact: the snapshot must carry the loaded set forward.
        log.replace_for_compaction(
            Message::user("[summary]"),
            vec![Message::user("question 2")],
            pre_loaded.iter().cloned().collect(),
        );

        // The materialized view shrank, but events are append-only.
        let post_loaded = extract_loaded_tool_names_from_events(log.events());
        assert!(
            post_loaded.iter().any(|name| name == "fixture_deferred"),
            "compaction must preserve the loaded-tools active set via the snapshot"
        );

        // The active tool set the agent sends to the API still includes fixture_deferred
        // post-compaction.
        let post_tools = registry.definitions_active_with_loaded(&post_loaded);
        assert!(post_tools.iter().any(|t| t.name == "fixture_deferred"));

        // The post-compaction event log must have grown, never shrunk.
        let boundary_count = log
            .events()
            .iter()
            .filter(|e| matches!(e, Event::CompactBoundary { .. }))
            .count();
        assert_eq!(boundary_count, 1);
        let append_count = log
            .events()
            .iter()
            .filter(|e| matches!(e, Event::Append(_)))
            .count();
        // 3 pre-compaction Appends + 1 tail Append = 4.
        assert_eq!(append_count, 4);
    }

    /// Same invariant, but exercises every pairwise permission toggle (16 combinations). Catches
    /// any permission state that sneaks back into the cacheable prefix.
    #[tokio::test]
    async fn permission_independence_all_levels() {
        use crate::{
            permission::{Permission, SharedPermission},
            prompt::build_system_prompt,
            store::Store,
            tools::ToolRegistry,
        };

        let store = Store::for_test().await;
        let shared_permission =
            SharedPermission::new(Permission::Read, crate::permission::EnabledPermissions::ALL);
        let shared_session_id = crate::session::SharedSessionId::default();
        let todo_list = crate::todo::SharedTodoList::default();
        let registry = ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability: crate::sandbox::detect(),
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe: crate::sandbox::BackendProbe::Missing {
                        reason: "test fixture".to_string(),
                    },
                    builtin_filter: crate::config::BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                skills_agent_managed: false,
                memories: crate::store::memory::MemoryStore::detached(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: shared_session_id,
                todo_list,
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    shared_permission.clone(),
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    std::sync::Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("default web client config should build cleanly");

        let provider = provider_for_test();
        // All four. The property is most load-bearing for `Workspace`: it sits between `read` and
        // `unrestricted` in the Shift+Tab cycle, so a user reaching `unrestricted` passes through
        // it and would invalidate the cached prefix on the way if the array moved.
        let levels = [
            Permission::None,
            Permission::Read,
            Permission::Workspace,
            Permission::Unrestricted,
        ];

        let mut bodies = Vec::with_capacity(levels.len());
        for &level in &levels {
            shared_permission.set_unchecked(level);
            let system = build_system_prompt(true, None);
            let tools = registry.definitions_active(&[]);
            let messages = vec![Message::user("hello")];
            assert!(
                !tools.is_empty(),
                "{level} produced no tools, so the pairwise equality below would be vacuous"
            );
            bodies.push(provider.build_request_body(
                &system,
                &messages,
                &tools,
                true,
                ThinkingOverride::Inherit,
                &crate::provider::Attribution::default(),
            ));
        }

        // Every pair must agree on the cacheable prefix.
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                assert_eq!(
                    bodies[i]["system"], bodies[j]["system"],
                    "system prompt differs between {:?} and {:?}",
                    levels[i], levels[j]
                );
                assert_prefix_stable(&bodies[i], &bodies[j], 1);
            }
        }
    }

    #[test]
    fn tool_loop_prefix_is_stable() {
        let provider = provider_for_test();
        let system = "You are a helpful assistant.";
        let tools = tools_for_test();

        // Iteration 1 of tool loop: user asks, model about to respond
        let messages_iter1 = vec![Message::user("Read /tmp/test.txt")];
        let body_iter1 = provider.build_request_body(
            system,
            &messages_iter1,
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // Iteration 2: model made a tool call, tool result came back
        let messages_iter2 = vec![
            Message::user("Read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "hello world".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body_iter2 = provider.build_request_body(
            system,
            &messages_iter2,
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // Iteration 3: model made another tool call
        let messages_iter3 = vec![
            Message::user("Read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "hello world".to_string(),
                    }],
                    is_error: false,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_2".to_string(),
                    name: "execute_command".to_string(),
                    input: serde_json::json!({"command": "wc -l /tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_2".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "1 /tmp/test.txt".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body_iter3 = provider.build_request_body(
            system,
            &messages_iter3,
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // Prefix from iter1 is stable in iter2 and iter3
        assert_prefix_stable(&body_iter1, &body_iter2, 1);
        assert_prefix_stable(&body_iter2, &body_iter3, 3);
        assert_prefix_stable(&body_iter1, &body_iter3, 1);
    }

    #[test]
    fn exactly_one_message_cache_control_per_request() {
        let provider = provider_for_test();
        let system = "You are a helpful assistant.";
        let tools = tools_for_test();

        // Single message
        let body1 = provider.build_request_body(
            system,
            &[Message::user("hello")],
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert_eq!(count_message_cache_controls(&body1), 1);

        // Three messages
        let body3 = provider.build_request_body(
            system,
            &[
                Message::user("hello"),
                Message::assistant_text("hi"),
                Message::user("bye"),
            ],
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert_eq!(count_message_cache_controls(&body3), 1);

        // Five messages with tool use
        let body5 = provider.build_request_body(
            system,
            &[
                Message::user("read file"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "t1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "/tmp/x"}),
                    }],
                },
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "t1".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "data".to_string(),
                        }],
                        is_error: false,
                    }],
                },
                Message::assistant_text("Here's the file."),
                Message::user("thanks"),
            ],
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert_eq!(count_message_cache_controls(&body5), 1);
    }

    #[test]
    fn cache_control_shifts_to_new_last_message() {
        let provider = provider_for_test();
        let system = "system";

        // Build with 2 messages: cache_control should be on message[1]
        let messages_a = vec![Message::user("hello"), Message::assistant_text("hi")];
        let body_a = provider.build_request_body(
            system,
            &messages_a,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let msgs_a = body_a["messages"].as_array().unwrap();

        // Message 0 should NOT have cache_control
        let block_0 = &msgs_a[0]["content"].as_array().unwrap()[0];
        assert!(block_0.get("cache_control").is_none());
        // Message 1 (last) SHOULD have cache_control
        let block_1 = &msgs_a[1]["content"].as_array().unwrap()[0];
        assert!(block_1.get("cache_control").is_some());

        // Now append a third message: cache_control should move to message[2]
        let messages_b = vec![
            Message::user("hello"),
            Message::assistant_text("hi"),
            Message::user("bye"),
        ];
        let body_b = provider.build_request_body(
            system,
            &messages_b,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let msgs_b = body_b["messages"].as_array().unwrap();

        // Messages 0 and 1 should NOT have cache_control
        assert!(
            msgs_b[0]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            msgs_b[1]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        // Message 2 (new last) SHOULD have cache_control
        assert!(
            msgs_b[2]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_some()
        );
    }

    #[test]
    fn system_prompt_identical_across_turns() {
        let provider = provider_for_test();
        let system = "You are a helpful assistant.";
        let tools = tools_for_test();

        let body1 = provider.build_request_body(
            system,
            &[Message::user("turn 1")],
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let body2 = provider.build_request_body(
            system,
            &[
                Message::user("turn 1"),
                Message::assistant_text("response 1"),
                Message::user("turn 2"),
            ],
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let body3 = provider.build_request_body(
            system,
            &[
                Message::user("turn 1"),
                Message::assistant_text("response 1"),
                Message::user("turn 2"),
                Message::assistant_text("response 2"),
                Message::user("turn 3"),
            ],
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // System prompt must be byte-identical across all turns.
        assert_eq!(body1["system"], body2["system"]);
        assert_eq!(body2["system"], body3["system"]);

        // Model, max_tokens, metadata must also be identical.
        assert_eq!(body1["model"], body2["model"]);
        assert_eq!(body1["max_tokens"], body2["max_tokens"]);
        assert_eq!(body1["metadata"], body2["metadata"]);
        assert_eq!(body2["model"], body3["model"]);
        assert_eq!(body2["max_tokens"], body3["max_tokens"]);
        assert_eq!(body2["metadata"], body3["metadata"]);
    }

    #[test]
    fn tool_schemas_stable_across_turns() {
        let provider = provider_for_test();
        let tools = tools_for_test();

        let body1 = provider.build_request_body(
            "system",
            &[Message::user("a")],
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let body2 = provider.build_request_body(
            "system",
            &[
                Message::user("a"),
                Message::assistant_text("b"),
                Message::user("c"),
            ],
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );

        // Tool schemas (including cache_control on the last tool) must be identical when the same
        // tools are provided.
        assert_eq!(body1["tools"], body2["tools"]);
    }

    #[test]
    fn multi_block_message_cache_control_on_last_block_only() {
        let provider = provider_for_test();

        // An assistant message with text + tool_use (multiple blocks)
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Let me read that file.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                },
            ],
        }];
        let body = provider.build_request_body(
            "system",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let message = &body["messages"].as_array().unwrap()[0];
        let blocks = message["content"].as_array().unwrap();

        // First block (text) should NOT have cache_control
        assert!(blocks[0].get("cache_control").is_none());
        // Second block (tool_use, the last block of the last message) SHOULD
        assert!(blocks[1].get("cache_control").is_some());
    }

    #[test]
    fn long_conversation_prefix_stability() {
        let provider = provider_for_test();
        let system = "You are a helpful assistant.";
        let tools = tools_for_test();

        // Build up a 10-turn conversation incrementally and verify each step preserves the prefix
        // from the previous step.
        let mut messages: Vec<Message> = Vec::new();
        let mut previous: Option<(serde_json::Value, usize)> = None;

        for turn in 0..10 {
            messages.push(Message::user(format!("User message {turn}")));
            let body = provider.build_request_body(
                system,
                &messages,
                &tools,
                true,
                ThinkingOverride::Inherit,
                &crate::provider::Attribution::default(),
            );

            if let Some((prev_body, prev_msg_count)) = &previous {
                // The shared prefix is exactly the messages that were in the previous request body.
                assert_prefix_stable(prev_body, &body, *prev_msg_count);
            }

            assert_eq!(count_message_cache_controls(&body), 1);

            let message_count = messages.len();
            // Simulate assistant response
            messages.push(Message::assistant_text(format!("Response {turn}")));
            previous = Some((body, message_count));
        }
    }

    #[test]
    fn tool_loop_with_multiple_sequential_calls() {
        let provider = provider_for_test();
        let system = "system";
        let tools = tools_for_test();

        // Simulate a user request that triggers 4 sequential tool calls. Each iteration of the loop
        // adds an assistant tool_use + user tool_result pair. Verify the prefix is stable across
        // all iterations.
        let mut messages: Vec<Message> = vec![Message::user("do several things")];

        let mut previous_body: Option<serde_json::Value> = None;
        let mut previous_len = 0;

        for i in 0..4 {
            let body = provider.build_request_body(
                system,
                &messages,
                &tools,
                true,
                ThinkingOverride::Inherit,
                &crate::provider::Attribution::default(),
            );

            if let Some(prev) = &previous_body {
                assert_prefix_stable(prev, &body, previous_len);
            }

            assert_eq!(
                count_message_cache_controls(&body),
                1,
                "iteration {i} should have exactly 1 message cache_control"
            );

            previous_len = messages.len();
            previous_body = Some(body);

            // Simulate tool call and result
            messages.push(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: format!("toolu_{i}"),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": format!("/tmp/file{}", i)}),
                }],
            });
            messages.push(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: format!("toolu_{i}"),
                    content: vec![ToolResultContent::Text {
                        text: format!("contents of file{i}"),
                    }],
                    is_error: false,
                }],
            });
        }

        // Final body after all tool calls
        let final_body = provider.build_request_body(
            system,
            &messages,
            &tools,
            true,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert_prefix_stable(previous_body.as_ref().unwrap(), &final_body, previous_len);
        assert_eq!(count_message_cache_controls(&final_body), 1);
    }

    #[test]
    fn empty_messages_produces_no_cache_control() {
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "system",
            &[],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        assert_eq!(count_message_cache_controls(&body), 0);
        assert!(body["messages"].as_array().unwrap().is_empty());
    }

    #[test]
    fn cache_control_on_tool_result_block() {
        let provider = provider_for_test();

        // When the last message is a tool_result, cache_control should still appear on its last
        // content block.
        let messages = vec![
            Message::user("read file"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "file data".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body = provider.build_request_body(
            "system",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let messages = body["messages"].as_array().unwrap();

        // Only the tool_result message (last) should have cache_control
        assert!(
            messages[0]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            messages[1]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            messages[2]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_some()
        );
        assert_eq!(count_message_cache_controls(&body), 1);
    }

    #[test]
    fn claude_cache_control_on_last_message_only() {
        let provider = provider_for_test();

        let messages = vec![
            Message::user("first"),
            Message::assistant_text("response"),
            Message::user("second"),
        ];
        let body = provider.build_request_body(
            "system",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let claude_messages = body["messages"].as_array().unwrap();

        let first_content = claude_messages[0]["content"].as_array().unwrap();
        assert!(first_content[0].get("cache_control").is_none());

        let second_content = claude_messages[1]["content"].as_array().unwrap();
        assert!(second_content[0].get("cache_control").is_none());

        let third_content = claude_messages[2]["content"].as_array().unwrap();
        assert!(third_content[0].get("cache_control").is_some());
    }

    #[test]
    fn claude_tools_carry_no_cache_control() {
        let provider = provider_for_test();

        let tools = vec![
            ToolDefinition::new(
                "read_file".to_string(),
                "Read a file".to_string(),
                serde_json::json!({"type": "object"}),
            ),
            ToolDefinition::new(
                "write_file".to_string(),
                "Write a file".to_string(),
                serde_json::json!({"type": "object"}),
            ),
        ];
        let body = provider.build_request_body(
            "system",
            &[Message::user("hi")],
            &tools,
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let claude_tools = body["tools"].as_array().unwrap();

        // No tool carries cache_control: the rolling last-message breakpoint caches the
        // tools+system prefix, matching the captured Claude Code CLI wire.
        assert!(claude_tools[0].get("cache_control").is_none());
        assert!(claude_tools[1].get("cache_control").is_none());
    }

    #[test]
    fn claude_no_message_cache_control_when_empty() {
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "system",
            &[],
            &[],
            false,
            ThinkingOverride::Inherit,
            &crate::provider::Attribution::default(),
        );
        let claude_messages = body["messages"].as_array().unwrap();
        assert!(claude_messages.is_empty());
    }

    /// A minimal in-process OAuth refresh endpoint that counts hits. Returns a valid refresh
    /// response on every call so the provider path completes; the test then asserts the hit count.
    /// `state_expiry` distinguishes a well-behaved issuer from one that answers without an
    /// `expires_in`, which would otherwise make the token due again on arrival.
    async fn run_mock_refresh_endpoint(
        listener: tokio::net::TcpListener,
        hits: Arc<std::sync::atomic::AtomicUsize>,
        states_expiry: bool,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(_) => return,
            };
            let hits = Arc::clone(&hits);
            tokio::spawn(async move {
                // Drain enough of the request to know we got a full POST body. The OAuth endpoint
                // sends a small JSON body; read until we see two CRLFs (header end) and then
                // enough bytes to satisfy Content-Length.
                let mut buffer = Vec::with_capacity(2048);
                let mut headers_end: Option<usize> = None;
                let mut content_length: Option<usize> = None;
                while headers_end.is_none() {
                    let mut chunk = [0u8; 1024];
                    let n = match socket.read(&mut chunk).await {
                        Ok(0) => return,
                        Ok(n) => n,
                        Err(_) => return,
                    };
                    buffer.extend_from_slice(&chunk[..n]);
                    if let Some(index) = find_crlf_crlf(&buffer) {
                        headers_end = Some(index);
                        content_length = parse_content_length(&buffer[..index]);
                    }
                }
                if let (Some(end), Some(len)) = (headers_end, content_length) {
                    let body_start = end + 4;
                    while buffer.len() < body_start + len {
                        let mut chunk = [0u8; 1024];
                        let n = match socket.read(&mut chunk).await {
                            Ok(0) => break,
                            Ok(n) => n,
                            Err(_) => return,
                        };
                        buffer.extend_from_slice(&chunk[..n]);
                    }
                }
                hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                let mut body = serde_json::json!({
                    "access_token": "fresh-token-xyz",
                    "refresh_token": "fresh-refresh",
                });
                if states_expiry && let Some(object) = body.as_object_mut() {
                    object.insert("expires_in".to_string(), serde_json::json!(3600));
                }
                let body = body.to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    }

    /// Answer every refresh with one fixed status and body, so a test can pin what the classifier
    /// is handed. `truncate` overstates `Content-Length`, which is how a *successful* exchange
    /// whose body never arrives is reproduced.
    async fn run_mock_refusing_refresh_endpoint(
        listener: tokio::net::TcpListener,
        status_line: &'static str,
        body: &'static str,
        truncate: bool,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        while let Ok((mut socket, _)) = listener.accept().await {
            let mut buffer = Vec::with_capacity(2048);
            loop {
                let mut chunk = [0u8; 1024];
                let read = match socket.read(&mut chunk).await {
                    Ok(0) => break,
                    Ok(read) => read,
                    Err(_) => return,
                };
                buffer.extend_from_slice(&chunk[..read]);
                let Some(end) = find_crlf_crlf(&buffer) else {
                    continue;
                };
                if buffer.len() >= end + 4 + parse_content_length(&buffer[..end]).unwrap_or(0) {
                    break;
                }
            }

            let declared = if truncate {
                body.len() + 64
            } else {
                body.len()
            };
            let response = format!(
                "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {declared}\r\n\r\n{body}"
            );
            if socket.write_all(response.as_bytes()).await.is_err() {
                return;
            }
            if socket.shutdown().await.is_err() {
                tracing::debug!("mock refresh endpoint failed to shut its socket down cleanly");
            }
        }
    }

    /// What the token endpoint answered decides the classification, at the site rather than only in
    /// the classifier.
    ///
    /// The wiring rather than the rule: `crate::error::oauth_refresh_error` has its own tests, and
    /// this asserts that the refresh path calls it. A bare `MekaError::Provider` is dropped by the
    /// agent loop, so a 503 from the token endpoint would end the turn while a 503 from the
    /// messages endpoint two lines later is retried; and since `ensure_valid_credential` runs
    /// inside `complete` and `stream`, that is a live turn lost to an outage with nothing to do
    /// with the conversation. The 400 is the opposite failure: it must *not* be retried, and it has
    /// to name the profile, because two accounts of one backend can coexist and "log in again"
    /// alone does not say where.
    #[tokio::test]
    async fn what_the_token_endpoint_answered_decides_whether_a_refresh_is_retried() {
        for (status_line, body, truncate, retryable) in [
            (
                "503 Service Unavailable",
                r#"{"error":"temporarily_unavailable"}"#,
                false,
                true,
            ),
            (
                "400 Bad Request",
                r#"{"error":"invalid_grant"}"#,
                false,
                false,
            ),
            // A success meka cannot read back: the grant is spent and only a login recovers it.
            ("200 OK", r#"{"access_token":"#, true, false),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind mock OAuth endpoint");
            let local = listener.local_addr().expect("local addr");
            tokio::spawn(run_mock_refusing_refresh_endpoint(
                listener,
                status_line,
                body,
                truncate,
            ));

            let credential = AuthCredential::OAuthToken {
                access_token: "stale".to_string(),
                refresh_token: Some("rt".to_string()),
                expires_at: Some(crate::oauth::now_epoch_millis()),
                account_id: None,
            };
            let provider = ClaudeSubscriptionProvider::new(
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::ClaudeSubscription,
                    credential,
                    "claude-sonnet-4-20250514".to_string(),
                )
                .base_url(None)
                .client_id(None)
                .oauth_token_url(Some(format!("http://{local}/")))
                .token_store(None)
                .credential_key(Some("work".to_string()))
                .thinking(ThinkingMode::Off, 10000)
                .device_id("a".repeat(64))
                .effort(None)
                .thinking_display(crate::config::ThinkingDisplay::Summarized)
                .max_output_tokens(None)
                .max_request_bytes(None),
            )
            .expect("build test provider");

            let error = provider
                .ensure_valid_credential()
                .await
                .expect_err("the endpoint did not hand back a usable token");

            // The prefix is asserted as well as the variant because the two backends now compose
            // their messages from one shared exchange: a reworded `context` would change what a
            // user greps for in both at once, silently.
            match error {
                MekaError::RetryableProvider { message, .. } if retryable => assert!(
                    message.starts_with("OAuth token refresh failed ("),
                    "{status_line}: {message}"
                ),
                MekaError::Provider(message) if !retryable => {
                    assert!(
                        message.starts_with("OAuth token refresh"),
                        "{status_line}: {message}"
                    );
                    assert!(
                        message.contains("meka account login work"),
                        "{status_line} should name the profile to log in to: {message}"
                    );
                }
                other => panic!("{status_line} was classified wrongly: {other}"),
            }
        }
    }

    fn find_crlf_crlf(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn parse_content_length(headers: &[u8]) -> Option<usize> {
        let headers = std::str::from_utf8(headers).ok()?;
        for line in headers.split("\r\n") {
            if let Some((name, value)) = line.split_once(':')
                && name.trim().eq_ignore_ascii_case("content-length")
            {
                return value.trim().parse().ok();
            }
        }
        None
    }

    /// When many tasks hit `ensure_valid_credential` against a near-expiry credential at the same
    /// instant, exactly **one** refresh API call must fire. The remaining tasks observe the refresh
    /// that already happened via the post-write-lock re-check inside `ensure_valid_credential` and
    /// return the fresh token without re-firing the refresh. This is the invariant relied on by
    /// multi-session ACP where two sessions can race the same credential at the same time.
    #[tokio::test]
    async fn oauth_refresh_fires_once_under_concurrent_demand() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock OAuth endpoint");
        let local = listener.local_addr().expect("local addr");
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        tokio::spawn(run_mock_refresh_endpoint(listener, Arc::clone(&hits), true));

        // Credential whose access token already counts as "expiring soon" (the threshold is 5
        // minutes / 300_000 ms). Setting expires_at to "now" forces every caller into the slow path
        // immediately.
        let credential = AuthCredential::OAuthToken {
            access_token: "stale".to_string(),
            refresh_token: Some("rt".to_string()),
            expires_at: Some(crate::oauth::now_epoch_millis()),
            account_id: None,
        };

        let provider = Arc::new(
            ClaudeSubscriptionProvider::new(
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::ClaudeSubscription,
                    credential,
                    "claude-sonnet-4-20250514".to_string(),
                )
                .base_url(None)
                .client_id(None)
                .oauth_token_url(Some(format!("http://{local}/")))
                .token_store(None)
                .credential_key(Some("test".to_string()))
                .thinking(ThinkingMode::Off, 10000)
                .device_id("a".repeat(64))
                .effort(Some("high".to_string()))
                .thinking_display(crate::config::ThinkingDisplay::Summarized)
                .max_output_tokens(None)
                .max_request_bytes(None),
            )
            .expect("build test provider"),
        );

        // Fire many concurrent callers. The exact count isn't load- bearing; we just want enough to
        // make a fan-out plausible if the gate broke.
        let mut handles = Vec::new();
        for _ in 0..16 {
            let provider = Arc::clone(&provider);
            handles.push(tokio::spawn(async move {
                provider
                    .ensure_valid_credential()
                    .await
                    .map(|(_, value)| value)
            }));
        }
        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            results.push(handle.await.expect("join").expect("ensure_valid"));
        }

        // Every caller must return the same fresh token, which proves they observed the refresh
        // that landed and didn't double-refresh.
        for header in &results {
            assert_eq!(header, "Bearer fresh-token-xyz", "stale token leaked",);
        }

        let observed_hits = hits.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            observed_hits, 1,
            "exactly one refresh API call must fire under concurrent demand; got {observed_hits}",
        );
    }

    /// An issuer that answers a refresh without an `expires_in` must not put every later request
    /// back through the refresh.
    ///
    /// `expires_at: None` reads as due, which is right for a *stored* token of unknown age and
    /// wrong for one that has just been minted. Handing the `None` straight back meant the
    /// credential was due again the instant it arrived, so every request took the write lock,
    /// re-read the database and ran a full OAuth round trip -- serialized, and rotating the
    /// refresh token each pass, which is the state most likely to end in an `invalid_grant`
    /// nobody can explain.
    #[tokio::test]
    async fn a_refresh_without_an_expiry_does_not_refresh_again_on_the_next_request() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock OAuth endpoint");
        let local = listener.local_addr().expect("local addr");
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        tokio::spawn(run_mock_refresh_endpoint(
            listener,
            Arc::clone(&hits),
            false,
        ));

        let credential = AuthCredential::OAuthToken {
            access_token: "stale".to_string(),
            refresh_token: Some("rt".to_string()),
            expires_at: Some(crate::oauth::now_epoch_millis()),
            account_id: None,
        };
        let provider = ClaudeSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ClaudeSubscription,
                credential,
                "claude-sonnet-4-20250514".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(Some(format!("http://{local}/")))
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .thinking(ThinkingMode::Off, 10000)
            .device_id("a".repeat(64))
            .effort(Some("high".to_string()))
            .thinking_display(crate::config::ThinkingDisplay::Summarized)
            .max_output_tokens(None)
            .max_request_bytes(None),
        )
        .expect("build test provider");

        for _ in 0..3 {
            let (_, header) = provider
                .ensure_valid_credential()
                .await
                .expect("ensure_valid");
            assert_eq!(header, "Bearer fresh-token-xyz");
        }

        assert_eq!(
            hits.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the token the refresh returned must be usable without refreshing it again",
        );
    }

    /// One listener standing in for both the API and the token endpoint, so a request that is
    /// refused, refreshed and sent again can be watched end to end.
    struct RejectingApi {
        /// How many `/v1/messages` calls to answer 401 before answering 200.
        reject_first: usize,
        messages_hits: std::sync::atomic::AtomicUsize,
        token_hits: std::sync::atomic::AtomicUsize,
        /// The `Authorization` header of each `/v1/messages` call, in order.
        authorizations: std::sync::Mutex<Vec<String>>,
    }

    async fn run_rejecting_api(listener: tokio::net::TcpListener, api: Arc<RejectingApi>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(_) => return,
            };
            let api = Arc::clone(&api);
            tokio::spawn(async move {
                let mut buffer = Vec::with_capacity(4096);
                loop {
                    let mut chunk = [0u8; 2048];
                    let read = match socket.read(&mut chunk).await {
                        Ok(0) => return,
                        Ok(read) => read,
                        Err(_) => return,
                    };
                    buffer.extend_from_slice(&chunk[..read]);
                    if let Some(end) = find_crlf_crlf(&buffer)
                        && buffer.len()
                            >= end + 4 + parse_content_length(&buffer[..end]).unwrap_or(0)
                    {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&buffer).to_string();
                let path = head
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_string();
                let authorization = head
                    .lines()
                    .find(|line| line.to_ascii_lowercase().starts_with("authorization:"))
                    .and_then(|line| line.split_once(':'))
                    .map(|(_, value)| value.trim().to_string())
                    .unwrap_or_default();
                let (status, reason, body) = if path.starts_with("/v1/oauth/token") {
                    api.token_hits
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    (
                        200,
                        "OK",
                        r#"{"access_token":"fresh-token-xyz","refresh_token":"fresh-refresh","expires_in":3600}"#,
                    )
                } else {
                    let hit = api
                        .messages_hits
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    api.authorizations.lock().expect("lock").push(authorization);
                    if hit < api.reject_first {
                        (
                            401,
                            "Unauthorized",
                            r#"{"type":"error","error":{"type":"authentication_error","message":"invalid token"}}"#,
                        )
                    } else {
                        (
                            200,
                            "OK",
                            r#"{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#,
                        )
                    }
                };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    }

    /// A provider against the listener at `local`, holding a token the store believes is good for
    /// another hour. Only the backend's word says otherwise.
    fn provider_against(local: std::net::SocketAddr) -> ClaudeSubscriptionProvider {
        ClaudeSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ClaudeSubscription,
                AuthCredential::OAuthToken {
                    access_token: "stale-token".to_string(),
                    refresh_token: Some("rt".to_string()),
                    expires_at: Some(crate::oauth::now_epoch_millis() + 3_600_000),
                    account_id: None,
                },
                "claude-sonnet-4-20250514".to_string(),
            )
            .base_url(Some(format!("http://{local}")))
            .client_id(None)
            .oauth_token_url(Some(format!("http://{local}/v1/oauth/token")))
            .token_store(None)
            .credential_key(Some("work".to_string()))
            .thinking(ThinkingMode::Off, 10000)
            .device_id("a".repeat(64))
            .effort(None)
            .thinking_display(crate::config::ThinkingDisplay::Summarized)
            .max_output_tokens(None)
            .max_request_bytes(None),
        )
        .expect("build test provider")
    }

    /// A 401 is the backend saying the stored expiry is wrong. The token is refreshed once and the
    /// same call sent again on the new one, so a revoked or shortened token costs one round trip
    /// rather than every turn until the expiry meka believed in has passed.
    #[tokio::test]
    async fn a_rejected_token_is_refreshed_once_and_the_call_sent_again() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock API");
        let local = listener.local_addr().expect("local addr");
        let api = Arc::new(RejectingApi {
            reject_first: 1,
            messages_hits: std::sync::atomic::AtomicUsize::new(0),
            token_hits: std::sync::atomic::AtomicUsize::new(0),
            authorizations: std::sync::Mutex::new(Vec::new()),
        });
        tokio::spawn(run_rejecting_api(listener, Arc::clone(&api)));

        let provider = provider_against(local);
        let message = provider
            .complete(CompletionRequest::new("", &[Message::user("hi")], &[]))
            .await
            .expect("the call sent again on the refreshed token succeeds")
            .message;
        assert_eq!(message.text_content(), "hi");
        assert_eq!(
            api.messages_hits.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "one refused call, one retried"
        );
        assert_eq!(
            api.token_hits.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "exactly one refresh"
        );
        assert_eq!(
            *api.authorizations.lock().expect("lock"),
            vec![
                "Bearer stale-token".to_string(),
                "Bearer fresh-token-xyz".to_string()
            ],
            "the retry carries the refreshed token, not the refused one"
        );
    }

    /// A second refusal is the account, not the token: the grant behind the refresh is gone and
    /// only signing in again produces one the backend will take, so the error says so rather than
    /// refreshing forever.
    #[tokio::test]
    async fn a_token_rejected_twice_names_the_login_remedy() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock API");
        let local = listener.local_addr().expect("local addr");
        let api = Arc::new(RejectingApi {
            reject_first: usize::MAX,
            messages_hits: std::sync::atomic::AtomicUsize::new(0),
            token_hits: std::sync::atomic::AtomicUsize::new(0),
            authorizations: std::sync::Mutex::new(Vec::new()),
        });
        tokio::spawn(run_rejecting_api(listener, Arc::clone(&api)));

        let provider = provider_against(local);
        let error = provider
            .complete(CompletionRequest::new("", &[Message::user("hi")], &[]))
            .await
            .expect_err("a token refused after its refresh is a failure");
        match error {
            MekaError::Provider(message) => assert!(
                message.contains("meka account login work"),
                "the remedy names the profile: {message}"
            ),
            other => panic!("expected a provider error naming the remedy, got {other:?}"),
        }
        assert_eq!(
            api.messages_hits.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "refused, refreshed, refused again, and no third attempt"
        );
        assert_eq!(api.token_hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
