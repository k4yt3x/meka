//! `chatgpt-subscription`: the Responses API billed to a ChatGPT subscription.
//!
//! Talks the Responses API to `chatgpt.com/backend-api/codex/responses`, authenticated by the
//! bearer token + `ChatGPT-Account-ID` header issued by the Codex OAuth flow. Mirrors how OpenAI's
//! own first-party Codex CLI authenticates so the wire shape matches.
//!
//! The protocol itself lives in [`super::responses_wire`], shared with the API-key
//! [`super::responses`] backend. What is particular to this one is the endpoint, the OAuth
//! credential, the Codex client headers, and the two reasoning parameters Codex sends: the
//! `include` of encrypted reasoning content and the `reasoning.summary` that makes the reasoning
//! visible. Those two stay here because this backend's endpoint is always ChatGPT.

mod auth;

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use self::auth::{extract_account_id, extract_expiration_seconds};
use super::responses_wire::{
    build_request_body, include_encrypted_reasoning, request_reasoning_summary,
};
use crate::{
    conversation::Message,
    error::{MekaError, Result},
    provider::{
        AccountIdentity, AccountUsage, CompletionRequest, DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID,
        DailyUsage, ExtraUsage, Provider, StreamEvent, ToolDefinition, UsageHistory, UsageWindow,
    },
    store::{AuthCredential, TokenStore},
};

/// Default OAuth token endpoint. Refresh requests POST here as JSON.
const DEFAULT_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// `originator` request header value. Mirrors Codex's `codex_cli_rs` slot, flagged as the calling
/// tool so OpenAI can attribute traffic.
const ORIGINATOR: &str = "meka_cli";

/// The `chatgpt-subscription` backend: one profile's model and the OAuth credential it bills.
pub(crate) struct ChatGptSubscriptionProvider {
    client: reqwest::Client,
    credential: tokio::sync::RwLock<AuthCredential>,
    /// Serializes refreshes without blocking readers. Held across the database and network awaits
    /// a refresh performs; `credential` is not.
    refresh_gate: tokio::sync::Mutex<()>,
    /// The access token a request site got a 401 for, read by every `ensure_valid_credential`
    /// until a refresh installs a replacement. Same contract as the Claude provider's: a refused
    /// token is refreshed whatever the stored expiry says, and its identity rather than a flag is
    /// what lets two requests refused in the same window both get the replacement.
    rejected_access_token: std::sync::Mutex<Option<String>>,
    base_url: String,
    /// What `base_url` names, settled once so every request path appends to it the same way.
    base_url_shape: ChatGptBaseUrlShape,
    model: String,
    client_id: String,
    oauth_token_url: String,
    token_store: Option<Arc<TokenStore>>,
    /// Account name this provider's credential is stored under, so refreshed tokens are written
    /// back to the correct `account_credentials` row.
    credential_key: String,
    /// The settled `reasoning.effort` for the request body, resolved once at construction from the
    /// profile's override. `None` (the unconfigured case) skips the reasoning block entirely, so
    /// the Responses API applies its own default.
    resolved_effort: Option<String>,
    /// Per-request output token cap from the profile; `None` leaves the Responses API default.
    max_output_tokens: Option<u64>,
    /// See [`crate::config::ProfileConfig::max_request_bytes`]; unset means no ceiling here.
    max_request_bytes: Option<usize>,
    user_agent: String,
}

impl ChatGptSubscriptionProvider {
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
            effort: reasoning_effort,
            max_output_tokens,
            max_request_bytes,
            ..
        } = settings;
        // chatgpt.com is fronted by Cloudflare; enabling the cookie jar lets bot-clearance cookies
        // (e.g. `__cf_bm`) persist across requests.
        let client = crate::provider::build_http_client("chatgpt-subscription", |builder| {
            builder.cookie_store(true)
        })?;

        let resolved_effort = crate::provider::resolve_effort_level(reasoning_effort.as_deref());
        let base_url = crate::provider::normalize_base_url(
            base_url
                .as_deref()
                .unwrap_or(crate::provider::DEFAULT_CHATGPT_BASE_URL),
        );
        // The same refusal `meka account add` makes, for a `base_url` that reached config.toml by
        // another route: an unusable endpoint is reported here, before a turn is spent on it.
        let base_url_shape = chatgpt_base_url_shape(&base_url)?;
        Ok(Self {
            client,
            credential: tokio::sync::RwLock::new(credential),
            refresh_gate: tokio::sync::Mutex::new(()),
            rejected_access_token: std::sync::Mutex::new(None),
            base_url,
            base_url_shape,
            model,
            client_id: client_id
                .unwrap_or_else(|| DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID.to_string()),
            oauth_token_url: oauth_token_url.unwrap_or_else(|| DEFAULT_TOKEN_URL.to_string()),
            token_store,
            credential_key,
            resolved_effort,
            max_output_tokens,
            max_request_bytes,
            user_agent: format!(
                "meka/{} ({}; {})",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
        })
    }

    /// The settled reasoning-effort to send as `reasoning.effort` (see [`Self::resolved_effort`]).
    fn wire_effort(&self) -> Option<String> {
        self.resolved_effort.clone()
    }

    /// The request body: the shared Responses encoding, plus the one thing this backend may add
    /// that its API-key sibling may not.
    ///
    /// A named method rather than inline in `stream` so the `include` can be asserted without a
    /// live endpoint. It is the half of the split that has to keep *sending*, and a test that only
    /// covered the other half would let it fall away silently.
    fn build_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> serde_json::Value {
        let mut body = build_request_body(
            &self.model,
            system_prompt,
            messages,
            tools,
            self.wire_effort().as_deref(),
            self.max_output_tokens,
            true,
        );
        // Safe here and only here: this backend's endpoint is always ChatGPT, and the first-party
        // Codex client asks for the same two things, a summary so the reasoning is visible, and
        // the encrypted content so it survives a stateless round trip. Summary first: it settles
        // the `reasoning` object the `include` keys off.
        request_reasoning_summary(&mut body);
        include_encrypted_reasoning(&mut body);
        body
    }

    /// Returns the URL the request POSTs to. Codex's own client appends `/backend-api`
    /// automatically when the base URL is one of the chatgpt.com domains; the path is kept explicit
    /// here so a profile whose `base_url` names a custom proxy doesn't need its author to know the
    /// rewrite rule.
    fn responses_url(&self) -> String {
        let base = &self.base_url;
        match self.base_url_shape {
            ChatGptBaseUrlShape::Root => format!("{base}/backend-api/codex/responses"),
            ChatGptBaseUrlShape::CodexRoot => format!("{base}/responses"),
        }
    }

    /// The `/backend-api` root the account endpoints hang off: `base_url` with a trailing `codex`
    /// segment removed, and `/backend-api` appended when the path has none.
    ///
    /// The account endpoints live beside `codex`, not under it: built from a `/backend-api/codex`
    /// base verbatim they would land on `/backend-api/codex/wham/usage`, and `meka account` would
    /// fail on a profile whose turns work.
    fn backend_api_root(&self) -> String {
        let base = &self.base_url;
        match self.base_url_shape {
            ChatGptBaseUrlShape::Root => format!("{base}/backend-api"),
            ChatGptBaseUrlShape::CodexRoot => {
                base.strip_suffix("/codex").unwrap_or(base).to_string()
            }
        }
    }

    /// URL of the ChatGPT-backend usage endpoint (`/wham/usage`), which lives under `/backend-api`
    /// alongside the responses endpoint.
    fn usage_url(&self) -> String {
        format!("{}/wham/usage", self.backend_api_root())
    }

    /// URL of the ChatGPT-backend token-usage-profile endpoint (`/wham/profiles/me`).
    fn profiles_url(&self) -> String {
        format!("{}/wham/profiles/me", self.backend_api_root())
    }

    /// GET `/wham/usage` and parse it. Shared by `fetch_usage` (rate-limit windows) and
    /// `fetch_identity` (the `plan_type` field), which both read this one payload.
    async fn fetch_wham_usage(&self) -> Result<CodexUsageResponse> {
        let response_text = self.fetch_json_endpoint(self.usage_url(), "usage").await?;
        serde_json::from_str(&response_text)
            .map_err(|error| MekaError::Provider(format!("invalid Codex usage JSON: {error}")))
    }

    /// Record that the backend refused the current access token, so every credential read until
    /// it is replaced refreshes it instead of trusting the stored expiry.
    async fn note_credential_rejected(&self) {
        tracing::warn!(
            "chatgpt-subscription rejected the access token; refreshing it and retrying once"
        );
        let refused = match &*self.credential.read().await {
            AuthCredential::OAuthToken { access_token, .. } => Some(access_token.clone()),
            AuthCredential::ApiKey(_) => None,
        };
        *crate::sync::lock(&self.rejected_access_token) = refused;
    }

    /// GET one of the account endpoints (usage, profile) as text, with the one retry after a 401
    /// that a completion gets. `what` names the call in the transport and read errors.
    ///
    /// Not `apply_headers`: that sets `Accept: text/event-stream` for the SSE responses call, and
    /// these endpoints return plain JSON.
    async fn fetch_json_endpoint(&self, url: String, what: &str) -> Result<String> {
        let response = crate::oauth::send_with_one_refresh(
            self,
            crate::error::ProviderRequest::Auxiliary,
            |error| {
                crate::error::provider_transport_error(
                    &format!("Codex {what} request failed"),
                    error,
                    None,
                )
            },
            || async {
                let (access_token, account_id) = self.ensure_valid_credential().await?;
                let mut request = self
                    .client
                    .get(&url)
                    .header("Authorization", crate::text::bearer(&access_token))
                    .header("originator", ORIGINATOR)
                    .header("User-Agent", &self.user_agent)
                    .header("Accept", "application/json");
                if let Some(account_id) = account_id.as_deref() {
                    request = request.header("ChatGPT-Account-ID", account_id);
                }
                Ok(request)
            },
            &tokio_util::sync::CancellationToken::new(),
        )
        .await?;
        let status = response.status();
        let retry_after = crate::error::parse_retry_after(response.headers());
        let text = response.text().await.map_err(|error| {
            crate::error::provider_transport_error(
                &format!("failed to read Codex {what} response"),
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

    /// Returns `(bearer_token, account_id)`, refreshing the access token first if it's within 5
    /// minutes of expiry. The account_id is `Option<String>` because free-tier accounts may not
    /// have one (Codex's auth/manager.rs treats the missing claim as non-fatal).
    async fn ensure_valid_credential(&self) -> Result<(String, Option<String>)> {
        // Compared, not consumed: the rejection stands until a refresh replaces the token it
        // names, so a second request carrying the same refused bearer refreshes too instead of
        // finding a flag the first one spent.
        let refused = crate::sync::lock(&self.rejected_access_token).clone();
        let (rejected, entry_access_token) = {
            let credential = self.credential.read().await;
            let AuthCredential::OAuthToken {
                access_token,
                expires_at,
                account_id,
                refresh_token,
            } = &*credential
            else {
                return Err(MekaError::Provider(
                    "chatgpt-subscription requires an OAuth token, not an API key".to_string(),
                ));
            };

            let rejected = refused.as_deref() == Some(access_token.as_str());
            if !rejected
                && !crate::oauth::oauth_needs_refresh(
                    *expires_at,
                    refresh_token.is_some(),
                    crate::oauth::now_epoch_millis(),
                )
            {
                return Ok((access_token.clone(), account_id.clone()));
            }
            let entry_access_token = access_token.clone();
            drop(credential);
            (rejected, entry_access_token)
        };

        // Only refreshers queue here. `credential` is taken for the reads and writes themselves and
        // never held across the database or network awaits below: with its write lock as the
        // refresh gate, a provider endpoint that goes silent wedges every reader in the process,
        // not just the task refreshing. See the Claude provider for the same contract.
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
                    tracing::warn!("failed to re-read Codex OAuth token before refresh: {error}");
                }
            }
        }

        // Double-check after the DB re-read: another caller (in this process or a sibling meka) may
        // already have rotated to a still-valid access token.
        let derived_from = {
            let credential = self.credential.read().await;
            if let AuthCredential::OAuthToken {
                access_token,
                expires_at,
                account_id,
                refresh_token,
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
                    return Ok((access_token.clone(), account_id.clone()));
                }
            }
            credential.clone()
        };
        let refresh_token = match &derived_from {
            AuthCredential::OAuthToken { refresh_token, .. } => refresh_token.clone(),
            AuthCredential::ApiKey(_) => None,
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

        let prior_account_id = match &derived_from {
            AuthCredential::OAuthToken { account_id, .. } => account_id.clone(),
            AuthCredential::ApiKey(_) => None,
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

        let (token_value, account_id) = match &new_credential {
            AuthCredential::OAuthToken {
                access_token,
                account_id,
                ..
            } => (access_token.clone(), account_id.clone()),
            // Unreachable as things stand, and kept anyway. `store_refreshed_credential` only
            // hands back a credential of the same kind this refresh was derived from, and that is
            // an `OAuthToken` at every path reaching here. If that ever stops being true this is
            // the difference between a clear refusal and an empty bearer token on the wire.
            AuthCredential::ApiKey(_) => {
                return Err(MekaError::Provider(
                    "the stored credential for this profile is not an OAuth token".to_string(),
                ));
            }
        };

        *self.credential.write().await = new_credential;
        // The refused token is gone, whatever the issuer minted; a rejection of the replacement is
        // recorded afresh by the request that meets it.
        crate::sync::lock(&self.rejected_access_token).take();
        Ok((token_value, account_id))
    }

    async fn refresh_oauth_token(
        &self,
        refresh_token: &str,
        prior_account_id: Option<String>,
    ) -> Result<AuthCredential> {
        tracing::info!("refreshing Codex OAuth token");

        #[derive(Deserialize)]
        struct RefreshResponse {
            id_token: Option<String>,
            access_token: Option<String>,
            refresh_token: Option<String>,
        }

        // The exchange itself is shared with the Claude backend; what is this backend's own is
        // everything below, because ChatGPT's issuer states neither an expiry nor an account and
        // both have to be read out of the JWTs it returns.
        let data: RefreshResponse =
            crate::oauth::exchange_refresh_token(crate::oauth::RefreshExchange {
                client: &self.client,
                token_url: &self.oauth_token_url,
                client_id: &self.client_id,
                refresh_token,
                account: &self.credential_key,
                context: "Codex OAuth token refresh",
            })
            .await?;

        let access_token = data.access_token.ok_or_else(|| {
            MekaError::Provider("Codex refresh response missing access_token".to_string())
        })?;

        // Re-extract `chatgpt_account_id` from the new id_token if the server returned one: the
        // workspace association can change. A refresh that returns no id token, or one without
        // the claim, says nothing about the account, so the one already known stands; blanking
        // it would bill a workspace subscriber's traffic to the personal account, durably.
        let account_id = data
            .id_token
            .as_deref()
            .and_then(|id_token| extract_account_id(id_token).ok().flatten())
            .or(prior_account_id);

        // expires_at comes from the access_token JWT's `exp` claim. A token carrying no readable
        // `exp`, or one too large to hold in milliseconds, gets the assumed lifetime: `None` reads
        // as due and would send every later request back through this whole path, rotating the
        // refresh token each time, while a far-future stamp would pin whatever the issuer actually
        // minted for the rest of the process. The 401 that corrects a wrong expiry forces one
        // refresh; a bounded guess keeps that the exception.
        let expires_at = Some(match extract_expiration_seconds(&access_token) {
            Ok(Some(seconds)) => seconds.checked_mul(1000).unwrap_or_else(|| {
                crate::oauth::oauth_assumed_expiry(crate::oauth::now_epoch_millis())
            }),
            _ => crate::oauth::oauth_assumed_expiry(crate::oauth::now_epoch_millis()),
        });

        Ok(AuthCredential::OAuthToken {
            access_token,
            refresh_token: data
                .refresh_token
                .or_else(|| Some(refresh_token.to_string())),
            expires_at,
            account_id,
        })
    }

    fn apply_headers(
        &self,
        request: reqwest::RequestBuilder,
        access_token: &str,
        account_id: Option<&str>,
    ) -> reqwest::RequestBuilder {
        let mut request = request
            .header("Authorization", crate::text::bearer(access_token))
            .header("originator", ORIGINATOR)
            .header("User-Agent", &self.user_agent)
            .header("Accept", "text/event-stream");
        // No `Content-Type` here: the shared send in `responses_wire` sets it when it attaches the
        // serialized body, and `reqwest` appends a second header rather than replacing the first,
        // which the backend refuses as an unsupported content type.
        if let Some(account_id) = account_id {
            request = request.header("ChatGPT-Account-ID", account_id);
        }
        request
    }
}

#[async_trait]
impl super::responses_wire::ResponsesBackend for ChatGptSubscriptionProvider {
    fn client(&self) -> &reqwest::Client {
        &self.client
    }

    fn endpoint(&self) -> String {
        self.responses_url()
    }

    fn request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> serde_json::Value {
        self.build_body(system_prompt, messages, tools)
    }

    fn max_request_bytes(&self) -> Option<usize> {
        self.max_request_bytes
    }

    async fn authenticated_request(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder> {
        let (access_token, account_id) = self.ensure_valid_credential().await?;
        Ok(self.apply_headers(request, &access_token, account_id.as_deref()))
    }
}

#[async_trait]
impl crate::oauth::RefreshesCredential for ChatGptSubscriptionProvider {
    /// Sent once more after a 401, on a credential refreshed for the purpose. A second refusal
    /// gets the login remedy instead.
    async fn refresh_after_rejection(&self) -> bool {
        self.note_credential_rejected().await;
        true
    }

    fn with_login_remedy(&self, error: MekaError) -> MekaError {
        crate::oauth::with_login_remedy(error, &self.credential_key)
    }
}

#[async_trait]
impl Provider for ChatGptSubscriptionProvider {
    async fn complete(
        &self,
        request: CompletionRequest<'_>,
        cancellation: CancellationToken,
    ) -> Result<crate::provider::Completion> {
        super::responses_wire::complete(self, request, cancellation).await
    }

    async fn stream(
        &self,
        request: CompletionRequest<'_>,
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        super::responses_wire::stream(self, request, event_sender, cancellation).await
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }

    async fn fetch_usage(&self) -> Result<Option<AccountUsage>> {
        Ok(Some(self.fetch_wham_usage().await?.into_account_usage()))
    }

    async fn fetch_history(&self) -> Result<Option<UsageHistory>> {
        let text = self
            .fetch_json_endpoint(self.profiles_url(), "profile")
            .await?;
        let parsed: CodexProfileResponse = serde_json::from_str(&text)
            .map_err(|error| MekaError::Provider(format!("invalid Codex profile JSON: {error}")))?;
        Ok(Some(parsed.into_history()))
    }

    async fn fetch_identity(&self) -> Result<Option<AccountIdentity>> {
        // The plan is the one identity field the usage payload carries; name, org and role need
        // `accounts/check`, so they stay `None`.
        let plan = self.fetch_wham_usage().await?.plan_type;
        Ok(Some(AccountIdentity {
            display_name: None,
            email: None,
            plan,
            tier: None,
            subscription_status: None,
            organization: None,
            role: None,
        }))
    }
}

/// The subset of the ChatGPT backend `GET /wham/usage` body that is rendered. Mirrors the fields
/// the Codex CLI reads (`RateLimitStatusPayload`), tolerant of absent/null buckets.
#[derive(Deserialize)]
struct CodexUsageResponse {
    #[serde(default)]
    plan_type: Option<String>,
    #[serde(default)]
    rate_limit: Option<CodexRateLimit>,
    #[serde(default)]
    credits: Option<CodexCredits>,
    #[serde(default)]
    spend_control: Option<CodexSpendControl>,
}

#[derive(Deserialize)]
struct CodexRateLimit {
    #[serde(default)]
    primary_window: Option<CodexWindow>,
    #[serde(default)]
    secondary_window: Option<CodexWindow>,
}

#[derive(Deserialize)]
struct CodexWindow {
    #[serde(default)]
    used_percent: Option<f64>,
    #[serde(default)]
    limit_window_seconds: Option<i64>,
    #[serde(default)]
    reset_at: Option<i64>,
}

#[derive(Deserialize)]
struct CodexCredits {
    #[serde(default)]
    has_credits: Option<bool>,
    /// Dollar string, e.g. `"9.99"` or `"$9.99"`.
    #[serde(default)]
    balance: Option<String>,
}

#[derive(Deserialize)]
struct CodexSpendControl {
    #[serde(default)]
    individual_limit: Option<CodexIndividualLimit>,
}

#[derive(Deserialize)]
struct CodexIndividualLimit {
    /// Dollar string of the amount spent against the cap.
    #[serde(default)]
    used: Option<String>,
    #[serde(default)]
    used_percent: Option<f64>,
}

impl CodexUsageResponse {
    fn into_account_usage(self) -> AccountUsage {
        let mut windows = Vec::new();
        if let Some(rate_limit) = self.rate_limit {
            push_codex_window(&mut windows, rate_limit.primary_window, "Primary");
            push_codex_window(&mut windows, rate_limit.secondary_window, "Secondary");
        }
        let note = self
            .plan_type
            .filter(|plan| !plan.is_empty())
            .map(|plan| format!("plan: {plan}"));
        AccountUsage {
            windows,
            extra_usage: codex_extra_usage(self.credits, self.spend_control),
            note,
        }
    }
}

/// Parse a dollar string like `"$9.99"` / `"9.99"` / `"1,234.50"` into an `f64`.
fn parse_dollars(value: &str) -> Option<f64> {
    value
        .trim()
        .trim_start_matches('$')
        .replace(',', "")
        .parse::<f64>()
        .ok()
}

/// Normalize Codex's `credits` + `spend_control` blocks into [`ExtraUsage`].
fn codex_extra_usage(
    credits: Option<CodexCredits>,
    spend_control: Option<CodexSpendControl>,
) -> Option<ExtraUsage> {
    if credits.is_none() && spend_control.is_none() {
        return None;
    }
    let (has_credits, balance) = match credits {
        Some(credits) => (
            credits.has_credits.unwrap_or(false),
            credits.balance.as_deref().and_then(parse_dollars),
        ),
        None => (false, None),
    };
    let (used, utilization) = match spend_control.and_then(|control| control.individual_limit) {
        Some(limit) => (
            limit.used.as_deref().and_then(parse_dollars),
            limit.used_percent,
        ),
        None => (None, None),
    };
    Some(ExtraUsage {
        // Extra usage is active if the account holds credits or has recorded spend against a cap;
        // keying only on `has_credits` would mislabel spend-only accounts as "disabled".
        enabled: has_credits || used.is_some(),
        utilization,
        used,
        balance,
        currency: None,
    })
}

fn push_codex_window(windows: &mut Vec<UsageWindow>, window: Option<CodexWindow>, fallback: &str) {
    if let Some(window) = window
        && let Some(used_percent) = window.used_percent
    {
        windows.push(UsageWindow {
            label: codex_window_label(window.limit_window_seconds, fallback),
            used_percent,
            resets_at: window.reset_at,
        });
    }
}

/// Subset of Codex's `GET /wham/profiles/me` body (`TokenUsageProfile`).
#[derive(Deserialize)]
struct CodexProfileResponse {
    #[serde(default)]
    stats: Option<CodexProfileStats>,
}

#[derive(Deserialize)]
struct CodexProfileStats {
    #[serde(default)]
    lifetime_tokens: Option<i64>,
    #[serde(default)]
    peak_daily_tokens: Option<i64>,
    #[serde(default)]
    current_streak_days: Option<i64>,
    #[serde(default)]
    longest_streak_days: Option<i64>,
    #[serde(default)]
    daily_usage_buckets: Vec<CodexDailyBucket>,
}

#[derive(Deserialize)]
struct CodexDailyBucket {
    start_date: String,
    tokens: i64,
}

impl CodexProfileResponse {
    fn into_history(self) -> UsageHistory {
        let stats = self.stats;
        UsageHistory {
            lifetime_tokens: stats.as_ref().and_then(|s| s.lifetime_tokens),
            peak_daily_tokens: stats.as_ref().and_then(|s| s.peak_daily_tokens),
            current_streak_days: stats.as_ref().and_then(|s| s.current_streak_days),
            longest_streak_days: stats.as_ref().and_then(|s| s.longest_streak_days),
            first_used: None,
            daily: stats
                .map(|s| {
                    s.daily_usage_buckets
                        .into_iter()
                        .map(|bucket| DailyUsage {
                            date: bucket.start_date,
                            tokens: bucket.tokens,
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

/// Human label for a window from its duration (seconds). Common durations get friendly names; the
/// rest fall back to the primary/secondary position label.
fn codex_window_label(limit_window_seconds: Option<i64>, fallback: &str) -> String {
    let Some(minutes) = limit_window_seconds.map(|seconds| seconds / 60) else {
        return fallback.to_string();
    };
    match minutes {
        m if m == 7 * 24 * 60 => "Weekly".to_string(),
        m if m % (24 * 60) == 0 => format!("{}-day", m / (24 * 60)),
        m if m % 60 == 0 => format!("{}-hour", m / 60),
        m => format!("{m}-min"),
    }
}

/// What a `chatgpt-subscription` account's `base_url` names, which decides what this backend
/// appends to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatGptBaseUrlShape {
    /// A root whose path has neither a `backend-api` nor a `codex` segment (`https://chatgpt.com`,
    /// or a proxy forwarding from any other prefix): meka appends `/backend-api/codex/responses`
    /// for turns and `/backend-api/wham/...` for the account endpoints.
    Root,
    /// A path ending in `/backend-api/codex`, the Codex client's own layout: meka appends
    /// `/responses` for turns and reaches the account endpoints beside `codex`.
    CodexRoot,
}

/// Classify a `base_url`, refusing every shape the appended paths would not reach.
///
/// One definition for `meka account add` and for building the provider, so a URL the first
/// accepts is one the second can use. Judged on path segments rather than substrings, so a host
/// named `codex.example.com` is not routed as if its path already ended in `/codex`.
pub(crate) fn chatgpt_base_url_shape(base_url: &str) -> Result<ChatGptBaseUrlShape> {
    let normalized = crate::provider::normalize_base_url(base_url);
    let segments: Vec<String> = reqwest::Url::parse(&normalized)
        .ok()
        .and_then(|url| {
            url.path_segments().map(|segments| {
                segments
                    .filter(|segment| !segment.is_empty())
                    .map(str::to_string)
                    .collect()
            })
        })
        .ok_or_else(|| {
            MekaError::Installation(format!(
                "`base_url` '{base_url}' is not a URL the chatgpt-subscription backend can build \
                 requests on"
            ))
        })?;
    let is_reserved = |segment: &String| segment == "backend-api" || segment == "codex";
    let reserved_count = segments
        .iter()
        .filter(|segment| is_reserved(segment))
        .count();
    if reserved_count == 0 {
        return Ok(ChatGptBaseUrlShape::Root);
    }
    if reserved_count == 2
        && let [.., before_last, last] = segments.as_slice()
        && before_last == "backend-api"
        && last == "codex"
    {
        return Ok(ChatGptBaseUrlShape::CodexRoot);
    }
    Err(MekaError::Installation(format!(
        "`base_url` '{base_url}' is not a shape the chatgpt-subscription backend accepts: give the \
         origin, or a path ending in '/backend-api/codex'"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{StopReason, openai::responses_wire::aggregate_stream};

    fn credential_for_test() -> AuthCredential {
        AuthCredential::OAuthToken {
            access_token: "access-test".to_string(),
            refresh_token: Some("refresh-test".to_string()),
            // 1 day in the future to avoid the refresh path during construction.
            expires_at: Some(crate::oauth::now_epoch_millis() + 86_400_000),
            account_id: Some("workspace-test".to_string()),
        }
    }

    fn provider_for_test() -> ChatGptSubscriptionProvider {
        ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(Some("high".to_string()))
            .max_output_tokens(None),
        )
        .expect("provider")
    }

    /// A stream that never started is retryable here too.
    ///
    /// The Codex sibling of the pair in `provider::anthropic::subscription`: its own `.send()`
    /// site, so its own wiring into [`crate::error::provider_transport_error`] and its own chance
    /// to regress to a bare `MekaError::Provider` that the agent loop discards.
    /// `credential_for_test`'s expiry is a day out, so `ensure_valid_credential` takes no
    /// refresh round trip first.
    #[tokio::test]
    async fn a_stream_that_could_not_start_reports_a_retryable_failure() {
        // Bound and dropped, so the port is refused rather than answered or hung.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5".to_string(),
            )
            .base_url(Some(format!("http://127.0.0.1:{port}/v1")))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(Some("high".to_string()))
            .max_output_tokens(None),
        )
        .expect("provider");
        let (sender, _receiver) = mpsc::channel(8);
        let error = provider
            .stream(
                CompletionRequest::new("", &[Message::user("hello")], &[]),
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

    /// The turn request names its content type once. `apply_headers` used to set it too, beside
    /// the shared send that attaches the serialized body; `reqwest` appends rather than replaces,
    /// and the backend refuses a request with two of them as an unsupported content type.
    #[tokio::test]
    async fn a_turn_request_carries_one_content_type_header() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a mock responses endpoint");
        let local = listener.local_addr().expect("local addr");
        let (head_sender, head_receiver) = tokio::sync::oneshot::channel::<String>();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut buffer = Vec::with_capacity(4096);
            loop {
                let mut chunk = [0u8; 2048];
                let read = match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(read) => read,
                };
                buffer.extend_from_slice(&chunk[..read]);
                if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let end = buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .unwrap_or(buffer.len());
            let head = String::from_utf8_lossy(&buffer[..end]).to_ascii_lowercase();
            // The test does not need an answer, so the shortest refusal will do.
            let response =
                "HTTP/1.1 400 Bad Request\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
            if socket.write_all(response.as_bytes()).await.is_err() {
                return;
            }
            if socket.shutdown().await.is_err() {
                return;
            }
            if head_sender.send(head).is_err() {
                tracing::debug!("the test dropped its receiver before the head arrived");
            }
        });

        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5".to_string(),
            )
            .base_url(Some(format!("http://{local}")))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(Some("high".to_string()))
            .max_output_tokens(None),
        )
        .expect("provider");
        let (sender, _receiver) = mpsc::channel(8);
        // The refusal is the point of the mock, not of the test.
        if let Ok(()) = provider
            .stream(
                CompletionRequest::new("", &[Message::user("hello")], &[]),
                sender,
                CancellationToken::new(),
            )
            .await
        {
            panic!("a 400 from the endpoint must not read as a completed stream");
        }
        let head = head_receiver.await.expect("the mock saw the request");
        let content_types = head
            .lines()
            .filter(|line| line.starts_with("content-type:"))
            .count();
        assert_eq!(
            content_types, 1,
            "the turn request must carry exactly one Content-Type header; head:\n{head}"
        );
    }

    /// The history probe classifies a dead endpoint the same way the turn path does.
    ///
    /// Its own `.send()` and response-read, so its own chance to bypass the shared classifier.
    ///
    /// Nothing retries this -- its caller is `meka account stats`, outside any retry loop -- so
    /// classifying it changes only how the failure reads. Worth pinning anyway: it is the same
    /// classifier the turn path depends on, and a probe that stops calling it has grown its own
    /// private rules.
    #[tokio::test]
    async fn the_history_probe_reports_a_dead_endpoint_as_retryable() {
        // Bound and dropped, so the port is refused rather than answered or hung.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5".to_string(),
            )
            .base_url(Some(format!("http://127.0.0.1:{port}/v1")))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(Some("high".to_string()))
            .max_output_tokens(None),
        )
        .expect("provider");
        let error = provider
            .fetch_history()
            .await
            .expect_err("nothing is listening there");

        assert!(
            matches!(error, MekaError::RetryableProvider { .. }),
            "{error}"
        );
    }

    /// This backend keeps the `include` its API-key sibling refuses to send.
    ///
    /// The other half of the split asserted in `openai-responses`: there, an OpenAI extension must
    /// never reach an endpoint that may not implement it; here, the endpoint is always ChatGPT and
    /// the first-party Codex client asks for the same thing, so reasoning survives the stateless
    /// round trip. Dropping it here would be silent -- the requests would still succeed, just
    /// without reasoning carried across turns.
    #[test]
    fn the_subscription_asks_chatgpt_to_round_trip_its_reasoning() {
        let body = provider_for_test().build_body("s", &[Message::user("hi")], &[]);
        assert_eq!(body["reasoning"]["effort"], "high");
        let include = body["include"].as_array().expect("include");
        assert!(
            include
                .iter()
                .any(|value| value == "reasoning.encrypted_content"),
            "{body}"
        );
    }

    /// The summary is the only part of the reasoning a person ever sees. Without it the model
    /// still thinks, the stream carries no summary deltas, and a long think renders as a hang.
    #[test]
    fn the_subscription_asks_chatgpt_to_summarize_its_reasoning() {
        let body = provider_for_test().build_body("s", &[Message::user("hi")], &[]);
        assert_eq!(body["reasoning"]["summary"], "auto", "{body}");
    }

    /// Neither ask may hinge on an effort being configured: with none, the shared body omits
    /// `reasoning` entirely, so there is nothing for the `include` to attach to, and the profile a
    /// user gets by default would ask ChatGPT for neither a summary nor encrypted reasoning. Codex
    /// sends `reasoning` on every request and omits only the fields it has no value for.
    #[test]
    fn an_unconfigured_profile_still_asks_for_a_summary_and_encrypted_reasoning() {
        let unconfigured = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5.6-sol".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");
        let body = unconfigured.build_body("s", &[Message::user("hi")], &[]);

        assert!(body["reasoning"].get("effort").is_none(), "{body}");
        assert_eq!(body["reasoning"]["summary"], "auto", "{body}");
        assert_eq!(
            body["include"],
            serde_json::json!(["reasoning.encrypted_content"]),
            "{body}"
        );
    }

    #[test]
    fn an_unconfigured_profile_sends_no_reasoning_effort() {
        // Effort belongs to the provider: unset means the Responses API applies its own default,
        // which meka asks for by omitting the field rather than by naming a tier.
        let unconfigured = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5.6-sol".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");
        assert_eq!(unconfigured.resolved_effort(), None);

        let configured = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5.6-sol".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(Some("medium".to_string()))
            .max_output_tokens(None),
        )
        .expect("provider");
        assert_eq!(configured.resolved_effort().as_deref(), Some("medium"));
    }

    #[test]
    fn usage_url_default_appends_backend_api_wham() {
        assert_eq!(
            provider_for_test().usage_url(),
            "https://chatgpt.com/backend-api/wham/usage"
        );
    }

    #[test]
    fn codex_usage_maps_windows_and_note() {
        // Shaped like the ChatGPT-backend `/wham/usage` body (RateLimitStatusPayload).
        let body = r#"{
            "plan_type": "plus",
            "rate_limit": {
                "allowed": true,
                "limit_reached": false,
                "primary_window": {"used_percent": 42, "limit_window_seconds": 18000, "reset_at": 123},
                "secondary_window": {"used_percent": 84, "limit_window_seconds": 604800, "reset_at": 456}
            },
            "credits": {"has_credits": true, "unlimited": false, "balance": "9.99"}
        }"#;
        let usage = serde_json::from_str::<CodexUsageResponse>(body)
            .expect("parse")
            .into_account_usage();
        assert_eq!(usage.windows.len(), 2);
        // 18000s = 300min -> 5-hour; 604800s = 10080min -> Weekly.
        assert_eq!(usage.windows[0].label, "5-hour");
        assert_eq!(usage.windows[0].used_percent, 42.0);
        assert_eq!(usage.windows[0].resets_at, Some(123));
        assert_eq!(usage.windows[1].label, "Weekly");
        // Plan stays in the note; credits move to extra_usage.
        assert_eq!(usage.note.as_deref(), Some("plan: plus"));
        let extra = usage.extra_usage.expect("extra_usage");
        assert!(extra.enabled);
        assert_eq!(extra.balance, Some(9.99));
    }

    #[test]
    fn codex_extra_usage_parses_spend_control() {
        let body = r#"{
            "plan_type": "pro",
            "credits": {"has_credits": true, "unlimited": false, "balance": "$5.00"},
            "spend_control": {"individual_limit": {"used": "$3.50", "used_percent": 70}}
        }"#;
        let extra = serde_json::from_str::<CodexUsageResponse>(body)
            .unwrap()
            .into_account_usage()
            .extra_usage
            .expect("extra_usage");
        assert!(extra.enabled);
        assert_eq!(extra.balance, Some(5.0));
        assert_eq!(extra.used, Some(3.5));
        assert_eq!(extra.utilization, Some(70.0));
    }

    #[test]
    fn codex_extra_usage_spend_only_is_enabled() {
        // No purchased credits, but recorded spend against a cap: must render as enabled, not
        // "disabled · $X spent".
        let body = r#"{
            "spend_control": {"individual_limit": {"used": "$3.50", "used_percent": 70}}
        }"#;
        let extra = serde_json::from_str::<CodexUsageResponse>(body)
            .unwrap()
            .into_account_usage()
            .extra_usage
            .expect("extra_usage");
        assert!(extra.enabled);
        assert_eq!(extra.used, Some(3.5));
    }

    #[test]
    fn codex_window_missing_used_percent_is_skipped_not_fatal() {
        // A partial window object (no `used_percent`) degrades to being dropped rather than failing
        // the whole payload; the complete sibling window still parses.
        let body = r#"{
            "rate_limit": {
                "primary_window": {"limit_window_seconds": 18000, "reset_at": 123},
                "secondary_window": {"used_percent": 84, "limit_window_seconds": 604800}
            }
        }"#;
        let usage = serde_json::from_str::<CodexUsageResponse>(body)
            .expect("partial window must not fail the parse")
            .into_account_usage();
        assert_eq!(usage.windows.len(), 1);
        assert_eq!(usage.windows[0].label, "Weekly");
        assert_eq!(usage.windows[0].used_percent, 84.0);
    }

    #[test]
    fn codex_profile_maps_history() {
        let body = r#"{
            "stats": {
                "lifetime_tokens": 1200000,
                "peak_daily_tokens": 45000,
                "current_streak_days": 3,
                "longest_streak_days": 12,
                "daily_usage_buckets": [
                    {"start_date": "2026-06-30", "tokens": 8100},
                    {"start_date": "2026-07-01", "tokens": 12300}
                ]
            }
        }"#;
        let history = serde_json::from_str::<CodexProfileResponse>(body)
            .unwrap()
            .into_history();
        assert_eq!(history.lifetime_tokens, Some(1_200_000));
        assert_eq!(history.current_streak_days, Some(3));
        assert_eq!(history.daily.len(), 2);
        assert_eq!(history.daily[1].date, "2026-07-01");
        assert_eq!(history.daily[1].tokens, 12300);
    }

    #[test]
    fn codex_usage_empty_rate_limit_is_no_windows() {
        let usage = serde_json::from_str::<CodexUsageResponse>(r#"{"plan_type": "pro"}"#)
            .unwrap()
            .into_account_usage();
        assert!(usage.windows.is_empty());
        assert_eq!(usage.note.as_deref(), Some("plan: pro"));
    }

    #[test]
    fn responses_url_default_appends_backend_api_codex() {
        let provider = provider_for_test();
        assert_eq!(
            provider.responses_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn responses_url_user_supplied_backend_api_path_preserved() {
        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5".to_string(),
            )
            .base_url(Some("https://example.com/backend-api/codex".to_string()))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");
        assert_eq!(
            provider.responses_url(),
            "https://example.com/backend-api/codex/responses"
        );
    }

    /// A host name is not a path. `https://codex.example.com` contains `/codex` as a substring,
    /// and was routed to `/responses` on a host that expected `/backend-api/codex/responses`.
    #[test]
    fn a_host_named_codex_is_not_a_codex_path() {
        for (base, expected) in [
            (
                "https://codex.example.com",
                "https://codex.example.com/backend-api/codex/responses",
            ),
            (
                "https://proxy.example.com/codex-mirror",
                "https://proxy.example.com/codex-mirror/backend-api/codex/responses",
            ),
        ] {
            let provider = ChatGptSubscriptionProvider::new(
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::ChatGptSubscription,
                    credential_for_test(),
                    "gpt-5".to_string(),
                )
                .base_url(Some(base.to_string()))
                .client_id(None)
                .oauth_token_url(None)
                .token_store(None)
                .credential_key(Some("test".to_string()))
                .effort(None)
                .max_output_tokens(None),
            )
            .expect("provider");
            assert_eq!(provider.responses_url(), expected, "for {base}");
        }
    }

    /// The two shapes the request paths are built by appending to, and nothing in between: a
    /// `/backend-api` alone would take `/responses` and reach nothing, and a `/codex` under any
    /// other prefix puts the account endpoints somewhere no server has them.
    #[test]
    fn a_base_url_is_one_of_two_shapes_or_refused() {
        for (base, shape) in [
            ("https://chatgpt.com", ChatGptBaseUrlShape::Root),
            ("https://chatgpt.com/", ChatGptBaseUrlShape::Root),
            (
                "https://proxy.example.com/openai",
                ChatGptBaseUrlShape::Root,
            ),
            ("https://codex.example.com", ChatGptBaseUrlShape::Root),
            (
                "https://chatgpt.com/backend-api/codex",
                ChatGptBaseUrlShape::CodexRoot,
            ),
            (
                "https://proxy.example.com/v1/backend-api/codex/",
                ChatGptBaseUrlShape::CodexRoot,
            ),
        ] {
            assert_eq!(
                chatgpt_base_url_shape(base).expect(base),
                shape,
                "for {base}"
            );
        }
        for base in [
            "https://chatgpt.com/backend-api",
            "https://proxy.example.com/codex",
            "https://chatgpt.com/codex/backend-api",
            "https://chatgpt.com/backend-api/codex/v1",
            "https://chatgpt.com/backend-api/codex/backend-api/codex",
            "chatgpt.com",
        ] {
            let error = chatgpt_base_url_shape(base).expect_err(base);
            // The class as well as the words. The message names the operator's endpoint, and as
            // `Config` it was a refusal `meka serve` handed back verbatim in a 422; a caller can
            // do nothing about a `base_url` in someone else's `config.toml`. `meka account add`
            // asks the same question and still prints it, because there the reader typed it.
            assert!(
                matches!(error, MekaError::Installation(_)),
                "a `base_url` shape is the operator's to fix: {error:?}"
            );
            let error = error.to_string();
            assert!(error.contains(base), "names the value: {error}");
        }
    }

    /// The config.toml door: a hand-edited `base_url` is refused where the provider is built, so
    /// the first turn reports the shape rather than a 404 from a path nothing serves.
    #[test]
    fn building_the_provider_refuses_a_base_url_of_neither_shape() {
        let error = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5".to_string(),
            )
            .base_url(Some("https://chatgpt.com/backend-api".to_string()))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .err()
        .expect("refused")
        .to_string();
        assert!(error.contains("/backend-api/codex"), "{error}");
    }

    #[test]
    fn responses_url_strips_trailing_slash() {
        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5".to_string(),
            )
            .base_url(Some("https://chatgpt.com/".to_string()))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");
        assert_eq!(
            provider.responses_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[tokio::test]
    async fn ensure_valid_credential_returns_token_and_account_id() {
        let provider = provider_for_test();
        let (bearer, account_id) = provider
            .ensure_valid_credential()
            .await
            .expect("valid credential");
        assert_eq!(bearer, "access-test");
        assert_eq!(account_id.as_deref(), Some("workspace-test"));
    }

    #[tokio::test]
    async fn ensure_valid_credential_rejects_api_key() {
        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                AuthCredential::ApiKey("sk-test".to_string()),
                "gpt-5".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");
        let result = provider.ensure_valid_credential().await;
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }

    #[tokio::test]
    async fn ensure_valid_credential_no_refresh_token_when_expired() {
        // Token already expired, no refresh available → error.
        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                AuthCredential::OAuthToken {
                    access_token: "old".to_string(),
                    refresh_token: None,
                    expires_at: Some(crate::oauth::now_epoch_millis() - 1_000),
                    account_id: None,
                },
                "gpt-5".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");
        let result = provider.ensure_valid_credential().await;
        assert!(matches!(result, Err(MekaError::Provider(ref m)) if m.contains("expired")));
    }

    /// The Codex refresh path classifies by what the token endpoint answered, like its Claude twin.
    ///
    /// The sibling of `anthropic::subscription::tests::
    /// what_the_token_endpoint_answered_decides_whether_a_refresh_is_retried`. The classification
    /// they exercise lives in one place (`crate::error::oauth_refresh_error`, reached through
    /// `crate::oauth::exchange_refresh_token`), so this is a wiring test: what it proves is that
    /// *this* backend reaches that path, with its own `context`, and that its answer still
    /// comes back as this backend's own message. A bare `MekaError::Provider` here kills a turn
    /// `ensure_valid_credential` was called from the middle of, on an outage at the token endpoint.
    #[tokio::test]
    async fn what_the_codex_token_endpoint_answered_decides_whether_a_refresh_is_retried() {
        for (status_line, body, retryable) in [
            (
                "503 Service Unavailable",
                r#"{"error":"temporarily_unavailable"}"#,
                true,
            ),
            ("400 Bad Request", r#"{"error":"invalid_grant"}"#, false),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind mock OAuth endpoint");
            let local = listener.local_addr().expect("local addr");
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let mut scratch = [0u8; 4096];
                // One read is enough to know the request arrived; the response follows whatever
                // was sent, and the body is small enough to land in a single segment.
                if socket.read(&mut scratch).await.is_err() {
                    return;
                }
                let response = format!(
                    "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    status_line,
                    body.len(),
                    body
                );
                if socket.write_all(response.as_bytes()).await.is_err() {
                    return;
                }
                if socket.shutdown().await.is_err() {
                    tracing::debug!("mock Codex refresh endpoint did not shut down cleanly");
                }
            });

            let credential = AuthCredential::OAuthToken {
                access_token: "stale".to_string(),
                refresh_token: Some("rt".to_string()),
                expires_at: Some(crate::oauth::now_epoch_millis()),
                account_id: None,
            };
            let provider = ChatGptSubscriptionProvider::new(
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::ChatGptSubscription,
                    credential,
                    "gpt-5".to_string(),
                )
                .base_url(None)
                .client_id(None)
                .oauth_token_url(Some(format!("http://{local}/")))
                .token_store(None)
                .credential_key(Some("work".to_string()))
                .effort(None)
                .max_output_tokens(None),
            )
            .expect("build test provider");

            let error = provider
                .ensure_valid_credential()
                .await
                .expect_err("the endpoint did not hand back a usable token");

            // The `Codex` prefix is asserted because both backends now compose their messages from
            // one shared exchange, and this is the half that would lose its own voice if the
            // `context` a call site passes stopped being read.
            match error {
                MekaError::RetryableProvider { message, .. } if retryable => assert!(
                    message.starts_with("Codex OAuth token refresh failed ("),
                    "{status_line}: {message}"
                ),
                MekaError::Provider(message) if !retryable => {
                    assert!(
                        message.starts_with("Codex OAuth token refresh"),
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

    #[tokio::test]
    async fn aggregate_stream_folds_events_into_message() {
        // chatgpt-subscription is streaming-only; it satisfies `complete` by folding its own SSE.
        // Feed the event sequence a summary turn would emit and assert it aggregates into
        // one assistant text message carrying the reported stop reason, with no spurious
        // notices.
        let (sender, receiver) = mpsc::channel::<StreamEvent>(16);
        sender
            .send(StreamEvent::TextDelta("Summary: ".to_string()))
            .await
            .unwrap();
        sender
            .send(StreamEvent::TextDelta("all done.".to_string()))
            .await
            .unwrap();
        sender
            .send(StreamEvent::MessageEnd {
                stop_reason: StopReason::EndTurn,
            })
            .await
            .unwrap();
        drop(sender);

        let crate::provider::Completion {
            message,
            stop_reason,
            notices,
            ..
        } = aggregate_stream(receiver).await;
        assert_eq!(message.text_content(), "Summary: all done.");
        assert!(matches!(stop_reason, StopReason::EndTurn));
        assert!(notices.is_empty());
    }

    /// Answer every request with one fresh Codex token, counting the requests.
    async fn answer_every_refresh(
        listener: tokio::net::TcpListener,
        hits: Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        while let Ok((mut socket, _)) = listener.accept().await {
            let hits = Arc::clone(&hits);
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
                    let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n")
                    else {
                        continue;
                    };
                    let head = String::from_utf8_lossy(&buffer[..end]).to_ascii_lowercase();
                    let length = head
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if buffer.len() >= end + 4 + length {
                        break;
                    }
                }
                hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let body = r#"{"access_token":"fresh-access","refresh_token":"fresh-refresh","id_token":null}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \
                     {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    }

    /// A rejection buys one forced refresh whatever the stored expiry says, and only one: the
    /// refresh that replaces the refused token clears it.
    #[tokio::test]
    async fn a_rejection_forces_one_refresh_despite_a_valid_expiry() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock token endpoint");
        let local = listener.local_addr().expect("local addr");
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        tokio::spawn(answer_every_refresh(listener, Arc::clone(&hits)));

        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                AuthCredential::OAuthToken {
                    access_token: "stale".to_string(),
                    refresh_token: Some("rt".to_string()),
                    expires_at: Some(crate::oauth::now_epoch_millis() + 3_600_000),
                    account_id: Some("workspace".to_string()),
                },
                "gpt-5".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(Some(format!("http://{local}/token")))
            .token_store(None)
            .credential_key(Some("work".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");

        let (token, _) = provider
            .ensure_valid_credential()
            .await
            .expect("a token the store believes in is handed out as is");
        assert_eq!(token, "stale");
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 0);

        provider.note_credential_rejected().await;
        let (token, account_id) = provider
            .ensure_valid_credential()
            .await
            .expect("the rejection forces a refresh");
        assert_eq!(token, "fresh-access");
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
        // The endpoint answered with no id token, which says nothing about the account.
        assert_eq!(
            account_id.as_deref(),
            Some("workspace"),
            "a refresh that names no account keeps the one already known"
        );

        let (token, _) = provider
            .ensure_valid_credential()
            .await
            .expect("the rejection was cleared by the refresh");
        assert_eq!(token, "fresh-access");
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "no second refresh: the new expiry is trusted again"
        );
    }

    /// Two requests refused in the same window both get the replacement, for one refresh. With a
    /// flag in place of the token's identity, the second read found the flag spent, trusted the
    /// stored expiry, and sent the refused bearer again.
    #[tokio::test]
    async fn two_rejections_of_one_token_cost_one_refresh_and_both_get_the_new_one() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock token endpoint");
        let local = listener.local_addr().expect("local addr");
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        tokio::spawn(answer_every_refresh(listener, Arc::clone(&hits)));

        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                AuthCredential::OAuthToken {
                    access_token: "stale".to_string(),
                    refresh_token: Some("rt".to_string()),
                    expires_at: Some(crate::oauth::now_epoch_millis() + 3_600_000),
                    account_id: None,
                },
                "gpt-5".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(Some(format!("http://{local}/token")))
            .token_store(None)
            .credential_key(Some("work".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");

        // Two request sites, both refused for "stale".
        provider.note_credential_rejected().await;
        provider.note_credential_rejected().await;
        let (first, second) = tokio::join!(
            provider.ensure_valid_credential(),
            provider.ensure_valid_credential()
        );
        let (first, _) = first.expect("first refreshes");
        let (second, _) = second.expect("second waits for the refresh");
        assert_eq!(first, "fresh-access");
        assert_eq!(
            second, "fresh-access",
            "the refused bearer is never sent again"
        );
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// A token endpoint that records the grant each refresh presents.
    async fn answer_refreshes_recording_grants(
        listener: tokio::net::TcpListener,
        grants: Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        while let Ok((mut socket, _)) = listener.accept().await {
            let grants = Arc::clone(&grants);
            tokio::spawn(async move {
                let mut buffer = Vec::with_capacity(4096);
                let body_start = loop {
                    let mut chunk = [0u8; 2048];
                    let read = match socket.read(&mut chunk).await {
                        Ok(0) => return,
                        Ok(read) => read,
                        Err(_) => return,
                    };
                    buffer.extend_from_slice(&chunk[..read]);
                    let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n")
                    else {
                        continue;
                    };
                    let head = String::from_utf8_lossy(&buffer[..end]).to_ascii_lowercase();
                    let length = head
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if buffer.len() >= end + 4 + length {
                        break end + 4;
                    }
                };
                let grant = serde_json::from_slice::<serde_json::Value>(&buffer[body_start..])
                    .ok()
                    .and_then(|body| body.get("refresh_token")?.as_str().map(str::to_string))
                    .unwrap_or_default();
                grants.lock().expect("grants").push(grant);
                let body = r#"{"access_token":"fresh-access","refresh_token":"fresh-refresh","id_token":null}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \
                     {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    }

    /// A store row older than the credential in memory is not installed over it before a refresh.
    ///
    /// The row falls behind in one way: a refresh in this process whose persist failed. The next
    /// refresh re-read the row, installed the stale credential, and presented its retired refresh
    /// token to the issuer, which answered `invalid_grant` for a session that held a live one.
    #[tokio::test]
    async fn a_stale_row_is_not_installed_over_a_newer_credential_in_memory() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock token endpoint");
        let local = listener.local_addr().expect("local addr");
        let grants = Arc::new(std::sync::Mutex::new(Vec::new()));
        tokio::spawn(answer_refreshes_recording_grants(
            listener,
            Arc::clone(&grants),
        ));

        let store = crate::store::Store::for_test().await;
        let token_store = Arc::new(store.token_store());
        let now = crate::oauth::now_epoch_millis();
        token_store
            .save_account_credential("work", &AuthCredential::OAuthToken {
                access_token: "older".to_string(),
                refresh_token: Some("rt-spent".to_string()),
                expires_at: Some(now - 7_200_000),
                account_id: None,
            })
            .await
            .expect("the row the failed persist left behind");

        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                AuthCredential::OAuthToken {
                    access_token: "newer-but-due".to_string(),
                    refresh_token: Some("rt-live".to_string()),
                    expires_at: Some(now - 60_000),
                    account_id: None,
                },
                "gpt-5".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(Some(format!("http://{local}/token")))
            .token_store(Some(token_store))
            .credential_key(Some("work".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");

        let (token, _) = provider
            .ensure_valid_credential()
            .await
            .expect("the refresh runs on the live grant");
        assert_eq!(token, "fresh-access");
        assert_eq!(
            *grants.lock().expect("grants"),
            vec!["rt-live".to_string()],
            "the issuer must see the live refresh token, not the one the stale row holds"
        );
    }

    /// A credential with nothing to refresh with names the remedy. This was the one exit from the
    /// refresh path that did not, so a revoked token whose issuer returned no refresh token read as
    /// an expiry with no way forward.
    #[tokio::test]
    async fn a_token_with_no_refresh_token_names_the_login_remedy() {
        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                AuthCredential::OAuthToken {
                    access_token: "expired".to_string(),
                    refresh_token: None,
                    expires_at: Some(crate::oauth::now_epoch_millis() - 60_000),
                    account_id: None,
                },
                "gpt-5".to_string(),
            )
            .base_url(None)
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("work".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");

        let error = provider
            .ensure_valid_credential()
            .await
            .expect_err("nothing to refresh with");
        assert!(
            error.to_string().contains("meka account login work"),
            "{error}"
        );
    }

    /// A `base_url` ending in `/backend-api/codex`, which `responses_url` accepts, puts the account
    /// endpoints beside `codex`, not under it.
    #[test]
    fn account_urls_for_a_codex_base_land_beside_it() {
        let provider = ChatGptSubscriptionProvider::new(
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::ChatGptSubscription,
                credential_for_test(),
                "gpt-5".to_string(),
            )
            .base_url(Some("https://chatgpt.com/backend-api/codex".to_string()))
            .client_id(None)
            .oauth_token_url(None)
            .token_store(None)
            .credential_key(Some("test".to_string()))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("provider");
        assert_eq!(
            provider.responses_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            provider.usage_url(),
            "https://chatgpt.com/backend-api/wham/usage"
        );
        assert_eq!(
            provider.profiles_url(),
            "https://chatgpt.com/backend-api/wham/profiles/me"
        );
    }
}
