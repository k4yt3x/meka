//! What every OAuth-backed credential shares: expiry arithmetic, the refresh exchange and its
//! cross-process lock, the rejection-and-retry policy, and the PKCE material a login mints.

use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngExt;
use sha2::{Digest, Sha256};

use crate::{
    error::{MekaError, ProviderRequest, Result},
    store::{AuthCredential, TokenStore},
};

/// How long before an access token's stated expiry to refresh it, so a token cannot expire between
/// the header being built and the request arriving.
const OAUTH_REFRESH_SKEW: Duration = Duration::from_secs(5 * 60);
/// How long a refreshed access token is assumed to last when its issuer states no expiry.
///
/// An *unlabeled stored* credential being due is right: refreshing it is how it becomes labeled.
/// An unlabeled credential coming back *from that refresh* is a different thing (the issuer
/// answered and still said nothing), and treating it the same way puts every subsequent request
/// on the slow path: the credential write lock, a database re-read, and a full OAuth round trip
/// that rotates the refresh token, all serialized behind one another. Assuming a short life bounds
/// the staleness without the storm, and a token that dies sooner is corrected by its 401.
const OAUTH_ASSUMED_LIFETIME: Duration = Duration::from_secs(60 * 60);
/// `duration` after an epoch-milliseconds instant, saturating. The credential rows store expiry as
/// epoch milliseconds, which is the one place a `Duration` here meets an integer.
fn epoch_millis_after(now_millis: i64, duration: Duration) -> i64 {
    now_millis.saturating_add(i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
}
/// Expiry to record for a refreshed token whose issuer named none.
pub(crate) fn oauth_assumed_expiry(now_millis: i64) -> i64 {
    epoch_millis_after(now_millis, OAUTH_ASSUMED_LIFETIME)
}
/// Whether an OAuth access token should be refreshed before the next request.
///
/// `expires_at: None` means the issuer did not say when the token expires, which is not a promise
/// that it never will. Read that way, an unlabeled token is a 401 on every request with no refresh
/// ever attempted. It is treated as due, but only when there is a refresh token to act on: without
/// one, the only thing left is to send it and let the 401 speak.
///
/// The refresh paths stamp [`oauth_assumed_expiry`] rather than handing `None` straight back, so
/// "due" here stays a one-shot rather than a per-request loop.
pub(crate) fn oauth_needs_refresh(
    expires_at: Option<i64>,
    has_refresh_token: bool,
    now_millis: i64,
) -> bool {
    match expires_at {
        Some(expiry) => epoch_millis_after(now_millis, OAUTH_REFRESH_SKEW) >= expiry,
        None => has_refresh_token,
    }
}
/// Whether a response is the backend refusing the credential it was sent.
///
/// One predicate for every request site on the two subscription backends, so a 401 means the same
/// thing on a completion, a usage probe and a profile read: refresh once and send again. A 403 is
/// not this; it is an account that lacks the permission, and no refresh changes that.
pub(crate) fn credential_was_rejected(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::UNAUTHORIZED
}
/// A request site's answer to a rejected credential.
///
/// The API-key backends take the defaults: nothing to refresh, nothing to add to the refusal. The
/// subscription backends replace the token once and name the login remedy on the second refusal.
#[async_trait::async_trait]
pub(crate) trait RefreshesCredential {
    /// Whether a rejected credential has been replaced for one more attempt.
    async fn refresh_after_rejection(&self) -> bool {
        false
    }
    /// The error to return when the credential is rejected again.
    fn with_login_remedy(&self, error: MekaError) -> MekaError {
        error
    }
}

/// Send a request, and once more on a credential the backend refused if `refresher` can replace it.
///
/// One rule for every request to a subscription backend, so a 401 means the same thing on a
/// completion, a usage probe and a profile read: refresh once and send again, and on the second
/// refusal return the error with the login remedy. `build` produces each attempt's request, so a
/// backend that refreshes does it there. `transport_error` names a send that never got an answer;
/// any status other than a rejection is the caller's to read.
///
/// `cancellation` ends the wait for the response headers, and only that;
/// `crate::provider::read_whole_reply` is the other half. Dropping the send is what aborts a
/// request on the wire, so a stop reaches a whole reply the provider is still generating as fast as
/// it reaches a stream, and this is the one place every request a turn makes goes out. `build` is
/// deliberately outside the race: a backend may be refreshing its credential there, an exchange
/// followed by a store write that must not be torn, so a stop that arrives during it lands once the
/// refreshed credential is on disk.
pub(crate) async fn send_with_one_refresh<R, B, F>(
    refresher: &R,
    request: ProviderRequest,
    transport_error: impl Fn(&reqwest::Error) -> MekaError,
    mut build: B,
    cancellation: &tokio_util::sync::CancellationToken,
) -> Result<reqwest::Response>
where
    R: RefreshesCredential + Sync + ?Sized,
    B: FnMut() -> F,
    F: Future<Output = Result<reqwest::RequestBuilder>>,
{
    let mut retried_after_rejection = false;
    loop {
        let request_builder = build().await?;
        let sent = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
            sent = request_builder.send() => sent,
        };
        let response = sent.map_err(|error| transport_error(&error))?;
        let status = response.status();
        if !credential_was_rejected(status) {
            return Ok(response);
        }
        if !retried_after_rejection && refresher.refresh_after_rejection().await {
            retried_after_rejection = true;
            continue;
        }
        let retry_after = crate::error::parse_retry_after(response.headers());
        let text = response.text().await.map_err(|error| {
            crate::error::provider_transport_error("failed to read response", &error, retry_after)
        })?;
        return Err(
            refresher.with_login_remedy(crate::error::provider_http_error(
                status,
                &text,
                retry_after,
                request,
            )),
        );
    }
}

/// The error for a credential the backend refused *after* the one refresh a rejection buys.
///
/// Names the remedy, because at this point the token was minted moments ago and refused anyway:
/// the grant behind it is gone, and only signing in again produces one the backend will take. The
/// same words `oauth_refresh_error` uses, so a user greps for one phrase.
pub(crate) fn with_login_remedy(error: MekaError, account: &str) -> MekaError {
    match error {
        MekaError::Provider(message) => MekaError::Provider(format!(
            "{message}; run `meka account login {account}` to sign in again"
        )),
        other => other,
    }
}
/// Milliseconds since the Unix epoch, the unit [`oauth_needs_refresh`] and every stored
/// `expires_at` are in.
///
/// Clamped at zero for a clock before 1970 rather than propagating: every caller is asking "is
/// this token due", and a machine that far out of step has already answered yes.
pub(crate) fn now_epoch_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}
/// Whether a refresh that lost its swap should switch to what the row holds instead.
///
/// Two conditions, and both are about the request in hand rather than about write order.
///
/// It has to be the same *kind* of credential as the one the refresh minted (`refreshed`). A row
/// that held an OAuth token and now holds an API key is a profile somebody repurposed with
/// `meka account add --api-key-stdin`, not a token this refresh has been overtaken by, and a
/// subscription backend cannot authenticate with it at all.
///
/// And it has to be *live*. The winner may have stored a token and then sat idle past its
/// lifetime, so adopting on write order alone would authenticate this very request with a token
/// that is already dead, having thrown away the good one this refresh just minted. Neither
/// outcome writes to the row (what it holds is not this process's to change), so the only
/// question is which token to spend the turn on.
pub(crate) fn is_worth_adopting(refreshed: &AuthCredential, current: &AuthCredential) -> bool {
    match (refreshed, current) {
        (AuthCredential::ApiKey(_), AuthCredential::ApiKey(_)) => true,
        (
            AuthCredential::OAuthToken { .. },
            AuthCredential::OAuthToken {
                expires_at,
                refresh_token,
                ..
            },
        ) => !oauth_needs_refresh(*expires_at, refresh_token.is_some(), now_epoch_millis()),
        _ => false,
    }
}
/// Whether a credential re-read from the store may replace the one in memory before a refresh.
///
/// The row is normally at least as new as memory: the same token, or one a sibling process has
/// rotated to, and a refresh must derive from the newest so the issuer sees a live refresh token.
/// It is older in one case, a refresh in this process whose persist failed, and installing it
/// spent a refresh token the issuer had already retired (`invalid_grant`) while the live one sat in
/// memory. Judged by expiry because that is the one field every refresh advances; an unknown
/// expiry on either side defers to the row.
pub(crate) fn row_is_at_least_as_new(memory: &AuthCredential, row: &AuthCredential) -> bool {
    match (memory, row) {
        (
            AuthCredential::OAuthToken {
                expires_at: Some(memory_expiry),
                ..
            },
            AuthCredential::OAuthToken {
                expires_at: Some(row_expiry),
                ..
            },
        ) => row_expiry >= memory_expiry,
        _ => true,
    }
}
/// How long to wait for another process to finish rotating a profile's credential before going
/// ahead anyway.
///
/// Bounded rather than blocking, because a wedged holder must not wedge every other meka on the
/// machine. Going ahead is safe: [`store_refreshed_credential`]'s compare-and-swap is what makes
/// the outcome correct, and this only spares the wasted refresh in the common case.
pub(crate) const CREDENTIAL_LOCK_WAIT: Duration = Duration::from_secs(10);
/// Wait, briefly, for exclusive use of a profile's credential across processes.
///
/// `None` means the wait ran out or the lock could not be asked for at all. The caller proceeds
/// either way: this reduces contention, it does not establish correctness.
pub(crate) async fn await_credential_lock(
    store: &TokenStore,
    profile: &str,
) -> Option<crate::fs::FileLock> {
    let deadline = tokio::time::Instant::now() + CREDENTIAL_LOCK_WAIT;
    loop {
        match store.try_lock_account_credential(profile) {
            Ok(Some(lock)) => return Some(lock),
            Ok(None) => {}
            // Not "someone has it" but "we could not ask": an unwritable lock directory, or
            // descriptors exhausted. Retrying would not help and refusing to refresh would be
            // worse than refreshing unserialized, so stop waiting and let the swap arbitrate.
            Err(error) => {
                tracing::debug!("failed to take the credential lock for '{profile}': {error}");
                return None;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            tracing::debug!(
                "another process has been refreshing '{profile}' for {CREDENTIAL_LOCK_WAIT:?}; going ahead"
            );
            return None;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
/// How long an OAuth refresh round trip may take before it is abandoned.
///
/// A whole-request deadline is right here in a way it never is for a turn: this is a small request
/// to a well-known endpoint and nothing legitimate about it takes minutes. It also runs behind the
/// gate that serializes a profile's refreshes, and *inside* `complete`/`stream` via
/// `ensure_valid_credential`, so a token endpoint that accepts the connection and then goes quiet
/// would otherwise hold both the next refresher and the user's turn in progress.
///
/// It cannot usefully go below `CONNECT_TIMEOUT`, which is the trap in tightening it: reqwest's
/// per-request `timeout` runs from the start of connecting, not from the request being sent, so a
/// value under 30 seconds would abandon a connection this same client is still willing to spend 30
/// seconds establishing. The two numbers being equal means a slow connect can consume the whole
/// refresh budget, which is the correct precedence: the total is what bounds the user's wait.
pub(crate) const REFRESH_TIMEOUT: Duration = Duration::from_secs(30);
/// Everything an OAuth refresh-token exchange needs, named rather than positional.
///
/// A struct because the alternative is six parameters of which five are `&str`, where transposing
/// two of them still compiles and produces a request that fails at the issuer with a message about
/// the wrong thing. Named fields make the two call sites say which is which.
pub(crate) struct RefreshExchange<'a> {
    pub(crate) client: &'a reqwest::Client,
    /// The issuer's token endpoint.
    pub(crate) token_url: &'a str,
    /// The OAuth client this profile authenticated as.
    pub(crate) client_id: &'a str,
    /// The grant being spent. Single-use under rotation; see
    /// [`crate::error::oauth_refresh_error`].
    pub(crate) refresh_token: &'a str,
    /// The account whose credential this is, named in the error so a user knows which one to log
    /// back in to.
    pub(crate) account: &'a str,
    /// The call site's own description of the request ("OAuth token refresh", "Codex OAuth token
    /// refresh"), so its messages do not all read alike.
    pub(crate) context: &'a str,
}
/// Spend a refresh token at an issuer's token endpoint and hand back its decoded response.
///
/// One copy for both subscription backends: POST the grant, classify the answer, decode the
/// payload. Only the payload differs (Claude's issuer states `expires_in` and an account uuid,
/// ChatGPT's states neither and has to be read out of the JWTs), so that is the generic parameter.
/// Two hand-written copies of a classification are two chances to get it wrong and two places to
/// remember when it changes. Both call sites keep a test of their own even so, since each still
/// has to prove it reaches this function at all.
///
/// Errors are classified where they belong rather than here: a call that got no answer through
/// [`crate::error::provider_transport_error`], and an answered one through
/// [`crate::error::oauth_refresh_error`], which is the function that knows a spent grant from an
/// unwell server.
pub(crate) async fn exchange_refresh_token<T: serde::de::DeserializeOwned>(
    exchange: RefreshExchange<'_>,
) -> Result<T> {
    let response = exchange
        .client
        .post(exchange.token_url)
        .timeout(REFRESH_TIMEOUT)
        // One order for both backends, which otherwise list the same three fields in two different
        // orders. `serde_json` is built with `preserve_order`, so this does change the bytes the
        // Claude endpoint receives; it cannot change what they mean, because a JSON object is an
        // unordered collection (RFC 8259 §1) and nothing here is signed over. Worth saying only
        // because this is the backend whose request shape is otherwise reproduced from a capture.
        .json(&serde_json::json!({
            "client_id": exchange.client_id,
            "grant_type": "refresh_token",
            "refresh_token": exchange.refresh_token,
        }))
        .send()
        .await
        .map_err(|error| {
            // Retryable like any other call that got no answer. The grant's fate is the one
            // wrinkle, and `oauth_refresh_error` records why a possible replay is still the better
            // trade than ending a turn on a blip.
            crate::error::provider_transport_error(
                &format!("{} request failed", exchange.context),
                &error,
                None,
            )
        })?;

    let status = response.status();
    let retry_after = crate::error::parse_retry_after(response.headers());
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|error| {
            tracing::warn!(
                "failed to read the OAuth refresh error body for '{account}': {error}",
                account = exchange.account
            );
            String::new()
        });
        return Err(crate::error::oauth_refresh_error(
            &format!("{} failed", exchange.context),
            status,
            &body,
            retry_after,
            exchange.account,
        ));
    }

    response.json::<T>().await.map_err(|error| {
        crate::error::oauth_refresh_error(
            &format!(
                "{context} failed to read the response",
                context = exchange.context
            ),
            status,
            &crate::error::format_reqwest_error(&error),
            None,
            exchange.account,
        )
    })
}
/// Persist a refreshed credential, and answer with the one that should actually be used.
///
/// A refresh is derived from the credential it read, so it may only replace *that* credential.
/// Where the row has moved on (another process refreshed first, or a `meka account login`
/// completed while this round trip was in flight) the stored value is newer than what this
/// refresh produced, and adopting it is both correct and what keeps the issuer's live token and the
/// database in agreement. A blind upsert leaves them disagreeing silently, and the symptom arrives
/// at the *next* launch as `invalid_grant` with nothing naming the cause.
///
/// A write that cannot be made at all is a warning rather than an error: the session in hand still
/// has a working token, and failing the turn over a persistence problem would be the worse trade.
///
/// `observed_version` is the row's version as the refresh saw it when it re-read the row; the swap
/// replaces the row only if it is still at that version. A refresh that could not read the row
/// first (`None`) asks for the version now, so it still cannot overwrite a sibling's newer write
/// unseen.
pub(crate) async fn store_refreshed_credential(
    store: &TokenStore,
    account: &str,
    observed_version: Option<&str>,
    refreshed: AuthCredential,
) -> AuthCredential {
    let version = match observed_version {
        Some(version) => version.to_string(),
        None => match store.account_credential_version(account).await {
            Ok(Some(version)) => version,
            Ok(None) => {
                tracing::warn!(
                    "'{account}' no longer has a stored credential; the refreshed token was not \
                     persisted"
                );
                return refreshed;
            }
            Err(error) => {
                tracing::warn!(
                    "failed to read the stored credential's version for '{account}': {error}; the \
                     next launch will need `meka account login`"
                );
                return refreshed;
            }
        },
    };
    match store
        .replace_account_credential(account, &version, &refreshed)
        .await
    {
        Ok(crate::store::CredentialWrite::Stored) => refreshed,
        // Adopted only when it is worth adopting: "newer" here means newer in *write order*, which
        // is neither "unexpired" nor "the same kind of credential". See [`is_worth_adopting`].
        Ok(crate::store::CredentialWrite::Superseded(current))
            if is_worth_adopting(&refreshed, &current.credential) =>
        {
            tracing::info!(
                "'{account}' was re-authenticated elsewhere while this refresh was in flight; adopting \
                 the stored credential"
            );
            current.credential
        }
        Ok(crate::store::CredentialWrite::Superseded(_)) => {
            tracing::warn!(
                "'{account}' was rewritten during this refresh with a credential this session cannot \
                 use; continuing on the refreshed token"
            );
            refreshed
        }
        // The account's credential was removed mid-refresh, which is `meka account remove` or a
        // `logout`. Writing this token back would resurrect an account the user just disconnected;
        // this process finishes its turn on what it has and the next launch asks for a login.
        Ok(crate::store::CredentialWrite::Gone) => {
            tracing::warn!(
                "'{account}' no longer has a stored credential; the refreshed token was not persisted"
            );
            refreshed
        }
        Err(error) => {
            tracing::warn!(
                "failed to persist the refreshed token for '{account}': {error}; the next \
                 launch will need `meka account login`"
            );
            refreshed
        }
    }
}
/// A PKCE verifier and its S256 challenge, for a login's authorization request.
pub(crate) fn generate_pkce_pair() -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::rng().fill(&mut bytes);
    let code_verifier = URL_SAFE_NO_PAD.encode(bytes);
    let digest = Sha256::digest(code_verifier.as_bytes());
    let code_challenge = URL_SAFE_NO_PAD.encode(digest);
    (code_verifier, code_challenge)
}
/// A fresh `state` value for a login's authorization request.
pub(crate) fn generate_state() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stop must reach a request the provider is still answering. The wait for a response is
    /// the one place a whole reply spends its time and nothing else on this side may end it, so
    /// the send is raced against the token here, where every request goes out. The peer holds
    /// the connection open and never answers, which is what a model still generating looks like
    /// from this side.
    #[tokio::test]
    async fn a_stop_ends_the_wait_for_a_response_at_once() {
        struct NoRefresh;
        impl RefreshesCredential for NoRefresh {}

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        let peer = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.expect("accept");
            std::future::pending::<()>().await;
        });
        let client = reqwest::Client::new();
        let cancellation = tokio_util::sync::CancellationToken::new();
        let stop = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                cancellation.cancel();
            }
        });
        let started = std::time::Instant::now();
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            send_with_one_refresh(
                &NoRefresh,
                crate::error::ProviderRequest::Completion,
                |error| MekaError::Provider(error.to_string()),
                || async { Ok(client.get(format!("http://{address}/v1/messages"))) },
                &cancellation,
            ),
        )
        .await
        .expect("the stop must end the wait, not the test's own deadline");
        stop.await.expect("the stop landed");
        assert!(
            matches!(outcome, Err(MekaError::Interrupted)),
            "a stop is reported as the interruption it is: {outcome:?}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "the stop must end the wait at once, not when the peer gives up: {:?}",
            started.elapsed()
        );
        peer.abort();
    }

    /// A refresh that loses its swap adopts what the row holds only when that can authenticate the
    /// request in hand.
    ///
    /// "Superseded" is a statement about write order and nothing else, and adopting on write order
    /// alone throws away a token the issuer has just minted in favor of one that may be dead or
    /// may not even be the same kind of credential. The failure is quiet in both directions: an
    /// expired adoption 401s the very request it was fetched for, and an API key adopted into a
    /// subscription profile is sent as a bearer token that endpoint has never accepted.
    #[tokio::test]
    async fn a_superseding_credential_is_adopted_only_when_it_can_authenticate() {
        let store = crate::store::Store::for_test().await;
        let token_store = store.token_store();
        let oauth = |access: &str, expires_at: Option<i64>| AuthCredential::OAuthToken {
            access_token: access.to_string(),
            refresh_token: Some("refresh".to_string()),
            expires_at,
            account_id: None,
        };
        let hour = 3_600_000;
        let minted = || oauth("just-minted", Some(now_epoch_millis() + hour));
        // Every case below is a lost swap: a version the row is no longer at, which is what a
        // sibling's write between the read and this swap leaves behind.
        let stale_version = Some("2000-01-01T00:00:00+00:00");
        let plant = async |credential: AuthCredential| {
            token_store
                .save_account_credential("work", &credential)
                .await
                .expect("plant what the winner left");
        };
        let access_token = |credential: &AuthCredential| match credential {
            AuthCredential::OAuthToken { access_token, .. } => access_token.clone(),
            AuthCredential::ApiKey(key) => key.clone(),
        };

        plant(oauth("theirs", Some(now_epoch_millis() + hour))).await;
        assert_eq!(
            access_token(
                &store_refreshed_credential(&token_store, "work", stale_version, minted()).await
            ),
            "theirs",
            "the premise: a live credential from the winner is the one to use"
        );

        plant(oauth("stale", Some(now_epoch_millis() - hour))).await;
        assert_eq!(
            access_token(
                &store_refreshed_credential(&token_store, "work", stale_version, minted()).await
            ),
            "just-minted",
            "a token that won the write and then expired cannot authenticate this request"
        );

        plant(AuthCredential::ApiKey("repurposed".to_string())).await;
        assert_eq!(
            access_token(
                &store_refreshed_credential(&token_store, "work", stale_version, minted()).await
            ),
            "just-minted",
            "and a profile somebody repurposed to an API key is not a newer version of this token"
        );
    }
    /// An issuer that omits `expires_in` is not promising the token is eternal. Reading `None` as
    /// "valid forever" meant the refresh path was never entered, so the credential 401'd on every
    /// request for the life of the process with nothing naming why.
    #[test]
    fn an_unlabeled_expiry_is_refreshed_rather_than_trusted_forever() {
        let now = 1_700_000_000_000;
        assert!(oauth_needs_refresh(None, true, now));

        // With no refresh token there is nothing to act on, so send it and let the 401 speak.
        assert!(!oauth_needs_refresh(None, false, now));

        // A stated expiry is still honored, skew included, and still refuses to refresh early.
        assert!(oauth_needs_refresh(Some(now + 60_000), true, now));
        assert!(!oauth_needs_refresh(Some(now + 600_000), true, now));

        // A `now` near the top of the range must not wrap into "not yet due".
        assert!(oauth_needs_refresh(Some(i64::MAX), true, i64::MAX));
    }
    /// And "due" has to stay a one-shot.
    ///
    /// Handing `None` straight back out of a refresh whose issuer stated no expiry made the token
    /// due again the instant it arrived, so every request re-entered the slow path -- credential
    /// write lock, database re-read, full OAuth round trip -- serialized, rotating the refresh
    /// token on each pass. The refresh paths stamp an assumed expiry instead.
    #[test]
    fn a_refresh_that_states_no_expiry_is_not_due_again_immediately() {
        let now = 1_700_000_000_000;
        assert!(
            !oauth_needs_refresh(Some(oauth_assumed_expiry(now)), true, now),
            "a token that just arrived must not already be due",
        );
        assert!(
            oauth_needs_refresh(
                Some(oauth_assumed_expiry(now)),
                true,
                epoch_millis_after(now, OAUTH_ASSUMED_LIFETIME)
            ),
            "but the assumption expires: it bounds staleness, it does not trust the token forever",
        );
    }
    #[test]
    fn generate_pkce_pair_challenge_is_sha256_of_verifier() {
        let (verifier, challenge) = generate_pkce_pair();
        let expected = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        assert_eq!(challenge, expected);
    }
    #[test]
    fn two_generated_states_differ() {
        assert_ne!(generate_state(), generate_state());
    }
}
