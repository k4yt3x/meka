//! Bearer-token authentication for `meka serve`. Reads `Authorization: Bearer <token>` from each
//! incoming request, matches against the configured token list using constant-time comparison
//! (so a near-miss can't be distinguished from an early mismatch via timing), and attaches the
//! matched principal to the request extensions for downstream handlers to consult.
//!
//! No tenants: all configured tokens share the same session namespace. Scopes (`sessions:r`,
//! `sessions:w`, `skills:r`, `mcp:r`) gate which endpoints a token can hit. See the HTTP API docs
//! for the full scope catalog.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::Next,
    response::Response,
};
use subtle::ConstantTimeEq;

use super::errors::{ErrorKind, ProblemDetail};
use crate::host::http::config::ResolvedServeToken;

/// Authenticated principal, attached as a request extension by [`bearer_auth`].
#[derive(Debug, Clone)]
pub(crate) struct Principal {
    /// Stable per-token identifier, the first 8 bytes of a SHA-256 of the token, hex-encoded.
    /// Used as the per-token key in the idempotency cache and structured logs (never the raw
    /// token, which we don't want appearing in observability output).
    pub(crate) token_id: String,
    pub(crate) scopes: Arc<[String]>,
}

impl Principal {
    /// Returns true iff this principal holds the named scope.
    pub(crate) fn has_scope(&self, required: &str) -> bool {
        self.scopes.iter().any(|s| s == required)
    }
}

/// One configured token paired with the `Principal` it resolves to. The fingerprint and scope
/// `Arc` are computed once at construction so per-request `lookup` only pays for the constant-time
/// compare plus a cheap `Principal` clone on match.
struct AuthEntry {
    token: String,
    principal: Principal,
}

/// Shared state the middleware reads on every request. Owned by the top-level `ServerState` and
/// cloned into the middleware via axum's `State` extractor.
///
/// The table is replaced whole each time the server serves `config.toml`, so a token added,
/// removed or rescoped there applies to the next request, and an open stream whose token no
/// longer grants what it needs ends ([`Self::revoked`]).
#[derive(Clone)]
pub(crate) struct AuthRegistry {
    entries: Arc<tokio::sync::watch::Sender<Arc<[AuthEntry]>>>,
}

// Manual `Debug` so the raw token strings never reach logs: only the count and the non-secret
// fingerprints are printed, mirroring `ResolvedServeToken`'s redacting `Debug`.
impl std::fmt::Debug for AuthRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entries = Arc::clone(&self.entries.borrow());
        f.debug_struct("AuthRegistry")
            .field("token_count", &entries.len())
            .field(
                "token_ids",
                &entries
                    .iter()
                    .map(|entry| entry.principal.token_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl AuthRegistry {
    pub(crate) fn new(tokens: Vec<ResolvedServeToken>) -> Self {
        let (entries, _receiver) = tokio::sync::watch::channel(Self::entries_of(tokens));
        Self {
            entries: Arc::new(entries),
        }
    }

    fn entries_of(tokens: Vec<ResolvedServeToken>) -> Arc<[AuthEntry]> {
        tokens
            .into_iter()
            .map(|entry| AuthEntry {
                principal: Principal {
                    token_id: token_fingerprint(&entry.token),
                    scopes: entry.scopes.into(),
                },
                token: entry.token,
            })
            .collect()
    }

    /// Replace the table with `tokens`, which the next request is checked against.
    pub(crate) fn replace(&self, tokens: Vec<ResolvedServeToken>) {
        self.entries.send_replace(Self::entries_of(tokens));
    }

    /// Resolves once the table no longer grants `principal` every scope in `required`: its token
    /// removed or changed, or one of the scopes taken from it. Never resolves while it does, so a
    /// stream opened under the token ends when the token stops covering it.
    ///
    /// The table is read before every wait, the first included: one replaced between the request's
    /// own check and this subscription is a change the subscription would never report.
    pub(crate) fn revoked(
        &self,
        principal: &Principal,
        required: &'static [&'static str],
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        let mut receiver = self.entries.subscribe();
        let token_id = principal.token_id.clone();
        async move {
            loop {
                let grants = receiver.borrow_and_update().iter().any(|entry| {
                    entry.principal.token_id == token_id
                        && required
                            .iter()
                            .all(|scope| entry.principal.has_scope(scope))
                });
                if !grants {
                    return;
                }
                if receiver.changed().await.is_err() {
                    std::future::pending::<()>().await;
                }
            }
        }
    }

    /// Constant-time lookup. Iterates every configured token even on early match so a timing
    /// observer cannot distinguish "matched index 0" from "matched index N" (or "no match"). Each
    /// request therefore costs `O(N * token_length)`, which is fine for the handful of tokens a
    /// deployment configures.
    pub(crate) fn lookup(&self, presented: &str) -> Option<Principal> {
        let presented = presented.as_bytes();
        let entries = Arc::clone(&self.entries.borrow());
        let mut matched: Option<&Principal> = None;
        for entry in entries.iter() {
            // ConstantTimeEq returns Choice::from(1u8) on match. Bytes-of-different-length never
            // match (no early return based on length difference).
            let candidate = entry.token.as_bytes();
            let is_match: bool = if candidate.len() == presented.len() {
                candidate.ct_eq(presented).into()
            } else {
                // Dummy ct_eq to equalize timing on length-mismatch.
                let _: bool = candidate.ct_eq(candidate).into();
                false
            };
            if is_match && matched.is_none() {
                matched = Some(&entry.principal);
            }
        }
        matched.cloned()
    }
}

/// Stable, non-reversible per-token identifier for logs and per-token cache keys. SHA-256
/// truncated to 8 bytes (16 hex chars), enough collision resistance for log correlation, never
/// long enough to reconstruct the token. `pub(crate)` so other modules (startup logging,
/// idempotency cache keying) share one source of truth for the fingerprint format.
pub(crate) fn token_fingerprint(token: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(token.as_bytes());
    let mut out = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// axum middleware: extracts the bearer token, looks it up in the registry, and attaches the
/// matched [`Principal`] to the request via `Extensions`. Routes that want to gate on a scope
/// extract the principal and call [`Principal::has_scope`] (or, preferred, use a small
/// extractor helper that does both in one step).
pub(crate) async fn bearer_auth(
    State(registry): State<AuthRegistry>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Response {
    let presented = match extract_bearer(&headers) {
        Ok(token) => token,
        Err(problem) => {
            tracing::debug!("authentication refused: {title}", title = problem.title);
            return problem
                .instance(request.uri().path().to_string())
                .into_response();
        }
    };
    let Some(principal) = registry.lookup(&presented) else {
        let problem = ProblemDetail::new(
            ErrorKind::Auth,
            StatusCode::UNAUTHORIZED,
            "bearer token does not match any configured entry",
        )
        .instance(request.uri().path().to_string());
        return problem.into_response();
    };
    request.extensions_mut().insert(principal);
    next.run(request).await
}

// `ProblemDetail` is ~128 bytes (it carries several `String`s and a `BTreeMap`). The lint flags
// the `Err` variant as large; for an auth-failure path that fires at most once per request, the
// extra stack space is fine and boxing would just shuffle the same allocation under the heap.
fn extract_bearer(headers: &HeaderMap) -> Result<String, ProblemDetail> {
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return Err(ProblemDetail::new(
            ErrorKind::Auth,
            StatusCode::UNAUTHORIZED,
            "Authorization header is required",
        ));
    };
    let value = value.to_str().map_err(|_| {
        // Non-ASCII bytes → header is present but unparseable: "invalid", not "missing".
        ProblemDetail::new(
            ErrorKind::Auth,
            StatusCode::UNAUTHORIZED,
            "Authorization header is not a valid ASCII string",
        )
    })?;
    // RFC 7235 §2.1 says the auth-scheme is case-insensitive and must be followed by at least one
    // whitespace character before the token. Leading whitespace before the scheme is tolerated.
    let trimmed = value.trim_start();
    let token = trimmed
        .get(..6)
        .filter(|prefix| prefix.eq_ignore_ascii_case("Bearer"))
        .and_then(|_| trimmed.get(6..))
        .and_then(|rest| {
            // The byte after "Bearer" must be ASCII whitespace per the grammar.
            rest.chars().next().filter(|c| c.is_ascii_whitespace())?;
            Some(rest.trim_start())
        });
    let Some(token) = token else {
        return Err(ProblemDetail::new(
            ErrorKind::Auth,
            StatusCode::UNAUTHORIZED,
            "Authorization header must use the `Bearer` scheme",
        ));
    };
    Ok(token.trim().to_string())
}

use axum::response::IntoResponse;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::http::config::TokenSource;

    fn entry(token: &str, scopes: &[&str]) -> ResolvedServeToken {
        ResolvedServeToken {
            token: token.to_string(),
            description: None,
            scopes: scopes.iter().map(|s| (*s).to_string()).collect(),
            source: TokenSource::Inline,
        }
    }

    #[test]
    fn lookup_matches_exact_token() {
        let registry = AuthRegistry::new(vec![entry("sk_test_a", &["sessions:r", "sessions:w"])]);
        let principal = registry.lookup("sk_test_a").expect("match");
        assert!(principal.has_scope("sessions:r"));
        assert!(principal.has_scope("sessions:w"));
        // Token_id is the fingerprint, not the raw token. Never log the raw value.
        assert_ne!(principal.token_id, "sk_test_a");
        assert_eq!(principal.token_id.len(), 16);
    }

    #[test]
    fn lookup_rejects_unknown_token() {
        let registry = AuthRegistry::new(vec![entry("sk_test_a", &["sessions:r"])]);
        assert!(registry.lookup("sk_test_b").is_none());
    }

    #[test]
    fn lookup_rejects_prefix_attack() {
        // Constant-time compare must reject a presented token that's a prefix of a configured
        // one (and vice versa); the length-mismatch branch in `lookup` handles this.
        let registry = AuthRegistry::new(vec![entry("sk_test_secret", &["sessions:r"])]);
        assert!(registry.lookup("sk_test_").is_none());
        assert!(registry.lookup("sk_test_secrets").is_none());
    }

    /// A token dropped between the request's check and the stream's subscription is revoked at
    /// once, and a stream that needs two scopes ends when either goes.
    #[tokio::test(start_paused = true)]
    async fn a_revocation_reads_the_table_it_subscribes_to() {
        let registry = AuthRegistry::new(vec![entry("sk_writer", &["sessions:r", "sessions:w"])]);
        let principal = registry.lookup("sk_writer").expect("match");
        registry.replace(Vec::new());
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            registry.revoked(&principal, &["sessions:r"]),
        )
        .await
        .expect("revoked before the subscription");

        registry.replace(vec![entry("sk_writer", &["sessions:r", "sessions:w"])]);
        let principal = registry.lookup("sk_writer").expect("match");
        let revoked = registry.revoked(&principal, &["sessions:r", "sessions:w"]);
        tokio::pin!(revoked);
        registry.replace(vec![entry("sk_writer", &["sessions:r"])]);
        tokio::time::timeout(std::time::Duration::from_secs(1), &mut revoked)
            .await
            .expect("revoked when `sessions:w` goes");
    }

    /// A replaced table applies to the next lookup, and a stream's revocation resolves once its
    /// token loses the scope it needs, and not while a change leaves that scope in place.
    #[tokio::test(start_paused = true)]
    async fn a_replaced_table_applies_at_once_and_revokes_what_it_drops() {
        let registry = AuthRegistry::new(vec![entry("sk_reader", &["sessions:r", "mcp:r"])]);
        let principal = registry.lookup("sk_reader").expect("match");
        let revoked = registry.revoked(&principal, &["sessions:r"]);
        tokio::pin!(revoked);

        registry.replace(vec![entry("sk_reader", &["sessions:r"])]);
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), &mut revoked)
                .await
                .is_err(),
            "the scope it needs is still granted"
        );

        registry.replace(vec![entry("sk_other", &["sessions:r"])]);
        assert!(registry.lookup("sk_reader").is_none());
        assert!(registry.lookup("sk_other").is_some());
        tokio::time::timeout(std::time::Duration::from_secs(1), &mut revoked)
            .await
            .expect("the token is gone, so the stream ends");
    }

    #[test]
    fn has_scope_exact_match() {
        let p = Principal {
            token_id: "abc".into(),
            scopes: vec!["sessions:r".into(), "mcp:r".into()].into(),
        };
        assert!(p.has_scope("sessions:r"));
        assert!(p.has_scope("mcp:r"));
        assert!(!p.has_scope("sessions:w"));
    }

    #[test]
    fn fingerprint_is_stable_and_truncated() {
        let a = token_fingerprint("sk_test");
        let b = token_fingerprint("sk_test");
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        let c = token_fingerprint("sk_test_other");
        assert_ne!(a, c);
    }
}
