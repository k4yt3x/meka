//! The scope catalog and the check every authenticated handler runs against it.
//!
//! Scopes gate verbs, not rows: there are no tenants, so every configured token shares one session
//! namespace and a scope only decides which endpoints a token may reach. See
//! [`crate::host::http::auth`] for how a token resolves to a [`Principal`].
//!
//! The catalog lives here rather than next to the router because two very distant places need to
//! agree on it: the handlers, which demand a scope, and config resolution, which warns about a
//! configured scope no handler will ever ask for.

use axum::http::StatusCode;

use super::{
    auth::Principal,
    errors::{ErrorKind, ProblemDetail},
};

/// Every scope meka recognizes.
///
/// Session-scoped operations (turns, compaction, rewind, export, background tasks) sit under
/// `sessions:*`, because the thing being read or changed is one conversation. The stores meka owns
/// process-wide get their own pairs, so an operator can hand a bridge the ability to run turns
/// without also handing it the ability to empty the memory store.
///
/// Kept sorted, and kept in lockstep with the scope table in the HTTP API docs and the `bearerAuth`
/// description in [`crate::host::http::openapi`].
pub(crate) const KNOWN_SCOPES: &[&str] = &[
    "mcp:r",
    "mcp:w",
    "memory:r",
    "memory:w",
    "schedule:r",
    "schedule:w",
    "sessions:r",
    "sessions:w",
    "skills:r",
    "skills:w",
];

/// Scopes that admit the server-level discovery endpoints (`/v1/info`, `/v1/skills`, `/v1/mcp`,
/// `/v1/profiles`).
///
/// Any read scope is enough, deliberately: a token configured with `sessions:r` so a client can
/// list sessions should also be able to see which model it is talking to, without an operator
/// having to also grant `mcp:r` and `skills:r` for what is not sensitive information.
pub(crate) const ANY_READ_SCOPES: &[&str] =
    &["sessions:r", "mcp:r", "skills:r", "memory:r", "schedule:r"];

// `ProblemDetail` is ~128 bytes and only constructed on the rejection path. Same trade-off as
// `extract_bearer` in auth.rs; see the rationale there.
/// Require one named scope. The rejection names the missing scope, so a client that gets a 403
/// learns what to ask its operator for rather than having to diff against the docs.
pub(crate) fn require(principal: &Principal, scope: &str) -> Result<(), ProblemDetail> {
    if principal.has_scope(scope) {
        return Ok(());
    }
    Err(ProblemDetail::new(
        ErrorKind::AuthScope,
        StatusCode::FORBIDDEN,
        format!("scope '{scope}' is required"),
    ))
}

/// Require at least one of `scopes`. Used by the discovery endpoints; see [`ANY_READ_SCOPES`].
pub(crate) fn require_any(principal: &Principal, scopes: &[&str]) -> Result<(), ProblemDetail> {
    if scopes.iter().any(|scope| principal.has_scope(scope)) {
        return Ok(());
    }
    let names = scopes
        .iter()
        .map(|scope| format!("'{scope}'"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(ProblemDetail::new(
        ErrorKind::AuthScope,
        StatusCode::FORBIDDEN,
        format!("one of {names} is required"),
    ))
}

/// A scope a handler requires, as a type: the requirement is the handler's signature rather than a
/// call it has to remember to make, so a handler that takes a [`Scoped`] cannot run without it.
pub(crate) trait Required: Send + Sync + 'static {
    /// Any one of these admits the caller.
    const SCOPES: &'static [&'static str];
}

pub(crate) struct McpRead;
pub(crate) struct McpWrite;
pub(crate) struct MemoryRead;
pub(crate) struct MemoryWrite;
pub(crate) struct ScheduleRead;
pub(crate) struct ScheduleWrite;
pub(crate) struct SessionsRead;
pub(crate) struct SessionsWrite;
pub(crate) struct SkillsRead;
pub(crate) struct SkillsWrite;
/// Any read scope at all: the read-only surfaces that describe the server rather than a store.
pub(crate) struct AnyRead;

impl Required for McpRead {
    const SCOPES: &'static [&'static str] = &["mcp:r"];
}
impl Required for McpWrite {
    const SCOPES: &'static [&'static str] = &["mcp:w"];
}
impl Required for MemoryRead {
    const SCOPES: &'static [&'static str] = &["memory:r"];
}
impl Required for MemoryWrite {
    const SCOPES: &'static [&'static str] = &["memory:w"];
}
impl Required for ScheduleRead {
    const SCOPES: &'static [&'static str] = &["schedule:r"];
}
impl Required for ScheduleWrite {
    const SCOPES: &'static [&'static str] = &["schedule:w"];
}
impl Required for SessionsRead {
    const SCOPES: &'static [&'static str] = &["sessions:r"];
}
impl Required for SessionsWrite {
    const SCOPES: &'static [&'static str] = &["sessions:w"];
}
impl Required for SkillsRead {
    const SCOPES: &'static [&'static str] = &["skills:r"];
}
impl Required for SkillsWrite {
    const SCOPES: &'static [&'static str] = &["skills:w"];
}
impl Required for AnyRead {
    const SCOPES: &'static [&'static str] = ANY_READ_SCOPES;
}

/// The caller, admitted for `R`. Extracting it is the scope check.
pub(crate) struct Scoped<R: Required> {
    pub(crate) principal: Principal,
    required: std::marker::PhantomData<R>,
}

impl<S, R> axum::extract::FromRequestParts<S> for Scoped<R>
where
    S: Send + Sync,
    R: Required,
{
    type Rejection = ProblemDetail;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        // The auth layer puts the principal on every authenticated route; a route without one is a
        // routing mistake, not a caller's, and says so as a 500 rather than a 403.
        let principal = parts
            .extensions
            .get::<Principal>()
            .cloned()
            .ok_or_else(|| {
                ProblemDetail::new(
                    ErrorKind::Internal,
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "no principal on an authenticated route",
                )
            })?;
        admitted(&principal, R::SCOPES)?;
        Ok(Self {
            principal,
            required: std::marker::PhantomData,
        })
    }
}

/// The single-scope wording for a single scope, the any-of wording otherwise.
fn admitted(principal: &Principal, scopes: &[&str]) -> Result<(), ProblemDetail> {
    match scopes {
        [scope] => require(principal, scope),
        _ => require_any(principal, scopes),
    }
}

/// Warn about a configured scope no handler will ever demand.
///
/// A warning rather than a hard error, in both directions: rejecting would mean a config written
/// for a newer meka cannot start an older binary, and staying silent would mean a plausible typo
/// like `sessions:write` grants nothing at all while looking like it grants everything. Called once
/// per token at config-resolve time.
pub(crate) fn warn_unknown(scopes: &[String], token_description: Option<&str>) {
    for scope in scopes {
        if KNOWN_SCOPES.contains(&scope.as_str()) {
            continue;
        }
        let unknown = crate::text::unknown_name("scope", scope, KNOWN_SCOPES);
        match token_description {
            Some(description) => tracing::warn!(
                "serve token '{description}' declares a scope that grants nothing: {unknown}"
            ),
            None => {
                tracing::warn!("a serve token declares a scope that grants nothing: {unknown}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn principal(scopes: &[&str]) -> Principal {
        Principal {
            token_id: "test".to_string(),
            scopes: scopes
                .iter()
                .map(|s| (*s).to_string())
                .collect::<Vec<_>>()
                .into(),
        }
    }

    #[test]
    fn require_admits_the_exact_scope_and_nothing_else() {
        let holder = principal(&["memory:r"]);
        assert!(require(&holder, "memory:r").is_ok());
        assert!(require(&holder, "memory:w").is_err());
        // Read does not imply write and write does not imply read: the catalog is flat.
        let writer = principal(&["memory:w"]);
        assert!(require(&writer, "memory:r").is_err());
    }

    /// The 403 body has to name the scope, or a client holding the wrong token cannot tell which
    /// of several scopes an endpoint wanted.
    #[test]
    fn rejection_names_the_missing_scope() {
        let problem = require(&principal(&[]), "schedule:w").expect_err("no scopes held");
        assert_eq!(problem.status, 403);
        let detail = problem.detail.expect("detail is always set");
        assert!(detail.contains("schedule:w"), "{detail}");
    }

    #[test]
    fn require_any_admits_a_single_match() {
        assert!(require_any(&principal(&["skills:r"]), ANY_READ_SCOPES).is_ok());
        assert!(require_any(&principal(&["sessions:w"]), ANY_READ_SCOPES).is_err());
    }

    #[test]
    fn require_any_rejection_lists_every_candidate() {
        let problem = require_any(&principal(&[]), &["mcp:r", "mcp:w"]).expect_err("no scopes");
        let detail = problem.detail.expect("detail is always set");
        assert!(detail.contains("'mcp:r'"), "{detail}");
        assert!(detail.contains("'mcp:w'"), "{detail}");
    }

    /// Every scope a handler can demand must be in the catalog, or config resolution warns about
    /// a scope that actually works. `ANY_READ_SCOPES` is the easiest one to forget when a new
    /// subsystem lands.
    #[test]
    fn any_read_scopes_are_all_cataloged() {
        for scope in ANY_READ_SCOPES {
            assert!(
                KNOWN_SCOPES.contains(scope),
                "'{scope}' is demanded by a handler but missing from KNOWN_SCOPES"
            );
        }
    }

    #[test]
    fn known_scopes_are_sorted_and_unique() {
        let mut sorted = KNOWN_SCOPES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.as_slice(), KNOWN_SCOPES);
    }

    /// `warn_unknown` has no return value; this pins the classification it warns on so a rename of
    /// a real scope cannot silently start warning about every token that uses it.
    #[test]
    fn typo_scopes_are_not_in_the_catalog() {
        for typo in ["sessions:write", "session:r", "skills", "memory:rw", ""] {
            assert!(!KNOWN_SCOPES.contains(&typo), "'{typo}' must not be known");
        }
    }

    #[test]
    fn principal_scopes_arc_is_cheap_to_clone() {
        // Guards the `Arc<[String]>` representation the middleware depends on for per-request
        // cloning; a switch to `Vec<String>` would silently make every request allocate.
        let holder = principal(&["sessions:r"]);
        let cloned = holder.clone();
        assert!(Arc::ptr_eq(&holder.scopes, &cloned.scopes));
    }
}
