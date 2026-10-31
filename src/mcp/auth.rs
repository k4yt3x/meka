//! OAuth / JWT authentication and HTTP callback handling for MCP HTTP transports. Also houses the
//! probe that classifies an unauthenticated response (RFC 6750 + RFC 9728), the best-effort token
//! revocation path (RFC 7009), and the SQLite-backed credential store threaded into `rmcp`'s
//! `AuthorizationManager`.

use async_trait::async_trait;
use rmcp::{
    ClientServiceExt,
    transport::{
        AuthClient, AuthError, AuthorizationManager, ClientCredentialsConfig, CredentialStore,
        StoredCredentials, auth::OAuthState,
    },
};

use super::{McpRunningService, handler::MekaClientHandler};
use crate::{
    config::McpAuthConfig,
    error::{MekaError, Result},
    oauth::{Completion, LoginPrompt, Paste},
    store::TokenStore,
};

/// The endpoint a stored MCP credential is bound to and a server is reached at: `url` in the
/// canonical form RFC 8707 gives a resource, which the MCP authorization spec names a server by.
/// Scheme and host lowercase and a default port dropped, as the URL parser leaves them, and no
/// fragment or trailing slash. A string that is not a URL stands for itself.
pub(crate) fn canonical_endpoint(url: &str) -> String {
    let url = url.trim();
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return url.to_string();
    };
    let mut canonical = parsed.origin().ascii_serialization();
    canonical.push_str(parsed.path().trim_end_matches('/'));
    if let Some(query) = parsed.query() {
        canonical.push('?');
        canonical.push_str(query);
    }
    canonical
}

/// Whether a credential of `kind` stored for `bound` may be presented to the server at `url`. An
/// OAuth bundle at the resource it was issued for alone: a token is bound to its audience (RFC
/// 8707, the MCP authorization spec). A bearer or a client secret anywhere on the origin it was
/// given for, which is where a client sends a credential it holds (RFC 9110's protection space),
/// and nowhere else: a credential never follows a server to another host.
pub(crate) fn presentable(kind: crate::store::McpCredentialKind, bound: &str, url: &str) -> bool {
    match kind {
        crate::store::McpCredentialKind::OAuth => bound == canonical_endpoint(url),
        crate::store::McpCredentialKind::Bearer | crate::store::McpCredentialKind::ClientSecret => {
            !bound.is_empty() && same_origin(bound, url)
        }
    }
}

/// The origin of `endpoint`, the most of it a listing or a refusal shows: a path or a query may
/// carry what the file keeps in an environment variable. `None` for a credential bound to no
/// server.
pub(crate) fn endpoint_origin(endpoint: &str) -> Option<String> {
    if endpoint.is_empty() {
        return None;
    }
    Some(match reqwest::Url::parse(endpoint) {
        Ok(parsed) => parsed.origin().ascii_serialization(),
        Err(_) => endpoint.to_string(),
    })
}

/// The `secret` stored for `server_name`, when it may be presented to `url`; refused by name when
/// it was stored for another endpoint, rather than sent there or silently left out.
pub(crate) async fn presented_secret(
    token_store: &TokenStore,
    server_name: &str,
    secret: crate::error::McpSecretKind,
    url: &str,
) -> Result<Option<String>> {
    let kind = crate::store::McpCredentialKind::from(secret);
    let Some(stored) = token_store.load_mcp_credentials(server_name, kind).await? else {
        return Ok(None);
    };
    if presentable(kind, &stored.endpoint, url) {
        return Ok(Some(stored.secret));
    }
    Err(MekaError::McpSecretForAnotherUrl {
        server_name: server_name.to_string(),
        secret,
        stored_for: endpoint_origin(&stored.endpoint),
    })
}

/// Whether two URLs name the same origin: scheme, host and port, the tuple that decides who
/// receives a credential. Path is excluded: an MCP endpoint at `/mcp` and its authorization
/// server's metadata at the site root are one origin. Two strings that are not URLs are the same
/// only when equal, and one URL is never the same as a string that is not one.
pub(crate) fn same_origin(left: &str, right: &str) -> bool {
    match (reqwest::Url::parse(left), reqwest::Url::parse(right)) {
        (Ok(left), Ok(right)) => {
            left.scheme() == right.scheme()
                && left.host_str() == right.host_str()
                && left.port_or_known_default() == right.port_or_known_default()
        }
        (Err(_), Err(_)) => left == right,
        _ => false,
    }
}

/// The revocation a door owes for the OAuth grant it cleared (RFC 7009), read before the clear and
/// sent once it commits: the local clear is what a logout or a removal guarantees, and the
/// authorization server answering slowly or not at all must hold up nothing behind it. Nothing is
/// owed when no grant was stored or it held no token.
#[must_use = "a cleared grant stays live at the provider until this is sent"]
pub(crate) struct Revocation(Option<RevocationRequest>);

/// One RFC 7009 request: the token, and the client that holds it.
struct RevocationRequest {
    server_name: String,
    /// The authorization server's issuer, whose metadata names the revocation endpoint: the one
    /// the grant records, else the origin it was stored for, whose endpoints a discovery that
    /// found no metadata falls back to.
    issuer: String,
    /// The refresh token, which revokes the grant whole where the server supports it, else the
    /// access token.
    token: String,
    token_type_hint: &'static str,
    client_id: String,
    /// The secret a confidential client authenticates with, which RFC 7009 asks of it.
    client_secret: Option<String>,
}

impl std::fmt::Debug for Revocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(request) => formatter
                .debug_struct("Revocation")
                .field("server_name", &request.server_name)
                .field("issuer", &request.issuer)
                .finish_non_exhaustive(),
            None => formatter.write_str("Revocation(none)"),
        }
    }
}

impl Revocation {
    /// Nothing owed.
    pub(crate) const fn none() -> Self {
        Self(None)
    }

    /// Whether a request is owed.
    #[cfg(test)]
    pub(crate) const fn is_owed(&self) -> bool {
        self.0.is_some()
    }

    /// The client secret the request authenticates with.
    #[cfg(test)]
    pub(crate) fn client_secret_for_test(&self) -> Option<&str> {
        self.0.as_ref()?.client_secret.as_deref()
    }

    /// Send what is owed, best-effort: a failure is the log's to say, since the grant is already
    /// cleared here and nothing a caller does changes the authorization server's answer.
    pub(crate) async fn send(self) {
        let Some(request) = self.0 else {
            return;
        };
        let server_name = request.server_name.clone();
        if let Err(error) = request.post().await {
            tracing::warn!("failed to revoke the OAuth tokens at '{server_name}': {error}");
        }
    }
}

/// The revocation of the grant stored for `server_name`, owed once its clear commits. A grant that
/// cannot be read is logged and revokes nothing: the clear goes on. The client secret goes along
/// only when it was stored for the grant's own origin, which is where the flow that minted the
/// grant presented it; one stored for another endpoint belongs to another client.
pub(crate) async fn stored_revocation(
    token_store: &TokenStore,
    server_name: &str,
) -> Result<Revocation> {
    let Some(stored) = token_store
        .load_mcp_credentials(server_name, crate::store::McpCredentialKind::OAuth)
        .await?
    else {
        return Ok(Revocation::none());
    };
    let credentials: StoredCredentials = match serde_json::from_str(&stored.secret) {
        Ok(credentials) => credentials,
        Err(error) => {
            tracing::warn!(
                "failed to read the stored OAuth tokens for '{server_name}' to revoke them: {error}"
            );
            return Ok(Revocation::none());
        }
    };
    let Some(issuer) = credentials
        .issuer
        .clone()
        .or_else(|| endpoint_origin(&stored.endpoint))
    else {
        tracing::warn!(
            "the stored OAuth tokens for '{server_name}' name no issuer and no endpoint to revoke \
             them at"
        );
        return Ok(Revocation::none());
    };
    // The token response as rmcp serializes it, the standard fields by their RFC 6749 names.
    let tokens = serde_json::to_value(&credentials.token_response).unwrap_or_default();
    let field = |name: &str| {
        tokens
            .get(name)
            .and_then(|value| value.as_str())
            .map(str::to_string)
    };
    let (token, token_type_hint) = match (field("refresh_token"), field("access_token")) {
        (Some(refresh), _) => (refresh, "refresh_token"),
        (None, Some(access)) => (access, "access_token"),
        (None, None) => return Ok(Revocation::none()),
    };
    let client_secret = token_store
        .load_mcp_credentials(server_name, crate::store::McpCredentialKind::ClientSecret)
        .await?
        .filter(|secret| {
            presentable(
                crate::store::McpCredentialKind::ClientSecret,
                &secret.endpoint,
                &stored.endpoint,
            )
        })
        .map(|secret| secret.secret);
    Ok(Revocation(Some(RevocationRequest {
        server_name: server_name.to_string(),
        issuer,
        token,
        token_type_hint,
        client_id: credentials.client_id,
        client_secret,
    })))
}

/// `value` in the form encoding RFC 6749 Appendix B asks of a client's identity in HTTP Basic.
fn form_encoded(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

/// Where the authorization server `issuer` publishes its metadata, in the order the MCP
/// authorization spec tries them: RFC 8414's path, then OpenID Connect Discovery's, each inserted
/// ahead of the issuer's own path, and for an issuer with a path OpenID Connect Discovery's
/// appended form last.
fn metadata_candidates(issuer: &str) -> Vec<String> {
    let Ok(parsed) = reqwest::Url::parse(issuer) else {
        return Vec::new();
    };
    let origin = parsed.origin().ascii_serialization();
    let path = parsed.path().trim_end_matches('/');
    let mut candidates = vec![
        format!("{origin}/.well-known/oauth-authorization-server{path}"),
        format!("{origin}/.well-known/openid-configuration{path}"),
    ];
    if !path.is_empty() {
        candidates.push(format!("{origin}{path}/.well-known/openid-configuration"));
    }
    candidates
}

impl RevocationRequest {
    async fn post(self) -> std::result::Result<(), String> {
        let Self {
            server_name,
            issuer,
            token,
            token_type_hint,
            client_id,
            client_secret,
        } = self;
        // The `issuer` URL was stored during the original flow, so its origin is trusted; the
        // network path and any redirect are not. reqwest follows redirects by default,
        // which would let a MITM send the metadata fetch to an attacker host and the access
        // token after it, so redirects are off, the body is size-capped, and
        // `revocation_endpoint` is pinned to the issuer's origin.
        const MAX_METADATA_BODY_BYTES: usize = 256 * crate::text::KIB;
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| format!("failed to build the HTTP client: {error}"))?;

        let mut revocation_endpoint: Option<String> = None;
        for url in &metadata_candidates(&issuer) {
            let Ok(response) = http.get(url).send().await else {
                continue;
            };
            if !response.status().is_success() {
                continue;
            }
            // Bounded as it arrives rather than after: Content-Length is server-supplied, and a
            // body read whole before the cap is checked is a body the cap did not
            // bound.
            let Some(bytes) = read_bounded(response, MAX_METADATA_BODY_BYTES).await else {
                continue;
            };
            if let Some(endpoint) = extract_revocation_endpoint(&bytes, MAX_METADATA_BODY_BYTES) {
                revocation_endpoint = Some(endpoint);
                break;
            }
        }

        let Some(endpoint) = revocation_endpoint else {
            return Err(format!(
                "server '{server_name}' does not advertise a revocation_endpoint"
            ));
        };

        validate_revocation_endpoint_origin(&issuer, &endpoint)?;

        let mut request = http.post(&endpoint).header(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        );
        let mut body = vec![
            ("token", token.as_str()),
            ("token_type_hint", token_type_hint),
        ];
        // A confidential client authenticates as RFC 6749 section 2.3.1 has it, by HTTP Basic over
        // the pair encoded by its Appendix B, the form encoding; a public one names itself in the
        // body.
        match &client_secret {
            Some(secret) => {
                request = request.basic_auth(form_encoded(&client_id), Some(form_encoded(secret)));
            }
            None => body.push(("client_id", client_id.as_str())),
        }
        let body = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(body)
            .finish();

        let response = request.body(body).send().await.map_err(|error| {
            format!(
                "failed to POST the revocation: {}",
                crate::error::format_reqwest_error(&error)
            )
        })?;
        // RFC 7009: successful revocation is 200 OK with an empty body; unknown tokens also return
        // 200 OK. Non-2xx is a genuine failure.
        if !response.status().is_success() {
            return Err(format!(
                "the revocation endpoint returned HTTP {}",
                response.status().as_u16()
            ));
        }
        Ok(())
    }
}

/// Classification of an unauthenticated probe to an HTTP MCP endpoint. The MCP authorization spec
/// (2025-03-26) layers on RFC 6750 + RFC 9728: a server that requires auth answers unauthenticated
/// requests with `401` and a `WWW-Authenticate: Bearer …` challenge, optionally advertising a
/// `resource_metadata` URL we can fetch to learn which authorization servers + scopes to use.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum McpAuthProbe {
    /// Server answered 2xx: reachable and doesn't require auth.
    Open,
    /// Server answered 401 / 403 with a `Bearer` challenge. The optional URL is the RFC 9728
    /// protected-resource-metadata document.
    AuthRequired { resource_metadata: Option<String> },
    /// Reachable but some other status (405, 404, …). Record it so the caller can surface it
    /// without claiming auth is or isn't needed.
    Unexpected { status: u16 },
    /// No answer at all (DNS, TLS, timeout).
    Unreachable { message: String },
}

/// Probe an MCP HTTP endpoint to see whether it requires OAuth.
///
/// Runs an unauthenticated `GET` with a 3 s wall-clock timeout and redirects disabled, so a
/// compromised DNS cannot make an attacker host authoritative about the real server. The body is
/// ignored; the verdict comes entirely from the status line and the `WWW-Authenticate` header.
pub(crate) async fn probe_http_auth(url: &str) -> McpAuthProbe {
    let http = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .redirect(reqwest::redirect::Policy::none())
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return McpAuthProbe::Unreachable {
                message: format!("failed to build the HTTP client: {error}"),
            };
        }
    };
    let response = match http.get(url).send().await {
        Ok(response) => response,
        Err(error) => {
            return McpAuthProbe::Unreachable {
                message: crate::error::format_reqwest_error(&error),
            };
        }
    };
    let status = response.status().as_u16();
    let www_authenticate = response
        .headers()
        .get(reqwest::header::WWW_AUTHENTICATE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    classify_probe_response(status, www_authenticate.as_deref())
}

/// Pure classifier for the probe: takes the status code + optional `WWW-Authenticate` header and
/// returns the [`McpAuthProbe`] verdict. Extracted so the RFC-6750 / RFC-9728 parsing can be
/// unit-tested without a live HTTP server.
fn classify_probe_response(status: u16, www_authenticate: Option<&str>) -> McpAuthProbe {
    if (200..300).contains(&status) {
        return McpAuthProbe::Open;
    }
    if status == 401 || status == 403 {
        let header = www_authenticate.unwrap_or("");
        // RFC 6750 §3: the challenge must start with `Bearer`. Match case-insensitively; tolerate
        // the scheme with or without any `key=value` parameters following.
        let first = header.split(',').next().unwrap_or("").trim();
        let is_bearer = first.eq_ignore_ascii_case("Bearer")
            || first
                .split_once(|c: char| c.is_whitespace())
                .map(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
                .unwrap_or(false);
        if !is_bearer {
            return McpAuthProbe::Unexpected { status };
        }
        return McpAuthProbe::AuthRequired {
            resource_metadata: extract_bearer_param(header, "resource_metadata"),
        };
    }
    McpAuthProbe::Unexpected { status }
}

/// Extract a parameter value from an RFC 6750 `WWW-Authenticate: Bearer …` challenge, quoted or
/// bare, with trailing commas and mixed whitespace tolerated. `None` when the key is absent.
fn extract_bearer_param(header: &str, key: &str) -> Option<String> {
    // Drop the `Bearer` scheme prefix; everything after is a comma-separated parameter list.
    let index = header.find(|c: char| c.is_whitespace())?;
    let params = &header[index..];
    for pair in params.split(',') {
        let pair = pair.trim();
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        if !k.trim().eq_ignore_ascii_case(key) {
            continue;
        }
        let v = v.trim();
        let unquoted = v
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(v);
        return Some(unquoted.to_string());
    }
    None
}

/// The body, or `None` once it has grown past `cap` or the connection failed.
async fn read_bounded(mut response: reqwest::Response, cap: usize) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.ok()? {
        if bytes.len().saturating_add(chunk.len()) > cap {
            return None;
        }
        bytes.extend_from_slice(&chunk);
    }
    Some(bytes)
}

/// Parse an OAuth authorization-server metadata JSON document and return the `revocation_endpoint`
/// string, if any. Rejects bodies larger than `max_bytes` and invalid JSON. Split from
/// [`Revocation::send`] so the size-cap and extraction logic are testable without a live HTTP
/// server.
fn extract_revocation_endpoint(bytes: &[u8], max_bytes: usize) -> Option<String> {
    if bytes.len() > max_bytes {
        return None;
    }
    let metadata = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
    metadata
        .get("revocation_endpoint")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Verify that `endpoint` has the same scheme, host, and effective port as `issuer`. Prevents a
/// compromised metadata document from redirecting the access-token POST to an attacker-controlled
/// host.
fn validate_revocation_endpoint_origin(
    issuer: &str,
    endpoint: &str,
) -> std::result::Result<(), String> {
    let issuer_url = reqwest::Url::parse(issuer)
        .map_err(|error| format!("stored issuer '{issuer}' is not a valid URL: {error}"))?;
    let endpoint_url = reqwest::Url::parse(endpoint)
        .map_err(|error| format!("revocation_endpoint '{endpoint}' is not a valid URL: {error}"))?;
    if endpoint_url.scheme() != issuer_url.scheme()
        || endpoint_url.host_str() != issuer_url.host_str()
        || endpoint_url.port_or_known_default() != issuer_url.port_or_known_default()
    {
        return Err(format!(
            "revocation_endpoint '{endpoint}' is on a different origin than issuer '{issuer}'; refusing to send the token"
        ));
    }
    Ok(())
}

pub(super) async fn connect_http_with_oauth(
    server_name: &str,
    url: &str,
    auth_config: &McpAuthConfig,
    transport_config: rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig,
    token_store: Option<&TokenStore>,
    login_prompt: Option<std::sync::Arc<dyn LoginPrompt>>,
    handler: MekaClientHandler,
) -> Result<McpRunningService> {
    // The client half of this server's identity, from the store, for the flows that present one:
    // `client_credentials` requires one and `oauth` takes one for a confidential client, while
    // `client_credentials_jwt` signs instead of sharing a secret, so a secret it holds is never
    // asked whether it may be presented.
    let presents_a_secret = !matches!(auth_config, McpAuthConfig::ClientCredentialsJwt { .. });
    let client_secret = match token_store {
        Some(store) if presents_a_secret => {
            presented_secret(
                store,
                server_name,
                crate::error::McpSecretKind::ClientSecret,
                url,
            )
            .await?
        }
        _ => None,
    };

    let auth_manager = match auth_config {
        McpAuthConfig::ClientCredentials { client_id, scopes } => {
            // Required for this flow, and there is nothing to fall back to: without it meka cannot
            // prove which client it is and the token endpoint will refuse.
            let secret = client_secret.as_deref().ok_or_else(|| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!(
                    "no client secret stored for '{server_name}', which `client_credentials` \
                     presents"
                ),
            })?;
            authenticate_client_credentials(server_name, url, client_id, secret, scopes.as_deref())
                .await?
        }
        McpAuthConfig::ClientCredentialsJwt {
            client_id,
            signing_key_path,
            signing_algorithm,
            scopes,
        } => {
            authenticate_client_credentials_jwt(
                server_name,
                url,
                client_id,
                signing_key_path,
                signing_algorithm.as_deref(),
                scopes.as_deref(),
            )
            .await?
        }
        McpAuthConfig::OAuth {
            client_id,
            scopes,
            redirect_port,
        } => {
            // Optional here: a public client has none, and a confidential one's is whatever the
            // store holds. Either way this is the only place it can come from now.
            authenticate_oauth_authorization_code(
                OAuthLogin {
                    server_name,
                    url,
                    client_id: client_id.as_deref(),
                    client_secret: client_secret.as_deref(),
                    scopes: scopes.as_deref(),
                    redirect_port: *redirect_port,
                },
                token_store,
                login_prompt,
            )
            .await?
        }
    };

    let auth_client = AuthClient::new(reqwest::Client::new(), auth_manager);
    let transport =
        rmcp::transport::StreamableHttpClientTransport::with_client(auth_client, transport_config);

    handler
        .serve_with_lifecycle(transport, super::handler::http_lifecycle())
        .await
        .map_err(|error| MekaError::McpConnection {
            server_name: server_name.to_string(),
            message: format!("HTTP connection failed: {error}"),
        })
}

async fn authenticate_client_credentials(
    server_name: &str,
    url: &str,
    client_id: &str,
    client_secret: &str,
    scopes: Option<&[String]>,
) -> Result<AuthorizationManager> {
    // The token is asked for the server it will be presented to, the resource the MCP
    // authorization spec has every token request name (RFC 8707), which is a fact of the URL and
    // not a setting.
    let config = ClientCredentialsConfig::ClientSecret {
        client_id: client_id.to_string(),
        client_secret: client_secret.to_string(),
        scopes: scopes.map(|s| s.to_vec()).unwrap_or_default(),
        resource: Some(canonical_endpoint(url)),
    };

    let mut oauth_state = OAuthState::new(url, None)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth: {error}"),
        })?;

    oauth_state
        .authenticate_client_credentials(config)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("client credentials authentication failed: {error}"),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after client credentials authentication".to_string(),
        })
}

async fn authenticate_client_credentials_jwt(
    server_name: &str,
    url: &str,
    client_id: &str,
    signing_key_path: &str,
    signing_algorithm: Option<&str>,
    scopes: Option<&[String]>,
) -> Result<AuthorizationManager> {
    // Open the key once and check permissions on the open fd to close the stat-then-read TOCTOU
    // window: a separate `metadata(path)` followed by `read(path)` could land on a different inode
    // if the path was swapped between syscalls. `File::metadata` walks the open descriptor.
    let mut key_file =
        std::fs::File::open(signing_key_path).map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to open signing key '{signing_key_path}': {error}"),
        })?;
    require_private_key_permissions_on_fd(server_name, signing_key_path, &key_file)?;
    let mut signing_key = Vec::new();
    {
        use std::io::Read;
        key_file
            .read_to_end(&mut signing_key)
            .map_err(|error| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to read signing key '{signing_key_path}': {error}"),
            })?;
    }

    let algorithm = parse_jwt_signing_algorithm(server_name, signing_algorithm)?;

    let config = ClientCredentialsConfig::PrivateKeyJwt {
        client_id: client_id.to_string(),
        signing_key,
        signing_algorithm: algorithm,
        scopes: scopes.map(|s| s.to_vec()).unwrap_or_default(),
        // As `authenticate_client_credentials` names it.
        resource: Some(canonical_endpoint(url)),
        token_endpoint_audience: None,
    };

    let mut oauth_state = OAuthState::new(url, None)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth: {error}"),
        })?;

    oauth_state
        .authenticate_client_credentials(config)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("JWT client credentials authentication failed: {error}"),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after JWT authentication".to_string(),
        })
}

/// On Unix, refuse to read a JWT signing key that is group- or world-accessible: a key another
/// local user can read lets them forge JWTs to the MCP server. The same 0600 policy the store and
/// `config.toml` get.
///
/// Takes the open `File` so the permission check and the subsequent read share the same inode. A
/// stat-then-read pair on the path could be swapped between syscalls.
///
/// No-op elsewhere: Windows uses ACLs, which are not audited.
fn require_private_key_permissions_on_fd(
    server_name: &str,
    path: &str,
    file: &std::fs::File,
) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = file.metadata().map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to stat signing key '{path}': {error}"),
        })?;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("signing key '{path}' has mode {mode:o}; it must be 0600"),
            });
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (server_name, path, file);
    }
    Ok(())
}

fn parse_jwt_signing_algorithm(
    server_name: &str,
    algorithm: Option<&str>,
) -> Result<rmcp::transport::JwtSigningAlgorithm> {
    use rmcp::transport::JwtSigningAlgorithm;

    match algorithm.unwrap_or("RS256") {
        "RS256" => Ok(JwtSigningAlgorithm::RS256),
        "RS384" => Ok(JwtSigningAlgorithm::RS384),
        "RS512" => Ok(JwtSigningAlgorithm::RS512),
        "ES256" => Ok(JwtSigningAlgorithm::ES256),
        "ES384" => Ok(JwtSigningAlgorithm::ES384),
        other => Err(MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!(
                "unsupported signing algorithm '{other}': expected RS256, RS384, RS512, ES256, or ES384"
            ),
        }),
    }
}

/// What an authorization-code login is for: the server, and the client meka presents itself as.
struct OAuthLogin<'a> {
    server_name: &'a str,
    url: &'a str,
    client_id: Option<&'a str>,
    client_secret: Option<&'a str>,
    scopes: Option<&'a [String]>,
    redirect_port: Option<u16>,
}

async fn authenticate_oauth_authorization_code(
    login: OAuthLogin<'_>,
    token_store: Option<&TokenStore>,
    login_prompt: Option<std::sync::Arc<dyn LoginPrompt>>,
) -> Result<AuthorizationManager> {
    let OAuthLogin {
        server_name,
        url,
        client_id,
        client_secret,
        scopes,
        redirect_port,
    } = login;
    let scope_strings: Vec<String> = scopes.map(|s| s.to_vec()).unwrap_or_default();

    let mut manager = AuthorizationManager::new(url)
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth manager: {error}"),
        })?;

    // Set up persistent credential storage if available. The cell is held here as well as inside
    // the adapter, so the interactive flow below can clear it; see the reset after
    // `initialize_from_store`.
    let last_read = std::sync::Arc::new(std::sync::Mutex::new(LastRead::Nothing));
    if let Some(store) = token_store {
        manager.set_credential_store(SqliteCredentialStore {
            token_store: store.clone(),
            server_name: server_name.to_string(),
            endpoint: canonical_endpoint(url),
            last_read: std::sync::Arc::clone(&last_read),
        });
    }

    let has_stored_credentials =
        manager
            .initialize_from_store()
            .await
            .map_err(|error| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to load stored credentials: {error}"),
            })?;

    if has_stored_credentials {
        tracing::info!("loaded stored OAuth credentials for MCP server '{server_name}'");
        return Ok(manager);
    }

    // No stored credentials; run the interactive browser flow, if anyone is there to complete it.
    let Some(login_prompt) = login_prompt else {
        return Err(MekaError::McpLoginRequired {
            server_name: server_name.to_string(),
        });
    };

    // Bound here, once a login is going to run, and not at the top: every connect of an OAuth
    // server comes through this function, and binding first would refuse a server with a good
    // stored bundle whenever the port is taken, by another meka or by a second server sharing
    // `redirect_port`. Bound before the URL is built so a random port (`redirect_port = None`) can
    // be learned and put in `redirect_uri`.
    let bind_port = redirect_port.unwrap_or(0);
    let callback_listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{bind_port}"))
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to bind callback server on port {bind_port}: {error}"),
        })?;
    let actual_port = callback_listener
        .local_addr()
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to read the callback listener's address: {error}"),
        })?
        .port();
    let redirect_uri = format!("http://127.0.0.1:{actual_port}/callback");

    // Cleared first, and this is load-bearing. `initialize_from_store` can itself write (on an
    // issuer change it loads, saves cleared tokens, and returns `false`), and if that write loses
    // its swap the adapter is left `Superseded`, which refuses every later write. The interactive
    // flow performs no further `load`, so the credential the user is about to authorize in a
    // browser would be silently dropped and the next launch would ask again. Nothing this flow
    // mints is derived from a stored value, so there is nothing here for a swap to protect.
    forget_what_was_read(&last_read);
    tracing::info!("starting OAuth authorization flow for MCP server '{server_name}'");

    let mut oauth_state = OAuthState::Unauthorized(manager);

    // A configured `client_id` skips dynamic registration, which `start_authorization` would
    // otherwise do; configuring the client needs the metadata resolved first.
    if let Some(id) = client_id {
        if let OAuthState::Unauthorized(ref mut manager) = oauth_state {
            let resolution =
                manager
                    .resolve_metadata()
                    .await
                    .map_err(|error| MekaError::McpAuth {
                        server_name: server_name.to_string(),
                        message: format!("OAuth metadata discovery failed: {error}"),
                    })?;
            manager.set_metadata(resolution.metadata);

            let mut oauth_client_config =
                rmcp::transport::auth::OAuthClientConfig::new(id.to_string(), redirect_uri.clone());
            if let Some(secret) = client_secret {
                oauth_client_config = oauth_client_config.with_client_secret(secret.to_string());
            }
            if !scope_strings.is_empty() {
                oauth_client_config = oauth_client_config.with_scopes(scope_strings.clone());
            }
            manager
                .configure_client(oauth_client_config)
                .map_err(|error| MekaError::McpAuth {
                    server_name: server_name.to_string(),
                    message: format!("failed to configure OAuth client: {error}"),
                })?;

            let scope_refs: Vec<&str> = scope_strings.iter().map(|s| s.as_str()).collect();
            let auth_url = manager
                .get_authorization_url(&scope_refs)
                .await
                .map_err(|error| MekaError::McpAuth {
                    server_name: server_name.to_string(),
                    message: format!("failed to get authorization URL: {error}"),
                })?;

            let session = rmcp::transport::AuthorizationSession::for_scope_upgrade(
                std::mem::replace(
                    manager,
                    AuthorizationManager::new("http://localhost")
                        .await
                        .map_err(|error| MekaError::McpAuth {
                            server_name: server_name.to_string(),
                            message: format!("internal error: {error}"),
                        })?,
                ),
                auth_url,
                &redirect_uri,
            );
            oauth_state = OAuthState::Session(session);
        }
    } else {
        let request = rmcp::transport::auth::AuthorizationRequest::new(redirect_uri.clone())
            .with_scopes(scope_strings.clone())
            .with_client_name("meka");
        oauth_state
            .start_authorization(request)
            .await
            .map_err(|error| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to start OAuth authorization: {error}"),
            })?;
    }

    let auth_url =
        oauth_state
            .get_authorization_url()
            .await
            .map_err(|error| MekaError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to get authorization URL: {error}"),
            })?;

    login_prompt.authorize_at(&auth_url);

    let expected_state = state_in_authorization_url(&auth_url);
    let callback = await_oauth_callback(
        callback_listener,
        login_prompt.as_ref(),
        expected_state.as_deref(),
    )
    .await
    .map_err(|error| MekaError::McpAuth {
        server_name: server_name.to_string(),
        message: format!("OAuth callback failed: {error}"),
    })?;

    // The `iss` the authorization server answered with (RFC 9207), which rmcp checks against the
    // issuer it discovered and requires where the server's metadata says it sends one.
    oauth_state
        .handle_callback_with_issuer(&callback.code, &callback.state, callback.issuer.as_deref())
        .await
        .map_err(|error| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("OAuth token exchange failed: {error}"),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| MekaError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after authorization".to_string(),
        })
}

/// What an authorization response carries back (RFC 6749 section 4.1.2, RFC 9207): the code, the
/// `state` that ties it to this login, and the issuer that sent it, when it says.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct OAuthCallback {
    pub(crate) code: String,
    pub(crate) state: String,
    pub(crate) issuer: Option<String>,
}

/// Overall wall-clock budget for the OAuth callback wait, shared by both the TCP accept path and
/// the paste-URL fallback.
pub(crate) const OAUTH_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Wait for the authorization code.
///
/// The common case is the authorization server redirecting the user's browser to the localhost
/// listener, and the code is read out of that request. When meka runs on a different host than
/// the browser (SSH sessions, containers) the browser cannot reach back, so the TCP accept is
/// raced against a prompt that takes the full callback URL pasted from the browser's address bar,
/// which shows it even when the connection is refused. Paste mode is only offered when the prompt
/// can read one.
async fn await_oauth_callback(
    listener: tokio::net::TcpListener,
    prompt: &dyn LoginPrompt,
    expected_state: Option<&str>,
) -> std::result::Result<OAuthCallback, String> {
    let deadline = tokio::time::Instant::now() + OAUTH_CALLBACK_TIMEOUT;
    let accepts_paste = prompt.accepts_paste();
    prompt.awaiting(Completion {
        listener_seconds: Some(OAUTH_CALLBACK_TIMEOUT.as_secs()),
        paste: accepts_paste.then_some(Paste::CallbackUrl),
    });
    if !accepts_paste {
        return accept_http_callback(listener, expected_state, deadline).await;
    }

    tokio::select! {
        result = accept_http_callback(listener, expected_state, deadline) => result,
        result = read_pasted_callback(prompt, expected_state, deadline) => result,
    }
}

/// This login's `state`, as the authorization URL carries it: what a callback must bring back to be
/// this login's, checked ahead of the exchange so a stale tab or another login's paste is refused
/// and the wait goes on.
fn state_in_authorization_url(auth_url: &str) -> Option<String> {
    url::Url::parse(auth_url)
        .ok()?
        .query_pairs()
        .find(|(key, _)| key == "state")
        .map(|(_, value)| value.into_owned())
}

/// Accept one HTTP request on the bound listener, validate it's the OAuth callback, extract what it
/// carries, and send back a success page. Loops past non-callback requests (favicons,
/// preflights) until the shared deadline elapses.
async fn accept_http_callback(
    listener: tokio::net::TcpListener,
    expected_state: Option<&str>,
    overall_deadline: tokio::time::Instant,
) -> std::result::Result<OAuthCallback, String> {
    use tokio::io::AsyncWriteExt;

    loop {
        let remaining = overall_deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(format!(
                "authorization timed out after {}s",
                OAUTH_CALLBACK_TIMEOUT.as_secs()
            ));
        }

        let accept_result = tokio::time::timeout(remaining, listener.accept()).await;
        let (mut stream, _addr) = match accept_result {
            Err(_) => {
                return Err(format!(
                    "authorization timed out after {}s",
                    OAUTH_CALLBACK_TIMEOUT.as_secs()
                ));
            }
            Ok(Err(error)) => return Err(format!("failed to accept connection: {error}")),
            Ok(Ok(pair)) => pair,
        };

        // A browser may send favicon or preflight requests to the callback origin; a path other
        // than `/callback?...` gets a 404 so it stops retrying, and the wait continues.
        let buffer = match crate::oauth::read_callback_request(&mut stream, overall_deadline).await
        {
            None => {
                return Err(format!(
                    "authorization timed out after {}s",
                    OAUTH_CALLBACK_TIMEOUT.as_secs()
                ));
            }
            Some(crate::oauth::CallbackRequest::Abandoned) => continue,
            Some(crate::oauth::CallbackRequest::Oversized) => {
                tracing::debug!("OAuth callback: dropped a request with oversized headers");
                if let Err(error) = stream
                    .write_all(b"HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await {
                    tracing::debug!("failed to answer the callback client: {error}");
                }
                continue;
            }
            Some(crate::oauth::CallbackRequest::Head(buffer)) => buffer,
        };

        let request = String::from_utf8_lossy(&buffer);
        match parse_callback_query(&request) {
            Ok(callback)
                if expected_state.is_some_and(|expected| expected != callback.state.as_str()) =>
            {
                // A stale tab, or another login's browser: not this login's code, and refused
                // ahead of the exchange so the one this login asked for is still waited on.
                tracing::warn!("OAuth state mismatch; refusing the callback");
                let response_body = "<!DOCTYPE html><html><body>\
                    <h1>Authorization refused</h1>\
                    <p>This callback is not the login meka is waiting for.</p>\
                    </body></html>";
                let response = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                if let Err(error) = stream.write_all(response.as_bytes()).await {
                    tracing::debug!("failed to send callback response: {error}");
                }
                continue;
            }
            Ok(callback) => {
                // No claim of success: the exchange comes after this page, and may still fail.
                let response_body = "<!DOCTYPE html><html><body>\
                    <h1>Return to meka</h1>\
                    <p>The authorization was received; meka is completing the login. \
                    You can close this tab.</p>\
                    </body></html>";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                if let Err(error) = stream.write_all(response.as_bytes()).await {
                    tracing::debug!("failed to send callback response: {error}");
                }
                return Ok(callback);
            }
            Err(CallbackParseError::NotCallbackPath) => {
                // Almost certainly a browser preflight or favicon request. Respond 404 and keep
                // waiting for the real callback.
                tracing::debug!("OAuth callback: ignored non-callback request on callback port");
                if let Err(error) = stream
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await
                {
                    tracing::debug!("failed to answer the callback client: {error}");
                }
                continue;
            }
            Err(CallbackParseError::Malformed(message)) => {
                if let Err(error) = stream
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await {
                    tracing::debug!("failed to answer the callback client: {error}");
                }
                // Ends the login only when it is this login's answer: an error or a missing code
                // under another state, or none, is anyone's to send.
                if expected_state.is_some_and(|expected| {
                    crate::oauth::callback_state(&request).as_deref() != Some(expected)
                }) {
                    tracing::warn!("OAuth callback without this login's state; still waiting");
                    continue;
                }
                return Err(message);
            }
        }
    }
}

/// Paste-URL fallback for the OAuth callback, for a browser that cannot reach the bound listener:
/// read lines the prompt hands over until one carries a code and this login's `state`, refusing
/// every other through the prompt and reading again, as the account flows do, since a line the
/// person mistyped spends nothing.
async fn read_pasted_callback(
    prompt: &dyn LoginPrompt,
    expected_state: Option<&str>,
    deadline: tokio::time::Instant,
) -> std::result::Result<OAuthCallback, String> {
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(format!(
                "authorization timed out after {}s",
                OAUTH_CALLBACK_TIMEOUT.as_secs()
            ));
        }
        match tokio::time::timeout(remaining, prompt.read_pasted_line()).await {
            Err(_) => {
                return Err(format!(
                    "authorization timed out after {}s",
                    OAUTH_CALLBACK_TIMEOUT.as_secs()
                ));
            }
            Ok(Err(error)) => return Err(error),
            // Nothing will be pasted: the prompt's input closed before the user pasted anything.
            // Not fatal; the TCP branch of the `select!` keeps waiting.
            Ok(Ok(None)) => return std::future::pending().await,
            Ok(Ok(Some(line))) => match parse_pasted_callback(&line) {
                Ok(callback)
                    if expected_state
                        .is_none_or(|expected| expected == callback.state.as_str()) =>
                {
                    return Ok(callback);
                }
                Ok(_) => prompt.paste_refused("OAuth state mismatch; refusing the callback"),
                Err(reason) => prompt.paste_refused(&reason),
            },
        }
    }
}

/// Extract what a pasted callback URL carries. Accepts either the full URL or just the
/// query string, percent-decodes the values, and surfaces the `error=…` parameter (sanitized) when
/// the authorization server declines.
pub(crate) fn parse_pasted_callback(input: &str) -> std::result::Result<OAuthCallback, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("no callback URL pasted".to_string());
    }
    // Drop any URL fragment, then narrow to whatever sits after `?`.
    let before_hash = trimmed.split('#').next().unwrap_or(trimmed);
    let query = match before_hash.find('?') {
        Some(index) => &before_hash[index + 1..],
        None => before_hash,
    };
    let mut code = None;
    let mut state = None;
    let mut issuer = None;
    let mut error_param: Option<String> = None;
    // The same decoder as `parse_callback_query`: a redirect query is form-encoded, so `+` is a
    // space, which `percent_decode_str` would leave a literal `+` and the token exchange would
    // then reject.
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        // Strict UTF-8: silently mangling a security-sensitive parameter (e.g. swapping invalid
        // bytes for U+FFFD) could let a tampered `state` value match the expected one despite
        // differing bytes.
        if value.contains('\u{FFFD}') {
            return Err(format!(
                "OAuth callback parameter '{key}' is not valid UTF-8"
            ));
        }
        let decoded = value.into_owned();
        match key.as_ref() {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "iss" => issuer = Some(decoded),
            "error" => error_param = Some(decoded),
            _ => {}
        }
    }
    if let Some(error) = error_param {
        return Err(format!(
            "authorization server returned error: {}",
            crate::text::sanitize_text(&error)
        ));
    }
    let code = code.ok_or_else(|| "missing 'code' parameter in pasted URL".to_string())?;
    let state = state.ok_or_else(|| "missing 'state' parameter in pasted URL".to_string())?;
    Ok(OAuthCallback {
        code,
        state,
        issuer,
    })
}

#[derive(Debug)]
enum CallbackParseError {
    /// Request wasn't to /callback at all (e.g. /favicon.ico, /).
    NotCallbackPath,
    /// Request targets /callback but failed to parse (missing code/state, etc).
    Malformed(String),
}

fn parse_callback_query(request: &str) -> std::result::Result<OAuthCallback, CallbackParseError> {
    let first_line = request
        .lines()
        .next()
        .ok_or_else(|| CallbackParseError::Malformed("empty HTTP request".into()))?;

    let path = first_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| CallbackParseError::Malformed("malformed HTTP request line".into()))?;

    // Compare path component only, case-insensitive, anchored to /callback. `/` or `/favicon.ico`
    // fall through to `NotCallbackPath`.
    let (path_component, query_string) = match path.split_once('?') {
        Some((p, q)) => (p, q),
        None => (path, ""),
    };
    if !path_component.eq_ignore_ascii_case("/callback") {
        return Err(CallbackParseError::NotCallbackPath);
    }
    if query_string.is_empty() {
        return Err(CallbackParseError::Malformed(
            "no query parameters in callback URL".into(),
        ));
    }

    let mut code = None;
    let mut state = None;
    let mut issuer = None;
    let mut error_param: Option<String> = None;

    // `form_urlencoded`, not a hand-rolled `split('&')` + `percent_decode_str`. A redirect query is
    // `application/x-www-form-urlencoded`, where `+` means a space; `percent_decode_str` leaves it
    // as a literal `+`, so a server that form-encodes a value containing a space hands back a code
    // that is then re-encoded as `%2B` and rejected at the token exchange with nothing to point at.
    for (key, value) in url::form_urlencoded::parse(query_string.as_bytes()) {
        // Stricter than the `Cow<str>` this yields for invalid UTF-8: reject rather than accept
        // a value with U+FFFD substituted into it. Same rationale as `parse_pasted_callback`.
        if value.contains('\u{FFFD}') {
            return Err(CallbackParseError::Malformed(format!(
                "OAuth callback parameter '{key}' is not valid UTF-8"
            )));
        }
        let decoded = value.into_owned();
        match key.as_ref() {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "iss" => issuer = Some(decoded),
            "error" => error_param = Some(decoded),
            _ => {}
        }
    }

    if let Some(error) = error_param {
        // Strip Cc/Cf so a hostile authorization server can't inject ANSI escapes or RTL overrides
        // through the error message.
        return Err(CallbackParseError::Malformed(format!(
            "authorization server returned error: {}",
            crate::text::sanitize_text(&error)
        )));
    }

    let code = code.ok_or_else(|| {
        CallbackParseError::Malformed("missing 'code' parameter in callback".into())
    })?;
    let state = state.ok_or_else(|| {
        CallbackParseError::Malformed("missing 'state' parameter in callback".into())
    })?;

    Ok(OAuthCallback {
        code,
        state,
        issuer,
    })
}

/// What this adapter knows about the row it writes to, which is what decides how it writes.
///
/// rmcp's `CredentialStore` is a `load`/`save` pair with nothing in between, so a refresh arrives
/// as a bare "store this" with no way to say what it was derived from. Tracking the read is what
/// supplies that.
#[derive(Debug, Clone)]
enum LastRead {
    /// Nothing has been read, so there is nothing to conflict with: the interactive authorization
    /// flow, where replacing whatever is there is the point of it.
    Nothing,
    /// The JSON last handed to rmcp. A save compares against it and replaces only that.
    Json(String),
    /// A save found the row holding something else.
    ///
    /// Distinct from [`Self::Nothing`], and the distinction is the whole of it. Both mean "there
    /// is nothing to compare against", but they want opposite writes: `Nothing` wants an
    /// unconditional one, and this wants none at all. Collapsed (a lost swap recorded as
    /// "unread"), a process reverts to blind upserts for the rest of its run after losing a single
    /// race, quietly undoing the guarantee the swap exists for. Every token rmcp mints from here
    /// descends from a credential another process has already superseded, and that is true of the
    /// next one and the one after it, not just the one that lost.
    Superseded,
}

struct SqliteCredentialStore {
    token_store: TokenStore,
    server_name: String,
    /// The server's URL in [`canonical_endpoint`]'s form: the one endpoint a stored bundle is
    /// presented to, and the one a bundle this adapter stores is bound to.
    endpoint: String,
    /// What this adapter last saw in the row. See [`LastRead`].
    last_read: std::sync::Arc<std::sync::Mutex<LastRead>>,
}

/// Put a credential cell back to "nothing has been read", for a flow that is about to mint a
/// credential from scratch rather than derive one from what is stored.
///
/// A poisoned mutex is recovered from for the same reason [`SqliteCredentialStore::remember`]
/// recovers: the cell holds one value that every writer replaces whole.
fn forget_what_was_read(cell: &std::sync::Arc<std::sync::Mutex<LastRead>>) {
    *crate::sync::lock(cell) = LastRead::Nothing;
}

impl SqliteCredentialStore {
    /// Record what the row holds, after a read, a write that did not land, or a clear.
    ///
    /// A poisoned mutex is recovered from rather than propagated: the cell holds one value that
    /// every writer replaces whole, and refusing to track a read because an unrelated thread
    /// panicked would only make the next save unconditional.
    fn remember(&self, last: LastRead) {
        *crate::sync::lock(&self.last_read) = last;
    }

    fn last_read(&self) -> LastRead {
        crate::sync::lock(&self.last_read).clone()
    }
}

#[async_trait]
impl CredentialStore for SqliteCredentialStore {
    /// Held across rmcp's load, token request and save, so two meka processes on one server do not
    /// both spend its refresh token, which an issuer that rotates them refuses as reuse: the second
    /// loads what the first stored and refreshes from that.
    async fn acquire_refresh_guard(
        &self,
    ) -> std::result::Result<Option<rmcp::transport::auth::CredentialRefreshGuard>, AuthError> {
        Ok(
            crate::oauth::await_mcp_credential_lock(&self.token_store, &self.server_name)
                .await
                .map(rmcp::transport::auth::CredentialRefreshGuard::new),
        )
    }

    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        match self
            .token_store
            .load_mcp_credentials(&self.server_name, crate::store::McpCredentialKind::OAuth)
            .await
        {
            // Stored for another endpoint, so not this server's to present: as nothing stored,
            // which a login replaces.
            Ok(Some(stored))
                if !presentable(
                    crate::store::McpCredentialKind::OAuth,
                    &stored.endpoint,
                    &self.endpoint,
                ) =>
            {
                tracing::debug!(
                    "the stored OAuth tokens for MCP server '{server_name}' were issued for another \
                     endpoint; not presenting them",
                    server_name = self.server_name
                );
                self.remember(LastRead::Nothing);
                Ok(None)
            }
            Ok(Some(stored)) => {
                let credentials: StoredCredentials =
                    serde_json::from_str(&stored.secret).map_err(|error| {
                        AuthError::InternalError(format!(
                            "failed to deserialize stored credentials: {error}"
                        ))
                    })?;
                self.remember(LastRead::Json(stored.secret));
                Ok(Some(credentials))
            }
            // Nothing stored means nothing to conflict with, and that has to be *recorded*: a
            // `Superseded` left over from a swap this adapter lost would otherwise outlive the row
            // it was about, and refuse to write to a profile that is now empty.
            Ok(None) => {
                self.remember(LastRead::Nothing);
                Ok(None)
            }
            Err(error) => Err(AuthError::InternalError(format!(
                "failed to load credentials from the store: {error}"
            ))),
        }
    }

    async fn save(&self, credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        let json = serde_json::to_string(&credentials).map_err(|error| {
            AuthError::InternalError(format!("failed to serialize credentials: {error}"))
        })?;

        let stored = match self.last_read() {
            LastRead::Json(expected) => self
                .token_store
                .replace_mcp_credentials(&self.server_name, &expected, &json)
                .await
                .map_err(|error| {
                    AuthError::InternalError(format!(
                        "failed to save credentials to the store: {error}"
                    ))
                })?,
            LastRead::Nothing => {
                self.token_store
                    .save_mcp_credentials(
                        &self.server_name,
                        crate::store::McpCredentialKind::OAuth,
                        &json,
                        &self.endpoint,
                    )
                    .await
                    .map_err(|error| {
                        AuthError::InternalError(format!(
                            "failed to save credentials to the store: {error}"
                        ))
                    })?;
                true
            }
            // Nothing this adapter can mint is writable any more: it all descends from a
            // credential the row has already moved past. Saying so once was enough; repeating it
            // per refresh would turn one fact into a line an hour.
            LastRead::Superseded => {
                tracing::debug!(
                    "not persisting the refreshed token for MCP server '{server_name}': the stored \
                     credential has moved on",
                    server_name = self.server_name
                );
                return Ok(());
            }
        };

        if stored {
            self.remember(LastRead::Json(json));
        } else {
            // Reported rather than raised. rmcp holds the token it just minted whatever this says,
            // and failing the save would abort a connection that is otherwise working; what matters
            // is that the *stored* credential stays the newest one, which is what the next process
            // to start will load.
            tracing::info!(
                "MCP server '{server_name}' was re-authenticated elsewhere during this refresh; \
                 keeping the stored credential",
                server_name = self.server_name
            );
            self.remember(LastRead::Superseded);
        }
        Ok(())
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        // Back to `Nothing` rather than to `Superseded`: a clear is this process deciding the row
        // should hold nothing, so a fresh authorization after it is entitled to write.
        self.remember(LastRead::Nothing);
        // Only the bundle this adapter owns. rmcp calls `clear` when the authorization server's
        // issuer changes, meaning the tokens it holds are bound to an issuer that is no longer the
        // right one, which is a statement about the tokens and nothing else. A confidential
        // client's `client_secret` is not bound to an issuer, was typed by the user, and
        // after this change meka is its only holder, so clearing every kind here would send
        // them back to the provider to reissue one that was never invalid.
        self.token_store
            .clear_mcp_credentials_of_kind(
                &self.server_name,
                crate::store::McpCredentialKind::OAuth,
            )
            .await
            .map_err(|error| {
                AuthError::InternalError(format!(
                    "failed to clear credentials from the store: {error}"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A server is named by its URL in one canonical form, whatever the spelling of its scheme,
    /// host, default port or trailing slash, and a token is presented to that name alone; a
    /// bearer and a client secret anywhere on their origin, and never across one.
    #[test]
    fn a_credential_is_presented_where_it_was_given_and_nowhere_else() {
        use crate::store::McpCredentialKind::{Bearer, ClientSecret, OAuth};
        assert_eq!(
            canonical_endpoint(" HTTPS://Mcp.Example.com:443/mcp/#frag "),
            "https://mcp.example.com/mcp"
        );
        assert_eq!(
            canonical_endpoint("https://mcp.example.com/"),
            "https://mcp.example.com"
        );
        assert_eq!(
            canonical_endpoint("https://mcp.example.com/mcp?tenant=a"),
            "https://mcp.example.com/mcp?tenant=a"
        );
        let bound = canonical_endpoint("https://a.example/mcp");
        assert!(presentable(OAuth, &bound, "https://A.example/mcp/"));
        assert!(
            !presentable(OAuth, &bound, "https://a.example/other"),
            "a token's audience is the resource it was issued for"
        );
        for kind in [Bearer, ClientSecret] {
            assert!(presentable(kind, &bound, "https://a.example/other"));
            assert!(
                !presentable(kind, &bound, "https://b.example/mcp"),
                "{kind}"
            );
            assert!(!presentable(kind, &bound, "http://a.example/mcp"), "{kind}");
            assert!(
                !presentable(kind, "", "https://a.example/mcp"),
                "a credential bound to no server is presented to none"
            );
        }
        assert!(!presentable(OAuth, "", "https://a.example/mcp"));
    }

    /// A revocation reads the grant as rmcp stores it: the refresh token where there is one, the
    /// access token otherwise, the issuer the grant records or else the origin it was stored for,
    /// and the client secret a confidential client authenticates with, when it was stored for the
    /// grant's own origin and so belongs to the client the grant was minted for.
    #[tokio::test]
    async fn a_revocation_reads_the_grant_as_rmcp_stores_it() {
        use crate::store::McpCredentialKind;
        let manager = crate::store::Store::for_test().await;
        let token_store = manager.token_store();
        let request = |revocation: Revocation| revocation.0;
        assert!(
            request(stored_revocation(&token_store, "docs").await.expect("read")).is_none(),
            "no grant, nothing to revoke"
        );

        token_store
            .save_mcp_credentials(
                "docs",
                McpCredentialKind::OAuth,
                r#"{"client_id":"stub-client","token_response":{"access_token":"at","token_type":"bearer","refresh_token":"rt"},"issuer":"https://auth.example"}"#,
                "https://docs.example/mcp",
            )
            .await
            .expect("seed");
        token_store
            .save_mcp_credentials(
                "docs",
                McpCredentialKind::ClientSecret,
                "cs",
                "https://docs.example/mcp",
            )
            .await
            .expect("seed");
        let revocation = request(stored_revocation(&token_store, "docs").await.expect("read"))
            .expect("a grant to revoke");
        assert_eq!(revocation.issuer, "https://auth.example");
        assert_eq!(revocation.token, "rt");
        assert_eq!(revocation.token_type_hint, "refresh_token");
        assert_eq!(revocation.client_id, "stub-client");
        assert_eq!(revocation.client_secret.as_deref(), Some("cs"));

        token_store
            .save_mcp_credentials(
                "fallback",
                McpCredentialKind::OAuth,
                r#"{"client_id":"c","token_response":{"access_token":"at","token_type":"bearer"}}"#,
                "https://fallback.example/mcp",
            )
            .await
            .expect("seed");
        token_store
            .save_mcp_credentials(
                "fallback",
                McpCredentialKind::ClientSecret,
                "for-another-provider",
                "https://other.example/mcp",
            )
            .await
            .expect("seed");
        let revocation = request(
            stored_revocation(&token_store, "fallback")
                .await
                .expect("read"),
        )
        .expect("a grant to revoke");
        assert_eq!(
            revocation.issuer, "https://fallback.example",
            "a discovery that found no metadata took the origin's endpoints"
        );
        assert_eq!(revocation.token, "at");
        assert_eq!(revocation.token_type_hint, "access_token");
        assert!(
            revocation.client_secret.is_none(),
            "a secret stored for another endpoint is another client's"
        );
    }

    /// An issuer's metadata is looked for where the MCP authorization spec says: RFC 8414's
    /// well-known path and OpenID Connect Discovery's, inserted ahead of the issuer's own path, and
    /// for an issuer with a path, OpenID Connect Discovery's appended to it.
    #[test]
    fn an_issuer_s_metadata_is_looked_for_where_the_specifications_put_it() {
        assert_eq!(metadata_candidates("https://auth.example/"), [
            "https://auth.example/.well-known/oauth-authorization-server",
            "https://auth.example/.well-known/openid-configuration",
        ]);
        assert_eq!(metadata_candidates("https://auth.example/tenant/one"), [
            "https://auth.example/.well-known/oauth-authorization-server/tenant/one",
            "https://auth.example/.well-known/openid-configuration/tenant/one",
            "https://auth.example/tenant/one/.well-known/openid-configuration",
        ]);
        assert!(metadata_candidates("not a url").is_empty());
    }

    /// A client's identity in HTTP Basic is encoded as RFC 6749 Appendix B has it, the form
    /// encoding, which leaves `-`, `.`, `_` and `*` as they are and writes a space as `+`.
    #[test]
    fn basic_credentials_are_form_encoded() {
        assert_eq!(form_encoded("a-b_c.d*e"), "a-b_c.d*e");
        assert_eq!(form_encoded("a b:c%"), "a+b%3Ac%25");
    }

    /// Two URLs are one origin when a credential sent to either reaches the same host.
    #[test]
    fn same_origin_compares_who_receives_the_token() {
        assert!(
            same_origin("https://a.example/mcp", "https://a.example/"),
            "the issuer sits at the root and the endpoint under a path; comparing paths would \
             report a mismatch on every correctly configured server"
        );
        assert!(same_origin(
            "https://a.example",
            "https://a.example:443/mcp"
        ));
        assert!(
            !same_origin("https://a.example/mcp", "https://b.example/mcp"),
            "a different host is the whole point of reporting this"
        );
        assert!(
            !same_origin("https://a.example/mcp", "http://a.example/mcp"),
            "and so is a downgraded scheme"
        );
        assert!(
            !same_origin("https://a.example", "https://a.example:8443"),
            "a different port is a different server"
        );
        assert!(same_origin("not a url", "not a url"));
        assert!(
            !same_origin("not a url", "https://a.example"),
            "a URL is never the same origin as a string that is not one"
        );
    }

    /// A refresh holds the server's credential lock from rmcp's load to its save, so another
    /// process's refresh of the same server waits rather than spending the same refresh token; a
    /// different server's is not held up.
    #[tokio::test]
    async fn a_refresh_holds_the_server_s_credential_lock() {
        let store = crate::store::Store::for_test().await;
        let token_store = store.token_store();
        let adapter = SqliteCredentialStore {
            token_store: token_store.clone(),
            server_name: "docs".to_string(),
            endpoint: "https://docs.example/mcp".to_string(),
            last_read: std::sync::Arc::new(std::sync::Mutex::new(LastRead::Nothing)),
        };
        let guard = adapter
            .acquire_refresh_guard()
            .await
            .expect("guard")
            .expect("the lock is free");
        assert!(
            token_store
                .try_lock_mcp_credential("docs")
                .expect("ask")
                .is_none(),
            "another refresh of the server got in"
        );
        assert!(
            token_store
                .try_lock_mcp_credential("other")
                .expect("ask")
                .is_some()
        );
        drop(guard);
        assert!(
            token_store
                .try_lock_mcp_credential("docs")
                .expect("ask")
                .is_some()
        );
    }

    /// Losing one compare-and-swap must not turn every later write into an unconditional one.
    ///
    /// The state after a lost swap must not be recorded as "nothing was read", which is also the
    /// state the interactive authorization flow is in, and that state takes the blind-upsert arm.
    /// So a process that lost a single race went back to overwriting the row for the rest of its
    /// run, with every token it wrote descended from a credential another process had already
    /// superseded. The invariant [`TokenStore::replace_mcp_credentials`] states -- that the stored
    /// credential never moves backwards -- held only until the first race.
    #[tokio::test]
    async fn a_lost_swap_does_not_return_the_adapter_to_blind_writes() {
        let manager = crate::store::Store::for_test().await;
        let token_store = manager.token_store();
        // Built by deserializing rather than by a struct literal: `StoredCredentials` is
        // `#[non_exhaustive]`, so rmcp reserves the right to add a field and only its own crate may
        // name them all.
        let credentials = |client_id: &str| -> StoredCredentials {
            serde_json::from_value(serde_json::json!({ "client_id": client_id }))
                .expect("the shape rmcp stores")
        };
        let json =
            |client_id: &str| serde_json::to_string(&credentials(client_id)).expect("serialize");
        token_store
            .save_mcp_credentials(
                "docs",
                crate::store::McpCredentialKind::OAuth,
                &json("original"),
                "https://docs.example/mcp",
            )
            .await
            .expect("seed");

        let adapter = SqliteCredentialStore {
            token_store: token_store.clone(),
            server_name: "docs".to_string(),
            endpoint: "https://docs.example/mcp".to_string(),
            last_read: std::sync::Arc::new(std::sync::Mutex::new(LastRead::Nothing)),
        };
        adapter.load().await.expect("load").expect("seeded");

        // Another process re-authenticates while this adapter holds what it read.
        token_store
            .save_mcp_credentials(
                "docs",
                crate::store::McpCredentialKind::OAuth,
                &json("theirs"),
                "https://docs.example/mcp",
            )
            .await
            .expect("the other process writes");

        adapter
            .save(credentials("ours"))
            .await
            .expect("a lost swap is reported, not raised");
        assert_eq!(
            token_store
                .load_mcp_credentials("docs", crate::store::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .map(|stored| stored.secret)
                .as_deref(),
            Some(json("theirs").as_str()),
            "the swap must lose"
        );

        // The next refresh in the same run. Its token descends from the same superseded
        // credential, so it is no more writable than the one that lost.
        adapter.save(credentials("ours-again")).await.expect("save");
        assert_eq!(
            token_store
                .load_mcp_credentials("docs", crate::store::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .map(|stored| stored.secret)
                .as_deref(),
            Some(json("theirs").as_str()),
            "and every write after it must lose too, rather than reverting to a blind upsert"
        );
    }

    /// rmcp calls `clear` when the authorization server's issuer changes. That is a statement
    /// about the tokens it holds, not about the client secret they were obtained with.
    ///
    /// The secret was typed by the user and, since it stopped being a config key, meka is its only
    /// holder. Clearing every kind here would make an issuer migration at the provider silently
    /// cost the user a credential that was never invalid, discovered only when the next login
    /// fails for want of it.
    #[tokio::test]
    async fn the_adapter_clears_only_the_bundle_it_owns() {
        use crate::store::McpCredentialKind;

        let manager = crate::store::Store::for_test().await;
        let token_store = manager.token_store();
        for (kind, secret) in [
            (McpCredentialKind::ClientSecret, "cs-not-a-real-secret"),
            (McpCredentialKind::Bearer, "bearer-not-a-real-token"),
            (McpCredentialKind::OAuth, r#"{"client_id":"x"}"#),
        ] {
            token_store
                .save_mcp_credentials("docs", kind, secret, "https://docs.example/mcp")
                .await
                .expect("seed");
        }

        let adapter = SqliteCredentialStore {
            token_store: token_store.clone(),
            server_name: "docs".to_string(),
            endpoint: "https://docs.example/mcp".to_string(),
            last_read: std::sync::Arc::new(std::sync::Mutex::new(LastRead::Nothing)),
        };
        adapter.clear().await.expect("clear");

        assert!(
            token_store
                .load_mcp_credentials("docs", McpCredentialKind::OAuth)
                .await
                .expect("load")
                .is_none(),
            "the bundle bound to the old issuer must go"
        );
        assert_eq!(
            token_store
                .load_mcp_credentials("docs", McpCredentialKind::ClientSecret)
                .await
                .expect("load")
                .map(|stored| stored.secret)
                .as_deref(),
            Some("cs-not-a-real-secret"),
            "the client secret is not bound to an issuer and must survive"
        );
        assert_eq!(
            token_store
                .load_mcp_credentials("docs", McpCredentialKind::Bearer)
                .await
                .expect("load")
                .map(|stored| stored.secret)
                .as_deref(),
            Some("bearer-not-a-real-token"),
            "nor is a bearer this adapter never owned"
        );
    }

    /// The other half: a save that should land, does.
    ///
    /// Both of the adapter's arms are exercised, because a `save` that quietly writes nothing is
    /// invisible for the life of the process, which keeps using the token it holds in memory. The
    /// user finds out at the next start, when the credential loaded is the one from before the
    /// refresh, expired.
    #[tokio::test]
    async fn a_save_that_should_land_reaches_the_store() {
        let manager = crate::store::Store::for_test().await;
        let token_store = manager.token_store();
        let credentials = |client_id: &str| -> StoredCredentials {
            serde_json::from_value(serde_json::json!({ "client_id": client_id }))
                .expect("the shape rmcp stores")
        };
        let json =
            |client_id: &str| serde_json::to_string(&credentials(client_id)).expect("serialize");
        let stored = async || {
            token_store
                .load_mcp_credentials("docs", crate::store::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .map(|stored| stored.secret)
        };

        let adapter = SqliteCredentialStore {
            token_store: token_store.clone(),
            server_name: "docs".to_string(),
            endpoint: "https://docs.example/mcp".to_string(),
            last_read: std::sync::Arc::new(std::sync::Mutex::new(LastRead::Nothing)),
        };

        // The blind-upsert arm: the interactive flow depositing a first bundle.
        adapter.save(credentials("first")).await.expect("save");
        assert_eq!(
            stored().await.as_deref(),
            Some(json("first").as_str()),
            "the first bundle must reach the store"
        );

        // The compare-and-swap arm: a refresh of what this adapter last read.
        adapter.load().await.expect("load").expect("just written");
        adapter.save(credentials("refreshed")).await.expect("save");
        assert_eq!(
            stored().await.as_deref(),
            Some(json("refreshed").as_str()),
            "an uncontended refresh must land"
        );
    }

    /// A client secret is asked whether it may be presented only by a flow that presents one: a
    /// `client_credentials_jwt` server holding a secret stored for another endpoint goes on to
    /// its own signing, and refuses over that, never over the secret it does not send.
    #[tokio::test]
    async fn a_jwt_client_is_not_held_to_a_secret_it_never_sends() {
        let store = crate::store::Store::for_test().await.token_store();
        store
            .save_mcp_credentials(
                "jwt",
                crate::store::McpCredentialKind::ClientSecret,
                "cs",
                "https://other.example/mcp",
            )
            .await
            .expect("seed");
        let auth = McpAuthConfig::ClientCredentialsJwt {
            client_id: "c".to_string(),
            signing_key_path: "/nonexistent/meka-test-signing-key.pem".to_string(),
            signing_algorithm: None,
            scopes: None,
        };
        let Err(error) = connect_http_with_oauth(
            "jwt",
            "https://a.example/mcp",
            &auth,
            rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(
                "https://a.example/mcp",
            ),
            Some(&store),
            None,
            MekaClientHandler::new("jwt".to_string(), crate::mcp::McpClientContext::new()),
        )
        .await
        else {
            panic!("a key that is not there cannot sign");
        };
        assert!(
            error.to_string().contains("signing key"),
            "refused over its own key, not the secret: {error}"
        );
    }

    /// The issuer an authorization response names (RFC 9207) is read off the callback whichever
    /// way it arrives, for the exchange to check against the issuer it discovered, and its
    /// absence is said as absence.
    #[test]
    fn a_callback_carries_the_issuer_it_names() {
        let request = "GET /callback?code=c&state=s&iss=https%3A%2F%2Fauth.example HTTP/1.1\r\n";
        let callback = parse_callback_query(request).expect("should parse");
        assert_eq!(callback.issuer.as_deref(), Some("https://auth.example"));
        let pasted = parse_pasted_callback(
            "http://127.0.0.1:1/callback?code=c&state=s&iss=https%3A%2F%2Fauth.example",
        )
        .expect("should parse");
        assert_eq!(pasted.issuer.as_deref(), Some("https://auth.example"));
        let without = parse_pasted_callback("code=c&state=s").expect("should parse");
        assert_eq!(without.issuer, None);
    }

    #[test]
    fn parse_callback_query_valid() {
        let request = "GET /callback?code=abc123&state=xyz789 HTTP/1.1\r\nHost: localhost\r\n";
        let OAuthCallback { code, state, .. } =
            parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz789");
    }

    #[test]
    fn parse_callback_query_reversed_order() {
        let request = "GET /callback?state=xyz789&code=abc123 HTTP/1.1\r\n";
        let OAuthCallback { code, state, .. } =
            parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz789");
    }

    #[test]
    fn parse_callback_query_missing_code() {
        let request = "GET /callback?state=xyz789 HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => assert!(m.contains("code")),
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn parse_callback_query_missing_state() {
        let request = "GET /callback?code=abc123 HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => assert!(m.contains("state")),
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn parse_callback_query_no_query_string() {
        let request = "GET /callback HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(_) => {}
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn parse_callback_query_ignores_favicon() {
        let request = "GET /favicon.ico HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        assert!(matches!(err, CallbackParseError::NotCallbackPath));
    }

    #[test]
    fn parse_callback_query_ignores_root() {
        let request = "GET / HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        assert!(matches!(err, CallbackParseError::NotCallbackPath));
    }

    #[test]
    fn parse_callback_query_url_decodes_state() {
        let request = "GET /callback?code=abc&state=xyz%3D%3D HTTP/1.1\r\n";
        let OAuthCallback { code, state, .. } =
            parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc");
        assert_eq!(state, "xyz==");
    }

    #[test]
    fn parse_callback_query_rejects_invalid_utf8_state() {
        // %FF and %FE are invalid as UTF-8; lossy decoding would silently turn them into U+FFFD,
        // which can let a tampered `state` parameter match a stored one despite differing bytes.
        // Strict mode rejects.
        let request = "GET /callback?code=abc&state=%FF%FE HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("non-utf8 state must be rejected");
        match err {
            CallbackParseError::Malformed(m) => {
                assert!(
                    m.contains("not valid UTF-8"),
                    "unexpected error message: {m}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn parse_callback_query_surfaces_oauth_error() {
        let request = "GET /callback?error=access_denied HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => {
                assert!(m.contains("access_denied"));
            }
            other => panic!("expected Malformed with error, got {other:?}"),
        }
    }

    #[test]
    fn extract_revocation_endpoint_returns_string() {
        let body = br#"{"revocation_endpoint":"https://auth.example.com/revoke","other":"x"}"#;
        assert_eq!(
            extract_revocation_endpoint(body, 1024),
            Some("https://auth.example.com/revoke".to_string())
        );
    }

    #[test]
    fn extract_revocation_endpoint_rejects_oversize_body() {
        let body = br#"{"revocation_endpoint":"https://auth.example.com/revoke"}"#;
        assert!(extract_revocation_endpoint(body, 8).is_none());
    }

    #[test]
    fn extract_revocation_endpoint_returns_none_without_field() {
        let body = br#"{"issuer":"https://auth.example.com"}"#;
        assert!(extract_revocation_endpoint(body, 1024).is_none());
    }

    #[test]
    fn extract_revocation_endpoint_rejects_malformed_json() {
        assert!(extract_revocation_endpoint(b"not json", 1024).is_none());
    }

    #[test]
    fn validate_revocation_endpoint_accepts_same_origin() {
        validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "https://auth.example.com/revoke",
        )
        .expect("same host should pass");
    }

    #[test]
    fn validate_revocation_endpoint_accepts_same_origin_with_path() {
        validate_revocation_endpoint_origin(
            "https://auth.example.com/realms/r",
            "https://auth.example.com/realms/r/protocol/openid-connect/revoke",
        )
        .expect("subpath on same host should pass");
    }

    #[test]
    fn validate_revocation_endpoint_rejects_different_host() {
        let err = validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "https://attacker.com/steal",
        )
        .expect_err("cross-host should be rejected");
        assert!(err.contains("different origin"));
    }

    #[test]
    fn validate_revocation_endpoint_rejects_http_to_https_downgrade() {
        let err = validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "http://auth.example.com/revoke",
        )
        .expect_err("scheme change should be rejected");
        assert!(err.contains("different origin"));
    }

    #[test]
    fn validate_revocation_endpoint_rejects_different_port() {
        let err = validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "https://auth.example.com:8443/revoke",
        )
        .expect_err("port change should be rejected");
        assert!(err.contains("different origin"));
    }

    #[cfg(unix)]
    #[test]
    fn require_private_key_permissions_rejects_world_readable() {
        use std::{io::Write, os::unix::fs::PermissionsExt};
        let dir = tempfile::tempdir().expect("tempdir");
        let key_path = dir.path().join("key.pem");
        let mut f = std::fs::File::create(&key_path).expect("create key");
        f.write_all(b"---BEGIN---").expect("write");
        drop(f);
        // 0644: readable by other users.
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o644))
            .expect("chmod 0644");
        let file = std::fs::File::open(&key_path).expect("open key");
        let err = require_private_key_permissions_on_fd(
            "srv",
            key_path.to_str().expect("utf-8 path"),
            &file,
        )
        .expect_err("loose permissions must be rejected");
        let message = format!("{err}");
        assert!(
            message.contains("0600") || message.contains("permissions"),
            "unexpected error: {message}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn require_private_key_permissions_accepts_0600() {
        use std::{io::Write, os::unix::fs::PermissionsExt};
        let dir = tempfile::tempdir().expect("tempdir");
        let key_path = dir.path().join("key.pem");
        let mut f = std::fs::File::create(&key_path).expect("create key");
        f.write_all(b"---BEGIN---").expect("write");
        drop(f);
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600))
            .expect("chmod 0600");
        let file = std::fs::File::open(&key_path).expect("open key");
        require_private_key_permissions_on_fd("srv", key_path.to_str().expect("utf-8 path"), &file)
            .expect("0600 must pass");
    }

    #[cfg(unix)]
    #[test]
    fn require_private_key_permissions_reports_missing_file() {
        // Missing file errors at `File::open` rather than inside the permissions check; assert the
        // open-side error message matches the user-facing wording in
        // `authenticate_client_credentials_jwt`.
        let err = std::fs::File::open("/nonexistent/key.pem").expect_err("missing file must error");
        let message = format!("{err}");
        assert!(message.contains("No such file") || message.contains("not found"));
    }

    #[test]
    fn validate_revocation_endpoint_rejects_malformed_urls() {
        assert!(
            validate_revocation_endpoint_origin("not-a-url", "https://auth.example.com/revoke")
                .is_err()
        );
        assert!(
            validate_revocation_endpoint_origin("https://auth.example.com", "also-not-a-url")
                .is_err()
        );
    }

    #[test]
    fn parse_callback_query_sanitizes_oauth_error() {
        // A malicious authorization server includes an ANSI escape and an RTL override in its
        // `error` parameter. The resulting error message must not carry those codepoints to the
        // terminal.
        let request = "GET /callback?error=bad%1B%5B2Jstuff%E2%80%AErtl HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => {
                assert!(!m.contains('\u{001B}'), "ANSI escape leaked: {m:?}");
                assert!(!m.contains('\u{202E}'), "RTL override leaked: {m:?}");
                assert!(m.contains("bad"));
                assert!(m.contains("stuff"));
                assert!(m.contains("rtl"));
            }
            other => panic!("expected Malformed with error, got {other:?}"),
        }
    }

    #[test]
    fn classify_probe_open_on_2xx() {
        assert_eq!(classify_probe_response(200, None), McpAuthProbe::Open);
        assert_eq!(classify_probe_response(204, None), McpAuthProbe::Open);
        assert_eq!(
            classify_probe_response(200, Some("Bearer realm=\"x\"")),
            McpAuthProbe::Open,
            "2xx always wins over any WWW-Authenticate header"
        );
    }

    #[test]
    fn classify_probe_auth_required_with_resource_metadata() {
        // What Notion-style MCP servers actually emit.
        let header = r#"Bearer realm="mcp", resource_metadata="https://mcp.notion.com/.well-known/oauth-protected-resource""#;
        assert_eq!(
            classify_probe_response(401, Some(header)),
            McpAuthProbe::AuthRequired {
                resource_metadata: Some(
                    "https://mcp.notion.com/.well-known/oauth-protected-resource".to_string()
                ),
            }
        );
    }

    #[test]
    fn classify_probe_auth_required_without_resource_metadata() {
        assert_eq!(
            classify_probe_response(401, Some("Bearer realm=\"mcp\"")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_auth_required_bare_bearer() {
        // Some servers emit just `Bearer` with no parameters.
        assert_eq!(
            classify_probe_response(401, Some("Bearer")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_is_case_insensitive_on_scheme() {
        assert_eq!(
            classify_probe_response(401, Some("bearer realm=\"x\"")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_401_without_bearer_is_unexpected() {
        // A 401 with e.g. Basic / Digest auth is not MCP-spec compliant. Surface it as Unexpected
        // rather than pretending it's OAuth.
        assert_eq!(
            classify_probe_response(401, Some("Basic realm=\"x\"")),
            McpAuthProbe::Unexpected { status: 401 }
        );
        assert_eq!(
            classify_probe_response(401, None),
            McpAuthProbe::Unexpected { status: 401 }
        );
    }

    #[test]
    fn classify_probe_403_with_bearer_is_auth_required() {
        // Some implementations return 403 for missing auth.
        assert_eq!(
            classify_probe_response(403, Some("Bearer realm=\"mcp\"")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_other_statuses_are_unexpected() {
        assert_eq!(
            classify_probe_response(405, None),
            McpAuthProbe::Unexpected { status: 405 }
        );
        assert_eq!(
            classify_probe_response(500, None),
            McpAuthProbe::Unexpected { status: 500 }
        );
    }

    #[test]
    fn extract_bearer_param_handles_quoting_and_spacing() {
        let header = r#"Bearer realm="x",resource_metadata="https://y/z""#;
        assert_eq!(
            extract_bearer_param(header, "resource_metadata"),
            Some("https://y/z".to_string())
        );
        let unquoted = r#"Bearer realm=x, resource_metadata=https://y/z"#;
        assert_eq!(
            extract_bearer_param(unquoted, "resource_metadata"),
            Some("https://y/z".to_string())
        );
    }

    #[test]
    fn extract_bearer_param_returns_none_when_missing() {
        assert_eq!(
            extract_bearer_param("Bearer realm=\"x\"", "resource_metadata"),
            None
        );
    }

    /// A form-encoded space arrives as a space, not a literal `+`.
    ///
    /// A redirect query is `application/x-www-form-urlencoded`, where `+` is a space.
    /// `percent_decode_str` does not know that, so a hand-rolled split returned the `+` verbatim
    /// and the token exchange re-encoded it as `%2B` -- a failure the server reports opaquely and
    /// nothing local explains. `form_urlencoded::parse` is the decoder that matches the encoder.
    #[test]
    fn callback_query_decodes_form_encoding_not_just_percent_escapes() {
        let OAuthCallback { code, state, .. } = parse_callback_query(
            "GET /callback?code=a+b&state=c%20d HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .expect("a well-formed callback");
        assert_eq!(code, "a b", "`+` is a space in a form-encoded query");
        assert_eq!(state, "c d", "and `%20` still is too");
    }

    /// Invalid UTF-8 in a callback parameter is refused rather than silently substituted.
    ///
    /// `form_urlencoded::parse` yields a lossy `Cow`, so without this check a `state` carrying
    /// invalid bytes would come back with U+FFFD in place of them and be compared against the
    /// stored value as though it were what the server sent.
    #[test]
    fn callback_query_refuses_invalid_utf8_rather_than_substituting() {
        let result = parse_callback_query(
            "GET /callback?code=ok&state=%FF%FE HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert!(
            matches!(result, Err(CallbackParseError::Malformed(_))),
            "expected a malformed-parameter refusal, got {result:?}"
        );
    }

    /// A host has no one at a terminal, so a server with no stored credential is refused with the
    /// command that does, rather than the connector printing a URL and reading stdin.
    #[tokio::test]
    async fn without_a_login_prompt_a_missing_credential_names_the_login_command() {
        let outcome = authenticate_oauth_authorization_code(
            OAuthLogin {
                server_name: "acme",
                url: "http://127.0.0.1:9/",
                client_id: None,
                client_secret: None,
                scopes: None,
                redirect_port: None,
            },
            None,
            None,
        )
        .await;
        let Err(error) = outcome else {
            panic!("nothing is stored and nobody can be asked, so this must be refused");
        };
        let message = error.to_string();
        assert!(
            message.contains("meka mcp login acme"),
            "the refusal must name the remedy: {message}"
        );
    }

    /// A scripted prompt: the lines it hands over in order, and every refusal it was told.
    struct ScriptedPrompt {
        lines: std::sync::Mutex<std::collections::VecDeque<String>>,
        refusals: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl LoginPrompt for ScriptedPrompt {
        fn authorize_at(&self, _url: &str) {}

        fn awaiting(&self, _completion: Completion) {}

        fn accepts_paste(&self) -> bool {
            true
        }

        async fn read_pasted_line(&self) -> std::result::Result<Option<String>, String> {
            Ok(self.lines.lock().unwrap().pop_front())
        }

        fn paste_refused(&self, reason: &str) {
            self.refusals.lock().unwrap().push(reason.to_string());
        }
    }

    /// A paste the flow cannot read, and one carrying another login's state, are each refused
    /// through the prompt and read past, so the line that carries this login's code is the one
    /// exchanged; a mistyped line spends nothing.
    #[tokio::test]
    async fn a_pasted_callback_is_refused_and_read_again_until_it_is_this_login_s() {
        let prompt = ScriptedPrompt {
            lines: std::sync::Mutex::new(
                [
                    "nonsense",
                    "http://127.0.0.1:1/callback?code=a&state=theirs",
                    "http://127.0.0.1:1/callback?code=b&state=ours",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            refusals: std::sync::Mutex::new(Vec::new()),
        };
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let OAuthCallback { code, state, .. } =
            read_pasted_callback(&prompt, Some("ours"), deadline)
                .await
                .expect("the third line is this login's");
        assert_eq!((code.as_str(), state.as_str()), ("b", "ours"));
        assert_eq!(*prompt.refusals.lock().unwrap(), [
            "missing 'code' parameter in pasted URL",
            "OAuth state mismatch; refusing the callback",
        ]);
    }

    /// The `state` the authorization URL carries is the one a callback must bring back.
    #[test]
    fn the_authorization_url_names_the_state_a_callback_must_carry() {
        assert_eq!(
            state_in_authorization_url(
                "http://127.0.0.1:1/authorize?response_type=code&state=s-1&redirect_uri=x"
            )
            .as_deref(),
            Some("s-1")
        );
        assert_eq!(state_in_authorization_url("not a url"), None);
    }

    #[test]
    fn parse_pasted_callback_accepts_full_url() {
        // Exact shape returned by Notion after the user authorizes.
        let input = "http://127.0.0.1:46437/callback?code=1d5d872b-594c-8153-a5e0-0002d8f4be0f%3AGEE3YdaPJhHZpMMa%3AjZ2YV0BC0TYheYBtoSB16LmRDTgIZ6zM&state=82Nw67m4su5AeMSfCFcXAw";
        let OAuthCallback { code, state, .. } = parse_pasted_callback(input).expect("should parse");
        // Percent-encoded colons in the code must be decoded.
        assert_eq!(
            code,
            "1d5d872b-594c-8153-a5e0-0002d8f4be0f:GEE3YdaPJhHZpMMa:jZ2YV0BC0TYheYBtoSB16LmRDTgIZ6zM"
        );
        assert_eq!(state, "82Nw67m4su5AeMSfCFcXAw");
    }

    /// A redirect query is form-encoded, so `+` is a space. Decoded as a literal `+`, a code that
    /// carried one failed the token exchange on the paste path alone; the listener path had the
    /// right decoder all along.
    #[test]
    fn parse_pasted_callback_decodes_form_encoding_like_the_listener() {
        let OAuthCallback { code, state, .. } =
            parse_pasted_callback("http://127.0.0.1:1/callback?code=a+b%2Bc&state=s+t")
                .expect("should parse");
        assert_eq!(code, "a b+c");
        assert_eq!(state, "s t");
    }

    #[test]
    fn parse_pasted_callback_accepts_query_only() {
        let OAuthCallback { code, state, .. } =
            parse_pasted_callback("code=abc&state=xyz").expect("should parse");
        assert_eq!(code, "abc");
        assert_eq!(state, "xyz");
    }

    #[test]
    fn parse_pasted_callback_trims_whitespace_and_fragment() {
        let OAuthCallback { code, state, .. } =
            parse_pasted_callback("   http://127.0.0.1:1/callback?code=a&state=b#fragment  \n")
                .expect("should parse");
        assert_eq!(code, "a");
        assert_eq!(state, "b");
    }

    #[test]
    fn parse_pasted_callback_rejects_empty_input() {
        let err = parse_pasted_callback("   \n").expect_err("empty input should fail");
        assert!(err.contains("no callback URL"));
    }

    #[test]
    fn parse_pasted_callback_surfaces_server_error_sanitized() {
        let input = "http://127.0.0.1:1/callback?error=bad%1B%5B2Jstuff%E2%80%AErtl&state=z";
        let err = parse_pasted_callback(input).expect_err("should surface error");
        assert!(!err.contains('\u{001B}'), "ANSI leaked: {err}");
        assert!(!err.contains('\u{202E}'), "RTL leaked: {err}");
        assert!(err.contains("bad"));
    }

    #[test]
    fn parse_pasted_callback_missing_code() {
        let err = parse_pasted_callback("state=xyz").expect_err("should fail");
        assert!(err.contains("missing 'code'"));
    }

    #[test]
    fn parse_pasted_callback_missing_state() {
        let err = parse_pasted_callback("code=abc").expect_err("should fail");
        assert!(err.contains("missing 'state'"));
    }

    #[test]
    fn parse_jwt_signing_algorithm_defaults() {
        let alg = parse_jwt_signing_algorithm("test", None).expect("should parse");
        assert!(matches!(alg, rmcp::transport::JwtSigningAlgorithm::RS256));
    }

    #[test]
    fn parse_jwt_signing_algorithm_all_values() {
        for name in &["RS256", "RS384", "RS512", "ES256", "ES384"] {
            assert!(parse_jwt_signing_algorithm("test", Some(name)).is_ok());
        }
    }

    #[test]
    fn parse_jwt_signing_algorithm_invalid() {
        assert!(parse_jwt_signing_algorithm("test", Some("HS256")).is_err());
    }
}
