//! Account and MCP credentials: the `account_credentials` and `mcp_credentials` tables, behind
//! [`TokenStore`].

use serde::{Deserialize, Serialize};

use super::*;

/// An account's credential as the row holds it, with the version the row was at.
///
/// The version is the row's `updated_at`, the same value
/// [`TokenStore::account_credential_version`] answers, and it is what a refresh presents to
/// [`TokenStore::replace_account_credential`] as "the row I read". It stands in for the
/// credential's bytes so the swap does not depend on how `serde` happens to spell a value this
/// build reads back out.
#[derive(Debug, Clone)]
pub(crate) struct StoredCredential {
    pub(crate) credential: AuthCredential,
    pub(crate) version: String,
}

/// What became of a compare-and-swap on an account's stored credential.
///
/// No equality derive: comparing credentials is not something callers should be doing, and
/// [`AuthCredential`]'s own `Debug` redacts, so this stays printable without leaking a token.
#[derive(Debug, Clone)]
pub(crate) enum CredentialWrite {
    /// The row was still at the version this write was derived from, and now holds the new value.
    Stored,
    /// Something else wrote first. This is what the row holds now.
    ///
    /// Newer in *write order*, which is the only thing this type knows and less than it sounds: it
    /// is neither necessarily unexpired nor necessarily the same kind of credential. Whether it is
    /// worth switching to is the caller's question (see `crate::oauth::is_worth_adopting`), and
    /// retrying is never the answer, because the value this write was derived from is gone.
    Superseded(Box<StoredCredential>),
    /// The account has no stored credential at all: removed while this write was in flight.
    /// Re-creating it would resurrect an account the user just disconnected.
    Gone,
}
/// Which kind of secret a `mcp_credentials` row holds, and therefore how its `secret` column is to
/// be read.
///
/// Stored rather than inferred from the value's shape. Sniffing would make every reader depend on
/// what rmcp's serialization happens to look like this week, which is a fact about someone else's
/// system and would go stale with nothing to notice.
///
/// Three variants rather than "secret or not", because a server can hold two at once and they are
/// not interchangeable: a confidential `auth = "oauth"` client keeps its long-lived
/// [`Self::ClientSecret`] *and* the [`Self::OAuth`] bundle obtained with it, and a token refresh
/// must rewrite only the second. That is what `(server_name, kind)` keys the table on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum McpCredentialKind {
    /// A static token sent verbatim as `Authorization: Bearer <token>`. It *is* the credential:
    /// nothing is exchanged for it and meka never rewrites it.
    Bearer,
    /// The client half of an OAuth client's identity, presented to an authorization server to
    /// obtain an access token. Long-lived, and meka never rewrites it either.
    ClientSecret,
    /// An rmcp OAuth bundle, obtained by the authorization-code flow and refreshed in place by the
    /// adapter. The only kind meka itself rewrites.
    OAuth,
}
impl McpCredentialKind {
    /// The stored discriminator. Values are part of the schema, so they are written out here rather
    /// than derived from the variant names, which are free to be renamed.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Bearer => "bearer",
            Self::ClientSecret => "client_secret",
            Self::OAuth => "oauth",
        }
    }

    /// What to call this kind when telling the user about it. Deliberately not [`Self::as_str`]:
    /// that one is a schema value and must not drift to suit a sentence.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Bearer => "bearer token",
            Self::ClientSecret => "client secret",
            Self::OAuth => "OAuth tokens",
        }
    }
}
/// The credential tables of the store, handed out by [`Store::token_store`].
#[derive(Clone)]
pub(crate) struct TokenStore {
    pub(super) connection: Arc<Connection>,
    /// Where the per-account credential locks live, shared with the session locks because they are
    /// the same kind of thing: a claim one process makes on a name, released by the OS when it
    /// exits. See [`TokenStore::try_lock_account_credential`].
    pub(super) lock_dir: PathBuf,
    /// Keeps an in-memory store's lock directory alive for as long as any handle on that store is,
    /// exactly as [`Store`] does. Without it a `TokenStore` outliving its store would
    /// be locking files under a directory that had already been removed.
    pub(super) _ephemeral_lock_dir: Option<Arc<EphemeralLockDir>>,
}
impl TokenStore {
    /// Try to take the lock that serializes one account's credential rotation across processes.
    /// `None` means another meka is already refreshing that account.
    ///
    /// Separate from the session locks beside it because the thing being protected is different: a
    /// session lock says who owns a conversation, this says who is allowed to spend a refresh
    /// token. Two processes refreshing the same account both present the token the other is about
    /// to invalidate, and against an issuer with a reuse window both succeed, leaving the
    /// database holding the *older* of the two, superseded, with the next launch getting
    /// `invalid_grant` and nothing naming why.
    ///
    /// The account name is hashed rather than used directly: it is a TOML table key with no charset
    /// rule behind it, so `[accounts."a/b"]` would otherwise name a path. A readable prefix is
    /// kept so an operator listing the directory can see what these files are.
    pub(crate) fn try_lock_account_credential(&self, account: &str) -> Result<Option<FileLock>> {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(account.as_bytes());
        let readable: String = account
            .chars()
            .filter(|character| character.is_ascii_alphanumeric() || *character == '-')
            .take(24)
            .collect();
        let path = self.lock_dir.join(format!(
            "{}{}-{:x}.lock",
            ACCOUNT_LOCK_PREFIX,
            readable,
            // The digest's leading four bytes: injective on `u32`, so two accounts never share a
            // file, and short enough that the name stays readable. Rendered by `{:x}`, which drops
            // leading zeros, so this is up to eight hex digits rather than always eight.
            u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]])
        ));
        try_lock_file(&path)
    }

    /// Load the stored credential (API key or OAuth bundle) for an account, keyed by the
    /// user-chosen account name. The credential is stored as a serialized [`AuthCredential`].
    pub(crate) async fn load_account_credential(
        &self,
        account: &str,
    ) -> Result<Option<AuthCredential>> {
        Ok(self
            .load_account_credential_versioned(account)
            .await?
            .map(|stored| stored.credential))
    }

    /// [`Self::load_account_credential`] with the row's version, for a reader that will later
    /// offer a replacement derived from what it read.
    pub(crate) async fn load_account_credential_versioned(
        &self,
        account: &str,
    ) -> Result<Option<StoredCredential>> {
        let account = account.to_string();
        let row: Option<(String, String)> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT credentials_json, updated_at FROM account_credentials WHERE account = ?1",
                    rusqlite::params![account],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                );
                match result {
                    Ok(row) => Ok(Some(row)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load account credential: {error}"))
            })?;

        match row {
            Some((json, version)) => {
                let credential = serde_json::from_str(&json).map_err(|error| {
                    MekaError::Database(format!(
                        "failed to parse stored account credential: {error}"
                    ))
                })?;
                Ok(Some(StoredCredential {
                    credential,
                    version,
                }))
            }
            None => Ok(None),
        }
    }

    /// A value that changes whenever an account's stored credential does, without reading the
    /// credential.
    ///
    /// For a holder that built something *from* a credential and needs to know whether the thing it
    /// built is still the right one. [`crate::provider::ProviderRegistry`] is the caller: it keeps
    /// a built provider for reuse, and the writer that supersedes the credential is usually
    /// another process (`meka account login` while `meka serve` runs), so a push cannot reach
    /// it. Comparing this against the value the provider was built from is the whole check.
    ///
    /// `None` for an account with no stored credential, which is a distinct answer from any
    /// version: a memo built while one existed must not survive `meka account remove`.
    ///
    /// The row's `updated_at` rather than a digest of the credential, because the point is to avoid
    /// pulling a secret into memory for a comparison. Every writer here stamps it
    /// ([`Self::save_account_credential`], [`Self::replace_account_credential`]), so a write the
    /// value of which happens to be unchanged still moves it, and the cost is a rebuild rather than
    /// a wrong answer.
    pub(crate) async fn account_credential_version(&self, account: &str) -> Result<Option<String>> {
        let account = account.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT updated_at FROM account_credentials WHERE account = ?1",
                    rusqlite::params![account],
                    |row| row.get::<_, String>(0),
                );
                match result {
                    Ok(version) => Ok(Some(version)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!(
                    "failed to read account credential version: {error}"
                ))
            })
    }

    /// Replace an account's credential only if the stored one is still what the caller derived its
    /// new value from.
    ///
    /// This is the door a *refresh* goes through, and the reason it is a compare-and-swap is that a
    /// refresh is not an assignment: it is a value computed from the old one, over a network round
    /// trip long enough for something else to have written. Two of those somethings are real. Two
    /// meka processes refreshing at once both present the same refresh token, and against an issuer
    /// with a reuse window *both* succeed, so a blind upsert leaves the database holding whichever
    /// finished last, which is the token the issuer has already superseded, and the next launch
    /// gets `invalid_grant` with nothing naming why. And a `meka account login` completing during
    /// a slow refresh is simply overwritten, silently, by a credential minted before it.
    ///
    /// Returns [`CredentialWrite::Superseded`] with what the row holds now, so the caller can
    /// decide whether to switch to it. Newer in write order is not the same as usable; see
    /// `crate::oauth::is_worth_adopting`.
    ///
    /// Keyed on the row's version, its `updated_at`, which every writer here stamps, rather than
    /// on the stored JSON. Comparing the bytes would re-serialize the value this build read back
    /// out, so the first field added to [`AuthCredential`] would make every row written before it
    /// unswappable: `serde` fills the new field on the way in and writes it on the way out, and no
    /// refresh would land again until the user signed in afresh.
    pub(crate) async fn replace_account_credential(
        &self,
        account: &str,
        expected_version: &str,
        credential: &AuthCredential,
    ) -> Result<CredentialWrite> {
        let json = serde_json::to_string(credential).map_err(|error| {
            MekaError::Database(format!("failed to serialize account credential: {error}"))
        })?;
        let now = chrono::Utc::now().to_rfc3339();
        let account_for_db = account.to_string();
        let expected_version = expected_version.to_string();
        let changed = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE account_credentials SET credentials_json = ?3, updated_at = ?4 \
                     WHERE account = ?1 AND updated_at = ?2",
                    rusqlite::params![account_for_db, expected_version, json, now],
                )
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save account credential: {error}"))
            })?;
        if changed == 1 {
            return Ok(CredentialWrite::Stored);
        }
        // Zero rows means the row moved, or that there is no row at all: an account whose
        // credential was deleted mid-refresh. Both are "somebody else decided what this account
        // holds", and in neither case may a token minted from a superseded one be written back.
        match self.load_account_credential_versioned(account).await? {
            Some(current) => Ok(CredentialWrite::Superseded(Box::new(current))),
            None => Ok(CredentialWrite::Gone),
        }
    }

    /// Persist (or replace) the credential for an account, keyed by account name.
    ///
    /// The unconditional door, for a caller whose credential is not derived from a stored one: a
    /// fresh `meka account login` or `account add`, where overwriting whatever is there is the
    /// point. A refresh wants [`Self::replace_account_credential`].
    pub(crate) async fn save_account_credential(
        &self,
        account: &str,
        credential: &AuthCredential,
    ) -> Result<()> {
        let account = account.to_string();
        let json = serde_json::to_string(credential).map_err(|error| {
            MekaError::Database(format!("failed to serialize account credential: {error}"))
        })?;
        let now = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO account_credentials (account, credentials_json, updated_at) \
                     VALUES (?1, ?2, ?3) \
                     ON CONFLICT(account) DO UPDATE SET \
                         credentials_json = excluded.credentials_json, \
                         updated_at = excluded.updated_at",
                    rusqlite::params![account, json, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save account credential: {error}"))
            })
    }

    /// Remove the stored credential for an account (used by `account remove`).
    pub(crate) async fn delete_account_credential(&self, account: &str) -> Result<()> {
        let account = account.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM account_credentials WHERE account = ?1",
                    rusqlite::params![account],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to delete account credential: {error}"))
            })
    }

    /// Every account name that has a stored credential, sorted.
    ///
    /// Credentials are keyed by account name and nothing enforces that the name still exists in
    /// `config.toml`: deleting an `[accounts.<name>]` block by hand leaves its API key or OAuth
    /// refresh token here indefinitely. Without this query the leftover cannot be named, so it
    /// cannot be reported or removed; `meka account list` diffs the result against the configured
    /// accounts.
    pub(crate) async fn list_credential_accounts(&self) -> Result<Vec<String>> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                let mut statement = connection
                    .prepare("SELECT account FROM account_credentials ORDER BY account")?;
                let accounts = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(accounts)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to list account credentials: {error}"))
            })
    }

    /// One server's stored secret of `kind`, or `None`.
    ///
    /// Matching on `kind` as well as the name is what makes "this server has no bearer" and "this
    /// server has an OAuth bundle" different answers, rather than handing an OAuth blob to a caller
    /// that would read it as a token.
    pub(crate) async fn load_mcp_credentials(
        &self,
        server_name: &str,
        kind: McpCredentialKind,
    ) -> Result<Option<String>> {
        let server_name = server_name.to_string();
        let kind = kind.as_str();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT secret FROM mcp_credentials WHERE server_name = ?1 AND kind = ?2",
                    rusqlite::params![server_name, kind],
                    |row| row.get::<_, String>(0),
                );

                match result {
                    Ok(secret) => Ok(Some(secret)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load MCP credentials: {error}"))
            })
    }

    /// Replace an MCP server's credentials only if the stored ones are still the ones the caller
    /// last read. Returns whether the write landed.
    ///
    /// The same compare-and-swap as [`Self::replace_account_credential`], for the same reason: two
    /// meka processes refreshing one server's OAuth token both write, and a blind upsert leaves the
    /// database holding whichever finished last rather than the one the issuer considers current.
    /// It arbitrates less here because rmcp owns the refresh and hands this adapter only a
    /// `load`/`save` pair, so the losing process keeps using its own token for the rest of its run.
    /// What it does guarantee is that the *stored* credential is never moved backwards, which is
    /// what the next process to start will load.
    ///
    /// Scoped to the `oauth` row because that is the only kind anything refreshes. A bearer and a
    /// client secret are written once by the user and read thereafter, so a compare-and-swap over
    /// them would arbitrate a race that cannot happen.
    pub(crate) async fn replace_mcp_credentials(
        &self,
        server_name: &str,
        expected_json: &str,
        json: &str,
    ) -> Result<bool> {
        let server_name = server_name.to_string();
        let expected_json = expected_json.to_string();
        let json = json.to_string();
        let now = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE mcp_credentials SET secret = ?3, updated_at = ?4 \
                     WHERE server_name = ?1 AND secret = ?2 AND kind = 'oauth'",
                    rusqlite::params![server_name, expected_json, json, now],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| {
                MekaError::Database(format!("failed to save MCP credentials: {error}"))
            })
    }

    /// Persist (or replace) one of an MCP server's secrets unconditionally, for a caller whose
    /// value is not derived from a stored one: `meka mcp add` / `login` reading from stdin, and the
    /// interactive authorization flow depositing its first bundle.
    ///
    /// The conflict target is the whole key, so writing one kind leaves the server's other kinds
    /// untouched. That is what lets a confidential client hold its client secret while its OAuth
    /// bundle is replaced.
    pub(crate) async fn save_mcp_credentials(
        &self,
        server_name: &str,
        kind: McpCredentialKind,
        secret: &str,
    ) -> Result<()> {
        let server_name = server_name.to_string();
        let kind = kind.as_str();
        let secret = secret.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO mcp_credentials (server_name, kind, secret, updated_at)
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(server_name, kind) DO UPDATE SET
                         secret = excluded.secret,
                         updated_at = excluded.updated_at",
                    rusqlite::params![server_name, kind, secret, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save MCP credentials: {error}"))
            })
    }

    /// Drop just one of a server's secrets.
    ///
    /// `meka mcp login` clears the OAuth row before running the flow, so a stale bundle cannot be
    /// picked up mid-authorization. It must leave the other kinds alone: a confidential client's
    /// `client_secret` is an *input* to the flow it is about to run, and clearing everything would
    /// delete the credential the login needs to succeed.
    pub(crate) async fn clear_mcp_credentials_of_kind(
        &self,
        server_name: &str,
        kind: McpCredentialKind,
    ) -> Result<()> {
        let server_name = server_name.to_string();
        let kind = kind.as_str();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM mcp_credentials WHERE server_name = ?1 AND kind = ?2",
                    rusqlite::params![server_name, kind],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to clear MCP credentials: {error}"))
            })
    }

    /// Drop every secret this server has, whatever kind. `meka mcp logout` and `remove`.
    pub(crate) async fn clear_mcp_credentials(&self, server_name: &str) -> Result<()> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM mcp_credentials WHERE server_name = ?1",
                    rusqlite::params![server_name],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to clear MCP credentials: {error}"))
            })
    }

    /// Whether this server has a stored credential at all, whatever kind.
    ///
    /// `meka mcp remove` asks this rather than loading, because it is about to delete every kind
    /// and only needs to know there is something to delete. Loading with a guessed kind would make
    /// a server that authenticates the other way look like a name that does not exist.
    pub(crate) async fn has_mcp_credentials(&self, server_name: &str) -> Result<bool> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let count: i64 = connection.query_row(
                    "SELECT COUNT(*) FROM mcp_credentials WHERE server_name = ?1",
                    rusqlite::params![server_name],
                    |row| row.get(0),
                )?;
                Ok(count > 0)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to look up MCP credentials: {error}"))
            })
    }

    /// Every MCP server name that has a stored credential of any kind, sorted.
    ///
    /// The counterpart to [`Self::list_credential_accounts`], and stale for the same reason:
    /// deleting an `[[mcp.servers]]` entry by hand strands its secret here. `meka mcp list` diffs
    /// the result against the configured servers.
    ///
    /// Kind-agnostic on purpose: an orphaned bearer is exactly as much of a leak as an orphaned
    /// OAuth bundle, and the report exists to name what is still lying around.
    pub(crate) async fn list_mcp_credential_servers(&self) -> Result<Vec<String>> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                // DISTINCT because the table is keyed by `(server_name, kind)`: a confidential
                // OAuth client holds two rows and a bearer beside them would make three, and this
                // answers "which servers have a secret", not "how many secrets are there".
                let mut statement = connection.prepare(
                    "SELECT DISTINCT server_name FROM mcp_credentials ORDER BY server_name",
                )?;
                let servers = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(servers)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to list MCP credentials: {error}"))
            })
    }
}
/// What an account authenticates with, as the `account_credentials` row serializes it.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) enum AuthCredential {
    ApiKey(String),
    OAuthToken {
        access_token: String,
        refresh_token: Option<String>,
        expires_at: Option<i64>,
        /// Provider-specific identity carried alongside the bearer token. Only
        /// `chatgpt-subscription` populates it, with the `chatgpt_account_id` read out of the
        /// id_token JWT and sent on every request as `ChatGPT-Account-ID`; Claude OAuth leaves it
        /// `None`.
        account_id: Option<String>,
    },
}
/// Hand-written so a credential cannot reach a log through a `{:?}` on any struct that holds one.
///
/// The derived impl prints the bearer token verbatim, and a provider struct is exactly the kind of
/// thing that ends up inside a `tracing::debug!` or an error's `{:?}`. Lengths are kept because
/// they are what a "wrong key pasted" diagnosis needs.
impl std::fmt::Debug for AuthCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(key) => f
                .debug_tuple("ApiKey")
                .field(&format_args!("[REDACTED len={}]", key.len()))
                .finish(),
            Self::OAuthToken {
                access_token,
                refresh_token,
                expires_at,
                account_id,
            } => f
                .debug_struct("OAuthToken")
                .field(
                    "access_token",
                    &format_args!("[REDACTED len={}]", access_token.len()),
                )
                .field(
                    "refresh_token",
                    &refresh_token
                        .as_ref()
                        .map(|token| format_args!("[REDACTED len={}]", token.len()).to_string()),
                )
                .field("expires_at", expires_at)
                .field("account_id", account_id)
                .finish(),
        }
    }
}
impl AuthCredential {
    /// The request header this credential is presented in, as `(name, value)`.
    pub(crate) fn auth_header(&self) -> (&'static str, String) {
        match self {
            AuthCredential::ApiKey(key) => ("x-api-key", key.clone()),
            AuthCredential::OAuthToken { access_token, .. } => {
                ("Authorization", crate::text::bearer(access_token))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mcp_credentials_round_trip() {
        let store = Store::for_test().await;
        let token_store = store.token_store();

        assert!(
            token_store
                .load_mcp_credentials("srv", crate::store::McpCredentialKind::OAuth)
                .await
                .expect("load absent")
                .is_none(),
            "no credentials should exist yet"
        );

        token_store
            .save_mcp_credentials(
                "srv",
                crate::store::McpCredentialKind::OAuth,
                r#"{"tokens":{"access_token":"at1"}}"#,
            )
            .await
            .expect("save");
        assert_eq!(
            token_store
                .load_mcp_credentials("srv", crate::store::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .as_deref(),
            Some(r#"{"tokens":{"access_token":"at1"}}"#)
        );

        // Upsert: second save replaces the first.
        token_store
            .save_mcp_credentials(
                "srv",
                crate::store::McpCredentialKind::OAuth,
                r#"{"tokens":{"access_token":"at2"}}"#,
            )
            .await
            .expect("save again");
        assert_eq!(
            token_store
                .load_mcp_credentials("srv", crate::store::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .as_deref(),
            Some(r#"{"tokens":{"access_token":"at2"}}"#)
        );

        token_store
            .clear_mcp_credentials("srv")
            .await
            .expect("clear");
        assert!(
            token_store
                .load_mcp_credentials("srv", crate::store::McpCredentialKind::OAuth)
                .await
                .expect("load after clear")
                .is_none()
        );
    }

    #[tokio::test]
    async fn mcp_credentials_are_scoped_per_server() {
        let store = Store::for_test().await;
        let token_store = store.token_store();
        token_store
            .save_mcp_credentials("a", crate::store::McpCredentialKind::OAuth, "alpha")
            .await
            .expect("save a");
        token_store
            .save_mcp_credentials("b", crate::store::McpCredentialKind::OAuth, "beta")
            .await
            .expect("save b");
        token_store
            .clear_mcp_credentials("a")
            .await
            .expect("clear a");
        assert!(
            token_store
                .load_mcp_credentials("a", crate::store::McpCredentialKind::OAuth)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            token_store
                .load_mcp_credentials("b", crate::store::McpCredentialKind::OAuth)
                .await
                .unwrap()
                .as_deref(),
            Some("beta")
        );
    }

    /// One server is one name in the listing, however many secrets it holds.
    ///
    /// The listing is a set of server names, and the composite key makes it tempting to forget
    /// that: a confidential OAuth client has two rows, and `meka mcp list` would name it twice in
    /// the orphaned-credential report, which reads as two different strandings to clean up.
    #[tokio::test]
    async fn a_server_with_several_kinds_is_listed_once() {
        use crate::store::McpCredentialKind;

        let store = Store::for_test().await;
        let token_store = store.token_store();
        for kind in [
            McpCredentialKind::Bearer,
            McpCredentialKind::ClientSecret,
            McpCredentialKind::OAuth,
        ] {
            token_store
                .save_mcp_credentials("api", kind, "not-a-real-secret")
                .await
                .expect("save");
        }
        token_store
            .save_mcp_credentials("docs", McpCredentialKind::OAuth, "not-a-real-secret")
            .await
            .expect("save");

        assert_eq!(
            token_store
                .list_mcp_credential_servers()
                .await
                .expect("list"),
            vec!["api".to_string(), "docs".to_string()],
            "three kinds on one server is still one name"
        );
    }

    /// The case `PRIMARY KEY (server_name, kind)` exists for.
    ///
    /// A confidential `auth = "oauth"` client holds two secrets at once: the long-lived client
    /// secret it authenticates *with*, and the bundle it obtained. Refreshing the bundle must not
    /// touch the secret. Under a `server_name`-only key these collide, and the server would work
    /// until its first token refresh and fail from then on.
    #[tokio::test]
    async fn one_server_holds_a_client_secret_and_an_oauth_bundle_at_once() {
        use crate::store::McpCredentialKind;

        let store = Store::for_test().await;
        let token_store = store.token_store();

        token_store
            .save_mcp_credentials(
                "api",
                McpCredentialKind::ClientSecret,
                "cs-not-a-real-secret",
            )
            .await
            .expect("save client secret");
        token_store
            .save_mcp_credentials("api", McpCredentialKind::OAuth, r#"{"access_token":"at1"}"#)
            .await
            .expect("save bundle");

        // A refresh: rmcp's adapter compare-and-swaps the bundle it last read.
        assert!(
            token_store
                .replace_mcp_credentials(
                    "api",
                    r#"{"access_token":"at1"}"#,
                    r#"{"access_token":"at2"}"#,
                )
                .await
                .expect("refresh"),
            "the refresh should have matched the stored bundle"
        );

        assert_eq!(
            token_store
                .load_mcp_credentials("api", McpCredentialKind::OAuth)
                .await
                .expect("load bundle")
                .as_deref(),
            Some(r#"{"access_token":"at2"}"#),
            "the refresh should have moved the bundle"
        );
        assert_eq!(
            token_store
                .load_mcp_credentials("api", McpCredentialKind::ClientSecret)
                .await
                .expect("load client secret")
                .as_deref(),
            Some("cs-not-a-real-secret"),
            "the refresh must not have touched the client secret"
        );
    }

    /// The two strings a kind carries answer to different masters: `as_str` is schema and must
    /// never move, `label` is prose and is free to. Printing the schema token where the prose
    /// belongs is the drift worth pinning, since it reads as almost right.
    #[test]
    fn a_kind_says_one_thing_to_the_schema_and_another_to_the_user() {
        use crate::store::McpCredentialKind;

        for (kind, stored, shown) in [
            (McpCredentialKind::Bearer, "bearer", "bearer token"),
            (
                McpCredentialKind::ClientSecret,
                "client_secret",
                "client secret",
            ),
            (McpCredentialKind::OAuth, "oauth", "OAuth tokens"),
        ] {
            assert_eq!(kind.as_str(), stored, "the stored discriminator moved");
            assert_eq!(kind.label(), shown, "the label the user reads moved");
            assert_ne!(
                kind.label(),
                kind.as_str(),
                "a label that is the schema value is a schema token leaking into the UI"
            );
        }
    }

    /// `mcp login` clears before authorizing, and must clear only what it is about to replace.
    #[tokio::test]
    async fn clearing_one_kind_leaves_the_others() {
        use crate::store::McpCredentialKind;

        let store = Store::for_test().await;
        let token_store = store.token_store();
        for (kind, secret) in [
            (McpCredentialKind::Bearer, "bearer-not-a-real-token"),
            (McpCredentialKind::ClientSecret, "cs-not-a-real-secret"),
            (McpCredentialKind::OAuth, r#"{"access_token":"at1"}"#),
        ] {
            token_store
                .save_mcp_credentials("api", kind, secret)
                .await
                .expect("save");
        }

        token_store
            .clear_mcp_credentials_of_kind("api", McpCredentialKind::OAuth)
            .await
            .expect("clear oauth");

        assert!(
            token_store
                .load_mcp_credentials("api", McpCredentialKind::OAuth)
                .await
                .expect("load oauth")
                .is_none()
        );
        assert!(
            token_store
                .load_mcp_credentials("api", McpCredentialKind::Bearer)
                .await
                .expect("load bearer")
                .is_some(),
            "clearing the OAuth bundle must leave the bearer"
        );
        assert!(
            token_store
                .load_mcp_credentials("api", McpCredentialKind::ClientSecret)
                .await
                .expect("load client secret")
                .is_some(),
            "clearing the OAuth bundle must leave the client secret, which the login needs"
        );
        assert!(
            token_store
                .has_mcp_credentials("api")
                .await
                .expect("has any after clearing one kind"),
            "two kinds remain, so the server still has credentials"
        );
    }

    #[tokio::test]
    async fn oauth_token_round_trip_preserves_all_fields() {
        let store = Store::for_test().await;
        let token_store = store.token_store();

        let credential = AuthCredential::OAuthToken {
            access_token: "access-1".to_string(),
            refresh_token: Some("refresh-1".to_string()),
            expires_at: Some(1_700_000_000_000),
            account_id: Some("account-abc".to_string()),
        };

        token_store
            .save_account_credential("chatgpt-subscription", &credential)
            .await
            .expect("save");

        let loaded = token_store
            .load_account_credential("chatgpt-subscription")
            .await
            .expect("load")
            .expect("present");

        match loaded {
            AuthCredential::OAuthToken {
                access_token,
                refresh_token,
                expires_at,
                account_id,
            } => {
                assert_eq!(access_token, "access-1");
                assert_eq!(refresh_token.as_deref(), Some("refresh-1"));
                assert_eq!(expires_at, Some(1_700_000_000_000));
                assert_eq!(account_id.as_deref(), Some("account-abc"));
            }
            _ => panic!("expected OAuthToken"),
        }
    }

    #[tokio::test]
    async fn oauth_token_round_trip_account_id_optional() {
        // Claude OAuth doesn't populate `account_id`; make sure round-tripping a `None` value
        // works without losing other fields.
        let store = Store::for_test().await;
        let token_store = store.token_store();

        let credential = AuthCredential::OAuthToken {
            access_token: "claude-token".to_string(),
            refresh_token: None,
            expires_at: None,
            account_id: None,
        };

        token_store
            .save_account_credential("claude", &credential)
            .await
            .expect("save");

        let loaded = token_store
            .load_account_credential("claude")
            .await
            .expect("load");

        match loaded {
            Some(AuthCredential::OAuthToken {
                access_token,
                account_id,
                ..
            }) => {
                assert_eq!(access_token, "claude-token");
                assert!(account_id.is_none());
            }
            _ => panic!("expected OAuthToken with account_id=None"),
        }
    }

    /// Two providers can persist independently with different `account_id` values. This test
    /// verifies the provider PK keeps chatgpt-subscription and a hypothetical future OAuth provider
    /// isolated.
    #[tokio::test]
    async fn oauth_token_two_providers_independent() {
        let store = Store::for_test().await;
        let token_store = store.token_store();

        let codex_credential = AuthCredential::OAuthToken {
            access_token: "codex-access".to_string(),
            refresh_token: Some("codex-refresh".to_string()),
            expires_at: Some(2_000_000_000_000),
            account_id: Some("workspace-1".to_string()),
        };
        let claude_credential = AuthCredential::OAuthToken {
            access_token: "claude-access".to_string(),
            refresh_token: Some("claude-refresh".to_string()),
            expires_at: Some(3_000_000_000_000),
            account_id: None,
        };

        token_store
            .save_account_credential("chatgpt-subscription", &codex_credential)
            .await
            .expect("save codex");
        token_store
            .save_account_credential("claude", &claude_credential)
            .await
            .expect("save claude");

        let codex_loaded = token_store
            .load_account_credential("chatgpt-subscription")
            .await
            .expect("load codex")
            .expect("present");
        let claude_loaded = token_store
            .load_account_credential("claude")
            .await
            .expect("load claude")
            .expect("present");

        if let AuthCredential::OAuthToken { account_id, .. } = codex_loaded {
            assert_eq!(account_id.as_deref(), Some("workspace-1"));
        } else {
            panic!("expected OAuthToken");
        }
        if let AuthCredential::OAuthToken { account_id, .. } = claude_loaded {
            assert!(account_id.is_none());
        } else {
            panic!("expected OAuthToken");
        }
    }

    #[tokio::test]
    async fn api_key_credential_round_trip() {
        let store = Store::for_test().await;
        let token_store = store.token_store();

        let credential = AuthCredential::ApiKey("sk-secret-123".to_string());
        token_store
            .save_account_credential("personal", &credential)
            .await
            .expect("save");

        let loaded = token_store
            .load_account_credential("personal")
            .await
            .expect("load")
            .expect("present");

        match loaded {
            AuthCredential::ApiKey(key) => assert_eq!(key, "sk-secret-123"),
            _ => panic!("expected ApiKey"),
        }
    }

    #[tokio::test]
    async fn delete_account_credential_removes_entry() {
        let store = Store::for_test().await;
        let token_store = store.token_store();

        token_store
            .save_account_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save");
        token_store
            .delete_account_credential("work")
            .await
            .expect("delete");

        assert!(
            token_store
                .load_account_credential("work")
                .await
                .expect("load")
                .is_none(),
            "credential must be gone after delete"
        );
        // Deleting a missing profile is a no-op, not an error.
        token_store
            .delete_account_credential("work")
            .await
            .expect("delete missing is a no-op");
    }

    /// A refresh may only replace the credential it was derived from.
    ///
    /// Two meka processes refreshing at once both present the same refresh token, and against an
    /// issuer with a reuse window both succeed. A blind upsert leaves the database holding
    /// whichever refresh finished last, which is the token the issuer has already superseded, and
    /// the symptom arrives at the *next* launch as `invalid_grant` with nothing naming the cause.
    /// The loser adopts the winner's credential instead, so the store and the issuer agree.
    #[tokio::test]
    async fn a_refresh_cannot_overwrite_a_credential_it_did_not_read() {
        let store = Store::for_test().await;
        let token_store = store.token_store();
        let original = AuthCredential::ApiKey("original".to_string());
        token_store
            .save_account_credential("work", &original)
            .await
            .expect("save");

        let original_version = token_store
            .load_account_credential_versioned("work")
            .await
            .expect("load")
            .expect("stored")
            .version;

        // The winner: the row is still at the version it read, so its write lands.
        let winner = AuthCredential::ApiKey("winner".to_string());
        match token_store
            .replace_account_credential("work", &original_version, &winner)
            .await
            .expect("swap")
        {
            CredentialWrite::Stored => {}
            other => panic!("expected the first write to land, got {other:?}"),
        }

        // The loser: derived from the same version, which the row has moved past.
        let loser = AuthCredential::ApiKey("loser".to_string());
        match token_store
            .replace_account_credential("work", &original_version, &loser)
            .await
            .expect("swap")
        {
            CredentialWrite::Superseded(current) => match current.credential {
                AuthCredential::ApiKey(key) => assert_eq!(
                    key, "winner",
                    "the loser must be handed what the row holds, to adopt rather than retry"
                ),
                other => panic!("expected an ApiKey, got {other:?}"),
            },
            other => panic!("expected the second write to be refused, got {other:?}"),
        }

        match token_store
            .load_account_credential("work")
            .await
            .expect("load")
            .expect("still stored")
        {
            AuthCredential::ApiKey(key) => assert_eq!(key, "winner"),
            other => panic!("expected an ApiKey, got {other:?}"),
        }
    }

    /// A profile disconnected mid-refresh must stay disconnected. Re-creating the row would put
    /// back an account the user had just removed, and it would come back working.
    #[tokio::test]
    async fn a_refresh_does_not_resurrect_a_removed_profile() {
        let store = Store::for_test().await;
        let token_store = store.token_store();
        let original = AuthCredential::ApiKey("original".to_string());
        token_store
            .save_account_credential("work", &original)
            .await
            .expect("save");
        token_store
            .delete_account_credential("work")
            .await
            .expect("remove");

        match token_store
            .replace_account_credential(
                "work",
                "2026-01-01T00:00:00+00:00",
                &AuthCredential::ApiKey("new".into()),
            )
            .await
            .expect("swap")
        {
            CredentialWrite::Gone => {}
            other => panic!("expected the write to find nothing to replace, got {other:?}"),
        }
        assert!(
            token_store
                .load_account_credential("work")
                .await
                .expect("load")
                .is_none(),
            "the profile must stay removed"
        );
    }

    /// The swap is keyed on the row's version, not on how the credential happens to be spelled.
    /// Compared as bytes, a row written by another build -- one that had not yet learned a field
    /// this one fills in and writes back out -- was unswappable, and every refresh after an upgrade
    /// was refused until the user signed in again.
    #[tokio::test]
    async fn a_row_spelled_differently_is_still_swappable_at_its_version() {
        let store = Store::for_test().await;
        let token_store = store.token_store();
        token_store
            .save_account_credential("work", &AuthCredential::ApiKey("original".to_string()))
            .await
            .expect("save");
        let version = token_store
            .load_account_credential_versioned("work")
            .await
            .expect("load")
            .expect("stored")
            .version;
        // The same value in another spelling, at the same version: what an older or newer build
        // leaves behind.
        store
            .connection
            .call(|connection| {
                connection.execute(
                    "UPDATE account_credentials SET credentials_json = ?1 WHERE account = 'work'",
                    ["{ \"ApiKey\" : \"original\" }"],
                )
            })
            .await
            .expect("respell the row");

        match token_store
            .replace_account_credential(
                "work",
                &version,
                &AuthCredential::ApiKey("refreshed".to_string()),
            )
            .await
            .expect("swap")
        {
            CredentialWrite::Stored => {}
            other => {
                panic!("the row is at the version that was read, so the swap lands: {other:?}")
            }
        }
    }

    /// The write-ahead log inherits the database's owner-only mode. SQLite derives the `-wal` and
    /// `-shm` modes from the main file, and this is the boundary that keeps a credential the store
    /// holds from another account on the machine; pinned so a change to how the store is opened
    /// cannot widen it unnoticed.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_write_ahead_log_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("data").join("meka.db");
        let store = Store::open(Some(&db_path), &Default::default())
            .await
            .expect("open");
        // A write, so the WAL exists.
        store
            .token_store()
            .save_account_credential("work", &AuthCredential::ApiKey("secret".to_string()))
            .await
            .expect("save");
        let wal = temp_dir.path().join("data").join("meka.db-wal");
        let mode = std::fs::metadata(&wal)
            .expect("the WAL exists after a write")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            mode, 0o600,
            "the WAL carries credentials too (got {mode:o})"
        );
    }

    /// The MCP half of the same rule. rmcp hands the adapter a bare `save`, so what it compares
    /// against is the credentials it last *read* -- and a write derived from something the row no
    /// longer holds would move the stored credential backwards onto a token another process has
    /// already replaced.
    #[tokio::test]
    async fn an_mcp_refresh_cannot_overwrite_credentials_it_did_not_read() {
        let store = Store::for_test().await;
        let token_store = store.token_store();
        token_store
            .save_mcp_credentials(
                "docs",
                crate::store::McpCredentialKind::OAuth,
                "{\"token\":\"original\"}",
            )
            .await
            .expect("save");

        assert!(
            token_store
                .replace_mcp_credentials(
                    "docs",
                    "{\"token\":\"original\"}",
                    "{\"token\":\"first\"}"
                )
                .await
                .expect("swap"),
            "the writer that still holds what it read wins"
        );
        assert!(
            !token_store
                .replace_mcp_credentials(
                    "docs",
                    "{\"token\":\"original\"}",
                    "{\"token\":\"second\"}"
                )
                .await
                .expect("swap"),
            "and the writer holding a superseded copy is refused"
        );
        assert_eq!(
            token_store
                .load_mcp_credentials("docs", crate::store::McpCredentialKind::OAuth)
                .await
                .expect("load")
                .as_deref(),
            Some("{\"token\":\"first\"}"),
            "the stored credential must never move backwards"
        );
    }

    /// The lock that keeps two processes from spending the same refresh token at once.
    ///
    /// Per profile, not per store: one profile refreshing must not stall an unrelated one, which is
    /// the whole reason the file is named after the profile rather than the database.
    #[tokio::test]
    async fn the_credential_lock_is_per_profile() {
        let store = Store::for_test().await;
        let token_store = store.token_store();

        let held = token_store
            .try_lock_account_credential("work")
            .expect("ask")
            .expect("nobody holds it");
        assert!(
            token_store
                .try_lock_account_credential("work")
                .expect("ask")
                .is_none(),
            "a second holder of the same profile must be refused"
        );
        assert!(
            token_store
                .try_lock_account_credential("personal")
                .expect("ask")
                .is_some(),
            "a different profile is a different lock"
        );

        drop(held);
        assert!(
            token_store
                .try_lock_account_credential("work")
                .expect("ask")
                .is_some(),
            "and it is released when the holder goes"
        );
    }

    /// A profile name is a TOML table key with no charset rule behind it, so it cannot be a file
    /// name directly: `[accounts."../../etc/passwd"]` would otherwise name a path. Stripping is
    /// what keeps the file inside the lock directory, and the hash is what keeps two names that
    /// strip alike from sharing a lock -- one profile's refresh blocking an unrelated one's, for as
    /// long as it takes.
    #[tokio::test]
    async fn the_credential_lock_handles_a_profile_name_that_is_not_a_file_name() {
        let store = Store::for_test().await;
        let token_store = store.token_store();

        let _escaping = token_store
            .try_lock_account_credential("../../etc/passwd")
            .expect("ask")
            .expect("nobody holds it");
        let inside: Vec<_> = std::fs::read_dir(&store.lock_dir)
            .expect("read the lock dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(ACCOUNT_LOCK_PREFIX)
            })
            .collect();
        assert_eq!(
            inside.len(),
            1,
            "a path-shaped profile name must still land inside the lock directory"
        );

        // `a/b` and `a.b` both strip to `ab`, so without the hash they would be one lock.
        let _first = token_store
            .try_lock_account_credential("a/b")
            .expect("ask")
            .expect("nobody holds it");
        assert!(
            token_store
                .try_lock_account_credential("a.b")
                .expect("ask")
                .is_some(),
            "two profiles that strip to the same readable name must not share a lock"
        );
    }

    /// The listing queries exist so a credential whose config entry was deleted by hand can still
    /// be named. Nothing else in the codebase enumerates either table, so an unlisted row is an
    /// invisible one: a live API key or OAuth refresh token no surface can report or remove.
    #[tokio::test]
    async fn credential_listings_name_every_stored_row() {
        let store = Store::for_test().await;
        let token_store = store.token_store();

        assert!(
            token_store
                .list_credential_accounts()
                .await
                .expect("list empty")
                .is_empty()
        );
        assert!(
            token_store
                .list_mcp_credential_servers()
                .await
                .expect("list empty")
                .is_empty()
        );

        token_store
            .save_account_credential("work", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save work");
        token_store
            .save_account_credential("archive", &AuthCredential::ApiKey("key".to_string()))
            .await
            .expect("save archive");
        token_store
            .save_mcp_credentials(
                "linear",
                crate::store::McpCredentialKind::OAuth,
                r#"{"tokens":{"access_token":"at"}}"#,
            )
            .await
            .expect("save linear");

        // Sorted, so the reported order doesn't depend on insertion order.
        assert_eq!(
            token_store.list_credential_accounts().await.expect("list"),
            vec!["archive".to_string(), "work".to_string()]
        );
        assert_eq!(
            token_store
                .list_mcp_credential_servers()
                .await
                .expect("list"),
            vec!["linear".to_string()]
        );

        // The two tables are independent: clearing one must not hide rows in the other.
        token_store
            .delete_account_credential("work")
            .await
            .expect("delete");
        assert_eq!(
            token_store.list_credential_accounts().await.expect("list"),
            vec!["archive".to_string()]
        );
        assert_eq!(
            token_store
                .list_mcp_credential_servers()
                .await
                .expect("list"),
            vec!["linear".to_string()]
        );
    }
}
