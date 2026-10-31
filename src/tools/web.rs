//! The web tool: `web_fetch`, an HTTP GET with HTML-to-markdown conversion or a multimodal image
//! return.

use std::sync::LazyLock;

use async_trait::async_trait;
use futures::StreamExt;
use html2md::rewrite_html_custom_with_url;

use super::{
    Tool, ToolOutput,
    util::{compile_user_regex, optional_str, redirects_to_scratchpad, require_str},
};
use crate::{
    config::{MinTlsVersion, WebClientConfig},
    error::{MekaError, Result},
    image::{ImageHandling, classify_content_type},
    permission::Permission,
    provider::ToolDefinition,
    tools::util::build_image_tool_output,
};

/// The two switches that keep a fetch from an address, and which of them refuses a given one.
///
/// One guard for both because they hold the same three doors: the URL `web_fetch` is given, every
/// redirect hop the client follows, and every name the client resolves. The private switch is
/// `[web].refuse_private_destinations`; the own-address switch is `[permissions].network =
/// "remote"`, under which `web_fetch` reaches other machines alone, as the shell does.
#[derive(Debug, Clone, Copy)]
struct DestinationGuard {
    refuse_private: bool,
    refuse_own: bool,
}

impl DestinationGuard {
    /// Whether either switch is on, which is what decides whether the client's resolver and
    /// redirect policy judge at all.
    fn is_on(self) -> bool {
        self.refuse_private || self.refuse_own
    }

    /// The switch that refuses `address`, named as the user set it, or `None` where the address
    /// is admitted. The private switch answers first where both are on, since every own address
    /// is a private one and that switch is the broader setting.
    fn refusal(self, address: std::net::IpAddr) -> Option<&'static str> {
        if self.refuse_private && is_private_destination(address) {
            return Some("`[web].refuse_private_destinations`");
        }
        if self.refuse_own && is_own_address(address) {
            return Some("`[permissions].network = \"remote\"`");
        }
        None
    }
}

/// The redirect policy: `0` follows no redirect and returns the 3xx as the response, any other
/// cap follows that many hops. With a guard on, a hop naming a refused literal address is refused
/// as well; a hop naming a host is judged by the resolver when it connects.
fn redirect_policy(max_redirects: usize, guard: DestinationGuard) -> reqwest::redirect::Policy {
    if !guard.is_on() {
        return if max_redirects == 0 {
            reqwest::redirect::Policy::none()
        } else {
            reqwest::redirect::Policy::limited(max_redirects)
        };
    }
    reqwest::redirect::Policy::custom(move |attempt| {
        if max_redirects == 0 {
            return attempt.stop();
        }
        if attempt.previous().len() > max_redirects {
            return attempt.error("too many redirects");
        }
        match refused_literal_destination(attempt.url().as_str(), guard) {
            Some((address, switch)) => {
                attempt.error(format!("a redirect to {address}, which {switch} refuses"))
            }
            None => attempt.follow(),
        }
    })
}

/// The address a URL names outright and the switch that refuses it, when it names a refused one:
/// the door for a literal, which hyper connects to without resolving. A host name answers `None`
/// here and is judged by the guard's resolver when it is resolved.
fn refused_literal_destination(
    url: &str,
    guard: DestinationGuard,
) -> Option<(std::net::IpAddr, &'static str)> {
    let address = match url::Url::parse(url).ok()?.host()? {
        url::Host::Ipv4(address) => std::net::IpAddr::V4(address),
        url::Host::Ipv6(address) => std::net::IpAddr::V6(address),
        url::Host::Domain(_) => return None,
    };
    guard.refusal(address).map(|switch| (address, switch))
}

/// The resolver a guard installs: every name the client connects to is looked up here, and one
/// that answers with an address a switch refuses is refused whole.
///
/// On the client's resolver rather than on the URL, so every connection is judged on the
/// addresses it would actually use: a redirect's target, a literal address, and a name that
/// answers differently on a second lookup, which a check made on the first would let through. A
/// name that answers with an admitted address beside a refused one is refused too, since the
/// client would connect to whichever it tried first. With a proxy configured the proxy resolves
/// the target and this sees only the proxy's own name, which the config page says.
impl reqwest::dns::Resolve for DestinationGuard {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        let guard = *self;
        Box::pin(async move {
            let addresses: Vec<std::net::SocketAddr> =
                tokio::net::lookup_host((host.as_str(), 0)).await?.collect();
            if let Some((address, switch)) = addresses
                .iter()
                .find_map(|address| guard.refusal(address.ip()).map(|switch| (address, switch)))
            {
                return Err(format!(
                    "'{host}' resolves to {}, which {switch} refuses",
                    address.ip()
                )
                .into());
            }
            Ok(Box::new(addresses.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// Whether an address is this machine's: loopback, unspecified, or one an interface here holds.
/// What `[permissions].network = "remote"` keeps `web_fetch` from, as a command under it is kept
/// off this machine; another machine on the LAN stays reachable, which is what sets the switch
/// apart from `[web].refuse_private_destinations`.
///
/// An interface's address is told by routing rather than by binding: connecting a UDP socket
/// sends nothing and makes the kernel pick the source address for the destination, which is the
/// destination itself exactly when the destination is local. A bind would say yes to any address
/// on a host with `ip_nonlocal_bind` on. An address this machine cannot route to, or whose family
/// it has no stack for, is not one it holds, which keeps a dual-stack name reachable from a host
/// without IPv6; any other error leaves the question unanswered and is read as "own", since
/// admitting is the direction that loses the boundary.
fn is_own_address(address: std::net::IpAddr) -> bool {
    if address.is_loopback() || address.is_unspecified() {
        return true;
    }
    if let std::net::IpAddr::V6(v6) = address
        && let Some(v4) = v6.to_ipv4_mapped()
    {
        return is_own_address(std::net::IpAddr::V4(v4));
    }
    let unspecified: std::net::IpAddr = match address {
        std::net::IpAddr::V4(_) => std::net::Ipv4Addr::UNSPECIFIED.into(),
        std::net::IpAddr::V6(_) => std::net::Ipv6Addr::UNSPECIFIED.into(),
    };
    let probe = std::net::UdpSocket::bind((unspecified, 0)).and_then(|socket| {
        socket
            .connect((address, ROUTE_PROBE_PORT))
            .and_then(|()| socket.local_addr())
    });
    match probe {
        Ok(local) => local.ip() == address,
        Err(error) => !is_unreachable_from_here(&error),
    }
}

/// The port the route probe in [`is_own_address`] connects to: any nonzero one serves, since a
/// connected UDP socket sends nothing; the discard port names the intent.
const ROUTE_PROBE_PORT: u16 = 9;

/// Whether a route probe's error says the address is beyond this machine: no route to it, or no
/// stack for its family at all. Every address an interface here holds has a route.
fn is_unreachable_from_here(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::NetworkUnreachable
            | std::io::ErrorKind::HostUnreachable
            | std::io::ErrorKind::AddrNotAvailable
    ) || error.raw_os_error() == Some(ADDRESS_FAMILY_UNSUPPORTED)
}

/// `EAFNOSUPPORT`: the socket for the family cannot be made here, so no address of that family is
/// this machine's. Not among `std::io::ErrorKind`'s names, so matched by number.
#[cfg(unix)]
const ADDRESS_FAMILY_UNSUPPORTED: i32 = libc::EAFNOSUPPORT;
/// `WSAEAFNOSUPPORT`, Winsock's spelling of the same.
#[cfg(windows)]
const ADDRESS_FAMILY_UNSUPPORTED: i32 = 10047;

/// Whether an address is one a fetch must not reach under `[web].refuse_private_destinations`:
/// loopback, private, link-local, carrier-grade NAT, unspecified and broadcast, and an IPv6
/// address that maps one of those.
fn is_private_destination(address: std::net::IpAddr) -> bool {
    match address {
        std::net::IpAddr::V4(v4) => {
            let [first, second, ..] = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || (first == 100 && (64..=127).contains(&second))
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| is_private_destination(std::net::IpAddr::V4(v4)))
        }
    }
}

/// Whether `web_fetch` is registered under `network`: the fetch runs in the meka process, where
/// no sandbox reaches, so the one way to keep a session off the network is to give it no tool
/// that is on it. One predicate for the registry and for `meka tool list`, which shows the tool
/// as disabled rather than dropping its row.
pub(crate) fn web_fetch_registered(network: crate::config::Network) -> bool {
    network != crate::config::Network::None
}

/// Whether `web_fetch` refuses this machine's own addresses under `network`: how the in-process
/// tool holds `remote`, which keeps a session off this machine while other machines stay in
/// reach. One predicate for the registry's tool and its client.
pub(crate) fn web_fetch_refuses_own_addresses(network: crate::config::Network) -> bool {
    network == crate::config::Network::Remote
}

/// Build the `reqwest::Client` for `web_fetch` from the resolved
/// [`WebClientConfig`].
///
/// Refused as [`MekaError::Installation`] rather than built from a fallback that ignores the
/// user's intent. `crate::host::build_shared_deps` builds one at startup, which is what makes a bad
/// proxy URL or CA file fail the process with the message at the terminal before any session
/// exists; each session's registry then builds its own from the same settings. The message names
/// a path or a URL out of `config.toml`, which is why its class keeps it off the wire on the hosts
/// a remote caller reaches. `refuse_own_addresses` is what `[permissions].network = "remote"`
/// asks of the client, beside the `[web]` switch the config carries.
pub(crate) fn build_web_client(
    config: &WebClientConfig,
    refuse_own_addresses: bool,
) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .user_agent(&config.user_agent)
        .timeout(config.request_timeout);
    // The guard has two doors here, since hyper resolves a name through the client's resolver and
    // connects to a literal address without asking it: the resolver below judges every name, and
    // the redirect policy judges every hop a literal names, as `web_fetch` judges the URL it is
    // given.
    let guard = DestinationGuard {
        refuse_private: config.refuse_private_destinations,
        refuse_own: refuse_own_addresses,
    };
    if guard.is_on() {
        builder = builder.dns_resolver(std::sync::Arc::new(guard));
    }

    if let Some(timeout) = config.connect_timeout {
        builder = builder.connect_timeout(timeout);
    }
    if let Some(timeout) = config.read_timeout {
        builder = builder.read_timeout(timeout);
    }

    builder = builder.redirect(redirect_policy(config.max_redirects, guard));

    match config.proxy.as_deref() {
        None => {}
        Some("") | Some("none") => {
            // Explicit opt-out of reqwest's env-proxy auto-detection. Useful to override a
            // host-level `HTTP_PROXY` env var without unsetting it.
            builder = builder.no_proxy();
        }
        Some(url) => {
            // `reqwest::Proxy::all` accepts `"not-a-url"` as `http://not-a-url/`, which would
            // silently route traffic through a non-existent host.
            const ALLOWED_SCHEMES: &[&str] = &[
                "http://",
                "https://",
                "socks5://",
                "socks5h://",
                "socks4://",
            ];
            if !ALLOWED_SCHEMES.iter().any(|s| url.starts_with(s)) {
                return Err(MekaError::Installation(format!(
                    "`[web].proxy` '{}' is not a proxy URL: expected one of {}",
                    url,
                    ALLOWED_SCHEMES.join(", ")
                )));
            }
            let proxy = reqwest::Proxy::all(url).map_err(|error| {
                MekaError::Installation(format!(
                    "`[web].proxy` '{url}' is not a proxy URL: {error}"
                ))
            })?;
            builder = builder.proxy(proxy);
        }
    }

    if let Some(path) = &config.ca_cert_file {
        let bytes = std::fs::read(path).map_err(|error| {
            MekaError::Installation(format!(
                "failed to read `[web].ca_cert_file` '{}': {}",
                path.display(),
                error
            ))
        })?;
        let certs = reqwest::Certificate::from_pem_bundle(&bytes).map_err(|error| {
            MekaError::Installation(format!(
                "`[web].ca_cert_file` '{}' is not valid PEM: {}",
                path.display(),
                error
            ))
        })?;
        // `from_pem_bundle` returns an empty list for a file with no PEM blocks, which would ship
        // a client with zero added CAs.
        if certs.is_empty() {
            return Err(MekaError::Installation(format!(
                "`[web].ca_cert_file` '{}' holds no PEM certificates",
                path.display()
            )));
        }
        for cert in certs {
            builder = builder.add_root_certificate(cert);
        }
    }

    if config.https_only {
        builder = builder.https_only(true);
    }

    if let Some(version) = config.min_tls_version {
        let minimum = match version {
            MinTlsVersion::V1_0 => reqwest::tls::Version::TLS_1_0,
            MinTlsVersion::V1_1 => reqwest::tls::Version::TLS_1_1,
            MinTlsVersion::V1_2 => reqwest::tls::Version::TLS_1_2,
            MinTlsVersion::V1_3 => reqwest::tls::Version::TLS_1_3,
        };
        builder = builder.min_tls_version(minimum);
    }

    if config.danger_accept_invalid_certs {
        tracing::warn!(
            "`[web].danger_accept_invalid_certs` is enabled; a forged certificate can spoof any HTTPS response"
        );
        builder = builder.danger_accept_invalid_certs(true);
    }
    if config.danger_accept_invalid_hostnames {
        tracing::warn!(
            "`[web].danger_accept_invalid_hostnames` is enabled; a certificate for any name can \
             spoof any HTTPS response"
        );
        builder = builder.danger_accept_invalid_hostnames(true);
    }

    builder
        .build()
        .map_err(|error| MekaError::Installation(format!("failed to build web client: {error}")))
}

/// Matches the open/close tags of `<nav>` and `<footer>` elements (with any attributes). The name
/// is anchored by the trailing `(\s|>|/)` group so sibling elements like `<navbar>` or a custom
/// `<nav-menu>` are left untouched.
#[allow(
    clippy::expect_used,
    reason = "a literal regex; a parse failure is a typo the first test run catches"
)]
static BOILERPLATE_CONTAINER_TAG: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"(?i)<(/?)(?:nav|footer)(\s[^>]*)?>").expect("static regex")
});

/// Rewrite `<nav>` / `<footer>` container tags to `<div>` before HTML-to-markdown conversion.
///
/// `fast_html2md` unconditionally drops the entire subtree of `head, nav, footer, script, noscript,
/// style` as boilerplate (it calls lol_html's `el.remove()`, which deletes the element and all its
/// content). That is reasonable for `script` / `style`, but modern sites (Next.js and friends) put
/// primary navigation and useful footer links inside `<nav>` / `<footer>`, so those links (text and
/// href alike) silently vanish from the converted markdown. Renaming just the open/close tags to a
/// neutral `<div>` keeps the content while leaving `script` / `style` / `head` stripping intact.
fn keep_boilerplate_container_content(html: &str) -> std::borrow::Cow<'_, str> {
    BOILERPLATE_CONTAINER_TAG.replace_all(html, "<${1}div${2}>")
}

/// Convert fetched HTML to Markdown exactly the way `web_fetch` does. Two steps: rewrite `<nav>` /
/// `<footer>` containers so their links survive [`keep_boilerplate_container_content`], then run
/// `fast_html2md` with the document's URL as the base so root-relative links (`/docs`) resolve to
/// absolute URLs (`https://host/docs`) the model can follow directly. A `None` base leaves relative
/// links relative (the converter only rewrites hrefs that start with `/`).
fn html_to_markdown(html: &str, base_url: &Option<url::Url>) -> String {
    rewrite_html_custom_with_url(
        &keep_boilerplate_container_content(html),
        &None,
        false,
        base_url,
    )
}

/// Default `limit` applied when the caller doesn't pass one. Single source of truth for both the
/// parameter unwrap and the description shown to the agent. Pass `0` to disable the cap.
const DEFAULT_LIMIT_CHARS: usize = 30_000;

fn apply_headers(
    mut builder: reqwest::RequestBuilder,
    input: &serde_json::Value,
) -> reqwest::RequestBuilder {
    if let Some(headers) = input.get("headers").and_then(|h| h.as_object()) {
        for (key, value) in headers {
            if let Some(value_str) = value.as_str() {
                builder = builder.header(key.as_str(), value_str);
            }
        }
    }
    builder
}

pub(super) struct FetchUrlTool {
    /// `[web].refuse_private_destinations`: whether a URL naming a private literal address is
    /// refused before the request, the one door of the guard the client itself cannot hold.
    pub(super) refuse_private_destinations: bool,
    /// Whether a URL naming one of this machine's addresses is refused at the same door, which is
    /// how the tool holds `[permissions].network = "remote"`.
    pub(super) refuse_own_addresses: bool,
    pub(crate) client: reqwest::Client,
}

#[async_trait]
impl Tool for FetchUrlTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "web_fetch".to_string(),
            description: "Fetch a URL as Markdown, or raw HTML with `raw`. Supported raster images are returned as image content, with non-native formats converted to PNG; `limit`, `regex`, and `raw` do not apply to them. Request images only when image input is enabled.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 0,
                        "default": DEFAULT_LIMIT_CHARS,
                        "description": format!(
                            "Maximum number of characters to return. Default: {DEFAULT_LIMIT_CHARS}. Set to 0 for no limit."
                        )
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers. Overrides defaults (e.g., User-Agent).",
                        "additionalProperties": { "type": "string" }
                    },
                    "regex": {
                        "type": "string",
                        "description": "Optional regex pattern. If provided, only matching content is returned (all matches joined by newlines)."
                    },
                    "raw": {
                        "type": "boolean",
                        "default": false,
                        "description": "Return raw HTML instead of converting it to markdown. Default: false."
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["url"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let url = require_str(&input, "url", "web_fetch")?;
        let guard = DestinationGuard {
            refuse_private: self.refuse_private_destinations,
            refuse_own: self.refuse_own_addresses,
        };
        if let Some((address, switch)) = refused_literal_destination(&url, guard) {
            return Err(MekaError::ToolExecution {
                tool_name: "web_fetch".to_string(),
                message: format!("'{url}' names {address}, which {switch} refuses"),
            });
        }
        // Every wait below is raced against the turn's cancellation, as `shell_execute` races
        // its child: the agent awaits every tool future, so a fetch that ignored the token held a
        // canceled turn until the peer answered or `[web] request_timeout` ran out. Dropping the
        // send or the body read aborts the request; a conversion already running off the runtime
        // finishes on its own and is discarded.
        let cancellation = context.cancellation;

        let request = apply_headers(self.client.get(&url), &input);
        let response = unless_canceled(&cancellation, request.send())
            .await?
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "web_fetch".to_string(),
                message: format!(
                    "failed to fetch '{}': {}",
                    url,
                    crate::error::format_reqwest_error(&error)
                ),
            })?;

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolOutput::text(format!("HTTP {status} for '{url}'"), true));
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();

        // Capture the document's final (post-redirect) URL to resolve relative links against. Taken
        // before `read_body_capped` consumes the response; re-parsed through our own `url` crate so
        // the type matches `html_to_markdown` regardless of reqwest's `url` re-export.
        let document_url: Option<url::Url> = url::Url::parse(response.url().as_str()).ok();

        let body_bytes = unless_canceled(&cancellation, read_body_capped(response)).await??;

        // A response the server labeled as an image becomes a multimodal Image block rather than
        // going through html2md. `Content-Type` only gates whether to try: the media type comes
        // from the bytes, and a body that isn't an image at all (an HTML error page served
        // as `image/png`, which is common) falls through to the text path below instead of
        // shipping a block the provider will reject.
        if !matches!(
            classify_content_type(&content_type),
            ImageHandling::Unsupported
        ) {
            let sniffed = crate::image::classify_bytes(&body_bytes);
            if !matches!(sniffed, ImageHandling::Unsupported) {
                let marker = format!("Image fetched from {url}");
                // Off the runtime, as `file_read` does it: decoding and re-encoding a
                // multi-megapixel image is tens of milliseconds of pure CPU, and on the runtime it
                // blocks every other task on that worker.
                let marker = marker.clone();
                return unless_canceled(
                    &cancellation,
                    tokio::task::spawn_blocking(move || {
                        build_image_tool_output(&marker, sniffed, &body_bytes)
                    }),
                )
                .await?
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "web_fetch".to_string(),
                    message: format!("image decode task failed: {error}"),
                });
            }
        }

        let html = String::from_utf8_lossy(&body_bytes).into_owned();

        let raw = super::util::optional_bool(&input, "raw", "web_fetch")?.unwrap_or(false);
        // Parsing and converting up to ten megabytes of HTML is pure CPU, and on the runtime it
        // blocks every other task on that worker.
        let body = if raw {
            html
        } else {
            let document_url = document_url.clone();
            unless_canceled(
                &cancellation,
                tokio::task::spawn_blocking(move || html_to_markdown(&html, &document_url)),
            )
            .await?
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "web_fetch".to_string(),
                message: format!("HTML conversion task failed: {error}"),
            })?
        };

        // When the caller redirects to the scratchpad we produce full content regardless of
        // `limit`; the scratchpad is the overflow buffer.
        let limit = if redirects_to_scratchpad(&input) {
            0
        } else {
            super::util::optional_count(&input, "limit", "web_fetch")?
                .unwrap_or(DEFAULT_LIMIT_CHARS)
        };

        // The regex runs against the whole document, before any truncation: run after, `limit`
        // would decide which matches existed, and "No matches found" would read as a fact about
        // the page. The cap then applies to the match list.
        let matched = match optional_str(&input, "regex", "web_fetch")? {
            Some(pattern) => {
                let re = compile_user_regex(pattern, "web_fetch")?;
                let mut matches = re.find_iter(&body).map(|found| found.as_str());
                let Some(first) = matches.next() else {
                    return Ok(ToolOutput::text(
                        "No matches found for the given regex pattern.".to_string(),
                        false,
                    ));
                };
                let mut joined = first.to_string();
                let mut joined_chars = first.chars().count();
                // Past `limit` the rest is cut anyway, and a pattern that matches every character
                // of a large page would otherwise build a list as long as the page first.
                for found in matches {
                    if limit > 0 && joined_chars > limit {
                        break;
                    }
                    joined.push('\n');
                    joined.push_str(found);
                    joined_chars += 1 + found.chars().count();
                }
                Some(joined)
            }
            None => None,
        };
        let content = matched.unwrap_or(body);

        Ok(ToolOutput::text(truncate_to_chars(content, limit), false))
    }
}

/// `content` cut to its first `limit` characters with a note saying so, or whole when it fits or
/// `limit` is zero. Characters, as the schema promises: a count of bytes cut non-ASCII text far
/// short of the number asked for, and a small limit could return none of it.
fn truncate_to_chars(content: String, limit: usize) -> String {
    if limit == 0 {
        return content;
    }
    match content.char_indices().nth(limit) {
        Some((cut, _)) => format!(
            "{}\n\n... (truncated, showing first {limit} characters)",
            &content[..cut]
        ),
        None => content,
    }
}

/// `future`'s output, or [`MekaError::Interrupted`] the moment the turn is canceled, whichever
/// comes first. The future is dropped on a stop, which is what aborts a request in flight.
async fn unless_canceled<T>(
    cancellation: &tokio_util::sync::CancellationToken,
    future: impl std::future::Future<Output = T>,
) -> Result<T> {
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => Err(MekaError::Interrupted),
        output = future => Ok(output),
    }
}

/// The most a single HTTP response body may occupy in memory, decompressed.
const MAX_RESPONSE_BYTES: usize = 10 * crate::text::MIB;

/// Read a response body into memory, refusing to grow past [`MAX_RESPONSE_BYTES`].
///
/// Streamed rather than buffered through `text()` so the cap is checked incrementally: reqwest is
/// built with gzip, deflate and brotli, so a small compressed payload can expand into gigabytes,
/// and `text()` would have allocated all of it before anything could object. `Content-Length` is
/// checked first when the server offers one, which turns the common case into one refusal instead
/// of ten megabytes of reading.
async fn read_body_capped(response: reqwest::Response) -> Result<Vec<u8>> {
    if let Some(len) = response.content_length()
        && len as usize > MAX_RESPONSE_BYTES
    {
        return Err(MekaError::ToolExecution {
            tool_name: "web_fetch".to_string(),
            message: format!(
                "response Content-Length {len} exceeds cap {MAX_RESPONSE_BYTES} bytes"
            ),
        });
    }

    let mut body_bytes: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| MekaError::ToolExecution {
            tool_name: "web_fetch".to_string(),
            message: format!("failed to read response body: {error}"),
        })?;
        if body_bytes.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(MekaError::ToolExecution {
                tool_name: "web_fetch".to_string(),
                message: format!(
                    "response body exceeded {MAX_RESPONSE_BYTES} bytes during streaming \
                     (possible decompression bomb)"
                ),
            });
        }
        body_bytes.extend_from_slice(&chunk);
    }
    Ok(body_bytes)
}

#[cfg(test)]
mod tests {
    // The raw, un-pre-processed converter, used below to demonstrate the boilerplate-drop that
    // `html_to_markdown` works around. Production code goes through `html_to_markdown`.
    use html2md::rewrite_html;
    use regex::Regex;

    use super::*;

    #[test]
    fn nav_links_survive_markdown_conversion() {
        // `fast_html2md` drops the whole subtree of `<nav>` and `<footer>`, link text and href
        // included; the pre-pass rewrites those containers to `<div>` so the links survive.
        let html = r#"<nav class="x"><a href="/docs">Docs</a></nav>"#;
        assert_eq!(rewrite_html(html, false), "");
        let fixed = rewrite_html(&keep_boilerplate_container_content(html), false);
        assert!(fixed.contains("[Docs](/docs)"), "got: {fixed:?}");

        let footer = r#"<footer><a href="/terms">Terms</a></footer>"#;
        let fixed_footer = rewrite_html(&keep_boilerplate_container_content(footer), false);
        assert!(
            fixed_footer.contains("[Terms](/terms)"),
            "got: {fixed_footer:?}"
        );
    }

    #[test]
    fn keep_boilerplate_container_content_is_bounded() {
        // <navbar> / custom <nav-menu> share a prefix with <nav> but must not be rewritten.
        assert_eq!(
            keep_boilerplate_container_content("<navbar>x</navbar>"),
            "<navbar>x</navbar>"
        );
        assert_eq!(
            keep_boilerplate_container_content("<nav-menu>x</nav-menu>"),
            "<nav-menu>x</nav-menu>"
        );
        // Real nav/footer tags (with and without attributes) become div, preserving attributes.
        assert_eq!(
            keep_boilerplate_container_content(r#"<nav class="top"><a>x</a></nav>"#),
            r#"<div class="top"><a>x</a></div>"#
        );
        // <script> is still stripped by the converter even though we don't touch it here.
        let md = rewrite_html(
            &keep_boilerplate_container_content("<div>keep<script>var x=1;</script></div>"),
            false,
        );
        assert_eq!(md.trim(), "keep");
    }

    #[test]
    fn html_to_markdown_resolves_relative_links() {
        let html = r#"<a href="/docs">Docs</a>"#;
        // With a base URL, root-relative hrefs become absolute and followable.
        let base = url::Url::parse("https://example.test/").expect("base url");
        let absolute = html_to_markdown(html, &Some(base));
        assert!(
            absolute.contains("[Docs](https://example.test/docs)"),
            "got: {absolute:?}"
        );
        // Without a base URL, the href is preserved verbatim (still better than being dropped).
        let relative = html_to_markdown(html, &None);
        assert!(relative.contains("[Docs](/docs)"), "got: {relative:?}");
    }

    #[test]
    fn html_to_markdown_synthetic_page_end_to_end() {
        // Synthetic page (not a real site) exercising the full web_fetch conversion: a nav and a
        // footer holding the only links, plus a script that must be stripped. Mirrors the layout of
        // modern SPA sites where primary navigation lives in <nav>/<footer>.
        let html = r#"
            <!doctype html><html>
            <head><title>Widget Co</title><style>.a{color:red}</style></head>
            <body>
              <nav class="topbar">
                <a href="/">Home</a><a href="/products">Products</a><a href="/docs">Docs</a>
                <a href="https://app.widget.test">Launch</a>
              </nav>
              <main>
                <h1>Widget Co</h1>
                <p>Durable widgets. See our <a href="/pricing">pricing</a>.</p>
                <script>trackVisitor("secret-token");</script>
              </main>
              <footer>
                <a href="/legal/terms">Terms</a><a href="https://status.widget.test">Status</a>
              </footer>
            </body></html>
        "#;
        let base = url::Url::parse("https://widget.test/").expect("base url");
        let md = html_to_markdown(html, &Some(base));

        // Nav links survive the boilerplate strip and are resolved to absolute URLs.
        assert!(md.contains("[Home](https://widget.test/)"), "got: {md}");
        assert!(
            md.contains("[Products](https://widget.test/products)"),
            "got: {md}"
        );
        assert!(md.contains("[Docs](https://widget.test/docs)"), "got: {md}");
        // Footer links survive too.
        assert!(
            md.contains("[Terms](https://widget.test/legal/terms)"),
            "got: {md}"
        );
        // Body link resolved; absolute links pass through unchanged.
        assert!(
            md.contains("[pricing](https://widget.test/pricing)"),
            "got: {md}"
        );
        assert!(md.contains("[Launch](https://app.widget.test"), "got: {md}");
        assert!(
            md.contains("[Status](https://status.widget.test"),
            "got: {md}"
        );
        // Body text is kept; the script (and its payload) is dropped.
        assert!(md.contains("Widget Co"), "got: {md}");
        assert!(!md.contains("trackVisitor"), "script not stripped: {md}");
        assert!(!md.contains("secret-token"), "script not stripped: {md}");
    }

    #[test]
    fn build_web_client_defaults_succeeds() {
        let config = WebClientConfig::default();
        assert!(build_web_client(&config, false).is_ok());
    }

    #[test]
    fn build_web_client_with_socks_proxy_succeeds() {
        let config = WebClientConfig {
            proxy: Some("socks5h://127.0.0.1:1080".to_string()),
            ..WebClientConfig::default()
        };
        // We don't actually connect; we just verify reqwest accepts the proxy URL shape.
        assert!(build_web_client(&config, false).is_ok());
    }

    #[test]
    fn build_web_client_with_http_proxy_succeeds() {
        let config = WebClientConfig {
            proxy: Some("http://proxy.local:8080".to_string()),
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&config, false).is_ok());
    }

    #[test]
    fn build_web_client_explicit_none_proxy_succeeds() {
        // `"none"` → `.no_proxy()`, suppresses env-var auto-detection.
        let config = WebClientConfig {
            proxy: Some("none".to_string()),
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&config, false).is_ok());
    }

    #[test]
    fn build_web_client_rejects_bad_proxy() {
        let config = WebClientConfig {
            proxy: Some("not-a-url".to_string()),
            ..WebClientConfig::default()
        };
        let error = build_web_client(&config, false).expect_err("bad proxy URL should fail");
        let message = format!("{error}");
        assert!(
            message.contains("[web].proxy") && message.contains("not-a-url"),
            "expected proxy error naming the bad value, got: {message}"
        );
        // The class, not just the words: as `Config` this would be a refusal every host relays
        // verbatim, and `meka serve` would publish the operator's proxy URL in a 422.
        assert!(
            matches!(error, MekaError::Installation(_)),
            "a client meka cannot build is the operator's to fix: {error:?}"
        );
    }

    #[test]
    fn build_web_client_missing_ca_cert_errors() {
        let config = WebClientConfig {
            ca_cert_file: Some(std::path::PathBuf::from(
                "/definitely/does/not/exist/ca.pem",
            )),
            ..WebClientConfig::default()
        };
        let error = build_web_client(&config, false).expect_err("missing CA file should fail");
        let message = format!("{error}");
        assert!(
            message.contains("[web].ca_cert_file")
                && message.contains("/definitely/does/not/exist"),
            "expected CA error naming the path, got: {message}"
        );
        // See `build_web_client_rejects_bad_proxy`: the path in that message is exactly why the
        // class matters.
        assert!(
            matches!(error, MekaError::Installation(_)),
            "a CA file the operator named is the operator's to fix: {error:?}"
        );
    }

    #[test]
    fn build_web_client_non_pem_ca_cert_errors() {
        // An existing but non-PEM file produces a clear parse error rather than a silent failure.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not-a-cert.bin");
        std::fs::write(&path, b"this is definitely not a PEM").expect("write");
        let config = WebClientConfig {
            ca_cert_file: Some(path),
            ..WebClientConfig::default()
        };
        let error = build_web_client(&config, false).expect_err("non-PEM CA file should fail");
        let message = format!("{error}");
        assert!(
            message.contains("[web].ca_cert_file"),
            "expected CA error, got: {message}"
        );
    }

    #[test]
    fn build_web_client_zero_redirects_builds() {
        // max_redirects = 0 → Policy::none(); reqwest accepts it.
        let config = WebClientConfig {
            max_redirects: 0,
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&config, false).is_ok());
    }

    #[test]
    fn build_web_client_https_only_builds() {
        let config = WebClientConfig {
            https_only: true,
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&config, false).is_ok());
    }

    #[test]
    fn build_web_client_with_min_tls_1_2_succeeds() {
        let config = WebClientConfig {
            min_tls_version: Some(MinTlsVersion::V1_2),
            ..WebClientConfig::default()
        };
        // rustls (our pinned backend) supports TLS 1.2; must build.
        assert!(build_web_client(&config, false).is_ok());
    }

    #[test]
    fn build_web_client_with_danger_flags_builds() {
        // The `warn!` per flag is not asserted: tracing capture would add plumbing for negligible
        // value.
        let config = WebClientConfig {
            danger_accept_invalid_certs: true,
            danger_accept_invalid_hostnames: true,
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&config, false).is_ok());
    }

    #[test]
    fn apply_headers_adds_headers() {
        let client = reqwest::Client::new();
        let input = serde_json::json!({
            "url": "https://example.com",
            "headers": {
                "X-Custom": "test-value",
                "Accept-Language": "en-US"
            }
        });
        let request = apply_headers(client.get("https://example.com"), &input)
            .build()
            .unwrap();
        assert_eq!(request.headers().get("X-Custom").unwrap(), "test-value");
        assert_eq!(request.headers().get("Accept-Language").unwrap(), "en-US");
    }

    #[test]
    fn apply_headers_overrides_user_agent() {
        let client = reqwest::Client::builder()
            .user_agent("default-agent")
            .build()
            .unwrap();
        let input = serde_json::json!({
            "headers": { "User-Agent": "custom-agent" }
        });
        let request = apply_headers(client.get("https://example.com"), &input)
            .build()
            .unwrap();
        assert_eq!(request.headers().get("User-Agent").unwrap(), "custom-agent");
    }

    #[test]
    fn apply_headers_no_headers() {
        let client = reqwest::Client::new();
        let input = serde_json::json!({"url": "https://example.com"});
        let request = apply_headers(client.get("https://example.com"), &input)
            .build()
            .unwrap();
        assert!(request.headers().get("X-Custom").is_none());
    }

    #[test]
    fn apply_headers_skips_non_string_values() {
        let client = reqwest::Client::new();
        let input = serde_json::json!({
            "headers": {
                "X-Valid": "good",
                "X-Invalid": 123
            }
        });
        let request = apply_headers(client.get("https://example.com"), &input)
            .build()
            .unwrap();
        assert_eq!(request.headers().get("X-Valid").unwrap(), "good");
        assert!(request.headers().get("X-Invalid").is_none());
    }

    #[test]
    fn regex_filters_content() {
        let content = "Hello world\nfoo 123 bar\nbaz 456 qux\nend";
        let re = Regex::new(r"\d+").unwrap();
        let matches: Vec<&str> = re.find_iter(content).map(|m| m.as_str()).collect();
        assert_eq!(matches.join("\n"), "123\n456");
    }

    #[test]
    fn regex_no_matches() {
        let content = "Hello world";
        let re = Regex::new(r"\d+").unwrap();
        let matches: Vec<&str> = re.find_iter(content).map(|m| m.as_str()).collect();
        assert!(matches.is_empty());
    }

    #[test]
    #[allow(
        clippy::invalid_regex,
        reason = "intentionally invalid: tests parser rejection"
    )]
    fn regex_invalid_pattern() {
        assert!(Regex::new(r"[invalid").is_err());
    }

    #[test]
    fn fetch_url_definition_has_headers_regex_and_raw() {
        let tool = FetchUrlTool {
            refuse_private_destinations: false,

            refuse_own_addresses: false,
            client: reqwest::Client::new(),
        };
        let def = tool.definition();
        let props = &def.parameters["properties"];
        assert!(props.get("headers").is_some());
        assert!(props.get("regex").is_some());
        assert!(props.get("raw").is_some());
    }

    /// The agent awaits every tool future, so a fetch that ignores the turn's token holds a
    /// canceled turn until the peer answers or `[web] request_timeout` runs out. The peer here
    /// accepts the connection and never answers.
    #[tokio::test]
    async fn a_fetch_stops_when_the_turn_is_canceled() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        let peer = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.expect("accept");
            std::future::pending::<()>().await;
        });
        let tool = FetchUrlTool {
            refuse_private_destinations: false,

            refuse_own_addresses: false,
            client: build_web_client(&WebClientConfig::default(), false).expect("client"),
        };
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
            tool.execute(
                serde_json::json!({ "url": format!("http://{address}/") }),
                crate::tools::ToolContext::detached(cancellation),
            ),
        )
        .await
        .expect("the stop must end the fetch, not the test's own deadline");
        stop.await.expect("the stop landed");
        assert!(
            matches!(outcome, Err(MekaError::Interrupted)),
            "a stop is reported as the interruption it is: {outcome:?}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "the stop must end the fetch at once: {:?}",
            started.elapsed()
        );
        peer.abort();
    }

    #[test]
    fn the_limit_counts_characters_not_bytes() {
        assert_eq!(
            truncate_to_chars("你好吗".to_string(), 2),
            "你好\n\n... (truncated, showing first 2 characters)"
        );
        assert_eq!(truncate_to_chars("你好吗".to_string(), 3), "你好吗");
        assert_eq!(
            truncate_to_chars("abcd".to_string(), 2),
            "ab\n\n... (truncated, showing first 2 characters)"
        );
        assert_eq!(truncate_to_chars("abcd".to_string(), 0), "abcd");
    }

    /// A canary on the response cap, which `web_fetch` reads through. It catches an accidental
    /// bump in either direction; end-to-end coverage of the streaming check itself needs a real
    /// server and lives in the manual verification step.
    #[test]
    fn fetch_url_size_cap_is_10_mib() {
        assert_eq!(MAX_RESPONSE_BYTES, 10_485_760);
    }

    #[test]
    fn redirects_to_scratchpad_logic() {
        // Mirrors the branch used in `web_fetch`'s execute. When redirecting, we force `limit` to 0
        // (unlimited).
        let with = serde_json::json!({ "scratchpad": "out", "limit": 100 });
        let without = serde_json::json!({ "limit": 100 });
        assert!(redirects_to_scratchpad(&with));
        assert!(!redirects_to_scratchpad(&without));
    }

    /// What `[web].refuse_private_destinations` keeps a fetch from: loopback, private, link-local,
    /// carrier-grade NAT, unspecified and broadcast, in both families and through a mapped
    /// address; a public address is admitted.
    #[test]
    fn a_private_destination_is_every_address_a_fetch_must_not_reach() {
        use std::net::IpAddr;
        for refused in [
            "127.0.0.1",
            "127.9.9.9",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "255.255.255.255",
            "::1",
            "::",
            "fc00::1",
            "fe80::1",
            "::ffff:10.0.0.1",
        ] {
            let address: IpAddr = refused.parse().expect("an address");
            assert!(is_private_destination(address), "{refused} must be refused");
        }
        for admitted in ["93.184.216.34", "1.1.1.1", "2606:4700:4700::1111"] {
            let address: IpAddr = admitted.parse().expect("an address");
            assert!(
                !is_private_destination(address),
                "{admitted} must be admitted"
            );
        }
    }

    /// What `[permissions].network = "remote"` keeps a fetch from: this machine's own addresses,
    /// loopback and unspecified in both families, a mapped loopback, and an address an interface
    /// here holds; a public address and a private one no interface here holds are admitted, which
    /// is the difference from the private switch.
    #[test]
    fn an_own_address_is_one_this_machine_holds() {
        use std::net::IpAddr;
        for refused in [
            "127.0.0.1",
            "127.9.9.9",
            "0.0.0.0",
            "::1",
            "::",
            "::ffff:127.0.0.1",
        ] {
            let address: IpAddr = refused.parse().expect("an address");
            assert!(is_own_address(address), "{refused} must be refused");
        }
        for admitted in ["93.184.216.34", "2606:4700:4700::1111", "10.255.255.254"] {
            let address: IpAddr = admitted.parse().expect("an address");
            assert!(!is_own_address(address), "{admitted} must be admitted");
        }
        // The address this machine would send from, learned without sending: a route is needed
        // to learn it, and a host without one has nothing further to assert.
        let outbound = std::net::UdpSocket::bind("0.0.0.0:0")
            .and_then(|socket| socket.connect("1.1.1.1:53").map(|()| socket))
            .and_then(|socket| socket.local_addr())
            .map(|address| address.ip());
        match outbound {
            Ok(address) if !address.is_unspecified() => {
                assert!(is_own_address(address), "{address} is this machine's");
            }
            _ => eprintln!("skipping the interface leg: no route to learn an address from"),
        }
    }

    /// `web_fetch` is withheld under `none` alone; under `remote` it is registered and refuses
    /// this machine's addresses.
    #[test]
    fn web_fetch_is_registered_unless_the_network_is_none() {
        use crate::config::Network;
        assert!(web_fetch_registered(Network::Host));
        assert!(web_fetch_registered(Network::Remote));
        assert!(!web_fetch_registered(Network::None));
        assert!(!web_fetch_refuses_own_addresses(Network::Host));
        assert!(web_fetch_refuses_own_addresses(Network::Remote));
    }

    /// The own-address guard refuses a loopback literal and `localhost` naming the switch, and
    /// admits a private address no interface here holds, on both the resolver door and the
    /// literal door.
    #[tokio::test]
    async fn the_own_address_guard_judges_a_name_by_its_addresses() {
        use reqwest::dns::Resolve;
        let guard = DestinationGuard {
            refuse_private: false,
            refuse_own: true,
        };
        for refused in ["127.0.0.1", "localhost"] {
            let name: reqwest::dns::Name = refused.parse().expect("a name");
            let error = guard
                .resolve(name)
                .await
                .err()
                .unwrap_or_else(|| panic!("{refused} must be refused"))
                .to_string();
            assert!(
                error.contains(refused) && error.contains("`[permissions].network = \"remote\"`"),
                "{error}"
            );
        }
        let name: reqwest::dns::Name = "10.255.255.254".parse().expect("a name");
        assert!(
            guard.resolve(name).await.is_ok(),
            "a private address this machine does not hold stays reachable under `remote`"
        );
        // The address this machine would send from, learned without sending; a host with no route
        // has nothing further to assert.
        let outbound = std::net::UdpSocket::bind("0.0.0.0:0")
            .and_then(|socket| socket.connect("1.1.1.1:53").map(|()| socket))
            .and_then(|socket| socket.local_addr())
            .map(|address| address.ip());
        if let Ok(address) = outbound
            && !address.is_unspecified()
        {
            let name: reqwest::dns::Name = address.to_string().parse().expect("a name");
            let error = guard
                .resolve(name)
                .await
                .err()
                .unwrap_or_else(|| panic!("{address}, this machine's own, must be refused"))
                .to_string();
            assert!(
                error.contains("`[permissions].network = \"remote\"`"),
                "{error}"
            );
        }
        let refused = refused_literal_destination("http://127.0.0.1:1/x", guard);
        assert_eq!(
            refused.map(|(address, switch)| (address.to_string(), switch)),
            Some((
                "127.0.0.1".to_string(),
                "`[permissions].network = \"remote\"`"
            ))
        );
        assert!(refused_literal_destination("http://10.255.255.254/x", guard).is_none());
        assert!(refused_literal_destination("http://example.com/x", guard).is_none());
    }

    /// The guard refuses a name on its addresses, a literal and `localhost` alike, with the
    /// address in the refusal, and admits a public literal.
    #[tokio::test]
    async fn the_private_destination_guard_judges_a_name_by_its_addresses() {
        use reqwest::dns::Resolve;
        let guard = DestinationGuard {
            refuse_private: true,
            refuse_own: false,
        };
        for refused in ["127.0.0.1", "localhost", "10.0.0.7"] {
            let name: reqwest::dns::Name = refused.parse().expect("a name");
            let error = guard
                .resolve(name)
                .await
                .err()
                .unwrap_or_else(|| panic!("{refused} must be refused"))
                .to_string();
            assert!(
                error.contains(refused) && error.contains("refuse_private_destinations"),
                "{error}"
            );
        }
        let name: reqwest::dns::Name = "93.184.216.34".parse().expect("a name");
        let addresses: Vec<std::net::SocketAddr> = guard
            .resolve(name)
            .await
            .expect("a public literal is admitted")
            .collect();
        assert_eq!(addresses.len(), 1);
        assert_eq!(addresses[0].ip().to_string(), "93.184.216.34");
    }

    /// The switch holds both doors: `web_fetch` refuses a URL naming a loopback literal before
    /// any connection is made, and the client refuses a redirect hop naming one; without the
    /// switch the same fetch reaches the listener.
    #[tokio::test]
    async fn the_switch_keeps_a_fetch_off_a_loopback_listener() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listen");
        let origin = format!("http://{}", listener.local_addr().expect("address"));
        let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        tokio::spawn({
            let accepted = accepted.clone();
            let origin = origin.clone();
            async move {
                while let Ok((mut socket, _)) = listener.accept().await {
                    accepted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let mut request = [0u8; 1024];
                    let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut request).await;
                    let reply = if request.starts_with(b"GET /hop ") {
                        format!(
                            "HTTP/1.1 302 Found\r\nlocation: {origin}/private\r\ncontent-length: 0\r\n\r\n"
                        )
                    } else {
                        "HTTP/1.1 204 No Content\r\ncontent-length: 0\r\n\r\n".to_string()
                    };
                    let _ =
                        tokio::io::AsyncWriteExt::write_all(&mut socket, reply.as_bytes()).await;
                }
            }
        });
        // Both switches hold both doors the same way; each is tried on its own.
        for (refuse_private, refuse_own, switch) in [
            (true, false, "refuse_private_destinations"),
            (false, true, "`[permissions].network = \"remote\"`"),
        ] {
            let config = WebClientConfig {
                refuse_private_destinations: refuse_private,
                ..WebClientConfig::default()
            };
            let tool = FetchUrlTool {
                refuse_private_destinations: refuse_private,
                refuse_own_addresses: refuse_own,
                client: build_web_client(&config, refuse_own).expect("client"),
            };
            let error = tool
                .execute(
                    serde_json::json!({ "url": format!("{origin}/") }),
                    crate::tools::ToolContext::detached(tokio_util::sync::CancellationToken::new()),
                )
                .await
                .expect_err("the tool refuses the literal")
                .to_string();
            assert!(
                error.contains("names 127.0.0.1") && error.contains(switch),
                "{error}"
            );
            assert_eq!(
                accepted.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "nothing may have connected"
            );

            // The client's own door, reached without the tool's: the first hop is accepted, the
            // redirect to a loopback literal is not.
            let error = build_web_client(&config, refuse_own)
                .expect("client")
                .get(format!("{origin}/hop"))
                .send()
                .await
                .expect_err("the hop is refused");
            let chain = std::iter::successors(Some(&error as &dyn std::error::Error), |error| {
                error.source()
            })
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(": ");
            assert!(chain.contains(switch), "{chain}");
            assert_eq!(accepted.swap(0, std::sync::atomic::Ordering::SeqCst), 1);
        }

        let open = build_web_client(&WebClientConfig::default(), false).expect("client");
        let response = open
            .get(format!("{origin}/"))
            .send()
            .await
            .expect("reaches the listener");
        assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
    }
}
