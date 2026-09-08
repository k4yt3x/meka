//! The web tool: `fetch_url`, an HTTP GET with HTML-to-markdown conversion or a multimodal image
//! return.

use std::sync::LazyLock;

use async_trait::async_trait;
use futures::StreamExt;
use html2md::rewrite_html_custom_with_url;

use super::{
    Tool, ToolOutput,
    util::{compile_user_regex, redirects_to_scratchpad, require_str},
};
use crate::{
    config::{MinTlsVersion, WebClientConfig},
    error::{MekaError, Result},
    image::{ImageHandling, classify_content_type},
    permission::Permission,
    provider::ToolDefinition,
    tools::util::build_image_tool_output,
};

/// Build the `reqwest::Client` for `fetch_url` from the resolved
/// [`WebClientConfig`].
///
/// Refused as [`MekaError::Installation`] rather than built from a fallback that ignores the
/// user's intent. `crate::host::build_shared_deps` builds one at startup, which is what makes a bad
/// proxy URL or CA file fail the process with the message at the terminal before any session
/// exists; each session's registry then builds its own from the same settings. The message names
/// a path or a URL out of `config.toml`, which is why its class keeps it off the wire on the hosts
/// a remote caller reaches.
pub(crate) fn build_web_client(config: &WebClientConfig) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .user_agent(&config.user_agent)
        .timeout(config.request_timeout);

    if let Some(timeout) = config.connect_timeout {
        builder = builder.connect_timeout(timeout);
    }
    if let Some(timeout) = config.read_timeout {
        builder = builder.read_timeout(timeout);
    }

    // `0` → no redirects at all (Policy::none). Any non-zero cap maps to Policy::limited(n).
    let policy = if config.max_redirects == 0 {
        reqwest::redirect::Policy::none()
    } else {
        reqwest::redirect::Policy::limited(config.max_redirects)
    };
    builder = builder.redirect(policy);

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

/// Convert fetched HTML to Markdown exactly the way `fetch_url` does. Two steps: rewrite `<nav>` /
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
    pub(crate) client: reqwest::Client,
}

#[async_trait]
impl Tool for FetchUrlTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_url".to_string(),
            description: "Fetch a web page and return its content as markdown. Set 'raw' \
                          to true to return untreated HTML. If the URL resolves to a \
                          supported raster image (PNG, JPEG, GIF, WebP, BMP, TIFF, \
                          ICO, HDR, EXR, TGA, PNM, QOI, DDS, or Farbfeld), the image \
                          is returned as a multimodal content block directly. \
                          Non-native formats are transparently converted to PNG. \
                          `limit`, `regex`, and `raw` do not apply to image \
                          responses. Only fetch image URLs if the current model \
                          supports vision input."
                .to_string(),
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
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let url = require_str(&input, "url", "fetch_url")?;

        let request = apply_headers(self.client.get(&url), &input);
        let response = request
            .send()
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "fetch_url".to_string(),
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

        let body_bytes = read_body_capped(response).await?;

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
                // Off the runtime, as `read_file` does it: decoding and re-encoding a
                // multi-megapixel image is tens of milliseconds of pure CPU, and on the runtime it
                // blocks every other task on that worker.
                let marker = marker.clone();
                return tokio::task::spawn_blocking(move || {
                    build_image_tool_output(&marker, sniffed, &body_bytes)
                })
                .await
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "fetch_url".to_string(),
                    message: format!("image decode task failed: {error}"),
                });
            }
        }

        let html = String::from_utf8_lossy(&body_bytes).into_owned();

        let raw = input["raw"].as_bool().unwrap_or(false);
        // Parsing and converting up to ten megabytes of HTML is pure CPU, and on the runtime it
        // blocks every other task on that worker.
        let body = if raw {
            html
        } else {
            let document_url = document_url.clone();
            tokio::task::spawn_blocking(move || html_to_markdown(&html, &document_url))
                .await
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "fetch_url".to_string(),
                    message: format!("HTML conversion task failed: {error}"),
                })?
        };

        // When the caller redirects to the scratchpad we produce full content regardless of
        // `limit`; the scratchpad is the overflow buffer.
        let limit = if redirects_to_scratchpad(&input) {
            0
        } else {
            input["limit"]
                .as_u64()
                .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
                .unwrap_or(DEFAULT_LIMIT_CHARS)
        };

        // The regex runs against the whole document, before any truncation: run after, `limit`
        // would decide which matches existed, and "No matches found" would read as a fact about
        // the page. The cap then applies to the match list.
        let matched = match input.get("regex").and_then(|value| value.as_str()) {
            Some(pattern) => {
                let re = compile_user_regex(pattern, "fetch_url")?;
                let matches: Vec<&str> = re.find_iter(&body).map(|found| found.as_str()).collect();
                if matches.is_empty() {
                    return Ok(ToolOutput::text(
                        "No matches found for the given regex pattern.".to_string(),
                        false,
                    ));
                }
                Some(matches.join("\n"))
            }
            None => None,
        };
        let content = matched.unwrap_or(body);

        let content = if limit > 0 && content.len() > limit {
            format!(
                "{}\n\n... (truncated, showing first {limit} characters)",
                &content[..content.floor_char_boundary(limit)],
            )
        } else {
            content
        };

        Ok(ToolOutput::text(content, false))
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
            tool_name: "fetch_url".to_string(),
            message: format!(
                "response Content-Length {len} exceeds cap {MAX_RESPONSE_BYTES} bytes"
            ),
        });
    }

    let mut body_bytes: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| MekaError::ToolExecution {
            tool_name: "fetch_url".to_string(),
            message: format!("failed to read response body: {error}"),
        })?;
        if body_bytes.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(MekaError::ToolExecution {
                tool_name: "fetch_url".to_string(),
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
        // Synthetic page (not a real site) exercising the full fetch_url conversion: a nav and a
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
        assert!(build_web_client(&config).is_ok());
    }

    #[test]
    fn build_web_client_with_socks_proxy_succeeds() {
        let config = WebClientConfig {
            proxy: Some("socks5h://127.0.0.1:1080".to_string()),
            ..WebClientConfig::default()
        };
        // We don't actually connect; we just verify reqwest accepts the proxy URL shape.
        assert!(build_web_client(&config).is_ok());
    }

    #[test]
    fn build_web_client_with_http_proxy_succeeds() {
        let config = WebClientConfig {
            proxy: Some("http://proxy.local:8080".to_string()),
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&config).is_ok());
    }

    #[test]
    fn build_web_client_explicit_none_proxy_succeeds() {
        // `"none"` → `.no_proxy()`, suppresses env-var auto-detection.
        let config = WebClientConfig {
            proxy: Some("none".to_string()),
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&config).is_ok());
    }

    #[test]
    fn build_web_client_rejects_bad_proxy() {
        let config = WebClientConfig {
            proxy: Some("not-a-url".to_string()),
            ..WebClientConfig::default()
        };
        let error = build_web_client(&config).expect_err("bad proxy URL should fail");
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
        let error = build_web_client(&config).expect_err("missing CA file should fail");
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
        let error = build_web_client(&config).expect_err("non-PEM CA file should fail");
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
        assert!(build_web_client(&config).is_ok());
    }

    #[test]
    fn build_web_client_https_only_builds() {
        let config = WebClientConfig {
            https_only: true,
            ..WebClientConfig::default()
        };
        assert!(build_web_client(&config).is_ok());
    }

    #[test]
    fn build_web_client_with_min_tls_1_2_succeeds() {
        let config = WebClientConfig {
            min_tls_version: Some(MinTlsVersion::V1_2),
            ..WebClientConfig::default()
        };
        // rustls (our pinned backend) supports TLS 1.2; must build.
        assert!(build_web_client(&config).is_ok());
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
        assert!(build_web_client(&config).is_ok());
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
            client: reqwest::Client::new(),
        };
        let def = tool.definition();
        let props = &def.parameters["properties"];
        assert!(props.get("headers").is_some());
        assert!(props.get("regex").is_some());
        assert!(props.get("raw").is_some());
    }

    /// A canary on the response cap, which `fetch_url` reads through. It catches an accidental
    /// bump in either direction; end-to-end coverage of the streaming check itself needs a real
    /// server and lives in the manual verification step.
    #[test]
    fn fetch_url_size_cap_is_10_mib() {
        assert_eq!(MAX_RESPONSE_BYTES, 10_485_760);
    }

    #[test]
    fn redirects_to_scratchpad_logic() {
        // Mirrors the branch used in fetch_url::execute. When redirecting, we force `limit` to 0
        // (unlimited).
        let with = serde_json::json!({ "scratchpad": "out", "limit": 100 });
        let without = serde_json::json!({ "limit": 100 });
        assert!(redirects_to_scratchpad(&with));
        assert!(!redirects_to_scratchpad(&without));
    }
}
