use std::sync::LazyLock;

use async_trait::async_trait;
use html2md::rewrite_html;
use regex::Regex;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::image::{ImageHandling, build_image_tool_output, classify_content_type};
use crate::permission::Permission;
use crate::provider::ToolDefinition;

use super::util::{redirects_to_scratchpad, require_str};
use super::{Tool, ToolOutput};

// Static CSS selectors for search result parsing (parsed once, reused on every call).
static DDG_RESULT: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse(".result").expect("static CSS selector"));
static DDG_LINK: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse("a.result__a").expect("static CSS selector"));
static DDG_SNIPPET: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse(".result__snippet").expect("static CSS selector"));
static GOOGLE_BLOCK: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse("div.g").expect("static CSS selector"));
static GOOGLE_TITLE: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse("h3").expect("static CSS selector"));
static GOOGLE_LINK: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse("a[href]").expect("static CSS selector"));
static BING_BLOCK: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse(".b_algo").expect("static CSS selector"));
static BING_TITLE: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse("h2 a").expect("static CSS selector"));
static BING_SNIPPET: LazyLock<scraper::Selector> =
    LazyLock::new(|| scraper::Selector::parse(".b_caption p").expect("static CSS selector"));

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
    pub client: reqwest::Client,
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
                          is returned as a multimodal content block directly — \
                          non-native formats are transparently converted to PNG. \
                          `max_length`, `regex`, and `raw` do not apply to image \
                          responses. Only fetch image URLs if the current model \
                          supports vision input."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum number of characters to return. Default: 30000. Set to 0 for no limit."
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
                        "description": "If true, return raw HTML instead of converting to markdown. Defaults to false."
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["url"]
            }),
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let url = require_str(&input, "url", "fetch_url")?;

        let request = apply_headers(self.client.get(&url), &input);
        let response = request
            .send()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("failed to fetch '{}': {}", url, error),
            })?;

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolOutput::text(
                format!("HTTP {} for '{}'", status, url),
                true,
            ));
        }

        // If the response is a supported image, return a multimodal Image
        // block directly rather than running the binary body through html2md.
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();

        let handling = classify_content_type(&content_type);
        if !matches!(handling, ImageHandling::Unsupported) {
            let bytes = response
                .bytes()
                .await
                .map_err(|error| AgshError::ToolExecution {
                    tool_name: "fetch_url".to_string(),
                    message: format!("failed to read image bytes: {}", error),
                })?;

            let marker = format!("Image fetched from {}", url);
            return Ok(build_image_tool_output(&marker, handling, &bytes));
        }

        let html = response
            .text()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("failed to read response body: {}", error),
            })?;

        let raw = input["raw"].as_bool().unwrap_or(false);
        let body = if raw {
            html
        } else {
            rewrite_html(&html, false)
        };

        // When the caller redirects to the scratchpad we produce full content
        // regardless of max_length — the scratchpad is the overflow buffer.
        let max_length = if redirects_to_scratchpad(&input) {
            0
        } else {
            input["max_length"]
                .as_u64()
                .map(|value| value as usize)
                .unwrap_or(30000)
        };

        let content = if max_length > 0 && body.len() > max_length {
            format!(
                "{}\n\n... (truncated, showing first {} characters)",
                &body[..body.floor_char_boundary(max_length)],
                max_length
            )
        } else {
            body
        };

        let content = if let Some(pattern) = input.get("regex").and_then(|v| v.as_str()) {
            let re = Regex::new(pattern).map_err(|error| AgshError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("invalid regex '{}': {}", pattern, error),
            })?;
            let matches: Vec<&str> = re.find_iter(&content).map(|m| m.as_str()).collect();
            if matches.is_empty() {
                "No matches found for the given regex pattern.".to_string()
            } else {
                matches.join("\n")
            }
        } else {
            content
        };

        Ok(ToolOutput::text(content, false))
    }
}

pub(super) struct WebSearchTool {
    pub client: reqwest::Client,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web and return results. Supports DuckDuckGo (default), \
                Google, and Bing search engines."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "engine": {
                        "type": "string",
                        "description": "Search engine to use: 'duckduckgo' (default), 'google', or 'bing'",
                        "enum": ["duckduckgo", "google", "bing"]
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers. Overrides defaults (e.g., User-Agent).",
                        "additionalProperties": { "type": "string" }
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let query = require_str(&input, "query", "web_search")?;
        let engine = input
            .get("engine")
            .and_then(|engine| engine.as_str())
            .unwrap_or("duckduckgo");

        let (url, query_params): (&str, Vec<(&str, &str)>) = match engine {
            "google" => (
                "https://www.google.com/search",
                vec![("q", query.as_str()), ("hl", "en")],
            ),
            "bing" => ("https://www.bing.com/search", vec![("q", query.as_str())]),
            _ => (
                "https://html.duckduckgo.com/html/",
                vec![("q", query.as_str())],
            ),
        };

        let request = apply_headers(self.client.get(url).query(&query_params), &input);
        let response = request
            .send()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "web_search".to_string(),
                message: format!("search request failed: {}", error),
            })?;

        let html = response
            .text()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "web_search".to_string(),
                message: format!("failed to read search response: {}", error),
            })?;

        let results = match engine {
            "google" => parse_google_results(&html),
            "bing" => parse_bing_results(&html),
            _ => parse_duckduckgo_results(&html),
        };

        if results.is_empty() {
            Ok(ToolOutput::text(
                "No search results found.".to_string(),
                false,
            ))
        } else {
            Ok(ToolOutput::text(results, false))
        }
    }
}

fn parse_duckduckgo_results(html: &str) -> String {
    let document = scraper::Html::parse_document(html);
    let mut results = Vec::new();

    for block in document.select(&DDG_RESULT) {
        let link = match block.select(&DDG_LINK).next() {
            Some(link) => link,
            None => continue,
        };

        let title: String = link.text().collect();
        if title.trim().is_empty() {
            continue;
        }

        let url = link.value().attr("href").map(|href| {
            if let Some(pos) = href.find("uddg=") {
                let encoded = &href[pos + 5..];
                let replaced = encoded.replace('+', " ");
                let decoded = percent_encoding::percent_decode_str(&replaced).decode_utf8_lossy();
                decoded.split('&').next().unwrap_or(&decoded).to_string()
            } else {
                href.to_string()
            }
        });

        let snippet: Option<String> = block
            .select(&DDG_SNIPPET)
            .next()
            .map(|element| element.text().collect());

        let mut result_text = format!("{}. **{}**", results.len() + 1, title.trim());
        if let Some(url) = &url {
            result_text.push_str(&format!("\n   URL: {}", url));
        }
        if let Some(snippet) = &snippet {
            let snippet = snippet.trim();
            if !snippet.is_empty() {
                result_text.push_str(&format!("\n   {}", snippet));
            }
        }
        results.push(result_text);
        if results.len() >= 10 {
            break;
        }
    }

    results.join("\n\n")
}

fn parse_google_results(html: &str) -> String {
    let document = scraper::Html::parse_document(html);
    let mut results = Vec::new();

    for block in document.select(&GOOGLE_BLOCK) {
        let title: String = match block.select(&GOOGLE_TITLE).next() {
            Some(h3) => h3.text().collect(),
            None => continue,
        };
        if title.trim().is_empty() {
            continue;
        }

        let url = block.select(&GOOGLE_LINK).find_map(|a| {
            a.value()
                .attr("href")
                .filter(|href| href.starts_with("http"))
                .map(|href| href.to_string())
        });

        let mut result_text = format!("{}. **{}**", results.len() + 1, title.trim());
        if let Some(url) = &url {
            result_text.push_str(&format!("\n   URL: {}", url));
        }
        results.push(result_text);
        if results.len() >= 10 {
            break;
        }
    }

    results.join("\n\n")
}

fn parse_bing_results(html: &str) -> String {
    let document = scraper::Html::parse_document(html);
    let mut results = Vec::new();

    for block in document.select(&BING_BLOCK) {
        let title_element = match block.select(&BING_TITLE).next() {
            Some(a) => a,
            None => continue,
        };
        let title: String = title_element.text().collect();
        if title.trim().is_empty() {
            continue;
        }

        let url = title_element
            .value()
            .attr("href")
            .filter(|href| href.starts_with("http"))
            .map(|href| href.to_string());

        let snippet: Option<String> = block
            .select(&BING_SNIPPET)
            .next()
            .map(|p| p.text().collect());

        let mut result_text = format!("{}. **{}**", results.len() + 1, title.trim());
        if let Some(url) = &url {
            result_text.push_str(&format!("\n   URL: {}", url));
        }
        if let Some(snippet) = &snippet {
            let snippet = snippet.trim();
            if !snippet.is_empty() {
                result_text.push_str(&format!("\n   {}", snippet));
            }
        }
        results.push(result_text);
        if results.len() >= 10 {
            break;
        }
    }

    results.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duckduckgo_results() {
        let html = r#"<html><body>
            <div class="result">
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1&rut=x">
                    First Result
                </a>
                <a class="result__snippet">First snippet text.</a>
            </div>
            <div class="result">
                <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpage2&rut=y">
                    Second Result
                </a>
                <a class="result__snippet">Second snippet text.</a>
            </div>
        </body></html>"#;
        let results = parse_duckduckgo_results(html);
        assert!(results.contains("First Result"));
        assert!(results.contains("Second Result"));
        assert!(results.contains("example.com/page1"));
        assert!(results.contains("example.com/page2"));
        assert!(results.contains("First snippet text."));
        assert!(results.contains("Second snippet text."));
    }

    #[test]
    fn test_parse_google_results() {
        let html = r#"<html><body>
            <div class="g">
                <a href="https://example.com/page1"><h3>First Google Result</h3></a>
            </div>
            <div class="g">
                <a href="https://example.com/page2"><h3>Second Google Result</h3></a>
            </div>
        </body></html>"#;
        let results = parse_google_results(html);
        assert!(results.contains("First Google Result"));
        assert!(results.contains("Second Google Result"));
        assert!(results.contains("example.com/page1"));
        assert!(results.contains("example.com/page2"));
    }

    #[test]
    fn test_parse_bing_results() {
        let html = r#"<html><body>
            <li class="b_algo">
                <h2><a href="https://example.com/page1">First Bing Result</a></h2>
                <div class="b_caption"><p>First Bing snippet.</p></div>
            </li>
            <li class="b_algo">
                <h2><a href="https://example.com/page2">Second Bing Result</a></h2>
                <div class="b_caption"><p>Second Bing snippet.</p></div>
            </li>
        </body></html>"#;
        let results = parse_bing_results(html);
        assert!(results.contains("First Bing Result"));
        assert!(results.contains("Second Bing Result"));
        assert!(results.contains("example.com/page1"));
        assert!(results.contains("example.com/page2"));
        assert!(results.contains("First Bing snippet."));
        assert!(results.contains("Second Bing snippet."));
    }

    #[test]
    fn test_parse_empty_results() {
        assert!(parse_duckduckgo_results("<html><body></body></html>").is_empty());
        assert!(parse_google_results("<html><body></body></html>").is_empty());
        assert!(parse_bing_results("<html><body></body></html>").is_empty());
    }

    #[test]
    fn test_apply_headers_adds_headers() {
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
    fn test_apply_headers_overrides_user_agent() {
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
    fn test_apply_headers_no_headers() {
        let client = reqwest::Client::new();
        let input = serde_json::json!({"url": "https://example.com"});
        let request = apply_headers(client.get("https://example.com"), &input)
            .build()
            .unwrap();
        assert!(request.headers().get("X-Custom").is_none());
    }

    #[test]
    fn test_apply_headers_skips_non_string_values() {
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
    fn test_regex_filters_content() {
        let content = "Hello world\nfoo 123 bar\nbaz 456 qux\nend";
        let re = Regex::new(r"\d+").unwrap();
        let matches: Vec<&str> = re.find_iter(content).map(|m| m.as_str()).collect();
        assert_eq!(matches.join("\n"), "123\n456");
    }

    #[test]
    fn test_regex_no_matches() {
        let content = "Hello world";
        let re = Regex::new(r"\d+").unwrap();
        let matches: Vec<&str> = re.find_iter(content).map(|m| m.as_str()).collect();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_regex_invalid_pattern() {
        assert!(Regex::new(r"[invalid").is_err());
    }

    #[test]
    fn test_fetch_url_definition_has_headers_regex_and_raw() {
        let tool = FetchUrlTool {
            client: reqwest::Client::new(),
        };
        let def = tool.definition();
        let props = &def.parameters["properties"];
        assert!(props.get("headers").is_some());
        assert!(props.get("regex").is_some());
        assert!(props.get("raw").is_some());
    }

    #[test]
    fn test_redirects_to_scratchpad_logic() {
        // Mirrors the branch used in fetch_url::execute. When redirecting,
        // we force max_length = 0 (unlimited).
        let with = serde_json::json!({ "scratchpad": "out", "max_length": 100 });
        let without = serde_json::json!({ "max_length": 100 });
        assert!(redirects_to_scratchpad(&with));
        assert!(!redirects_to_scratchpad(&without));
    }

    #[test]
    fn test_web_search_definition_has_headers() {
        let tool = WebSearchTool {
            client: reqwest::Client::new(),
        };
        let def = tool.definition();
        let props = &def.parameters["properties"];
        assert!(props.get("headers").is_some());
    }
}
