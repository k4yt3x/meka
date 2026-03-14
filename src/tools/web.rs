use async_trait::async_trait;
use html2md::rewrite_html;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::ToolDefinition;

use super::util::{ceil_char_boundary, require_str};
use super::{Tool, ToolOutput};

pub(super) struct FetchUrlTool {
    pub user_agent: String,
}

#[async_trait]
impl Tool for FetchUrlTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_url".to_string(),
            description: "Fetch a web page and return its content as markdown text.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
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

        let client = reqwest::Client::builder()
            .user_agent(&self.user_agent)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("failed to create HTTP client: {}", error),
            })?;

        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("failed to fetch '{}': {}", url, error),
            })?;

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolOutput {
                content: format!("HTTP {} for '{}'", status, url),
                is_error: true,
            });
        }

        let html = response
            .text()
            .await
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "fetch_url".to_string(),
                message: format!("failed to read response body: {}", error),
            })?;

        let markdown = rewrite_html(&html, false);

        let max_length = 50000;
        let content = if markdown.len() > max_length {
            format!(
                "{}\n\n... (truncated, showing first {} characters)",
                &markdown[..markdown.floor_char_boundary(max_length)],
                max_length
            )
        } else {
            markdown
        };

        Ok(ToolOutput {
            content,
            is_error: false,
        })
    }
}

pub(super) struct WebSearchTool {
    pub user_agent: String,
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

        let client = reqwest::Client::builder()
            .user_agent(&self.user_agent)
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|error| AgshError::ToolExecution {
                tool_name: "web_search".to_string(),
                message: format!("failed to create HTTP client: {}", error),
            })?;

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

        let response = client
            .get(url)
            .query(&query_params)
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
            Ok(ToolOutput {
                content: "No search results found.".to_string(),
                is_error: false,
            })
        } else {
            Ok(ToolOutput {
                content: results,
                is_error: false,
            })
        }
    }
}

fn parse_duckduckgo_results(html: &str) -> String {
    // DuckDuckGo HTML results have a structure where each result is in a
    // div with class "result". We extract titles and snippets using simple
    // string matching since we don't want a full HTML parser dependency
    // just for this.
    let mut results = Vec::new();
    let mut position = 0;

    while let Some(result_start) = html[position..].find("class=\"result__a\"") {
        let absolute_start = position + result_start;

        let url = extract_href(
            &html[..html.floor_char_boundary(absolute_start + 200)],
            absolute_start,
        );

        let title = if let Some(tag_end) = html[absolute_start..].find('>') {
            let text_start = absolute_start + tag_end + 1;
            if let Some(close_tag) = html[text_start..].find("</a>") {
                strip_html_tags(&html[text_start..text_start + close_tag])
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let snippet =
            if let Some(snippet_start) = html[absolute_start..].find("class=\"result__snippet\"") {
                let snippet_abs = absolute_start + snippet_start;
                if let Some(tag_end) = html[snippet_abs..].find('>') {
                    let text_start = snippet_abs + tag_end + 1;
                    if let Some(close_tag) = html[text_start..].find("</") {
                        strip_html_tags(&html[text_start..text_start + close_tag])
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

        if !title.is_empty() {
            let mut result_text = format!("{}. **{}**", results.len() + 1, title.trim());
            if let Some(url) = &url {
                result_text.push_str(&format!("\n   URL: {}", url));
            }
            if !snippet.is_empty() {
                result_text.push_str(&format!("\n   {}", snippet.trim()));
            }
            results.push(result_text);

            if results.len() >= 10 {
                break;
            }
        }

        position = absolute_start + 1;
    }

    results.join("\n\n")
}

fn extract_href(html: &str, near_position: usize) -> Option<String> {
    let search_start = html.floor_char_boundary(near_position.saturating_sub(500));
    let search_region = &html[search_start..near_position];

    if let Some(href_pos) = search_region.rfind("href=\"") {
        let url_start = search_start + href_pos + 6;
        if let Some(url_end) = html[url_start..].find('"') {
            let url = &html[url_start..url_start + url_end];
            if let Some(uddg_pos) = url.find("uddg=") {
                let encoded_url = &url[uddg_pos + 5..];
                let decoded = urldecode(encoded_url);
                let clean_url = decoded.split('&').next().unwrap_or(&decoded);
                return Some(clean_url.to_string());
            }
            return Some(url.to_string());
        }
    }
    None
}

fn urldecode(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars();

    while let Some(character) = chars.next() {
        if character == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            } else {
                result.push('%');
                result.push_str(&hex);
            }
        } else if character == '+' {
            result.push(' ');
        } else {
            result.push(character);
        }
    }

    result
}

fn strip_html_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;

    for character in html.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(character),
            _ => {}
        }
    }

    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn parse_google_results(html: &str) -> String {
    let mut results = Vec::new();
    let mut position = 0;

    while let Some(block_start) = html[position..].find("<div class=\"g\"") {
        let absolute_start = position + block_start;
        let block_end = html[absolute_start..]
            .find("<div class=\"g\"")
            .map(|next| {
                if next == 0 {
                    html[ceil_char_boundary(html, absolute_start + 20)..]
                        .find("<div class=\"g\"")
                        .map(|p| ceil_char_boundary(html, absolute_start + 20) + p)
                        .unwrap_or(html.len())
                } else {
                    absolute_start + next
                }
            })
            .unwrap_or(html.len());

        let block = &html[absolute_start..block_end];

        let url = if let Some(href_start) = block.find("href=\"") {
            let url_start = href_start + 6;
            block[url_start..]
                .find('"')
                .map(|url_end| block[url_start..url_start + url_end].to_string())
                .filter(|url| url.starts_with("http"))
        } else {
            None
        };

        let title = if let Some(h3_start) = block.find("<h3") {
            if let Some(tag_end) = block[h3_start..].find('>') {
                let text_start = h3_start + tag_end + 1;
                block[text_start..]
                    .find("</h3>")
                    .map(|close| strip_html_tags(&block[text_start..text_start + close]))
            } else {
                None
            }
        } else {
            None
        };

        if let Some(title) = title
            && !title.trim().is_empty()
        {
            let mut result_text = format!("{}. **{}**", results.len() + 1, title.trim());
            if let Some(url) = &url {
                result_text.push_str(&format!("\n   URL: {}", url));
            }
            results.push(result_text);

            if results.len() >= 10 {
                break;
            }
        }

        position = absolute_start + 1;
    }

    results.join("\n\n")
}

fn parse_bing_results(html: &str) -> String {
    let mut results = Vec::new();
    let mut position = 0;

    while let Some(block_start) = html[position..].find("class=\"b_algo\"") {
        let absolute_start = position + block_start;
        let block_end = html[ceil_char_boundary(html, absolute_start + 14)..]
            .find("class=\"b_algo\"")
            .map(|next| ceil_char_boundary(html, absolute_start + 14) + next)
            .unwrap_or(html.len());

        let block = &html[absolute_start..block_end];

        let url = if let Some(href_start) = block.find("href=\"") {
            let url_start = href_start + 6;
            block[url_start..]
                .find('"')
                .map(|url_end| block[url_start..url_start + url_end].to_string())
                .filter(|url| url.starts_with("http"))
        } else {
            None
        };

        let title = if let Some(h2_start) = block.find("<h2") {
            if let Some(a_start) = block[h2_start..].find("<a ") {
                let a_abs = h2_start + a_start;
                if let Some(tag_end) = block[a_abs..].find('>') {
                    let text_start = a_abs + tag_end + 1;
                    block[text_start..]
                        .find("</a>")
                        .map(|close| strip_html_tags(&block[text_start..text_start + close]))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let snippet = block.find("class=\"b_caption\"").and_then(|cap_start| {
            block[cap_start..].find("<p").and_then(|p_start| {
                let p_abs = cap_start + p_start;
                block[p_abs..].find('>').and_then(|tag_end| {
                    let text_start = p_abs + tag_end + 1;
                    block[text_start..]
                        .find("</p>")
                        .map(|close| strip_html_tags(&block[text_start..text_start + close]))
                })
            })
        });

        if let Some(title) = title
            && !title.trim().is_empty()
        {
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

        position = absolute_start + 1;
    }

    results.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("<b>hello</b>"), "hello");
        assert_eq!(strip_html_tags("no tags here"), "no tags here");
        assert_eq!(strip_html_tags("&amp; test"), "& test");
    }

    #[test]
    fn test_urldecode() {
        assert_eq!(urldecode("hello%20world"), "hello world");
        assert_eq!(urldecode("test%2Fpath"), "test/path");
        assert_eq!(urldecode("a+b"), "a b");
    }
}
