use async_trait::async_trait;
use html2md::rewrite_html;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::ToolDefinition;

use super::util::require_str;
use super::{Tool, ToolOutput};

pub(super) struct FetchUrlTool {
    pub client: reqwest::Client,
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
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum number of characters to return. Default: 50000. Set to 0 for no limit."
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

        let response =
            self.client
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

        let max_length = input["max_length"]
            .as_u64()
            .map(|value| value as usize)
            .unwrap_or(50000);

        let content = if max_length > 0 && markdown.len() > max_length {
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

        let response = self
            .client
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
    let document = scraper::Html::parse_document(html);
    let result_selector = scraper::Selector::parse(".result").unwrap();
    let link_selector = scraper::Selector::parse("a.result__a").unwrap();
    let snippet_selector = scraper::Selector::parse(".result__snippet").unwrap();
    let mut results = Vec::new();

    for block in document.select(&result_selector) {
        let link = match block.select(&link_selector).next() {
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
            .select(&snippet_selector)
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
    let block_selector = scraper::Selector::parse("div.g").unwrap();
    let title_selector = scraper::Selector::parse("h3").unwrap();
    let link_selector = scraper::Selector::parse("a[href]").unwrap();
    let mut results = Vec::new();

    for block in document.select(&block_selector) {
        let title: String = match block.select(&title_selector).next() {
            Some(h3) => h3.text().collect(),
            None => continue,
        };
        if title.trim().is_empty() {
            continue;
        }

        let url = block.select(&link_selector).find_map(|a| {
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
    let block_selector = scraper::Selector::parse(".b_algo").unwrap();
    let title_selector = scraper::Selector::parse("h2 a").unwrap();
    let snippet_selector = scraper::Selector::parse(".b_caption p").unwrap();
    let mut results = Vec::new();

    for block in document.select(&block_selector) {
        let title_element = match block.select(&title_selector).next() {
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
            .select(&snippet_selector)
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
}
