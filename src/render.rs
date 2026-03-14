use std::io::{self, Write};

use crossterm::style::{Color, Stylize};
use termimad::MadSkin;

pub struct StreamingRenderer {
    buffer: String,
    skin: MadSkin,
    started: bool,
}

impl StreamingRenderer {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            skin: MadSkin::default_dark(),
            started: false,
        }
    }

    pub fn push_delta(&mut self, delta: &str) -> io::Result<()> {
        let delta = if self.started {
            delta
        } else {
            let trimmed = delta.trim_start_matches('\n');
            if trimmed.is_empty() {
                return Ok(());
            }
            self.started = true;
            trimmed
        };

        self.buffer.push_str(delta);

        // Render complete paragraphs (text before double newline) through termimad.
        // Keep the tail (incomplete paragraph) buffered for later.
        while let Some(boundary) = self.buffer.find("\n\n") {
            let complete = self.buffer[..boundary + 2].to_string();
            self.buffer = self.buffer[boundary + 2..].to_string();
            self.render_block(&complete)?;
        }

        // For single newline-terminated lines outside of code blocks,
        // render them immediately to give a streaming feel
        if !self.in_code_block() && !self.in_table() {
            while let Some(newline_pos) = self.buffer.find('\n') {
                // Only flush if this isn't the start of a potential double-newline
                if newline_pos + 1 < self.buffer.len() || !self.buffer.ends_with('\n') {
                    let line = self.buffer[..newline_pos + 1].to_string();
                    self.buffer = self.buffer[newline_pos + 1..].to_string();
                    self.render_block(&line)?;
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            self.render_block(&remaining)?;
        }
        io::stdout().flush()
    }

    fn render_block(&self, text: &str) -> io::Result<()> {
        let formatted = self.skin.term_text(text);
        print!("{}", formatted);
        io::stdout().flush()
    }

    fn in_code_block(&self) -> bool {
        let fence_count = self.buffer.matches("```").count();
        !fence_count.is_multiple_of(2)
    }

    fn in_table(&self) -> bool {
        self.buffer.trim_start().starts_with('|')
    }
}

pub fn render_tool_indicator(name: &str, input: &serde_json::Value) {
    let display_name = tool_display_name(name);
    let indicator = match tool_primary_param(name, input) {
        Some(value) => {
            let truncated = truncate_display(value, 80);
            format!("[tool {}(`{}`)]", display_name, truncated)
        }
        None => format!("[tool {}]", display_name),
    };
    println!("{}", indicator.with(Color::DarkCyan));
}

pub fn render_session_id(label: &str, id: &str) {
    eprintln!("{}", format!("{}: {}", label, id).with(Color::DarkGrey));
}

pub fn render_hint(message: &str) {
    eprintln!("{}", message.with(Color::DarkGrey));
}

fn tool_display_name(name: &str) -> &str {
    match name {
        "execute_command" => "Shell",
        "read_file" => "ReadFile",
        "write_file" => "WriteFile",
        "edit_file" => "EditFile",
        "find_files" => "FindFiles",
        "search_contents" => "SearchContents",
        "fetch_url" => "FetchUrl",
        "web_search" => "WebSearch",
        other => other,
    }
}

fn tool_primary_param<'a>(name: &str, input: &'a serde_json::Value) -> Option<&'a str> {
    let key = match name {
        "execute_command" => "command",
        "read_file" | "write_file" | "edit_file" => "path",
        "find_files" | "search_contents" => "pattern",
        "fetch_url" => "url",
        "web_search" => "query",
        _ => return None,
    };
    input.get(key).and_then(|v| v.as_str())
}

fn truncate_display(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        value.to_string()
    } else {
        let truncated: String = value.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_display_short() {
        assert_eq!(truncate_display("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_display_exact() {
        assert_eq!(truncate_display("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_display_long() {
        assert_eq!(truncate_display("hello world", 5), "hello...");
    }

    #[test]
    fn test_truncate_display_empty() {
        assert_eq!(truncate_display("", 5), "");
    }

    #[test]
    fn test_tool_display_name_mappings() {
        assert_eq!(tool_display_name("execute_command"), "Shell");
        assert_eq!(tool_display_name("read_file"), "ReadFile");
        assert_eq!(tool_display_name("write_file"), "WriteFile");
        assert_eq!(tool_display_name("edit_file"), "EditFile");
        assert_eq!(tool_display_name("find_files"), "FindFiles");
        assert_eq!(tool_display_name("search_contents"), "SearchContents");
        assert_eq!(tool_display_name("fetch_url"), "FetchUrl");
        assert_eq!(tool_display_name("web_search"), "WebSearch");
        assert_eq!(tool_display_name("custom_tool"), "custom_tool");
    }

    #[test]
    fn test_tool_primary_param() {
        let input = serde_json::json!({"command": "ls", "path": "/tmp"});
        assert_eq!(tool_primary_param("execute_command", &input), Some("ls"));
        assert_eq!(tool_primary_param("read_file", &input), Some("/tmp"));
        assert_eq!(tool_primary_param("unknown_tool", &input), None);
    }

    #[test]
    fn test_tool_primary_param_missing() {
        let input = serde_json::json!({"other": "value"});
        assert_eq!(tool_primary_param("execute_command", &input), None);
    }

    #[test]
    fn test_streaming_renderer_basic() {
        let mut renderer = StreamingRenderer::new();
        renderer.push_delta("hello").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_streaming_renderer_strips_leading_newlines() {
        let mut renderer = StreamingRenderer::new();
        renderer.push_delta("\n\nhello").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_in_code_block() {
        let renderer = StreamingRenderer::new();
        assert!(!renderer.in_code_block());
    }

    #[test]
    fn test_in_table() {
        let mut renderer = StreamingRenderer::new();
        renderer.buffer = "| col1 | col2 |".to_string();
        assert!(renderer.in_table());
        renderer.buffer = "not a table".to_string();
        assert!(!renderer.in_table());
    }
}
