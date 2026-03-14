use std::io::{self, Write};

use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor, Stylize};
use termimad::MadSkin;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderMode {
    Rich,
    #[default]
    Raw,
}

impl std::fmt::Display for RenderMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderMode::Rich => write!(formatter, "rich"),
            RenderMode::Raw => write!(formatter, "raw"),
        }
    }
}

impl std::str::FromStr for RenderMode {
    type Err = String;

    fn from_str(string: &str) -> std::result::Result<Self, Self::Err> {
        match string.to_lowercase().as_str() {
            "rich" => Ok(RenderMode::Rich),
            "raw" => Ok(RenderMode::Raw),
            other => Err(format!(
                "unknown render mode '{}' (expected 'rich' or 'raw')",
                other
            )),
        }
    }
}

pub struct StreamingRenderer {
    buffer: String,
    skin: MadSkin,
    mode: RenderMode,
    started: bool,
    raw_in_code_block: bool,
    raw_table_buffer: Vec<String>,
}

impl StreamingRenderer {
    pub fn new(mode: RenderMode) -> Self {
        Self {
            buffer: String::new(),
            skin: MadSkin::default_dark(),
            mode,
            started: false,
            raw_in_code_block: false,
            raw_table_buffer: Vec::new(),
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

        if self.mode == RenderMode::Raw {
            // In raw mode, flush complete lines immediately for streaming feel.
            // Buffer only the last incomplete line. Table rows are accumulated
            // and flushed together so columns can be aligned.
            while let Some(newline_pos) = self.buffer.find('\n') {
                let line = self.buffer[..newline_pos + 1].to_string();
                self.buffer = self.buffer[newline_pos + 1..].to_string();

                let trimmed = line.trim();
                if trimmed.starts_with('|') && trimmed.ends_with('|') {
                    self.raw_table_buffer.push(line);
                } else {
                    self.flush_raw_table()?;
                    self.render_block(&line)?;
                }
            }
            return Ok(());
        }

        // Rich mode: render complete paragraphs (text before double newline) through termimad.
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
        if self.mode == RenderMode::Raw && !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            let trimmed = remaining.trim();
            if trimmed.starts_with('|') && trimmed.ends_with('|') {
                self.raw_table_buffer.push(remaining);
            } else {
                self.flush_raw_table()?;
                self.render_block(&remaining)?;
            }
        }
        self.flush_raw_table()?;
        if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            self.render_block(&remaining)?;
        }
        io::stdout().flush()
    }

    fn render_block(&mut self, text: &str) -> io::Result<()> {
        match self.mode {
            RenderMode::Rich => {
                let formatted = self.skin.term_text(text);
                print!("{}", formatted);
            }
            RenderMode::Raw => {
                self.render_raw_line(text)?;
            }
        }
        io::stdout().flush()
    }

    fn render_raw_line(&mut self, line: &str) -> io::Result<()> {
        let trimmed = line.trim_end_matches('\n');
        let stdout = io::stdout();
        let mut out = stdout.lock();

        if trimmed.starts_with("```") {
            self.raw_in_code_block = !self.raw_in_code_block;
            if self.raw_in_code_block {
                write!(out, "{}", SetForegroundColor(Color::DarkYellow))?;
            } else {
                write!(out, "{}", ResetColor)?;
            }
            writeln!(out)?;
            return Ok(());
        }

        if self.raw_in_code_block {
            writeln!(out, "{}", trimmed)?;
            return Ok(());
        }

        if trimmed.is_empty() {
            writeln!(out)?;
            return Ok(());
        }

        if let Some(heading) = trimmed
            .strip_prefix("# ")
            .or_else(|| trimmed.strip_prefix("## "))
            .or_else(|| trimmed.strip_prefix("### "))
            .or_else(|| trimmed.strip_prefix("#### "))
        {
            write!(
                out,
                "{}{}{}{}",
                SetAttribute(Attribute::Bold),
                SetForegroundColor(Color::Cyan),
                heading,
                ResetColor,
            )?;
            write!(out, "{}", SetAttribute(Attribute::NoBold))?;
            writeln!(out)?;
            return Ok(());
        }

        render_inline_styles(&mut out, trimmed)?;
        writeln!(out)?;

        Ok(())
    }

    fn flush_raw_table(&mut self) -> io::Result<()> {
        if self.raw_table_buffer.is_empty() {
            return Ok(());
        }

        let rows = std::mem::take(&mut self.raw_table_buffer);

        // Parse each row into cells
        let mut parsed_rows: Vec<Vec<String>> = Vec::new();
        let mut separator_indices: Vec<usize> = Vec::new();

        for (row_index, row) in rows.iter().enumerate() {
            let trimmed = row.trim().trim_matches('|');
            let cells: Vec<String> = trimmed
                .split('|')
                .map(|cell| cell.trim().to_string())
                .collect();

            if is_separator_row(&cells) {
                separator_indices.push(row_index);
            }
            parsed_rows.push(cells);
        }

        // Compute max display width per column (skip separator rows for width calculation)
        let column_count = parsed_rows.iter().map(|row| row.len()).max().unwrap_or(0);
        let mut max_widths = vec![0usize; column_count];

        for (row_index, cells) in parsed_rows.iter().enumerate() {
            if separator_indices.contains(&row_index) {
                continue;
            }
            for (column_index, cell) in cells.iter().enumerate() {
                let width = unicode_width::UnicodeWidthStr::width(cell.as_str());
                if width > max_widths[column_index] {
                    max_widths[column_index] = width;
                }
            }
        }

        let stdout = io::stdout();
        let mut out = stdout.lock();

        for (row_index, cells) in parsed_rows.iter().enumerate() {
            if separator_indices.contains(&row_index) {
                // Render separator with correct widths
                write!(out, "|")?;
                for (column_index, _) in cells.iter().enumerate() {
                    let width = if column_index < max_widths.len() {
                        max_widths[column_index]
                    } else {
                        3
                    };
                    write!(out, " {:-<width$} |", "", width = width)?;
                }
                writeln!(out)?;
                continue;
            }

            write!(out, "|")?;
            for column_index in 0..column_count {
                let cell = cells.get(column_index).map(|s| s.as_str()).unwrap_or("");
                let display_width = unicode_width::UnicodeWidthStr::width(cell);
                let target_width = if column_index < max_widths.len() {
                    max_widths[column_index]
                } else {
                    display_width
                };
                let padding = target_width.saturating_sub(display_width);
                write!(out, " {}{} |", cell, " ".repeat(padding))?;
            }
            writeln!(out)?;
        }

        Ok(())
    }

    fn in_code_block(&self) -> bool {
        let fence_count = self.buffer.matches("```").count();
        !fence_count.is_multiple_of(2)
    }

    fn in_table(&self) -> bool {
        self.buffer.trim_start().starts_with('|')
    }
}

fn is_separator_row(cells: &[String]) -> bool {
    cells.iter().all(|cell| {
        let trimmed = cell.trim();
        !trimmed.is_empty()
            && trimmed
                .chars()
                .all(|character| character == '-' || character == ':')
    })
}

fn render_inline_styles(out: &mut impl Write, text: &str) -> io::Result<()> {
    let chars: Vec<char> = text.chars().collect();
    let length = chars.len();
    let mut index = 0;

    while index < length {
        // **bold** or __bold__
        if index + 1 < length
            && ((chars[index] == '*' && chars[index + 1] == '*')
                || (chars[index] == '_' && chars[index + 1] == '_'))
        {
            let marker = chars[index];
            if let Some(end) = find_closing_double(&chars, index + 2, marker) {
                write!(out, "{}", SetAttribute(Attribute::Bold))?;
                let inner: String = chars[index + 2..end].iter().collect();
                write!(out, "{}", inner)?;
                write!(out, "{}", SetAttribute(Attribute::NoBold))?;
                index = end + 2;
                continue;
            }
        }

        // *italic* or _italic_ (single, not preceded by another marker)
        if (chars[index] == '*' || chars[index] == '_')
            && (index + 1 >= length || chars[index + 1] != chars[index])
        {
            let marker = chars[index];
            if let Some(end) = find_closing_single(&chars, index + 1, marker) {
                write!(out, "{}", SetAttribute(Attribute::Italic))?;
                let inner: String = chars[index + 1..end].iter().collect();
                write!(out, "{}", inner)?;
                write!(out, "{}", SetAttribute(Attribute::NoItalic))?;
                index = end + 1;
                continue;
            }
        }

        // `code`
        if chars[index] == '`'
            && let Some(end) = find_closing_backtick(&chars, index + 1)
        {
            write!(out, "{}", SetForegroundColor(Color::DarkYellow))?;
            let inner: String = chars[index + 1..end].iter().collect();
            write!(out, "{}", inner)?;
            write!(out, "{}", ResetColor)?;
            index = end + 1;
            continue;
        }

        write!(out, "{}", chars[index])?;
        index += 1;
    }

    Ok(())
}

fn find_closing_double(chars: &[char], start: usize, marker: char) -> Option<usize> {
    let mut index = start;
    while index + 1 < chars.len() {
        if chars[index] == marker && chars[index + 1] == marker {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn find_closing_single(chars: &[char], start: usize, marker: char) -> Option<usize> {
    let mut index = start;
    while index < chars.len() {
        if chars[index] == marker {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn find_closing_backtick(chars: &[char], start: usize) -> Option<usize> {
    let mut index = start;
    while index < chars.len() {
        if chars[index] == '`' {
            return Some(index);
        }
        index += 1;
    }
    None
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
        let mut renderer = StreamingRenderer::new(RenderMode::Rich);
        renderer.push_delta("hello").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_streaming_renderer_strips_leading_newlines() {
        let mut renderer = StreamingRenderer::new(RenderMode::Rich);
        renderer.push_delta("\n\nhello").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_in_code_block() {
        let renderer = StreamingRenderer::new(RenderMode::Rich);
        assert!(!renderer.in_code_block());
    }

    #[test]
    fn test_in_table() {
        let mut renderer = StreamingRenderer::new(RenderMode::Rich);
        renderer.buffer = "| col1 | col2 |".to_string();
        assert!(renderer.in_table());
        renderer.buffer = "not a table".to_string();
        assert!(!renderer.in_table());
    }

    #[test]
    fn test_streaming_renderer_raw_mode() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.push_delta("**bold** text\n").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_render_mode_default() {
        assert_eq!(RenderMode::default(), RenderMode::Raw);
    }

    #[test]
    fn test_is_separator_row() {
        assert!(is_separator_row(&["---".to_string(), "---".to_string()]));
        assert!(is_separator_row(&[":---:".to_string(), "---:".to_string()]));
        assert!(!is_separator_row(&[
            "hello".to_string(),
            "world".to_string()
        ]));
        assert!(!is_separator_row(&["".to_string()]));
    }

    #[test]
    fn test_raw_table_buffering() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer
            .push_delta("| A | BB |\n|---|----|\n| C | DD |\n\n")
            .unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_raw_table_last_row_no_trailing_newline() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer
            .push_delta("| A | BB |\n|---|----|\n| C | DD |")
            .unwrap();
        renderer.finish().unwrap();
    }
}
